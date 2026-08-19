"""HybridMemoryAgent — episodic memory (Qdrant) + stable profile (Feast).

Bonus challenge for Lab 19. The two halves of the lab meet here:

    Vector Store  (NB1-NB3)  ->  what the user *said and read*   (episodic)
    Feature Store (NB4, NB8) ->  who the user *is*               (profile)

`recall()` fuses them into one context string. No LLM call — the assembled
string IS the deliverable, so the design decision stays visible instead of
disappearing into a prompt.

Reuses `app.embeddings.Embedder` so EMBEDDING_BACKEND still selects the model,
and reuses the RRF k=60 constant from `app.search` so the fusion behaves
identically to NB2.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
import warnings
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

MEMORY_COLLECTION = "bonus_memory"
RRF_K = 60                  # same fusion constant as app/search.py
AFFINITY_WEIGHT = 0.5       # profile arm nudges the ranking, never rules it

# topic_affinity is one token in the feature store; expand it into a phrase
# the embedder can actually place in vector space.
AFFINITY_QUERY = {
    "cloud": "điện toán đám mây, kubernetes, container, hạ tầng cloud",
    "ai_ml": "trí tuệ nhân tạo, mô hình học máy, embedding, LLM",
    "security": "bảo mật, mã hoá, xác thực, lỗ hổng an ninh",
    "database": "cơ sở dữ liệu, truy vấn SQL, chỉ mục, replica",
    "devops": "devops, CI/CD, pipeline triển khai, docker",
}
CHUNK_CHARS = 320           # ~80 tokens of Vietnamese — see ARCHITECTURE.md §D1
OVERLAP_SENTENCES = 1

PROFILE_FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
]
ACTIVITY_FEATURES = [
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]

# Function words carry no retrieval signal, and on a *per-user* memory store
# (tens of chunks, not 1000 docs) BM25's IDF is too thin to discount them on
# its own: without this list, BM25("Recommend đọc gì tiếp") ranks purely on the
# stopword "đọc" and the shortest chunk wins on length normalisation.
STOPWORDS = {
    "tôi", "toi", "bạn", "ban", "gì", "gi", "gia", "về", "ve", "của", "cua",
    "và", "va", "là", "la", "có", "co", "không", "khong", "cho", "với", "voi",
    "đã", "da", "đang", "dang", "được", "duoc", "này", "nay", "đó", "do",
    "một", "mot", "các", "cac", "những", "nhung", "thì", "thi", "mà", "ma",
    "ở", "o", "trong", "trên", "tren", "khi", "nếu", "neu", "đọc", "doc",
    "tiếp", "tiep", "gần", "gan", "đây", "day", "cùng", "cung", "hay",
    "the", "a", "an", "of", "to", "in", "on", "for", "is", "are", "what",
    "how", "me", "my", "i", "you", "give", "show", "tell", "recommend",
}

_SENTENCE_RE = re.compile(r"(?<=[.!?;\n])\s+")
_WORD_RE = re.compile(r"[0-9a-zà-ỹđ_/.-]+", re.IGNORECASE)


def fold(text: str) -> str:
    """Strip Vietnamese diacritics: 'co giãn' -> 'co gian', 'đám' -> 'dam'.

    VN users routinely type without dấu, so the folded form is indexed
    alongside the raw form (ARCHITECTURE.md §D-VN).
    """
    s = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    return "".join(c for c in s if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Raw tokens + diacritic-folded tokens, minus stopwords."""
    raw = [t for t in _WORD_RE.findall(text.lower()) if t not in STOPWORDS]
    folded = [f for t in raw if (f := fold(t)) != t and f not in STOPWORDS]
    return raw + folded


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Pack sentences up to `max_chars`, overlapping by one sentence."""
    sents = [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
    if not sents:
        return []
    chunks, buf = [], []
    for s in sents:
        if buf and sum(len(x) for x in buf) + len(s) > max_chars:
            chunks.append(" ".join(buf))
            buf = buf[-OVERLAP_SENTENCES:]          # carry context across the seam
        buf.append(s)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


@dataclass
class MemoryHit:
    text: str
    score: float


@dataclass
class HybridMemoryAgent:
    """Per-user episodic memory with profile-aware recall."""

    feast_repo: str = "app/feast_repo"
    embedder: Embedder = field(default_factory=Embedder)
    client: QdrantClient = None
    _chunks: dict[str, list[str]] = field(default_factory=dict)
    _bm25: dict[str, BM25Okapi] = field(default_factory=dict)
    _n_points: int = 0

    def __post_init__(self) -> None:
        if self.client is None:
            mode = os.getenv("QDRANT_MODE", "memory")
            self.client = (QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
                           if mode == "server" else QdrantClient(":memory:"))
        names = {c.name for c in self.client.get_collections().collections}
        if MEMORY_COLLECTION in names:
            self.client.delete_collection(MEMORY_COLLECTION)
        self.client.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config=models.VectorParams(
                size=self.embedder.dim, distance=models.Distance.COSINE),
        )
        # Single shared collection + user_id payload filter (not one collection
        # per user) — the isolation tradeoff is argued in ARCHITECTURE.md §D2.
        with warnings.catch_warnings():
            # Local Qdrant filters correctly but ignores payload indexes and
            # warns every run; on a server this is what keeps the filter cheap.
            warnings.simplefilter("ignore")
            self.client.create_payload_index(
                MEMORY_COLLECTION, "user_id",
                field_schema=models.PayloadSchemaType.KEYWORD)
        self._store = self._open_feast()

    def _open_feast(self):
        try:
            from feast import FeatureStore
            return FeatureStore(repo_path=self.feast_repo)
        except Exception as exc:                     # not applied / not installed
            print(f"  [warn] Feast unavailable ({type(exc).__name__}); "
                  "profile falls back to defaults. Run NB4 first.")
            return None

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        chunks = chunk_text(text)
        if not chunks:
            return
        vectors = list(self.embedder.embed(chunks))
        now = time.time()
        self.client.upsert(
            collection_name=MEMORY_COLLECTION,
            points=[
                models.PointStruct(
                    id=self._n_points + i, vector=v.tolist(),
                    payload={"user_id": user_id, "text": c, "ts": now},
                )
                for i, (c, v) in enumerate(zip(chunks, vectors))
            ],
        )
        self._n_points += len(chunks)
        self._chunks.setdefault(user_id, []).extend(chunks)
        self._bm25.pop(user_id, None)                # lazily rebuilt on next recall

    # ── read path ───────────────────────────────────────────────────────
    def _features(self, user_id: str) -> dict:
        defaults = {"reading_speed_wpm": 200, "preferred_language": "vi",
                    "topic_affinity": "unknown", "queries_last_hour": 0,
                    "distinct_topics_24h": 0}
        if self._store is None:
            return defaults
        got = self._store.get_online_features(
            features=PROFILE_FEATURES + ACTIVITY_FEATURES,
            entity_rows=[{"user_id": user_id}],
        ).to_dict()
        # A TTL-expired feature comes back as None, not as a missing key.
        return {k: (got[k][0] if got.get(k) and got[k][0] is not None else v)
                for k, v in defaults.items()}

    def _keyword(self, query: str, user_id: str, depth: int) -> list[str]:
        chunks = self._chunks.get(user_id, [])
        if not chunks:
            return []
        if user_id not in self._bm25:
            self._bm25[user_id] = BM25Okapi([tokenize(c) for c in chunks])
        scores = self._bm25[user_id].get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:depth]
        return [chunks[i] for i in ranked if scores[i] > 0]

    def _semantic(self, query: str, user_id: str, depth: int) -> list[str]:
        qv = next(self.embedder.embed([query])).tolist()
        hits = self.client.query_points(
            collection_name=MEMORY_COLLECTION, query=qv, limit=depth,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id))]),
        ).points
        return [h.payload["text"] for h in hits]

    def search_memories(self, query: str, user_id: str, top_k: int = 3,
                        affinity: str | None = None) -> list[MemoryHit]:
        """RRF over THREE rankers, not two: BM25 + vector + profile affinity.

        The third arm is what makes the feature store change the *ranking*
        rather than only decorate the printed context. It carries half weight
        (AFFINITY_WEIGHT) so a stale profile can nudge but never overrule what
        the user actually asked — the tradeoff argued in ARCHITECTURE.md §D2.
        """
        depth = max(top_k * 5, 20)
        arms = [(self._keyword(query, user_id, depth), 1.0),
                (self._semantic(query, user_id, depth), 1.0)]
        if affinity and affinity != "unknown":
            # Semantic arm, NOT BM25: affinity is a *concept*, and lexical
            # matching on the bare word backfires — a note reading "không liên
            # quan cloud" scores top on BM25("cloud"). See ARCHITECTURE.md §D2.
            arms.append((self._semantic(AFFINITY_QUERY.get(affinity, affinity),
                                        user_id, depth), AFFINITY_WEIGHT))

        rrf: dict[str, float] = {}
        for ranked, weight in arms:
            for rank, text in enumerate(ranked, start=1):
                rrf[text] = rrf.get(text, 0.0) + weight / (RRF_K + rank)
        top = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
        return [MemoryHit(text=t, score=s) for t, s in top]

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Top-K memories + profile features -> one assembled context string."""
        f = self._features(user_id)
        hits = self.search_memories(query, user_id, affinity=f["topic_affinity"])
        memories = ("\n".join(f"  {i}. [{h.score:.4f}] {h.text}"
                              for i, h in enumerate(hits, 1))
                    or "  (chưa có ký ức nào khớp)")
        return (
            f"[QUERY] {query}\n"
            f"[PROFILE] user={user_id} · lang={f['preferred_language']} · "
            f"reads {f['reading_speed_wpm']} wpm · affinity={f['topic_affinity']}\n"
            f"[ACTIVITY] {f['queries_last_hour']} queries/1h · "
            f"{f['distinct_topics_24h']} topics/24h\n"
            f"[MEMORIES]\n{memories}"
        )
