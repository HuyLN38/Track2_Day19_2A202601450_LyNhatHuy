#!/usr/bin/env python3
"""5-query demo for HybridMemoryAgent — `python bonus/demo.py` exits 0.

Seeds one user's episodic memory, then runs the five query shapes from
BONUS-CHALLENGE.md and prints the assembled context for each.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonus.agent import (  # noqa: E402
    AFFINITY_QUERY, AFFINITY_WEIGHT, RRF_K, HybridMemoryAgent,
)

USER = "u_001"          # profile: vi · 187 wpm · affinity=cloud (from NB4 Parquet)

# Episodic memory: things this user read / said / noted. Deliberately mixes
# vi and en inside single sentences — that is how VN engineers actually write.
SEED_MEMORIES = [
    "Hôm nay tôi đọc tài liệu về Kubernetes HPA. Horizontal Pod Autoscaler "
    "theo dõi CPU và custom metrics rồi tự động tăng giảm số replica. "
    "Ghi chú: nhớ set resource requests, nếu không HPA không có baseline để tính.",

    "Đọc bài về cluster autoscaler trên GKE. Khi pod ở trạng thái Pending vì "
    "thiếu tài nguyên, autoscaler sẽ thêm node mới vào node pool. Cơ chế này "
    "co giãn hạ tầng tự động theo tải thực tế, khác với HPA ở tầng pod.",

    "Note về cloud security: OWASP Top 10 cho LLM, mục LLM08 nói về "
    "vector store poisoning và rò rỉ dữ liệu chéo tenant. Giải pháp là filter "
    "theo tenant_id ngay trong index chứ không post-filter sau khi ANN trả về.",

    "Tôi đã đọc guide về mã hoá dữ liệu at-rest trên S3 với KMS. "
    "Bucket policy nên deny mọi request không có header x-amz-server-side-encryption. "
    "Kèm ghi chú về Nghị định 13/2023 khi lưu dữ liệu cá nhân người dùng VN.",

    "Ghi chú ngắn: so sánh Redis và DynamoDB làm online store cho Feast. "
    "Redis nhanh hơn ở p99 nhưng phải tự lo HA; DynamoDB đắt hơn nhưng managed.",

    "Đọc về service worker để hỗ trợ offline trên frontend. Cache-first cho "
    "static assets, network-first cho API. Không liên quan cloud nhưng lưu lại.",
]

# (label, query, indices into SEED_MEMORIES that a correct top-1 may return)
QUERIES = [
    ("1 · vector-only hit",      "Tôi đã đọc gì về Kubernetes?",        {0, 1}),
    ("2 · needs profile",        "Recommend đọc gì tiếp",               {0, 1, 2, 3}),
    ("3 · needs fresh activity", "Tôi đang quan tâm gì gần đây?",       set(range(6))),
    ("4 · paraphrase",           "Tài liệu về tự động mở rộng hạ tầng?", {0, 1}),
    ("5 · mixed + profile",      "Cho tôi summary cloud security",      {2, 3}),
]


def _fuse(arms: list[tuple[list[str], float]]) -> list[str]:
    """Standalone RRF so each arm can be scored in isolation."""
    sc: dict[str, float] = {}
    for ranked, w in arms:
        for rank, t in enumerate(ranked, start=1):
            sc[t] = sc.get(t, 0.0) + w / (RRF_K + rank)
    return [t for t, _ in sorted(sc.items(), key=lambda kv: -kv[1])]


def ablation(agent: HybridMemoryAgent) -> None:
    """Which arm actually earns its place? NB2's golden-set table, POC-scale."""
    def source_of(text: str) -> int:
        return next((i for i, m in enumerate(SEED_MEMORIES) if text[:40] in m), -1)

    names = ["BM25", "vector", "hybrid", "hybrid+profile"]
    score = dict.fromkeys(names, 0)
    print(f"{'query':40}" + "".join(f"{n:>16}" for n in names))
    for _, q, gold in QUERIES:
        kw = agent._keyword(q, USER, 20)
        sem = agent._semantic(q, USER, 20)
        aff = agent._semantic(AFFINITY_QUERY["cloud"], USER, 20)
        got = {
            "BM25": _fuse([(kw, 1.0)]),
            "vector": _fuse([(sem, 1.0)]),
            "hybrid": _fuse([(kw, 1.0), (sem, 1.0)]),
            "hybrid+profile": _fuse([(kw, 1.0), (sem, 1.0), (aff, AFFINITY_WEIGHT)]),
        }
        marks = []
        for n in names:
            hit = bool(got[n]) and source_of(got[n][0]) in gold
            score[n] += hit
            marks.append("hit" if hit else "-")
        print(f"{q[:38]:40}" + "".join(f"{m:>16}" for m in marks))
    print(f"{'hit@1 (out of 5)':40}" + "".join(f"{score[n]:>16}" for n in names))


def main() -> int:
    print("=" * 72)
    print("HybridMemoryAgent — bonus demo (Vector Store + Feature Store)")
    print("=" * 72)

    agent = HybridMemoryAgent()
    for m in SEED_MEMORIES:
        agent.remember(m, user_id=USER)
    n = len(agent._chunks.get(USER, []))
    print(f"\nSeeded {len(SEED_MEMORIES)} memories -> {n} chunks "
          f"({agent.embedder.model_name}, {agent.embedder.dim}d)\n")

    for label, q, _gold in QUERIES:
        print("-" * 72)
        print(f"### {label}")
        print("-" * 72)
        print(agent.recall(q, user_id=USER))
        print()

    print("-" * 72)
    print("### ablation — does each retrieval arm earn its place?")
    print("-" * 72)
    ablation(agent)
    print()

    # Privacy check: a different user must see none of u_001's memories.
    print("-" * 72)
    print("### isolation check — u_042 asks the same question")
    print("-" * 72)
    print(agent.recall("Tôi đã đọc gì về Kubernetes?", user_id="u_042"))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
