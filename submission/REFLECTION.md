# Reflection — Lab 19

**Tên:** Lý Nhật Huy
**Cohort:** A20K-K4
**Path đã chạy:** both

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

- **Exact queries:** BM25 cùng Hybrid dẫn đầu (~96.7%). Nhóm query này mang
  thuật ngữ và mã kỹ thuật cố định, nên việc đối sánh từ nguyên văn (lexical
  match) của BM25 vốn đã là phương án tốt nhất.
- **Paraphrase queries:** Ưu thế nghiêng về Vector/Hybrid, bởi câu hỏi được
  diễn đạt lại và không còn giữ đúng từ khoá xuất hiện trong tài liệu — chỉ
  tín hiệu ngữ nghĩa mới bắt được liên hệ.
- **Mixed queries:** Hybrid thắng tuyệt đối (100%). RRF ($k=60$) hoà trộn
  được cả hai nguồn tín hiệu: điểm khớp từ khoá của BM25 và khoảng cách
  vector của mô hình semantic.

**Khi KHÔNG dùng Hybrid:**

1. **Pure BM25:** Truy vấn định danh chính xác (SKU, Log ID, mã hợp đồng),
   tra từ điển; hoặc khi hệ thống đòi độ trễ rất thấp (<5ms) và muốn tiết
   giảm hạ tầng (khỏi vận hành thêm Vector DB).
2. **Pure Vector:** Tìm kiếm đa phương thức (ảnh/âm thanh), hoặc truy vấn
   dựa trên ngữ cảnh mà từ khoá không xuất hiện (cross-lingual, sắc thái
   cảm xúc).
3. **Tránh Hybrid:** Khi ngân sách tính toán bị siết, vì Hybrid phải chạy
   song song 2 retriever rồi hợp nhất bằng RRF — chi phí gấp đôi.

---

## Điều ngạc nhiên nhất khi làm lab này

Target-encoding leakage trong Feature Store có thể phá hỏng mô hình ngay cả
khi mọi chỉ số trên tập train trông rất đẹp; và PIT join (Point-in-Time) là
thứ xử lý dứt điểm kiểu rò rỉ dữ liệu theo trục thời gian này.

---

## Bonus challenge

- [x] Đã làm bonus (xem [`bonus/`](../bonus/))
- [ ] Pair work với: _không — làm solo_

`bonus/` chứa POC **HybridMemoryAgent**: episodic memory trong Qdrant (filter
theo `user_id`) ghép với stable profile + recent activity từ Feast online
store, hợp nhất bằng RRF k=60 trên **3 ranker** (BM25 · vector · topic
affinity w=0.5).

- [`bonus/ARCHITECTURE.md`](../bonus/ARCHITECTURE.md) — sơ đồ Mermaid, 3 quyết
  định kèm tradeoff, lựa chọn đã loại bỏ, phần bối cảnh tiếng Việt, và mục
  giới hạn còn tồn đọng.
- [`bonus/agent.py`](../bonus/agent.py) — `remember()` / `recall()`.
- [`bonus/demo.py`](../bonus/demo.py) — 5 query + bảng ablation + kiểm tra
  isolation giữa hai user. `python bonus/demo.py` exit 0.

Điều đo được mà tôi không đoán trước: ở quy mô per-user (6 ký ức), hybrid chỉ
**hoà** với vector thuần (5/5) chứ không thắng như trên corpus 1000 doc của
NB2 — BM25 cần khối lượng để IDF có nghĩa, mà bộ nhớ mỗi user bắt đầu từ số
không. Bỏ stopword tiếng Việt kéo arm BM25 từ 3/5 lên 4/5.
