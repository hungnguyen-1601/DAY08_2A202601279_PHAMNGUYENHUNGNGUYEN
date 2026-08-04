# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về dịch vụ và chính sách đại học liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
┌──────────────────────── DATA LAYER ────────────────────────┐
│ Task 1: PDF/HTML chính sách (RMIT)  →  data/landing/legal/ │
│ Task 2: Crawl 5 bài tin tức (JSON)  →  data/landing/news/  │
│ Task 3: MarkItDown convert          →  data/standardized/  │
│ Task 4: RecursiveCharacterTextSplitter (size=500, overlap= │
│         50) + all-MiniLM-L6-v2 (384 dim) → ChromaDB (508   │
│         chunks, cosine)                                    │
└────────────────────────────┬───────────────────────────────┘
                             │
        User ──► Streamlit Chatbot (app.py)
                             │
                             ▼
┌───────────────── RETRIEVAL (Task 9) ───────────────────────┐
│  ├─ Semantic Search (Task 5, ChromaDB + HyDE) ──┐          │
│  ├─ Lexical Search  (Task 6, BM25/TF-IDF) ──────┤          │
│  │                                              ▼          │
│  │                    Merge + Rerank RRF k=60 (Task 7)     │
│  │                    (+ Jina cross-encoder nếu có key)    │
│  └─ Nếu cosine top-1 < 0.48 → Fallback PageIndex           │
│     Vectorless (Task 8)                                    │
└────────────────────────────┬───────────────────────────────┘
                             ▼
┌───────────────── GENERATION (Task 10) ─────────────────────┐
│ Reorder chunks (chống lost-in-the-middle) → format context │
│ → OpenRouter LLM → câu trả lời kèm citation [Source, Year] │
│ → không đủ evidence: "I cannot verify this information"    │
└────────────────────────────┬───────────────────────────────┘
                             ▼
     UI: chat history (session_state) + expander nguồn tham khảo

┌──────────────── EVALUATION (group_project) ────────────────┐
│ golden_dataset.json (15 Q&A) → eval_pipeline.py            │
│ Metrics: Hit@5, MRR, Context Recall, Citation Presence/    │
│ Validity, Abstention Accuracy + A/B: Hybrid+RRF+fallback   │
│ vs Dense-only → results.md + evaluation_results.json       │
└────────────────────────────────────────────────────────────┘
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Phạm Nguyễn Hùng Nguyên (hungnguyen-1601) | 2A202601279 | **Role 1 — Team Leader & Architect:** điều phối nhóm, review & merge PR, duyệt config chunking/RRF/threshold, tổng hợp README & kiến trúc, xác nhận 35/35 test passed | ✅ Hoàn thành |
| An (Anbt0106) | | **Role 2 — Data & Retrieval Specialist:** Task 1 (thu thập văn bản chính sách), Task 4 (chunking + indexing ChromaDB), Task 7 (RRF reranking), Task 9 (retrieval pipeline, fallback threshold 0.48), tích hợp generation vào app.py | ✅ Hoàn thành |
| Cảnh (zangzang1303) | | **Role 3 — Frontend & Chatbot Dev:** Task 2 (crawl tin tức), Task 5 (semantic search + HyDE), Task 8 (PageIndex vectorless fallback), Task 10 (generation có citation), giao diện Streamlit chatbot | ✅ Hoàn thành |
| (ngovan15121977-bit) | | **Role 4 — Evaluation & QA Engineer:** Task 3 (convert Markdown), Task 6 (lexical search BM25/TF-IDF), golden_dataset.json 15 Q&A, eval_pipeline.py, báo cáo A/B results.md, chạy pytest | ✅ Hoàn thành |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```
