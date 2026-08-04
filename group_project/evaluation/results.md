# RAG Evaluation Results

## Ý nghĩa các metric

- **Hit@5:** tỷ lệ câu hỏi lấy được đúng tài liệu trong 5 kết quả đầu.
- **MRR:** nguồn đúng càng gần top 1 thì điểm càng cao.
- **Context Recall:** tỷ lệ từ khóa kỳ vọng xuất hiện trong context retrieval.
- **Citation Presence/Validity:** câu trả lời có dẫn nguồn và nguồn có thật trong context.
- **Abstention Accuracy:** tỷ lệ câu ngoài phạm vi được từ chối an toàn.

## A/B Retrieval

| Metric | Hybrid + RRF + fallback | Dense-only | Delta | Winner |
|---|---:|---:|---:|---|
| Hit@5 | 0.0000 | 0.0000 | +0.0000 | tie |
| MRR | 0.0000 | 0.0000 | +0.0000 | tie |
| Context Recall | 0.5714 | 0.1429 | +0.4286 | hybrid_rerank_fallback |
| Latency ms | 13629.1045 | 8749.2675 | +4879.8370 | dense_only |
| Source Diversity | 0.4000 | 0.2000 | +0.2000 | hybrid_rerank_fallback |

### Trạng thái cấu hình

- Hybrid errors: **0**.
- Dense-only errors: **0**.

## Generation

| Metric | Score | Hướng đọc |
|---|---:|---|
| citation_presence | 0.0000 | Thấp: prompt chưa buộc citation đủ mạnh. |
| citation_validity | 0.0000 | Dưới 1.0: có citation không thuộc context. |
| keyword_answer_recall | 0.0952 | Thấp: câu trả lời thiếu ý chính. |
| abstention_accuracy | 0.0000 | Thấp: nguy cơ trả lời ngoài phạm vi. |

## DeepSeek LLM Judge

> Skipped: DEEPSEEK_API_KEY is not configured

## Bottom 3 retrieval cases

| Case | Hit | MRR | Context Recall | Nguyên nhân gợi ý |
|---|---:|---:|---:|---|
| course_registration_002 | 0 | 0.000 | 0.000 | Sai/thiếu nguồn |
| dormitory_002 | 0 | 0.000 | 0.333 | Sai/thiếu nguồn |
| dormitory_003 | 0 | 0.000 | 0.333 | Sai/thiếu nguồn |

## Hướng cải thiện

1. Hit@5/MRR thấp: kiểm tra chunking, query expansion và trọng số fusion.
2. Context Recall thấp: tăng top_k có kiểm soát hoặc cải thiện metadata/heading.
3. Citation thấp: siết system prompt và kiểm tra citation sau generation.
4. Hybrid chậm: cache embedding, BM25 corpus và PageIndex.

> Chi tiết từng test case nằm trong `evaluation_results.json`.
