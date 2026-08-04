"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# TODO: Chọn LLM model (OpenRouter model ID)
LLM_MODEL = "openai/gpt-4o-mini"  # hoặc model ":free" nếu chưa có credit


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    Strategy: giữ các chunk có điểm cao ở đầu và cuối prompt, đặt chunk kém quan trọng ở giữa.
    """
    if not chunks:
        return []
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]
    back = chunks[1::2][::-1]
    return front + back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.
    """
    if not chunks:
        return ""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source") or metadata.get("doc_id") or metadata.get("source_file") or f"Source {i}"
        doc_type = metadata.get("type") or metadata.get("doc_type") or "unknown"
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n{content}"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def _fallback_keyword_search(query: str, top_k: int) -> list[dict]:
    """Fallback retrieval using local markdown files when semantic/lexical APIs are unavailable."""
    if not DATA_DIR.exists():
        return []

    query_terms = [term for term in re.findall(r"\w+", query.lower()) if term]
    if not query_terms:
        return []

    results = []
    for md_file in sorted(DATA_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        text = content.lower()
        score = sum(1 for term in query_terms if term in text)
        if score <= 0:
            continue

        rel_path = md_file.relative_to(ROOT_DIR).as_posix()
        results.append(
            {
                "content": content[:2500],
                "score": float(score),
                "metadata": {
                    "source": rel_path,
                    "type": "local_fallback",
                },
                "source": "local",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _retrieve_chunks(query: str, top_k: int) -> list[dict]:
    """Retrieve relevant chunks using task 9 if available, otherwise local fallback."""
    try:
        from .task9_retrieval_pipeline import retrieve as task9_retrieve

        chunks = task9_retrieve(query, top_k=top_k)
        if chunks:
            return chunks
    except Exception:
        pass

    try:
        from .task5_semantic_search import semantic_search

        dense_results = semantic_search(query, top_k=max(top_k * 2, 5))
    except Exception:
        dense_results = []

    try:
        from .task6_lexical_search import lexical_search

        sparse_results = lexical_search(query, top_k=max(top_k * 2, 5))
    except Exception:
        sparse_results = []

    combined = []
    seen = set()
    for item in dense_results + sparse_results:
        content = (item.get("content") or "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        combined.append(
            {
                **item,
                "source": item.get("source") or "hybrid",
                "metadata": item.get("metadata") or {},
            }
        )

    if combined:
        combined.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return combined[:top_k]

    return _fallback_keyword_search(query, top_k)


def _call_llm(prompt: str) -> str:
    """Try OpenRouter/OpenAI-compatible API, then fall back to a deterministic local response."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No API key configured")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        try:
            import openai

            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            raise RuntimeError("LLM call failed") from exc


def _build_fallback_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return 'Tôi không thể xác minh thông tin này từ nguồn hiện có.'

    first = chunks[0]
    metadata = first.get("metadata") or {}
    source = metadata.get("source") or metadata.get("doc_id") or metadata.get("source_file") or "nguồn được cung cấp"
    excerpt = (first.get("content") or "").strip().replace("\n", " ")
    excerpt = excerpt[:280]

    if len(chunks) == 1:
        return f"Dựa trên nguồn {source}, thông tin liên quan là: {excerpt} [Source: {source}]"

    second = chunks[1]
    second_meta = second.get("metadata") or {}
    second_source = second_meta.get("source") or second_meta.get("doc_id") or second_meta.get("source_file") or "nguồn được cung cấp"
    second_excerpt = (second.get("content") or "").strip().replace("\n", " ")
    second_excerpt = second_excerpt[:180]
    return (
        f"Dựa trên các nguồn được cung cấp, thông tin liên quan là: {excerpt} [Source: {source}] "
        f"và {second_excerpt} [Source: {second_source}]"
    )


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.
    """
    chunks = _retrieve_chunks(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""

    try:
        answer = _call_llm(user_message)
    except Exception:
        answer = _build_fallback_answer(query, reordered)

    retrieval_source = "hybrid" if reordered else "none"
    if reordered and all(item.get("source") == "pageindex" for item in reordered):
        retrieval_source = "pageindex"

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
