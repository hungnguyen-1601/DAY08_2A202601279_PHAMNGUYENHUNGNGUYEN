"""
Task 6 — Lexical Search (BM25 & TF-IDF)

Hướng dẫn:
    1. Tokenize văn bản tiếng Việt đơn giản
    2. Tạo BM25 index bằng BM25Okapi
    3. Tạo TF-IDF index bằng TfidfVectorizer
    4. Chạy pytest tests/test_individual.py cho Task 4-6

Cài đặt:
    pip install rank-bm25 scikit-learn

Bonus: hỗ trợ cả TF-IDF ngoài BM25 mặc định — chọn qua tham số `method`.
"""

import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data" / "standardized"


def simple_vietnamese_tokenize(text: str) -> list[str]:
    """
    Tokenize tiếng Việt đơn giản: lowercase + tách theo khoảng trắng/dấu câu.
    Không dùng thư viện NLP phức tạp (underthesea/pyvi) — đủ dùng cho BM25/TF-IDF cơ bản.
    """
    text = text.lower()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)  # giữ chữ cái có dấu tiếng Việt
    tokens = text.split()
    return tokens


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ file .md trong data/standardized/ (legal + news).
    Returns:
        [{"doc_id": str, "content": str}, ...]
    """
    docs = []
    for md_file in DATA_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        docs.append({
            "doc_id": md_file.stem,
            "content": content,
        })
    return docs


def build_bm25_index(docs: list[dict]):
    """Tạo BM25 index từ danh sách document đã tokenize."""
    tokenized_corpus = [simple_vietnamese_tokenize(d["content"]) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def build_tfidf_index(docs: list[dict]):
    """Tạo TF-IDF index từ danh sách document."""
    corpus = [d["content"] for d in docs]
    vectorizer = TfidfVectorizer(
        tokenizer=simple_vietnamese_tokenize,
        lowercase=False,
        token_pattern=None,  # tắt warning vì đã dùng tokenizer tùy chỉnh
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


def lexical_search(query: str, top_k: int = 10, method: str = "bm25") -> list[dict]:
    """
    Tìm kiếm lexical trên toàn bộ document đã convert.
    Mặc định dùng BM25; truyền method="tfidf" để dùng TF-IDF (bonus).

    Args:
        query: câu truy vấn
        top_k: số kết quả trả về (mặc định 10)
        method: "bm25" (mặc định) hoặc "tfidf"

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sắp xếp giảm dần theo score
    """
    docs = load_documents()
    if not docs:
        return []

    query_tokens = simple_vietnamese_tokenize(query)

    if method == "bm25":
        bm25 = build_bm25_index(docs)
        scores = bm25.get_scores(query_tokens)
    elif method == "tfidf":
        vectorizer, tfidf_matrix = build_tfidf_index(docs)
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    else:
        raise ValueError(f"Method không hợp lệ: {method}. Chọn 'bm25' hoặc 'tfidf'.")

    results = [
        {
            "content": docs[i]["content"],
            "score": float(scores[i]),
            "metadata": {"doc_id": docs[i]["doc_id"]},
        }
        for i in range(len(docs))
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    print("=" * 50)
    print("Task 6: Lexical Search (BM25 & TF-IDF)")
    print("=" * 50)

    test_query = "học phí"
    print(f"\nQuery: '{test_query}'")

    print("\n--- BM25 (mac dinh) ---")
    for r in lexical_search(test_query, top_k=3):
        print(f"  [{r['score']:.4f}] {r['metadata']['doc_id']}")

    print("\n--- TF-IDF (bonus) ---")
    for r in lexical_search(test_query, top_k=3, method="tfidf"):
        print(f"  [{r['score']:.4f}] {r['metadata']['doc_id']}")