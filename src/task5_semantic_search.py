"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def _load_embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence_transformers is required for semantic_search. "
            "Install it with `pip install sentence-transformers`."
        ) from exc

    return SentenceTransformer(EMBEDDING_MODEL)


def _get_chroma_collection():
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "chromadb is required for semantic_search. "
            "Install it with `pip install chromadb`."
        ) from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except ValueError:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def _generate_hypothetical_doc(query: str) -> str:
    """Generate a HyDE-style hypothetical document for improved retrieval."""
    query = query.strip()
    if not query:
        return ""

    prompt = (
        "Generate a concise and informative hypothetical document that answers the following user query "
        "as if it were written by a university policy and services knowledge base. "
        "Focus on the relevant concepts and terminology in a factual style.\n\n"
        f"User query: {query}\n\nHypothetical document:"
    )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            import openai

            openai.api_key = openai_api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a concise document writer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=180,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass

    # Fallback if OpenAI isn't available or not configured.
    return (
        "This document discusses the university policy and service details relevant to the query: "
        f"{query}. It covers tuition fees, scholarship eligibility, registration, accommodation, library services, "
        "and student support information as appropriate for the question."
    )


def _embed_text(text: str, model: Any = None) -> list[float]:
    if model is None:
        model = _load_embedding_model()

    if not text or not text.strip():
        return []

    emb = model.encode([text], show_progress_bar=False, batch_size=1)[0]
    if hasattr(emb, "tolist"):
        return emb.tolist()
    return list(emb)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Semantic search using HyDE and cosine similarity over the ChromaDB vector store.
    """
    query = (query or "").strip()
    if not query:
        return []

    hypothetical_doc = _generate_hypothetical_doc(query)
    try:
        embedding_vector = _embed_text(hypothetical_doc)
    except Exception:
        return []
    if not embedding_vector:
        return []

    try:
        collection = _get_chroma_collection()
        results = collection.query(
            query_embeddings=[embedding_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        score = 1.0 - float(distance)
        if score < 0:
            score = 0.0
        output.append(
            {
                "content": content,
                "score": round(score, 4),
                "metadata": metadata or {},
            }
        )

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    sample_query = "what is the tuition fee"
    results = semantic_search(sample_query, top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
