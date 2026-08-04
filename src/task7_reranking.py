"""
Task 7 - Reranking Module.

This module implements local reranking methods for the RAG pipeline. The main
method is Reciprocal Rank Fusion (RRF), which merges ranked lists from dense
semantic search and lexical search without mixing incompatible raw score scales.

Important for Task 9: RRF scores are fusion scores based on rank only. Do not
use them as semantic relevance scores for fallback thresholds.
"""

from math import sqrt


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder reranking placeholder.

    Task 7 uses RRF by default because it is local, fast, and does not need an API key.
    """
    raise NotImplementedError("Cross-encoder reranking is not configured; use method='rrf'.")


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance: select candidates that are relevant and diverse.
    """
    if not query_embedding or not candidates or top_k <= 0:
        return []

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            embedding = candidates[idx].get("embedding", [])
            relevance = _cosine_sim(query_embedding, embedding)
            redundancy = 0.0
            for selected_idx in selected:
                redundancy = max(
                    redundancy,
                    _cosine_sim(embedding, candidates[selected_idx].get("embedding", [])),
                )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["score"] = float(item.get("score", 0.0))
        results.append(item)
    return results


def _candidate_key(item: dict) -> str:
    metadata = item.get("metadata") or {}
    return str(metadata.get("doc_id") or metadata.get("source") or item.get("content", ""))


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Merge ranked lists with Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank_r(d)))
    """
    if not ranked_lists or top_k <= 0:
        return []

    rrf_scores = {}
    best_items = {}
    best_original_scores = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = _candidate_key(item)
            if not key:
                continue

            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            original_score = float(item.get("score", 0.0))
            if key not in best_items or original_score > best_original_scores[key]:
                best_items[key] = item
                best_original_scores[key] = original_score

    sorted_keys = sorted(
        rrf_scores,
        key=lambda key: (rrf_scores[key], best_original_scores.get(key, 0.0)),
        reverse=True,
    )

    results = []
    for key in sorted_keys[:top_k]:
        item = best_items[key].copy()
        item["score"] = rrf_scores[key]
        item["rrf_score"] = rrf_scores[key]
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """
    Unified reranking interface.

    For the default RRF method, a single candidate list is sorted by its existing
    retrieval score and fused as one ranked list. For hybrid dense+lexical fusion,
    call rerank_rrf([dense_results, sparse_results], top_k=...).
    """
    if not candidates or top_k <= 0:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding for MMR.")
    if method == "rrf":
        ranked_candidates = sorted(
            candidates,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        return rerank_rrf([ranked_candidates], top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dense_results = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {"doc_id": "tuition"}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {"doc_id": "scholarship"}},
    ]
    lexical_results = [
        {"content": "Scholarship eligibility requirements", "score": 4.2, "metadata": {"doc_id": "scholarship"}},
        {"content": "Tuition fee payment schedule", "score": 3.4, "metadata": {"doc_id": "tuition"}},
    ]
    for result in rerank_rrf([dense_results, lexical_results], top_k=2):
        print(f"[{result['score']:.4f}] {result['content']}")
