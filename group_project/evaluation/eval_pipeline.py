"""Two-layer evaluation for the e-commerce RAG pipeline.

Offline metrics are deterministic and require no network. RAGAS is optional
and runs only when an OpenAI-compatible judge is configured.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Callable


EVALUATION_DIR = Path(__file__).parent
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_JSON_PATH = EVALUATION_DIR / "evaluation_results.json"
RESULTS_PATH = EVALUATION_DIR / "results.md"
REQUIRED_FIELDS = {
    "id", "category", "question", "expected_answer", "expected_keywords",
    "expected_sources", "expected_context", "is_answerable",
}
UNVERIFIED_MESSAGE = "I cannot verify this information"
CITATION_PATTERN = re.compile(r"\[([^\[\],]+),\s*([^\[\]]+)\]")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    validation = validate_golden_dataset(dataset)
    if validation["errors"]:
        raise ValueError("Invalid golden dataset: " + "; ".join(validation["errors"]))
    return dataset


def validate_golden_dataset(dataset: object) -> dict:
    """Return case count, category distribution and every schema error."""
    errors: list[str] = []
    if not isinstance(dataset, list):
        return {"case_count": 0, "categories": {}, "errors": ["root must be a list"]}
    if len(dataset) < 15:
        errors.append(f"dataset needs at least 15 cases, found {len(dataset)}")
    ids: list[str] = []
    categories: Counter[str] = Counter()
    ood_count = 0
    for index, case in enumerate(dataset):
        if not isinstance(case, dict):
            errors.append(f"case {index} must be an object")
            continue
        missing = REQUIRED_FIELDS - case.keys()
        if missing:
            errors.append(f"case {index} missing: {', '.join(sorted(missing))}")
            continue
        ids.append(str(case["id"]))
        categories[str(case["category"])] += 1
        if not isinstance(case["expected_keywords"], list):
            errors.append(f"{case['id']}: expected_keywords must be a list")
        if not isinstance(case["expected_sources"], list):
            errors.append(f"{case['id']}: expected_sources must be a list")
        if not isinstance(case["is_answerable"], bool):
            errors.append(f"{case['id']}: is_answerable must be boolean")
        if case["is_answerable"] is False:
            ood_count += 1
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append("duplicate ids: " + ", ".join(duplicates))
    if len(categories) < 5:
        errors.append("dataset must cover at least 5 categories")
    if ood_count == 0:
        errors.append("dataset must include out-of-domain cases")
    return {"case_count": len(dataset), "categories": dict(categories), "errors": errors}


def _source_name(result: dict) -> str:
    return str((result.get("metadata") or {}).get("source", ""))


def _keyword_recall(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    normalized = _normalize(text)
    return sum(_normalize(keyword) in normalized for keyword in keywords) / len(keywords)


def evaluate_retrieval(
    retriever: Callable[..., list[dict]], dataset: list[dict], top_k: int = 5
) -> dict:
    """Evaluate source hit, reciprocal rank, context coverage and latency."""
    rows = []
    for case in dataset:
        started = time.perf_counter()
        error = None
        try:
            results = retriever(case["question"], top_k=top_k)
        except Exception as exc:
            results = []
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        expected = set(case["expected_sources"])
        sources = [_source_name(item) for item in results]
        first_rank = next((rank for rank, source in enumerate(sources, 1) if source in expected), None)
        hit = 1.0 if (first_rank is not None or not case["is_answerable"]) else 0.0
        reciprocal_rank = 1.0 / first_rank if first_rank else (1.0 if not case["is_answerable"] and not results else 0.0)
        context = "\n".join(str(item.get("content", "")) for item in results)
        rows.append({
            "id": case["id"], "category": case["category"], "hit_at_k": hit,
            "reciprocal_rank": reciprocal_rank,
            "keyword_context_recall": _keyword_recall(context, case["expected_keywords"]),
            "latency_ms": latency_ms, "result_count": len(results),
            "source_diversity": len(set(sources)) / len(results) if results else 0.0,
            "retrieved_sources": sources, "error": error,
        })
    answerable_rows = [row for row, case in zip(rows, dataset) if case["is_answerable"]]
    scored = answerable_rows or rows
    return {
        "summary": {
            "hit_at_k": mean(row["hit_at_k"] for row in scored) if scored else 0.0,
            "mrr": mean(row["reciprocal_rank"] for row in scored) if scored else 0.0,
            "keyword_context_recall": mean(row["keyword_context_recall"] for row in scored) if scored else 0.0,
            "mean_latency_ms": mean(row["latency_ms"] for row in rows) if rows else 0.0,
            "source_diversity": mean(row["source_diversity"] for row in scored) if scored else 0.0,
            "error_count": sum(row["error"] is not None for row in rows),
        },
        "cases": rows,
    }


def _citation_sources(answer: str) -> list[str]:
    return [match.group(1).strip() for match in CITATION_PATTERN.finditer(answer)]


def evaluate_generation(
    generator: Callable[[str], dict], dataset: list[dict]
) -> dict:
    """Evaluate citations, answer coverage and safe out-of-domain abstention."""
    rows = []
    for case in dataset:
        started = time.perf_counter()
        error = None
        try:
            response = generator(case["question"])
        except Exception as exc:
            response = {"answer": "", "sources": []}
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        answer = str(response.get("answer", ""))
        source_names = {_source_name(item) for item in response.get("sources", [])}
        citations = _citation_sources(answer)
        citation_presence = 1.0 if citations else 0.0
        citation_validity = (
            sum(citation in source_names for citation in citations) / len(citations)
            if citations else 0.0
        )
        abstained = _normalize(UNVERIFIED_MESSAGE) in _normalize(answer)
        rows.append({
            "id": case["id"], "category": case["category"],
            "citation_presence": citation_presence,
            "citation_validity": citation_validity,
            "keyword_answer_recall": _keyword_recall(answer, case["expected_keywords"]),
            "abstained": abstained, "is_answerable": case["is_answerable"],
            "latency_ms": latency_ms, "answer": answer,
            "source_names": sorted(source_names), "citations": citations, "error": error,
        })
    answerable = [row for row in rows if row["is_answerable"]]
    ood = [row for row in rows if not row["is_answerable"]]
    return {
        "summary": {
            "citation_presence": mean(row["citation_presence"] for row in answerable) if answerable else 0.0,
            "citation_validity": mean(row["citation_validity"] for row in answerable) if answerable else 0.0,
            "keyword_answer_recall": mean(row["keyword_answer_recall"] for row in answerable) if answerable else 0.0,
            "abstention_accuracy": mean(row["abstained"] for row in ood) if ood else 0.0,
            "mean_latency_ms": mean(row["latency_ms"] for row in rows) if rows else 0.0,
            "error_count": sum(row["error"] is not None for row in rows),
        },
        "cases": rows,
    }


DEEPSEEK_JUDGE_PROMPT = """Bạn là giám khảo độc lập đánh giá một hệ thống RAG.
Chấm bốn metric từ 0.0 đến 1.0:
1. faithfulness: mọi khẳng định trong ANSWER có được CONTEXT hỗ trợ không.
2. answer_relevance: ANSWER có trả lời trực tiếp QUESTION không.
3. context_recall: CONTEXT có chứa đủ thông tin trong EXPECTED ANSWER không.
4. context_precision: bao nhiêu phần CONTEXT thực sự hữu ích cho QUESTION.

Chỉ trả về một JSON object hợp lệ, không Markdown, đúng schema:
{"faithfulness": 0.0, "answer_relevance": 0.0, "context_recall": 0.0,
 "context_precision": 0.0, "reason": "giải thích ngắn bằng tiếng Việt"}
"""


def _parse_judge_response(text: str) -> dict:
    """Parse and validate the bounded score object returned by DeepSeek."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("DeepSeek judge did not return a JSON object")
    payload = json.loads(match.group(0))
    required = {"faithfulness", "answer_relevance", "context_recall", "context_precision"}
    missing = required - payload.keys()
    if missing:
        raise ValueError("DeepSeek judge missing metrics: " + ", ".join(sorted(missing)))
    for name in required:
        score = float(payload[name])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"DeepSeek judge score outside 0..1: {name}={score}")
        payload[name] = score
    payload["reason"] = str(payload.get("reason", ""))
    return payload


def evaluate_with_deepseek(generator: Callable[[str], dict], dataset: list[dict]) -> dict:
    """Use DeepSeek as an LLM judge; one request scores all four metrics."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return {"status": "skipped", "reason": "DEEPSEEK_API_KEY is not configured"}
    from openai import OpenAI

    model = os.getenv("DEEPSEEK_JUDGE_MODEL", "deepseek-chat")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    rows = []
    errors = []
    for case in dataset:
        if not case["is_answerable"]:
            continue
        started = time.perf_counter()
        try:
            generated = generator(case["question"])
            contexts = [str(item.get("content", "")) for item in generated.get("sources", [])]
            context_text = "\n\n---\n\n".join(contexts)[:12_000]
            user_prompt = (
                f"QUESTION:\n{case['question']}\n\n"
                f"ANSWER:\n{generated.get('answer', '')}\n\n"
                f"EXPECTED ANSWER:\n{case['expected_answer']}\n\n"
                f"CONTEXT:\n{context_text or '[EMPTY CONTEXT]'}"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DEEPSEEK_JUDGE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                stream=False,
            )
            raw = response.choices[0].message.content or ""
            scores = _parse_judge_response(raw)
            rows.append({
                "id": case["id"], "category": case["category"], **scores,
                "latency_ms": (time.perf_counter() - started) * 1000,
            })
        except Exception as exc:
            errors.append({"id": case["id"], "error": f"{type(exc).__name__}: {exc}"})
    metric_names = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    summary = {
        name: mean(row[name] for row in rows) if rows else 0.0
        for name in metric_names
    }
    return {
        "status": "completed" if not errors else "partial",
        "provider": "deepseek", "model": model, "summary": summary,
        "case_count": len(rows), "expected_case_count": sum(case["is_answerable"] for case in dataset),
        "cases": rows, "errors": errors,
    }


def compare_configs(
    dataset: list[dict], hybrid_retriever: Callable[..., list[dict]],
    dense_retriever: Callable[..., list[dict]], top_k: int = 5,
) -> dict:
    configs = {
        "hybrid_rerank_fallback": evaluate_retrieval(hybrid_retriever, dataset, top_k),
        "dense_only": evaluate_retrieval(dense_retriever, dataset, top_k),
    }
    first = configs["hybrid_rerank_fallback"]["summary"]
    second = configs["dense_only"]["summary"]
    keys = ["hit_at_k", "mrr", "keyword_context_recall", "mean_latency_ms", "source_diversity"]
    delta = {key: first[key] - second[key] for key in keys}
    winner = {}
    for key in keys:
        if first["error_count"] or second["error_count"]:
            winner[key] = "not_comparable"
        elif first[key] == second[key]:
            winner[key] = "tie"
        elif key == "mean_latency_ms":
            winner[key] = "hybrid_rerank_fallback" if first[key] < second[key] else "dense_only"
        else:
            winner[key] = "hybrid_rerank_fallback" if first[key] > second[key] else "dense_only"
    return {"configs": configs, "delta": delta, "winner": winner}


def export_results(results: dict, comparison: dict, path: Path = RESULTS_PATH) -> None:
    payload = {"offline": results, "comparison": comparison}
    RESULTS_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    a = comparison["configs"]["hybrid_rerank_fallback"]["summary"]
    b = comparison["configs"]["dense_only"]["summary"]
    generation = results.get("generation", {}).get("summary", {})
    answerable_ids = {
        row["id"] for row in results.get("generation", {}).get("cases", [])
        if row.get("is_answerable", True)
    }
    retrieval_cases = comparison["configs"]["hybrid_rerank_fallback"]["cases"]
    failed_cases = sorted(
        [row for row in retrieval_cases if not answerable_ids or row["id"] in answerable_ids],
        key=lambda row: (row["hit_at_k"], row["keyword_context_recall"], row["reciprocal_rank"]),
    )[:3]
    lines = [
        "# RAG Evaluation Results", "", "## Ý nghĩa các metric", "",
        "- **Hit@5:** tỷ lệ câu hỏi lấy được đúng tài liệu trong 5 kết quả đầu.",
        "- **MRR:** nguồn đúng càng gần top 1 thì điểm càng cao.",
        "- **Context Recall:** tỷ lệ từ khóa kỳ vọng xuất hiện trong context retrieval.",
        "- **Citation Presence/Validity:** câu trả lời có dẫn nguồn và nguồn có thật trong context.",
        "- **Abstention Accuracy:** tỷ lệ câu ngoài phạm vi được từ chối an toàn.", "",
        "## A/B Retrieval", "",
        "| Metric | Hybrid + RRF + fallback | Dense-only | Delta | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label in [("hit_at_k", "Hit@5"), ("mrr", "MRR"), ("keyword_context_recall", "Context Recall"), ("mean_latency_ms", "Latency ms"), ("source_diversity", "Source Diversity")]:
        lines.append(f"| {label} | {a[key]:.4f} | {b[key]:.4f} | {comparison['delta'][key]:+.4f} | {comparison['winner'][key]} |")
    lines += [
        "", "### Trạng thái cấu hình", "",
        f"- Hybrid errors: **{a['error_count']}**.",
        f"- Dense-only errors: **{b['error_count']}**.",
    ]
    if a["error_count"] or b["error_count"]:
        lines.append("- A/B được đánh dấu `not_comparable`; cài đủ dependencies/index rồi chạy lại trước khi kết luận config thắng.")
    lines += ["", "## Generation", "", "| Metric | Score | Hướng đọc |", "|---|---:|---|"]
    descriptions = {
        "citation_presence": "Thấp: prompt chưa buộc citation đủ mạnh.",
        "citation_validity": "Dưới 1.0: có citation không thuộc context.",
        "keyword_answer_recall": "Thấp: câu trả lời thiếu ý chính.",
        "abstention_accuracy": "Thấp: nguy cơ trả lời ngoài phạm vi.",
    }
    for key, description in descriptions.items():
        lines.append(f"| {key} | {generation.get(key, 0):.4f} | {description} |")
    judge = results.get("deepseek_judge", {})
    lines += ["", "## DeepSeek LLM Judge", ""]
    if judge.get("status") == "skipped":
        lines.append(f"> Skipped: {judge.get('reason', 'không rõ nguyên nhân')}")
    else:
        lines += [
            f"- Status: **{judge.get('status', 'unknown')}**",
            f"- Model: **{judge.get('model', 'unknown')}**",
            f"- Cases: **{judge.get('case_count', 0)}/{judge.get('expected_case_count', 0)}**", "",
            "| Metric | Score | Ý nghĩa |", "|---|---:|---|",
        ]
        judge_meanings = {
            "faithfulness": "Câu trả lời có bám bằng chứng retrieval không.",
            "answer_relevance": "Câu trả lời có trực tiếp giải quyết câu hỏi không.",
            "context_recall": "Context có đủ ý từ đáp án kỳ vọng không.",
            "context_precision": "Tỷ lệ context thực sự hữu ích cho câu hỏi.",
        }
        for key, meaning in judge_meanings.items():
            lines.append(f"| {key} | {judge.get('summary', {}).get(key, 0):.4f} | {meaning} |")
    lines += ["", "## Bottom 3 retrieval cases", "", "| Case | Hit | MRR | Context Recall | Nguyên nhân gợi ý |", "|---|---:|---:|---:|---|"]
    for row in failed_cases:
        reason = "Sai/thiếu nguồn" if not row["hit_at_k"] else "Nguồn đúng xếp thấp hoặc context thiếu keyword"
        lines.append(f"| {row['id']} | {row['hit_at_k']:.0f} | {row['reciprocal_rank']:.3f} | {row['keyword_context_recall']:.3f} | {reason} |")
    lines += [
        "", "## Hướng cải thiện", "",
        "1. Hit@5/MRR thấp: kiểm tra chunking, query expansion và trọng số fusion.",
        "2. Context Recall thấp: tăng top_k có kiểm soát hoặc cải thiện metadata/heading.",
        "3. Citation thấp: siết system prompt và kiểm tra citation sau generation.",
        "4. Hybrid chậm: cache embedding, BM25 corpus và PageIndex.", "",
        "> Chi tiết từng test case nằm trong `evaluation_results.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from src.task10_generation import generate_with_citation
    from src.task5_semantic_search import semantic_search
    from src.task9_retrieval_pipeline import retrieve

    dataset = load_golden_dataset()
    hybrid = evaluate_retrieval(retrieve, dataset)
    generation = evaluate_generation(generate_with_citation, dataset)
    comparison = compare_configs(dataset, retrieve, semantic_search)
    deepseek_judge = evaluate_with_deepseek(generate_with_citation, dataset)
    export_results({"retrieval": hybrid, "generation": generation, "deepseek_judge": deepseek_judge}, comparison)
    print(f"Evaluated {len(dataset)} cases")
    print(f"Hit@5={hybrid['summary']['hit_at_k']:.3f} MRR={hybrid['summary']['mrr']:.3f}")
    print(f"DeepSeek judge status: {deepseek_judge['status']}")
    print(f"Reports: {RESULTS_JSON_PATH} and {RESULTS_PATH}")


if __name__ == "__main__":
    main()