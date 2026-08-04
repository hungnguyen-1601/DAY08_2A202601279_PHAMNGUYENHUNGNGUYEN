"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fpdf import FPDF
from pageindex.client import PageIndexAPIError, PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PAGEINDEX_DATA_DIR = Path(__file__).parent.parent / "data" / "pageindex"
PDF_DIR = PAGEINDEX_DATA_DIR / "pdfs"
DOC_INDEX_FILE = PAGEINDEX_DATA_DIR / "doc_index.json"


def _ensure_api_key() -> None:
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY is not set in .env")


def _ensure_dirs() -> None:
    PAGEINDEX_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def _wrap_text(pdf: FPDF, text: str, max_width: float) -> List[str]:
    words = text.split(" ")
    lines: List[str] = []
    current = []

    for word in words:
        if not current:
            current.append(word)
            continue

        candidate = " ".join(current + [word])
        if pdf.get_string_width(candidate) <= max_width:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    page_width = pdf.w - 2 * pdf.l_margin
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            pdf.ln(5)
            continue

        if pdf.get_string_width(line) <= page_width:
            pdf.cell(0, 6, line, ln=1)
            continue

        for wrapped in _wrap_text(pdf, line, page_width):
            pdf.cell(0, 6, wrapped, ln=1)

    pdf.output(str(pdf_path))


def _save_doc_index(index: Dict[str, Dict[str, str]]) -> None:
    _ensure_dirs()
    DOC_INDEX_FILE.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_doc_index() -> Dict[str, Dict[str, str]]:
    if not DOC_INDEX_FILE.exists():
        return {}
    return json.loads(DOC_INDEX_FILE.read_text(encoding="utf-8"))


def _wait_for_retrieval(
    client: PageIndexClient,
    retrieval_id: str,
    timeout: float = 30.0,
    interval: float = 1.0,
) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        retrieval = client.get_retrieval(retrieval_id)
        status = str(retrieval.get("status", "")).lower()

        if status in {"completed", "ready", "succeeded"}:
            return retrieval

        if retrieval.get("retrieved_nodes"):
            return retrieval

        time.sleep(interval)

    return retrieval


def _parse_retrieval_nodes(
    retrieval: Dict[str, Any], doc_meta: Dict[str, str]
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    rank = 1

    for node in retrieval.get("retrieved_nodes", []):
        for group in node.get("relevant_contents", []) or []:
            for item in group or []:
                content = item.get("relevant_content") or item.get("content") or ""
                if not content:
                    continue

                section = item.get("section_title") or item.get("title") or ""
                score = 1.0 / (rank + 1)
                results.append(
                    {
                        "content": content,
                        "score": float(score),
                        "metadata": {
                            "doc_id": doc_meta.get("doc_id", ""),
                            "source_file": doc_meta.get("source_file", ""),
                            "section": section,
                        },
                        "source": "pageindex",
                    }
                )
                rank += 1

    return results


def upload_documents() -> Dict[str, Dict[str, str]]:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    _ensure_api_key()
    _ensure_dirs()

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_index = _load_doc_index()

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        rel_path = str(md_file.relative_to(STANDARDIZED_DIR))
        if rel_path in doc_index:
            print(f"Skipping existing document: {rel_path}")
            continue

        pdf_path = PDF_DIR / f"{md_file.stem}.pdf"
        print(f"Converting markdown to PDF: {rel_path}")
        _markdown_to_pdf(md_file, pdf_path)

        print(f"Uploading to PageIndex: {rel_path}")
        response = client.submit_document(str(pdf_path))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return a doc_id for {rel_path}")

        doc_index[rel_path] = {
            "doc_id": str(doc_id),
            "source_file": rel_path,
            "pdf_path": str(pdf_path.relative_to(Path(__file__).parent.parent)),
        }
        print(f"  ✓ Uploaded {rel_path} -> {doc_id}")

    _save_doc_index(doc_index)
    print(f"Finished uploading {len(doc_index)} documents.")
    return doc_index


def pageindex_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.
    """
    api_key = os.getenv("PAGEINDEX_API_KEY", "")
    if not api_key:
        print("⚠ PAGEINDEX_API_KEY chưa set trong .env. PageIndex search sẽ trả về []")
        return []

    doc_index = _load_doc_index()
    if not doc_index:
        print("⚠ Không tìm thấy PageIndex document index. Chạy upload_documents() trước khi truy vấn.")
        return []

    client = PageIndexClient(api_key=api_key)
    all_results: List[Dict[str, Any]] = []

    for rel_path, doc_meta in doc_index.items():
        doc_id = doc_meta.get("doc_id")
        if not doc_id:
            continue

        try:
            query_resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = query_resp.get("retrieval_id") or query_resp.get("id")
            if not retrieval_id:
                continue

            retrieval = _wait_for_retrieval(client, retrieval_id)
            all_results.extend(_parse_retrieval_nodes(retrieval, doc_meta))
        except PageIndexAPIError as exc:
            print(f"PageIndex API error for {rel_path}: {exc}")
        except Exception as exc:
            print(f"PageIndex query failed for {rel_path}: {exc}")

    if not all_results:
        return []

    deduped: Dict[str, Dict[str, Any]] = {}
    for item in all_results:
        key = item["content"].strip()[:300]
        existing = deduped.get(key)
        if existing is None or item["score"] > existing["score"]:
            deduped[key] = item

    results = sorted(deduped.values(), key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
