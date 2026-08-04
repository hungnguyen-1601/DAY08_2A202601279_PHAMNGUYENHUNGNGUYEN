"""
Task 1 - Thu thap van ban chinh sach/quy dinh dich vu dai hoc.

Huong dan:
    1. Tim toi thieu 3 van ban chinh sach (PDF/DOCX) tu trang cong khai cua mot truong dai hoc.
    2. Tai ve va luu vao data/landing/legal/
    3. Dat ten file ro rang, khong dau, mo ta dung noi dung.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
LEGAL_DOCS = [
    (
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/documents/pdfs/study-at-rmit/tuition-fees/student-fees-and-charges-guide-06-2026.pdf",
        "student-fees-and-charges-guide-rmit-2026.pdf",
    ),
    (
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/documents/pdfs/study-at-rmit/scholarships/english-pdf/rmit-university-vietnam-scholarship-terms-and-conditions.pdf",
        "scholarship-terms-and-conditions-rmit.pdf",
    ),
    (
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/documents/pdfs/students/accommodation/accommodation-advice-for-international-students-in-vietnam.pdf",
        "accommodation-advice-international-students-rmit.pdf",
    ),
]


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    filepath = DATA_DIR / filename
    if filepath.exists() and filepath.stat().st_size > 1024:
        print(f"Already exists: {filepath}")
        return filepath

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        filepath.write_bytes(response.read())

    if filepath.stat().st_size <= 1024:
        raise ValueError(f"Downloaded file is too small and may be invalid: {filepath}")

    print(f"Downloaded: {filepath}")
    return filepath


def download_all():
    setup_directory()
    with ThreadPoolExecutor(max_workers=min(8, len(LEGAL_DOCS))) as executor:
        list(executor.map(lambda item: download_file(*item), LEGAL_DOCS))


if __name__ == "__main__":
    download_all()
