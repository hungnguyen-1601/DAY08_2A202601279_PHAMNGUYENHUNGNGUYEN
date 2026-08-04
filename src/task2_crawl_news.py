"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://www.rmit.edu.vn/vi/tin-tuc/tat-ca-tin-tuc/2026/jul/hien-thuc-hoa-tam-nhin-100-nam-cua-ha-noi",
    "https://www.rmit.edu.vn/vi/tin-tuc/tat-ca-tin-tuc/2026/jul/hoi-nghi-quoc-te-ban-ve-tuong-lai-cua-chuoi-cung-ung-tai-tao",
    "https://www.rmit.edu.vn/vi/tin-tuc/tat-ca-tin-tuc/2026/jul/bo-giao-duc-va-dao-tao-phoi-hop-voi-dai-hoc-rmit-nang-cao-chat-luong-thiet-ke-hoc-truc-tuyen",
    "https://www.rmit.edu.vn/vi/tin-tuc/tat-ca-tin-tuc/2026/jul/de-xuat-bao-ve-tre-em-tren-mang-xa-hoi-khong-chi-la-cau-chuyen-an-toan",
    "https://www.rmit.edu.vn/vi/tin-tuc/tat-ca-tin-tuc/2026/jul/duong-den-chu-quyen-ai-chuyen-mon-hoa-thay-vi-chay-dua",
]


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_title(soup: BeautifulSoup) -> str:
    title_tag = soup.find("h1")
    if title_tag and title_tag.get_text(strip=True):
        return _clean_text(title_tag.get_text(separator=" ", strip=True))

    if soup.title and soup.title.string:
        return _clean_text(soup.title.string)

    return "Untitled Article"


def _extract_article_content(soup: BeautifulSoup) -> str:
    article = soup.find("article")
    if article:
        paragraphs = [
            _clean_text(p.get_text(separator=" ", strip=True))
            for p in article.find_all(["p", "li"])
            if _clean_text(p.get_text(separator=" ", strip=True))
        ]
        if paragraphs:
            return "\n\n".join(paragraphs)

    selectors = [
        "div[class*='content']",
        "div[class*='article']",
        "main",
        "section[class*='content']",
    ]

    for selector in selectors:
        region = soup.select_one(selector)
        if region:
            paragraphs = [
                _clean_text(p.get_text(separator=" ", strip=True))
                for p in region.find_all(["p", "li"])
                if _clean_text(p.get_text(separator=" ", strip=True))
            ]
            if paragraphs:
                return "\n\n".join(paragraphs)

    paragraphs = [
        _clean_text(p.get_text(separator=" ", strip=True))
        for p in soup.find_all("p")
        if _clean_text(p.get_text(separator=" ", strip=True))
    ]

    return "\n\n".join(paragraphs)


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    title = _extract_title(soup)
    content = _extract_article_content(soup)

    if not content:
        raise RuntimeError(f"Không thể trích xuất nội dung từ {url}")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
