# check_ready.py
from pathlib import Path

legal = list(Path("data/landing/legal").glob("*"))
news = list(Path("data/landing/news").glob("*.json"))

print(f"Legal files: {len(legal)} (cần >=3)")
for f in legal:
    print(f"  - {f.name}")

print(f"News files: {len(news)} (cần >=5)")
for f in news:
    print(f"  - {f.name}")