"""Check the results of the multi-university scraper."""
import json
from pathlib import Path

files = sorted(Path("data/processed").glob("all_universities_kenya_requirements_*.json"))
if not files:
    print("No results files found!")
    exit(1)

with open(files[-1], "r") as f:
    data = json.load(f)

for r in data:
    name = r["institution"]
    text_len = r.get("text_length", 0)
    print(f"=== {name} === text_len={text_len}")
    text = r.get("page_text")
    if text:
        for kw in ["postgraduate", "post-graduate", "masters", "msc", "master", "bachelor", "degree", "honours", "second class", "2:1", "2:2", "kcse", "ielts", "english", "entry", "requirement"]:
            idx = text.lower().find(kw.lower())
            if idx >= 0:
                s = max(0, idx-80)
                e = min(len(text), idx+200)
                excerpt = text[s:e].replace("\n", " ")
                print(f"  [{kw}] ...{excerpt}...")
    else:
        print(f"  ERROR: {r.get('error', '')}")
    print()
