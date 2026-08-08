"""Explore Aston University site for Kenya postgraduate entry requirements."""
import re
from playwright.sync_api import sync_playwright

BASE = "https://www.aston.ac.uk"
OUT = open("aston_explore_output.txt", "w", encoding="utf-8")

def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    OUT.write(msg + "\n")
    OUT.flush()

def goto_robust(page, url, timeout=60000):
    strategies = ["domcontentloaded", "commit", "load"]
    for s in strategies:
        try:
            resp = page.goto(url, timeout=timeout, wait_until=s)
            status = resp.status if resp else "none"
            log(f"[nav {s}] status={status}")
            return True
        except Exception as exc:
            log(f"[nav {s}] failed: {type(exc).__name__}: {str(exc)[:100]}")
            page.wait_for_timeout(3000)
    return False

def get_links(page, url, pattern, wait_ms=7000):
    ok = goto_robust(page, url)
    if not ok:
        log(f"Could not navigate to {url}")
        return
    page.wait_for_timeout(wait_ms)
    log(f"\n=== {url} ===")
    try:
        log("TITLE:", page.title())
        links = page.eval_on_selector_all(
            "a[href]",
            f"els => {{ const re = new RegExp({pattern!r}, 'i'); return els.filter(a => re.test(a.href + ' ' + (a.innerText||''))).map(a => [a.innerText.trim().slice(0,70), a.href]); }}",
        )
        seen = set()
        for t, h in links:
            if h not in seen:
                seen.add(h)
                log(f"{t or '(no text)':<70} => {h}")
    except Exception as exc:
        log("Link extraction failed:", exc)

def dump_text(page, url, wait_ms=8000):
    ok = goto_robust(page, url)
    if not ok:
        log(f"Could not navigate to {url}")
        return
    page.wait_for_timeout(wait_ms)
    log(f"\n=== TEXT DUMP {url} ===")
    try:
        log("TITLE:", page.title())
        text = page.inner_text("body")
        log(f"PAGE TEXT LENGTH: {len(text)}")
        keywords = ["kenya", "entry requirement", "qualification", "postgraduate",
                     "english", "bachelor", "degree", "upper second", "2:1",
                     "second class", "msc", "master", "country", "africa"]
        for kw in keywords:
            for m in re.finditer(kw, text, re.I):
                s = max(0, m.start() - 200)
                e = min(len(text), m.end() + 300)
                log(f"--- {kw.upper()} ---")
                log(text[s:e].replace("\n", " "))
                break
    except Exception as exc:
        log("Text dump failed:", exc)

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    pg.set_default_timeout(60000)

    # Africa page - look for Kenya
    get_links(pg, f"{BASE}/international/aston-in-your-country/africa", r"kenya|east|country|all")
    dump_text(pg, f"{BASE}/international/aston-in-your-country/africa")
    
    # Postgraduate course page
    get_links(pg, f"{BASE}/postgraduate/courses", r"msc|master|course|data|computer|business|management|finance")
    
    # International students page for entry requirements
    dump_text(pg, f"{BASE}/international/students-applying-to-aston")
    
    # Try to find course finder page
    dump_text(pg, f"{BASE}/courses")
    
    # Look for specific course pages
    get_links(pg, f"{BASE}/postgraduate/courses", r"msc|business|management|data|computer|cyber|finance|marketing|hr|supply|logistics|project")
    
    b.close()

OUT.close()
print("Done! Check aston_explore_output.txt")
