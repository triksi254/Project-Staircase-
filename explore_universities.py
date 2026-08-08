"""Explore multiple universities to find Kenya entry requirements pages."""
import re
from playwright.sync_api import sync_playwright

OUT = open("universities_explore_output.txt", "w", encoding="utf-8")

UNIVERSITIES = [
    {"name": "Buckinghamshire New University", "key": "bucks_new_university", "base": "https://www.bucks.ac.uk", "patterns": [r"kenya", r"international.*countr|countr.*international", r"entry.requirement"]},
    {"name": "Birmingham City University", "key": "birmingham_city_university", "base": "https://www.bcu.ac.uk", "patterns": [r"kenya", r"international.*countr|countr.*international", r"entry.requirement"]},
    {"name": "University of Salford", "key": "salford_university", "base": "https://www.salford.ac.uk", "patterns": [r"kenya", r"international.*countr|countr.*international", r"entry.requirement"]},
    {"name": "Robert Gordon University", "key": "robert_gordon_university", "base": "https://www.rgu.ac.uk", "patterns": [r"kenya", r"international.*countr|countr.*international", r"entry.requirement"]},
    {"name": "University of South Wales", "key": "south_wales_university", "base": "https://www.southwales.ac.uk", "patterns": [r"kenya", r"international.*countr|countr.*international", r"entry.requirement"]},
    {"name": "University of Central Lancashire", "key": "central_lancashire_university", "base": "https://www.uclan.ac.uk", "patterns": [r"kenya", r"international.*countr|countr.*international", r"entry.requirement"]},
    {"name": "University of Hertfordshire", "key": "hertfordshire_university", "base": "https://www.herts.ac.uk", "patterns": [r"kenya", r"international.*countr|countr.*international", r"entry.requirement"]},
]

def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    OUT.write(msg + "\n")
    OUT.flush()

def get_links(page, url, pattern, wait_ms=6000):
    try:
        resp = page.goto(url, timeout=45000, wait_until="domcontentloaded")
        status = resp.status if resp else "none"
        log(f"[nav] {url} status={status}")
    except Exception as exc:
        log(f"[nav] {url} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
        page.wait_for_timeout(3000)
        return
    page.wait_for_timeout(wait_ms)
    try:
        links = page.eval_on_selector_all(
            "a[href]",
            f"els => {{ const re = new RegExp({pattern!r}, 'i'); return els.filter(a => re.test(a.href + ' ' + (a.innerText||''))).map(a => [a.innerText.trim().slice(0,60), a.href]); }}",
        )
        seen = set()
        for t, h in links:
            if h not in seen:
                seen.add(h)
                log(f"  {t or '(no text)':<60} => {h}")
    except Exception as exc:
        log("  Link extraction failed:", exc)

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0")
    pg.set_default_timeout(45000)

    for uni in UNIVERSITIES:
        log("\n" + "=" * 80)
        log(f"UNIVERSITY: {uni['name']} ({uni['base']})")
        log("=" * 80)
        for pat in uni["patterns"]:
            get_links(pg, uni["base"] + "/international", pat)
            get_links(pg, uni["base"] + "/international/countries", pat)
            get_links(pg, uni["base"], pat)

    b.close()

OUT.close()
print("Done! Check universities_explore_output.txt")
