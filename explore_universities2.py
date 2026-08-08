"""Deep-dive exploration of each university's Kenya entry requirements page."""
import re
from playwright.sync_api import sync_playwright

OUT = open("universities_explore2_output.txt", "w", encoding="utf-8")

def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    OUT.write(msg + "\n")
    OUT.flush()

def goto_robust(page, url, timeout=45000):
    try:
        resp = page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        status = resp.status if resp else "none"
        log(f"[nav] {url} status={status}")
        return status
    except Exception as exc:
        log(f"[nav] {url} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
        page.wait_for_timeout(3000)
        return None

def get_links(page, url, pattern, wait_ms=5000):
    status = goto_robust(page, url)
    if status is None or status >= 400:
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

def dump_keywords(page, url, wait_ms=6000):
    status = goto_robust(page, url)
    if status is None or status >= 400:
        return
    page.wait_for_timeout(wait_ms)
    try:
        text = page.inner_text("body")
        log(f"\n  --- TEXT from {url} (len={len(text)}) ---")
        keywords = ["kenya", "postgraduate", "postgraduate taught", "masters", "msc",
                     "entry requirement", "bachelor", "honours", "second class", "upper second",
                     "2:1", "2.1", "2:2", "2.2", "degree", "english", "ielts"]
        for kw in keywords:
            idx = text.lower().find(kw.lower())
            if idx >= 0:
                s = max(0, idx - 150)
                e = min(len(text), idx + 300)
                log(f"    [{kw}] ...{text[s:e].replace(chr(10), ' ')}...")
    except Exception as exc:
        log("  Text extraction failed:", exc)

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36")
    pg.set_default_timeout(45000)

    # 1. BCU Kenya page
    log("\n" + "=" * 80)
    log("BIRMINGHAM CITY UNIVERSITY - Kenya page")
    log("=" * 80)
    dump_keywords(pg, "https://www.bcu.ac.uk/international/bcu-in-your-country/kenya")

    # 2. Salford - country page
    log("\n" + "=" * 80)
    log("UNIVERSITY OF SALFORD - country pages")
    log("=" * 80)
    get_links(pg, "https://www.salford.ac.uk/international/your-country-or-region", r"kenya|africa|country|region")
    dump_keywords(pg, "https://www.salford.ac.uk/international/your-country-or-region")

    # 3. RGU - country page
    log("\n" + "=" * 80)
    log("ROBERT GORDON UNIVERSITY - country pages")
    log("=" * 80)
    get_links(pg, "https://www.rgu.ac.uk/study/international-students/your-country-or-territory", r"kenya|africa|country|territory")
    dump_keywords(pg, "https://www.rgu.ac.uk/study/international-students/your-country-or-territory")

    # 4. UCLan - country page (redirects to lancashire.ac.uk)
    log("\n" + "=" * 80)
    log("UNIVERSITY OF CENTRAL LANCASHIRE - country page")
    log("=" * 80)
    get_links(pg, "https://www.lancashire.ac.uk/international-students/country", r"kenya|africa|country|region")
    dump_keywords(pg, "https://www.lancashire.ac.uk/international-students/country")

    # 5. Hertfordshire - country pages
    log("\n" + "=" * 80)
    log("UNIVERSITY OF HERTFORDSHIRE - country pages")
    log("=" * 80)
    get_links(pg, "https://www.herts.ac.uk/international", r"kenya|country|countr|entry|requirement")
    get_links(pg, "https://www.herts.ac.uk/international/entry-requirements", r"kenya|country|countr|entry|requirement")
    dump_keywords(pg, "https://www.herts.ac.uk/international/entry-requirements")

    # 6. South Wales - try with different approach (maybe https works or different URL)
    log("\n" + "=" * 80)
    log("UNIVERSITY OF SOUTH WALES - country pages")
    log("=" * 80)
    get_links(pg, "https://www.southwales.ac.uk/international/country-information", r"kenya|country|africa")
    dump_keywords(pg, "https://www.southwales.ac.uk/international/country-information")

    # 7. Bucks - try different URLs
    log("\n" + "=" * 80)
    log("BUCKINGHAMSHIRE NEW UNIVERSITY - country pages")
    log("=" * 80)
    get_links(pg, "https://bucks.ac.uk/international", r"kenya|country|countr")
    get_links(pg, "https://www.bucks.ac.uk/international/your-country", r"kenya|country|countr")

    b.close()

OUT.close()
print("Done! Check universities_explore2_output.txt")
