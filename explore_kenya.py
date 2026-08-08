"""Explore Aston Kenya entry requirements page and MSc course pages."""
import re
from playwright.sync_api import sync_playwright

BASE = "https://www.aston.ac.uk"
OUT = open("kenya_explore_output.txt", "w", encoding="utf-8")

def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    OUT.write(msg + "\n")
    OUT.flush()

def goto_robust(page, url, timeout=60000):
    for s in ["domcontentloaded", "commit", "load"]:
        try:
            resp = page.goto(url, timeout=timeout, wait_until=s)
            status = resp.status if resp else "none"
            log(f"[nav {s}] status={status}")
            return True
        except Exception as exc:
            log(f"[nav {s}] failed: {type(exc).__name__}: {str(exc)[:100]}")
            page.wait_for_timeout(3000)
    return False

def dump_full_text(page, url, wait_ms=8000):
    ok = goto_robust(page, url)
    if not ok:
        log(f"Could not navigate to {url}")
        return
    page.wait_for_timeout(wait_ms)
    log(f"\n=== FULL TEXT: {url} ===")
    try:
        log("TITLE:", page.title())
        text = page.inner_text("body")
        log(f"PAGE TEXT LENGTH: {len(text)}")
        # Write the full text
        log("----- BEGIN PAGE TEXT -----")
        OUT.write(text + "\n")
        OUT.flush()
        log("----- END PAGE TEXT -----")
    except Exception as exc:
        log("Text dump failed:", exc)

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    pg.set_default_timeout(60000)

    # Kenya entry requirements page
    dump_full_text(pg, f"{BASE}/international/aston-in-your-country/africa/kenya")
    
    # Also find some MSc courses to check entry requirements
    # The courses page uses a filter URL for postgraduate
    dump_full_text(pg, f"{BASE}/courses?title=&field_course_level_target_id%5B2%5D=2&field_course_level_target_id%5B5%5D=5&clearing=All&course_title_autocomplete=0")
    
    b.close()

OUT.close()
print("Done! Check kenya_explore_output.txt")
