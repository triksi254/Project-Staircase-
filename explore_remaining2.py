"""Explore remaining universities - country-specific pages."""
from playwright.sync_api import sync_playwright

OUT = open("remaining2_output.txt", "w", encoding="utf-8")

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

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36")
    pg.set_default_timeout(45000)

    # 1. BUCKS - Your Country page
    log("\n" + "=" * 80)
    log("BUCKS - Your Country page")
    log("=" * 80)
    s = goto_robust(pg, "https://www.bucks.ac.uk/study/international/your-country")
    if s and s < 400:
        pg.wait_for_timeout(5000)
        # Look for Kenya
        links = pg.eval_on_selector_all("a[href]", "els => els.filter(a => a.innerText.toLowerCase().includes('kenya')).map(a => [a.innerText.trim(), a.href])")
        for t, h in links:
            log(f"  Kenya link: {t} => {h}")
        body = pg.inner_text("body")
        log(f"  Body length: {len(body)}")
        for kw in ["postgraduate", "kenya", "masters", "entry", "requirement"]:
            idx = body.lower().find(kw.lower())
            if idx >= 0:
                s = max(0, idx-150)
                e = min(len(body), idx+300)
                log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")

    # 2. BUCKS - General Entry Requirements
    log("\n" + "=" * 80)
    log("BUCKS - General Entry Requirements")
    log("=" * 80)
    s = goto_robust(pg, "https://www.bucks.ac.uk/study/general-entry-requirements")
    if s and s < 400:
        pg.wait_for_timeout(5000)
        body = pg.inner_text("body")
        log(f"  Body length: {len(body)}")
        for kw in ["postgraduate", "masters", "bachelor", "entry", "kcse", "kenya", "english"]:
            idx = body.lower().find(kw.lower())
            if idx >= 0:
                s = max(0, idx-150)
                e = min(len(body), idx+300)
                log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")

    # 3. SOUTH WALES - Your Country page
    log("\n" + "=" * 80)
    log("SOUTH WALES - Your Country page")
    log("=" * 80)
    s = goto_robust(pg, "https://www.southwales.ac.uk/international/your-country/")
    if s and s < 400:
        pg.wait_for_timeout(5000)
        links = pg.eval_on_selector_all("a[href]", "els => els.filter(a => a.innerText.toLowerCase().includes('kenya')).map(a => [a.innerText.trim(), a.href])")
        for t, h in links:
            log(f"  Kenya link: {t} => {h}")
        body = pg.inner_text("body")
        log(f"  Body length: {len(body)}")
        for kw in ["postgraduate", "masters", "kenya", "entry", "requirement", "bachelor"]:
            idx = body.lower().find(kw.lower())
            if idx >= 0:
                s = max(0, idx-150)
                e = min(len(body), idx+300)
                log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")

    # 4. UCLAN - search for Kenya on more pages
    log("\n" + "=" * 80)
    log("UCLAN - looking for Kenya on pages 4-9")
    log("=" * 80)
    for pg_num in range(4, 10):
        url = f"https://www.lancashire.ac.uk/international-students/country?page={pg_num}"
        s = goto_robust(pg, url)
        if s and s < 400:
            pg.wait_for_timeout(3000)
            body = pg.inner_text("body")
            if "kenya" in body.lower():
                log(f"  Found Kenya on page {pg_num}!")
                links = pg.eval_on_selector_all("a[href]", "els => els.filter(a => a.innerText.toLowerCase().includes('kenya')).map(a => [a.innerText.trim(), a.href])")
                for t, h in links:
                    log(f"  Kenya link: {t} => {h}")
                # Extract relevant text
                idx = body.lower().find("kenya")
                if idx >= 0:
                    s = max(0, idx-200)
                    e = min(len(body), idx+500)
                    log(f"  ...[{body[s:e].replace(chr(10), ' ')}]...")
                break

    # 5. HERTFORDSHIRE - try to find country-specific page
    log("\n" + "=" * 80)
    log("HERTFORDSHIRE - finding country-specific page")
    log("=" * 80)
    s = goto_robust(pg, "https://www.herts.ac.uk/international/apply")
    if s and s < 400:
        pg.wait_for_timeout(5000)
        body = pg.inner_text("body")
        log(f"  Body length: {len(body)}")
        for kw in ["country", "kenya", "postgraduate", "entry", "requirement", "international"]:
            idx = body.lower().find(kw.lower())
            if idx >= 0:
                s = max(0, idx-150)
                e = min(len(body), idx+300)
                log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")

    # 6. SALFORD - try the Kenya page with a different approach
    log("\n" + "=" * 80)
    log("SALFORD - Kenya page (retry)")
    log("=" * 80)
    s = goto_robust(pg, "https://www.salford.ac.uk/international/your-country-or-region/kenya")
    if s and s < 400:
        pg.wait_for_timeout(8000)
        body = pg.inner_text("body")
        log(f"  Body length: {len(body)}")
        for kw in ["postgraduate", "masters", "entry", "requirement", "bachelor", "kcse", "english"]:
            idx = body.lower().find(kw.lower())
            if idx >= 0:
                s = max(0, idx-200)
                e = min(len(body), idx+500)
                log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")

    b.close()
OUT.close()
print("Done! Check remaining2_output.txt")
