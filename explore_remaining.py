"""Explore remaining universities to find Kenya-specific entry requirements."""
from playwright.sync_api import sync_playwright

OUT = open("remaining_explore_output.txt", "w", encoding="utf-8")

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
        return status, url
    except Exception as exc:
        log(f"[nav] {url} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
        page.wait_for_timeout(3000)
        return None, url

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36")
    pg.set_default_timeout(45000)

    # 1. Salford - Kenya page deep dive
    log("\n" + "=" * 80)
    log("SALFORD - Kenya page deep dive")
    log("=" * 80)
    status, url = goto_robust(pg, "https://www.salford.ac.uk/international/your-country-or-region/kenya")
    if status and status < 400:
        pg.wait_for_timeout(8000)
        # Try clicking any accordion buttons
        try:
            btns = pg.query_selector_all("button, .accordion-trigger, [data-toggle], .toggle")
            log(f"Found {len(btns)} interactive elements")
            for btn in btns:
                try:
                    text = btn.inner_text()
                    log(f"  Button: '{text.strip()[:50]}'")
                    btn.click()
                    pg.wait_for_timeout(1000)
                except:
                    pass
        except Exception as exc:
            log(f"  Button interaction: {exc}")
        pg.wait_for_timeout(2000)
        body = pg.inner_text("body")
        log(f"\n  Full body text ({len(body)} chars):")
        for kw in ["postgraduate", "post-graduate", "masters", "msc", "bachelor", "honours", "second class", "2:1", "2:2", "kcse", "entry", "requirement", "degree", "english"]:
            idx = body.lower().find(kw.lower())
            if idx >= 0:
                s = max(0, idx-200)
                e = min(len(body), idx+500)
                log(f"\n  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")

    # 2. BCU - Kenya page (try with longer timeout and scroll)
    log("\n" + "=" * 80)
    log("BCU - Kenya page")
    log("=" * 80)
    status, url = goto_robust(pg, "https://www.bcu.ac.uk/international/bcu-in-your-country/kenya")
    if status and status < 400:
        pg.wait_for_timeout(5000)
        body = pg.inner_text("body")
        log(f"  Body length: {len(body)}")
        if len(body) > 100:
            for kw in ["postgraduate", "masters", "bachelor", "entry", "degree", "kcse", "english"]:
                idx = body.lower().find(kw.lower())
                if idx >= 0:
                    s = max(0, idx-200)
                    e = min(len(body), idx+500)
                    log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")

    # 3. UCLan - Kenya page
    log("\n" + "=" * 80)
    log("UCLAN - Finding Kenya page")
    log("=" * 80)
    status, url = goto_robust(pg, "https://www.lancashire.ac.uk/international-students/country")
    pg.wait_for_timeout(3000)
    # Try clicking on page 2 etc
    for pg_num in [1, 2, 3]:
        url = f"https://www.lancashire.ac.uk/international-students/country?page={pg_num}"
        status, _ = goto_robust(pg, url)
        if status and status < 400:
            pg.wait_for_timeout(3000)
            body = pg.inner_text("body")
            if "kenya" in body.lower() or "Kenya" in body:
                log(f"  Found Kenya on page {pg_num}!")
                # Find the Kenya link
                links = pg.eval_on_selector_all("a[href]", f"els => els.filter(a => a.innerText.toLowerCase().includes('kenya')).map(a => a.href)")
                for l in links:
                    log(f"  Kenya link: {l}")
                log(f"  Body: {body[:2000]}")
                break

    # 4. Hertfordshire - Kenya page
    log("\n" + "=" * 80)
    log("HERTFORDSHIRE - Finding Kenya page")
    log("=" * 80)
    status, url = goto_robust(pg, "https://www.herts.ac.uk/international")
    if status and status < 400:
        pg.wait_for_timeout(3000)
        links = pg.eval_on_selector_all("a[href]", "els => els.filter(a => /kenya|country|entry.requirement/i.test(a.href + ' ' + a.innerText)).map(a => [a.innerText.trim().slice(0,50), a.href])")
        for t, h in links:
            log(f"  {t:<55} {h}")
        body = pg.inner_text("body")
        # Look for "by country" or "international entry requirements" links
        for kw in ["by country", "country", "entry requirement", "requirement"]:
            idx = body.lower().find(kw.lower())
            if idx >= 0:
                s = max(0, idx-100)
                e = min(len(body), idx+300)
                log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")

    # Also try the apply page
    status, url = goto_robust(pg, "https://www.herts.ac.uk/international/apply")
    if status and status < 400:
        pg.wait_for_timeout(3000)
        links = pg.eval_on_selector_all("a[href]", "els => els.filter(a => /kenya|country|entry.requirement/i.test(a.href + ' ' + a.innerText)).map(a => [a.innerText.trim().slice(0,50), a.href])")
        for t, h in links:
            log(f"  {t:<55} {h}")

    # 5. Bucks - try different approach
    log("\n" + "=" * 80)
    log("BUCKS - trying different URLs")
    log("=" * 80)
    for url in ["https://www.bucks.ac.uk/", "https://www.bucks.ac.uk/study", "https://www.bucks.ac.uk/about/international"]:
        status, _ = goto_robust(pg, url)
        if status and status < 400:
            pg.wait_for_timeout(3000)
            links = pg.eval_on_selector_all("a[href]", "els => els.filter(a => /kenya|country|international|entry.requirement/i.test(a.href + ' ' + a.innerText)).map(a => [a.innerText.trim().slice(0,50), a.href])")
            if links:
                for t, h in links:
                    log(f"  {t:<55} {h}")

    # 6. South Wales - try different approach
    log("\n" + "=" * 80)
    log("SOUTH WALES - trying different URLs")
    log("=" * 80)
    for url in ["https://www.southwales.ac.uk/international/", "https://www.southwales.ac.uk/", "https://www.southwales.ac.uk/study/"]:
        status, _ = goto_robust(pg, url)
        if status and status < 400:
            pg.wait_for_timeout(3000)
            links = pg.eval_on_selector_all("a[href]", "els => els.filter(a => /kenya|country|international|entry.requirement/i.test(a.href + ' ' + a.innerText)).map(a => [a.innerText.trim().slice(0,50), a.href])")
            if links:
                for t, h in links:
                    log(f"  {t:<55} {h}")

    b.close()
OUT.close()
print("Done! Check remaining_explore_output.txt")
