"""Deep-dive into failing universities to find exact Kenya PG requirement data."""
from playwright.sync_api import sync_playwright

OUT = open("failing_explore_output.txt", "w", encoding="utf-8")

def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    OUT.write(msg + "\n")
    OUT.flush()

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0", viewport={"width": 1366, "height": 900})

    # 1. SALFORD - explore the Kenya page structure
    log("=" * 80)
    log("SALFORD - Kenya page structure")
    log("=" * 80)
    pg.goto("https://www.salford.ac.uk/international/your-country-or-region/kenya", wait_until="domcontentloaded")
    pg.wait_for_timeout(5000)
    # Get all links
    links = pg.eval_on_selector_all("a[href]", "els => els.map(a => [a.innerText.trim().slice(0,60), a.href, a.className || ''])")
    log("All links on page:")
    for t, h, c in links:
        if any(kw in (t + h).lower() for kw in ["entry", "requirement", "postgraduate", "masters", "kenya", "admission"]):
            log(f"  {t:<60} => {h}")
    # Get all headings
    headings = pg.eval_on_selector_all("h1,h2,h3,h4,h5,h6", "els => els.map(h => h.innerText.trim().slice(0,80))")
    log("Headings:")
    for h in headings:
        if any(kw in h.lower() for kw in ["entry", "requirement", "postgraduate", "kenya", "admission"]):
            log(f"  {h}")

    # Try clicking "Entry requirements" link
    log("\nTrying to click Entry requirements link...")
    for t in ["Entry requirements", "entry requirements"]:
        link = pg.get_by_text(t, exact=False).first
        try:
            link.click(timeout=5000)
            pg.wait_for_timeout(3000)
            log(f"  Clicked '{t}', new URL: {pg.url}")
            body = pg.inner_text("body")
            log(f"  Body length: {len(body)}")
            for kw in ["postgraduate", "masters", "bachelor", "second class", "degree", "kcse", "ielts"]:
                idx = body.lower().find(kw.lower())
                if idx >= 0:
                    s = max(0, idx-100)
                    e = min(len(body), idx+300)
                    log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")
        except Exception as ex:
            log(f"  Could not click '{t}': {ex}")

    # 2. SOUTH WALES - Kenya page
    log("\n" + "=" * 80)
    log("SOUTH WALES - Kenya page")
    log("=" * 80)
    pg.goto("https://www.southwales.ac.uk/international/your-country/kenya/", wait_until="domcontentloaded")
    pg.wait_for_timeout(5000)
    body = pg.inner_text("body")
    log(f"Body length: {len(body)}")
    # Get all text
    for kw in ["postgraduate", "masters", "bachelor", "second class", "kcse", "ielts", "entry", "degree"]:
        idx = body.lower().find(kw.lower())
        if idx >= 0:
            s = max(0, idx-100)
            e = min(len(body), idx+300)
            log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")
    # Get links
    links = pg.eval_on_selector_all("a[href]", "els => els.map(a => [a.innerText.trim().slice(0,50), a.href])")
    for t, h in links:
        log(f"  {t:<55} => {h}")

    # 3. BUCKS - Your Country page
    log("\n" + "=" * 80)
    log("BUCKS - Your Country page")
    log("=" * 80)
    pg.goto("https://www.bucks.ac.uk/study/international/your-country", wait_until="domcontentloaded")
    pg.wait_for_timeout(8000)
    body = pg.inner_text("body")
    log(f"Body length: {len(body)}")
    # Find Kenya link
    links = pg.eval_on_selector_all("a[href]", "els => els.filter(a => a.innerText.toLowerCase().includes('kenya')).map(a => [a.innerText.trim().slice(0,50), a.href])")
    for t, h in links:
        log(f"  Kenya link: {t} => {h}")
    for kw in ["postgraduate", "masters", "entry", "requirement", "degree", "kcse", "english"]:
        idx = body.lower().find(kw.lower())
        if idx >= 0:
            s = max(0, idx-100)
            e = min(len(body), idx+300)
            log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")

    # 4. HERTFORDSHIRE - entry requirements page
    log("\n" + "=" * 80)
    log("HERTFORDSHIRE - entry requirements")
    log("=" * 80)
    pg.goto("https://www.herts.ac.uk/international/apply/application-requirements", wait_until="domcontentloaded")
    pg.wait_for_timeout(8000)
    # Try clicking accordions
    for sel in ["button[aria-expanded]", "button[data-bs-toggle='collapse']", ".accordion-item button", "button.collapsed"]:
        btns = pg.query_selector_all(sel)
        for btn in btns:
            try:
                expanded = btn.get_attribute("aria-expanded")
                if expanded == "false" or expanded is None:
                    btn.click()
                    pg.wait_for_timeout(500)
            except:
                pass
    pg.wait_for_timeout(2000)
    body = pg.inner_text("body")
    log(f"Body length: {len(body)}")
    for kw in ["postgraduate", "masters", "msc", "bachelor", "kcse", "kenya", "ielts", "english", "entry", "degree", "second class"]:
        idx = body.lower().find(kw.lower())
        if idx >= 0:
            s = max(0, idx-100)
            e = min(len(body), idx+300)
            log(f"  [{kw}] ...{body[s:e].replace(chr(10), ' ')}...")

    b.close()
OUT.close()
print("Done! Check failing_explore_output.txt")
