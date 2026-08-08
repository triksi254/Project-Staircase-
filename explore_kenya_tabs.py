"""Explore the tabs on Aston Kenya page - especially Postgraduate requirements."""
import re
from playwright.sync_api import sync_playwright

BASE = "https://www.aston.ac.uk"
URL = f"{BASE}/international/aston-in-your-country/africa/kenya"
OUT = open("kenya_tabs_output.txt", "w", encoding="utf-8")

def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    OUT.write(msg + "\n")
    OUT.flush()

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    pg.set_default_timeout(60000)

    try:
        resp = pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        log("Initial nav status:", resp.status if resp else "none")
        pg.wait_for_timeout(5000)

        # Find all tabs/buttons on the page
        log("\n=== TABS / BUTTONS ===")
        try:
            tabs = pg.eval_on_selector_all(
                "button, a, [role='tab'], [role='button']",
                "els => els.map(e => { const t = (e.innerText||'').trim().replace(/\\s+/g,' '); return { tag: e.tagName, text: t.slice(0,80), id: e.id, cls: e.className.toString().slice(0,60), href: e.getAttribute('href') || '' }; }).filter(x => x.text && x.text.length < 80)"
            )
            for t in tabs:
                log(f"[{t['tag']}] id={t['id']!r} cls={t['cls']!r} href={t['href']!r} :: {t['text']}")
        except Exception as exc:
            log("Tab extraction failed:", exc)

        # Try clicking elements with text 'Postgraduate'
        log("\n=== CLICKING POSTGRADUATE ===")
        try:
            clicked = pg.evaluate("""
                () => {
                    const els = [...document.querySelectorAll('button, a, [role=tab], [role=button], li, div, span')];
                    const targets = els.filter(e => (e.innerText||'').trim() === 'Postgraduate' && e.offsetParent !== null);
                    if (targets.length) {
                        targets[0].click();
                        return targets.length + ' elements found, clicked first: ' + targets[0].tagName + '.' + targets[0].className;
                    }
                    return 'No exact Postgraduate clickable element found';
                }
            """)
            log("Click result:", clicked)
        except Exception as exc:
            log("Click failed:", exc)

        pg.wait_for_timeout(4000)

        # Dump page text after click
        text = pg.inner_text("body")
        log("\n=== PAGE TEXT AFTER CLICK (len={}) ===".format(len(text)))

        # Look for key requirement terms
        keywords = ["kenya", "postgraduate", "entry requirement", "bachelor", "degree",
                     "upper second", "2:1", "second class", "upper second class",
                     "msc", "master", "degree classification", "accredited", "recognition",
                     "applicants must", "normally require", "honours", "cgpa", "gpa"]
        lines = text.split("\n")
        for kw in keywords:
            hits = []
            for i, line in enumerate(lines):
                if kw.lower() in line.lower():
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    ctx = " | ".join(l.strip() for l in lines[start:end] if l.strip())
                    hits.append(ctx)
            if hits:
                log(f"\n--- {kw.upper()} ({len(hits)} hits) ---")
                for h in hits[:5]:
                    log(h[:300])

        # Also check the tab structure with JS
        log("\n=== TAB DOM ANALYSIS ===")
        try:
            tab_info = pg.evaluate("""
                () => {
                    const results = [];
                    const all = document.querySelectorAll('*');
                    for (const el of all) {
                        const t = (el.innerText||'').trim();
                        if (t && t.length < 60 && /postgraduate|undergraduate|country information|english language/i.test(t)) {
                            results.push({tag: el.tagName, text: t, role: el.getAttribute('role'), cls: el.className.toString().slice(0,80)});
                        }
                    }
                    return results.slice(0, 30);
                }
            """)
            for ti in tab_info:
                log(f"[{ti['tag']}] role={ti['role']!r} cls={ti['cls']!r} :: {ti['text']}")
        except Exception as exc:
            log("Tab DOM analysis failed:", exc)

    finally:
        b.close()

OUT.close()
print("Done! Check kenya_tabs_output.txt")
