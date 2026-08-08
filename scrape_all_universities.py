"""Scrape Kenya Postgraduate MSc entry requirements from all 7 universities."""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# Suppress non-critical Playwright logs
import logging
logging.getLogger("playwright").setLevel(logging.WARNING)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSITIES = [
    {
        "name": "Buckinghamshire New University",
        "key": "bucks_new_university",
        "url": "https://www.bucks.ac.uk/study/general-entry-requirements",
        "alt_urls": ["https://www.bucks.ac.uk/study/international/your-country"],
        "has_accordion": False,
    },
    {
        "name": "Birmingham City University",
        "key": "birmingham_city_university",
        "url": "https://www.bcu.ac.uk/international/bcu-in-your-country/kenya",
        "alt_urls": ["https://www.bcu.ac.uk/international/your-application/entry-requirements"],
        "has_accordion": True,
    },
    {
        "name": "University of Salford",
        "key": "salford_university",
        "url": "https://www.salford.ac.uk/international/your-country-or-region/kenya",
        "alt_urls": ["https://www.salford.ac.uk/study/postgraduate"],
        "has_accordion": True,
    },
    {
        "name": "Robert Gordon University",
        "key": "robert_gordon_university",
        "url": "https://www.rgu.ac.uk/study/international-students/your-country-or-territory/kenya",
        "alt_urls": ["https://www.rgu.ac.uk/study/international-students/english-language-requirements"],
        "has_accordion": False,
    },
    {
        "name": "University of South Wales",
        "key": "south_wales_university",
        "url": "https://www.southwales.ac.uk/international/your-country/kenya/",
        "alt_urls": ["https://www.southwales.ac.uk/international/your-country/"],
        "has_accordion": False,
    },
    {
        "name": "University of Central Lancashire",
        "key": "central_lancashire_university",
        "url": "https://www.lancashire.ac.uk/international-students/country/kenya",
        "alt_urls": ["https://www.lancashire.ac.uk/international-students/english-requirements"],
        "has_accordion": True,
    },
    {
        "name": "University of Hertfordshire",
        "key": "hertfordshire_university",
        "url": "https://www.herts.ac.uk/international/apply",
        "alt_urls": ["https://www.herts.ac.uk/international/apply/application-requirements"],
        "has_accordion": True,
    },
]


def extract_requirements(text: str) -> Dict[str, Optional[str]]:
    """Extract key requirement fields from body text."""
    result = {
        "postgraduate_entry_requirement": None,
        "english_language_requirement": None,
        "bachelor_degree_requirement": None,
        "foundation_requirement": None,
        "scholarship_info": None,
        "contact_info": None,
    }

    lines = text.split("\n")
    full_text = " ".join(line.strip() for line in lines if line.strip())

    # Postgraduate entry requirements
    pg_patterns = [
        r"(?i)(?:postgraduate|post-graduate|masters|master\'s|msc|ma\b|mba|meng).{0,300}(?:bachelor|degree|honours|second class|upper|lower|2:?[12]|2\.\s*[12])",
        r"(?i)(?:bachelor|degree|honours).{0,200}(?:second class|upper|lower|2:?[12]|2\.\s*[12]).{0,200}(?:postgraduate|post-graduate|masters)",
    ]
    for pat in pg_patterns:
        m = re.search(pat, full_text)
        if m:
            result["postgraduate_entry_requirement"] = m.group(0)[:1000].strip()
            break

    # Bachelor degree requirement
    bach_pat = r"(?i)(?:bachelor|degree|honours).{0,300}(?:second class|upper|lower|2:?[12]|2\.\s*[12]|first class|gpa|grade|recognis)"
    m = re.search(bach_pat, full_text)
    if m:
        result["bachelor_degree_requirement"] = m.group(0)[:800].strip()

    # English language requirement
    eng_pat = r"(?i)(?:english|ielts|toefl|language).{0,200}(?:requirement|grade|score|level|band|min).{0,300}"
    m = re.search(eng_pat, full_text)
    if m:
        result["english_language_requirement"] = m.group(0)[:800].strip()

    # Foundation requirement
    found_pat = r"(?i)(?:foundation|foundation year|international foundation).{0,300}(?:year|study|kcse|grade|subject)"
    m = re.search(found_pat, full_text)
    if m:
        result["foundation_requirement"] = m.group(0)[:600].strip()

    # Scholarship info
    schol_pat = r"(?i)(?:scholarship|bursary|funding|award).{0,200}(?:international|available|offer|amount)"
    m = re.search(schol_pat, full_text)
    if m:
        result["scholarship_info"] = m.group(0)[:500].strip()

    # Contact info
    contact_pat = r"(?i)(?:contact|email|phone|regional manager|representative).{0,200}(?:@|\.com|\.ac|whatsapp|\+[0-9])"
    m = re.search(contact_pat, full_text)
    if m:
        result["contact_info"] = m.group(0)[:400].strip()

    return result


def _click_accordions(page) -> None:
    """Click all accordion/expandable buttons to reveal hidden content."""
    try:
        # Try various selectors for accordion triggers
        accordion_selectors = [
            "button[aria-expanded]",
            ".accordion-trigger",
            ".accordion-header button",
            "[data-toggle='collapse']",
            ".toggle",
            ".collapse-toggle",
            "button.collapsed",
            "button[data-bs-toggle='collapse']",
            "details summary",
            ".accordion-item button",
            "button[aria-controls]",
            "button:has(span)",
        ]
        clicked = set()
        for sel in accordion_selectors:
            buttons = page.query_selector_all(sel)
            for btn in buttons:
                try:
                    outer = btn.inner_html()
                    if outer in clicked:
                        continue
                    clicked.add(outer)
                    # Check if it's collapsed/not expanded
                    expanded = btn.get_attribute("aria-expanded")
                    if expanded == "false" or expanded is None:
                        btn.click()
                        page.wait_for_timeout(500)
                except Exception:
                    pass
        logger.info("  Clicked {} accordion elements", len(clicked))
    except Exception as exc:
        logger.debug("  Accordion click error: {}", exc)


def scrape_university(uni: Dict) -> List[Dict]:
    """Scrape a university's Kenya entry requirements."""
    from playwright.sync_api import sync_playwright

    records = []
    url = uni["url"]
    all_urls = [url] + uni.get("alt_urls", [])

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
            viewport={"width": 1366, "height": 900},
        )
        page = ctx.new_page()

        for target_url in all_urls:
            try:
                resp = page.goto(target_url, timeout=45000, wait_until="domcontentloaded")
                status = resp.status if resp else "none"
                logger.info("{} - {}: status={}", uni["name"], target_url, status)

                if status and status >= 400:
                    continue

                page.wait_for_timeout(5000)

                # Click accordion buttons to reveal hidden content if applicable
                if uni.get("has_accordion", False):
                    _click_accordions(page)
                    page.wait_for_timeout(2000)

                # Try to get full page text
                body_text = page.inner_text("body")
                text_len = len(body_text)
                logger.info("  Text length: {}", text_len)

                if text_len < 100:
                    continue

                # Extract requirements
                reqs = extract_requirements(body_text)

                record = {
                    "institution": uni["name"],
                    "institution_key": uni["key"],
                    "source_url": target_url,
                    "text_length": text_len,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "page_text": body_text[:5000],  # Store first 5000 chars
                    **reqs,
                }
                records.append(record)
                break  # Successfully scraped, no need for alt URLs

            except Exception as exc:
                logger.warning("  Failed: {} - {}", type(exc).__name__, str(exc)[:100])
                continue

        ctx.close()
        b.close()

    return records


def main():
    all_records: List[Dict] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for uni in UNIVERSITIES:
        logger.info("=" * 60)
        logger.info("Scraping: {}", uni["name"])
        logger.info("=" * 60)
        try:
            records = scrape_university(uni)
            if not records:
                logger.warning("  No data scraped for {}", uni["name"])
                all_records.append({
                    "institution": uni["name"],
                    "institution_key": uni["key"],
                    "source_url": uni["url"],
                    "text_length": 0,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "page_text": None,
                    "postgraduate_entry_requirement": None,
                    "english_language_requirement": None,
                    "bachelor_degree_requirement": None,
                    "foundation_requirement": None,
                    "scholarship_info": None,
                    "contact_info": None,
                    "error": "Could not access page",
                })
            else:
                all_records.extend(records)
                r = records[0]
                if r.get("postgraduate_entry_requirement"):
                    logger.info("  ✓ Found PG entry requirement")
                if r.get("english_language_requirement"):
                    logger.info("  ✓ Found English requirement")
                if r.get("bachelor_degree_requirement"):
                    logger.info("  ✓ Found bachelor requirement")
        except Exception as exc:
            logger.exception("Fatal error scraping {}: {}", uni["name"], exc)

    # Save CSV
    csv_path = PROCESSED_DIR / f"all_universities_kenya_requirements_{timestamp}.csv"
    fieldnames = [
        "institution", "institution_key", "source_url", "text_length", "scraped_at",
        "postgraduate_entry_requirement", "english_language_requirement",
        "bachelor_degree_requirement", "foundation_requirement",
        "scholarship_info", "contact_info", "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    logger.info("\n" + "=" * 60)
    logger.info("CSV saved: {}", csv_path)
    logger.info("Total records: {}", len(all_records))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Kenya Postgraduate MSc Entry Requirements")
    print("=" * 70)
    for r in all_records:
        name = r["institution"]
        pg = r.get("postgraduate_entry_requirement")
        eng = r.get("english_language_requirement")
        err = r.get("error", "")
        status = "✓" if pg else (f"✗ ({err})" if err else "✗ (no data)")
        print(f"  {name:<45} {status}")

    # Save JSON
    json_path = PROCESSED_DIR / f"all_universities_kenya_requirements_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, default=str)

    print(f"\nFiles saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
