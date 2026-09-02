"""Scrape Kenya Postgraduate MSc entry requirements from all 7 universities."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# Ensure project root on sys.path when running from elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processors.requirements_extractor import extract_requirements  # noqa: E402
from scrapers.page_interaction import click_accordions, launch_browser  # noqa: E402

# Suppress non-critical Playwright logs
import logging
logging.getLogger("playwright").setLevel(logging.WARNING)

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
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


# NOTE: extract_requirements now lives in processors.requirements_extractor


# NOTE: _click_accordions moved to scrapers.page_interaction (click_accordions)


def scrape_university(uni: Dict) -> List[Dict]:
    """Scrape a university's Kenya entry requirements."""
    from playwright.sync_api import sync_playwright

    records = []
    url = uni["url"]
    all_urls = [url] + uni.get("alt_urls", [])

    with sync_playwright() as p:
        b, ctx, page = launch_browser(p)

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
                    click_accordions(page)
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
