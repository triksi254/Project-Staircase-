"""Targeted fix for remaining universities to extract Kenya PG entry requirements."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import logging
logging.getLogger("playwright").setLevel(logging.WARNING)

# Ensure project root on sys.path when running from elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processors.requirements_extractor import extract_requirements  # noqa: E402
from scrapers.page_interaction import click_accordions, launch_browser  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: extract_requirements now lives in processors.requirements_extractor


def scrape_salford() -> Optional[Dict]:
    """Salford: Kenya page with Entry Requirements accordion clicking."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b, ctx, page = launch_browser(p)
        try:
            resp = page.goto("https://www.salford.ac.uk/international/your-country-or-region/kenya",
                             timeout=45000, wait_until="domcontentloaded")
            if resp and resp.status >= 400:
                return None
            page.wait_for_timeout(5000)

            # Click ALL buttons and links that might reveal content
            # Try clicking "Entry requirements" link specifically
            for link_text in ["Entry requirements", "entry requirements", "Show more", "Read more"]:
                try:
                    link = page.get_by_text(link_text, exact=False).first
                    if link:
                        link.click()
                        page.wait_for_timeout(1000)
                except Exception:
                    pass

            # Click all accordion buttons
            click_accordions(page)

            page.wait_for_timeout(2000)
            body = page.inner_text("body")
            reqs = extract_requirements(body)
            return {
                "institution": "University of Salford",
                "institution_key": "salford_university",
                "source_url": "https://www.salford.ac.uk/international/your-country-or-region/kenya",
                "text_length": len(body),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                **reqs,
            }
        except Exception as exc:
            return {"institution": "University of Salford", "error": str(exc)}
        finally:
            ctx.close()
            b.close()


def scrape_southwales() -> Optional[Dict]:
    """South Wales: Kenya page with detailed extraction."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b, ctx, page = launch_browser(p)
        try:
            # First try the Kenya page directly
            resp = page.goto("https://www.southwales.ac.uk/international/your-country/kenya/",
                             timeout=45000, wait_until="domcontentloaded")
            status = resp.status if resp else "none"
            page.wait_for_timeout(5000)
            body = page.inner_text("body")
            if len(body) > 200:
                reqs = extract_requirements(body)
                return {
                    "institution": "University of South Wales",
                    "institution_key": "south_wales_university",
                    "source_url": "https://www.southwales.ac.uk/international/your-country/kenya/",
                    "text_length": len(body),
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    **reqs,
                }
        except Exception:
            pass
        try:
            # Fallback to general country page
            resp = page.goto("https://www.southwales.ac.uk/international/your-country/",
                             timeout=45000, wait_until="domcontentloaded")
            page.wait_for_timeout(5000)
            body = page.inner_text("body")
            reqs = extract_requirements(body)
            return {
                "institution": "University of South Wales",
                "institution_key": "south_wales_university",
                "source_url": "https://www.southwales.ac.uk/international/your-country/",
                "text_length": len(body),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                **reqs,
            }
        except Exception as exc:
            return {"institution": "University of South Wales", "error": str(exc)}
        finally:
            ctx.close()
            b.close()


def scrape_bucks() -> Optional[Dict]:
    """Bucks: try to get Kenya-specific info from the Your Country page."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b, ctx, page = launch_browser(p)
        try:
            # Try the International page with longer timeout
            resp = page.goto("https://www.bucks.ac.uk/study/international/your-country",
                             timeout=60000, wait_until="domcontentloaded")
            if resp and resp.status < 400:
                page.wait_for_timeout(8000)
                # Look for Kenya link
                kenya_link = page.query_selector("a[href*='kenya' i]")
                if kenya_link:
                    kenya_link.click()
                    page.wait_for_timeout(5000)
                body = page.inner_text("body")
                if len(body) > 200:
                    reqs = extract_requirements(body)
                    return {
                        "institution": "Buckinghamshire New University",
                        "institution_key": "bucks_new_university",
                        "source_url": page.url,
                        "text_length": len(body),
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                        **reqs,
                    }
        except Exception:
            pass
        # Fallback to general entry requirements
        try:
            resp = page.goto("https://www.bucks.ac.uk/study/general-entry-requirements",
                             timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)
            body = page.inner_text("body")
            reqs = extract_requirements(body)
            return {
                "institution": "Buckinghamshire New University",
                "institution_key": "bucks_new_university",
                "source_url": "https://www.bucks.ac.uk/study/general-entry-requirements",
                "text_length": len(body),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                **reqs,
            }
        except Exception as exc:
            return {"institution": "Buckinghamshire New University", "error": str(exc)}
        finally:
            ctx.close()
            b.close()


def scrape_hertfordshire() -> Optional[Dict]:
    """Hertfordshire: try to find Kenya-specific entry requirements."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b, ctx, page = launch_browser(p)
        try:
            resp = page.goto("https://www.herts.ac.uk/international/apply/application-requirements",
                             timeout=60000, wait_until="domcontentloaded")
            if resp and resp.status < 400:
                page.wait_for_timeout(8000)
                # Try accordion clicks
                click_accordions(page)
                page.wait_for_timeout(2000)
                body = page.inner_text("body")
                if len(body) > 200:
                    reqs = extract_requirements(body)
                    return {
                        "institution": "University of Hertfordshire",
                        "institution_key": "hertfordshire_university",
                        "source_url": "https://www.herts.ac.uk/international/apply/application-requirements",
                        "text_length": len(body),
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                        **reqs,
                    }
        except Exception:
            pass
        try:
            resp = page.goto("https://www.herts.ac.uk/international/apply",
                             timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(5000)
            body = page.inner_text("body")
            reqs = extract_requirements(body)
            return {
                "institution": "University of Hertfordshire",
                "institution_key": "hertfordshire_university",
                "source_url": "https://www.herts.ac.uk/international/apply",
                "text_length": len(body),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                **reqs,
            }
        except Exception as exc:
            return {"institution": "University of Hertfordshire", "error": str(exc)}
        finally:
            ctx.close()
            b.close()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_records = []

    # Load existing results from the main scrape
    existing_files = sorted(Path("data/processed").glob("all_universities_kenya_requirements_*.json"))
    if existing_files:
        with open(existing_files[-1], "r") as f:
            all_records = json.load(f)

    # Extract existing records into a dict by key
    existing = {r["institution_key"]: r for r in all_records}

    print("=" * 70)
    print("FIXING REMAINING UNIVERSITIES")
    print("=" * 70)

    # 1. Salford - fix accordion
    print("\n1. University of Salford (fixing accordion)...")
    r = scrape_salford()
    if r and r.get("postgraduate_entry_requirement"):
        existing["salford_university"] = r
        print(f"   ✓ Found PG: {r['postgraduate_entry_requirement'][:100]}...")
    else:
        print(f"   ✗ Still no data: {r.get('error', 'unknown')}")

    # 2. South Wales - fix
    print("\n2. University of South Wales...")
    r = scrape_southwales()
    if r and r.get("postgraduate_entry_requirement"):
        existing["south_wales_university"] = r
        print(f"   ✓ Found PG: {r['postgraduate_entry_requirement'][:100]}...")
    else:
        print(f"   ✗ No data: {r.get('error', 'unknown')}")

    # 3. Bucks - fix
    print("\n3. Buckinghamshire New University...")
    r = scrape_bucks()
    if r and r.get("postgraduate_entry_requirement"):
        existing["bucks_new_university"] = r
        print(f"   ✓ Found PG: {r['postgraduate_entry_requirement'][:100]}...")
    else:
        print(f"   ✗ No data: {r.get('error', 'unknown')}")

    # 4. Hertfordshire - fix
    print("\n4. University of Hertfordshire...")
    r = scrape_hertfordshire()
    if r and r.get("postgraduate_entry_requirement"):
        existing["hertfordshire_university"] = r
        print(f"   ✓ Found PG: {r['postgraduate_entry_requirement'][:100]}...")
    else:
        print(f"   ✗ No data: {r.get('error', 'unknown')}")

    # Save updated results
    fieldnames = [
        "institution", "institution_key", "source_url", "text_length", "scraped_at",
        "postgraduate_entry_requirement", "english_language_requirement",
        "bachelor_degree_requirement", "foundation_requirement",
        "scholarship_info", "contact_info", "error",
    ]

    all_records = list(existing.values())
    csv_path = PROCESSED_DIR / f"all_universities_kenya_requirements_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    json_path = PROCESSED_DIR / f"all_universities_kenya_requirements_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for r in all_records:
        name = r["institution"]
        pg = r.get("postgraduate_entry_requirement")
        eng = r.get("english_language_requirement")
        err = r.get("error", "")
        status = "✓" if pg else (f"✗ ({err})" if err else "✗")
        print(f"  {name:<45} {status}")

    print(f"\nFiles saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
