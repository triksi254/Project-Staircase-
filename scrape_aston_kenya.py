"""Scrape Aston University Kenya qualification requirements for Postgraduate MSc.
Exports results to CSV.

Sources:
  - https://www.aston.ac.uk/international/aston-in-your-country/africa/kenya
  - English language requirements
  - MSc course pages

Output:
  data/processed/aston_university_kenya_requirements_<timestamp>.csv
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DIR  # noqa: E402

BASE = "https://www.aston.ac.uk"
KENYA_URL = f"{BASE}/international/aston-in-your-country/africa/kenya"
ENGLISH_URL = f"{BASE}/international/english-language-requirements"


class AstonKenyaScraper:
    """Scrape qualification requirements for Kenyan students applying for MSc."""

    SCRAPER_NAME = "aston_kenya_requirements"

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def scrape(self) -> List[Dict[str, Any]]:
        """Run the scraping pipeline."""
        from playwright.sync_api import sync_playwright

        self.metadata = {
            "institution": "aston_university",
            "country": "kenya",
            "data_type": "postgraduate_msc_requirements",
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "scraper_used": "playwright+manual_accordion",
        }

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
                viewport={"width": 1366, "height": 900},
                locale="en-GB",
            )
            page = context.new_page()
            page.set_default_timeout(60000)

            try:
                # --- Kenya page ---
                self._open_kenya_page(page)
                self._extract_kenya_requirements(page)

                # --- English language page ---
                self._open_english_language_page(page)
                self._extract_english_requirements(page)

            except Exception as exc:  # noqa: BLE001
                logger.exception("Scrape failed: {}", exc)
                self.errors.append(str(exc))
                # Save screenshot for debugging
                try:
                    shot_dir = Path("data") / "screenshots"
                    shot_dir.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(shot_dir / "aston_kenya_error.png"))
                    logger.info("Screenshot saved to {}", shot_dir / "aston_kenya_error.png")
                except Exception:  # noqa: BLE001
                    pass
            finally:
                context.close()
                browser.close()

        self.metadata["total_items"] = len(self.records)
        self.metadata["total_errors"] = len(self.errors)
        return self.records

    # ------------------------------------------------------------------ #
    # Kenya page
    # ------------------------------------------------------------------ #
    def _open_kenya_page(self, page) -> None:
        resp = page.goto(KENYA_URL, wait_until="domcontentloaded", timeout=60000)
        logger.info("Kenya page status: {}", resp.status if resp else "none")
        page.wait_for_timeout(3000)
        # Accept cookies if present
        try:
            accept = page.query_selector("button.agree-button")
            if accept:
                accept.click()
                page.wait_for_timeout(1000)
        except Exception:  # noqa: BLE001
            pass

    def _extract_kenya_requirements(self, page) -> None:
        """Extract all accordion sections from the Kenya page."""
        # Open each accordion toggler and capture its content
        toggler_sel = "a.ckeditor-accordion-toggler"
        try:
            togglers = page.query_selector_all(toggler_sel)
            logger.info("Found {} accordion togglers", len(togglers))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not list accordions: {}", exc)
            togglers = []

        # Map of section name to data
        sections: Dict[str, str] = {}

        for idx in range(len(togglers)):
            try:
                # Re-query each time because DOM may change after clicking
                current = page.query_selector_all(toggler_sel)
                if idx >= len(current):
                    break
                toggler = current[idx]
                title = (toggler.inner_text() or "").strip()
                if not title:
                    continue

                # Find the panel content next to this toggler
                # ckeditor accordions usually have a panel div
                expanded = toggler.get_attribute("aria-expanded")
                if expanded != "true":
                    try:
                        toggler.click()
                        page.wait_for_timeout(800)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not click '{}': {}", title, exc)

                # Extract content from the panel
                content = self._extract_accordion_content(page, title)
                if content:
                    sections[title] = content
                    logger.info("Extracted section '{}': {} chars", title, len(content))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Section extraction error: {}", exc)

        # Also capture the main intro text
        try:
            body_text = page.inner_text("main, #main-content, body")
            intro_match = re.search(
                r"Aston University is one of the leading career-focused universities.*?learn more\.?",
                body_text,
                re.S | re.I,
            )
            if intro_match:
                sections["Introduction"] = intro_match.group(0).strip()
        except Exception:  # noqa: BLE001
            pass

        # Convert sections into structured records
        self._records_from_sections(sections)

    def _extract_accordion_content(self, page, title: str) -> str:
        """Try to extract the visible panel content following an accordion toggler."""
        # Approach 1: aria-controls / aria-labelledby linkage
        try:
            linked = page.evaluate(
                """(title) => {
                    const togglers = document.querySelectorAll('a.ckeditor-accordion-toggler');
                    for (const t of togglers) {
                        if ((t.innerText||'').trim() === title) {
                            const controls = t.getAttribute('aria-controls');
                            if (controls) {
                                const panel = document.getElementById(controls);
                                if (panel) return panel.innerText.trim();
                            }
                        }
                    }
                    return null;
                }""",
                title,
            )
            if linked:
                return linked
        except Exception:  # noqa: BLE001
            pass

        # Approach 2: Get all visible text and find section boundaries
        try:
            text = page.inner_text("main, #main-content, body")
            # Find title position and extract up to the next toggler title
            markers = [
                "English Language Requirements",
                "International Foundation Year",
                "Undergraduate",
                "Postgraduate",
                "International Scholarships",
                "Contact us",
            ]
            lines = text.split("\n")
            start_idx = None
            for i, line in enumerate(lines):
                if line.strip().lower() == title.lower():
                    start_idx = i
                    break
            if start_idx is not None:
                end_idx = len(lines)
                for j in range(start_idx + 1, len(lines)):
                    stripped = lines[j].strip().lower()
                    if any(marker.lower() == stripped for marker in markers if marker.lower() != title.lower()):
                        end_idx = j
                        break
                section = "\n".join(lines[start_idx + 1 : end_idx]).strip()
                if section:
                    return section
        except Exception:  # noqa: BLE001
            pass

        return ""

    def _records_from_sections(self, sections: Dict[str, str]) -> None:
        """Turn extracted sections into structured records."""
        for section_name, content in sections.items():
            clean = re.sub(r"\s+", " ", content).strip()
            self.records.append(
                {
                    "institution": "Aston University",
                    "country": "Kenya",
                    "category": section_name,
                    "requirement_type": "postgraduate_msc",
                    "requirement": clean,
                    "source_url": KENYA_URL,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Also parse the Postgraduate section specifically into bullets
        pg = sections.get("Postgraduate", "")
        bullets = [b.strip() for b in re.split(r"[\n•·]", pg) if b.strip()]
        if bullets:
            for b in bullets:
                self.records.append(
                    {
                        "institution": "Aston University",
                        "country": "Kenya",
                        "category": "Postgraduate (MSc) - Bullet",
                        "requirement_type": "postgraduate_msc",
                        "requirement": b,
                        "source_url": KENYA_URL,
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

    # ------------------------------------------------------------------ #
    # English language page
    # ------------------------------------------------------------------ #
    def _open_english_language_page(self, page) -> None:
        resp = page.goto(ENGLISH_URL, wait_until="domcontentloaded", timeout=60000)
        logger.info("English language page status: {}", resp.status if resp else "none")
        page.wait_for_timeout(3000)

    def _extract_english_requirements(self, page) -> None:
        try:
            text = page.inner_text("main, #main-content, body")
            # Extract relevant English language requirement sections
            for section_title in ["Postgraduate", "IELTS", "English language requirements"]:
                idx = text.lower().find(section_title.lower())
                if idx >= 0:
                    snippet = text[idx : idx + 800].strip()
                    self.records.append(
                        {
                            "institution": "Aston University",
                            "country": "Kenya",
                            "category": f"English Language - {section_title}",
                            "requirement_type": "postgraduate_msc",
                            "requirement": re.sub(r"\s+", " ", snippet),
                            "source_url": ENGLISH_URL,
                            "scraped_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("English requirements extraction failed: {}", exc)


def export_csv(records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Path:
    """Export records to a CSV file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aston_university_kenya_requirements_{ts}.csv"
    path = PROCESSED_DIR / filename

    fieldnames = ["institution", "country", "category", "requirement_type", "requirement", "source_url", "scraped_at"]

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    logger.info("Exported {} records to CSV: {}", len(records), path)
    return path


def main() -> int:
    """Entry point."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    scraper = AstonKenyaScraper()
    records = scraper.scrape()

    if not records:
        print("No records scraped!")
        return 1

    # Export CSV
    csv_path = export_csv(records, scraper.metadata)

    # Also save JSON raw
    RAW_DIR = Path("data") / "raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RAW_DIR / f"aston_university_kenya_requirements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path.write_text(
        json.dumps({"metadata": scraper.metadata, "data": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print summary
    print()
    print("=" * 80)
    print("Scraping Results: aston_university/kenya/postgraduate_msc_requirements")
    print("=" * 80)
    print(f"Status: success")
    print(f"Scraped: {len(records)} items")
    print(f"Errors: {len(scraper.errors)}")
    print(f"Saved Files:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print("=" * 80)

    # Print the records
    print()
    print("Extracted Data:")
    for i, rec in enumerate(records, 1):
        print(f"\n[{i}] {rec['category']}")
        print(f"    {rec['requirement'][:200]}{'...' if len(rec['requirement']) > 200 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

