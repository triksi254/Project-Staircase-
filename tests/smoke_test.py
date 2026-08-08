"""End-to-end smoke test for the University Web Scraper.

Serves a sample HTML page locally, then runs the full pipeline:
scrape -> process -> FAQ enrich -> export (json/csv/database).

Usage:
    python tests/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DIR, RAW_DIR, get_settings  # noqa: E402
from faq.faq_integrator import FAQIntegrator  # noqa: E402
from processors.data_processor import DataProcessor  # noqa: E402
from scrapers.smart_scraper import SmartScraper  # noqa: E402
from storage.storage import CSVExporter, DatabaseStorage, JSONExporter  # noqa: E402

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Test University Courses</title></head>
<body>
  <div class="course-item" data-course-item>
    <h2 class="course-title">BSc Computer Science</h2>
    <a class="course-link" href="/courses/bsc-computer-science">View</a>
    <div class="course-description">A comprehensive degree in computer science covering software and fees.</div>
    <div class="qualification-level">BSc</div>
    <div class="course-duration">3 years</div>
    <div class="study-mode">Full-time</div>
    <div class="fees-amount">£9,250</div>
  </div>
  <div class="course-item" data-course-item>
    <h2 class="course-title">MSc Data Science</h2>
    <a class="course-link" href="/courses/msc-data-science">View</a>
    <div class="course-description">Advanced postgraduate course on data science and machine learning.</div>
    <div class="qualification-level">MSc</div>
    <div class="course-duration">1 year</div>
    <div class="study-mode">Part-time</div>
    <div class="fees-amount">£12,000</div>
  </div>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(SAMPLE_HTML.encode("utf-8"))

    def log_message(self, format, *args):  # noqa: A002
        pass


def main():
    # 1. Start local server
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    base_url = f"http://127.0.0.1:{port}"
    print(f"[smoke] Local test server running at {base_url}")

    # 2. Scrape using SmartScraper (force HTML since it's a static local page)
    scraper = SmartScraper("aston_university", "courses", force_html=True, delay=0)
    raw_items = scraper.scrape(url=base_url)
    metadata = scraper.metadata.to_dict()
    print(f"[smoke] Scraped {len(raw_items)} raw items using {metadata['scraper_used']}")
    assert len(raw_items) == 2, f"Expected 2 raw items, got {len(raw_items)}"

    # 3. Save raw
    raw_path = JSONExporter.export("aston_university", "courses", raw_items, metadata, RAW_DIR, data_kind="raw")
    print(f"[smoke] Raw saved to {raw_path}")

    # 4. Process
    processor = DataProcessor(validate=True)
    result = processor.process_courses(raw_items, institution="aston_university")
    print(f"[smoke] Processed {result.count} courses, {result.dropped} dropped, {len(result.errors)} errors")
    assert result.count == 2
    assert result.processed[0].level.value == "bachelor"
    assert result.processed[1].level.value == "master"
    assert result.processed[0].study_type.value == "full_time"

    # 5. FAQ enrich
    integrator = FAQIntegrator()
    enriched = integrator.enrich_records(result.processed, "courses")
    faq_count = sum(len(r.related_faqs) for r in enriched)
    print(f"[smoke] FAQ enriched records with {faq_count} related FAQs")
    assert faq_count > 0

    # 6. Export JSON + CSV
    enriched_items = [m.model_dump(mode="json") for m in enriched]
    processed_path = JSONExporter.export("aston_university", "courses", enriched_items, metadata, PROCESSED_DIR)
    csv_path = CSVExporter.export("aston_university", "courses", enriched_items, metadata, PROCESSED_DIR)
    print(f"[smoke] JSON exported to {processed_path}")
    print(f"[smoke] CSV exported to {csv_path}")

    # 7. Database
    settings = get_settings()
    db = DatabaseStorage(settings.DATABASE_URL)
    db.save_records("aston_university", "courses", enriched)
    stats = db.get_statistics()
    print(f"[smoke] Database statistics: {stats}")
    assert stats["courses"] >= 2

    # 8. Verify JSON content
    data = json.loads(processed_path.read_text(encoding="utf-8"))
    assert data["metadata"]["institution"] == "aston_university"
    assert len(data["data"]) == 2
    assert data["data"][0]["title"] == "BSc Computer Science"
    assert "related_faqs" in data["data"][0]
    print("[smoke] JSON content verified")

    server.shutdown()
    print("\n[smoke] ALL SMOKE TESTS PASSED ")


if __name__ == "__main__":
    main()


