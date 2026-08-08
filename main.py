"""University Web Scraper - Entry Point.

Scrapes institutional data (courses, accommodation) from university websites.

Usage:
    python main.py --institution aston_university --data-type courses
    python main.py --batch batch_config_example.json --export json csv
    python main.py --institution oxford_university --data-type postgraduate --force-js
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Ensure project root on sys.path when running from elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR, PROCESSED_DIR, RAW_DIR, get_settings  # noqa: E402
from faq.faq_integrator import FAQIntegrator  # noqa: E402
from processors.data_processor import DataProcessor  # noqa: E402
from scrapers.smart_scraper import SmartScraper  # noqa: E402
from storage.storage import DatabaseStorage, JSONExporter, CSVExporter  # noqa: E402


def setup_logging() -> None:
    """Configure loguru logging."""
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    settings.ensure_directories()
    log_path = settings.LOG_FILE
    logger.add(
        log_path,
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="30 days",
        enqueue=True,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="University Web Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--institution", type=str, help="Institution key (e.g. aston_university)")
    parser.add_argument("--data-type", type=str, help="Data type (courses, accommodation, postgraduate)")
    parser.add_argument("--batch", type=str, help="Path to batch config JSON file")
    parser.add_argument(
        "--export",
        nargs="+",
        default=["json"],
        choices=["json", "csv", "database"],
        help="Export formats (default: json)",
    )
    parser.add_argument("--force-js", action="store_true", help="Force JavaScript scraper (Playwright)")
    parser.add_argument("--force-html", action="store_true", help="Force HTML scraper (BeautifulSoup)")
    parser.add_argument("--delay", type=float, default=None, help="Delay between requests (seconds)")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to scrape")
    parser.add_argument("--worker-id", type=int, default=None, help="Worker ID for distributed scraping")
    parser.add_argument("--no-validate", action="store_true", help="Disable Pydantic validation")
    parser.add_argument("--no-faq", action="store_true", help="Disable FAQ integration")
    return parser.parse_args()


def load_batch_config(path: str) -> List[Dict[str, Any]]:
    """Load batch job configuration from a JSON file."""
    batch_path = Path(path)
    if not batch_path.exists():
        # Try relative to project root
        batch_path = PROJECT_ROOT / path
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch config not found: {path}")
    data = json.loads(batch_path.read_text(encoding="utf-8"))
    return data.get("jobs", [])


def run_job(
    institution: str,
    data_type: str,
    formats: List[str],
    force_js: bool = False,
    force_html: bool = False,
    delay: Optional[float] = None,
    max_pages: Optional[int] = None,
    validate: bool = True,
    integrate_faq: bool = True,
) -> Dict[str, Any]:
    """Run a single scraping job end-to-end."""
    settings = get_settings()
    started = datetime.now(timezone.utc)

    logger.info("=" * 80)
    logger.info("Job: {} / {}", institution, data_type)
    logger.info("=" * 80)

    # 1. Scrape
    scraper = SmartScraper(
        institution,
        data_type,
        force_html=force_html,
        force_js=force_js,
        delay=delay,
        max_pages=max_pages,
    )
    raw_items = scraper.scrape()
    metadata = scraper.metadata.to_dict()

    # 2. Save raw data
    raw_path = JSONExporter.export(
        institution, data_type, raw_items, metadata, RAW_DIR, data_kind="raw"
    )

    # 3. Process / validate
    processor = DataProcessor(validate=validate)
    result = processor.process(raw_items, data_type, institution=institution)
    processed_items = [m.model_dump(mode="json") for m in result.processed]

    # 4. FAQ integration
    if integrate_faq and settings.INTEGRATE_FAQ:
        integrator = FAQIntegrator(settings=settings)
        enriched_records = integrator.enrich_records(result.processed, data_type)
        enriched_items = [m.model_dump(mode="json") for m in enriched_records]
    else:
        enriched_records = result.processed
        enriched_items = processed_items

    # 5. Storage / export
    saved_files: List[str] = []
    if "json" in formats:
        path = JSONExporter.export(
            institution, data_type, enriched_items, metadata, PROCESSED_DIR
        )
        saved_files.append(str(path))
    if "csv" in formats:
        path = CSVExporter.export(
            institution, data_type, enriched_items, metadata, PROCESSED_DIR
        )
        saved_files.append(str(path))
    if "database" in formats:
        db = DatabaseStorage(settings.DATABASE_URL)
        db.save_records(institution, data_type, enriched_records)
        stats = db.get_statistics()
    else:
        stats = {}

    # 6. Summary
    completed = datetime.now(timezone.utc)
    summary = {
        "institution": institution,
        "data_type": data_type,
        "status": "success",
        "started": started.isoformat(),
        "completed": completed.isoformat(),
        "scraper": metadata.get("scraper_used"),
        "scraped": len(raw_items),
        "processed": len(processed_items),
        "enriched": len(enriched_items),
        "errors": len(result.errors) + metadata.get("total_errors", 0),
        "saved_files": saved_files,
        "database_stats": stats,
    }
    _print_summary(summary)
    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print a human-readable summary of a job run."""
    line = "=" * 80
    print()
    print(line)
    print(f"Scraping Results: {summary['institution']}/{summary['data_type']}")
    print(line)
    print(f"Status: {summary['status']}")
    print(f"Started: {summary['started']}")
    print(f"Completed: {summary['completed']}")
    print()
    print("Summary:")
    print(f"  Scraped: {summary['scraped']} items")
    print(f"  Processed: {summary['processed']} items")
    print(f"  Enriched: {summary['enriched']} items")
    print(f"  Errors: {summary['errors']}")
    if summary["database_stats"]:
        print()
        print("Database Statistics:")
        for key, val in summary["database_stats"].items():
            print(f"  Total {key.title()}: {val}")
    if summary["saved_files"]:
        print()
        print("Saved Files:")
        for f in summary["saved_files"]:
            print(f"  - {f}")
    print(line)
    print()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging()

    settings = get_settings()
    validate = not args.no_validate
    integrate_faq = not args.no_faq

    jobs: List[Dict[str, Any]] = []
    if args.batch:
        jobs = load_batch_config(args.batch)
        # Apply global flags to all jobs
        for job in jobs:
            job.setdefault("force_js", args.force_js)
            job.setdefault("force_html", args.force_html)
    elif args.institution and args.data_type:
        jobs = [
            {
                "institution": args.institution,
                "data_type": args.data_type,
                "force_js": args.force_js,
                "force_html": args.force_html,
            }
        ]
    else:
        logger.error("Provide --institution/--data-type or --batch")
        print(__doc__)
        return 2

    # Optional worker-id filtering (distributed scraping)
    if args.worker_id is not None:
        jobs = [job for i, job in enumerate(jobs) if i % 3 == args.worker_id % 3]
        logger.info("Worker {} handling {} jobs", args.worker_id, len(jobs))

    results = []
    for job in jobs:
        try:
            summary = run_job(
                job["institution"],
                job["data_type"],
                formats=args.export,
                force_js=job.get("force_js", False),
                force_html=job.get("force_html", False),
                delay=args.delay,
                max_pages=args.max_pages,
                validate=validate,
                integrate_faq=integrate_faq,
            )
            results.append(summary)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job failed for {}/{}: {}", job.get("institution"), job.get("data_type"), exc)
            results.append(
                {
                    "institution": job.get("institution"),
                    "data_type": job.get("data_type"),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

