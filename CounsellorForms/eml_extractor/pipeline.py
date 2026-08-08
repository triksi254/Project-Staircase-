"""Orchestrate the extraction pipeline: parse -> categorize -> validate -> export.

Logs a processing report with success/failure counts.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .categorizer import build_record
from .models import COLUMNS, AssessmentRecord, now_tag
from .parser import parse_eml
from .pii import has_pii, redact_pii


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}")


def process_file(
    eml_path: Path,
    replace_names: bool = True,
) -> Optional[AssessmentRecord]:
    """Parse and clean a single .eml file. Returns None on failure."""
    try:
        parsed = parse_eml(eml_path)
        # Redact PII embedded in the source filename (e.g. phone numbers).
        parsed["source_file"] = redact_pii(eml_path.name, replace_names=replace_names)
        rec = build_record(parsed, replace_names=replace_names)
        return rec
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR processing {eml_path.name}: {exc}")
        return None


def process_directory(
    root: Path,
    replace_names: bool = True,
) -> List[AssessmentRecord]:
    """Recursively process all .eml files under *root*."""
    eml_files = sorted(root.rglob("*.eml"))
    _log(f"Found {len(eml_files)} .eml files under {root}")

    records: List[AssessmentRecord] = []
    failed: List[str] = []
    for eml in eml_files:
        rec = process_file(eml, replace_names=replace_names)
        if rec is None:
            failed.append(eml.name)
        else:
            records.append(rec)

    _log(f"Processed: {len(records)} OK, {len(failed)} failed")
    if failed:
        _log("Failed files:")
        for name in failed[:20]:
            _log(f"  - {name}")
    return records


def export_csv(records: List[AssessmentRecord], out_path: Path) -> None:
    """Write records to a CSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_row())
    _log(f"Wrote CSV: {out_path} ({len(records)} rows)")


def export_json(records: List[AssessmentRecord], out_path: Path) -> None:
    """Write records to a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [rec.to_dict() for rec in records]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    _log(f"Wrote JSON: {out_path} ({len(records)} records)")


def run_pipeline(
    forms_root: Path,
    out_dir: Path,
    replace_names: bool = True,
) -> Dict[str, object]:
    """Run the full pipeline and return a summary."""
    records = process_directory(forms_root, replace_names=replace_names)

    tag = now_tag()
    csv_path = out_dir / f"assessment_forms_cleaned_{tag}.csv"
    json_path = out_dir / f"assessment_forms_cleaned_{tag}.json"

    export_csv(records, csv_path)
    export_json(records, json_path)

    # Quality summary
    total = len(records)
    clean = sum(1 for r in records if r.data_quality_flag == "CLEAN")
    check = sum(1 for r in records if r.data_quality_flag == "CHECK")
    review = sum(1 for r in records if r.data_quality_flag == "REVIEW")
    ratings: Dict[str, int] = {}
    for r in records:
        ratings[r.rating] = ratings.get(r.rating, 0) + 1

    # Final PII scan on the export.
    pii_leaks = 0
    for rec in records:
        for col in COLUMNS:
            val = getattr(rec, col, "")
            if isinstance(val, str) and has_pii(val):
                pii_leaks += 1
                break

    summary = {
        "total": total,
        "clean": clean,
        "check": check,
        "review": review,
        "ratings": ratings,
        "csv": str(csv_path),
        "json": str(json_path),
        "pii_leaks": pii_leaks,
    }
    _log(f"Summary: {summary}")
    return summary
