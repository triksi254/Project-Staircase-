"""Behavioural feature extraction from cleaned counsellor assessment records.

Transforms records produced by ``CounsellorForms/extract_assessment_forms.py``
into a feature matrix aligned with the research proposal's lead-scoring rubric:
passport status, English-test evidence, funding clarity, study destination,
course/intake readiness, qualification level, study gap, previous applications,
and note length.

Labels encode the counsellor rating as the proposal's persona classes:
``Cold = 0``, ``Good (Warm) = 1``, ``Excellent (Hot) = 2``.

Usage:
    python -m leads.features --input CounsellorForms/output/assessment_forms_cleaned_*.json
    python -m leads.features --out data/processed
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.schemas import QualificationLevel  # noqa: E402

# --------------------------------------------------------------------------- #
# Label encoding:  Cold=0, Good/Warm=1, Excellent/Hot=2  (-1 = unrated)
# --------------------------------------------------------------------------- #
RATING_LABELS = {
    "cold": 0,
    "good": 1,
    "warm": 1,
    "excellent": 2,
    "hot": 2,
}

# --------------------------------------------------------------------------- #
# Feature constants
# --------------------------------------------------------------------------- #
PASSPORT_EXPIRED_NOT_RENEWED = 0
PASSPORT_NONE = 1
PASSPORT_VALID = 2

FUNDING_UNKNOWN = 0
FUNDING_UNCLEAR = 1
FUNDING_PARTIAL = 2
FUNDING_CLEAR = 3

#: Output columns in stable order (label first, then features).
FEATURE_COLUMNS = [
    "crm_id",
    "label",
    "passport_status",
    "has_english_test",
    "english_band",
    "funding_method_present",
    "funding_clarity",
    "destination_uk",
    "has_course",
    "has_intake",
    "qual_level",
    "study_gap_mentioned",
    "previous_application_mentioned",
    "note_word_count",
]

QUAL_LEVEL_ORDER = [
    "unknown", "foundation", "diploma", "bachelor", "master", "phd",
]


# --------------------------------------------------------------------------- #
# Low-level parsing helpers
# --------------------------------------------------------------------------- #
def _record_text(record: Dict[str, Any]) -> str:
    """Join the free-text fields of a record into one searchable blob."""
    parts = [
        record.get("assessment_notes", ""),
        record.get("qualifications", ""),
        record.get("english_test", ""),
        record.get("funding_method", ""),
        record.get("study_destination", ""),
    ]
    extra = record.get("extra_fields", {})
    if isinstance(extra, dict):
        parts.extend(str(v) for v in extra.values() if v)
    return " ".join(str(p) for p in parts)


def _passport_status(record: Dict[str, Any]) -> int:
    """Score passport status from free text (Appendix D heuristic)."""
    text = _record_text(record).lower()
    if "yet to renew passport" in text or "needs to renew passport" in text:
        return PASSPORT_EXPIRED_NOT_RENEWED
    if re.search(r"\bno passport\b", text) or re.search(r"\bdoes ?n'?t have a passport\b", text):
        return PASSPORT_NONE
    return PASSPORT_VALID


def _english_test_state(record: Dict[str, Any]) -> tuple[int, float]:
    """Return (has_english_test, band) from the english_test field."""
    raw = str(record.get("english_test", "") or "").strip()
    lower = raw.lower()
    if not raw or lower in {"no", "none", "n/a", "na", "-"}:
        return 0, 0.0
    band_match = re.search(r"(\d+(?:\.\d+)?)", raw)
    band = float(band_match.group(1)) if band_match else 0.0
    return 1, band


def _funding_method_present(record: Dict[str, Any]) -> int:
    raw = str(record.get("funding_method", "") or "").strip()
    return 1 if raw else 0


def _funding_clarity(record: Dict[str, Any]) -> int:
    """Classify funding clarity on a 1-3 scale (proposal rubric).

    3 = confirmed source/amount, 2 = actively working on it, 1 = unclear,
    0 = unknown (no information provided).
    """
    method = str(record.get("funding_method", "") or "").lower()
    if not method.strip():
        return FUNDING_UNKNOWN

    unclear = (
        "no funds",
        "no source of funds",
        "yet to discuss",
        "not sure",
        "unknown",
        "unclear",
        "to be discussed",
    )
    partial = (
        "looking for",
        "scholarship",
        "bursary",
        "awaiting",
        "may get",
        "pending",
        "try",
        "undecided",
        "explore",
    )
    clear = (
        "self funded",
        "self-funded",
        "parents",
        "sponsor",
        "savings",
        "has funds",
        "has savings",
        "loan approved",
        "paid",
        "fully funded",
        "own funds",
    )

    if any(k in method for k in unclear):
        return FUNDING_UNCLEAR
    if any(k in method for k in partial):
        return FUNDING_PARTIAL
    if any(k in method for k in clear):
        return FUNDING_CLEAR
    # Some explicit yes/no phrasing we cannot classify confidently
    if "fund" in method or "finance" in method:
        return FUNDING_CLEAR
    return FUNDING_PARTIAL


def _destination_uk(record: Dict[str, Any]) -> int:
    dest = str(record.get("study_destination", "") or "").lower()
    return 1 if "uk" in dest else 0


def _has_field(record: Dict[str, Any], field: str) -> int:
    return 1 if str(record.get(field, "") or "").strip() else 0


def _qual_level(record: Dict[str, Any]) -> str:
    """Normalized qualification level from the qualifications free text."""
    raw = str(record.get("qualifications", "") or "").strip()
    if not raw or raw.lower() in {"na", "n/a", "none", "[university]"}:
        return "unknown"
    enum = QualificationLevel.normalize(raw)
    if enum is not None:
        return enum.value
    lowered = raw.lower()
    if any(k in lowered for k in ("phd", "doctorate", "dphil")):
        return "phd"
    if any(k in lowered for k in ("master", "msc", "mba", "ma ", "meng", "postgraduate", "post-graduate")):
        return "master"
    if any(k in lowered for k in ("bachelor", "bsc", "beng", "ba ", "honours", "degree")):
        return "bachelor"
    if "diploma" in lowered:
        return "diploma"
    if "foundation" in lowered:
        return "foundation"
    return "unknown"


def _study_gap_mentioned(record: Dict[str, Any]) -> int:
    text = _record_text(record)
    if re.search(r"(?i)\bstudy gap\b|gap year|since the completion of your last studies", text):
        return 1
    extra = record.get("extra_fields", {})
    if isinstance(extra, dict):
        for q, a in extra.items():
            if "study gap" in str(q).lower() and str(a).strip() and str(a).strip().lower() not in {"no", "none", "n/a"}:
                return 1
    return 0


def _previous_application_mentioned(record: Dict[str, Any]) -> int:
    extra = record.get("extra_fields", {})
    if isinstance(extra, dict):
        for q, a in extra.items():
            if "application" in str(q).lower():
                answer = str(a).strip().lower()
                if answer and answer not in {"no", "none", "not yet", "n/a", "-"}:
                    return 1
    if re.search(r"(?i)\bapplied to\b|application (?:made|submitted)|applied (?:to|at) a university", _record_text(record)):
        return 1
    return 0


def _note_word_count(record: Dict[str, Any]) -> int:
    notes = str(record.get("assessment_notes", "") or "")
    return len(notes.split())


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_features(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build one feature row per assessment record.

    ``crm_id`` is kept as a non-feature identifier; ``label`` is -1 for
    unrated records so downstream ML steps can filter them out.
    """
    rows: List[Dict[str, Any]] = []
    for rec in records:
        rating = str(rec.get("rating", "") or "").strip().lower()
        has_eng, band = _english_test_state(rec)
        qual = _qual_level(rec)
        rows.append(
            {
                "crm_id": str(rec.get("crm_id", "") or ""),
                "label": RATING_LABELS.get(rating, -1),
                "passport_status": _passport_status(rec),
                "has_english_test": has_eng,
                "english_band": band,
                "funding_method_present": _funding_method_present(rec),
                "funding_clarity": _funding_clarity(rec),
                "destination_uk": _destination_uk(rec),
                "has_course": _has_field(rec, "course"),
                "has_intake": _has_field(rec, "intake"),
                "qual_level": QUAL_LEVEL_ORDER.index(qual) if qual in QUAL_LEVEL_ORDER else 0,
                "study_gap_mentioned": _study_gap_mentioned(rec),
                "previous_application_mentioned": _previous_application_mentioned(rec),
                "note_word_count": _note_word_count(rec),
            }
        )
    return rows


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return dataset-level statistics for the extracted features."""
    rows = extract_features(records)
    total = len(rows)
    labeled = [r for r in rows if r["label"] >= 0]
    label_counts: Dict[str, int] = {name: 0 for name in ("Cold", "Good", "Excellent")}
    for r in labeled:
        label_counts[("Cold", "Good", "Excellent")[r["label"]]] += 1

    def _frac(field: str, value: Any) -> float:
        return round(sum(1 for r in rows if r[field] == value) / total, 4) if total else 0.0

    return {
        "total_records": total,
        "labeled_records": len(labeled),
        "label_counts": label_counts,
        "passport_valid_frac": _frac("passport_status", PASSPORT_VALID),
        "passport_none_frac": _frac("passport_status", PASSPORT_NONE),
        "funding_clear_frac": _frac("funding_clarity", FUNDING_CLEAR),
        "funding_unclear_frac": _frac("funding_clarity", FUNDING_UNCLEAR),
        "has_english_test_frac": _frac("has_english_test", 1),
        "destination_uk_frac": _frac("destination_uk", 1),
    }


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    """Write the feature matrix as CSV and JSON; returns (csv_path, json_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"lead_features_{ts}.csv"
    json_path = out_dir / f"lead_features_{ts}.json"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEATURE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    payload = {"metadata": {"source": "counsellor_assessment_forms", "rows": len(rows)}, "data": rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract lead-scoring behavioural features")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "CounsellorForms" / "output",
        help="Path to a cleaned assessment JSON file OR a directory containing them "
        "(default: CounsellorForms/output)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Output directory for the feature matrix (default: data/processed)",
    )
    args = parser.parse_args(argv)

    input_path: Path = args.input
    if input_path.is_dir():
        candidates = sorted(input_path.glob("assessment_forms_cleaned_*.json"))
        if not candidates:
            print(f"No assessment_forms_cleaned_*.json found in {input_path}", file=sys.stderr)
            return 2
        input_path = candidates[-1]

    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    records = json.loads(input_path.read_text(encoding="utf-8"))
    rows = extract_features(records)
    csv_path, json_path = write_outputs(rows, args.out)

    print("=" * 70)
    print("LEAD FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Input:    {input_path}")
    print(f"Records:  {len(records)}")
    summary = summarize(records)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("Saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())