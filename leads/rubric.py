"""Explainable rule-based lead-scoring rubric (Cold / Warm / Hot).

Converts feature rows from ``leads.features`` into a score in [0, 1] with an
auditable per-feature rationale. This is the calibration prior for the hybrid
score ``alpha * rule_score + (1 - alpha) * ml_score`` (proposal Page 8).
No ML stack needed (no numpy/pandas/scikit-learn); it imports ``leads.features``,
which in turn imports the scraper's Pydantic ``models.schemas``.

The label cuts below (0.35 / 0.68) are this module's own descriptive cuts, used
for the rubric-vs-counsellor agreement table. The hybrid score uses tertiles
(``leads.hybrid.COLD_WARM_CUT`` / ``WARM_HOT_CUT``); the two disagree only in
[1/3, 0.35) and [2/3, 0.68) (pinned by ``tests/test_config_consistency.py``).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leads.features import (  # noqa: E402
    FEATURE_COLUMNS,
    PASSPORT_EXPIRED_NOT_RENEWED,
    PASSPORT_NONE,
    PASSPORT_VALID,
    QUAL_LEVEL_ORDER,
    FUNDING_UNKNOWN,
    FUNDING_UNCLEAR,
    FUNDING_PARTIAL,
    FUNDING_CLEAR,
)

COLD_HOT_THRESHOLD = 0.35
WARM_HOT_THRESHOLD = 0.68

WEIGHTS: Dict[str, float] = {
    "passport_status": 0.30,
    "destination_uk": 0.10,
    "funding_clarity": 0.20,
    "has_course": 0.05,
    "has_intake": 0.05,
    "qual_level": 0.10,
    "english_test": 0.08,
    "study_gap": 0.04,
    "previous_app": 0.03,
    "note_completeness": 0.05,
}
WEIGHT_SUM = sum(WEIGHTS.values())  # == 1.0
ENGLISH_BAND_THRESHOLD = 6.0


class LeadScore:
    def __init__(self, score: float, label: str, contributions: List[Dict[str, Any]]):
        self.score = round(score, 4)
        self.label = label
        self.contributions = contributions

    @property
    def is_hot(self) -> bool:
        return self.label == "Hot"

    @property
    def is_cold(self) -> bool:
        return self.label == "Cold"

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "label": self.label, "contributions": self.contributions}

    def __repr__(self) -> str:
        return f"LeadScore(score={self.score:.2f}, label={self.label})"


# --------------------------------------------------------------------------- #
# Per-feature contribution calculators
# --------------------------------------------------------------------------- #
def _passport_contrib(value: int) -> float:
    """Map passport_status -> contribution in [0, WEIGHTS[passport_status]]."""
    if value == PASSPORT_VALID:
        return WEIGHTS["passport_status"]
    # expired/not renewed OR no passport -> strong negative signal
    return 0.0


def _funding_contrib(value: int) -> float:
    """Map funding_clarity -> contribution in [0, WEIGHTS[funding_clarity]]."""
    levels = {
        FUNDING_UNKNOWN: 0.0,
        FUNDING_UNCLEAR: 0.0,
        FUNDING_PARTIAL: 0.33,
        FUNDING_CLEAR: 1.0,
    }
    frac = levels.get(value, 0.0)
    return frac * WEIGHTS["funding_clarity"]


def _qual_contrib(qual_level: int) -> float:
    """qual_level is already 0..5 (index into QUAL_LEVEL_ORDER)."""
    max_level = len(QUAL_LEVEL_ORDER) - 1
    frac = qual_level / max_level if max_level else 0.0
    return frac * WEIGHTS["qual_level"]


def _english_contrib(row: Dict[str, Any]) -> float:
    """has_english_test + band quality."""
    if not row.get("has_english_test"):
        return 0.0
    band = float(row.get("english_band") or 0.0)
    if band == 0.0:
        return 0.5 * WEIGHTS["english_test"]  # test present but band not parsed
    if band >= ENGLISH_BAND_THRESHOLD:
        return WEIGHTS["english_test"]
    return (band / ENGLISH_BAND_THRESHOLD) * WEIGHTS["english_test"]


def _note_completeness_contrib(row: Dict[str, Any]) -> float:
    """Longer, evidence-bearing notes -> more ready lead (capped)."""
    words = int(row.get("note_word_count") or 0)
    frac = min(words / 40.0, 1.0)  # 40 words ~ full credit
    return frac * WEIGHTS["note_completeness"]


def _bool_contrib(flag: Any, weight_key: str) -> float:
    return (1.0 if flag else 0.0) * WEIGHTS[weight_key]


# --------------------------------------------------------------------------- #
# Core scorer
# --------------------------------------------------------------------------- #
def score_row(row: Dict[str, Any]) -> LeadScore:
    """Score a single feature row (output of leads.features.extract_features)."""
    contribs: List[Dict[str, Any]] = []

    def add(name: str, value: float, weight: float, detail: Optional[str] = None):
        contribs.append(
            {"feature": name, "contribution": round(value, 4), "weight": round(weight, 4), "detail": detail}
        )

    # 1. passport
    ps = int(row.get("passport_status") or 0)
    add("passport_status", _passport_contrib(ps), WEIGHTS["passport_status"],
        {PASSPORT_VALID: "valid passport",
         PASSPORT_EXPIRED_NOT_RENEWED: "expired / needs renewal",
         PASSPORT_NONE: "no passport reported"}.get(ps, "unknown"))

    # 2. destination_uk
    add("destination_uk", _bool_contrib(row.get("destination_uk"), "destination_uk"), WEIGHTS["destination_uk"])

    # 3. funding_clarity
    fc = int(row.get("funding_clarity") or 0)
    add("funding_clarity", _funding_contrib(fc), WEIGHTS["funding_clarity"], f"level={fc}")

    # 4. course readiness
    add("has_course", _bool_contrib(row.get("has_course"), "has_course"), WEIGHTS["has_course"])

    # 5. intake readiness
    add("has_intake", _bool_contrib(row.get("has_intake"), "has_intake"), WEIGHTS["has_intake"])

    # 6. qualification level
    ql = int(row.get("qual_level") or 0)
    add("qual_level", _qual_contrib(ql), WEIGHTS["qual_level"],
        QUAL_LEVEL_ORDER[ql] if ql < len(QUAL_LEVEL_ORDER) else "unknown")

    # 7. english test
    add("english_test", _english_contrib(row), WEIGHTS["english_test"])

    # 8. study gap (gap mentioned -> 0 contribution)
    sg = int(row.get("study_gap_mentioned") or 0)
    add("study_gap", (0.0 if sg else WEIGHTS["study_gap"]), WEIGHTS["study_gap"],
        "gap mentioned" if sg else "no gap")

    # 9. previous application (engagement signal)
    pa = int(row.get("previous_application_mentioned") or 0)
    add("previous_app", _bool_contrib(pa, "previous_app"), WEIGHTS["previous_app"])

    # 10. note completeness
    add("note_completeness", _note_completeness_contrib(row), WEIGHTS["note_completeness"])

    total = sum(c["contribution"] for c in contribs)
    score = total / WEIGHT_SUM
    label = "Hot" if score >= WARM_HOT_THRESHOLD else ("Cold" if score < COLD_HOT_THRESHOLD else "Warm")
    return LeadScore(score=score, label=label, contributions=contribs)


def score_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score a list of feature rows; returns rows annotated with score/label/contributions."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        ls = score_row(row)
        merged = dict(row)
        merged["lead_score"] = ls.score
        merged["lead_label"] = ls.label
        merged["score_contributions"] = ls.contributions
        out.append(merged)
    return out


# --------------------------------------------------------------------------- #
# Summary statistics (rubric-level -- calibrate alpha)
# --------------------------------------------------------------------------- #
def summarize_scores(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate rule-based score distribution for calibrating alpha."""
    n = len(scored)
    if n == 0:
        return {"total": 0, "labels": {}, "score_stats": {}}
    counts: Dict[str, int] = {"Cold": 0, "Warm": 0, "Hot": 0}
    scores = []
    for r in scored:
        counts[r["lead_label"]] = counts.get(r["lead_label"], 0) + 1
        scores.append(r["lead_score"])
    return {
        "total": n,
        "labels": counts,
        "score_stats": {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "mean": round(sum(scores) / n, 4),
            "hot_frac": round(counts.get("Hot", 0) / n, 4),
            "cold_frac": round(counts.get("Cold", 0) / n, 4),
        },
    }


def write_outputs(scored: List[Dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    """Write scored feature matrix (CSV + JSON)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"lead_scores_{ts}.csv"
    json_path = out_dir / f"lead_scores_{ts}.json"

    fieldnames = FEATURE_COLUMNS + ["lead_score", "lead_label", "score_contributions"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in scored:
            row = dict(r)
            row["score_contributions"] = json.dumps(r.get("score_contributions", []))
            writer.writerow({k: row.get(k) for k in fieldnames})

    payload = {"metadata": {"rows": len(scored)}, "data": scored}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rule-based Cold/Warm/Hot lead scoring (rubric prior for alpha)"
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Feature-matrix JSON/CSV from leads.features (default: latest data/processed/lead_features_*)",
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-row score + top contributions")
    args = parser.parse_args(argv)

    input_path: Path = args.input if args.input else None
    if input_path is None:
        candidates = sorted((PROJECT_ROOT / "data" / "processed").glob("lead_features_*.json"))
        if not candidates:
            print("No feature matrix found. Run: python -m leads.features first.", file=sys.stderr)
            return 2
        input_path = candidates[-1]

    rows: List[Dict[str, Any]]
    if input_path.suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    else:
        with open(input_path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))

    scored = score_rows(rows)
    csv_path, json_path = write_outputs(scored, args.out)
    summary = summarize_scores(scored)

    print("=" * 70)
    print("RULE-BASED LEAD SCORING (Cold / Warm / Hot)")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Records: {summary['total']}")
    print(f"Labels: {summary['labels']}")
    print(f"Scores: {summary['score_stats']}")
    print(f"Saved:  {csv_path}")
    print(f"        {json_path}")

    if args.verbose:
        for r in scored:
            print(f"\n  {r.get('crm_id', '?')}: {r['lead_label']} ({r['lead_score']:.2f})")
            for c in r.get("score_contributions", []):
                if c["contribution"] > 0:
                    print(f"    +{c['contribution']:.3f}  {c['feature']} ({c.get('detail', '')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
