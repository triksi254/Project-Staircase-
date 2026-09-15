"""Tests for the rule-based lead-scoring rubric (leads.rubric)."""
import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leads.rubric import (  # noqa: E402
    COLD_HOT_THRESHOLD,
    LeadScore,
    WEIGHTS,
    WARM_HOT_THRESHOLD,
    score_row,
    score_rows,
    summarize_scores,
)
from leads.features import PASSPORT_VALID, PASSPORT_EXPIRED_NOT_RENEWED, PASSPORT_NONE, FUNDING_CLEAR, FUNDING_UNCLEAR  # noqa: E402


def _base():
    return {
        "passport_status": PASSPORT_VALID,
        "destination_uk": 1,
        "funding_clarity": FUNDING_CLEAR,
        "has_course": 1,
        "has_intake": 1,
        "qual_level": 4,  # master
        "has_english_test": 1,
        "english_band": 7.0,
        "study_gap_mentioned": 0,
        "previous_application_mentioned": 1,
        "note_word_count": 20,
    }


def test_hot_lead_scores_0_95():
    r = score_row(_base())
    assert r.label == "Hot"
    assert r.score >= WARM_HOT_THRESHOLD


def test_cold_lead_no_passport_scores_cold():
    row = _base()
    row["passport_status"] = PASSPORT_NONE
    row["destination_uk"] = 0
    row["funding_clarity"] = FUNDING_UNCLEAR
    row["has_course"] = 0
    row["has_intake"] = 0
    row["has_english_test"] = 0
    row["english_band"] = 0.0
    row["study_gap_mentioned"] = 1
    row["previous_application_mentioned"] = 0
    row["note_word_count"] = 5
    r = score_row(row)
    assert r.label == "Cold"
    assert r.score < COLD_HOT_THRESHOLD


def test_expired_passport_drops_to_warm():
    row = _base()
    row["passport_status"] = PASSPORT_EXPIRED_NOT_RENEWED
    row["note_word_count"] = 30  # passport is the only change from a Hot lead
    r = score_row(row)
    # 0.95 base minus passport's 0.30 weight => 0.65, which is the Warm band
    assert r.label == "Warm"
    assert COLD_HOT_THRESHOLD <= r.score < WARM_HOT_THRESHOLD


def test_warm_band():
    row = _base()
    row["funding_clarity"] = FUNDING_UNCLEAR   # -0.20
    row["has_english_test"] = 0                 # -0.08
    row["english_band"] = 0.0
    row["note_word_count"] = 5                   # -0.05ish
    r = score_row(row)
    # 0.95 - 0.20 - 0.08 - ~0.05 => ~0.60, Warm band
    assert r.label == "Warm"
    assert COLD_HOT_THRESHOLD <= r.score < WARM_HOT_THRESHOLD


def test_score_row_returns_leadscore_and_contributions():
    r = score_row(_base())
    assert isinstance(r, LeadScore)
    assert len(r.contributions) == len(WEIGHTS)  # one entry per feature
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_contributions_sum_to_score():
    r = score_row(_base())
    total = round(sum(c["contribution"] for c in r.contributions) / sum(WEIGHTS.values()), 4)
    assert total == pytest.approx(r.score, abs=1e-4)


def test_score_rows_annotates():
    rows = [_base(), {**_base(), "passport_status": PASSPORT_NONE, "funding_clarity": FUNDING_UNCLEAR, "has_course": 0, "has_intake": 0, "has_english_test": 0, "english_band": 0.0, "study_gap_mentioned": 1, "previous_application_mentioned": 0, "note_word_count": 5}]
    scored = score_rows(rows)
    assert all("lead_score" in r and "lead_label" in r and "score_contributions" in r for r in scored)
    assert scored[0]["lead_label"] == "Hot"
    assert scored[1]["lead_label"] == "Cold"


def test_summarize_scores():
    rows = score_rows([_base(), _base(), {**_base(), "passport_status": PASSPORT_NONE, "destination_uk": 0, "funding_clarity": FUNDING_UNCLEAR, "has_course": 0, "has_intake": 0, "has_english_test": 0, "english_band": 0.0, "study_gap_mentioned": 1, "previous_application_mentioned": 0, "note_word_count": 5}])
    s = summarize_scores(rows)
    assert s["total"] == 3
    assert s["labels"]["Hot"] == 2
    assert s["labels"]["Cold"] == 1
    assert s["score_stats"]["min"] < s["score_stats"]["max"]


def test_empty_summarize():
    s = summarize_scores([])
    assert s["total"] == 0