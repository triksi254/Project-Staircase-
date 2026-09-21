"""Unit tests for the lead-scoring behavioural feature extractor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leads.features import (  # noqa: E402
    FUNDING_CLEAR,
    FUNDING_PARTIAL,
    FUNDING_UNCLEAR,
    FUNDING_UNKNOWN,
    PASSPORT_EXPIRED_NOT_RENEWED,
    PASSPORT_NONE,
    PASSPORT_VALID,
    extract_features,
    write_outputs,
)


def _record(**overrides):
    base = {
        "crm_id": "999001",
        "rating": "Good",
        "qualifications": "",
        "english_test": "",
        "study_destination": "UK",
        "course": "",
        "intake": "",
        "funding_method": "",
        "assessment_notes": "",
        "extra_fields": {},
    }
    base.update(overrides)
    return base


def _row(**overrides):
    return extract_features([_record(**overrides)])[0]


# --------------------------------------------------------------------------- #
# Label encoding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("rating", "expected"),
    [("Cold", 0), ("Good", 1), ("Warm", 1), ("Excellent", 2), ("Hot", 2), ("", -1), ("NA", -1)],
)
def test_label_mapping(rating, expected):
    assert _row(rating=rating)["label"] == expected


# --------------------------------------------------------------------------- #
# Passport status
# --------------------------------------------------------------------------- #
def test_passport_expired_not_renewed():
    row = _row(assessment_notes="Student is yet to renew passport.")
    assert row["passport_status"] == PASSPORT_EXPIRED_NOT_RENEWED


def test_passport_none():
    row = _row(assessment_notes="Has no passport at the moment.")
    assert row["passport_status"] == PASSPORT_NONE


def test_passport_valid_default():
    row = _row(assessment_notes="Good student, strong English.")
    assert row["passport_status"] == PASSPORT_VALID


# --------------------------------------------------------------------------- #
# English test
# --------------------------------------------------------------------------- #
def test_english_test_no_means_none():
    row = _row(english_test="no")
    assert row["has_english_test"] == 0
    assert row["english_band"] == 0.0


def test_english_test_band_parsed():
    row = _row(english_test="IELTS 6.5")
    assert row["has_english_test"] == 1
    assert row["english_band"] == 6.5


def test_english_test_kcse_present():
    row = _row(english_test="KCSE English C+")
    assert row["has_english_test"] == 1


# --------------------------------------------------------------------------- #
# Funding clarity
# --------------------------------------------------------------------------- #
def test_funding_clear():
    row = _row(funding_method="Self Funded")
    assert row["funding_clarity"] == FUNDING_CLEAR


def test_funding_partial():
    row = _row(funding_method="looking for scholarship funding")
    assert row["funding_clarity"] == FUNDING_PARTIAL


def test_funding_unclear():
    row = _row(funding_method="has no funds at the moment")
    assert row["funding_clarity"] == FUNDING_UNCLEAR


def test_funding_unknown():
    row = _row(funding_method="")
    assert row["funding_clarity"] == FUNDING_UNKNOWN


# --------------------------------------------------------------------------- #
# Other features
# --------------------------------------------------------------------------- #
def test_destination_uk_and_course_intake():
    row = _row(study_destination="Canada", course="MBA", intake="Sep 2026")
    assert row["destination_uk"] == 0
    assert row["has_course"] == 1
    assert row["has_intake"] == 1


def test_qual_level_bachelor_and_master():
    assert _row(qualifications="BSc Computer Science")["qual_level"] == 3  # bachelor
    assert _row(qualifications="MSc Data Science")["qual_level"] == 4  # master
    assert _row(qualifications="NA")["qual_level"] == 0  # unknown


def test_study_gap_and_previous_application():
    row = _row(
        assessment_notes="Has a study gap to explain.",
        extra_fields={"Have you made any application to any universities yet?": "University of Derby"},
    )
    assert row["study_gap_mentioned"] == 1
    assert row["previous_application_mentioned"] == 1


def test_note_word_count():
    row = _row(assessment_notes="clear funding and strong english")
    assert row["note_word_count"] == 5


# --------------------------------------------------------------------------- #
# Output helpers & row shape
# --------------------------------------------------------------------------- #
def test_extract_features_row_shape():
    rows = extract_features([_record()])
    assert len(rows) == 1
    assert set(rows[0].keys()) == {
        "crm_id", "label", "passport_status", "has_english_test", "english_band",
        "funding_method_present", "funding_clarity", "destination_uk", "has_course",
        "has_intake", "qual_level", "study_gap_mentioned",
        "previous_application_mentioned", "note_word_count",
    }


def test_write_outputs(tmp_path):
    rows = extract_features([_record(), _record(rating="Cold")])
    csv_path, json_path = write_outputs(rows, tmp_path)
    assert csv_path.exists() and json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["metadata"]["rows"] == 2
    assert len(data["data"]) == 2

# --------------------------------------------------------------------------- #
# data-quality diagnostics (silent problems made loud)
# --------------------------------------------------------------------------- #
def test_summarize_reports_repeated_leads_empty_notes_and_constants():
    from leads.features import summarize

    recs = [
        _record(crm_id="1", rating="Cold"),
        _record(crm_id="1", rating="Cold"),          # same lead, second form
        _record(crm_id="2", rating="Good"),
        _record(crm_id="3", rating="Excellent"),
    ]
    s = summarize(recs)
    assert s["unique_crm_ids"] == 3
    assert s["repeated_crm_ids"] == 1
    assert s["rows_in_repeated_crm_ids"] == 2
    assert s["assessment_notes_nonempty_frac"] == 0.0
    assert "note_word_count" in s["constant_feature_columns"]


def test_summarize_sees_notes_when_they_exist():
    from leads.features import summarize

    s = summarize([_record(assessment_notes="strong candidate, funds ready"),
                   _record(crm_id="2")])
    assert s["assessment_notes_nonempty_frac"] == 0.5
    assert "note_word_count" not in s["constant_feature_columns"]


def test_features_cli_warns_about_empty_notes_and_repeated_leads(tmp_path, capsys):
    from leads.features import main

    src = tmp_path / "assessment_forms_cleaned_1.json"
    src.write_text(json.dumps([_record(crm_id="1"), _record(crm_id="1")]),
                   encoding="utf-8")
    assert main(["--input", str(src), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "assessment_notes is empty in every record" in out
    assert "CRM id that occurs more than once" in out
