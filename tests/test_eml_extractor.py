"""Unit tests for the CounsellorForms EML extractor (parser, PII, categorizer).

These cover the most bespoke code in the repo: .eml parsing, the PII redaction
rules (including the contextual counsellor-name redaction that protects unseen
names), and record building.
"""
from __future__ import annotations

import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COUNSELLOR_ROOT = PROJECT_ROOT / "CounsellorForms"
if str(COUNSELLOR_ROOT) not in sys.path:
    sys.path.insert(0, str(COUNSELLOR_ROOT))

from eml_extractor.categorizer import build_record  # noqa: E402
from eml_extractor.pii import has_pii, redact_pii  # noqa: E402
from eml_extractor.parser import parse_eml  # noqa: E402
from eml_extractor.pipeline import process_file  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_eml(
    subject: str = "Re_ Kenya Office - Counsellor Assessment Form - Brian - Good - 999001 -",
    html: str | None = None,
    plain: str | None = None,
) -> bytes:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = "jotform@example.com"
    msg["Date"] = "Thu, 03 Apr 2025 03:34:07 -0400"
    if html:
        msg.attach(MIMEText(html, "html"))
    if plain:
        msg.attach(MIMEText(plain, "plain"))
    return msg.as_bytes()


HTML_TABLE = """<html><body>
<table id="emailFieldsTable">
  <tr><td id="question1">What are the highest qualifications?</td><td id="value1">Bachelor of Science - University of Nairobi</td></tr>
  <tr><td id="question2">English proficiency</td><td id="value2">IELTS 6.5</td></tr>
  <tr><td id="question3">Preferred study destination</td><td id="value3">UK</td></tr>
  <tr><td id="question4">Course of interest</td><td id="value4">MSc Data Science</td></tr>
  <tr><td id="question5">Intake</td><td id="value5">September 2026</td></tr>
  <tr><td id="question6">Funding method</td><td id="value6">Self funded</td></tr>
  <tr><td id="question7">Fund amount</td><td id="value7">17000</td></tr>
  <tr><td id="question8">Counsellor notes</td><td id="value8">Student has a valid passport.</td></tr>
</table>
</body></html>
"""

PLAIN_BODY = """Jotform Logo
What are the highest qualifications?
Bachelor of Science - University of Nairobi
English proficiency
IELTS 6.5
Preferred study destination
UK
Course of interest
MSc Data Science
"""


# --------------------------------------------------------------------------- #
# PII redaction
# --------------------------------------------------------------------------- #
def test_redact_email_phone_passport_amount():
    text = "Email me at jane@example.com or call +254 702 102 295 (passport 2024-2034), budget 17000 pounds"
    out = redact_pii(text)
    assert "[EMAIL_REDACTED]" in out
    assert "[PHONE_REDACTED]" in out
    assert "[PASSPORT_REDACTED]" in out
    assert "[AMOUNT_REDACTED]" in out
    assert "jane@example.com" not in out
    assert "17000" not in out


def test_redact_known_counsellor_name():
    out = redact_pii("Counsellor Teresia reviewed the lead")
    assert "Teresia" not in out
    assert "[NAME_REDACTED]" in out


def test_redact_contextual_unseen_name_in_filename():
    filename = "Re_ Kenya Office - Counsellor Assessment Form - Brian Omondi - Good - 999001 -.eml[1].eml"
    out = redact_pii(filename)
    assert "Brian" not in out and "Omondi" not in out
    assert "[NAME_REDACTED]" in out
    # The original filename contains recognizable PII; the redacted one does not.
    assert has_pii(filename) is True
    assert has_pii(out) is False


def test_redact_contextual_unseen_name_in_subject():
    subject = "Re_ Kenya Office - Counsellor Assessment Form - Brian Omondi - Good - 999001 -"
    out = redact_pii(subject)
    assert "Brian" not in out and "Omondi" not in out
    assert "[NAME_REDACTED]" in out


def test_redact_no_pii_leaves_text_unchanged():
    text = "This is a normal assessment note about funding and course preference."
    assert redact_pii(text) == text
    assert has_pii(text) is False


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def test_parse_eml_html_fields(tmp_path):
    eml = tmp_path / "sample.eml"
    eml.write_bytes(_make_eml(html=HTML_TABLE))
    parsed = parse_eml(eml)
    assert parsed["source"] == "html"
    fields = dict(parsed["fields"])
    assert "IELTS 6.5" in fields.get("English proficiency", "")
    assert "UK" in fields.get("Preferred study destination", "")
    assert parsed["subject"].startswith("Re_ Kenya Office")


def test_parse_eml_plain_fallback(tmp_path):
    eml = tmp_path / "plain.eml"
    eml.write_bytes(_make_eml(html="<html><body>no table here</body></html>", plain=PLAIN_BODY))
    parsed = parse_eml(eml)
    assert parsed["source"] == "plain"
    fields = dict(parsed["fields"])
    assert "IELTS 6.5" in fields.get("English proficiency", "")


# --------------------------------------------------------------------------- #
# Categorizer / pipeline
# --------------------------------------------------------------------------- #
def test_build_record_redacts_unseen_subject_name(tmp_path):
    eml = tmp_path / "unseen.eml"
    eml.write_bytes(_make_eml(html=HTML_TABLE))
    parsed = parse_eml(eml)
    parsed["source_file"] = eml.name
    rec = build_record(parsed, replace_names=True)
    # Single-word counsellor name from the subject must never survive.
    assert rec.counsellor_id == "[NAME_REDACTED]"
    assert rec.rating == "Good"
    assert rec.crm_id == "999001"


def test_process_file_end_to_end_no_pii(tmp_path):
    eml = tmp_path / "Re_ Kenya Office - Counsellor Assessment Form - Brian Omondi - Good - 999001 -.eml[1].eml"
    eml.write_bytes(_make_eml(html=HTML_TABLE))
    rec = process_file(eml, replace_names=True)
    assert rec is not None
    assert rec.crm_id == "999001"
    assert rec.rating == "Good"
    # None of the output fields may still contain the counsellor's name.
    for field in ("counsellor_id", "source_file", "assessment_notes"):
        value = getattr(rec, field, "")
        assert "Brian" not in value and "Omondi" not in value, f"PII leak in {field}: {value}"
    assert has_pii(rec.source_file) is False
    assert has_pii(rec.assessment_notes) is False
