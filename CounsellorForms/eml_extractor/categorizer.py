"""Map raw Jotform question/answer pairs to standardized categories.

Also applies PII redaction to the free-text fields and derives the data
quality flag.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .models import AssessmentRecord
from .pii import has_pii, redact_pii

# --------------------------------------------------------------------------- #
# Field-keywords -> standardized field
# --------------------------------------------------------------------------- #
# Order matters: more specific rules first.
FIELD_RULES: List[Tuple[str, List[str]]] = [
    ("crm_id", ["crm id", "lead id", "lead number", "application id"]),
    ("qualifications", ["qualification", "degree", "academic", "education", "highest qualification"]),
    ("english_test", ["english", "ielts", "toefl", "kcse", "proficiency", "language"]),
    ("study_destination", ["study destination", "destination", "which country", "preferred country", "country are you"]),
    ("course", ["course", "programme", "program", "subject", "study preference", "what course"]),
    ("intake", ["intake", "start date", "start term", "which intake", "entry", "when do you"]),
    ("funding_method", ["funding", "sponsor", "self funded", "loan", "scholarship", "how are you funding"]),
    ("fund_amount", ["fund amount", "amount arranged", "budget", "how much"]),
    ("assessment_notes", ["comments", "notes", "feedback", "observation", "assessment"]),
    ("counsellor_id", ["counsellor", "username", "agent", "staff"]),
    ("rating", ["rating", "student readiness", "assessment rating", "grade"]),
]

RATINGS = {"cold", "good", "excellent"}

# Question keywords that indicate a personal/generic field we should NOT
# force into a standardized category (kept as extra/redacted).
_GENERIC_QUESTION = re.compile(
    r"(?i)(^|\W)(name|email|phone|whatsapp|marital|visa refusal|dependant|"
    r"nationality|country of residence|address|date of birth|age)(\W|$)"
)


def _classify_field(question: str) -> Optional[str]:
    """Return the standardized field key for a question, or None."""
    q = question.lower()
    if _GENERIC_QUESTION.search(q):
        return None
    for key, keywords in FIELD_RULES:
        for kw in keywords:
            if kw in q:
                return key
    return None


def _extract_rating(value: str) -> str:
    """Normalize a rating value to (Cold|Good|Excellent) or ''."""
    v = value.strip().lower()
    for r in RATINGS:
        if r in v:
            return r.capitalize()
    return ""


def _extract_crm_id(value: str) -> str:
    """Pull a CRM id (a run of digits) from the value."""
    m = re.search(r"\d{4,}", value)
    return m.group(0) if m else value.strip()


def build_record(
    parsed: Dict[str, object],
    replace_names: bool = True,
) -> AssessmentRecord:
    """Build a cleaned AssessmentRecord from the parsed .eml output."""
    rec = AssessmentRecord()
    fields: List[Tuple[str, str]] = parsed.get("fields", [])  # type: ignore[assignment]
    subject = str(parsed.get("subject", ""))
    headers = parsed.get("headers", {})
    rec.source_file = str(parsed.get("source_file", ""))
    rec.extraction_source = str(parsed.get("source", ""))
    rec.email_date = str(headers.get("date", "")) if isinstance(headers, dict) else ""

    seen: Dict[str, str] = {}
    last_key: Optional[str] = None

    for question, answer in fields:
        key = _classify_field(question)
        if key is None:
            # Empty question: carry over the previous field's context.
            if not question.strip() and last_key and last_key != "assessment_notes":
                seen.setdefault(last_key, answer)
                continue
            # Generic personal field (e.g. Name, Phone, Email) -> redact value.
            q = question.lower()
            if replace_names and re.search(r"\bname\b", q):
                rec.extra_fields[redact_pii(question, replace_names)] = "[NAME_REDACTED]"
            else:
                rec.extra_fields[redact_pii(question, replace_names)] = redact_pii(
                    answer, replace_names
                )
            continue
        seen[key] = answer
        last_key = key

    # Extract counsellor id, rating, and crm id from subject if not in fields.
    subject_crm = re.search(r"\b(\d{6})\b", subject)
    if "crm_id" not in seen and subject_crm:
        seen["crm_id"] = subject_crm.group(1)

    if "counsellor_id" not in seen:
        m = re.search(r"-\s*([A-Za-z]+)\s*-\s*(Cold|Good|Excellent)", subject)
        if m:
            # The value captured by the subject regex is the counsellor's name,
            # which is PII. Redact it immediately (it is not a system ID) rather
            # than relying on the name-list in pii.py to recognise it later.
            seen["counsellor_id"] = "[NAME_REDACTED]" if replace_names else m.group(1)
            if "rating" not in seen:
                seen["rating"] = m.group(2)

    # Map seen values into the record, redacting where appropriate.
    for key, raw in seen.items():
        # CRM ID is a system ID that must be preserved (not PII).
        if key == "crm_id":
            rec.crm_id = _extract_crm_id(raw)
            continue

        value = redact_pii(raw, replace_names)

        if key == "rating":
            rec.rating = _extract_rating(value)
        elif key == "qualifications":
            rec.qualifications = value or "[UNIVERSITY]"
        elif key == "english_test":
            rec.english_test = value
        elif key == "study_destination":
            rec.study_destination = value
        elif key == "course":
            rec.course = value
        elif key == "intake":
            rec.intake = value
        elif key == "funding_method":
            rec.funding_method = value
        elif key == "fund_amount":
            rec.fund_amount = value or "[AMOUNT_REDACTED]"
        elif key == "assessment_notes":
            rec.assessment_notes = value
        elif key == "counsellor_id":
            rec.counsellor_id = value or "[COUNSELLOR_ID]"
            if (
                replace_names
                and "REDACTED" not in rec.counsellor_id
                and "COUNSELLOR_ID" not in rec.counsellor_id
            ):
                # A bare person identifier (name/username) is PII - redact it.
                rec.counsellor_id = "[NAME_REDACTED]"

    # Post-process: counsellor identity must never remain in the record when
    # names are being redacted (the pii name-list only covers known names).
    if (
        replace_names
        and rec.counsellor_id
        and "REDACTED" not in rec.counsellor_id
        and "COUNSELLOR_ID" not in rec.counsellor_id
    ):
        rec.counsellor_id = "[NAME_REDACTED]"

    # Quality flag.
    flag = "CLEAN"
    if not rec.rating or not rec.crm_id:
        flag = "CHECK"
    if has_pii(rec.assessment_notes) or has_pii(rec.qualifications):
        rec.pii_issues = "PII not fully redacted"
        flag = "REVIEW"
    if has_pii(rec.counsellor_id) and "REDACTED" not in rec.counsellor_id:
        rec.pii_issues = (rec.pii_issues + "; counsellor id").strip(";")
        flag = "REVIEW"
    rec.data_quality_flag = flag

    return rec
