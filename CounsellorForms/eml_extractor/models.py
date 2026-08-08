"""Data model for a cleaned assessment record (stdlib-only)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional


@dataclass
class AssessmentRecord:
    """A single cleaned, PII-free assessment form record."""

    # Identity / source
    source_file: str = ""
    crm_id: str = ""
    counsellor_id: str = "[COUNSELLOR_ID]"
    rating: str = ""
    email_date: str = ""

    # Assessment data
    qualifications: str = "[UNIVERSITY]"
    english_test: str = ""
    study_destination: str = ""
    course: str = ""
    intake: str = ""
    funding_method: str = ""
    fund_amount: str = "[AMOUNT_REDACTED]"
    assessment_notes: str = ""

    # Quality
    data_quality_flag: str = "CLEAN"
    extraction_source: str = ""
    pii_issues: str = ""

    # Extra fields captured from the form (kept for completeness)
    extra_fields: Dict[str, str] = field(default_factory=dict)

    def to_row(self) -> Dict[str, str]:
        """Return a flat dict of the standard output columns."""
        return {
            "source_file": self.source_file,
            "crm_id": self.crm_id,
            "counsellor_id": self.counsellor_id,
            "rating": self.rating,
            "email_date": self.email_date,
            "qualifications": self.qualifications,
            "english_test": self.english_test,
            "study_destination": self.study_destination,
            "course": self.course,
            "intake": self.intake,
            "funding_method": self.funding_method,
            "fund_amount": self.fund_amount,
            "assessment_notes": self.assessment_notes,
            "data_quality_flag": self.data_quality_flag,
            "extraction_source": self.extraction_source,
            "pii_issues": self.pii_issues,
        }

    def to_dict(self) -> Dict[str, object]:
        """Return dict including extra_fields for JSON output."""
        data = asdict(self)
        return data


# CSV/JSON output column order.
COLUMNS = [
    "source_file",
    "crm_id",
    "counsellor_id",
    "rating",
    "email_date",
    "qualifications",
    "english_test",
    "study_destination",
    "course",
    "intake",
    "funding_method",
    "fund_amount",
    "assessment_notes",
    "data_quality_flag",
    "extraction_source",
    "pii_issues",
]


def now_tag() -> str:
    """Return a filesystem-safe timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
