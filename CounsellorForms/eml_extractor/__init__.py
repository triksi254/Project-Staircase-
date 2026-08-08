"""EML assessment-form extraction package (stdlib-only).

Parses Jotform-generated .eml files located under ``CounsellorForms/Forms``,
extracts structured assessment data, redacts personally identifiable
information (PII), and exports cleaned records to CSV + JSON.

This module intentionally depends only on the Python standard library so it
can run on any Python 3 interpreter without installing third-party packages.
"""

from eml_extractor.pii import redact_pii, has_pii
from eml_extractor.parser import parse_eml
from eml_extractor.models import AssessmentRecord
from eml_extractor.categorizer import build_record
from eml_extractor.pipeline import (
    process_file,
    process_directory,
    run_pipeline,
    export_csv,
    export_json,
)

__all__ = [
    "redact_pii",
    "has_pii",
    "parse_eml",
    "AssessmentRecord",
    "build_record",
    "process_file",
    "process_directory",
    "run_pipeline",
    "export_csv",
    "export_json",
]
