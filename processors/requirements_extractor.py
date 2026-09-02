"""Canonical Kenya entry-requirement extraction from scraped page text.

Single source of truth for turning page text into the standardized requirement
fields used by the Kenya postgraduate-requirements scrapers
(``scrape_all_universities.py`` and ``scrape_fix_remaining.py``), so the same
regex rules aren't copy-pasted across scripts and drift apart.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

#: Output field keys produced by :func:`extract_requirements` (stable order).
FIELDS = (
    "postgraduate_entry_requirement",
    "english_language_requirement",
    "bachelor_degree_requirement",
    "foundation_requirement",
    "scholarship_info",
    "contact_info",
)


def extract_requirements(text: str) -> Dict[str, Optional[str]]:
    """Extract key requirement fields from body text."""
    result = {field: None for field in FIELDS}

    lines = text.split("\n")
    full_text = " ".join(line.strip() for line in lines if line.strip())

    # Postgraduate entry requirements
    pg_patterns = [
        r"(?i)(?:postgraduate|post-graduate|masters|master\'s|msc|ma\b|mba|meng).{0,300}(?:bachelor|degree|honours|second class|upper|lower|2:?[12]|2\.\s*[12])",
        r"(?i)(?:bachelor|degree|honours).{0,200}(?:second class|upper|lower|2:?[12]|2\.\s*[12]).{0,200}(?:postgraduate|post-graduate|masters)",
    ]
    for pat in pg_patterns:
        m = re.search(pat, full_text)
        if m:
            result["postgraduate_entry_requirement"] = m.group(0)[:1000].strip()
            break

    # Bachelor degree requirement
    bach_pat = r"(?i)(?:bachelor|degree|honours).{0,300}(?:second class|upper|lower|2:?[12]|2\.\s*[12]|first class|gpa|grade|recognis)"
    m = re.search(bach_pat, full_text)
    if m:
        result["bachelor_degree_requirement"] = m.group(0)[:800].strip()

    # English language requirement
    eng_pat = r"(?i)(?:english|ielts|toefl|language).{0,200}(?:requirement|grade|score|level|band|min).{0,300}"
    m = re.search(eng_pat, full_text)
    if m:
        result["english_language_requirement"] = m.group(0)[:800].strip()

    # Foundation requirement
    found_pat = r"(?i)(?:foundation|foundation year|international foundation).{0,300}(?:year|study|kcse|grade|subject)"
    m = re.search(found_pat, full_text)
    if m:
        result["foundation_requirement"] = m.group(0)[:600].strip()

    # Scholarship info
    schol_pat = r"(?i)(?:scholarship|bursary|funding|award).{0,200}(?:international|available|offer|amount)"
    m = re.search(schol_pat, full_text)
    if m:
        result["scholarship_info"] = m.group(0)[:500].strip()

    # Contact info
    contact_pat = r"(?i)(?:contact|email|phone|regional manager|representative).{0,200}(?:@|\.com|\.ac|whatsapp|\+[0-9])"
    m = re.search(contact_pat, full_text)
    if m:
        result["contact_info"] = m.group(0)[:400].strip()

    return result