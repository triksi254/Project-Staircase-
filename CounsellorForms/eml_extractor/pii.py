"""PII detection and redaction for assessment-form emails (stdlib-only).

Defines compiled regex and helper functions to redact personally
identifiable information (PII) found in Jotform assessment emails.
"""

from __future__ import annotations

import re
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Regex patterns
# --------------------------------------------------------------------------- #
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Passport / ID numbers: year-range format like 2024-2034.
PASSPORT_PATTERN = re.compile(r"\b\d{4}[\-–]\d{4}\b")
# IP addresses (IPv4)
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Phone numbers: optional country code (+..), optional separator/parens,
# then 9-12 digits (covers +254702102295, (+254) 0739232898, 0716524788).
PHONE_PATTERN = re.compile(
    r"(?<![\d-])(?:\+?\d{1,3}[\s\-.]?)?(?:\(\d{1,3}\)[\s\-.]?)?"
    r"(?:\d{3}[\s\-.]?\d{3}[\s\-.]?\d{3,4}|\d{10,12})(?![\d-])"
)
AMOUNT_PATTERN = re.compile(
    r"(?i)(?:[$£€]\s*[\d,]+(?:\.\d+)?(?:[km]|million|thousand)?"
    r"|\b(?:kes|ksh|usd|gbp|eur|aed)\s*[\d,]+(?:\.\d+)?"
    r"|\b[\d,]+\.?\d*\s*(?:million|m|k))\b"
)
ADDRESS_PATTERN = re.compile(
    r"(?i)\b(?:\d{1,5}[\s-])?(?:[a-z0-9.\-]+\s){1,4}?"
    r"(?:road|street|avenue|lane|boulevard|drive|close|park|estate|"
    r"floor|plaza|tower|box|chiromo|westlands|nairobi)\s*"
    r"(?:[a-z0-9.\-]+\s?){0,3}?"
)
SOCIAL_PATTERN = re.compile(
    r"(?i)\b(?:facebook\.com|twitter\.com|instagram\.com|linkedin\.com|youtube\.com|@)\S+"
)

NAME_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"(?i)\b(Teresia|Attina|Duncan|Ezra|Hansel|James|Mercy|Rosalinda|"
        r"Stephanie|Vanessa|Wanjiku|Veronicah)\b"
    )
]

# Ordered list of (pattern, marker) applied to free text. Passport/IP are
# applied before phone so passport ranges like "2024-2034" are not caught by
# the phone rule.
PII_RULES: List[re.Pattern] = [
    EMAIL_PATTERN,
    PASSPORT_PATTERN,
    IP_PATTERN,
    PHONE_PATTERN,
    AMOUNT_PATTERN,
    ADDRESS_PATTERN,
    SOCIAL_PATTERN,
]

_MARKER_BY_RULE = {
    EMAIL_PATTERN: "[EMAIL_REDACTED]",
    PASSPORT_PATTERN: "[PASSPORT_REDACTED]",
    IP_PATTERN: "[IP_REDACTED]",
    PHONE_PATTERN: "[PHONE_REDACTED]",
    AMOUNT_PATTERN: "[AMOUNT_REDACTED]",
    ADDRESS_PATTERN: "[ADDRESS_REDACTED]",
    SOCIAL_PATTERN: "[SOCIAL_REDACTED]",
}


def count_pii(text: str) -> Dict[str, int]:
    """Return a count of matches per PII category in *text*."""
    counts: Dict[str, int] = {}
    for rule in PII_RULES:
        counts[_MARKER_BY_RULE[rule]] = len(rule.findall(text))
    counts["[NAME_REDACTED]"] = sum(len(p.findall(text)) for p in NAME_PATTERNS)
    return counts


def redact_pii(text: object, replace_names: bool = True) -> str:
    """Redact PII in *text*, returning a cleaned string."""
    if text is None:
        return ""
    value = str(text).strip()
    if not value:
        return ""

    for _ in range(2):
        for rule in PII_RULES:
            value = rule.sub(_MARKER_BY_RULE[rule], value)
        if replace_names:
            for name_rule in NAME_PATTERNS:
                value = name_rule.sub("[NAME_REDACTED]", value)

    value = re.sub(r"\s+", " ", value).strip()
    return value


def has_pii(text: object) -> bool:
    """Return True if *text* still contains recognizable PII."""
    if text is None:
        return False
    value = str(text)
    for rule in PII_RULES:
        if rule.search(value):
            return True
    return any(name.search(value) for name in NAME_PATTERNS)
