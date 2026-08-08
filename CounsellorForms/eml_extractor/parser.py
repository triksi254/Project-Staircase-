"""Parse Jotform assessment-form .eml files (stdlib-only).

Uses the standard-library ``email`` package to parse the MIME structure and
extracts question/answer pairs. Prefers the HTML ``emailFieldsTable`` (the
authoritative form representation) and falls back to the plain-text body when
the HTML table is absent (e.g. some UK Office / Pre-Application variants).
"""

from __future__ import annotations

import email
import email.policy
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Question cells end with a question mark or a colon (e.g. "...?")
_QUESTION_HINT = re.compile(r"\?\s*$|:\s*$")


class _TableParser(HTMLParser):
    """Collect question/value pairs from the ``emailFieldsTable`` HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.field_rows: List[Tuple[str, str]] = []
        self._in_fields_table: bool = False
        self._in_cell: Optional[str] = None  # "question" | "value" | None
        self._buf: List[str] = []
        self._pending_question: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        attrs_lower = {k.lower(): v for k, v in attrs}

        if tag == "table":
            cell_id = attrs_lower.get("id", "")
            if "emailFieldsTable" in cell_id:
                self._in_fields_table = True

        if not self._in_fields_table:
            return

        if tag == "td":
            cell_id = attrs_lower.get("id", "")
            if "question" in cell_id:
                self._in_cell = "question"
                self._buf = []
            elif "value" in cell_id:
                self._in_cell = "value"
                self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_fields_table and self._in_cell:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._in_fields_table:
            return
        if tag == "td" and self._in_cell:
            text = self._flush_buf().strip()
            if self._in_cell == "question":
                self._pending_question = text
            elif self._in_cell == "value":
                if self._pending_question is not None:
                    self.field_rows.append((self._pending_question, text))
                self._pending_question = None
            self._in_cell = None
        if tag == "table":
            self._in_fields_table = False

    def _flush_buf(self) -> str:
        text = " ".join(self._buf)
        self._buf = []
        return text


def _extract_html_fields(html: Optional[str]) -> List[Tuple[str, str]]:
    """Parse question/value pairs out of the HTML body."""
    if not html:
        return []
    parser = _TableParser()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - tolerate malformed HTML
        return []
    return parser.field_rows


def _strip_html(html: str) -> str:
    """Remove HTML tags, returning plain text (for non-table sections)."""
    import html as html_module

    text = re.sub(r"<[^>]+>", " ", html)
    return html_module.unescape(re.sub(r"\s+", " ", text)).strip()


def _extract_plain_fields(body: str) -> List[Tuple[str, str]]:
    """Parse question/answer pairs from the plain-text Jotform body.

    The plain text body uses a ``Question ... Answer`` layout on consecutive
    lines. We pair a question line with the following non-empty line(s).
    """
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Jotform Logo" in line or "edit this submission" in line.lower():
            i += 1
            continue
        if _is_question(line):
            # Gather following lines until the next question-like line.
            answer_parts: List[str] = []
            j = i + 1
            while j < len(lines) and not _is_question(lines[j]):
                answer_parts.append(lines[j])
                j += 1
            pairs.append((line, " ".join(answer_parts)))
            i = j
        else:
            i += 1
    return pairs


def _is_question(line: str) -> bool:
    """Heuristic: a question line ends with '?' or ':' or is a known prompt."""
    if _QUESTION_HINT.search(line):
        return True
    known = (
        "qualifications",
        "english proficiency",
        "study destination",
        "course",
        "intake",
        "funding",
        "crm id",
        "enquiry stage",
        "username",
        "email address",
    )
    lower = line.lower()
    return any(k in lower for k in known)


def parse_eml(path: Path) -> Dict[str, object]:
    """Parse a single .eml file and return extracted data.

    Returns a dict with:
        - ``headers``: raw email headers (From, To, Subject, Date)
        - ``fields``: list of (question, answer) pairs
        - ``source``: "html" or "plain"
        - ``subject``: email subject
    """
    raw = path.read_bytes()
    msg = email.message_from_bytes(raw, policy=email.policy.default)

    subject = str(msg.get("Subject", "")).strip()
    from_addr = str(msg.get("From", "")).strip()
    to_addrs = str(msg.get("To", "")).strip()
    date = str(msg.get("Date", "")).strip()

    html_body: Optional[str] = None
    plain_body: Optional[str] = None

    for part in msg.walk():
        ctype = part.get_content_type()
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        try:
            text = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            text = payload.decode("utf-8", errors="ignore")

        if ctype == "text/html":
            html_body = (html_body or "") + text
        elif ctype == "text/plain":
            plain_body = (plain_body or "") + text

    # Prefer HTML table extraction.
    fields = _extract_html_fields(html_body)
    source = "html"
    if not fields and plain_body:
        fields = _extract_plain_fields(plain_body)
        source = "plain"

    return {
        "headers": {
            "from": from_addr,
            "to": to_addrs,
            "subject": subject,
            "date": date,
        },
        "fields": fields,
        "source": source,
        "subject": subject,
        "html_body": html_body,
        "plain_body": plain_body,
    }
