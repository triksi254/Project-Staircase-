"""Tests for per-lead chat transcript downloads (dissertation evidence)."""
import json
from datetime import datetime

from dashboard.app import (
    archive_entry,
    build_live_entry,
    transcript_rows,
    transcript_to_csv,
    transcript_to_json,
    transcript_to_markdown,
)
from chatbot.session_features import SessionTracker


def _tracker():
    return SessionTracker("LEAD-TEST-0001", datetime(2026, 9, 19))


def _entry_with_transcript():
    t = _tracker()
    t.add_turn("Do I need IELTS?",
               {"category": "English Language", "confidence": 0.8},
               datetime(2026, 9, 19, 12, 0, 0))
    chat = [
        {"role": "user", "text": "Do I need IELTS?"},
        {"role": "assistant", "text": "Yes — IELTS 6.0 overall."},
    ]
    turns_log = [{
        "user_msg": "Do I need IELTS?",
        "category": "English Language",
        "confidence": 0.8,
        "assistant_text": "Yes — IELTS 6.0 overall.",
        "ts": "2026-09-19T12:00:00+00:00",
    }]
    return archive_entry("LEAD-20260919-0001", t, 0.4, 0.5, "Warm",
                         chat=chat, turns_log=turns_log,
                         started_at="2026-09-19T11:59:00+00:00")


def test_archive_entry_stores_transcript():
    entry = _entry_with_transcript()
    assert entry["chat"][0]["text"] == "Do I need IELTS?"
    assert entry["turns_log"][0]["assistant_text"].startswith("Yes")
    assert entry["started_at"] == "2026-09-19T11:59:00+00:00"
    assert entry["turns"] == 1


def test_archive_entry_defaults_keep_old_callers_working():
    t = _tracker()
    entry = archive_entry("LEAD-20260919-0009", t, 0.2, 0.2, "Cold")
    assert entry["chat"] == []
    assert entry["turns_log"] == []
    assert transcript_rows(entry) == []


def test_transcript_rows_pairs_user_and_assistant():
    rows = transcript_rows(_entry_with_transcript())
    assert len(rows) == 1
    assert rows[0]["user_message"] == "Do I need IELTS?"
    assert rows[0]["assistant_response"] == "Yes — IELTS 6.0 overall."
    assert rows[0]["category"] == "English Language"


def test_transcript_exports_json_markdown_csv():
    entry = _entry_with_transcript()
    payload = json.loads(transcript_to_json(entry))
    assert payload["lead_id"] == "LEAD-20260919-0001"
    assert payload["turns_log"][0]["user_msg"] == "Do I need IELTS?"
    md = transcript_to_markdown(entry)
    assert "# Chat transcript — LEAD-20260919-0001" in md
    assert "**User:** Do I need IELTS?" in md
    assert "IELTS 6.0" in md
    csv_text = transcript_to_csv(entry)
    assert csv_text.splitlines()[0].startswith("turn_no,timestamp,")
    assert "Do I need IELTS?" in csv_text


def test_build_live_entry_uses_exported_at_for_active_lead():
    class _State(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    class _St:
        session_state = _State({
            "lead_id": "LEAD-20260919-0002",
            "chat": [{"role": "user", "text": "hi"}],
            "turns_log": [{
                "user_msg": "hi", "category": "General Enquiries",
                "confidence": 0.5,
                "assistant_text": "hello",
                "ts": "2026-09-19T12:00:00+00:00"}],
            "tracker_state": {
                "lead_id": "LEAD-20260919-0002",
                "started_at": "2026-09-19T12:00:00+00:00"},
        })

    entry = build_live_entry(_St())
    assert entry["lead_id"] == "LEAD-20260919-0002"
    assert "exported_at" in entry and "closed_at" not in entry
    assert transcript_rows(entry)[0]["assistant_response"] == "hello"
