"""Tests for chatbot.session_features + dashboard lead-identity helpers."""
from datetime import datetime, timedelta

from chatbot.session_features import (
    SessionTracker,
    category_entropy,
    empty_evidence,
    hybrid_for_session,
    rule_score_from_evidence,
)
from dashboard.app import (
    archive_entry,
    new_lead_id,
    should_show_new_badge,
)


def _turn(cat, text="hello world", conf=0.5):
    return text, {"category": cat, "confidence": conf}


def _tracker():
    return SessionTracker("LEAD-TEST-0001", datetime(2026, 9, 19))


def test_tracker_counts_turns():
    t = _tracker()
    base = datetime(2026, 9, 19, 12, 0, 0)
    for i, cat in enumerate(["Visa & Immigration", "Fees & Funding",
                             "Scholarships"]):
        msg, resp = _turn(cat, text=f"message number {i}")
        t.add_turn(msg, resp, base + timedelta(seconds=30 * i))
    feats = t.features()
    assert feats["message_count"] == 3
    assert len(t) == 3
    assert feats["returning_session"] == 0


def test_tracker_maps_visa_to_evidence():
    t = _tracker()
    msg, resp = _turn("Visa & Immigration", text="do I need a visa")
    t.add_turn(msg, resp, datetime(2026, 9, 19, 12, 0, 0))
    ev = t.rubric_evidence()
    assert ev["session_flags"]["visa_intent_mentioned"] is True
    assert ev["funding_method_present"] == 0


def test_tracker_unseen_is_not_positive():
    t = _tracker()
    ev = t.rubric_evidence()
    assert ev["has_english_test"] == 0
    assert ev["has_course"] == 0
    assert ev["has_intake"] == 0
    assert ev["funding_method_present"] == 0
    assert ev["session_flags"]["visa_intent_mentioned"] is False
    base = empty_evidence()
    for key in base:
        assert ev[key] == base[key]


def test_entropy_zero_on_single_category():
    assert category_entropy([]) == 0.0
    assert category_entropy(["Visa & Immigration"] * 4) == 0.0


def test_entropy_increases_with_mixed_categories():
    single = category_entropy(["Fees & Funding"] * 4)
    mixed = category_entropy(["Fees & Funding", "Visa & Immigration",
                              "Scholarships", "Accommodation"])
    assert mixed > single > 0.0 or (single == 0.0 and mixed > 0.0)


def test_rule_score_from_evidence_is_bounded():
    t = _tracker()
    assert 0.0 <= t.rule_score() <= 1.0
    msg, resp = _turn("Entry Requirements", text="what grades do I need")
    t.add_turn(msg, resp, datetime(2026, 9, 19, 12, 0, 0))
    assert 0.0 <= t.rule_score() <= 1.0
    assert 0.0 <= rule_score_from_evidence(empty_evidence()) <= 1.0
    assert 0.0 <= hybrid_for_session(0.2) <= 1.0


def test_new_lead_id_format_and_badge_window():
    first = new_lead_id(1, datetime(2026, 9, 19, 8, 0, 0))
    second = new_lead_id(2, datetime(2026, 9, 19, 8, 0, 0))
    assert first == "LEAD-20260919-0001"
    assert second == "LEAD-20260919-0002"
    assert should_show_new_badge(0) is True
    assert should_show_new_badge(2) is True
    assert should_show_new_badge(3) is False


def test_archive_entry_captures_final_score():
    t = _tracker()
    msg, resp = _turn("Visa & Immigration")
    t.add_turn(msg, resp, datetime(2026, 9, 19, 12, 0, 0))
    entry = archive_entry("LEAD-20260919-0001", t, 0.2, 0.2, "Cold")
    assert entry["lead_id"] == "LEAD-20260919-0001"
    assert entry["final_label"] == "Cold"
    assert entry["turns"] == 1
    assert 0.0 <= entry["final_hybrid_score"] <= 1.0
