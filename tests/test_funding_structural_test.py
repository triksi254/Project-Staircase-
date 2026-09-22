"""scripts/funding_structural_test.py: capture only, structural claim check.

Pins the report's shape and internal consistency. The script's own claim (that
``funding_clarity`` can never appear, whatever a funding turn's wording) held
at the time it was tagged ``funding_structural_diagnostic`` -- see the frozen
``artifacts/funding_structural_test.txt``, left untouched. ``funding_amount_extractor``
then gave ``rubric_evidence()`` a real, narrower way to observe it (an explicit
amount *and* a funding-method phrase in the same turn), so a live re-run of
this same script now finds it on 2 of the 6 turns
("I can pay £15,000..." and "My parents have £30,000 saved..."); the other
four still never set it (no qualifying amount+method phrasing), consistent
with the extractor being conservative by design.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "funding_structural_test.py"

REQUESTED = [
    "I can pay £15,000 per year towards tuition",
    "My parents have £30,000 saved for my education",
    "I'll need a £10,000 scholarship to make it work",
    "Can I pay fees in three instalments over the year?",
    "How much is tuition at Aston for international students?",
    "What is the total cost of a one-year masters at BCU?",
]


def _load():
    spec = importlib.util.spec_from_file_location("funding_structural_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_six_queries_are_the_requested_ones_in_order():
    assert list(_load().QUERIES) == REQUESTED


def test_capture_runs_one_session_and_two_turns_now_set_funding_clarity(tmp_path):
    m = _load()
    records = m.capture()
    assert [r["query"] for r in records] == REQUESTED
    assert len(records) == 6

    # turns 1-2 name an amount and a funding-method phrase; the extractor sets
    # a tier for each (turn 2's £30,000 overwrites turn 1's £15,000 tier).
    # Every later evidence dict now always carries the key, at whatever value
    # the last qualifying turn set (score_row() reads it in the same way).
    assert records[0]["evidence"]["funding_clarity"] == 2
    assert records[1]["evidence"]["funding_clarity"] == 3
    for r in records:
        assert r["funding_clarity_present_in_evidence"] is True
        assert r["funding_method_present"] in (0, 1)
        assert r["rule_score"] >= 0.0
    for r in records[1:]:
        # turns 3-6 name no qualifying amount+method pair; the tier holds.
        assert r["evidence"]["funding_clarity"] == 3
        assert r["funding_clarity_contribution"] == pytest.approx(0.20)

    out = tmp_path / "funding_structural_test.txt"
    assert m.main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "funding_clarity ever present in rubric_evidence(): True" in text
    assert "funding_clarity contribution ever > 0.0: True" in text
    for q in REQUESTED:
        assert q in text
