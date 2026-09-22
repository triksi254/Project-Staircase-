"""scripts/funding_structural_test.py: capture only, structural claim check.

Pins the report's shape and internal consistency, and the one claim the script
exists to check: ``funding_clarity`` never appears in ``rubric_evidence()`` and
its rubric contribution is always 0.0, whatever a funding turn's wording.
"""
import importlib.util
from pathlib import Path

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


def test_capture_runs_one_session_and_funding_clarity_never_moves(tmp_path):
    m = _load()
    records = m.capture()
    assert [r["query"] for r in records] == REQUESTED
    assert len(records) == 6

    for r in records:
        # the field score_row() actually reads is never set by a live chat
        assert r["funding_clarity_present_in_evidence"] is False
        assert "funding_clarity" not in r["evidence"]
        assert r["funding_clarity_contribution"] == 0.0
        assert r["funding_clarity_weight"] == 0.20
        # funding_method_present is set on Fees & Funding / Scholarships turns
        # but the rubric never reads it (no such WEIGHTS key)
        assert r["funding_method_present"] in (0, 1)
        assert r["rule_score"] >= 0.0

    out = tmp_path / "funding_structural_test.txt"
    assert m.main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "funding_clarity ever present in rubric_evidence(): False" in text
    assert "funding_clarity contribution ever > 0.0: False" in text
    for q in REQUESTED:
        assert q in text
