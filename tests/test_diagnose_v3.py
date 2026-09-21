"""scripts/diagnose_v3.py: the three diagnostics behind the ``diagnose_v3`` tag.

Diagnosis only - it changes no behaviour. The pure helpers are tested directly;
one integration test runs the whole report against the real SBERT retriever.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "diagnose_v3.py"


def _load():
    spec = importlib.util.spec_from_file_location("diagnose_v3", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rule_lattice_has_six_distinct_values_and_a_ceiling_of_0_7778():
    lat = _load().rule_lattice()
    assert len(lat) == 8                                   # 2^3 topic subsets
    assert sorted({round(r["score"], 4) for r in lat}) == [
        0.0, 0.2222, 0.2778, 0.5, 0.5556, 0.7778]
    top = max(lat, key=lambda r: r["score"])
    assert top["raw"] == pytest.approx(0.14) and top["mass"] == pytest.approx(0.18)
    assert top["score"] == pytest.approx(0.14 / 0.18)
    assert top["score"] < 1.0


def test_two_different_topic_pairs_collide_at_exactly_0_5():
    """LEAD-0001 and LEAD-0002 both read 0.500 because two pairs give raw 0.09."""
    lat = {tuple(r["topics"]): r for r in _load().rule_lattice()}
    a = lat[("Application Process", "English Language")]
    b = lat[("Entry Requirements", "English Language")]
    assert a["raw"] == pytest.approx(0.09) == pytest.approx(b["raw"])
    assert a["score"] == pytest.approx(0.5) == pytest.approx(b["score"])


def test_the_ceiling_is_the_unparsed_english_band():
    """Chat never supplies an IELTS band, so English earns half its weight."""
    m = _load()
    assert m.english_credit(0.0) == pytest.approx(0.04)
    assert m.english_credit(6.5) == pytest.approx(0.08)


def test_report_runs_and_states_the_three_findings(tmp_path):
    out = tmp_path / "diagnose_v3.txt"
    assert _load().main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    # 1. escalation
    assert "scope_check_invoked=False" in text
    assert text.count("decision_reason=multi_intent_guard") == 2
    assert "groups=[g1:ielts, g2:kcse, g5:aston]" in text
    assert "groups=[g1:ielts, g3:masters, g5:rgu]" in text
    # 2. rule-score ceiling
    assert "raw=0.0900" in text and "0.7778" in text
    # 3. classifier: category source differs per query
    assert text.count("category source: retrieved FAQ") == 2
    assert text.count("category source: classifier fallback") == 1
