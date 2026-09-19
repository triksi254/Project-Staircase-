"""Tests for chatbot.demo CLI + cv_report diagnostic."""
import json

from chatbot.classifier import cv_report, load_training_data
from chatbot.demo import format_human, run_cli, run_demo


def test_cv_report_five_folds_per_class_keys():
    texts, labels = load_training_data()
    out = cv_report(texts, labels, k=5)
    assert out["k"] == 5
    assert len(out["folds"]) == 5
    for fold in out["folds"]:
        assert "macro_f1" in fold
        assert "support" in fold and "per_class_f1" in fold
        assert len(fold["per_class_f1"]) == 9
        assert 0.0 <= fold["macro_f1"] <= 1.0


def test_demo_runs_on_known_query():
    out = run_demo("Do I need a visa?", rule=0.8)
    assert "Visa" in out["response"]["category"]
    assert isinstance(out["hybrid_score"], float)
    assert 0.0 <= out["hybrid_score"] <= 1.0


def test_demo_json_output_parseable():
    raw = run_cli(["Do I need a visa?", "--rule", "0.8", "--json"])
    # run_cli captures stdout; logging warnings go to stderr so this is pure JSON
    data = json.loads(raw)
    for key in ("query", "rule", "ml_proba", "alpha", "hybrid_score",
                "label", "escalate", "priority", "response", "candidates"):
        assert key in data, f"missing key {key}"


def test_demo_hot_lead_escalates():
    raw = run_cli(["Do I need a visa?", "--rule", "0.9",
                   "--ml-proba", "0.05,0.15,0.80", "--json"])
    data = json.loads(raw)
    assert data["label"] == "Hot"
    assert data["escalate"] is True
    assert data["priority"] == "high"


def test_demo_fallback_on_nonsense():
    raw = run_cli(["purple monkey dishwasher", "--rule", "0.5", "--json"])
    data = json.loads(raw)
    assert data["response"]["fallback"] is True


def test_displayed_threshold_matches_applied_threshold():
    """Regression: the printed threshold must be the one used for the decision.

    Previously format_human hardcoded 0.60 while respond() compared against
    EscalationPolicy.min_confidence (0.15), so a 0.37-confidence hit printed
    "(threshold 0.60)" alongside "Fallback: no".
    """
    from chatbot.responder import EscalationPolicy

    out = run_demo("Do I need a visa?", rule=0.8)
    assert out["min_confidence"] == EscalationPolicy().min_confidence
    human = format_human(out)
    assert f"(threshold {out['min_confidence']:.2f})" in human
    # The decision must be consistent with the displayed gate.
    below = out["response"]["confidence"] < out["min_confidence"]
    assert out["response"]["fallback"] is below


def test_min_confidence_gate_applies_on_answered_path():
    """Raising the gate above the top-1 score must force abstention."""
    out = run_demo("Do I need a visa?", rule=0.8, min_confidence=0.99)
    assert out["response"]["fallback"] is True
    assert out["escalate"] is True
    # A confident answer at the default gate is the contrasting case.
    base = run_demo("Do I need a visa?", rule=0.8)
    assert base["response"]["fallback"] is False


def test_min_confidence_flag_is_accepted_by_cli():
    raw = run_cli(["Do I need a visa?", "--rule", "0.8",
                   "--min-confidence", "0.99", "--json"])
    data = json.loads(raw)
    assert data["min_confidence"] == 0.99
    assert data["response"]["fallback"] is True
