"""Tests for leads.hybrid: convex combination, proba mapping, calibration."""
from leads.hybrid import (
    calibrate_alpha,
    calibrate_alpha_kfold,
    hybrid_score,
    load_artifacts,
    ml_proba_to_score,
    score_to_label,
)


def test_hybrid_midpoint():
    assert hybrid_score(0.8, 0.6, alpha=0.5) == 0.7


def test_hybrid_alpha_edges():
    assert hybrid_score(0.2, 0.9, alpha=1.0) == 0.2
    assert hybrid_score(0.2, 0.9, alpha=0.0) == 0.9


def test_hybrid_none_ml_returns_rule():
    assert hybrid_score(0.42, None, alpha=0.3) == 0.42


def test_proba_mapping_hot():
    assert ml_proba_to_score([0.0, 0.0, 1.0]) == 1.0
    assert ml_proba_to_score([1.0, 0.0, 0.0]) == 0.0
    assert ml_proba_to_score([0.0, 1.0, 0.0]) == 0.5
    assert ml_proba_to_score({"Cold": 0.2, "Warm": 0.3, "Hot": 0.5}) == 0.65


def test_score_to_label_thresholds():
    assert score_to_label(0.1) == "Cold"
    assert score_to_label(0.5) == "Warm"
    assert score_to_label(0.9) == "Hot"


def test_calibrate_prefers_rule_when_rule_is_perfect():
    y = [0, 2, 1, 2, 0]
    rule = [0.0, 1.0, 0.5, 1.0, 0.0]
    ml = [0.5, 0.5, 0.5, 0.5, 0.5]
    out = calibrate_alpha(y, rule, ml)
    assert out["best_alpha"] == 1.0
    assert out["best_macro_f1"] == 1.0
    assert len(out["results"]) == 11
    for row in out["results"]:
        assert 0.0 <= row["macro_f1"] <= 1.0
        assert 0.0 <= row["accuracy"] <= 1.0


def test_calibration_sanity_macro_f1_grid():
    import random
    rng = random.Random(7)
    y = [rng.choice([0, 1, 1, 1, 2]) for _ in range(60)]
    rule = [rng.random() for _ in range(60)]
    ml = [rng.random() for _ in range(60)]
    out = calibrate_alpha(y, rule, ml)
    assert len(out["results"]) == 11
    assert [r["alpha"] for r in out["results"]] != []
    for row in out["results"]:
        assert 0.0 <= row["macro_f1"] <= 1.0
    best = max(out["results"], key=lambda r: (r["macro_f1"], -r["mae"]))
    assert out["best_alpha"] == best["alpha"]


def test_calibrate_kfold_train_only():
    import random
    rng = random.Random(42)
    n = 60
    X = [[rng.random(), rng.random(), rng.random()] for _ in range(n)]
    y = [rng.choice([0, 1, 2]) for _ in range(n)]
    rule = [rng.random() for _ in range(n)]
    out = calibrate_alpha_kfold(X, rule, y, n_splits=3, n_estimators=10)
    assert out["n_train"] == n
    assert len(out["oof_ml_scores"]) == n
    assert len(out["results"]) == 11
    assert 0.0 <= out["best_alpha"] <= 1.0


def test_load_artifacts_warns_on_missing(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="leads.hybrid"):
        out = load_artifacts(tmp_path / "nope")
    assert out["files"] == {}
    assert "lead_model.pkl" in caplog.text
    assert "metrics.json" in caplog.text


def test_load_artifacts_missing_dir(tmp_path):
    out = load_artifacts(tmp_path / "nope")
    assert out["files"] == {}
