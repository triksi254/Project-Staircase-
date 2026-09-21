"""Regression tests for the live scoring path (chat -> evidence -> ML -> hybrid).

Each test pins a defect found in the end-of-build review:

* categorical model inputs were all zero (one-hot names never matched);
* the visa flag never reached the model;
* entropy was in nats live but bits in the training generator;
* live rule scores topped out at 0.23, so "Hot" was unreachable;
* unobservable form fields were asserted as negatives, not left unknown;
* model selection / imputation / failures were silent.

The trained model is replaced by a tiny forest fitted through the *same*
``leads.train_ml`` preprocessing, so the tests run on a clean clone (the real
``*.pkl`` artifacts are gitignored). One extra test exercises the real
artifact when it is present.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from chatbot.session_features import SessionTracker, category_entropy
from leads import hybrid as H

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

CATS = ["Entry Requirements", "Application Process", "English Language",
        "Fees & Funding", "Visa & Immigration", "Scholarships",
        "Accommodation", "Program Details"]


def _session(cats, words=30, gap=10):
    t0 = datetime(2026, 9, 21, 12, 0, 0)
    t = SessionTracker("LEAD-TEST", t0)
    for i, c in enumerate(cats):
        t.add_turn("word " * words, {"category": c, "confidence": 0.9},
                   t0 + timedelta(seconds=gap * (i + 1)))
    return t


def _fit_toy_model(tmp_path, name="lead_model.pkl", extra=None):
    """Fit a small forest through train_ml's own preprocessing and save it."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    from leads import train_ml as T
    from leads.personas import generate_sessions

    sessions, _ = generate_sessions(n=400, seed=3)
    df = pd.DataFrame(sessions)
    df["source"] = "synthetic"
    combined = T.align_schema(df)
    X, y, feats = T.build_matrix(combined)
    model = RandomForestClassifier(n_estimators=15, random_state=0,
                                   class_weight="balanced").fit(X, y)
    d = tmp_path / "artifacts"
    d.mkdir(exist_ok=True)
    joblib.dump(model, d / name)
    (d / "metrics.json").write_text(
        json.dumps({"schema": T.schema_report(combined)}), encoding="utf-8")
    if extra is not None:
        (d / "config.json").write_text(json.dumps(extra), encoding="utf-8")
    return d, feats, model


@pytest.fixture()
def toy(tmp_path):
    d, feats, model = _fit_toy_model(tmp_path)
    cfg = H.load_artifacts(d)
    H.build_ml_cache(cfg)
    yield d, feats, cfg
    H._invalidate_ml_cache()


# --------------------------------------------------------------------------- #
# categorical alignment
# --------------------------------------------------------------------------- #
def test_categorical_inputs_are_one_hot_aligned_with_model(toy):
    """Every categorical feature must switch on exactly one model column."""
    _, feats, _ = toy
    t = _session(["Entry Requirements", "English Language"])
    x = H.preprocess_for_prediction(t.rubric_evidence(), engagement=t.features())
    for cat in H.CATEGORICAL:
        cols = [c for c in feats if c.startswith(cat + "_")]
        assert cols, "toy model has no columns for %s" % cat
        assert sum(x[c] for c in cols) == 1.0, (cat, {c: x[c] for c in cols})


def test_observed_chat_signals_reach_the_model_vector(toy):
    t = _session(["Entry Requirements", "Visa & Immigration"])
    x = H.preprocess_for_prediction(t.rubric_evidence(), engagement=t.features())
    assert x["has_course_1"] == 1.0 and x["has_course_0"] == 0.0
    assert x["has_intake_0"] == 1.0 and x["has_intake_1"] == 0.0
    assert x["message_count"] == 2.0


def test_visa_flag_reaches_the_model(toy):
    with_visa = _session(["Visa & Immigration", "Fees & Funding"])
    without = _session(["Fees & Funding", "Scholarships"])
    xv = H.preprocess_for_prediction(with_visa.rubric_evidence(),
                                     engagement=with_visa.features())
    xn = H.preprocess_for_prediction(without.rubric_evidence(),
                                     engagement=without.features())
    assert xv["visa_intent_mentioned"] == 1.0
    assert xn["visa_intent_mentioned"] == 0.0


def test_explicit_config_builds_the_cache(tmp_path):
    """predict_ml_proba(config=...) must not depend on a prior global load."""
    d, _, _ = _fit_toy_model(tmp_path)
    cfg = H.load_artifacts(d)          # invalidates the module cache
    t = _session(["Entry Requirements", "English Language"])
    proba = H.predict_ml_proba(t.rubric_evidence(), engagement=t.features(),
                               config=cfg)
    H._invalidate_ml_cache()
    assert proba is not None and len(proba) == 3
    assert abs(sum(proba) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# unobservable fields are unknown, not negative
# --------------------------------------------------------------------------- #
UNOBSERVABLE = ("passport_status", "destination_uk", "qual_level",
                "previous_application_mentioned", "funding_clarity",
                "study_gap_mentioned", "note_word_count")


def test_unobservable_fields_are_not_asserted_by_chat_evidence():
    ev = _session(["Entry Requirements", "English Language"]).rubric_evidence()
    for key in UNOBSERVABLE:
        assert key not in ev, "chat cannot observe %r; it must stay unknown" % key


def test_missing_fields_are_imputed_from_the_training_table(tmp_path):
    """Unknown fields take the training mode / median, and say so."""
    extra = {"alpha": 0.5, "imputation": {
        "medians": {"avg_delay_s": 52.6, "funding_clarity": 3.0,
                    "note_word_count": 0.0, "returning_session": 0.0},
        "modes": {"passport_status": "2", "qual_level": "0",
                  "destination_uk": "1", "study_gap_mentioned": "0",
                  "previous_application_mentioned": "0"}}}
    d, _, _ = _fit_toy_model(tmp_path, extra=extra)
    cfg = H.load_artifacts(d)
    H.build_ml_cache(cfg)
    try:
        one_turn = _session(["Entry Requirements"])
        x = H.preprocess_for_prediction(one_turn.rubric_evidence(),
                                        engagement=one_turn.features())
    finally:
        H._invalidate_ml_cache()
    assert x["avg_delay_s"] == 52.6            # undefined for one turn -> median
    assert x["passport_status_2"] == 1.0       # unobserved -> training mode
    assert x["destination_uk_1"] == 1.0


def test_missing_imputation_table_is_announced(toy, caplog):
    """No silent 0.0: a fallback imputation must be logged."""
    H._ml_cache["_warned_imputation"] = False
    H._ml_cache["median_imputations"] = {}
    t = _session(["Entry Requirements"])
    with caplog.at_level(logging.WARNING, logger="leads.hybrid"):
        H.preprocess_for_prediction(t.rubric_evidence(), engagement=t.features())
    assert "imputation" in caplog.text.lower()


# --------------------------------------------------------------------------- #
# units and reachable range
# --------------------------------------------------------------------------- #
def test_entropy_is_in_bits_like_the_training_generator():
    assert category_entropy(CATS[:4]) == pytest.approx(2.0)
    assert category_entropy(CATS) == pytest.approx(3.0, abs=1e-3)
    assert category_entropy([CATS[0]] * 5) == 0.0


def test_single_turn_delay_is_undefined_not_instant():
    t = _session(["English Language"])
    assert t.features()["avg_delay_s"] is None
    assert _session(["English Language", "Fees & Funding"]) \
        .features()["avg_delay_s"] == 10.0


def test_hot_is_reachable_rule_only():
    t = _session(["Entry Requirements", "Application Process",
                  "English Language"])
    assert t.rule_score() >= 2.0 / 3.0
    assert H.score_to_label(t.hybrid(None)) == "Hot"


def test_hot_is_reachable_with_a_strong_ml_score():
    t = _session(["Entry Requirements", "English Language"])
    assert H.score_to_label(t.hybrid(ml_proba=[0.0, 0.0, 1.0])) == "Hot"


def test_live_scores_span_all_three_labels():
    labels = {
        H.score_to_label(_session(["Fees & Funding"]).rule_score()),
        H.score_to_label(_session(["Entry Requirements",
                                   "English Language"]).rule_score()),
        H.score_to_label(_session(["Entry Requirements", "Application Process",
                                   "English Language"]).rule_score()),
    }
    assert labels == {"Cold", "Warm", "Hot"}


def test_topic_only_chat_does_not_earn_form_credit():
    """Fees/visa/scholarship questions say nothing the rubric can score."""
    assert _session(["Fees & Funding", "Visa & Immigration",
                     "Scholarships", "Accommodation"]).rule_score() == 0.0


def test_visitor_word_count_is_not_note_completeness():
    """Chatty visitors must not earn the counsellor-notes credit."""
    short = _session(["Entry Requirements"], words=2)
    chatty = _session(["Entry Requirements"], words=200)
    assert short.rule_score() == chatty.rule_score()


def test_empty_session_scores_zero_regardless_of_ml():
    t = SessionTracker("LEAD-EMPTY", datetime(2026, 9, 21))
    assert t.rule_score() == 0.0
    assert t.hybrid(ml_proba=[0.0, 0.0, 1.0]) == 0.0


# --------------------------------------------------------------------------- #
# model selection and failures are explicit
# --------------------------------------------------------------------------- #
def test_fallback_model_choice_is_announced(tmp_path, caplog):
    d, _, _ = _fit_toy_model(tmp_path, name="lead_model_rf_full.pkl")
    with caplog.at_level(logging.WARNING, logger="leads.hybrid"):
        out = H.load_artifacts(d)
    H._invalidate_ml_cache()
    assert out["files"]["model"].endswith("lead_model_rf_full.pkl")
    assert "fallback" in caplog.text.lower()
    assert "rf_full" in caplog.text


def test_missing_config_does_not_claim_rule_only_when_model_loaded(tmp_path,
                                                                    caplog):
    d, _, _ = _fit_toy_model(tmp_path)
    with caplog.at_level(logging.WARNING, logger="leads.hybrid"):
        out = H.load_artifacts(d)
    H._invalidate_ml_cache()
    assert "model" in out
    assert "rule-only" not in caplog.text


def test_config_json_selects_the_model_and_alpha(tmp_path):
    d, _, _ = _fit_toy_model(tmp_path, name="lead_model_custom.pkl",
                             extra={"model_file": "lead_model_custom.pkl",
                                    "alpha": 0.3})
    cfg = H.load_artifacts(d)
    H._invalidate_ml_cache()
    assert cfg["files"]["model"].endswith("lead_model_custom.pkl")
    assert H.get_default_alpha(cfg) == 0.3


def test_default_alpha_without_config_is_the_documented_constant(tmp_path):
    cfg = H.load_artifacts(tmp_path / "nope")
    assert H.get_default_alpha(cfg) == H.DEFAULT_ALPHA == 0.5


def test_prediction_failure_is_logged_not_swallowed(toy, caplog):
    _, _, cfg = toy

    class _Boom:
        feature_names_in_ = cfg["model"].feature_names_in_

        def predict_proba(self, x):
            raise RuntimeError("shape mismatch")

    broken = dict(cfg, model=_Boom())
    t = _session(["Entry Requirements"])
    with caplog.at_level(logging.WARNING, logger="leads.hybrid"):
        out = H.predict_ml_proba(t.rubric_evidence(), engagement=t.features(),
                                 config=broken)
    assert out is None
    assert "shape mismatch" in caplog.text


# --------------------------------------------------------------------------- #
# the shipped artifact (skipped on a clean clone: *.pkl is gitignored)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (ARTIFACTS / "lead_model_rf_full.pkl").exists(),
                    reason="trained model binary is not tracked")
def test_shipped_model_receives_populated_categoricals():
    cfg = H.load_artifacts()
    H.build_ml_cache(cfg)
    try:
        names = H._ml_cache["feature_names"]
        t = _session(CATS[:3])
        x = H.preprocess_for_prediction(t.rubric_evidence(),
                                        engagement=t.features())
        cat_cols = [n for n in names
                    if any(n.startswith(c + "_") for c in H.CATEGORICAL)]
        assert len(cat_cols) == 19
        assert sum(1 for n in cat_cols if x[n] != 0.0) == len(H.CATEGORICAL)
        proba = H.predict_ml_proba(t.rubric_evidence(),
                                   engagement=t.features(), config=cfg)
        assert proba is not None and abs(sum(proba) - 1.0) < 1e-6
    finally:
        H._invalidate_ml_cache()


def test_shipped_config_json_documents_its_provenance():
    fp = ARTIFACTS / "config.json"
    assert fp.is_file(), "artifacts/config.json must ship with the repo"
    cfg = json.loads(fp.read_text(encoding="utf-8"))
    assert cfg["alpha"] == H.DEFAULT_ALPHA
    assert "uncalibrated" in cfg["alpha_source"].lower()
    imp = cfg["imputation"]
    assert set(H.CATEGORICAL) - {"has_course", "has_intake"} <= set(imp["modes"])
    assert "avg_delay_s" in imp["medians"]
    assert cfg["provenance"]["matches_metrics_schema"] is True


# --------------------------------------------------------------------------- #
# the rule score must be reproducible from what the panel shows
# --------------------------------------------------------------------------- #
def test_rule_breakdown_reconciles_with_the_score():
    """Panel showed +0.050 +0.040 +0.000 (=0.090) beside a rule score of 0.500:
    the score is that sum divided by the observable weight (0.18)."""
    from chatbot.session_features import LIVE_OBSERVABLE, live_rule_breakdown
    t = _session(["Application Process", "English Language", "English Language"])
    bd = live_rule_breakdown(t.rubric_evidence())
    assert bd["raw"] == pytest.approx(0.09)
    assert bd["mass"] == pytest.approx(0.18)
    assert bd["score"] == pytest.approx(0.5) == pytest.approx(t.rule_score())
    assert {l["feature"] for l in bd["lines"]} == set(LIVE_OBSERVABLE)
    assert sum(l["contribution"] for l in bd["lines"]) == pytest.approx(bd["raw"])
    assert sum(l["weight"] for l in bd["lines"]) == pytest.approx(bd["mass"])


def test_rule_score_is_defined_by_the_breakdown():
    """One function feeds both the score and the panel, so they cannot diverge."""
    from chatbot.session_features import live_rule_breakdown, rule_score_from_evidence
    for cats in (["Entry Requirements"], ["English Language", "Application Process"],
                 ["Fees & Funding"], list(CATS)):
        ev = _session(cats).rubric_evidence()
        assert rule_score_from_evidence(ev) == live_rule_breakdown(ev)["score"]


def test_rule_breakdown_logs_the_raw_score_before_normalisation(caplog):
    """The raw sum (0.0900) and the divisor (0.1800) are visible at DEBUG; the
    rule score is raw / mass with no clamp anywhere on the path."""
    t = _session(["Application Process", "English Language"])
    with caplog.at_level(logging.DEBUG, logger="chatbot.session_features"):
        t.rule_score()
    msgs = [r.getMessage() for r in caplog.records
            if r.name == "chatbot.session_features"]
    assert msgs, "live_rule_breakdown must log at DEBUG"
    msg = msgs[-1]
    assert "raw=0.0900" in msg and "mass=0.1800" in msg and "score=0.5000" in msg
    assert "has_intake=+0.050/0.050" in msg and "english_test=+0.040/0.080" in msg
