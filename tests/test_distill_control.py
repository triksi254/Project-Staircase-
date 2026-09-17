"""Tests for the self-distillation control in leads.eval_augmentation.

The control (``+distill``) must reproduce the +v2 training signal without any
synthetic rows: RF1 soft labels on the same train rows. These tests keep the
control honest -- real rows only, untouched holdout, valid metrics.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leads.eval_augmentation import (  # noqa: E402
    DISTILL_VARIANT,
    LABELS,
    SPLIT_SEED,
    _mechanism,
    bootstrap_cis,
    build_distill_frame,
    fit_label_model,
    run_distill_experiment,
    split_labeled_rows,
)
from leads.train_ml import SOURCE  # noqa: E402


PROFILE_BY_LABEL = {
    0: {"funding_clarity": 0, "destination_uk": 0, "has_course": 0,
        "has_intake": 0, "has_english_test": 0, "qual_level": 0,
        "passport_status": 1},
    1: {"funding_clarity": 2, "destination_uk": 1, "has_course": 1,
        "has_intake": 1, "has_english_test": 1, "qual_level": 3,
        "passport_status": 2},
    2: {"funding_clarity": 3, "destination_uk": 1, "has_course": 1,
        "has_intake": 1, "has_english_test": 1, "qual_level": 5,
        "passport_status": 2},
}


def _row(label, idx=0, **over):
    row = {
        "passport_status": 2, "qual_level": 4, "destination_uk": 1,
        "has_course": 1, "has_intake": 1, "study_gap_mentioned": 0,
        "previous_application_mentioned": 0, "funding_clarity": 3,
        "funding_method_present": 1, "has_english_test": 1,
        "english_band": 6.5, "note_word_count": 10 + idx,
        "label": label, SOURCE: "real",
    }
    row.update(PROFILE_BY_LABEL.get(label, {}))
    row.update(over)
    return row


def _real_frame(n_per_class=25):
    return pd.DataFrame(
        [_row(lbl, idx=i) for lbl in (0, 1, 2) for i in range(n_per_class)]
    )


def _split_and_model(n_per_class=25):
    frame = _real_frame(n_per_class=n_per_class)
    split = split_labeled_rows(frame, random_state=SPLIT_SEED)
    model = fit_label_model(split, bootstrap_pool="train", seed=SPLIT_SEED)
    return split, model


# --------------------------------------------------------------------------- #
# the control itself
# --------------------------------------------------------------------------- #
def test_distill_variant_runs():
    """+distill produces valid metrics and trains on no synthetic rows."""
    split, model = _split_and_model()
    out = run_distill_experiment(
        split["train"], split["holdout"], model,
        model_kind="rf", feature_set="full", split=split)

    # Required keys and a valid metrics dict.
    for key in ("headline_macro_f1", "generalization", "n_train_hard",
                "n_train_soft_probes", "n_train_total", "n_synthetic_rows"):
        assert key in out, "missing key: %s" % key
    gen = out["generalization"]
    for side in ("real_only", "real_plus_synthetic"):
        m = gen[side]
        assert {"macro_f1", "per_class", "y_true", "y_pred"} <= set(m)
        assert set(m["per_class"]) == set(LABELS)
        assert 0.0 <= m["macro_f1"] <= 1.0
        n = len(m["y_true"])
        assert n == len(m["y_pred"]) == split["n_holdout"]
        assert set(m["y_pred"]) <= {0, 1, 2}
    assert set(gen["delta_per_class_f1"]) == set(LABELS)
    assert "delta_macro_f1" in gen
    assert gen["n_real_test"] == split["n_holdout"]
    assert list(gen["real_holdout_index"]) == split["holdout_idx"]

    # The training set contains NO synthetic rows: the hard half and the
    # soft half are both copies of the same train rows.
    assert out["n_synthetic_rows"] == 0
    assert out["n_train_hard"] == split["n_train"]
    assert out["n_train_soft_probes"] == split["n_train"]
    # 688 hard examples + 688 soft probes (weight 1.0 each).
    assert out["n_train_total"] == 2 * split["n_train"]
    # Soft probes are expanded one row per class inside RF2's fit matrix.
    assert out["n_fit_rows"] == 4 * split["n_train"]
    probe = build_distill_frame(split["train"], model)
    assert set(probe[SOURCE].unique()) == {"real"}
    assert len(probe) == split["n_train"]
    # The probes must be exactly the train rows (never holdout rows).
    # Value-tuples repeat across the split by construction, so identify rows
    # by the global ``row_id`` carried through the split.
    assert "row_id" in probe.columns
    probe_ids = set(probe["row_id"].astype(int))
    assert probe_ids == set(split["train_row_ids"])
    assert probe_ids.isdisjoint(set(split["holdout_row_ids"]))


def test_distill_soft_labels_come_from_rf1():
    """The soft probes carry RF1's ``predict_proba`` on the train rows."""
    import numpy as np

    split, model = _split_and_model()
    from leads.train_ml import align_schema, build_matrix

    X_tr, _, _ = build_matrix(align_schema(split["train"]), False, False)
    ask = X_tr.reindex(columns=list(model.feature_names), fill_value=0.0)
    proba = model.model.predict_proba(
        pd.DataFrame(ask, columns=list(model.feature_names)))
    assert proba.shape == (split["n_train"], len(model.classes))
    # Every probe row's soft distribution matches RF1's own predict_proba
    # for that row (the cached values the v2 generator also uses).
    cached = np.asarray(
        [model.label_proba(i) for i in range(split["n_train"])])
    np.testing.assert_allclose(proba, cached, atol=1e-12)


# --------------------------------------------------------------------------- #
# mechanism arithmetic + bootstrap CIs
# --------------------------------------------------------------------------- #
def _fake_results(d_v2, d_distill):
    def _gen(delta):
        return {"generalization": {"delta_macro_f1": delta}} if delta \
            is not None else {"generalization": None}

    return {"full": {"generators": {
        "v2": _gen(d_v2),
        DISTILL_VARIANT: _gen(d_distill),
    }}}


def test_mechanism_classifies_self_distillation():
    mech = _mechanism({"results": _fake_results(0.1188, 0.1100)},
                      feature_set="full")
    assert mech["confirmed"] is True  # |0.1188 - 0.1100| < 0.02
    assert mech["tag"] == "SELF-DISTILLATION CONFIRMED"
    assert mech["increment"] == pytest.approx(0.0088, abs=1e-9)

    mech2 = _mechanism({"results": _fake_results(0.1188, 0.0400)},
                       feature_set="full")
    assert mech2["confirmed"] is False  # |0.1188 - 0.0400| >= 0.02
    assert mech2["tag"] == "AUGMENTATION EFFECT SURVIVES"

    missing = _mechanism({"results": _fake_results(0.1188, None)},
                         feature_set="full")
    assert missing["confirmed"] is False  # no control -> never confirmed


def test_bootstrap_cis_paired_and_flagged():
    """Delta CIs pair the resamples and flag CI-crossing-zero as not sig."""
    y_true = [lbl for lbl in (0, 1, 2) for _ in range(20)]
    base_pred = [0, 1, 1] * 20  # real-only misses some Hot

    def _gen(y_pred):
        return {"generalization": {"real_plus_synthetic": {
            "y_true": list(y_true), "y_pred": list(y_pred)}}}

    payload = {
        "results": {"full": {
            "real_only": {"y_true": y_true, "y_pred": list(base_pred)},
            "generators": {
                "v1": _gen([0] * len(y_true)),
                "v2": _gen(y_true),
                DISTILL_VARIANT: _gen(base_pred),
            },
        }},
    }
    cis = bootstrap_cis(payload, n_resamples=50, seed=SPLIT_SEED)
    fs = cis["variants"]["full"]
    assert fs["n_holdout"] == len(y_true)
    for name in ("real-only", "+v1", "+v2", "+distill"):
        v = fs["variants"][name]
        assert {"point", "mean", "lo", "hi"} <= set(v["macro_f1"])
        assert v["macro_f1"]["lo"] <= v["macro_f1"]["mean"] <= \
            v["macro_f1"]["hi"]
    # +v2 predicts perfectly -> strictly positive delta CI, significant.
    dv2 = fs["deltas"]["+v2"]
    assert dv2["macro"]["lo"] > 0
    assert dv2["significant_macro"] is True
    # +distill == real-only here -> delta 0, CI crosses zero, not significant.
    ddt = fs["deltas"]["+distill"]
    assert ddt["significant_macro"] is False
    # +v1 (all-Cold predictions) is strictly worse in point estimate; its
    # significance flag must exactly reflect whether its CI crosses zero.
    dv1 = fs["deltas"]["+v1"]
    assert dv1["macro"]["point"] < dv2["macro"]["point"]
    assert dv1["significant_macro"] == (
        dv1["macro"]["lo"] > 0 or dv1["macro"]["hi"] < 0)
    # Hot F1 CIs exist for every variant.
    for name in ("real-only", "+v1", "+v2", "+distill"):
        assert {"point", "mean", "lo", "hi"} <= set(
            fs["variants"][name]["hot_f1"])
