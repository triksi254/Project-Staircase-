"""Tests for the lead-scoring model training pipeline (leads.train_ml).

Frames are built in-memory because ``data/processed/`` is gitignored and not
available in a fresh clone.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leads.train_ml import (  # noqa: E402
    CATEGORICAL,
    ENGAGEMENT,
    LABELS,
    NUMERIC,
    SOURCE,
    TARGET,
    _ci95,
    align_schema,
    build_matrix,
    evaluate,
    feature_importances,
    fit_and_eval,
    make_model,
    schema_report,
)


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


def _row(label, source="real", idx=0, **over):
    row = {
        "passport_status": 2, "qual_level": 4, "destination_uk": 1,
        "has_course": 1, "has_intake": 1, "study_gap_mentioned": 0,
        "previous_application_mentioned": 0, "funding_clarity": 3,
        "funding_method_present": 1, "has_english_test": 1,
        "note_word_count": 10 + idx,
        TARGET: label, SOURCE: source,
    }
    row.update(PROFILE_BY_LABEL.get(label, {}))
    row.update(over)
    return row


def _frame(n_per_class=8, source="real"):
    return pd.DataFrame(
        [_row(lbl, source, idx=i) for lbl in (0, 1, 2)
         for i in range(n_per_class)]
    )


def test_align_schema_drops_unrated_label():
    real = pd.DataFrame([_row(0), _row(1), _row(2), _row(-1)])
    aligned = align_schema(real)
    assert len(aligned) == 3
    assert set(aligned[TARGET].unique()) <= {0, 1, 2}


def test_align_schema_adds_missing_columns():
    real = pd.DataFrame([_row(1)])
    real = real.drop(columns=["funding_clarity", "has_english_test"])
    aligned = align_schema(real)
    assert "funding_clarity" in aligned.columns
    assert "has_english_test" in aligned.columns
    assert pd.isna(aligned["funding_clarity"].iloc[0])


def test_align_schema_maps_string_labels():
    real = pd.DataFrame([
        {"passport_status": 2, TARGET: "Cold", SOURCE: "real"},
        {"passport_status": 2, TARGET: "Good", SOURCE: "real"},
        {"passport_status": 2, TARGET: "Excellent", SOURCE: "real"},
    ])
    aligned = align_schema(real)
    assert sorted(aligned[TARGET].tolist()) == [0, 1, 2]


def test_align_schema_ignores_empty_frame():
    aligned = align_schema(_frame(3), pd.DataFrame())
    assert len(aligned) == 9


def test_build_matrix_shapes_and_dtypes():
    X, y, feats = build_matrix(align_schema(_frame(4)))
    assert X.shape[0] == 12
    assert X.shape[1] == len(feats)
    assert set(y) == {0, 1, 2}
    assert all(str(d).startswith("float") or d == "float64" for d in X.dtypes)
    assert not X.isna().any().any()


def test_build_matrix_no_engagement_drops_columns():
    aligned = align_schema(_frame(3))
    _, _, full = build_matrix(aligned)
    _, _, lean = build_matrix(aligned, no_engagement=True)
    for col in ENGAGEMENT:
        assert not any(col == f or f.startswith(col + "_") for f in lean)
    assert len(lean) < len(full)


def test_build_matrix_ablate_english():
    aligned = align_schema(_frame(3))
    _, _, feats = build_matrix(aligned, ablate_english=True)
    assert "has_english_test" not in feats


def test_build_matrix_imputes_all_nan_engagement_for_real():
    # real frames have no engagement columns at all
    aligned = align_schema(_frame(3, source="real"))
    X, _, feats = build_matrix(aligned)
    for col in ENGAGEMENT:
        if col in feats:
            assert (X[col] == 0.0).all()


def test_evaluate_reports_all_labels_and_cm():
    m = evaluate([0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 1, 1])
    assert set(m["per_class"]) == set(LABELS)
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert len(m["confusion_matrix"]) == 3
    assert all(len(r) == 3 for r in m["confusion_matrix"])


def test_evaluate_handles_absent_class():
    m = evaluate([0, 0, 1, 1], [0, 0, 1, 1])
    assert m["per_class"]["Hot"]["support"] == 0
    assert m["per_class"]["Hot"]["f1"] == 0.0


def test_make_model_kinds_and_balanced_weights():
    assert isinstance(make_model("rf"), type(make_model("rf")))
    assert make_model("rf").class_weight == "balanced"
    assert make_model("logreg").named_steps["clf"].class_weight == "balanced"
    assert make_model("dummy") is not None
    with pytest.raises(ValueError):
        make_model("nope")


def test_fit_and_eval_and_importances():
    aligned = align_schema(_frame(20))
    X, y, feats = build_matrix(aligned)
    model, metrics = fit_and_eval("rf", X, y, X, y)
    assert metrics["macro_f1"] > 0.5
    imp = feature_importances(model, feats)
    assert imp
    assert len(imp) == len(feats)
    assert list(imp.values()) == sorted(imp.values(), reverse=True)


def test_feature_importances_logreg_pipeline():
    aligned = align_schema(_frame(20))
    X, y, feats = build_matrix(aligned)
    model, _ = fit_and_eval("logreg", X, y, X, y)
    imp = feature_importances(model, feats)
    assert len(imp) == len(feats)


def test_ci95_interval_contains_mean():
    ci = _ci95([0.5, 0.6, 0.7, 0.65, 0.55])
    assert ci["n_folds"] == 5
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]
    assert ci["std"] > 0


def test_ci95_single_value_is_degenerate():
    ci = _ci95([0.8])
    assert ci["std"] == 0.0
    assert ci["ci95_low"] == ci["ci95_high"] == ci["mean"]


def test_schema_report_lists_missing_and_labels():
    aligned = align_schema(_frame(3))
    report = schema_report(aligned)
    assert report["missing_expected"] == []
    assert sum(report["label_distribution"].values()) == 9
    assert CATEGORICAL and NUMERIC  # sanity: config not empty
