"""leads.alpha_calibration: train-only, lead-grouped OOF calibration of alpha."""
import json

import pandas as pd
import pytest

from leads.alpha_calibration import main, run_calibration


def _real_like(n=360, per_lead=2, seed=5):
    """Feature frame with the columns leads.features emits, repeated leads."""
    from leads.personas import generate_sessions
    sessions, _ = generate_sessions(n=n, seed=seed)
    df = pd.DataFrame(sessions)
    df["crm_id"] = [str(300000 + i // per_lead) for i in range(len(df))]
    return df


def test_calibration_uses_train_rows_only_and_reports_both_oof_modes():
    frame = _real_like()
    out = run_calibration(frame, seed=42, n_splits=3, n_estimators=25)
    assert out["protocol"]["n_train"] < len(frame)          # holdout excluded
    assert "lead-grouped" in out["protocol"]["split"]
    for key in ("lead_grouped_oof", "row_level_oof"):
        blk = out[key]
        assert len(blk["grid"]) == 11
        assert blk["best_alpha"] in [g["alpha"] for g in blk["grid"]]
        assert blk["best_macro_f1"] == max(g["macro_f1"] for g in blk["grid"])
    assert 0.0 <= out["rubric_label_agreement_on_train"] <= 1.0
    assert "does not transfer" in out["caveat"]


def test_calibration_never_uses_a_holdout_lead(monkeypatch):
    """The rows handed to the OOF routine are exactly the grouped train split."""
    import leads.hybrid as H
    from leads.eval_augmentation import split_labeled_rows

    frame = _real_like()
    seen = {}
    real_fn = H.calibrate_alpha_kfold

    def spy(X, rule, y, **kw):
        seen.setdefault("n", []).append(len(y))
        return real_fn(X, rule, y, **kw)

    monkeypatch.setattr(H, "calibrate_alpha_kfold", spy)
    run_calibration(frame, seed=42, n_splits=3, n_estimators=10)
    split = split_labeled_rows(frame, random_state=42, group_split=True)
    assert seen["n"] == [split["n_train"], split["n_train"]]
    assert split["lead_overlap"]["n_holdout_rows_with_train_sibling"] == 0


def test_cli_writes_the_artifact(tmp_path, monkeypatch):
    import leads.eval_augmentation as E
    monkeypatch.setattr(E, "load_real_frame", lambda: (_real_like(), "test"))
    dest = tmp_path / "alpha.json"
    # keep the RF small: patch the default size used by run_calibration
    import leads.alpha_calibration as A
    orig = A.run_calibration
    monkeypatch.setattr(A, "run_calibration",
                        lambda df, seed=42: orig(df, seed=seed, n_splits=3,
                                                 n_estimators=10))
    assert main(["--out", str(dest)]) == 0
    d = json.loads(dest.read_text(encoding="utf-8"))
    assert d["protocol"]["seed"] == 42 and "reading" in d


def test_shipped_calibration_artifact_is_consistent():
    from pathlib import Path
    fp = Path(__file__).resolve().parent.parent / "artifacts" / "alpha_calibration.json"
    d = json.loads(fp.read_text(encoding="utf-8"))
    g = d["lead_grouped_oof"]
    assert g["grid"][0]["alpha"] == 0.0 and g["grid"][-1]["alpha"] == 1.0
    # the ML-only end must beat the rule-only end on these labels (rubric agrees
    # with the counsellor only ~19% of the time)
    assert g["ml_only_macro_f1 (alpha=0)"] > g["rule_only_macro_f1 (alpha=1)"]
    # lead leakage flatters the OOF score
    assert d["row_level_oof"]["best_macro_f1"] >= g["best_macro_f1"]
