"""Calibrate the hybrid blend weight alpha (train-only, lead-grouped OOF).

``leads.hybrid.calibrate_alpha_kfold`` existed and was unit-tested but was never
run, so ``alpha = 0.5`` was an unexamined default. This module runs it once, on
the counsellor corpus, under the leakage-safe protocol:

* the holdout is never touched: only the train rows of a **lead-grouped** 80/20
  split (seed 42) are used;
* the out-of-fold ML scores come from ``StratifiedGroupKFold`` on the CRM id, so
  a lead's repeat assessments never sit on both sides of an OOF fold; the
  row-level (leaky) OOF is reported alongside to show how much the leak matters;
* the ML features are the counsellor-form profile features (no engagement), and
  the rule score is the full rubric on the raw feature rows.

IMPORTANT: this calibrates the *form* rubric + form model. The live chat rule
score is a different quantity (the rubric over chat-observable topics only), so
this alpha does not transfer to the dashboard; ``artifacts/config.json`` keeps
alpha at the documented default and says so.

Usage:
    python -m leads.alpha_calibration            # writes artifacts/alpha_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_PATH = PROJECT_ROOT / "artifacts" / "alpha_calibration.json"


def _summarise(res: Dict[str, Any]) -> Dict[str, Any]:
    by_alpha = {round(r["alpha"], 2): r for r in res["results"]}
    return {
        "best_alpha": res["best_alpha"],
        "best_macro_f1": round(res["best_macro_f1"], 4),
        "ml_only_macro_f1 (alpha=0)": round(by_alpha[0.0]["macro_f1"], 4),
        "rule_only_macro_f1 (alpha=1)": round(by_alpha[1.0]["macro_f1"], 4),
        "at_default_alpha_0.5_macro_f1": round(by_alpha[0.5]["macro_f1"], 4),
        "grid": [{"alpha": round(r["alpha"], 2),
                  "macro_f1": round(r["macro_f1"], 4),
                  "accuracy": round(r["accuracy"], 4),
                  "mae": round(r["mae"], 4)}
                 for r in sorted(res["results"], key=lambda d: d["alpha"])],
    }


def run_calibration(real_df, seed: int = 42, n_splits: int = 5,
                    n_estimators: int = 300) -> Dict[str, Any]:
    """Run the calibration; ``real_df`` is the feature frame of ``leads.features``."""
    from leads.eval_augmentation import split_labeled_rows
    from leads.hybrid import calibrate_alpha_kfold
    from leads.rubric import score_row
    from leads.train_ml import build_matrix, lead_groups

    split = split_labeled_rows(real_df, random_state=seed, group_split=True)
    train = split["train"]                       # labelled train rows (+ crm_id, row_id)
    X, y, _ = build_matrix(train, no_engagement=True)
    labelled = real_df[real_df["label"].isin([0, 1, 2])].reset_index(drop=True)
    raw = labelled.iloc[train["row_id"].to_numpy()]
    scored = [score_row(dict(r)) for r in raw.to_dict("records")]
    rule = [s.score for s in scored]
    agree = sum(1 for s, t in zip(scored, y)
                if {"Cold": 0, "Warm": 1, "Hot": 2}[s.label] == int(t)) / len(y)

    Xn = X.to_numpy(dtype=float)
    grouped = calibrate_alpha_kfold(Xn, rule, y, n_splits=n_splits,
                                    random_state=seed, n_estimators=n_estimators,
                                    groups=lead_groups(train))
    row_level = calibrate_alpha_kfold(Xn, rule, y, n_splits=n_splits,
                                      random_state=seed, n_estimators=n_estimators)
    g, r = _summarise(grouped), _summarise(row_level)
    return {
        "protocol": {
            "seed": seed,
            "split": "lead-grouped 80/20; holdout untouched",
            "n_train": int(len(y)),
            "ml_features": "counsellor-form profile features (no engagement)",
            "oof_folds": n_splits,
            "rf": {"n_estimators": n_estimators, "class_weight": "balanced"},
            "label_cuts": "leads.hybrid.score_to_label (tertiles)",
        },
        "rubric_label_agreement_on_train": round(agree, 4),
        "lead_grouped_oof": g,
        "row_level_oof": r,
        "reading": (
            "Lead-grouped OOF: best alpha %.1f (macro F1 %.4f) vs ML-only %.4f, "
            "rule-only %.4f and the default alpha=0.5 at %.4f. Row-level OOF "
            "(same-lead siblings across folds) gives best alpha %.1f at %.4f."
            % (g["best_alpha"], g["best_macro_f1"],
               g["ml_only_macro_f1 (alpha=0)"], g["rule_only_macro_f1 (alpha=1)"],
               g["at_default_alpha_0.5_macro_f1"], r["best_alpha"],
               r["best_macro_f1"])),
        "caveat": (
            "Calibrated on the counsellor-form rubric and profile model. The live "
            "chat rule score is the rubric over chat-observable topics only, so "
            "this alpha does not transfer to the dashboard; artifacts/config.json "
            "keeps the documented default."),
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Train-only alpha calibration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    from leads.eval_augmentation import load_real_frame
    real_df, provenance = load_real_frame()
    out = run_calibration(real_df, seed=args.seed)
    out["protocol"]["real_rows"] = provenance.split(":")[-1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(out["reading"])
    print("rubric label agreement on train rows: %.4f"
          % out["rubric_label_agreement_on_train"])
    print("Wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
