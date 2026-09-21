"""Build ``artifacts/config.json``: the explicit live-scoring configuration.

The dashboard serves a model that ``leads.hybrid.load_artifacts`` used to pick
by filename fallback, with no imputation table (so every missing value became a
silent 0.0) and no recorded alpha. This module makes those choices explicit and
reproducible:

* ``model_file`` / ``model_description``: which binary serves live scores and
  what it actually is;
* ``alpha`` + ``alpha_source``: the blend weight and why it has that value;
* ``imputation``: the training-frame median of every numeric feature and the
  mode of every categorical feature -- exactly what ``train_ml.build_matrix``
  used -- so a field the chat cannot observe is imputed as "typical", never
  invented;
* ``provenance``: what the table was computed from, and a check that the
  training frame reproduces the label / source counts recorded in
  ``artifacts/metrics_rf_full.json``.

Deterministic: the real rows come from the tracked
``CounsellorForms/output`` corpus (or ``data/processed`` when present) and the
synthetic rows are the v1 generator at seed 42, n=1000, i.e. the command
``python -m leads.personas --n 1000 --seed 42`` that trained ``rf_full``.

Usage:
    python -m leads.live_config           # (re)write artifacts/config.json
    python -m leads.live_config --check   # exit 1 if the shipped file drifts
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

ARTIFACTS = PROJECT_ROOT / "artifacts"
CONFIG_PATH = ARTIFACTS / "config.json"
MODEL_FILE = "lead_model_rf_full.pkl"
METRICS_FILE = "metrics_rf_full.json"
IMPORTANCE_FILE = "feature_importance_rf_full.json"
SYNTH_N, SYNTH_SEED = 1000, 42


def _synthetic_frame():
    """The v1 synthetic corpus that trained ``rf_full`` (regenerated, seed 42)."""
    import pandas as pd
    from leads.personas import generate_sessions
    sessions, _ = generate_sessions(n=SYNTH_N, seed=SYNTH_SEED)
    df = pd.DataFrame(sessions)
    df["source"] = "synthetic"
    return df


def build_config(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Compute the config dict (does not write)."""
    import pandas as pd
    from leads import train_ml as T
    from leads.eval_augmentation import load_real_frame
    from leads.hybrid import DEFAULT_ALPHA

    art = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS
    real, real_src = load_real_frame()
    combined = T.align_schema(real, _synthetic_frame())
    report = T.schema_report(combined)

    medians: Dict[str, float] = {}
    for col in T.NUMERIC:
        med = pd.to_numeric(combined[col], errors="coerce").median()
        medians[col] = 0.0 if pd.isna(med) else round(float(med), 6)
    modes: Dict[str, str] = {}
    for col in T.CATEGORICAL:
        series = combined[col].astype(object).fillna("unknown").astype(str)
        modes[col] = str(series.mode().iloc[0])

    metrics = json.loads((art / METRICS_FILE).read_text(encoding="utf-8"))
    matches = (report["label_distribution"] == metrics["schema"]["label_distribution"]
               and report["source_counts"] == metrics["schema"]["source_counts"])
    importance = json.loads((art / IMPORTANCE_FILE).read_text(encoding="utf-8"))
    engagement = round(sum(v for k, v in importance.items()
                           if k in T.ENGAGEMENT), 4)

    return {
        "alpha": DEFAULT_ALPHA,
        "alpha_source": (
            "uncalibrated default (leads.hybrid.DEFAULT_ALPHA). The form-rubric "
            "calibration is in artifacts/alpha_calibration.json; it does not "
            "transfer to the live chat rule score, which is computed over a "
            "different feature subset."),
        "model_file": MODEL_FILE,
        "model_description": (
            "rf_full: RandomForest trained on %d real counsellor rows + %d v1 "
            "synthetic sessions. About %d%% of its importance sits on "
            "synthetic-only engagement features, so the live score is an "
            "engagement-persona score, not a validated lead-conversion model. "
            "Provisional demo output." % (
                report["source_counts"]["real"],
                report["source_counts"]["synthetic"],
                round(100 * engagement))),
        "imputation": {"medians": medians, "modes": modes},
        "provenance": {
            "generated_by": "python -m leads.live_config",
            "real_rows": real_src.split(":")[-1] if ":" in real_src else real_src,
            "synthetic": "leads.personas v1, n=%d, seed=%d" % (SYNTH_N, SYNTH_SEED),
            "n_real": report["source_counts"]["real"],
            "n_synthetic": report["source_counts"]["synthetic"],
            "label_distribution": report["label_distribution"],
            "matches_metrics_schema": bool(matches),
            "metrics_file": METRICS_FILE,
            "engagement_importance_share": engagement,
        },
    }


def _dump(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, indent=2) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write/check artifacts/config.json")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the shipped file differs")
    ap.add_argument("--out", type=Path, default=CONFIG_PATH)
    args = ap.parse_args(argv)

    cfg = build_config()
    if not cfg["provenance"]["matches_metrics_schema"]:
        print("ERROR: regenerated training frame does not match "
              "artifacts/%s; refusing to write." % METRICS_FILE, file=sys.stderr)
        return 2
    if args.check:
        try:
            shipped = json.loads(args.out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print("MISSING/UNREADABLE: %s" % args.out, file=sys.stderr)
            return 1
        # provenance.real_rows names the source file, which legitimately differs
        # between a checkout with data/processed and a clean clone.
        for c in (cfg, shipped):
            c["provenance"].pop("real_rows", None)
        if cfg != shipped:
            print("DRIFT: %s differs from the recomputed table" % args.out,
                  file=sys.stderr)
            return 1
        print("OK: %s matches the recomputed table" % args.out)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_dump(cfg), encoding="utf-8")
    print("Wrote %s" % args.out)
    print("  model_file :", cfg["model_file"])
    print("  matches metrics schema:", cfg["provenance"]["matches_metrics_schema"])
    print("  medians    :", cfg["imputation"]["medians"])
    print("  modes      :", cfg["imputation"]["modes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
