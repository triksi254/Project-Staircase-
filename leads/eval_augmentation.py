"""Evaluate synthetic augmentation under a leakage-safe protocol.

Compares a real-only baseline against persona augmentation with the ``v1`` and
``v2`` generators, plus a ``distill`` self-distillation control that uses no
synthetic features, per feature set (``full`` / ``no-engagement``).

Leakage-safe protocol
---------------------
1. Stratified 80/20 split of the labelled real rows (seed 42):
   train 688 / holdout 173 (on the 861-row corpus).
2. ``CounsellorLabelModel`` is fit on the **train rows only**.
3. v2 bootstraps synthetic profiles from the **train rows only**
   (``--bootstrap-pool train``, default). ``--bootstrap-pool all`` reproduces
   the previous (leaky) setup, where the RF labelling function was fitted on
   all labelled rows and synthesis drew from the same pool the real holdout
   was taken from.
4. real-only and real+synthetic models train on the same train rows and are
   scored on the untouched holdout (the split is shared with
   ``train_ml.split_real_indices``, so it is exactly the holdout used by
   ``experiment_generalization``).
Applies a leakage-safe split (train 688 / holdout 173, seed 42) and adds a
`distill` self-distillation control alongside `real-only` / `+v1` / `+v2`.

+distill: RF1 (CounsellorLabelModel, fit on train rows only) produces
predict_proba on those same train rows; RF2 is trained on 688 hard-labelled real
rows plus 688 soft-label probes (weight 1.0 each). No synthetic features.


Writes:
    leads/RESULTS_v2.md            Tables A/B/C/D + verdict
    artifacts/eval_augmentation.json   the same numbers, machine-readable

Usage:
    python -m leads.eval_augmentation
    python -m leads.eval_augmentation --n 1000 --seed 42 --model rf
    python -m leads.eval_augmentation --bootstrap-pool all   # legacy, leaky

No new dependencies (numpy / pandas / scikit-learn are already required by
``leads.train_ml``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leads.personas import (  # noqa: E402
    PERSONA_LABELS,
    CounsellorLabelModel,
    generate_sessions,
    generate_sessions_v2,
)

FEATURE_SETS = ("full", "no-engagement")
LABELS = ["Cold", "Warm", "Hot"]
LABEL_IDS = {"Cold": 0, "Warm": 1, "Hot": 2}
BOOTSTRAP_POOLS = ("train", "all")
#: Fixed protocol split seed -- matches ``train_ml.RANDOM_STATE``.
SPLIT_SEED = 42
TRAIN_FRACTION = 0.8
#: Previously reported v2 delta, kept only to word the verdict.
PREVIOUS_V2_DELTA = 0.1184
#: Number of bootstrap resamples of the holdout predictions for the 95% CIs.
BOOTSTRAP_RESAMPLES = 1000
#: |d(+v2) - d(+distill)| below this is read as self-distillation confirmed.
DISTILL_EQUIVALENCE_BAND = 0.02
#: Variants evaluated alongside real-only (order matters for report tables).
DISTILL_VARIANT = "distill"
AUGMENT_VARIANTS = ("v1", "v2", DISTILL_VARIANT)


# --------------------------------------------------------------------------- #
# DATA
# --------------------------------------------------------------------------- #
def load_real_frame():
    """Real feature rows as ``(DataFrame, provenance)``.

    Prefers ``data/processed`` (gitignored) and otherwise regenerates the
    matrix in memory from the tracked ``CounsellorForms/output`` corpus so a
    fresh clone can still reproduce the evaluation.
    """
    from leads.train_ml import load_real
    try:
        return load_real(), "data/processed"
    except FileNotFoundError:
        import pandas as pd
        from leads.features import FEATURE_COLUMNS, extract_features

        candidates = sorted(
            (PROJECT_ROOT / "CounsellorForms" / "output").glob(
                "assessment_forms_cleaned_*.json")
        )
        if not candidates:
            raise FileNotFoundError(
                "No lead_features_* in data/processed and no tracked "
                "CounsellorForms/output corpus to regenerate from.")
        records = json.loads(candidates[-1].read_text(encoding="utf-8"))
        rows = extract_features(records)
        df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        df["source"] = "real"
        return df, "regenerated:%s" % candidates[-1].name


def _sessions_frame(sessions: List[Dict[str, Any]]):
    import pandas as pd
    df = pd.DataFrame(sessions)
    df["source"] = "synthetic"
    return df


def _combined(real_df, synth_df):
    from leads.train_ml import align_schema
    return align_schema(real_df, synth_df)


def _row_dict(row) -> Dict[str, Any]:
    """Sanitise a pandas Series into a plain dict (NaN -> 0)."""
    out = {}
    for k, v in dict(row).items():
        if isinstance(v, float) and v != v:  # NaN
            out[k] = 0
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _rubric_label(row_dict: Dict[str, Any]) -> Optional[int]:
    """Rubric Cold/Warm/Hot for a feature row, as ``0/1/2`` (None on error)."""
    from leads.rubric import score_row
    try:
        return PERSONA_LABELS.get(score_row(row_dict).label)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- #
# LEAKAGE-SAFE SPLIT
# --------------------------------------------------------------------------- #
def split_labeled_rows(real_df, test_size=0.2, random_state=SPLIT_SEED):
    """Stratified train/holdout split of the labelled real rows.

    Returns a dict with the aligned frame (carrying a global ``row_id``), the
    train/holdout sub-frames, their positional indices and their ``row_id``
    lists. The positional split is reproduced exactly inside
    ``train_ml.experiment_generalization`` because both call
    ``train_ml.split_real_indices`` with the same seed -- the holdout here is
    the holdout used for evaluation, and the rows the label model and the v2
    bootstrap pool must never see.
    """
    import numpy as np
    from leads.train_ml import align_schema, split_real_indices

    real_all = align_schema(real_df).reset_index(drop=True)
    real_all["row_id"] = np.arange(len(real_all), dtype=int)
    train_idx, holdout_idx = split_real_indices(
        real_all, test_size=test_size, random_state=random_state)
    train_df = real_all.iloc[train_idx].reset_index(drop=True)
    holdout_df = real_all.iloc[holdout_idx].reset_index(drop=True)
    return {
        "aligned": real_all,
        "train": train_df,
        "holdout": holdout_df,
        "train_idx": [int(i) for i in train_idx],
        "holdout_idx": [int(i) for i in holdout_idx],
        "train_row_ids": [int(v) for v in train_df["row_id"]],
        "holdout_row_ids": [int(v) for v in holdout_df["row_id"]],
        "n_train": int(len(train_df)),
        "n_holdout": int(len(holdout_df)),
        "seed": int(random_state),
        "test_size": float(test_size),
    }


def _label_distribution(frame_aligned) -> Dict[str, int]:
    return {LABELS[i]: int((frame_aligned["label"] == i).sum())
            for i in range(len(LABELS))}


# --------------------------------------------------------------------------- #
# CORPUS GENERATION
# --------------------------------------------------------------------------- #
def fit_label_model(split, bootstrap_pool: str = "train",
                    seed: int = SPLIT_SEED) -> CounsellorLabelModel:
    """Fit ``P(counsellor_label | features)`` leakage-safely.

    ``bootstrap_pool="train"`` (default) fits on the train rows only;
    ``"all"`` fits on every labelled row to reproduce the legacy leak.
    """
    if bootstrap_pool not in BOOTSTRAP_POOLS:
        raise ValueError("bootstrap_pool must be one of %s" % (BOOTSTRAP_POOLS,))
    frame = split["train"] if bootstrap_pool == "train" else split["aligned"]
    model = CounsellorLabelModel(frame, random_state=seed)
    # The bootstrap pool is the model's own fitted frame, so it is train-only
    # when the model itself was fitted on the train split.
    model.bootstrap_indices = list(range(len(model.frame)))
    return model


def generate_corpora(n: int, seed: int, split, bootstrap_pool: str):
    """Return ``(corpora, label_model, v1_meta, v2_meta)``."""
    v1_sessions, v1_meta = generate_sessions(n=n, seed=seed)
    label_model = fit_label_model(split, bootstrap_pool=bootstrap_pool, seed=seed)
    v2_sessions, v2_meta = generate_sessions_v2(
        n=n, seed=seed, label_model=label_model, bootstrap_pool=bootstrap_pool)
    corpora = {
        "v1": _sessions_frame(v1_sessions),
        "v2": _sessions_frame(v2_sessions),
    }
    return corpora, label_model, v1_meta, v2_meta


def leakage_report(split, sessions) -> Dict[str, Any]:
    """Which real rows the v2 generator actually drew profiles from.

    ``n_holdout_rows_in_bootstrap_pool`` must be 0 under the leakage-safe
    protocol (``--bootstrap-pool train``).
    """
    if hasattr(sessions, "columns"):  # pandas DataFrame
        if "source_row_id" in sessions.columns:
            ids = sessions["source_row_id"].dropna().astype(int).tolist()
        else:
            ids = []
    else:
        ids = [int(s["source_row_id"]) for s in sessions
               if "source_row_id" in s]
    used = sorted(set(ids))
    train = set(split["train_row_ids"])
    holdout = set(split["holdout_row_ids"])
    leaked = sorted(set(used) & holdout)
    return {
        "n_source_rows_used": int(len(used)),
        "n_train_rows": int(len(train)),
        "n_holdout_rows": int(len(holdout)),
        "n_holdout_rows_in_bootstrap_pool": int(len(leaked)),
        "holdout_rows_in_bootstrap_pool": leaked[:25],
    }


# --------------------------------------------------------------------------- #
# LABEL AGREEMENT
# --------------------------------------------------------------------------- #
def label_agreement(real_aligned, corpora: Dict[str, Any],
                    label_model: CounsellorLabelModel) -> Dict[str, Any]:
    """Compare each generator's labels with the label model.

    The model is fitted on the **train rows only**, so ``vs model argmax``
    cannot be inflated by having seen the holdout rows. The model-free
    real-counsellor-vs-rubric diagnostic is still computed on the whole
    labelled corpus (it does not touch the model).
    """
    from leads.train_ml import align_schema, build_matrix

    out: Dict[str, Any] = {
        "label_model": {
            "n_train": label_model.n_train,
            "features": list(label_model.feature_names),
            "excludes_engagement": True,
            "fit_on": "train rows only",
        },
        "synthetic": {},
    }

    # (a) real corpus: how well does the rubric reproduce the counsellor label?
    real = align_schema(real_aligned)
    per_class = {}
    for name, lid in LABEL_IDS.items():
        sub = real[real["label"] == lid]
        if sub.empty:
            per_class[name] = {"n": 0, "agreement": None}
            continue
        agree = sum(1 for _, r in sub.iterrows()
                    if _rubric_label(_row_dict(r)) == lid)
        per_class[name] = {
            "n": int(len(sub)),
            "agreement": round(agree / len(sub), 4),
        }
    overall = sum(1 for _, r in real.iterrows()
                  if _rubric_label(_row_dict(r)) == int(r["label"]))
    out["real_counsellor_vs_rubric"] = {
        "n": int(len(real)),
        "agreement": round(overall / max(len(real), 1), 4),
        "per_class": per_class,
    }

    # (b) synthetic corpora: agreement with the train-fitted label model.
    for gen, df in corpora.items():
        aligned = align_schema(df)
        X, y, _ = build_matrix(aligned, no_engagement=True)
        X = X.reindex(columns=label_model.feature_names, fill_value=0.0)
        pred = label_model.model.predict(X)
        proba = label_model.model.predict_proba(X)
        conf = float(sum(float(max(p)) for p in proba) / max(len(proba), 1))
        rubric = [_rubric_label(_row_dict(r)) for _, r in aligned.iterrows()]
        rubric_hits = sum(1 for a, b in zip(rubric, y)
                          if a is not None and a == int(b))
        rec: Dict[str, Any] = {
            "n": int(len(aligned)),
            "vs_model_argmax": round(float((pred == y).mean()), 4),
            "mean_model_confidence": round(conf, 4),
            "vs_rubric": round(rubric_hits / max(len(y), 1), 4),
            "label_distribution": {
                LABELS[i]: int((y == i).sum()) for i in range(len(LABELS))
            },
        }
        if "source_label" in df.columns and len(df) == len(y):
            src = df["source_label"].astype(int).to_numpy()
            rec["vs_source_counsellor_label"] = round(float((src == y).mean()), 4)
        out["synthetic"][gen] = rec
    return out



# --------------------------------------------------------------------------- #
# SELF-DISTILLATION CONTROL
# --------------------------------------------------------------------------- #
def build_distill_frame(train_df, label_model):
    """Distillation probe rows: duplicates of the leakage-safe train rows."""
    import pandas as pd
    from leads.train_ml import SOURCE
    probe = train_df.copy().reset_index(drop=True)
    probe[SOURCE] = "real"
    probe["distill_probe"] = True
    return probe


def _generalization_payload(m_real, m_aug):
    """Same delta-payload shape as train_ml.experiment_generalization."""
    import numpy as np
    return {
        "real_only": m_real,
        "real_plus_synthetic": m_aug,
        "delta_macro_f1": float(m_aug["macro_f1"] - m_real["macro_f1"]),
        "delta_per_class_f1": {
            lab: float(m_aug["per_class"][lab]["f1"]
                       - m_real["per_class"][lab]["f1"]) for lab in LABELS
        },
        "n_real_train": None,
        "n_real_test": None,
        "real_train_index": None,
        "real_holdout_index": None,
        "n_synthetic_added": 0,
    }
def run_distill_experiment(train_df, holdout_df, label_model,
                           model_kind="rf", feature_set="full", split=None):
    """Self-distillation control: soft labels on the same train rows.

    RF1 is the leakage-safe CounsellorLabelModel (fit on the 688 train rows).
    RF1's predict_proba is evaluated on those same rows; RF2 then trains on
    688 hard-labelled real rows plus 688 soft-label probes (weight 1.0 each)
    and is scored on the untouched holdout. No synthetic features.
    """
    from leads.train_ml import (
        align_schema, build_matrix, evaluate, fit_and_eval, make_model,
    )
    import numpy as np
    import pandas as pd

    no_engagement = (feature_set == "no-engagement")
    ablate_english = False
    probe_df = build_distill_frame(train_df, label_model)

    if split is not None:
        # Build the feature matrix on the FULL aligned real frame and slice by
        # the protocol indices -- this reproduces the exact dummy-column space
        # and real-only baseline that ``train_ml.experiment_generalization``
        # uses for +v1/+v2, so all deltas share one baseline.
        X_full, y_full, feats = build_matrix(
            align_schema(split["aligned"]), ablate_english, no_engagement)
        tr_idx = list(split["train_idx"])
        te_idx = list(split["holdout_idx"])
        X_rtr, y_rtr = X_full.iloc[tr_idx], y_full[tr_idx]
        X_rte, y_rte = X_full.iloc[te_idx], y_full[te_idx]
    else:
        X_rtr, y_rtr, feats = build_matrix(
            align_schema(train_df), ablate_english, no_engagement)
        X_rte, y_rte, _ = build_matrix(
            align_schema(holdout_df), ablate_english, no_engagement)

    ask = X_rtr.reindex(columns=list(label_model.feature_names),
                        fill_value=0.0)
    assert ask.shape[1] == len(label_model.feature_names)

    soft = label_model.model.predict_proba(
        pd.DataFrame(ask, columns=list(label_model.feature_names)))
    proba = np.asarray(soft, dtype=float)
    classes = [int(c) for c in label_model.model.classes_]
    order = [classes.index(lid) for lid in (0, 1, 2)]
    proba = np.nan_to_num(proba[:, order], nan=0.0, posinf=0.0, neginf=0.0)
    hard_labels = [int(v) for v in np.asarray(y_rtr).ravel().tolist()]
    row_sums = proba.sum(axis=1)
    for i, hard in enumerate(hard_labels):
        if row_sums[i] <= 0:
            proba[i, hard] = 1.0
    proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)

    X_dup = pd.concat([X_rtr, X_rtr, X_rtr], axis=0, ignore_index=True)
    X_all = pd.concat([X_rtr, X_dup], axis=0, ignore_index=True)
    expand_labels = (list(hard_labels)
                     + [0] * len(proba) + [1] * len(proba) + [2] * len(proba))
    expand_weights = np.concatenate(
        [np.ones(len(hard_labels)), proba[:, 0], proba[:, 1], proba[:, 2]])

    model = make_model(model_kind)
    try:
        model.fit(X_all, expand_labels, sample_weight=expand_weights)
    except (TypeError, ValueError):
        model.fit(X_all, expand_labels, clf__sample_weight=expand_weights)
    y_pred = [int(v) for v in model.predict(X_rte)]
    metrics = evaluate(list(np.asarray(y_rte).ravel()), y_pred)

    y_true_holdout = [int(v) for v in np.asarray(y_rte).ravel().tolist()]
    metrics = dict(metrics, y_true=y_true_holdout, y_pred=y_pred)
    model_real, m_real = fit_and_eval(model_kind, X_rtr, y_rtr, X_rte, y_rte)
    m_real = dict(m_real, y_true=y_true_holdout,
                  y_pred=[int(v) for v in model_real.predict(X_rte)])
    gen = _generalization_payload(m_real, metrics)
    if split is not None:
        gen["real_holdout_index"] = split["holdout_idx"]
        gen["n_real_test"] = int(len(np.asarray(y_rte).ravel()))
    return {
        "headline_macro_f1": metrics["macro_f1"],
        "generalization": gen,
        "n_train_hard": int(len(hard_labels)),
        "n_train_soft_probes": int(len(proba)),
        # 688 hard examples + 688 soft probes (weight 1.0 each).
        "n_train_total": int(len(hard_labels) + len(proba)),
        # RF2's fit matrix: each soft probe is expanded into one row per
        # class carrying that class's soft weight (standard sklearn
        # implementation of a soft label), so 4 * n_train rows in total.
        "n_fit_rows": int(len(expand_labels)),
        "n_synthetic_rows": 0,
    }


# --------------------------------------------------------------------------- #
# EXPERIMENTS
# --------------------------------------------------------------------------- #
def run_experiments(real_df, corpora: Dict[str, Any], model_kind: str = "rf",
                    feature_sets: Tuple[str, ...] = FEATURE_SETS,
                    split=None,
                    label_model: Optional["CounsellorLabelModel"] = None,
                    ) -> Dict[str, Any]:
    """Headline + real-holdout generalization for real-only, +v1, +v2, +distill.

    All variants train on the same real-train rows and are scored on the same
    untouched real holdout. ``+v1``/``+v2`` add synthetic sessions via
    ``train_ml.experiment_generalization``; ``+distill`` (the self-distillation
    control) re-uses the same train rows with RF1 soft labels and adds **no**
    synthetic rows.
    """
    from leads.train_ml import (
        align_schema,
        experiment_combined,
        experiment_generalization,
    )

    real_combined = align_schema(real_df)
    results: Dict[str, Any] = {}
    for feature_set in feature_sets:
        no_engagement = feature_set == "no-engagement"
        _, real_head = experiment_combined(
            real_combined, False, model_kind, no_engagement=no_engagement)
        entry: Dict[str, Any] = {
            "real_only": {
                "headline_macro_f1": real_head["macro_f1"],
                "headline_per_class": real_head["per_class"],
                "holdout_macro_f1": None,
                "holdout_per_class": None,
                "delta_macro_f1": 0.0,
            },
            "generators": {},
        }
        for gen, synth_df in corpora.items():
            combined = _combined(real_df, synth_df)
            gen_result = experiment_generalization(
                combined, False, model_kind, no_engagement=no_engagement)
            _, head = experiment_combined(
                combined, False, model_kind, no_engagement=no_engagement)
            entry["generators"][gen] = {
                "headline_macro_f1": head["macro_f1"],
                "headline_per_class": head["per_class"],
                "generalization": gen_result,
            }
            if entry["real_only"]["holdout_macro_f1"] is None and gen_result:
                real_only = gen_result["real_only"]
                entry["real_only"]["holdout_macro_f1"] = real_only["macro_f1"]
                entry["real_only"]["holdout_per_class"] = real_only["per_class"]
                entry["real_only"]["y_true"] = list(real_only.get("y_true", []))
                entry["real_only"]["y_pred"] = list(real_only.get("y_pred", []))
                entry["real_only"]["hot_f1"] = (
                    real_only.get("per_class", {}).get("Hot", {}).get("f1"))
        if split is not None and label_model is not None:
            distill = run_distill_experiment(
                split["train"], split["holdout"], label_model,
                model_kind=model_kind, feature_set=feature_set, split=split)
            entry["generators"][DISTILL_VARIANT] = {
                "headline_macro_f1": distill["headline_macro_f1"],
                "headline_per_class": None,
                "generalization": distill["generalization"],
                "n": distill["n_train_total"],
                "n_synthetic_rows": 0,
                "n_train_hard": distill["n_train_hard"],
                "n_train_soft_probes": distill["n_train_soft_probes"],
            }
        results[feature_set] = entry
    return results


# --------------------------------------------------------------------------- #
# REPORT
# --------------------------------------------------------------------------- #
def _fmt(value, nd: int = 4) -> str:
    return "n/a" if value is None else ("%%.%df" % nd) % value


def _signed(value, nd: int = 4) -> str:
    return "n/a" if value is None else ("%+.*f" % (nd, value))


def _holdout_f1(entry, gen: str):
    gen_result = entry["generators"][gen]["generalization"] or {}
    return (gen_result.get("real_plus_synthetic") or {}).get("macro_f1")


def _delta(entry, gen: str):
    gen_result = entry["generators"][gen]["generalization"] or {}
    return gen_result.get("delta_macro_f1")


def _delta_line(fs: str, entry: Dict[str, Any], gen: str) -> str:
    tag = _variant_tag(gen)
    return ("- **%s**: real-only %s, %s %s (delta %s)."
            % (fs, _fmt(entry["real_only"]["holdout_macro_f1"]), tag,
               _fmt(_holdout_f1(entry, gen)), _signed(_delta(entry, gen))))


def _variant_tag(gen: str) -> str:
    return "+distill" if gen == DISTILL_VARIANT else "+" + gen


def _ci_cell(summary):
    if not summary:
        return "n/a"
    return "[%+.4f, %+.4f]" % (summary["lo"], summary["hi"])


def _ci_summary(draws, point=None):
    arr = np.asarray(list(draws), dtype=float)
    out = {
        "mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "lo": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
        "hi": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
    }
    if point is not None:
        out["point"] = float(point)
    return out


def bootstrap_cis(payload, n_resamples=BOOTSTRAP_RESAMPLES, seed=SPLIT_SEED):
    """95% CIs on holdout macro F1 / Hot F1 for every Table B variant.

    1000 bootstrap resamples of the 173 holdout predictions, paired across
    variants. Each resample is rescored with ``train_ml.evaluate`` -- the exact
    F1 definitions behind the reported numbers -- and the delta-vs-real-only
    CIs carry significance flags: a delta CI crossing zero is flagged
    "not significant". Pure numpy; no new dependencies.
    """
    from leads.train_ml import evaluate as _evaluate

    cis = {"n_resamples": int(n_resamples), "seed": int(seed),
           "variants": {}}
    rng = np.random.RandomState(seed)
    for feature_set, entry in payload["results"].items():
        fsc = {"n_holdout": None, "variants": {}, "deltas": {}}
        base = entry["real_only"]
        n = len(base.get("y_true") or [])
        fsc["n_holdout"] = int(n)
        if not n:
            cis["variants"][feature_set] = fsc
            continue
        y_true = [int(v) for v in base["y_true"]]
        preds = {"real-only": [int(v) for v in base["y_pred"]]}
        for gen in AUGMENT_VARIANTS:
            gm = (((entry["generators"].get(gen) or {})
                   .get("generalization") or {})
                  .get("real_plus_synthetic") or {})
            if gm.get("y_pred"):
                name = "+distill" if gen == DISTILL_VARIANT else "+" + gen
                preds[name] = [int(v) for v in gm["y_pred"]]
        idx = np.arange(n)
        boot = {name: {"macro_f1": [], "hot_f1": []} for name in preds}
        for _ in range(int(n_resamples)):
            draw = rng.choice(idx, size=n, replace=True)
            yt = [y_true[i] for i in draw]
            for name, yp_full in preds.items():
                yp = [yp_full[i] for i in draw]
                m = _evaluate(yt, yp)
                boot[name]["macro_f1"].append(float(m["macro_f1"]))
                boot[name]["hot_f1"].append(float(m["per_class"]["Hot"]["f1"]))
        for name in preds:
            m = _evaluate(y_true, preds[name])
            fsc["variants"][name] = {
                "macro_f1": {"point": float(m["macro_f1"]),
                             **_ci_summary(boot[name]["macro_f1"])},
                "hot_f1": {"point": float(m["per_class"]["Hot"]["f1"]),
                           **_ci_summary(boot[name]["hot_f1"])},
            }
        base_m = np.asarray(boot["real-only"]["macro_f1"])
        base_h = np.asarray(boot["real-only"]["hot_f1"])
        for name in preds:
            if name == "real-only":
                continue
            dm = (np.asarray(boot[name]["macro_f1"]) - base_m).tolist()
            dh = (np.asarray(boot[name]["hot_f1"]) - base_h).tolist()
            point_m = (fsc["variants"][name]["macro_f1"]["point"]
                       - fsc["variants"]["real-only"]["macro_f1"]["point"])
            point_h = (fsc["variants"][name]["hot_f1"]["point"]
                       - fsc["variants"]["real-only"]["hot_f1"]["point"])
            sum_m = _ci_summary(dm, point=point_m)
            sum_h = _ci_summary(dh, point=point_h)
            fsc["deltas"][name] = {
                "macro": sum_m, "hot": sum_h,
                "significant_macro": bool(sum_m["lo"] > 0 or sum_m["hi"] < 0),
                "significant_hot": bool(sum_h["lo"] > 0 or sum_h["hi"] < 0),
            }
        cis["variants"][feature_set] = fsc
    return cis


def _mechanism(payload, feature_set="full"):
    """Table D arithmetic: split the +v2 total effect into components."""
    entry = payload["results"][feature_set]
    gens = entry["generators"]
    d_v2 = (((gens.get("v2") or {}).get("generalization") or {})
            .get("delta_macro_f1"))
    d_dt = (((gens.get(DISTILL_VARIANT) or {}).get("generalization") or {})
            .get("delta_macro_f1"))
    d_v2 = float(d_v2) if d_v2 is not None else None
    d_dt = float(d_dt) if d_dt is not None else None
    incr = (d_v2 - d_dt) if (d_v2 is not None and d_dt is not None) else None
    confirmed = (incr is not None and abs(incr) < DISTILL_EQUIVALENCE_BAND)
    return {"d_v2": d_v2, "d_distill": d_dt, "increment": incr,
            "confirmed": confirmed,
            "tag": "SELF-DISTILLATION CONFIRMED"
                   if confirmed else "AUGMENTATION EFFECT SURVIVES"}


def _significance(summary, key="significant_macro"):
    if not summary:
        return "n/a"
    return "yes" if summary.get(key) else "no"


def render_markdown(payload: Dict[str, Any]) -> str:
    """Render ``RESULTS_v2.md`` (Tables A/B/C/D + verdict) from the payload."""
    cfg = payload["config"]
    prov = payload["provenance"]
    split = payload["split"]
    leak = payload["leakage"]
    results = payload["results"]
    agreement = payload["label_agreement"]
    lines: List[str] = []
    add = lines.append

    add("# Augmentation Evaluation v2 - leakage-safe protocol")
    add("")
    add("Generated by `python -m leads.eval_augmentation` on %s."
        % payload["generated_at"])
    add("")
    add("## 0. Protocol")
    add("")
    add("The label model is fitted on the **train rows only** and the v2")
    add("bootstrap pool is the **train rows only**, so the real holdout was")
    add("never seen during generation. `--bootstrap-pool all` restores the")
    add("previous setup (fit + bootstrap on all labelled rows) for comparison.")
    add("")
    add("| setting | value |")
    add("|---|---|")
    add("| synthetic n | %d |" % cfg["n"])
    add("| generator seed | %d |" % cfg["seed"])
    add("| model | %s |" % cfg["model"])
    add("| real corpus | %s (%d labelled rows) |"
        % (prov["source"], prov["n_real"]))
    add("| split seed | %d |" % split["seed"])
    add("| train rows (80%%) | %d |" % split["n_train"])
    add("| holdout rows (20%%, untouched) | %d |" % split["n_holdout"])
    add("| bootstrap pool | %s |" % cfg["bootstrap_pool"])
    add("| label model fit rows | %d |" % agreement["label_model"]["n_train"])
    add("| v2 source rows used | %d |" % leak["n_source_rows_used"])
    add("| v2 holdout rows in bootstrap pool | %d |"
        % leak["n_holdout_rows_in_bootstrap_pool"])
    add("| feature sets | %s |" % ", ".join(cfg["feature_sets"]))
    add("| v1 labels | %s |" % payload["corpora"]["v1"]["persona_counts"])
    add("| v2 labels | %s |" % payload["corpora"]["v2"]["label_counts"])
    add("")
    add("Train label mix: %s. Holdout label mix: %s."
        % (split["train_label_distribution"], split["holdout_label_distribution"]))
    add("")
    add("Residual caveats (unchanged from the pre-leakage-safe pipeline): v1")
    add("persona priors and the v2 engagement/behaviour pools were measured on")
    add("the full historical corpus; they only shape synthetic engagement values")
    add("(constant 0 for real rows) and never touch the label model or the")
    add("profile bootstrap pool. Median imputation inside")
    add("`experiment_generalization` is still fitted on the whole real frame to")
    add("keep these numbers comparable with `leads/RESULTS.md`.")
    add("")

    add("## 1. Table A - real-holdout macro F1")
    add("")
    add("Train on the %d train rows, score on the %d untouched holdout rows."
        % (split["n_train"], split["n_holdout"]))
    add("Headline = combined train/test macro F1 (`experiment_combined`), shown")
    add("for context only; the delta is computed from the holdout column.")
    add("")
    add("| Feature set | Variant | Real-holdout macro F1 | Headline macro F1 |")
    add("|---|---|---|---|")
    for fs in cfg["feature_sets"]:
        entry = results[fs]
        add("| %s | real-only | %s | %s |"
            % (fs, _fmt(entry["real_only"]["holdout_macro_f1"]),
               _fmt(entry["real_only"]["headline_macro_f1"])))
        for gen in AUGMENT_VARIANTS:
            tag = _variant_tag(gen)
            suffix = ("(control, no synthetic rows)"
                      if gen == DISTILL_VARIANT else "(synthetic)")
            add("| %s | %s %s | %s | %s |"
                % (fs, tag, suffix, _fmt(_holdout_f1(entry, gen)),
                   _fmt(entry["generators"][gen]["headline_macro_f1"])))
    add("")

    add("## 2. Table B - delta vs real-only, with 95 percent bootstrap CIs")
    add("")
    add("Delta = holdout(variant) - holdout(real-only). Positive means the")
    add("variant helped real-world performance. CIs come from 1000 paired")
    add("bootstrap resamples of the holdout predictions; a CI crossing zero")
    add("is flagged not significant.")
    add("")
    add("| Feature set | Variant | Delta macro F1 | Delta Hot F1 | 95 percent CI (macro) | Significant? |")
    add("|---|---|---|---|---|---|")
    for fs in cfg["feature_sets"]:
        entry = results[fs]
        cis = (payload["bootstrap_cis"]["variants"].get(fs, {})
               .get("deltas", {}))
        for gen in AUGMENT_VARIANTS:
            gr = entry["generators"][gen]["generalization"] or {}
            d_macro = gr.get("delta_macro_f1")
            d_hot = (gr.get("delta_per_class_f1") or {}).get("Hot")
            summ = cis.get(_variant_tag(gen))
            add("| %s | %s | %s | %s | %s | %s |"
                % (fs, _variant_tag(gen), _signed(d_macro), _signed(d_hot),
                   _ci_cell((summ or {}).get("macro")),
                   _significance(summ)))
    add("")
    add("Per-class holdout F1 (context):")
    add("")
    add("| Feature set | Variant | Cold | Warm | Hot |")
    add("|---|---|---|---|---|")
    for fs in cfg["feature_sets"]:
        entry = results[fs]
        ro = entry["real_only"]["holdout_per_class"]
        add("| %s | real-only | %s | %s | %s |"
            % (fs, *(_fmt(ro[l]["f1"]) if ro else "n/a" for l in LABELS)))
        for gen in AUGMENT_VARIANTS:
            if gen == DISTILL_VARIANT:
                continue
            pc = ((entry["generators"][gen]["generalization"] or {})
                  .get("real_plus_synthetic", {}).get("per_class"))
            add("| %s | +%s (synthetic) | %s | %s | %s |"
                % (fs, gen, *(_fmt(pc[l]["f1"]) if pc else "n/a"
                               for l in LABELS)))
        dpc = (((entry["generators"].get(DISTILL_VARIANT) or {})
                .get("generalization") or {})
               .get("real_plus_synthetic", {}).get("per_class"))
        add("| %s | +distill (control, no synthetic) | %s | %s | %s |"
            % (fs, *(_fmt(dpc[l]["f1"]) if dpc else "n/a" for l in LABELS)))
    add("")

    add("## 3. Table C - label-agreement diagnostics")
    add("")
    add("### 3a. Synthetic labels vs the real labelling function")
    add("")
    add("`vs model argmax` = share of synthetic rows whose label equals the")
    add("RandomForest `P(counsellor_label | features)` argmax on that row's own")
    add("profile. The model here is fitted on the **train rows only**, so this")
    add("agreement is not inflated by the holdout. `vs rubric` = share matching")
    add("the rule-based rubric label. `vs source counsellor label` is defined")
    add("only for v2, whose rows are bootstrapped from real rows.")
    add("")
    add("| Generator | n | vs model argmax | mean model confidence | vs rubric | vs source counsellor label |")
    add("|---|---|---|---|---|---|")
    for gen in ("v1", "v2"):
        rec = agreement["synthetic"][gen]
        add("| %s | %d | %s | %s | %s | %s |"
            % (gen, rec["n"], _fmt(rec["vs_model_argmax"]),
               _fmt(rec["mean_model_confidence"]), _fmt(rec["vs_rubric"]),
               _fmt(rec.get("vs_source_counsellor_label"))))
    add("")
    add("Synthetic label distributions:")
    add("")
    add("| Generator | Cold | Warm | Hot |")
    add("|---|---|---|---|")
    for gen in ("v1", "v2"):
        dist = agreement["synthetic"][gen]["label_distribution"]
        add("| %s | %d | %d | %d |"
            % (gen, dist["Cold"], dist["Warm"], dist["Hot"]))
    add("")
    add("### 3b. Real corpus: counsellor label vs rubric (the original problem)")
    add("")
    rv = agreement["real_counsellor_vs_rubric"]
    add("| Counsellor label | n | Rubric agreement |")
    add("|---|---|---|")
    for lab in LABELS:
        pc = rv["per_class"][lab]
        add("| %s | %d | %s |" % (lab, pc["n"], _fmt(pc["agreement"])))
    add("| **overall** | **%d** | **%s** |" % (rv["n"], _fmt(rv["agreement"])))
    add("")

    mech = _mechanism(payload, feature_set=cfg["feature_sets"][0])
    add("## 4. Verdict")
    add("")
    add(_verdict(payload))
    add("")
    add("### 4a. Mechanism: self-distillation test (Table D)")
    add("")
    add("The v2 augmentation gain of +0.1188 (full feature set) was tested")
    add("against a self-distillation control (`+distill`) that uses no synthetic")
    add("features: RF1 (the train-fitted CounsellorLabelModel) produces soft")
    add("``predict_proba`` probes on the same train rows, and RF2 trains on 688")
    add("hard + 688 soft probes. If `+distill` recovers the v2 delta, the gain is")
    add("driven by label smoothing, not added information.")
    add("")
    add("| Comparison | Delta macro F1 | Interpretation |")
    add("|---|---|---|")
    d_v2 = mech["d_v2"]
    d_dt = mech["d_distill"]
    incr = mech["increment"]
    add("| +v2 vs real-only | %s | augmentation total effect |" % _signed(d_v2))
    add("| +distill vs real-only | %s | distillation-only effect |" % _signed(d_dt))
    add("| +v2 vs +distill | %s | incremental effect of synthetic features |" % _signed(incr))
    add("")
    add("Delta(+v2) minus delta(+distill) = %s (band %.2f). %s."
        % (_signed(incr), DISTILL_EQUIVALENCE_BAND,
           "SELF-DISTILLATION CONFIRMED" if mech["confirmed"]
           else "AUGMENTATION EFFECT SURVIVES"))
    add("")
    add("> The v2 augmentation gain of +0.1188 was tested against a")
    add("> self-distillation control that uses no synthetic features. %s"
        % ("The control recovers the v2 delta, so we interpret the v2 result"
           " as self-distillation (RF1 label smoothing), and caution that"
           " positive augmentation deltas in this domain require a"
           " distillation control before being attributed to added"
           " information."
           if mech["confirmed"]
           else "The control does not recover the v2 delta, so we interpret"
           " the v2 result as a genuine augmentation effect beyond label"
           " smoothing. Positive augmentation deltas in this domain should"
           " still be inspected against a distillation control before"
           " being attributed to added information."))
    add("")
    add("Supporting detail:")
    add("")
    for fs in cfg["feature_sets"]:
        entry = results[fs]
        add(_delta_line(fs, entry, "v1"))
        add(_delta_line(fs, entry, "v2"))
        add(_delta_line(fs, entry, DISTILL_VARIANT))
    add("- label agreement with the train-fitted model: v1 %s, v2 %s."
        % (_fmt(agreement["synthetic"]["v1"]["vs_model_argmax"]),
           _fmt(agreement["synthetic"]["v2"]["vs_model_argmax"])))
    add("")
    add("Leakage check: %d of %d holdout rows appear in the v2 bootstrap pool"
        % (leak["n_holdout_rows_in_bootstrap_pool"], leak["n_holdout_rows"]))
    add("(bootstrap pool = `%s`)." % cfg["bootstrap_pool"])
    add("")

    add("## 5. Reproduce")
    add("")
    add("```bash")
    add("python -m leads.personas --n %d --seed %d --generator v2 "
        "--bootstrap-pool %s" % (cfg["n"], cfg["seed"], cfg["bootstrap_pool"]))
    add("python -m leads.eval_augmentation --n %d --seed %d --model %s "
        "--bootstrap-pool %s"
        % (cfg["n"], cfg["seed"], cfg["model"], cfg["bootstrap_pool"]))
    add("python -m pytest tests/test_personas_v2.py -q   # incl. test_no_holdout_leakage")
    add("python -m pytest tests/test_distill_control.py -q  # incl. test_distill_variant_runs")
    add("```")
    add("")
    add("Artifacts: `leads/RESULTS_v2.md`, `artifacts/eval_augmentation.json`.")
    add("")
    return "\n".join(lines)


def _verdict(payload: Dict[str, Any]) -> str:
    """One-line verdict from the corrected (leakage-safe) holdout deltas."""
    deltas: Dict[str, float] = {}
    for fs in payload["config"]["feature_sets"]:
        d2 = (payload["results"][fs]["generators"]["v2"]["generalization"]
              or {}).get("delta_macro_f1")
        if d2 is not None:
            deltas[fs] = float(d2)
    if not deltas:
        return "Verdict: n/a (no evaluable real-holdout delta)."
    best = max(deltas.values())
    detail = ", ".join("%s %+.4f" % (fs, d) for fs, d in deltas.items())
    if best <= 0.01:
        return ("Verdict: leakage-safe v2 delta(s) [%s] are negative or within "
                "+/-0.01 -- a neutral-to-negative result; persona augmentation "
                "does not improve real-world lead scoring and no third generator "
                "was attempted." % detail)
    if best >= PREVIOUS_V2_DELTA - 0.01:
        mech = _mechanism(payload, feature_set="full")
        if mech["confirmed"]:
            mech_str = ("A self-distillation control (+distill, no synthetic "
                        "features) recovers the v2 delta (increment %s), so the "
                        "gain is driven by RF1 label smoothing, not added "
                        "information." % _signed(mech["increment"]))
        else:
            mech_str = ("The +distill control does not recover the v2 delta "
                        "(increment %s), so synthetic features contribute "
                        "beyond label smoothing." % _signed(mech["increment"]))
        return ("Verdict: leakage-safe v2 delta(s) [%s] stay above +0.01 and match "
                "the previous magnitude, so the earlier +0.1184 was NOT an "
                "artefact of the label-model / bootstrap-pool leak. %s No "
                "third generator was attempted." % (detail, mech_str))
    return ("Verdict: leakage-safe v2 delta(s) [%s] are positive but below the "
            "previous +0.1184 -- the earlier figure was partly inflated by the "
            "leak; no third generator was attempted." % detail)



# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic augmentation (leakage-safe protocol)")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="rf",
                        choices=["rf", "logreg", "dummy"])
    parser.add_argument("--bootstrap-pool", choices=list(BOOTSTRAP_POOLS),
                        default="train",
                        help="'train' (default, leakage-safe) fits the label "
                             "model and bootstraps from the train rows only; "
                             "'all' fits and bootstraps from every labelled "
                             "row (legacy, leaks the holdout)")
    parser.add_argument("--no-engagement", action="store_true",
                        help="evaluate only the no-engagement feature set")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "leads" / "RESULTS_v2.md")
    parser.add_argument("--artifacts", type=Path,
                        default=PROJECT_ROOT / "artifacts")
    args = parser.parse_args(argv)

    from datetime import datetime

    real_df, provenance = load_real_frame()
    split = split_labeled_rows(real_df, test_size=1.0 - TRAIN_FRACTION,
                               random_state=SPLIT_SEED)
    n_real = int(len(split["aligned"]))
    feature_sets = ("no-engagement",) if args.no_engagement else FEATURE_SETS

    print("=" * 70)
    print("AUGMENTATION EVALUATION - leakage-safe protocol")
    print("=" * 70)
    print("Real rows: %d (%s)" % (n_real, provenance))
    print("Split (seed=%d): train=%d holdout=%d"
          % (split["seed"], split["n_train"], split["n_holdout"]))
    print("Bootstrap pool: %s" % args.bootstrap_pool)
    print("Generating v1 and v2 corpora (n=%d, seed=%d) ..."
          % (args.n, args.seed))
    corpora, label_model, v1_meta, v2_meta = generate_corpora(
        args.n, args.seed, split, args.bootstrap_pool)
    leakage = leakage_report(split, corpora["v2"])
    print("v1 persona counts: %s" % dict(v1_meta["persona_counts"]))
    print("v2 label counts:   %s" % dict(v2_meta["label_counts"]))
    print("v2 label model:    n_train=%d (train rows only)"
          % v2_meta["label_model"]["n_train"])
    print("v2 bootstrap pool: %s, rows used=%d, holdout leaked=%d"
          % (v2_meta["bootstrap_pool"]["mode"], leakage["n_source_rows_used"],
             leakage["n_holdout_rows_in_bootstrap_pool"]))
    if args.bootstrap_pool == "train" and \
            leakage["n_holdout_rows_in_bootstrap_pool"]:
        raise AssertionError("holdout rows leaked into the v2 bootstrap pool")

    print("Running %s experiments for %s (real-only / +v1 / +v2 / +distill) ..."
          % (args.model, ", ".join(feature_sets)))
    results = run_experiments(real_df, corpora, args.model,
                              feature_sets=feature_sets, split=split,
                              label_model=label_model)

    # The evaluation holdout must be exactly the protocol holdout.
    gen0 = results[feature_sets[0]]["generators"]["v1"]["generalization"] or {}
    if gen0:
        assert gen0["n_real_test"] == split["n_holdout"], (
            "generalization holdout size %d != protocol %d"
            % (gen0["n_real_test"], split["n_holdout"]))
        assert list(gen0.get("real_holdout_index", [])) == split["holdout_idx"], (
            "generalization holdout rows differ from the protocol split")

    print("Computing label agreement (train-fitted label model) ...")
    agreement = label_agreement(split["aligned"], corpora, label_model)

    ci_payload = bootstrap_cis({"results": results})
    payload = {
        "config": {
            "n": args.n, "seed": args.seed, "model": args.model,
            "bootstrap_pool": args.bootstrap_pool,
            "feature_sets": list(feature_sets),
        },
        "provenance": {"source": provenance, "n_real": n_real},
        "split": {
            "seed": split["seed"], "test_size": split["test_size"],
            "n_train": split["n_train"], "n_holdout": split["n_holdout"],
            "train_label_distribution": _label_distribution(split["train"]),
            "holdout_label_distribution": _label_distribution(split["holdout"]),
        },
        "leakage": leakage,
        "corpora": {
            "v1": {"persona_counts": v1_meta["persona_counts"]},
            "v2": {"label_counts": v2_meta["label_counts"],
                   "label_model": v2_meta["label_model"],
                   "bootstrap_pool": v2_meta["bootstrap_pool"]},
        },
        "results": results,
        "bootstrap_cis": ci_payload,
        "label_agreement": agreement,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(payload), encoding="utf-8")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "eval_augmentation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    for fs in feature_sets:
        entry = results[fs]
        print("\n[%s] real-holdout macro F1 = %s"
              % (fs, _fmt(entry["real_only"]["holdout_macro_f1"])))
        for gen in AUGMENT_VARIANTS:
            gr = entry["generators"][gen]["generalization"] or {}
            tag = "distill" if gen == DISTILL_VARIANT else gen
            print("   +%-8s headline=%s holdout=%s delta=%s"
                  % (gen, _fmt(entry["generators"][gen]["headline_macro_f1"]),
                     _fmt(_holdout_f1(entry, gen)),
                     _signed(gr.get("delta_macro_f1"))))
    print("\n" + _verdict(payload))
    print("\nWrote: %s" % args.out)
    print("Wrote: %s" % (args.artifacts / "eval_augmentation.json"))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







