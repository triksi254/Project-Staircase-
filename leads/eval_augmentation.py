"""Evaluate synthetic augmentation under a leakage-safe protocol.

Compares a real-only baseline against persona augmentation with the ``v1`` and
``v2`` generators, plus a ``distill`` self-distillation control that uses no
synthetic features, per feature set (``full`` / ``no-engagement``).

Leakage-safe protocol
---------------------
1. Stratified 80/20 split of the labelled real rows (seed = ``--seed``/``--seeds``;
   42 by default): train 688 / holdout 173 (on the 861-row corpus).
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
Applies a leakage-safe split (train 688 / holdout 173, seed 42 by default) and
adds a `distill` self-distillation control alongside `real-only` / `+v1` / `+v2`.

+distill: RF1 (CounsellorLabelModel, fit on train rows only) produces
predict_proba on those same train rows; RF2 is trained on 688 hard-labelled real
rows plus 688 soft-label probes (weight 1.0 each). No synthetic features.

Lead-grouped protocol (``--group-split``)
-----------------------------------------
39% of the labelled counsellor rows belong to a CRM id that occurs more than
once, and under the row-level split above 38-42% of holdout rows have a
same-lead sibling in train (always with the same label). ``--group-split``
replaces the row split with a stratified *group* split on the CRM id, so no
lead straddles train and holdout (and therefore the v2 label model and
bootstrap pool, which are train-only, never see a sibling of a holdout row).
It writes ``leads/RESULTS_v2_grouped.md`` and ``artifacts/eval_augmentation_
grouped*.json``; the default (row-level) run and its tagged artifacts are
untouched. ``--render-only`` re-renders the markdown from an existing payload
without recomputing anything.

Seed robustness
---------------
``--seeds 1,7,42`` re-runs the *entire* pipeline once per seed: the same value
drives the train/holdout split, the CounsellorLabelModel fit, the v1/v2
generators and therefore the v2 bootstrap pool. The learner seed stays fixed at
42 so only the data seed varies. Holdout macro F1 and Hot F1 are collected per
variant per seed and reported in Table E / E2 with the mean/min/max of each
delta, plus a three-tier verdict (ROBUST / PARTIALLY ROBUST / SEED-SENSITIVE).


Writes:
    leads/RESULTS_v2.md            Tables A/B/C/D/E + verdict
    artifacts/eval_augmentation.json   the same numbers, machine-readable
    artifacts/eval_augmentation_seed_robustness.json   per-seed sweep

Usage:
    python -m leads.eval_augmentation
    python -m leads.eval_augmentation --n 1000 --seed 42 --model rf
    python -m leads.eval_augmentation --seeds 1,7,42       # seed robustness
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
#: Seed-42 full-set v2 delta measured with the pre-fix (leaky) bootstrap pool,
#: kept only to compare against the leakage-safe seed-42 row of the sweep.
PREVIOUS_V2_DELTA = 0.1184
PREVIOUS_V2_SEED = 42
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


def _combined(real_df, synth_df, group_split=False):
    from leads.train_ml import align_schema
    # crm_id is carried only for the lead-grouped protocol: the default path
    # keeps its exact dtypes so the tagged results stay reproducible.
    extra = ("crm_id",) if group_split else ()
    return align_schema(real_df, synth_df, extra_cols=extra)


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
def lead_overlap_report(groups, train_idx, holdout_idx):
    """How many holdout rows have a same-lead sibling in the train split."""
    train = {groups[i] for i in train_idx}
    total = len(holdout_idx)
    n = sum(1 for i in holdout_idx if groups[i] in train)
    return {"n_holdout_rows": int(total),
            "n_holdout_rows_with_train_sibling": int(n),
            "share": round(n / total, 4) if total else 0.0}


def split_labeled_rows(real_df, test_size=0.2, random_state=SPLIT_SEED,
                       group_split=False):
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
    from leads.train_ml import align_schema, lead_groups, split_real_indices

    extra = ("crm_id",) if group_split else ()
    real_all = align_schema(real_df, extra_cols=extra).reset_index(drop=True)
    real_all["row_id"] = np.arange(len(real_all), dtype=int)
    # Lead identity is measured on every run (grouped or not) so the report can
    # state how much of the holdout has a same-lead sibling in train.
    groups = lead_groups(
        align_schema(real_df, extra_cols=("crm_id",)).reset_index(drop=True))
    train_idx, holdout_idx = split_real_indices(
        real_all, test_size=test_size, random_state=random_state,
        groups=groups if group_split else None)
    train_df = real_all.iloc[train_idx].reset_index(drop=True)
    holdout_df = real_all.iloc[holdout_idx].reset_index(drop=True)
    overlap = lead_overlap_report(groups, train_idx, holdout_idx)
    if group_split:
        row_tr, row_te = split_real_indices(
            real_all, test_size=test_size, random_state=random_state)
        row_overlap = lead_overlap_report(groups, row_tr, row_te)
    else:
        row_overlap = overlap
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
        "grouped_by_lead": bool(group_split),
        "lead_overlap": overlap,
        "row_split_lead_overlap": row_overlap,
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
                    split_random_state: int = SPLIT_SEED,
                    group_split: bool = False,
                    ) -> Dict[str, Any]:
    """Headline + real-holdout generalization for real-only, +v1, +v2, +distill.

    All variants train on the same real-train rows and are scored on the same
    untouched real holdout. ``+v1``/``+v2`` add synthetic sessions via
    ``train_ml.experiment_generalization``; ``+distill`` (the self-distillation
    control) re-uses the same train rows with RF1 soft labels and adds **no**
    synthetic rows.

    ``split_random_state`` must equal the seed used to build ``split`` so the
    holdout evaluated here is exactly the protocol holdout (asserted by the
    caller) -- this is what makes a seed sweep measure the split and the
    generator together.
    """
    from leads.train_ml import (
        align_schema,
        experiment_combined,
        experiment_generalization,
    )

    extra = ("crm_id",) if group_split else ()
    real_combined = align_schema(real_df, extra_cols=extra)
    results: Dict[str, Any] = {}
    for feature_set in feature_sets:
        no_engagement = feature_set == "no-engagement"
        _, real_head = experiment_combined(
            real_combined, False, model_kind, no_engagement=no_engagement,
            random_state=split_random_state, group_by_lead=group_split)
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
            combined = _combined(real_df, synth_df, group_split)
            gen_result = experiment_generalization(
                combined, False, model_kind, no_engagement=no_engagement,
                split_random_state=split_random_state,
                group_by_lead=group_split)
            _, head = experiment_combined(
                combined, False, model_kind, no_engagement=no_engagement,
                random_state=split_random_state, group_by_lead=group_split)
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
# SEED ROBUSTNESS
# --------------------------------------------------------------------------- #
#: Evaluated variants, in report order (real-only is the delta baseline).
ROBUSTNESS_VARIANTS = ("real-only", "v1", "v2", DISTILL_VARIANT)
DELTA_VARIANTS = ("v1", "v2", DISTILL_VARIANT)


def parse_seeds(spec: str) -> List[int]:
    """Parse ``--seeds "1,7,42"`` -> ``[1, 7, 42]``.

    A single value (``"42"``) yields a one-element list, so the pre-existing
    single-seed command line reproduces the earlier result exactly.
    """
    seeds: List[int] = []
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def _holdout_hot_f1(entry: Dict[str, Any], gen: str):
    gen_result = entry["generators"][gen]["generalization"] or {}
    return ((gen_result.get("real_plus_synthetic") or {})
            .get("per_class", {}).get("Hot", {}).get("f1"))


def _real_only_hot_f1(entry: Dict[str, Any]):
    return ((entry["real_only"].get("holdout_per_class") or {})
            .get("Hot", {}).get("f1"))


def seed_robustness(per_seed: Dict[int, Dict[str, Any]],
                    feature_sets) -> Dict[str, Any]:
    """Aggregate per-seed holdout metrics into per-variant deltas + ranges.

    ``per_seed`` maps seed -> the payload returned by :func:`_pipeline_for_seed`.
    For every feature set this collects the holdout macro F1 and Hot F1 of each
    variant, the delta vs real-only per seed, and the mean/min/max of each
    delta across seeds.
    """
    seeds = sorted(per_seed)
    out: Dict[str, Any] = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "feature_sets": {},
    }
    for fs in feature_sets:
        rows: List[Dict[str, Any]] = []
        for seed in seeds:
            entry = per_seed[seed]["results"][fs]
            row: Dict[str, Any] = {
                "seed": int(seed),
                "real_only_macro": entry["real_only"]["holdout_macro_f1"],
                "real_only_hot": _real_only_hot_f1(entry),
            }
            for gen in DELTA_VARIANTS:
                gen_result = (entry["generators"][gen]["generalization"] or {})
                aug = gen_result.get("real_plus_synthetic") or {}
                row["%s_macro" % gen] = aug.get("macro_f1")
                row["%s_hot" % gen] = (aug.get("per_class", {})
                                       .get("Hot", {}).get("f1"))
                row["delta_%s_macro" % gen] = gen_result.get("delta_macro_f1")
                row["delta_%s_hot" % gen] = (gen_result.get("delta_per_class_f1")
                                             or {}).get("Hot")
            rows.append(row)

        summary: Dict[str, Any] = {}
        for gen in DELTA_VARIANTS:
            rec: Dict[str, Any] = {}
            for metric in ("macro", "hot"):
                key = "delta_%s_%s" % (gen, metric)
                vals = [float(r[key]) for r in rows if r[key] is not None]
                rec[metric] = {
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if vals else None,
                    "min": float(np.min(vals)) if vals else None,
                    "max": float(np.max(vals)) if vals else None,
                }
            summary[gen] = rec
        out["feature_sets"][fs] = {"rows": rows, "summary": summary}
    return out


def _seed_verdict(summary: Dict[str, Any], gen: str = "v2") -> Optional[Dict[str, str]]:
    """Three-tier robustness verdict for one feature set.

    Rules (evaluated in this order, on the across-seed macro delta):
      1. min > 0.02                                -> ROBUST
      2. min > 0 and max > 0.15 and min < 0.05     -> PARTIALLY ROBUST
      3. any seed <= 0                             -> SEED-SENSITIVE
      4. otherwise (positive but tiny everywhere)  -> INCONCLUSIVE
    """
    rec = (summary or {}).get(gen, {}).get("macro") or {}
    if not rec or rec.get("min") is None:
        return None
    lo, hi, mean = rec["min"], rec["max"], rec["mean"]
    rng = "mean %+.4f, range [%+.4f, %+.4f] over %d seeds" % (
        mean, lo, hi, rec.get("n", 0))
    if lo > 0.02:
        return {"tier": "ROBUST",
                "reason": "minimum across seeds is above +0.02 (%s)" % rng}
    if lo > 0 and hi > 0.15 and lo < 0.05:
        return {"tier": "PARTIALLY ROBUST",
                "reason": "direction stable but magnitude seed-dependent (%s)" % rng}
    if lo <= 0:
        return {"tier": "SEED-SENSITIVE",
                "reason": "at least one seed gives a non-positive delta (%s); "
                          "the point estimate is not reliable, report the range"
                          % rng}
    return {"tier": "INCONCLUSIVE",
            "reason": "positive at every seed but below the 0.02 stability "
                      "band (%s)" % rng}


def _overall_seed_verdict(robust: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """One-line verdict. Leads with the fair (no-engagement) configuration.

    The no-engagement set is the like-for-like comparison: real rows carry no
    engagement features, so in the ``full`` set engagement alone identifies a
    row's source. The ``full`` verdict is reported second, with that caveat.
    """
    sets = list(robust["feature_sets"])
    primary = "no-engagement" if "no-engagement" in sets else sets[0]
    block = robust["feature_sets"][primary]
    verdict = _seed_verdict(block["summary"], "v2")
    if verdict is None:
        return "Verdict: n/a (no evaluable across-seed delta for %s)." % primary
    detail = ""
    others = [fs for fs in sets if fs != primary]
    if others:
        parts = []
        for fs in others:
            v = _seed_verdict(robust["feature_sets"][fs]["summary"], "v2")
            if v:
                rec = robust["feature_sets"][fs]["summary"]["v2"]["macro"]
                parts.append("%s %s (range [%+.4f, %+.4f])"
                             % (fs, v["tier"], rec["min"], rec["max"]))
        if parts:
            detail = " Other feature sets: %s." % "; ".join(parts)
    note = (" Tiers are descriptive rules on the across-seed range (min > +0.02 "
            "= ROBUST), not significance tests, and %d seeds re-split the same "
            "rows." % robust["n_seeds"])
    if "full" in others:
        note += (" The full configuration's gain depends on engagement columns "
                 "that are constant (0) for every real row.")
    return "Verdict (%s configuration): **%s** -- %s.%s%s" % (
        primary, verdict["tier"], verdict["reason"], detail, note)


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


def _render_row_vs_grouped(payload: Dict[str, Any]) -> List[str]:
    """Same-seed comparison with the committed row-level payload (grouped runs)."""
    if not (payload.get("split") or {}).get("grouped_by_lead"):
        return []
    fp = PROJECT_ROOT / "artifacts" / "eval_augmentation.json"
    try:
        base = json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    a = (base.get("seed_robustness") or {}).get("feature_sets") or {}
    b = (payload.get("seed_robustness") or {}).get("feature_sets") or {}
    lines: List[str] = [
        "## 5b. Row-level vs lead-grouped split (same seeds)", "",
        "Deltas are holdout macro F1 minus the same run's real-only baseline;",
        "`mean [min, max]` over the seeds. Row-level = `leads/RESULTS_v2.md`.", "",
        "| Feature set | Variant | Row-level | Lead-grouped |",
        "|---|---|---|---|"]
    for fs in b:
        if fs not in a:
            continue
        for g in DELTA_VARIANTS:
            ra = a[fs]["summary"][g]["macro"]
            rb = b[fs]["summary"][g]["macro"]
            lines.append("| %s | +%s | %s [%s, %s] | %s [%s, %s] |" % (
                fs, g, _signed(ra["mean"]), _signed(ra["min"]), _signed(ra["max"]),
                _signed(rb["mean"]), _signed(rb["min"]), _signed(rb["max"])))
    lines += ["", "Real-only baseline (holdout macro F1 / Hot F1) per seed:", "",
              "| Feature set | Seed | Row-level macro / Hot | Lead-grouped macro / Hot |",
              "|---|---|---|---|"]
    for fs in b:
        if fs not in a:
            continue
        rows_a = {r["seed"]: r for r in a[fs]["rows"]}
        for r in b[fs]["rows"]:
            o = rows_a.get(r["seed"])
            if o is None:
                continue
            lines.append("| %s | %d | %s / %s | %s / %s |" % (
                fs, r["seed"], _fmt(o["real_only_macro"]), _fmt(o["real_only_hot"]),
                _fmt(r["real_only_macro"]), _fmt(r["real_only_hot"])))
    lines += ["", "*A grouped holdout is a different set of ~173 rows, so the",
              "baselines move by sampling noise (about 12-13 Hot rows each) in",
              "either direction; only the augmentation deltas are compared here.*", ""]
    return lines


def _render_seed_robustness(robust: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """Tables E / E2 -- holdout macro F1 and Hot F1 per variant per seed."""
    lines: List[str] = []
    add = lines.append
    add("## 5. Table E - Seed robustness")
    add("")
    add("Each seed drives **both** the train/holdout split and the v1/v2")
    add("generators (and therefore the CounsellorLabelModel fit and the v2")
    add("bootstrap pool), so a seed change re-runs the entire pipeline. The")
    add("learner seed stays fixed at 42 so only the data seed varies. Deltas")
    add("are holdout macro F1 / Hot F1 vs the same seed's real-only baseline.")
    add("")
    if robust["n_seeds"] < 2:
        add("> Single-seed run: min/max equal the point estimate, so the tier")
        add("> below is a point estimate, not a robustness assessment.")
        add("")
    for fs, block in robust["feature_sets"].items():
        summary = block["summary"]
        verdict = _seed_verdict(summary, "v2")
        add("### %s (%d seeds: %s)" % (fs, robust["n_seeds"],
                                       ", ".join(str(s) for s in robust["seeds"])))
        add("")
        add("#### Table E - holdout macro F1")
        add("")
        add("| Seed | real-only | +v1 | +v2 | +distill | D(+v2) | D(+distill) |")
        add("|---|---|---|---|---|---|---|")
        for row in block["rows"]:
            add("| %d | %s | %s | %s | %s | %s | %s |"
                % (row["seed"], _fmt(row["real_only_macro"]),
                   _fmt(row["v1_macro"]), _fmt(row["v2_macro"]),
                   _fmt(row["distill_macro"]), _signed(row["delta_v2_macro"]),
                   _signed(row["delta_distill_macro"])))
        add("| **mean [min, max]** | | | | | **%s [%s, %s]** | **%s [%s, %s]** |"
            % (_signed(summary["v2"]["macro"]["mean"]),
               _signed(summary["v2"]["macro"]["min"]),
               _signed(summary["v2"]["macro"]["max"]),
               _signed(summary[DISTILL_VARIANT]["macro"]["mean"]),
               _signed(summary[DISTILL_VARIANT]["macro"]["min"]),
               _signed(summary[DISTILL_VARIANT]["macro"]["max"])))
        add("")
        add("Delta summary (macro F1): " + "; ".join(
            "D(+%s) mean %s, range [%s, %s]"
            % (g, _signed(summary[g]["macro"]["mean"]),
               _signed(summary[g]["macro"]["min"]),
               _signed(summary[g]["macro"]["max"]))
            for g in ("v1", "v2", DISTILL_VARIANT)) + ".")
        add("")
        add("#### Table E2 - holdout Hot F1")
        add("")
        add("| Seed | real-only | +v1 | +v2 | +distill | D(+v2) | D(+distill) |")
        add("|---|---|---|---|---|---|---|")
        for row in block["rows"]:
            add("| %d | %s | %s | %s | %s | %s | %s |"
                % (row["seed"], _fmt(row["real_only_hot"]),
                   _fmt(row["v1_hot"]), _fmt(row["v2_hot"]),
                   _fmt(row["distill_hot"]), _signed(row["delta_v2_hot"]),
                   _signed(row["delta_distill_hot"])))
        add("| **mean [min, max]** | | | | | **%s [%s, %s]** | **%s [%s, %s]** |"
            % (_signed(summary["v2"]["hot"]["mean"]),
               _signed(summary["v2"]["hot"]["min"]),
               _signed(summary["v2"]["hot"]["max"]),
               _signed(summary[DISTILL_VARIANT]["hot"]["mean"]),
               _signed(summary[DISTILL_VARIANT]["hot"]["min"]),
               _signed(summary[DISTILL_VARIANT]["hot"]["max"])))
        add("")
        if verdict:
            add("Tier (%s): **%s** -- %s." % (fs, verdict["tier"], verdict["reason"]))
        else:
            add("Tier (%s): n/a." % fs)
        add("")
    add("### Seed-robustness verdict")
    add("")
    add(_overall_seed_verdict(robust, cfg))
    add("")
    return lines


def render_markdown(payload: Dict[str, Any]) -> str:
    """Render ``RESULTS_v2.md`` (Tables A/B/C/D/E + verdict) from the payload."""
    cfg = payload["config"]
    prov = payload["provenance"]
    split = payload["split"]
    leak = payload["leakage"]
    results = payload["results"]
    agreement = payload["label_agreement"]
    lines: List[str] = []
    add = lines.append

    grouped = bool(split.get("grouped_by_lead"))
    add("# Augmentation Evaluation v2 - %s"
        % ("lead-grouped split, leakage-safe protocol" if grouped
           else "leakage-safe protocol (row-level split)"))
    add("")
    add("Generated by `python -m leads.eval_augmentation` on %s."
        % payload["generated_at"])
    if payload.get("_rendered_from"):
        add("")
        add("> Re-rendered on %s from `%s` without recomputation: the numbers are"
            % (payload.get("_rendered_at", "?"), payload["_rendered_from"]))
        add("> unchanged; only the explanatory text was corrected (an earlier")
        add("> version hard-coded a seed-42 figure and a verdict that contradicted")
        add("> its own Table E).")
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
    add("| split type | %s |" % ("lead-grouped (a CRM id never straddles "
                                 "train and holdout)" if grouped
                                 else "row-level stratified"))
    lo = split.get("lead_overlap")
    if lo:
        add("| holdout rows with a same-lead sibling in train | %d of %d (%.0f%%) |"
            % (lo["n_holdout_rows_with_train_sibling"], lo["n_holdout_rows"],
               100 * lo["share"]))
    ro = split.get("row_split_lead_overlap")
    if grouped and ro:
        add("| ... under the row-level split (reference) | %d of %d (%.0f%%) |"
            % (ro["n_holdout_rows_with_train_sibling"], ro["n_holdout_rows"],
               100 * ro["share"]))
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
    add("Residual caveats (unchanged from the pre-leakage-safe pipeline):")
    add("")
    add("- The v1 profile marginals in `leads/personas.py` were measured on the")
    add("  full labelled corpus (holdout included). They shape v1's *profile*")
    add("  features; v1 still loses, so the bias is in v1's favour and the")
    add("  conclusion stands, but the leak is real for v1.")
    add("- The v1/v2 engagement (behaviour) pools are **hand-authored** ranges,")
    add("  not measured from data (no chat logs exist). In the `full` set they")
    add("  are the only signal separating synthetic from real rows.")
    add("- Median imputation inside `experiment_generalization` is fitted on the")
    add("  whole real frame to stay comparable with `leads/RESULTS.md`; real")
    add("  profile columns have no NaN and engagement is constant 0, so this")
    add("  changes nothing measurable.")
    add("")

    add("## 1. Table A - real-holdout macro F1")
    add("")
    add("*Tables A-D show a single run: generator/split seed %d (the first seed"
        % cfg["seed"])
    add("of the sweep, not a chosen one). Table E shows every seed.*")
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

    fs0 = cfg["feature_sets"][0]
    mech = _mechanism(payload, feature_set=fs0)
    add("## 4. Verdict")
    add("")
    add(_verdict(payload))
    add("")
    add("### 4a. Mechanism: self-distillation test (Table D)")
    add("")
    add("The v2 delta of %s (%s feature set, seed %d) was compared with a"
        % (_signed(mech["d_v2"]), fs0, cfg["seed"]))
    add("self-distillation control (`+distill`) that adds no synthetic rows: RF1")
    add("(the train-fitted CounsellorLabelModel) produces `predict_proba` on the")
    add("same train rows and RF2 trains on the 688 hard rows plus those soft")
    add("probes. If `+distill` reproduced the v2 delta, the gain would be label")
    add("smoothing rather than added information.")
    add("")
    add("| Comparison | Delta macro F1 | Interpretation |")
    add("|---|---|---|")
    d_v2 = mech["d_v2"]
    d_dt = mech["d_distill"]
    incr = mech["increment"]
    add("| +v2 vs real-only | %s | augmentation total effect |" % _signed(d_v2))
    add("| +distill vs real-only | %s | distillation-only effect |" % _signed(d_dt))
    add("| +v2 vs +distill | %s | difference between the two arms |" % _signed(incr))
    add("")
    add("Delta(+v2) minus delta(+distill) = %s (descriptive band %.2f; no CI)."
        % (_signed(incr), DISTILL_EQUIVALENCE_BAND))
    add("Tag: %s." % ("SELF-DISTILLATION CONFIRMED" if mech["confirmed"]
                      else "AUGMENTATION EFFECT SURVIVES"))
    add("")
    add("> %s"
        % ("The control reproduces the v2 delta to within the band, which is"
           " consistent with label smoothing rather than added information."
           if mech["confirmed"]
           else "The control does not reproduce the v2 delta within the band."
           " That does not by itself show that synthetic features add"
           " information, because the control is not like-for-like with +v2."))
    add(">")
    add("> **Caveat.** The control differs from +v2 in more than the presence of")
    add("> synthetic rows: (i) 4 x 688 rows with sample weights instead of 1000")
    add("> fresh bootstrap rows with sampled hard labels; (ii) its soft labels are")
    add("> RF1's *in-sample* probabilities; (iii) `class_weight=\"balanced\"` is")
    add("> recomputed on the expanded label vector, changing the effective class")
    add("> weighting; (iv) it has no engagement variation. The no-engagement rows")
    add("> of Table E are the cleaner test of whether synthetic *profiles* help.")
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
    add("(bootstrap pool = `%s`). This is a *row-level* check. Lead-level:" % cfg["bootstrap_pool"])
    if lo:
        add("%d of %d holdout rows (%.0f%%) have a same-lead sibling in the train"
            % (lo["n_holdout_rows_with_train_sibling"], lo["n_holdout_rows"],
               100 * lo["share"]))
        add("split%s." % ("" if not grouped else " (0 expected: the split is lead-grouped)"))
    else:
        add("not recorded in this payload (produced before the lead-level check).")
    add("")

    if payload.get("seed_robustness"):
        lines.extend(_render_seed_robustness(payload["seed_robustness"], cfg))
        lines.extend(_render_row_vs_grouped(payload))

    add("## 6. Limitations of this evaluation")
    add("")
    add("- The real holdout has about 12 Hot rows, so every Hot F1 rests on ~12")
    add("  positives; a single prediction moves it by 0.05-0.1.")
    add("- The bootstrap CIs resample the holdout *predictions* only: they capture")
    add("  test-sampling variance, not training variance or split variance.")
    add("- Seeds re-split the same rows, so holdouts overlap across seeds; the v2")
    add("  design was iterated against the seed-42 holdout. There is no final")
    add("  untouched test set.")
    add("- The +distill control is not like-for-like with +v2 (see 4a).")
    add("- In the `full` set real rows have engagement = 0 and synthetic rows > 0.")
    add("- %s" % ("Lead-level leakage is removed by the grouped split; compare with "
                  "`leads/RESULTS_v2.md` for the row-level numbers." if grouped
                  else "Lead-level leakage is NOT removed here (see the protocol table); "
                  "`leads/RESULTS_v2_grouped.md` reports the lead-grouped split."))
    add("")

    add("## 7. Reproduce")
    add("")
    add("```bash")
    add("python -m leads.personas --n %d --seed %d --generator v2 "
        "--bootstrap-pool %s" % (cfg["n"], cfg["seed"], cfg["bootstrap_pool"]))
    add("python -m leads.eval_augmentation --n %d --seed %d --model %s "
        "--bootstrap-pool %s%s"
        % (cfg["n"], cfg["seed"], cfg["model"], cfg["bootstrap_pool"],
           " --group-split" if grouped else ""))
    add("python -m pytest tests/test_personas_v2.py -q   # incl. test_no_holdout_leakage")
    add("python -m pytest tests/test_distill_control.py -q  # incl. test_distill_variant_runs")
    add("```")
    add("")
    add("Artifacts: `leads/RESULTS_v2.md`, `artifacts/eval_augmentation.json`.")
    add("")
    return "\n".join(lines)


def _previous_seed_check(payload: Dict[str, Any]) -> str:
    """Compare the leakage-safe seed-42 full delta with the pre-fix figure."""
    if (payload.get("split") or {}).get("grouped_by_lead"):
        return ""      # the pre-fix figure was a row-level number; see RESULTS_v2.md
    rows = ((((payload.get("seed_robustness") or {}).get("feature_sets") or {})
             .get("full") or {}).get("rows") or [])
    for r in rows:
        if r.get("seed") == PREVIOUS_V2_SEED and r.get("delta_v2_macro") is not None:
            d = float(r["delta_v2_macro"])
            diff = d - PREVIOUS_V2_DELTA
            if abs(diff) < 0.01:
                return ("The seed-%d full-set delta with the leakage-safe pool (%s) "
                        "matches the pre-fix figure (%s) to within %.4f, so the "
                        "earlier bootstrap-pool leak had no material effect on "
                        "that seed." % (PREVIOUS_V2_SEED, _signed(d),
                                        _signed(PREVIOUS_V2_DELTA), abs(diff)))
            return ("The seed-%d full-set delta with the leakage-safe pool (%s) "
                    "differs from the pre-fix figure (%s) by %s."
                    % (PREVIOUS_V2_SEED, _signed(d), _signed(PREVIOUS_V2_DELTA),
                       _signed(diff)))
    return ""


def _verdict(payload: Dict[str, Any]) -> str:
    """Verdict from this payload's own seed (plus the sweep, when present)."""
    cfg = payload["config"]
    deltas: Dict[str, float] = {}
    for fs in cfg["feature_sets"]:
        d2 = (payload["results"][fs]["generators"]["v2"]["generalization"]
              or {}).get("delta_macro_f1")
        if d2 is not None:
            deltas[fs] = float(d2)
    if not deltas:
        return "Verdict: n/a (no evaluable real-holdout delta)."
    detail = ", ".join("%s %+.4f" % (fs, d) for fs, d in deltas.items())
    parts = ["Verdict (seed %d): v2 holdout macro-F1 delta(s) [%s]."
             % (cfg["seed"], detail)]
    fair = deltas.get("no-engagement")
    if fair is not None:
        if fair <= 0.01:
            parts.append(
                "In the no-engagement configuration - the fair comparison, "
                "because real rows carry no engagement features - persona "
                "augmentation does not improve real-lead scoring (%s)."
                % _signed(fair))
        else:
            parts.append("In the no-engagement configuration augmentation "
                         "gains %s." % _signed(fair))
    full = deltas.get("full")
    if full is not None and full > 0.01 and (fair is None or fair <= 0.01):
        parts.append(
            "The positive full-set delta (%s) appears only when engagement "
            "columns are present; every real row has engagement = 0 there while "
            "every synthetic row has engagement > 0, so engagement identifies a "
            "row's source and the gain is not evidence that behavioural "
            "information transfers to real leads (a hypothesis, not proven)."
            % _signed(full))
    prev = _previous_seed_check(payload)
    if prev:
        parts.append(prev)
    parts.append("No third generator was attempted.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def _pipeline_for_seed(seed: int, real_df, provenance: str, args,
                       feature_sets) -> Dict[str, Any]:
    """Run the full protocol once for one seed and return its payload.

    The same ``seed`` drives the train/holdout split, the CounsellorLabelModel
    fit and the v1/v2 generators, so a seed change perturbs the whole pipeline.
    The learner seed inside ``make_model`` stays fixed at 42.
    """
    from datetime import datetime

    group_split = bool(getattr(args, "group_split", False))
    split = split_labeled_rows(real_df, test_size=1.0 - TRAIN_FRACTION,
                               random_state=seed, group_split=group_split)
    n_real = int(len(split["aligned"]))

    print("-" * 70)
    print("SEED %d: split seed=%d -> train=%d holdout=%d"
          % (seed, split["seed"], split["n_train"], split["n_holdout"]))
    lo = split["lead_overlap"]
    print("Lead-level overlap: %d of %d holdout rows have a same-lead sibling "
          "in train (%s split)" % (lo["n_holdout_rows_with_train_sibling"],
                                   lo["n_holdout_rows"],
                                   "lead-grouped" if group_split else "row-level"))
    print("Generating v1 and v2 corpora (n=%d, seed=%d) ..."
          % (args.n, seed))
    corpora, label_model, v1_meta, v2_meta = generate_corpora(
        args.n, seed, split, args.bootstrap_pool)
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
                              label_model=label_model,
                              split_random_state=seed, group_split=group_split)

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

    return {
        "config": {
            "n": args.n, "seed": seed, "model": args.model,
            "bootstrap_pool": args.bootstrap_pool,
            "feature_sets": list(feature_sets),
            "group_split": group_split,
        },
        "provenance": {"source": provenance, "n_real": n_real},
        "split": {
            "seed": split["seed"], "test_size": split["test_size"],
            "n_train": split["n_train"], "n_holdout": split["n_holdout"],
            "train_label_distribution": _label_distribution(split["train"]),
            "holdout_label_distribution": _label_distribution(split["holdout"]),
            "grouped_by_lead": split["grouped_by_lead"],
            "lead_overlap": split["lead_overlap"],
            "row_split_lead_overlap": split["row_split_lead_overlap"],
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic augmentation (leakage-safe protocol)")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42,
                        help="single-seed mode (kept for backward compatibility)")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seed list, e.g. '1,7,42'; each "
                             "seed drives the split AND the generators. "
                             "Overrides --seed when given.")
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
    parser.add_argument("--group-split", action="store_true",
                        help="lead-grouped train/holdout split on the CRM id "
                             "(no lead straddles); writes RESULTS_v2_grouped.md "
                             "and artifacts/eval_augmentation_grouped*.json")
    parser.add_argument("--tag", default=None,
                        help="suffix for the artifact filenames (default: "
                             "'' or '_grouped' with --group-split)")
    parser.add_argument("--render-only", action="store_true",
                        help="re-render the markdown from an existing payload; "
                             "recomputes nothing and writes no artifacts")
    parser.add_argument("--from-artifact", type=Path, default=None,
                        help="payload for --render-only (default: "
                             "artifacts/eval_augmentation<tag>.json)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path,
                        default=PROJECT_ROOT / "artifacts")
    args = parser.parse_args(argv)
    tag = args.tag if args.tag is not None else (
        "_grouped" if args.group_split else "")
    if args.out is None:
        args.out = PROJECT_ROOT / "leads" / (
            "RESULTS_v2_grouped.md" if args.group_split else "RESULTS_v2.md")

    if args.render_only:
        from datetime import date
        src = args.from_artifact or (args.artifacts / ("eval_augmentation%s.json" % tag))
        payload = json.loads(Path(src).read_text(encoding="utf-8"))
        payload["_rendered_from"] = "artifacts/%s" % Path(src).name
        payload["_rendered_at"] = date.today().isoformat()
        args.out.write_text(render_markdown(payload), encoding="utf-8")
        print("Re-rendered %s from %s (no recomputation)" % (args.out, src))
        return 0

    seeds = parse_seeds(args.seeds) if args.seeds else [int(args.seed)]
    feature_sets = ("no-engagement",) if args.no_engagement else FEATURE_SETS

    print("=" * 70)
    print("AUGMENTATION EVALUATION - leakage-safe protocol")
    print("=" * 70)
    real_df, provenance = load_real_frame()
    from leads.train_ml import align_schema as _align
    print("Real rows: %d labelled of %d loaded (%s)"
          % (len(_align(real_df)), len(real_df), provenance))
    print("Seeds: %s (each drives the split AND the generators)" % seeds)
    print("Bootstrap pool: %s" % args.bootstrap_pool)

    per_seed: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        per_seed[seed] = _pipeline_for_seed(
            seed, real_df, provenance, args, feature_sets)

    payload = per_seed[seeds[0]]
    robust = seed_robustness(per_seed, feature_sets)
    payload["seed_robustness"] = robust
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    # Persist the computed payload FIRST: a rendering failure must never lose
    # 12+ minutes of sweep compute (this is exactly how the earlier run died).
    payload_path = args.artifacts / ("eval_augmentation%s.json" % tag)
    payload_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    robust_path = args.artifacts / (
        "eval_augmentation%s_seed_robustness.json" % tag)
    if len(seeds) > 1:
        robust_path.write_text(
            json.dumps({"config": payload["config"],
                        "seed_robustness": robust},
                       indent=2, default=str), encoding="utf-8")
    try:
        args.out.write_text(render_markdown(payload), encoding="utf-8")
    except Exception as exc:  # rendering is best-effort; artifacts are safe
        print("WARNING: markdown render failed (%s: %s); payload already "
              "written to %s" % (type(exc).__name__, exc, payload_path),
              file=sys.stderr)

    for fs in feature_sets:
        entry = payload["results"][fs]
        print("\n[%s] real-holdout macro F1 = %s (seed %d)"
              % (fs, _fmt(entry["real_only"]["holdout_macro_f1"]),
                 payload["config"]["seed"]))
        for gen in AUGMENT_VARIANTS:
            gr = entry["generators"][gen]["generalization"] or {}
            print("   +%-8s headline=%s holdout=%s delta=%s"
                  % (gen, _fmt(entry["generators"][gen]["headline_macro_f1"]),
                     _fmt(_holdout_f1(entry, gen)),
                     _signed(gr.get("delta_macro_f1"))))
    print("\n" + _verdict(payload))
    if len(seeds) > 1:
        print(_overall_seed_verdict(robust, payload["config"]))
    print("\nWrote: %s" % args.out)
    print("Wrote: %s" % payload_path)
    if len(seeds) > 1:
        print("Wrote: %s" % robust_path)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







