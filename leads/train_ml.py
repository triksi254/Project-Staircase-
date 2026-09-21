"""Train the behavioural lead-scoring model on hybrid real + synthetic data.

Answers the dissertation's H2 (does the model reach macro F1 > 0.80?) and adds
the *generalization* experiment: does synthetic persona augmentation actually
improve performance on **real** leads?

Usage:
    python -m leads.train_ml                          # full run, saves artifacts
    python -m leads.train_ml --no-synthetic           # real-only baseline
    python -m leads.train_ml --ablate-english         # drop noisy english signal
    python -m leads.train_ml --no-engagement          # fair generalization test
    python -m leads.train_ml --cv 5 --cv-repeat 3     # repeated stratified CV
    python -m leads.train_ml --model logreg
    python -m leads.train_ml --model dummy            # sanity floor
    python -m leads.train_ml --tag _rf_nofull         # keep each variant apart

Outputs (to artifacts/):
    lead_model<tag>.pkl       (gitignored binary)
    feature_importance<tag>.json
    metrics<tag>.json         <- appendix artifact, fully reproducible

Adaptations from the reference design (this repo's actual schema + env):
  * Column names follow ``leads.features.FEATURE_COLUMNS`` /
    ``leads.personas.SESSION_COLUMNS``: ``study_gap_mentioned``,
    ``previous_application_mentioned``, ``question_category_entropy``,
    ``visa_intent_mentioned``, ``returning_session``, ``session_word_count``.
  * ``funding_method_present`` is included (present in both sources).
  * ``label == -1`` (counsellor left no rating) is dropped before training.
  * scikit-learn 1.9 removed ``LogisticRegression(multi_class=...)``; the
    multinomial default is used instead.
  * pandas 3.0: categoricals are coerced via ``astype(object)`` before
    ``fillna`` so fills never raise on integer dtypes.
  * The confusion matrix is written into ``metrics.json`` (no matplotlib
    dependency in this venv).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
LABELS = ["Cold", "Warm", "Hot"]
LABEL_MAP = {"Cold": 0, "Warm": 1, "Hot": 2, "Good": 1, "Excellent": 2}
VALID_LABELS = (0, 1, 2)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
# NOTE: this repo already has a top-level ``models/`` *Python package*
# (models/schemas.py, the scraper's pydantic models), so the ML artifacts go to
# a dedicated ``artifacts/`` directory instead of clobbering that package.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR

# Categorical (one-hot encoded). Names must exist in the source frames or they
# are added as NaN and bucketed as "unknown".
CATEGORICAL = [
    "passport_status",
    "qual_level",
    "destination_uk",
    "has_course",
    "has_intake",
    "study_gap_mentioned",
    "previous_application_mentioned",
]

TARGET = "label"
SOURCE = "source"

#: Engagement features exist only for synthetic sessions (NaN for real leads).
#: They are strong persona separators but constitute a confound when making
#: claims about *real-world* generalization, so ``--no-engagement`` drops them.
ENGAGEMENT = [
    "message_count",
    "avg_delay_s",
    "question_category_entropy",
    "visa_intent_mentioned",
    "returning_session",
    "session_word_count",
]

# Numeric (median-imputed). Engagement columns are synthetic-only (NaN for real
# leads); ``leads.hybrid.NUMERIC`` mirrors this list and a test pins the equality.
NUMERIC = [
    "funding_clarity",
    "funding_method_present",
    "has_english_test",
    "note_word_count",
] + ENGAGEMENT


# --------------------------------------------------------------------------- #
# LOADING
# --------------------------------------------------------------------------- #
def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "data" in payload:
        return pd.DataFrame(payload["data"])
    return pd.DataFrame(payload)


def _latest(patterns: List[str], data_dir: Path = DATA_DIR) -> Optional[Path]:
    """Newest matching file, preferring .csv over .json."""
    for pattern in patterns:
        found = sorted(data_dir.glob(pattern))
        csvs = [p for p in found if p.suffix == ".csv"]
        if csvs:
            return csvs[-1]
        if found:
            return found[-1]
    return None


def load_real(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Real feature rows from ``leads.features``."""
    path = _latest(["lead_features_*.csv", "lead_features_*.json"], data_dir)
    if path is None:
        raise FileNotFoundError(
            f"No lead_features_* file in {data_dir}. Run: python -m leads.features"
        )
    df = _read_table(path)
    df[SOURCE] = "real"
    return df


def load_synthetic(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Synthetic sessions from ``leads.personas``."""
    path = _latest(["synthetic_sessions_*.csv", "synthetic_sessions_*.json"], data_dir)
    if path is None:
        raise FileNotFoundError(
            f"No synthetic_sessions_* file in {data_dir}. Run: python -m leads.personas"
        )
    df = _read_table(path)
    df[SOURCE] = "synthetic"
    return df


# --------------------------------------------------------------------------- #
# LEAD IDENTITY (repeated assessments of one lead)
# --------------------------------------------------------------------------- #
def _clean_lead_id(value: Any) -> str:
    """Canonical CRM id string; ``""`` when the id is missing/blank."""
    import re
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    s = str(value).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    if re.fullmatch(r"\d+\.0+", s):          # "375732.0" from a float CSV read
        s = s.split(".")[0]
    return s


def lead_groups(frame: pd.DataFrame, id_col: str = "crm_id") -> List[str]:
    """One group id per row: the CRM id, or a unique id when it is missing.

    39% of the labelled counsellor rows belong to a CRM id that occurs more
    than once (repeat assessments of one lead, almost always the same label).
    Splitting rows independently therefore puts a same-lead sibling in train
    for 38-42% of holdout rows; splitting by these groups does not.
    """
    ids = list(frame[id_col]) if id_col in frame.columns else [None] * len(frame)
    return [(_clean_lead_id(v) or "__row%d" % i) for i, v in enumerate(ids)]


# --------------------------------------------------------------------------- #
# SCHEMA ALIGNMENT
# --------------------------------------------------------------------------- #
def align_schema(*frames: pd.DataFrame,
                 extra_cols: Tuple[str, ...] = ()) -> pd.DataFrame:
    """Bring frames onto a common column set and drop unusable labels.

    Missing columns become NaN (so the median imputer handles them). Rows whose
    ``label`` is not in {0, 1, 2} are removed -- this includes the ``-1``
    "unrated" rows produced by ``leads.features``. ``extra_cols`` (e.g.
    ``("crm_id",)``) are carried through untouched for grouping; they are never
    model features and the default (none) leaves every existing caller as is.
    """
    keep = CATEGORICAL + NUMERIC + [TARGET, SOURCE] + list(extra_cols)
    prepared: List[pd.DataFrame] = []
    for df in frames:
        if df is None or df.empty:
            continue
        df = df.copy()
        if TARGET in df.columns and not pd.api.types.is_integer_dtype(df[TARGET]):
            df[TARGET] = df[TARGET].map(
                lambda v: LABEL_MAP.get(str(v).strip().title(), np.nan)
                if not isinstance(v, (int, np.integer)) else v
            )
        for col in keep:
            if col not in df.columns:
                df[col] = np.nan
        prepared.append(df[keep])

    if not prepared:
        return pd.DataFrame(columns=keep)

    combined = pd.concat(prepared, ignore_index=True)
    combined = combined.dropna(subset=[TARGET])
    combined = combined[combined[TARGET].isin(VALID_LABELS)]
    combined[TARGET] = combined[TARGET].astype(int)
    return combined.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# FEATURE MATRIX
# --------------------------------------------------------------------------- #
def build_matrix(
    df: pd.DataFrame,
    ablate_english: bool = False,
    no_engagement: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Return ``(X, y, feature_names)`` for the given frame.

    NOTE: median imputation is fit on *this* frame. Call this separately per
    frame (real vs synthetic) inside the generalization experiment so that
    synthetic engagement medians never leak into the real-only baseline.
    """
    cats = list(CATEGORICAL)
    nums = list(NUMERIC)
    if ablate_english:
        nums = [c for c in nums if c != "has_english_test"]
    if no_engagement:
        nums = [c for c in nums if c not in ENGAGEMENT]

    # pandas 3.0: coerce to object before fillna so integer dtypes don't raise.
    X_cat = df[cats].astype(object).fillna("unknown").astype(str)
    X_cat = pd.get_dummies(X_cat, prefix=list(CATEGORICAL), dummy_na=False)
    X_cat = X_cat.astype(float)

    X_num = df[nums].apply(pd.to_numeric, errors="coerce")
    for col in nums:
        median = X_num[col].median()
        if pd.isna(median):
            median = 0.0
        X_num[col] = X_num[col].fillna(median)

    X = pd.concat([X_num.astype(float), X_cat], axis=1)
    y = df[TARGET].to_numpy(dtype=int)
    return X, y, list(X.columns)


# --------------------------------------------------------------------------- #
# MODELS
# --------------------------------------------------------------------------- #
def _n_jobs() -> int:
    """Joblib parallelism for the forest, overridable for long sweeps.

    ``LEADS_N_JOBS=1`` avoids spawning/reusing a loky worker pool across the
    dozens of fits in a multi-seed sweep (the pool can wedge on Windows when
    many short fits are queued back to back). Defaults to -1 (all cores).
    """
    raw = os.environ.get("LEADS_N_JOBS", "")
    try:
        return int(raw)
    except ValueError:
        return -1


def make_model(kind: str):
    """Build a classifier. ``class_weight='balanced'`` handles the thin Hot class."""
    if kind == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            class_weight="balanced", n_jobs=_n_jobs(), random_state=RANDOM_STATE,
        )
    if kind == "dummy":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown model kind: {kind}")


def evaluate(y_true, y_pred, label_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Macro F1 + per-class precision/recall/F1/support + confusion matrix."""
    names = label_names or LABELS
    labels = [LABEL_MAP[n] for n in names]
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in names
        },
        "confusion_matrix": cm.tolist(),
    }


def fit_and_eval(kind, X_train, y_train, X_test, y_test, label_names=None):
    model = make_model(kind)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, evaluate(y_test, y_pred, label_names)


def feature_importances(model, feature_names: List[str]) -> Dict[str, float]:
    """Tree importances, or |coef| mean for a Pipeline ending in a linear model."""
    est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        imp = np.abs(np.atleast_2d(est.coef_)).mean(axis=0)
    else:
        return {}
    if len(imp) != len(feature_names):
        return {}
    return dict(sorted(
        zip(feature_names, (float(v) for v in imp)),
        key=lambda kv: kv[1], reverse=True,
    ))


# --------------------------------------------------------------------------- #
# EXPERIMENTS
# --------------------------------------------------------------------------- #
def experiment_combined(combined, ablate_english, model_kind, test_size=0.2,
                        no_engagement=False, random_state=RANDOM_STATE,
                        group_by_lead=False):
    """Headline result (H2): train on combined, test on a combined holdout.

    ``group_by_lead`` keeps every CRM id on one side of the split (synthetic
    rows carry no id and stay singletons).
    """
    X, y, feats = build_matrix(combined, ablate_english, no_engagement)
    if group_by_lead:
        tr, te = split_real_indices(
            combined, test_size=test_size, random_state=random_state,
            groups=lead_groups(combined))
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state,
        )
    model, metrics = fit_and_eval(model_kind, X_tr, y_tr, X_te, y_te)
    metrics["n_train"] = int(len(y_tr))
    metrics["n_test"] = int(len(y_te))
    metrics["features"] = feats
    return model, metrics


def split_real_indices(real_frame, test_size=0.2,
                       random_state=RANDOM_STATE, groups=None):
    """Stratified positional ``(train_idx, holdout_idx)`` for real rows.

    Indices are positional into ``real_frame``, which must carry integer
    ``label`` values. Shared by :func:`experiment_generalization` and the
    leakage-safe augmentation evaluation (``leads.eval_augmentation``) so the
    two always use an *identical* split -- the real holdout seen at evaluation
    time is exactly the set excluded from label-model fitting and from the v2
    bootstrap pool.

    ``groups`` (one id per row, see :func:`lead_groups`) makes the split
    *lead-grouped*: a stratified group split (~``test_size`` of the rows) in
    which no group appears on both sides. With ``groups=None`` (the default)
    the split is the original row-level stratified ``train_test_split``,
    unchanged, so every tagged result stays reproducible.
    """
    y = real_frame[TARGET].to_numpy(dtype=int)
    idx = np.arange(len(y))
    if groups is None:
        train_idx, holdout_idx = train_test_split(
            idx, test_size=test_size, stratify=y, random_state=random_state,
        )
        return train_idx, holdout_idx
    grp = list(groups)
    if len(grp) != len(y):
        raise ValueError("groups must have one entry per row (%d != %d)"
                         % (len(grp), len(y)))
    n_splits = max(2, int(round(1.0 / test_size)))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    train_idx, holdout_idx = next(iter(skf.split(idx, y, groups=grp)))
    return np.sort(train_idx), np.sort(holdout_idx)


def experiment_generalization(combined, ablate_english, model_kind, test_size=0.2,
                              no_engagement=False,
                              split_random_state=RANDOM_STATE,
                              group_by_lead=False):
    """Does synthetic augmentation improve performance on **real** leads?

    Splits the real rows into train/test. Model A sees real-train only; Model B
    trains on real-train + all synthetic. Both are scored on the same holdout.

    ``split_random_state`` lets a caller drive the split from its own protocol
    seed (``leads.eval_augmentation`` does this so the holdout is identical to
    the protocol holdout for *every* seed in a seed-robustness sweep). It
    defaults to ``RANDOM_STATE`` so existing callers are unchanged.
    """
    real = combined[combined[SOURCE] == "real"].reset_index(drop=True)
    synth = combined[combined[SOURCE] == "synthetic"].reset_index(drop=True)
    if real.empty or synth.empty:
        return None

    X_real, y_real, feats = build_matrix(real, ablate_english, no_engagement)
    X_syn, y_syn, _ = build_matrix(synth, ablate_english, no_engagement)
    # Align synthetic one-hot columns onto the real feature space
    X_syn = X_syn.reindex(columns=feats, fill_value=0.0)

    tr_idx, te_idx = split_real_indices(
        real, test_size=test_size, random_state=split_random_state,
        groups=lead_groups(real) if group_by_lead else None)
    X_rtr, X_rte = X_real.iloc[tr_idx], X_real.iloc[te_idx]
    y_rtr, y_rte = y_real[tr_idx], y_real[te_idx]

    model_real, m_real = fit_and_eval(model_kind, X_rtr, y_rtr, X_rte, y_rte)

    X_aug = pd.concat([X_rtr, X_syn], axis=0, ignore_index=True)
    y_aug = np.concatenate([y_rtr, y_syn])
    model_aug, m_aug = fit_and_eval(model_kind, X_aug, y_aug, X_rte, y_rte)
    # Holdout predictions (paired with the protocol holdout) for bootstrap CIs
    # in ``leads.eval_augmentation``. Additive keys only; no split change.
    y_true_holdout = [int(v) for v in np.asarray(y_rte).ravel().tolist()]
    m_real = dict(m_real, y_true=y_true_holdout,
                  y_pred=[int(v) for v in model_real.predict(X_rte)])
    m_aug = dict(m_aug, y_true=y_true_holdout,
                 y_pred=[int(v) for v in model_aug.predict(X_rte)])

    return {
        "real_only": m_real,
        "real_plus_synthetic": m_aug,
        "delta_macro_f1": float(m_aug["macro_f1"] - m_real["macro_f1"]),
        "delta_per_class_f1": {
            lab: float(m_aug["per_class"][lab]["f1"] - m_real["per_class"][lab]["f1"])
            for lab in LABELS
        },
        "n_real_train": int(len(y_rtr)),
        "n_real_test": int(len(y_rte)),
        "real_train_index": [int(i) for i in tr_idx],
        "real_holdout_index": [int(i) for i in te_idx],
        "n_synthetic_added": int(len(y_syn)),
        "grouped_by_lead": bool(group_by_lead),
    }


def _ci95(values: List[float]) -> Dict[str, float]:
    """Normal-approximation 95% CI for a list of fold scores."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half = 1.96 * std / max(np.sqrt(arr.size), 1.0)
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
        "n_folds": int(arr.size),
    }


def experiment_cv(combined, ablate_english, model_kind, n_splits=5, n_repeats=1,
                  no_engagement=False, group_by_lead=False):
    """Repeated stratified k-fold: mean, std and 95% CI per metric.

    Repeats use different shuffle seeds so the interval is not a single lucky
    split -- important because real Hot has only ~61 examples.
    ``group_by_lead`` uses ``StratifiedGroupKFold`` on the CRM id so repeated
    assessments of one lead never straddle a fold (the row-level default lets
    a same-lead sibling, with the same label, sit in the training folds).
    """
    X, y, _ = build_matrix(combined, ablate_english, no_engagement)
    groups = lead_groups(combined) if group_by_lead else None
    macro, per_class = [], {lab: [] for lab in LABELS}
    for repeat in range(max(n_repeats, 1)):
        if groups is None:
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True,
                random_state=RANDOM_STATE + repeat,
            )
            folds = skf.split(X, y)
        else:
            skf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True,
                random_state=RANDOM_STATE + repeat,
            )
            folds = skf.split(X, y, groups=groups)
        for tr, te in folds:
            if groups is not None:
                assert not ({groups[i] for i in tr} & {groups[i] for i in te}), \
                    "a lead straddles a CV fold"
            _, m = fit_and_eval(model_kind, X.iloc[tr], y[tr], X.iloc[te], y[te])
            macro.append(m["macro_f1"])
            for lab in LABELS:
                per_class[lab].append(m["per_class"][lab]["f1"])
    return {
        "n_splits": n_splits,
        "n_repeats": max(n_repeats, 1),
        "grouped_by_lead": bool(group_by_lead),
        "macro_f1": _ci95(macro),
        "per_class": {lab: _ci95(vals) for lab, vals in per_class.items()},
    }


# --------------------------------------------------------------------------- #
# SCHEMA REPORT
# --------------------------------------------------------------------------- #
def schema_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Which expected columns are missing / present, and the label mix."""
    expected = CATEGORICAL + NUMERIC
    # What build_matrix imputes with, recorded so live inference can reuse it
    # instead of guessing (``leads.hybrid`` reads ``schema.imputation``).
    medians = {}
    for col in NUMERIC:
        if col in df.columns:
            med = pd.to_numeric(df[col], errors="coerce").median()
            medians[col] = 0.0 if pd.isna(med) else round(float(med), 6)
    modes = {}
    for col in CATEGORICAL:
        if col in df.columns and len(df):
            modes[col] = str(
                df[col].astype(object).fillna("unknown").astype(str).mode().iloc[0])
    return {
        "columns": list(df.columns),
        "missing_expected": [c for c in expected if c not in df.columns],
        "label_distribution": {
            LABELS[int(k)]: int(v)
            for k, v in df[TARGET].value_counts().sort_index().items()
            if int(k) in VALID_LABELS
        },
        "source_counts": {str(k): int(v) for k, v in df[SOURCE].value_counts().items()},
        "imputation": {"medians": medians, "modes": modes},
    }


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train the lead-scoring model on hybrid real + synthetic data",
    )
    ap.add_argument("--no-synthetic", action="store_true",
                    help="Train on real data only (baseline)")
    ap.add_argument("--ablate-english", action="store_true",
                    help="Drop has_english_test (flagged as noisy)")
    ap.add_argument("--no-engagement", action="store_true",
                    help="Drop synthetic-only engagement features (fair "
                         "generalization test on real leads)")
    ap.add_argument("--tag", default="",
                    help="Suffix for artifact filenames, e.g. --tag _rf_full")
    ap.add_argument("--model", default="rf", choices=["rf", "logreg", "dummy"])
    ap.add_argument("--cv", type=int, default=0,
                    help="If >1, run stratified k-fold CV with this many folds")
    ap.add_argument("--cv-repeat", type=int, default=1,
                    help="Number of repeated CV passes (different shuffle seeds)")
    ap.add_argument("--group-by-lead", action="store_true",
                    help="keep every CRM id on one side of each split / CV fold "
                         "(removes same-lead leakage; default is row-level)")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=MODELS_DIR)
    args = ap.parse_args(argv)

    print("=" * 68)
    print("LEAD-SCORING MODEL TRAINING (hybrid real + synthetic)")
    print("=" * 68)

    try:
        real = load_real(args.data_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Real rows loaded:       {len(real)}")

    extra = ("crm_id",) if args.group_by_lead else ()
    if args.no_synthetic:
        combined = align_schema(real, extra_cols=extra)
        print("Synthetic:              disabled (--no-synthetic)")
    else:
        synth = load_synthetic(args.data_dir)
        print(f"Synthetic rows loaded:  {len(synth)}")
        combined = align_schema(real, synth, extra_cols=extra)
    if args.group_by_lead:
        print("Split / CV:             lead-grouped (CRM id never straddles)")

    if combined.empty:
        print("ERROR: no usable labelled rows after schema alignment.", file=sys.stderr)
        return 2

    report = schema_report(combined)
    print(f"Combined rows:          {len(combined)}")
    print(f"Label distribution:     {report['label_distribution']}")
    print(f"Source mix:             {report['source_counts']}")
    print(f"English ablated:        {args.ablate_english}")
    print(f"Engagement dropped:     {args.no_engagement}")
    print(f"Model:                  {args.model}")
    if report["missing_expected"]:
        print(f"Missing expected cols:  {report['missing_expected']}")
    print("-" * 68)

    # [1] Headline: combined train/test
    model, metrics = experiment_combined(
        combined, args.ablate_english, args.model,
        no_engagement=args.no_engagement, group_by_lead=args.group_by_lead,
    )
    print("\n[1] Combined train/test (headline H2)")
    print(f"    macro F1 : {metrics['macro_f1']:.4f}")
    for lab in LABELS:
        m = metrics["per_class"][lab]
        print(f"    {lab:5s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")
    print("    confusion matrix (rows=true, cols=pred):")
    for lab, row in zip(LABELS, metrics["confusion_matrix"]):
        print(f"      {lab:5s} {row}")

    # [2] Feature importance
    imp = feature_importances(model, metrics["features"])
    print("\n[2] Top feature importances")
    for name, val in list(imp.items())[:12]:
        print(f"    {name:34s} {val:.4f}")

    # [3] Generalization: does synthetic help real-world performance?
    gen = None
    if not args.no_synthetic:
        gen = experiment_generalization(
            combined, args.ablate_english, args.model,
            no_engagement=args.no_engagement, group_by_lead=args.group_by_lead,
        )
        if gen:
            print("\n[3] Generalization (evaluated on real-only holdout)")
            print(f"    real-train only       macro F1 = "
                  f"{gen['real_only']['macro_f1']:.4f}")
            print(f"    real + synthetic      macro F1 = "
                  f"{gen['real_plus_synthetic']['macro_f1']:.4f}")
            print(f"    delta (augmentation)  = {gen['delta_macro_f1']:+.4f}")
            print(f"    per-class delta:      {gen['delta_per_class_f1']}")

    # [4] Optional repeated CV
    cv = None
    if args.cv and args.cv > 1:
        cv = experiment_cv(combined, args.ablate_english, args.model,
                           n_splits=args.cv, n_repeats=args.cv_repeat,
                           no_engagement=args.no_engagement,
                           group_by_lead=args.group_by_lead)
        print(f"\n[4] {args.cv}-fold stratified CV x{args.cv_repeat} repeat(s)")
        mac = cv["macro_f1"]
        print(f"    macro F1 = {mac['mean']:.4f} +/- {mac['std']:.4f}  "
              f"(95% CI {mac['ci95_low']:.4f}-{mac['ci95_high']:.4f}, "
              f"{mac['n_folds']} folds)")
        for lab in LABELS:
            pc = cv["per_class"][lab]
            print(f"    {lab:5s}  F1 = {pc['mean']:.3f} +/- {pc['std']:.3f}  "
                  f"(95% CI {pc['ci95_low']:.3f}-{pc['ci95_high']:.3f})")

    # ---- Save artifacts -----------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ""
    import joblib
    joblib.dump(model, args.out / f"lead_model{tag}.pkl")
    (args.out / f"feature_importance{tag}.json").write_text(
        json.dumps(imp, indent=2), encoding="utf-8",
    )
    payload = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "schema": report,
        "headline": metrics,
        "generalization": gen,
        "cv": cv,
    }
    (args.out / f"metrics{tag}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )
    print("\n" + "-" * 68)
    print(f"Saved: {args.out / f'lead_model{tag}.pkl'}")
    print(f"Saved: {args.out / f'feature_importance{tag}.json'}")
    print(f"Saved: {args.out / f'metrics{tag}.json'}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





