"""Hybrid lead score: alpha * rule + (1 - alpha) * ml.

Combines the explainable rule-based rubric score (leads.rubric) with the
behavioural ML model probability (leads.train_ml) into one score in [0, 1].

Conventions (same as leads.rubric / leads.personas):
  Cold = 0, Warm = 1, Hot = 2; rule_score and ml_score in [0, 1].
  ML 3-class probabilities map to scalar via expected value:
      ml_score = 0.0 * P(Cold) + 0.5 * P(Warm) + 1.0 * P(Hot)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "artifacts"

COLD, WARM, HOT = 0, 1, 2
LABEL_NAMES = ("Cold", "Warm", "Hot")

logger = logging.getLogger(__name__)


def _clip01(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"score must be numeric, got {x!r}")
    if v != v:
        raise ValueError("score must not be NaN")
    return max(0.0, min(1.0, v))


def ml_proba_to_score(proba) -> float:
    """Map 3-class ML probabilities to a scalar score in [0, 1]."""
    if isinstance(proba, dict):
        try:
            p = [float(proba.get("Cold", proba.get(0, 0.0))),
                 float(proba.get("Warm", proba.get(1, 0.0))),
                 float(proba.get("Hot", proba.get(2, 0.0)))]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bad proba dict {proba!r}: {exc}")
    else:
        try:
            p = [float(v) for v in proba]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bad proba {proba!r}: {exc}")
    if len(p) != 3:
        raise ValueError(f"proba must have 3 entries, got {len(p)}")
    if any(v < 0 for v in p):
        raise ValueError(f"proba must be non-negative, got {p}")
    total = sum(p)
    if total <= 0:
        raise ValueError(f"proba must sum to positive, got {p}")
    p = [v / total for v in p]
    return _clip01(0.5 * p[1] + 1.0 * p[2])


def _coerce_ml_score(ml) -> Optional[float]:
    if ml is None:
        return None
    if isinstance(ml, (dict, list, tuple)):
        return ml_proba_to_score(ml)
    try:
        import numpy as np
        if isinstance(ml, np.ndarray):
            if ml.size == 3:
                return ml_proba_to_score([float(v) for v in ml.tolist()])
            if ml.size == 1:
                return _clip01(float(ml.flat[0]))
    except ImportError:
        pass
    return _clip01(float(ml))


def hybrid_score(rule_score: float, ml_score=None, alpha: float = 0.5) -> float:
    """Return alpha * rule + (1 - alpha) * ml, clipped to [0, 1]."""
    rule = _clip01(rule_score)
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        raise ValueError(f"alpha must be numeric, got {alpha!r}")
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    ml = _coerce_ml_score(ml_score)
    if ml is None:
        return rule
    return _clip01(a * rule + (1.0 - a) * ml)


def score_to_label(score: float) -> str:
    """Map a [0, 1] hybrid score to Cold / Warm / Hot."""
    s = _clip01(score)
    if s < 1.0 / 3.0:
        return "Cold"
    if s < 2.0 / 3.0:
        return "Warm"
    return "Hot"

def _accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def calibrate_alpha_kfold(
    train_features,
    rule_scores,
    y_train,
    alphas=None,
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 300,
) -> Dict[str, Any]:
    """Calibrate alpha with 5-fold OOF ML scores on TRAIN rows only.

    Holdout rule: the 173-row holdout must never be indexed, sliced, or
    passed into this function. Pass the 688 training rows only.
    Runs StratifiedKFold(n_splits=5, shuffle=True, random_state=42);
    each fold fits RandomForestClassifier(n_estimators=300,
    class_weight="balanced", random_state=42, n_jobs=1) on 4/5 and takes
    predict_proba on 1/5; OOF rows pool to 688 ml_proba rows, mapped to
    scalar ML scores, then swept over alpha in {0.0..1.0} by macro F1.
    Returns calibrate_alpha result plus {"oof_ml_scores", "n_train"}.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    try:
        X = np.asarray(train_features, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"train_features must be numeric 2D: {exc}")
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("train_features must be a non-empty 2D matrix")
    yt = [int(v) for v in list(y_train)]
    rs = [_clip01(v) for v in list(rule_scores)]
    if not (X.shape[0] == len(yt) == len(rs)):
        raise ValueError("train_features, rule_scores, y_train must align")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_classes = 3
    oof = np.zeros((len(yt), n_classes))
    for tr_idx, va_idx in skf.split(X, yt):
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     class_weight="balanced",
                                     random_state=random_state, n_jobs=1)
        clf.fit(X[tr_idx], [yt[i] for i in tr_idx])
        proba = clf.predict_proba(X[va_idx])
        classes = list(clf.classes_)
        for row, probs in zip(va_idx, proba):
            for c, p in zip(classes, probs):
                oof[row, int(c)] = float(p)
    ml_scores = [ml_proba_to_score([float(v) for v in row]) for row in oof.tolist()]
    out = calibrate_alpha(yt, rs, ml_scores, alphas=alphas)
    out["oof_ml_scores"] = ml_scores
    out["n_train"] = len(yt)
    return out


def calibrate_alpha(y_true, rule_scores, ml_scores, alphas=None) -> Dict[str, Any]:
    """Grid-search alpha on labelled TRAIN-ONLY validation data (macro F1).

    Holdout rule: the 173-row holdout must never be indexed, sliced, or
    passed into this function. Pass train rows (688) only; callers that
    need fold-based ML scores should use calibrate_alpha_kfold below.
    Primary objective is macro F1 (accuracy misleads: Warm dominates with
    426 rows, so a Warm-only predictor looks fine); MAE breaks F1 ties.
    Returns {"best_alpha", "best_macro_f1", "best_accuracy", "results"}.
    Each results row: {"alpha", "macro_f1", "accuracy", "mae"}.
    """
    from sklearn.metrics import accuracy_score, f1_score

    yt = [int(v) for v in list(y_true)]
    rs = [_clip01(v) for v in list(rule_scores)]
    ms = [_coerce_ml_score(v) for v in list(ml_scores)]
    if not (len(yt) == len(rs) == len(ms)):
        raise ValueError("y_true, rule_scores, ml_scores must align")
    if any(v is None for v in ms):
        raise ValueError("ml_scores must not contain None for calibration")
    grid: List[float] = [float(a) for a in
                         (alphas if alphas is not None else [i / 10 for i in range(11)])]
    if not grid:
        raise ValueError("alphas must be non-empty")
    for a in grid:
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"alpha out of range: {a}")
    name_to_id = {"Cold": 0, "Warm": 1, "Hot": 2}
    results: List[Dict[str, float]] = []
    for a in grid:
        blended = [a * r + (1.0 - a) * m for r, m in zip(rs, ms)]
        preds = [name_to_id[score_to_label(s)] for s in blended]
        f1 = float(f1_score(yt, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(yt, preds))
        mae = sum(abs(s - (t / 2.0)) for s, t in zip(blended, yt)) / len(yt)
        results.append({"alpha": a, "macro_f1": f1, "accuracy": acc, "mae": mae})
    results.sort(key=lambda d: (-d["macro_f1"], d["mae"], d["alpha"]))
    best = results[0]
    return {"best_alpha": best["alpha"], "best_macro_f1": best["macro_f1"],
            "best_accuracy": best["accuracy"], "results": results}


def load_artifacts(models_dir=None) -> Dict[str, Any]:
    """Load saved ML artifacts if present; warn per missing file, never raise."""
    base = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    out: Dict[str, Any] = {"models_dir": str(base), "files": {}}
    pkl_path = base / "lead_model.pkl"
    if not pkl_path.is_file():
        rf_full = base / "lead_model_rf_full.pkl"
        if rf_full.is_file():
            pkl_path = rf_full
        else:
            tagged = sorted(base.glob("lead_model*.pkl"))
            tagged = [pp for pp in tagged if pp.name != "lead_model.pkl"]
            if tagged:
                pkl_path = tagged[0]

    if not pkl_path.is_file():
        logger.warning("load_artifacts: ML model missing at %s — returning rule-only config", pkl_path)
    else:
        try:
            import joblib
            out["model"] = joblib.load(pkl_path)
            out["files"]["model"] = str(pkl_path)
        except Exception:
            logger.warning("load_artifacts: ML model unreadable at %s — returning rule-only config", pkl_path)
    for key, fname in (("metrics", "metrics.json"),
                       ("feature_importance", "feature_importance.json"),
                       ("config", "config.json")):
        fp = base / fname
        if not fp.is_file():
            logger.warning("load_artifacts: %s missing at %s — returning rule-only config", key, fp)
            continue
        try:
            out[key] = json.loads(fp.read_text(encoding="utf-8"))
            out["files"][key] = str(fp)
        except (OSError, ValueError):
            logger.warning("load_artifacts: %s unreadable at %s — returning rule-only config", key, fp)
            continue
    # invalidate cache so the next predict_ml_proba call picks up the new model
    _invalidate_ml_cache()
    return out

#: Numeric feature columns the trained lead model consumes. Mirrors
#: ``leads.train_ml.NUMERIC`` (kept as a plain list here so live scoring never
#: imports the pandas/sklearn training module).
NUMERIC: List[str] = [
    "funding_clarity",
    "funding_method_present",
    "has_english_test",
    "note_word_count",
    # engagement: synthetic-only, NaN for real -> median-imputed
    "message_count",
    "avg_delay_s",
    "question_category_entropy",
    "visa_intent_mentioned",
    "returning_session",
    "session_word_count",
]

#: Categorical feature columns, one-hot expanded for the model. Mirrors
#: ``leads.train_ml.CATEGORICAL``; the per-feature value sets live in
#: ``KNOWN_CAT_VALUES`` below.
CATEGORICAL: List[str] = [
    "passport_status",
    "qual_level",
    "destination_uk",
    "has_course",
    "has_intake",
    "study_gap_mentioned",
    "previous_application_mentioned",
]

#: Known categorical values for each CATEGORICAL feature. These are the values
#: observed in the combined train/test frame (real + synthetic) used to train
#: the headline RandomForest model saved in ``artifacts/``.
KNOWN_CAT_VALUES: Dict[str, List[str]] = {
    "passport_status": ["none", "eu", "kenyan", "other", "unknown"],
    "qual_level": ["below_english", "english_qual", "high_school_diploma",
                   "undergraduate", "postgraduate", "unknown"],
    "destination_uk": ["no", "unknown"],
    "has_course": ["no", "unknown"],
    "has_intake": ["no", "unknown"],
    "study_gap_mentioned": ["no", "unknown"],
    "previous_application_mentioned": ["no", "yes", "unknown"],
}


#: Module-level cache for the loaded ML model + preprocessing parameters.
#: Invalidated whenever ``_invalidate_ml_cache()`` is called.
_ml_cache: Dict[str, Any] = {
    "config": None,
    "feature_names": None,
    "cat_values": None,
    "median_imputations": None,
}


def _invalidate_ml_cache() -> None:
    """Clear the ML prediction cache so the next call re-loads + re-derives."""
    _ml_cache["config"] = None
    _ml_cache["feature_names"] = None
    _ml_cache["cat_values"] = None
    _ml_cache["median_imputations"] = None


def _widen_cat_values(model_feature_names: List[str]) -> Dict[str, List[str]]:
    """Widen KNOWN_CAT_VALUES with any extra one-hot columns the model expects."""
    out: Dict[str, List[str]] = {k: list(v) for k, v in KNOWN_CAT_VALUES.items()}
    for col in model_feature_names:
        parts = col.split("_", 1)
        if len(parts) == 2 and parts[0] in out:
            val = parts[1]
            if val not in out[parts[0]]:
                out[parts[0]].append(val)
    return out


def _derive_feature_names(config: Dict[str, Any]) -> List[str]:
    """Return the ordered feature-name list the loaded model expects."""
    model = config.get("model")
    if model is not None and hasattr(model, "feature_names_in_"):
        names = getattr(model.feature_names_in_, "tolist",
                        lambda: list(model.feature_names_in_))()
        if names:
            return list(names)
    schema = config.get("metrics", {}).get("schema", {})
    feats = schema.get("features", []) if isinstance(schema, dict) else []
    if feats:
        return list(feats)
    return []


def _derive_median_imputations(config: Dict[str, Any]) -> Dict[str, float]:
    """Return numeric median imputation values from metrics config, else 0.0."""
    medians: Dict[str, float] = {}
    schema = config.get("metrics", {}).get("schema", {})
    if isinstance(schema, dict):
        for feat in NUMERIC:
            val = schema.get(feat, 0.0)
            try:
                medians[feat] = float(val)
            except (TypeError, ValueError):
                medians[feat] = 0.0
    return medians


def preprocess_for_prediction(
    evidence: Dict[str, Any],
    engagement: Optional[Dict[str, Any]] = None,
    medians: Optional[Dict[str, float]] = None,
    cat_values: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    """Build a feature dict from a live evidence row ready for model prediction.

    ``evidence`` is a dict as returned by ``SessionTracker.rubric_evidence()``
    (or ``leads.features`` row). ``engagement`` is an optional dict with keys
    ``message_count``, ``avg_delay_s``, ``question_category_entropy``,
    ``returning_session``, ``session_word_count``. ``medians`` and
    ``cat_values`` override the cache defaults; pass ``None`` to use cached
    values from the last ``load_artifacts`` call.

    The returned dict maps each expected model feature column to its numeric
    value; missing expected columns are filled with 0.0.
    """
    if cat_values is None:
        cat_values = _ml_cache.get("cat_values", KNOWN_CAT_VALUES)
    if medians is None:
        medians = _ml_cache.get("median_imputations", {})

    feats: Dict[str, float] = {}

    # -- numeric features --
    for feat in NUMERIC:
        val = evidence.get(feat)
        if val is None and engagement is not None:
            val = engagement.get(feat)
        try:
            feats[feat] = float(val)
        except (TypeError, ValueError):
            feats[feat] = float("nan")

    # median imputation (0.0 when no median stored for this feature)
    for feat in NUMERIC:
        v = feats[feat]
        if v != v:  # NaN
            feats[feat] = medians.get(feat, 0.0)

    # -- categorical features (one-hot) --
    for feat in CATEGORICAL:
        raw = evidence.get(feat, "unknown")
        try:
            raw = str(raw)
        except (TypeError, ValueError):
            raw = "unknown"
        norm = raw.strip().lower()
        for val in cat_values.get(feat, ["unknown"]):
            col = f"{feat}_{val}"
            feats[col] = 1.0 if norm == val.lower() else 0.0

    # fill any expected columns not yet set
    expected = _ml_cache.get("feature_names", [])
    if expected:
        for col in expected:
            if col not in feats:
                feats[col] = 0.0

    return feats


def predict_ml_proba(
    evidence: Dict[str, Any],
    engagement: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[List[float]]:
    """Return P(Cold), P(Warm), P(Hot) for a live evidence row, or None.

    Uses the model + preprocessing params from the last ``load_artifacts``
    call (or the explicit ``config`` dict). The artifacts are loaded lazily on
    the first call -- so a caller that never touched ``load_artifacts`` (the
    dashboard) still gets a prediction instead of a spurious ``None``.
    Returns None when the model is absent, not a classifier, or prediction
    fails for any reason.
    """
    if config is None:
        config = get_ml_config()
    model = config.get("model") if config else None
    if model is None or not hasattr(model, "predict_proba"):
        return None

    feature_names = _ml_cache.get("feature_names")
    if feature_names is None:
        return None

    feats = preprocess_for_prediction(evidence, engagement=engagement)
    import numpy as np
    # Column order comes from the model's own ``feature_names_in_`` (see
    # ``_derive_feature_names``), so passing a plain array is equivalent to a
    # frame with those names -- sklearn's warning about "valid feature names"
    # is a false positive here.
    import warnings
    x = np.array([[feats.get(col, 0.0) for col in feature_names]],
                  dtype=np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            proba = model.predict_proba(x)[0]
        return [float(v) for v in proba.tolist()]
    except Exception:
        return None


def build_ml_cache(config: Dict[str, Any]) -> None:
    """Derive + cache feature_names, cat_values, and median_imputations.

    Must be called after ``load_artifacts`` succeeds (config["model"] is set).
    Safe to call repeatedly; only re-derives when the config changes.
    """
    if config is _ml_cache.get("config"):
        return
    _ml_cache["config"] = config
    if config.get("model") is not None:
        _ml_cache["feature_names"] = _derive_feature_names(config)
        _ml_cache["cat_values"] = _widen_cat_values(_ml_cache["feature_names"])
    else:
        _ml_cache["feature_names"] = _derive_feature_names(config)
        _ml_cache["cat_values"] = dict(KNOWN_CAT_VALUES)
    _ml_cache["median_imputations"] = _derive_median_imputations(config)


def get_ml_config() -> Dict[str, Any]:
    """Return the cached ML config, loading artifacts on first call."""
    cfg = _ml_cache.get("config")
    if cfg is None:
        cfg = load_artifacts()
        build_ml_cache(cfg)
    return cfg
