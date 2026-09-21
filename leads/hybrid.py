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

#: Label cuts for the [0, 1] hybrid score (tertiles). ``chatbot.responder``
#: derives its Hot-escalation threshold from ``WARM_HOT_CUT``. The rubric's own
#: descriptive label cuts (``leads.rubric``: 0.35 / 0.68) are older and differ
#: slightly on purpose: they were used for the rubric-vs-counsellor agreement
#: table, which must not move. The two disagree only inside [1/3, 0.35) and
#: [2/3, 0.68) (pinned by ``tests/test_config_consistency.py``).
COLD_WARM_CUT = 1.0 / 3.0
WARM_HOT_CUT = 2.0 / 3.0

#: Default blend weight in ``alpha * rule + (1 - alpha) * ml``. This is an
#: *uncalibrated* default: ``calibrate_alpha_kfold`` exists but its result is
#: not a shipped input (see ``artifacts/alpha_calibration.json`` for the
#: train-only run). Every caller takes it from here or from ``config.json``.
DEFAULT_ALPHA = 0.5

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


def hybrid_score(rule_score: float, ml_score=None,
                 alpha: float = DEFAULT_ALPHA) -> float:
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
    if s < COLD_WARM_CUT:
        return "Cold"
    if s < WARM_HOT_CUT:
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
    groups=None,
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

    ``groups`` (optional, one id per row, e.g. the CRM id) switches the folds
    to ``StratifiedGroupKFold`` so repeated assessments of one lead never sit
    on both sides of an OOF fold (39% of the 861 labelled rows belong to a
    CRM id that occurs more than once).
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

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
    if groups is not None:
        grp = list(groups)
        if len(grp) != len(yt):
            raise ValueError("groups must align with train_features")
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                   random_state=random_state)
        folds = skf.split(X, yt, groups=grp)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)
        folds = skf.split(X, yt)
    n_classes = 3
    oof = np.zeros((len(yt), n_classes))
    for tr_idx, va_idx in folds:
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


def _select_model_file(base: Path, cfg: Dict[str, Any]):
    """Pick the model binary explicitly; return ``(path, how)`` or ``(None, None)``.

    ``how`` is ``"config"`` (named in config.json), ``"default"``
    (``lead_model.pkl``) or ``"fallback"`` (an experiment model picked by
    filename; the caller announces it).
    """
    named = cfg.get("model_file")
    if named:
        p = base / str(named)
        if p.is_file():
            return p, "config"
        logger.warning(
            "load_artifacts: config.json names model_file=%s but %s does not exist",
            named, p)
    default = base / "lead_model.pkl"
    if default.is_file():
        return default, "default"
    rf_full = base / "lead_model_rf_full.pkl"
    if rf_full.is_file():
        return rf_full, "fallback"
    tagged = [pp for pp in sorted(base.glob("lead_model*.pkl"))
              if pp.name != "lead_model.pkl"]
    if tagged:
        return tagged[0], "fallback"
    return None, None


def load_artifacts(models_dir=None) -> Dict[str, Any]:
    """Load saved ML artifacts if present; warn per missing file, never raise.

    ``config.json`` (tracked, written by ``scripts/write_live_config.py``) is
    read first: it names the model file, the default alpha and the imputation
    table used at inference. Model selection order: ``config.json`` ->
    ``model_file``; ``lead_model.pkl``; otherwise a **fallback** to
    ``lead_model_rf_full.pkl`` / the first tagged model, which is logged as a
    warning because that is an experiment model, not a deliberate deployment.
    """
    base = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    out: Dict[str, Any] = {"models_dir": str(base), "files": {}}

    cfg_path = base / "config.json"
    if cfg_path.is_file():
        try:
            out["config"] = json.loads(cfg_path.read_text(encoding="utf-8"))
            out["files"]["config"] = str(cfg_path)
        except (OSError, ValueError):
            logger.warning(
                "load_artifacts: config unreadable at %s — using built-in defaults "
                "(alpha=%s)", cfg_path, DEFAULT_ALPHA)
    else:
        logger.warning(
            "load_artifacts: config missing at %s — using built-in defaults "
            "(alpha=%s, no shipped imputation table)", cfg_path, DEFAULT_ALPHA)

    chosen, how = _select_model_file(base, out.get("config") or {})
    if chosen is None:
        logger.warning("load_artifacts: ML model missing at %s — returning rule-only config",
                       base / "lead_model.pkl")
    else:
        if how == "fallback":
            logger.warning(
                "load_artifacts: %s not found; FALLBACK to %s. This is a tagged "
                "experiment model, not a deliberate deployment - set model_file "
                "in config.json to choose explicitly.",
                base / "lead_model.pkl", chosen.name)
        try:
            import joblib
            out["model"] = joblib.load(chosen)
            out["files"]["model"] = str(chosen)
            out["model_selection"] = how
        except Exception as exc:
            logger.warning("load_artifacts: ML model unreadable at %s (%s) — "
                           "returning rule-only config", chosen, exc)
    for key, fname in (("metrics", "metrics.json"),
                       ("feature_importance", "feature_importance.json")):
        fp = base / fname
        if not fp.is_file():
            logger.warning("load_artifacts: %s missing at %s", key, fp)
            continue
        try:
            out[key] = json.loads(fp.read_text(encoding="utf-8"))
            out["files"][key] = str(fp)
        except (OSError, ValueError):
            logger.warning("load_artifacts: %s unreadable at %s", key, fp)
            continue
    # invalidate cache so the next predict_ml_proba call picks up the new model
    _invalidate_ml_cache()
    return out


def load_config_json(models_dir=None) -> Dict[str, Any]:
    """Read ``config.json`` only (no model load); ``{}`` when absent/unreadable."""
    base = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    fp = base / "config.json"
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def get_default_alpha(config: Optional[Dict[str, Any]] = None) -> float:
    """Blend weight from ``config.json`` when present, else ``DEFAULT_ALPHA``."""
    cfg = config if config is not None else get_ml_config()
    raw = ((cfg or {}).get("config") or {}).get("alpha", DEFAULT_ALPHA)
    try:
        a = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_ALPHA
    return a if 0.0 <= a <= 1.0 else DEFAULT_ALPHA


def active_model_info(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Which model is serving live scores, and how it was chosen (for the UI)."""
    cfg = config if config is not None else get_ml_config()
    path = (cfg or {}).get("files", {}).get("model")
    desc = ((cfg or {}).get("config") or {}).get("model_description")
    return {"file": Path(path).name if path else None,
            "selection": (cfg or {}).get("model_selection"),
            "description": desc}


#: Numeric feature columns the trained lead model consumes. Mirrors
#: ``leads.train_ml.NUMERIC`` (kept as a plain list here so live scoring never
#: imports the pandas/sklearn training module). ``tests/test_config_consistency``
#: fails if the two drift.
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
#: ``leads.train_ml.CATEGORICAL``.
CATEGORICAL: List[str] = [
    "passport_status",
    "qual_level",
    "destination_uk",
    "has_course",
    "has_intake",
    "study_gap_mentioned",
    "previous_application_mentioned",
]

#: Known values of each CATEGORICAL feature, as the *strings* ``pd.get_dummies``
#: produced when the model was trained: ``leads.features`` codes them as small
#: integers (passport 0 expired / 1 none / 2 valid, qual_level 0..5 index into
#: ``QUAL_LEVEL_ORDER``, the rest 0/1), so the model's columns are
#: ``passport_status_0``, ``has_course_1``, ... . Used only when the model's own
#: feature names cannot be read; otherwise the model's columns are authoritative.
KNOWN_CAT_VALUES: Dict[str, List[str]] = {
    "passport_status": ["0", "1", "2"],
    "qual_level": ["0", "1", "2", "3", "4", "5"],
    "destination_uk": ["0", "1"],
    "has_course": ["0", "1"],
    "has_intake": ["0", "1"],
    "study_gap_mentioned": ["0", "1"],
    "previous_application_mentioned": ["0", "1"],
}

#: Most frequent value of each categorical feature in the training frame
#: (real + v1 synthetic). Used to fill a field chat cannot observe, and only
#: when neither ``config.json`` nor ``metrics.json`` ships an imputation table.
#: "Unknown" is filled with the typical training value, never with a negative.
DEFAULT_CAT_MODES: Dict[str, str] = {
    "passport_status": "2",
    "qual_level": "0",
    "destination_uk": "1",
    "has_course": "1",
    "has_intake": "1",
    "study_gap_mentioned": "0",
    "previous_application_mentioned": "0",
}


#: Module-level cache for the loaded ML model + preprocessing parameters.
#: Invalidated whenever ``_invalidate_ml_cache()`` is called.
_ml_cache: Dict[str, Any] = {
    "config": None,
    "feature_names": None,
    "cat_values": None,
    "median_imputations": None,
    "mode_imputations": None,
    "_warned_imputation": False,
}


def _invalidate_ml_cache() -> None:
    """Clear the ML prediction cache so the next call re-loads + re-derives."""
    _ml_cache["config"] = None
    _ml_cache["feature_names"] = None
    _ml_cache["cat_values"] = None
    _ml_cache["median_imputations"] = None
    _ml_cache["mode_imputations"] = None
    _ml_cache["_warned_imputation"] = False


def _norm_cat(value: Any) -> Optional[str]:
    """Canonical string for a categorical value: 2, 2.0, "2", "2.0" -> "2".

    ``None`` / NaN / blank -> ``None`` (missing).
    """
    import numbers
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        f = float(value)
        if f != f:
            return None
        return str(int(f)) if f.is_integer() else str(f)
    s = str(value).strip().lower()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f.is_integer() else s


def _derive_cat_values(model_feature_names: List[str]) -> Dict[str, List[str]]:
    """Per categorical feature, the column suffixes the *model* was trained on.

    The model's own one-hot names are authoritative (matched by the longest
    feature-name prefix, so underscores inside feature names are safe);
    ``KNOWN_CAT_VALUES`` is the fallback when no names are available.
    """
    found: Dict[str, List[str]] = {feat: [] for feat in CATEGORICAL}
    for col in model_feature_names:
        for feat in sorted(CATEGORICAL, key=len, reverse=True):
            prefix = feat + "_"
            if col.startswith(prefix):
                found[feat].append(col[len(prefix):])
                break
    return {feat: (found[feat] or list(KNOWN_CAT_VALUES[feat]))
            for feat in CATEGORICAL}


def _derive_feature_names(config: Dict[str, Any]) -> List[str]:
    """Return the ordered feature-name list the loaded model expects."""
    model = config.get("model")
    if model is not None and hasattr(model, "feature_names_in_"):
        names = getattr(model.feature_names_in_, "tolist",
                        lambda: list(model.feature_names_in_))()
        if names:
            return list(names)
    schema = (config.get("metrics") or {}).get("schema", {})
    feats = schema.get("features", []) if isinstance(schema, dict) else []
    if feats:
        return list(feats)
    return []


def _imputation_table(config: Dict[str, Any]):
    """``(medians, modes)`` shipped with the artifacts.

    Sources, later wins: ``metrics.json`` -> ``schema.imputation`` (written by
    newer ``train_ml`` runs) then ``config.json`` -> ``imputation`` (written by
    ``scripts/write_live_config.py`` for the shipped model).
    """
    cfg_imp = ((config.get("config") or {}).get("imputation") or {})
    schema = (config.get("metrics") or {}).get("schema") or {}
    schema_imp = schema.get("imputation") or {} if isinstance(schema, dict) else {}
    medians: Dict[str, float] = {}
    modes: Dict[str, str] = {}
    for src in (schema_imp, cfg_imp):
        for feat, val in (src.get("medians") or {}).items():
            try:
                medians[feat] = float(val)
            except (TypeError, ValueError):
                continue
        for feat, val in (src.get("modes") or {}).items():
            norm = _norm_cat(val)
            if norm is not None:
                modes[feat] = norm
    return medians, modes


def _warn_imputation_once(features: List[str]) -> None:
    """A missing imputation table must be visible, not a silent 0.0."""
    if _ml_cache.get("_warned_imputation"):
        return
    _ml_cache["_warned_imputation"] = True
    logger.warning(
        "preprocess_for_prediction: no imputation table for %s; using built-in "
        "fallbacks (numeric 0.0, categorical training mode). Ship "
        "artifacts/config.json 'imputation' (scripts/write_live_config.py).",
        sorted(set(features)))


def preprocess_for_prediction(
    evidence: Dict[str, Any],
    engagement: Optional[Dict[str, Any]] = None,
    medians: Optional[Dict[str, float]] = None,
    cat_values: Optional[Dict[str, List[str]]] = None,
    modes: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Build the model's input dict from a live evidence row.

    ``evidence`` is ``SessionTracker.rubric_evidence()`` (or a
    ``leads.features`` row); ``engagement`` is ``SessionTracker.features()``.
    A numeric value is looked up in ``evidence``, then ``engagement``, then
    ``evidence["session_flags"]`` (where the visa flag lives). Anything still
    unknown is *imputed*, never invented: numerics take the training median,
    categoricals the training mode (a field chat cannot observe is "typical",
    not "negative"). ``medians`` / ``modes`` / ``cat_values`` override the cache.

    Categorical one-hot columns are named exactly as the model's own feature
    names (``passport_status_2`` ...), so every categorical feature switches on
    exactly one column.
    """
    if cat_values is None:
        cat_values = _ml_cache.get("cat_values") or KNOWN_CAT_VALUES
    if medians is None:
        medians = _ml_cache.get("median_imputations") or {}
    if modes is None:
        modes = _ml_cache.get("mode_imputations") or {}
    flags = evidence.get("session_flags") or {}
    engagement = engagement or {}

    fallback_used: List[str] = []
    feats: Dict[str, float] = {}

    # -- numeric features --
    for feat in NUMERIC:
        val = evidence.get(feat)
        if val is None:
            val = engagement.get(feat)
        if val is None:
            val = flags.get(feat)
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = float("nan")
        if v != v:  # NaN / missing -> training median
            if feat in medians:
                v = float(medians[feat])
            else:
                v = 0.0
                fallback_used.append(feat)
        feats[feat] = v

    # -- categorical features (one-hot, model column names) --
    for feat in CATEGORICAL:
        norm = _norm_cat(evidence.get(feat))
        if norm is None:  # not observed -> training mode
            norm = modes.get(feat)
            if norm is None:
                norm = DEFAULT_CAT_MODES.get(feat)
                fallback_used.append(feat)
        for val in cat_values.get(feat, KNOWN_CAT_VALUES.get(feat, [])):
            feats["%s_%s" % (feat, val)] = 1.0 if _norm_cat(val) == norm else 0.0

    if fallback_used:
        _warn_imputation_once(fallback_used)

    # any expected column still unset (e.g. a value never seen live) is 0.0
    for col in _ml_cache.get("feature_names") or []:
        feats.setdefault(col, 0.0)
    return feats


def predict_ml_proba(
    evidence: Dict[str, Any],
    engagement: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[List[float]]:
    """Return P(Cold), P(Warm), P(Hot) for a live evidence row, or None.

    Uses the model + preprocessing params of ``config`` (or, when omitted, the
    cached artifacts loaded lazily on first use -- so a caller that never
    touched ``load_artifacts``, like the dashboard, still gets a prediction).
    Returns None when the model is absent or not a classifier; a *failed*
    prediction also returns None but is logged with its reason.
    """
    if config is None:
        config = get_ml_config()
    else:
        build_ml_cache(config)
    model = config.get("model") if config else None
    if model is None or not hasattr(model, "predict_proba"):
        return None

    feature_names = _ml_cache.get("feature_names")
    if not feature_names:
        logger.warning("predict_ml_proba: model exposes no feature names "
                       "(feature_names_in_ / metrics schema); cannot align inputs")
        return None

    import numpy as np
    import warnings
    try:
        feats = preprocess_for_prediction(evidence, engagement=engagement)
        # Column order comes from the model's own ``feature_names_in_``, so a
        # plain array is equivalent to a frame with those names -- sklearn's
        # "valid feature names" warning is a false positive here.
        x = np.array([[feats.get(col, 0.0) for col in feature_names]],
                     dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            proba = model.predict_proba(x)[0]
        return [float(v) for v in proba.tolist()]
    except Exception as exc:
        logger.warning("predict_ml_proba failed (%s: %s) — falling back to the "
                       "rule-only score", type(exc).__name__, exc)
        return None


def build_ml_cache(config: Dict[str, Any]) -> None:
    """Derive + cache feature_names, cat_values and the imputation table.

    Safe to call repeatedly; only re-derives when the config object changes.
    """
    if config is _ml_cache.get("config"):
        return
    _ml_cache["config"] = config
    names = _derive_feature_names(config)
    _ml_cache["feature_names"] = names
    _ml_cache["cat_values"] = _derive_cat_values(names)
    medians, modes = _imputation_table(config)
    _ml_cache["median_imputations"] = medians
    _ml_cache["mode_imputations"] = modes
    _ml_cache["_warned_imputation"] = False


def get_ml_config() -> Dict[str, Any]:
    """Return the cached ML config, loading artifacts on first call."""
    cfg = _ml_cache.get("config")
    if cfg is None:
        cfg = load_artifacts()
        build_ml_cache(cfg)
    return cfg
