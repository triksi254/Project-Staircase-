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
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

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
        logger.warning("load_artifacts: ML model missing at %s — returning rule-only config", pkl_path)
    else:
        try:
            import pickle
            with open(pkl_path, "rb") as fh:
                out["model"] = pickle.load(fh)
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
    return out