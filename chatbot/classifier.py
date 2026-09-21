"""Question-type classifier: query text -> one of the 9 FAQ categories.

Trained on the enriched faq_corpus.json (98 entries with category field):
TF-IDF (1-2 grams) + LogisticRegression, class_weight='balanced'.
98 entries / 9 categories is thin (Visa & Immigration and General
Enquiries have 5 each). The 5-fold CV macro F1 is recorded in
``artifacts/classifier_cv.json`` (``python -m chatbot.classifier --out ...``);
note it is measured on the *corpus documents* (question + answer + keywords,
categories assigned by hand in ``scripts/enrich_corpus.py``), not on user
queries, so it does not estimate query-classification accuracy. Persists to
models/classifier.pkl (gitignored) + models/classifier_labels.json.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "faq_corpus.json"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "classifier.pkl"
LABELS_PATH = MODELS_DIR / "classifier_labels.json"

CATEGORIES = ["Entry Requirements", "Fees & Funding", "Scholarships",
              "Program Details", "Accommodation", "Visa & Immigration",
              "Application Process", "English Language", "General Enquiries"]

FALLBACK = "General Enquiries"


def load_training_data(path=None) -> Tuple[List[str], List[str]]:
    """Return (texts, labels) from the enriched corpus."""
    from chatbot.retriever import load_corpus
    entries = load_corpus(path or CORPUS_PATH)
    texts = [e.document for e in entries]
    labels = [e.category if e.category in CATEGORIES else FALLBACK for e in entries]
    return texts, labels


def train(texts: List[str], labels: List[str]):
    """Fit TF-IDF + LogisticRegression pipeline (balanced classes)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from chatbot.retriever import tokenize

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, lowercase=False,
                                  ngram_range=(1, 2), token_pattern=None)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   n_jobs=1, random_state=42)),
    ])
    pipe.fit(texts, labels)
    return pipe


def cross_validate(texts: List[str], labels: List[str], k: int = 5) -> Dict[str, Any]:
    """Stratified k-fold macro F1 (per-fold + mean). Never touches test data."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    folds: List[float] = []
    for tr, va in skf.split(texts, labels):
        model = train([texts[i] for i in tr], [labels[i] for i in tr])
        pred = model.predict([texts[i] for i in va])
        folds.append(float(f1_score([labels[i] for i in va], pred,
                                    average="macro", zero_division=0)))
    return {"folds": folds, "mean_macro_f1": sum(folds) / len(folds), "k": k}


def save(model, labels: List[str]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)
    LABELS_PATH.write_text(json.dumps({"categories": labels}, indent=2),
                           encoding="utf-8")


def load():
    """Load persisted classifier; (None, []) with warning when missing."""
    if not MODEL_PATH.is_file():
        logger.warning("classifier: model missing at %s — run train first", MODEL_PATH)
        return None, []
    try:
        with open(MODEL_PATH, "rb") as fh:
            return pickle.load(fh), CATEGORIES
    except (OSError, ValueError, EOFError) as exc:
        logger.warning("classifier: model unreadable at %s (%s)", MODEL_PATH, exc)
        return None, []


def predict(query: str, model=None) -> str:
    """Predict category; gibberish/empty -> FALLBACK (never raises)."""
    if not query or not query.strip():
        return FALLBACK
    mdl = model
    if mdl is None:
        mdl, _ = load()
    if mdl is None:
        return FALLBACK
    try:
        label = str(mdl.predict([query])[0])
    except ValueError:
        return FALLBACK
    return label if label in CATEGORIES else FALLBACK


def cv_report(texts, labels, k=5):
    """Per-fold diagnostic: support, per-class F1, macro F1. No model change."""
    from collections import Counter
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for i, (tr, va) in enumerate(skf.split(texts, labels)):
        model = train([texts[j] for j in tr], [labels[j] for j in tr])
        y_va = [labels[j] for j in va]
        pred = list(model.predict([texts[j] for j in va]))
        labs = sorted(set(labels))
        f1s = f1_score(y_va, pred, labels=labs, average=None, zero_division=0)
        sup = Counter(y_va)
        macro = float(sum(f1s) / len(f1s))
        print(f"fold {i}: n_val={len(va)} macro_f1={macro:.4f}")
        for lab, f1 in zip(labs, f1s):
            print(f"  {lab}: support={sup.get(lab, 0)} f1={float(f1):.4f}")
        folds.append({"fold": i, "n_val": len(va),
                      "support": {lab: int(sup.get(lab, 0)) for lab in labs},
                      "per_class_f1": {lab: float(f1) for lab, f1 in zip(labs, f1s)},
                      "macro_f1": macro})
    return {"k": k, "folds": folds}


def cv_summary(texts, labels, k=5):
    """``cv_report`` plus mean/std and provenance, as a JSON-ready dict."""
    import statistics
    out = cv_report(texts, labels, k=k)
    macros = [f["macro_f1"] for f in out["folds"]]
    return {
        "k": k,
        "n_documents": len(texts),
        "n_categories": len(set(labels)),
        "mean_macro_f1": sum(macros) / len(macros),
        "std_macro_f1": statistics.pstdev(macros) if len(macros) > 1 else 0.0,
        "folds": out["folds"],
        "note": ("Stratified CV over the corpus documents (question + answer + "
                 "keywords) with hand-assigned categories; NOT a measurement on "
                 "user queries."),
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Question-type classifier")
    ap.add_argument("--cv-verbose", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the CV summary (per-fold + mean) as JSON")
    args = ap.parse_args(argv)
    texts, labels = load_training_data()
    summary = cv_summary(texts, labels, k=args.k)
    if not args.cv_verbose:
        print(f"mean macro F1: {summary['mean_macro_f1']:.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
