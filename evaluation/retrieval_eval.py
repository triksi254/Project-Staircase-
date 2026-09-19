"""H1 retrieval evaluation against a hand-labelled gold set.

Gold format (``evaluation/gold_queries*.json``): a JSON list of objects with
``query``, ``gold_id``, ``category`` and ``expected_abstain``. ``gold_id`` is
the **raw JSON index** of the target FAQ (``FaqEntry.index``), or ``null``
when the corpus genuinely cannot answer the question (an abstention).

Two distinct score notions are reported, because conflating them would
overstate H1:

* ``top1_score`` - what ``Retriever.search`` returns: TF-IDF cosine plus a
  0.05 keyword bonus per matched keyword, capped at 1.0. This is the value
  ``chatbot.responder`` compares against ``min_confidence``, so it is what
  actually drives the abstention decision here.
* ``raw_cosine`` - the pure top-1 TF-IDF cosine, no keyword bonus. H1 is
  stated on this, per the proposal wording ("cosine similarity scores
  greater than 0.80").
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = PROJECT_ROOT / "evaluation" / "gold_queries.json"
DEFAULT_THRESHOLD = 0.60
SWEEP_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)
TOP_K = 5


def load_gold(path) -> List[Dict[str, Any]]:
    """Load and validate a gold set; ValueError on a malformed entry."""
    fp = Path(path)
    raw = json.loads(fp.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("gold set at %s must be a non-empty list" % fp)
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not str(item.get("query", "")).strip():
            raise ValueError("gold entry %d in %s needs a non-empty query" % (i, fp))
        gold_id = item.get("gold_id")
        if gold_id is not None and (isinstance(gold_id, bool)
                                    or not isinstance(gold_id, int)):
            raise ValueError("gold entry %d in %s: gold_id must be int or null"
                             % (i, fp))
        abstain = bool(item.get("expected_abstain", gold_id is None))
        if gold_id is not None and abstain:
            raise ValueError("gold entry %d in %s: expected_abstain=true "
                             "requires gold_id=null" % (i, fp))
        if gold_id is None and not abstain:
            raise ValueError("gold entry %d in %s: gold_id=null requires "
                             "expected_abstain=true" % (i, fp))
        out.append({"query": str(item["query"]).strip(), "gold_id": gold_id,
                    "category": str(item.get("category", "General Enquiries")),
                    "expected_abstain": abstain, "gold_set_index": i})
    return out


def raw_cosine(retriever, query: str) -> Optional[float]:
    """Pure top-1 TF-IDF cosine with no keyword bonus; None if unavailable.

    None when the retriever has no fitted vectorizer (keyword fallback active,
    or an injected stub), so callers must treat it as missing, not zero.
    """
    vectorizer = getattr(retriever, "_vectorizer", None)
    matrix = getattr(retriever, "_matrix", None)
    if vectorizer is None or matrix is None or not str(query).strip():
        return None
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    except (ValueError, ImportError):
        return None
    if not len(sims):
        return None
    return round(float(max(0.0, min(1.0, max(sims)))), 4)


def _mean(values) -> float:
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _std(values) -> float:
    vals = [v for v in values if v is not None]
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def _r4(value: float) -> float:
    return round(float(value), 4)


def build_retriever(corpus_path=None):
    """Build a live Retriever over the corpus (default data/faq_corpus.json)."""
    from chatbot.retriever import Retriever, load_corpus
    return Retriever(load_corpus(corpus_path) if corpus_path else load_corpus())


def evaluate(retriever, gold_queries: List[Dict[str, Any]],
             threshold: float = DEFAULT_THRESHOLD,
             top_k: int = TOP_K) -> Dict[str, Any]:
    """Score ``retriever`` on ``gold_queries`` at a single abstention gate.

    Ranks are matched on ``FaqEntry.index`` (stable under corpus reduction),
    never on list position. ``abstention_precision`` is computed over ALL
    queries - the denominator is every query the retriever abstained on.
    Restricting it to the abstain-expected subset would make precision
    identically equal to recall and measure nothing.
    ``false_abstention_rate`` is the operationally important error: the share
    of answerable queries whose (correct) answer the gate withheld.
    """
    per_query: List[Dict[str, Any]] = []
    for item in gold_queries:
        query, gold_id = item["query"], item["gold_id"]
        hits = retriever.search(query, top_k=top_k)
        top1_score = round(float(hits[0][1]), 4) if hits else 0.0
        rank = None
        if gold_id is not None:
            for i, (entry, _score) in enumerate(hits, 1):
                if getattr(entry, "index", None) == gold_id:
                    rank = i
                    break
        predicted_abstain = (not hits) or (top1_score < threshold)
        expected_abstain = bool(item["expected_abstain"])
        per_query.append({
            "query": query, "gold_id": gold_id, "category": item["category"],
            "expected_abstain": expected_abstain,
            "top1_score": top1_score,
            "raw_cosine": raw_cosine(retriever, query),
            "rank": rank,
            "predicted_abstain": predicted_abstain,
            "correct": bool(gold_id is not None and rank == 1),
            "recall_at_3": bool(rank is not None and rank <= 3),
            "recall_at_5": bool(rank is not None and rank <= 5),
            "false_abstention": bool(not expected_abstain and predicted_abstain),
        })

    answerable = [r for r in per_query if r["gold_id"] is not None]
    abstain_exp = [r for r in per_query if r["expected_abstain"]]
    predicted = [r for r in per_query if r["predicted_abstain"]]
    true_pos = [r for r in predicted if r["expected_abstain"]]

    per_category: Dict[str, Dict[str, Any]] = {}
    for cat in sorted({r["category"] for r in per_query}):
        rows = [r for r in per_query if r["category"] == cat]
        ans = [r for r in rows if r["gold_id"] is not None]
        per_category[cat] = {
            "n": len(rows), "n_answerable": len(ans),
            "rank_1": _r4(_mean([1.0 if r["rank"] == 1 else 0.0 for r in ans]))
            if ans else 0.0,
            "recall_at_3": _r4(_mean([1.0 if r["recall_at_3"] else 0.0
                                      for r in ans])) if ans else 0.0,
            "mrr": _r4(_mean([1.0 / r["rank"] if r["rank"] else 0.0
                              for r in ans])) if ans else 0.0,
            "thin": len(ans) < 5,
        }

    entries = getattr(retriever, "entries", [])
    known = {getattr(e, "index", None) for e in entries}
    missing = sorted({r["gold_id"] for r in answerable
                      if r["gold_id"] not in known}) if entries else []

    return {
        "n_queries": len(per_query),
        "n_answerable": len(answerable),
        "n_abstain_expected": len(abstain_exp),
        "rank_1_accuracy": _r4(_mean([1.0 if r["rank"] == 1 else 0.0
                                      for r in answerable])),
        "mrr": _r4(_mean([1.0 / r["rank"] if r["rank"] else 0.0
                          for r in answerable])),
        "recall_at_3": _r4(_mean([1.0 if r["recall_at_3"] else 0.0
                                  for r in answerable])),
        "recall_at_5": _r4(_mean([1.0 if r["recall_at_5"] else 0.0
                                  for r in answerable])),
        "mean_top1_score": _r4(_mean([r["top1_score"] for r in per_query])),
        "std_top1_score": _r4(_std([r["top1_score"] for r in per_query])),
        "mean_top1_score_answerable": _r4(_mean([r["top1_score"]
                                                 for r in answerable])),
        "mean_raw_cosine": _r4(_mean([r["raw_cosine"] for r in per_query])),
        "std_raw_cosine": _r4(_std([r["raw_cosine"] for r in per_query])),
        "mean_raw_cosine_answerable": _r4(_mean([r["raw_cosine"]
                                                 for r in answerable])),
        "abstention_precision": _r4(len(true_pos) / len(predicted)
                                    if predicted else 0.0),
        "abstention_recall": _r4(len(true_pos) / len(abstain_exp)
                                 if abstain_exp else 0.0),
        "false_abstention_rate": _r4(len([r for r in answerable
                                          if r["false_abstention"]])
                                     / len(answerable) if answerable else 0.0),
        "n_predicted_abstain": len(predicted),
        "gold_ids_missing": missing,
        "per_category": per_category,
        "per_query": per_query,
        "config": {"threshold": threshold, "top_k": top_k},
    }


def threshold_sweep(retriever, gold_queries: List[Dict[str, Any]],
                    thresholds=SWEEP_THRESHOLDS,
                    top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Run ``evaluate`` once per gate and tabulate the abstention trade-off.

    Retrieval-side metrics (rank-1, recall@k, cosines) are invariant across
    gates; only the abstention block moves. This measures one fixed model at
    several gates - it is not threshold tuning.
    """
    rows = []
    for t in thresholds:
        res = evaluate(retriever, gold_queries, threshold=float(t), top_k=top_k)
        rows.append({
            "threshold": float(t),
            "rank_1_accuracy": res["rank_1_accuracy"],
            "recall_at_3": res["recall_at_3"],
            "recall_at_5": res["recall_at_5"],
            "abstention_precision": res["abstention_precision"],
            "abstention_recall": res["abstention_recall"],
            "false_abstention_rate": res["false_abstention_rate"],
            "n_predicted_abstain": res["n_predicted_abstain"],
        })
    return rows


def format_sweep(rows: List[Dict[str, Any]]) -> str:
    """Render the threshold sweep as an aligned table."""
    head = ("  thresh  rank-1  R@3    R@5    abst-P  abst-R  false-abst  n_abstain")
    lines = [head]
    for r in rows:
        lines.append(
            "  %6.2f  %6.2f  %5.2f  %5.2f  %6.2f  %6.2f  %10.2f  %9d" % (
                r["threshold"], r["rank_1_accuracy"], r["recall_at_3"],
                r["recall_at_5"], r["abstention_precision"],
                r["abstention_recall"], r["false_abstention_rate"],
                r["n_predicted_abstain"]))
    return "\n".join(lines)


def format_human(metrics: Dict[str, Any], gold_path: str,
                 sweep: Optional[List[Dict[str, Any]]] = None) -> str:
    """Render the H1 report; raw cosine is reported beside the retriever score."""
    lines = [
        "Gold set: %s" % gold_path,
        "Queries:  %d (%d answerable, %d abstain-expected)"
        % (metrics["n_queries"], metrics["n_answerable"],
           metrics["n_abstain_expected"]),
        "Gate:     top1_score < %.2f => abstain" % metrics["config"]["threshold"],
        "",
        "Rank-1 accuracy:     %.2f" % metrics["rank_1_accuracy"],
        "MRR:                 %.2f" % metrics["mrr"],
        "Recall@3:            %.2f" % metrics["recall_at_3"],
        "Recall@5:            %.2f" % metrics["recall_at_5"],
        "",
        "  H1 must be stated on raw cosine (no keyword boost):",
        "Mean raw cosine:     %.2f (+/- %.2f)  all queries"
        % (metrics["mean_raw_cosine"], metrics["std_raw_cosine"]),
        "Mean raw cosine:     %.2f  answerable only"
        % metrics["mean_raw_cosine_answerable"],
        "  Retriever score (cosine + keyword bonus; drives the gate):",
        "Mean top-1 score:    %.2f (+/- %.2f)"
        % (metrics["mean_top1_score"], metrics["std_top1_score"]),
        "Mean top-1 score:    %.2f  answerable only"
        % metrics["mean_top1_score_answerable"],
        "",
        "Abstention precision: %.2f   (over %d predicted abstentions)"
        % (metrics["abstention_precision"], metrics["n_predicted_abstain"]),
        "Abstention recall:    %.2f" % metrics["abstention_recall"],
        "False abstention:     %.2f   (answerable queries withheld)"
        % metrics["false_abstention_rate"],
    ]
    if metrics["gold_ids_missing"]:
        lines.append("WARNING: gold ids absent from corpus: %s"
                     % metrics["gold_ids_missing"])
    lines += ["", "Per category:"]
    for cat, st in metrics["per_category"].items():
        flag = "  (thin)" if st["thin"] else ""
        lines.append("  %-22s n=%d/%d  rank-1 %.2f  recall@3 %.2f%s"
                     % (cat, st["n_answerable"], st["n"], st["rank_1"],
                        st["recall_at_3"], flag))
    if sweep:
        lines += ["", "Threshold sweep (same model, several gates):",
                  format_sweep(sweep)]
    return "\n".join(lines)


def assert_index_contract(retriever) -> None:
    """Assert ``FaqEntry.index`` equals its raw JSON position.

    Gold ``gold_id`` values address the corpus by raw JSON index. ``load_corpus``
    skips entries lacking a question or answer, so index == position holds only
    while every entry loads cleanly. Fail loudly rather than silently scoring
    against shifted targets if that ever stops being true.
    """
    for pos, entry in enumerate(getattr(retriever, "entries", [])):
        idx = getattr(entry, "index", None)
        if idx != pos:
            raise AssertionError(
                "gold_id addressing broken: entry at position %d carries index "
                "%r. Gold ids address the raw JSON corpus, so load_corpus must "
                "not skip entries." % (pos, idx))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H1 retrieval evaluation")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--sweep", action="store_true",
                    help="tabulate abstention metrics across several gates")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None,
                    help="also write the JSON payload here as UTF-8 "
                         "(preferred over shell redirection)")
    args = ap.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be in [0, 1]")
    gold = load_gold(args.gold)
    retriever = build_retriever()
    assert_index_contract(retriever)
    metrics = evaluate(retriever, gold, threshold=args.threshold,
                       top_k=args.top_k)
    sweep = (threshold_sweep(retriever, gold, top_k=args.top_k)
             if args.sweep else None)
    payload: Dict[str, Any] = {"gold": args.gold, "metrics": metrics}
    if sweep is not None:
        payload["sweep"] = sweep
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n",
                                  encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_human(metrics, args.gold, sweep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())