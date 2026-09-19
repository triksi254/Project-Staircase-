"""Paired TF-IDF vs Sentence-BERT comparison on identical gold sets.

Both retrievers are scored by the *same* code (``retrieval_eval.evaluate``) on
the *same* queries, so per-query outcomes are paired and the difference is
bootstrapped directly rather than compared as two independent means.

Reports three things the headline means hide:

* win / loss / tie counts on per-query rank - where SBERT helps or hurts, not
  just whether the average moved;
* rank-1 flips - queries where exactly one retriever puts the gold first;
* the raw-cosine distributions side by side, since that is the quantity H1's
  ">0.80" criterion is stated on and the two models are not on a common scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.adaptation_loop import paired_bootstrap
from evaluation.retrieval_eval import (DEFAULT_GOLD, SWEEP_THRESHOLDS, TOP_K,
                                       assert_index_contract, build_retriever,
                                       evaluate, load_gold, threshold_sweep)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KINDS = ("tfidf", "sbert")


def _rank_key(row) -> float:
    """Rank as a sortable number; not-retrieved sorts last."""
    return float(row["rank"]) if row["rank"] is not None else float("inf")


def _paired_detail(tfidf_rows: List[Dict[str, Any]],
                   sbert_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-query rank comparison, in gold order."""
    detail = []
    for a, b in zip(tfidf_rows, sbert_rows):
        if a["query"] != b["query"]:
            raise AssertionError("row order diverged between retrievers")
        ra, rb = _rank_key(a), _rank_key(b)
        if ra == rb:
            outcome = "tie"
        elif rb < ra:
            outcome = "sbert_win"
        else:
            outcome = "tfidf_win"
        detail.append({
            "query": a["query"], "gold_id": a["gold_id"],
            "category": a["category"], "expected_abstain": a["expected_abstain"],
            "tfidf_rank": a["rank"], "sbert_rank": b["rank"],
            "tfidf_raw_cosine": a["raw_cosine"],
            "sbert_raw_cosine": b["raw_cosine"],
            "tfidf_score": a["top1_score"], "sbert_score": b["top1_score"],
            "outcome": outcome,
            "rank1_flip": (a["rank"] == 1) != (b["rank"] == 1),
        })
    return detail


def _summarise_outcomes(detail: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"sbert_win": 0, "tfidf_win": 0, "tie": 0}
    for row in detail:
        counts[row["outcome"]] += 1
    counts["rank1_flips"] = sum(1 for row in detail if row["rank1_flip"])
    counts["n"] = len(detail)
    return counts


def compare_retrievers(gold_path=None, top_k: int = TOP_K,
                       thresholds=SWEEP_THRESHOLDS, n_bootstrap: int = 1000,
                       seed: int = 42) -> Dict[str, Any]:
    """Score both retrievers on one gold set and pair the per-query results."""
    gold_path = str(gold_path or DEFAULT_GOLD)
    gold = load_gold(gold_path)
    per_kind: Dict[str, Dict[str, Any]] = {}
    for kind in KINDS:
        retriever = build_retriever(kind)
        assert_index_contract(retriever)
        metrics = evaluate(retriever, gold, threshold=thresholds[-1],
                           top_k=top_k)
        per_kind[kind] = {
            "metrics": metrics,
            "sweep": threshold_sweep(retriever, gold, thresholds=thresholds,
                                     top_k=top_k),
        }

    # Paired comparison is defined on answerable queries only (rank metrics are
    # undefined for abstain-expected rows), matching how the summaries report.
    rows_a = [r for r in per_kind["tfidf"]["metrics"]["per_query"]
              if r["gold_id"] is not None]
    rows_b = [r for r in per_kind["sbert"]["metrics"]["per_query"]
              if r["gold_id"] is not None]
    bootstrap = {}
    for key, label in (("correct", "rank_1"), ("recall_at_3", "recall_at_3"),
                       ("recall_at_5", "recall_at_5")):
        bootstrap[label] = paired_bootstrap(rows_a, rows_b, key,
                                            n_bootstrap=n_bootstrap, seed=seed)

    detail = _paired_detail(rows_a, rows_b)
    summary = {}
    for kind in KINDS:
        m = per_kind[kind]["metrics"]
        summary[kind] = {
            "rank_1_accuracy": m["rank_1_accuracy"], "mrr": m["mrr"],
            "recall_at_3": m["recall_at_3"], "recall_at_5": m["recall_at_5"],
            "mean_raw_cosine": m["mean_raw_cosine"],
            "std_raw_cosine": m["std_raw_cosine"],
            "mean_raw_cosine_answerable": m["mean_raw_cosine_answerable"],
            "mean_top1_score_answerable": m["mean_top1_score_answerable"],
            "false_abstention_rate": m["false_abstention_rate"],
        }
    delta = {k: round(summary["sbert"][k] - summary["tfidf"][k], 4)
             for k in ("rank_1_accuracy", "mrr", "recall_at_3", "recall_at_5",
                       "mean_raw_cosine", "mean_raw_cosine_answerable")}
    return {
        "gold": gold_path,
        "n_queries": len(gold),
        "n_answerable": len(rows_a),
        "config": {"top_k": top_k, "thresholds": list(thresholds),
                   "n_bootstrap": n_bootstrap, "seed": seed},
        "summary": summary,
        "delta_sbert_minus_tfidf": delta,
        "bootstrap": bootstrap,
        "outcomes": _summarise_outcomes(detail),
        "per_query": detail,
        "sweep": {k: per_kind[k]["sweep"] for k in KINDS},
    }


def format_human(out: Dict[str, Any]) -> str:
    """Render the paired comparison (ASCII only, for any console codepage)."""
    a, b = out["summary"]["tfidf"], out["summary"]["sbert"]
    d = out["delta_sbert_minus_tfidf"]
    lines = [
        "Paired retrieval comparison: TF-IDF vs Sentence-BERT",
        "===================================================",
        "Gold set:  %s" % out["gold"],
        "Queries:   %d (%d answerable)" % (out["n_queries"], out["n_answerable"]),
        "",
        "Metric                    TF-IDF    SBERT     Delta",
        "-----------------------------------------------------",
        "Rank-1 accuracy           %6.2f    %6.2f    %+6.2f"
        % (a["rank_1_accuracy"], b["rank_1_accuracy"], d["rank_1_accuracy"]),
        "MRR                       %6.2f    %6.2f    %+6.2f"
        % (a["mrr"], b["mrr"], d["mrr"]),
        "Recall@3                  %6.2f    %6.2f    %+6.2f"
        % (a["recall_at_3"], b["recall_at_3"], d["recall_at_3"]),
        "Recall@5                  %6.2f    %6.2f    %+6.2f"
        % (a["recall_at_5"], b["recall_at_5"], d["recall_at_5"]),
        "Mean raw cosine (ans.)    %6.2f    %6.2f    %+6.2f"
        % (a["mean_raw_cosine_answerable"], b["mean_raw_cosine_answerable"],
           d["mean_raw_cosine_answerable"]),
        "",
        "Paired bootstrap (95% CI), sbert - tfidf:",
    ]
    for label in ("rank_1", "recall_at_3", "recall_at_5"):
        ci = out["bootstrap"][label]
        verdict = ("significant" if ci["ci_low"] > 0 or ci["ci_high"] < 0
                   else "crosses zero")
        lines.append("  %-12s %+.2f  [%+.2f, %+.2f]  %s"
                     % (label, ci["delta"], ci["ci_low"], ci["ci_high"],
                        verdict))
    o = out["outcomes"]
    lines += [
        "",
        "Per-query rank outcomes (n=%d): sbert_win %d, tfidf_win %d, tie %d"
        % (o["n"], o["sbert_win"], o["tfidf_win"], o["tie"]),
        "Rank-1 flips (exactly one retriever got it first): %d"
        % o["rank1_flips"],
    ]
    diffs = [r for r in out["per_query"] if r["outcome"] != "tie"]
    if diffs:
        lines += ["", "Queries whose rank differs (tfidf -> sbert):"]
        for r in diffs:
            fmt = lambda x: "-" if x is None else str(x)
            lines.append("  %-6s %-10s %2s -> %-2s  %s"
                         % (r["outcome"].replace("_win", "").upper(),
                            r["category"][:10], fmt(r["tfidf_rank"]),
                            fmt(r["sbert_rank"]), r["query"][:60]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Paired TF-IDF vs SBERT comparison")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated gates for the per-retriever sweeps "
                         "(default: %s)" % ",".join(str(t) for t in SWEEP_THRESHOLDS))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None,
                    help="write the JSON payload here as UTF-8")
    args = ap.parse_args(argv)
    thresholds = SWEEP_THRESHOLDS
    if args.thresholds:
        try:
            thresholds = tuple(float(x) for x in args.thresholds.split(",")
                               if x.strip())
        except ValueError:
            raise SystemExit("--thresholds must be comma-separated numbers")
        if not thresholds or any(not 0.0 <= t <= 1.0 for t in thresholds):
            raise SystemExit("--thresholds must be in [0, 1]")
    out = compare_retrievers(args.gold, top_k=args.top_k,
                             thresholds=thresholds,
                             n_bootstrap=args.n_bootstrap, seed=args.seed)
    if args.out:
        fp = Path(args.out)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(format_human(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())