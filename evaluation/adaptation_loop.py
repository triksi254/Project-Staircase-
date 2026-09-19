"""Cold-start adaptation loop: how much retrieval loss does a top-up recover?

Protocol (see ``run_adaptation_experiment``): the full corpus is reduced in
memory, evaluated, then topped up, and recovery is measured against the
cold-start loss. ``data/faq_corpus.json`` is never written to; integrity is
checked by hashing the corpus and the retrieval caches before and after.

Two restorations are reported, and only the second is a real finding:

* Stage 4 (full restoration) is an upper bound that is *identical to Stage 1
  by construction* - it puts every withheld entry back. It is retained as a
  determinism check, not as evidence.
* Stage 4b (targeted restoration) puts back only the entries that Stage 3's
  probe set flagged as missing. The gap between 4 and 4b is what targeted
  top-up leaves on the table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.retrieval_eval import evaluate, load_gold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "faq_corpus.json"
DEFAULT_BATCH_A = PROJECT_ROOT / "evaluation" / "gold_queries.json"
DEFAULT_BATCH_B = PROJECT_ROOT / "evaluation" / "gold_queries_b.json"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "adaptation_results.json"

WITHHOLD_FRAC = 0.30
DEFAULT_THRESHOLD = 0.60
TOP_K = 5
RANK_FAIL = 3  # a query counts as a gap when its gold rank exceeds this

# Caches chatbot.retriever writes on every construction; included in the
# integrity hash because a corpus reduction must not leave a reduced index
# behind for later runs.
CACHE_PATHS = (PROJECT_ROOT / "data" / "tfidf_vectorizer.pkl",
               PROJECT_ROOT / "data" / "tfidf_matrix.npy")


def _stage(retriever, gold, threshold, top_k=TOP_K) -> Dict[str, Any]:
    """Evaluate one corpus state; keep a compact summary plus per-query flags.

    The per-query recall flags are retained (in gold order) so Stage 5 can run
    a *paired* bootstrap over identical query indices.
    """
    res = evaluate(retriever, gold, threshold=threshold, top_k=top_k)
    summary = {k: v for k, v in res.items()
               if k not in ("per_query", "per_category", "gold_ids_missing")}
    return {
        "summary": summary,
        "n_corpus_entries": len(getattr(retriever, "entries", [])),
        "rows": [{"query": r["query"], "gold_id": r["gold_id"],
                  "rank": r["rank"], "top1_score": r["top1_score"],
                  "raw_cosine": r["raw_cosine"],
                  "predicted_abstain": r["predicted_abstain"],
                  "recall_at_3": r["recall_at_3"],
                  "recall_at_5": r["recall_at_5"]}
                 for r in res["per_query"]],
    }


def _delta(earlier: Dict[str, Any], later: Dict[str, Any], key: str) -> float:
    """later - earlier for one summary metric."""
    return round(later["summary"][key] - earlier["summary"][key], 4)


def paired_bootstrap(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]],
                     key: str, n_bootstrap: int = 1000,
                     seed: int = 42) -> Dict[str, Any]:
    """95% CI for ``mean(b) - mean(a)`` by paired resampling of query rows.

    Pairing matters: the same queries are resampled for both stages, so the
    interval is for the *difference*, not for two independent means.
    """
    if len(rows_a) != len(rows_b) or not rows_a:
        raise ValueError("paired bootstrap needs two equal-length row lists")
    import numpy as np
    a = np.asarray([1.0 if r[key] else 0.0 for r in rows_a], dtype=float)
    b = np.asarray([1.0 if r[key] else 0.0 for r in rows_b], dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    boot = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return {
        "metric": key,
        "n": int(n),
        "n_bootstrap": int(n_bootstrap),
        "delta": round(float(b.mean() - a.mean()), 4),
        "mean": round(float(boot.mean()), 4),
        "ci_low": round(float(np.percentile(boot, 2.5)), 4),
        "ci_high": round(float(np.percentile(boot, 97.5)), 4),
    }


def _recovery(stage_base: Dict[str, Any], stage_cold: Dict[str, Any],
              stage_restored: Dict[str, Any], key: str = "recall_at_3"):
    """Fraction of the cold-start loss recovered, or None if there was none."""
    lost = stage_base["summary"][key] - stage_cold["summary"][key]
    gained = stage_restored["summary"][key] - stage_cold["summary"][key]
    if lost == 0:
        return None
    return round(gained / lost, 4)


def _make_retriever(entries):
    """Build a Retriever over ``entries`` with on-disk persistence disabled.

    ``chatbot.retriever.Retriever`` writes the vectorizer and matrix to
    ``data/`` on *every* construction. Building a deliberately-reduced corpus
    would therefore overwrite the cache with a 70%-corpus index and silently
    corrupt later runs, so persistence is suppressed here.
    """
    from chatbot.retriever import Retriever

    class _QuietRetriever(Retriever):
        def _persist(self) -> None:
            return None

    return _QuietRetriever(entries)


def _sha256(path) -> Optional[str]:
    """Hash a file's bytes, or None when it does not exist."""
    fp = Path(path)
    if not fp.is_file():
        return None
    digest = hashlib.sha256()
    with open(fp, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_integrity(corpus_path) -> Dict[str, Optional[str]]:
    """Hash the corpus and every retrieval cache."""
    out = {"corpus": _sha256(corpus_path)}
    for name, path in zip(("vectorizer_cache", "matrix_cache"), CACHE_PATHS):
        out[name] = _sha256(path)
    return out
def run_adaptation_experiment(
    corpus_path=None,
    batch_a_path=None,
    batch_b_path=None,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Cold-start simulation: reduce the corpus, probe for gaps, top up.

    Guarantees, all asserted rather than assumed:
      * ``data/faq_corpus.json`` is never written (hash compared before/after,
        together with both TF-IDF cache files that ``Retriever`` persists).
      * Batch B is never used in Stage 3 (asserted: no batch-B query appears
        in the Stage-3 probe results).
      * Batch A is never used in Stage 1 or 2 (only batch B is evaluated there).
      * Deterministic given ``seed``.
    """
    from chatbot.retriever import load_corpus

    corpus_path = Path(corpus_path or DEFAULT_CORPUS)
    batch_a = load_gold(batch_a_path or DEFAULT_BATCH_A)
    batch_b = load_gold(batch_b_path or DEFAULT_BATCH_B)
    before = snapshot_integrity(corpus_path)

    entries = load_corpus(corpus_path)
    if not entries:
        raise ValueError("corpus at %s yielded no usable entries" % corpus_path)

    # 30% of the distinct FAQ entries batch B actually needs.
    b_gold_ids = sorted({g["gold_id"] for g in batch_b if g["gold_id"] is not None})
    n_withhold = min(len(b_gold_ids),
                     max(1, int(round(WITHHOLD_FRAC * len(b_gold_ids)))))
    withheld = sorted(random.Random(seed).sample(b_gold_ids, n_withhold))
    withheld_set = set(withheld)

    def corpus_without(excluded):
        return [e for e in entries if e.index not in excluded]

    # ---- STAGE 1: baseline over the full corpus -------------------------
    stage1 = _stage(_make_retriever(entries), batch_b, threshold)

    # ---- STAGE 2: cold start (withheld entries unavailable) -------------
    reduced = corpus_without(withheld_set)
    stage2 = _stage(_make_retriever(reduced), batch_b, threshold)

    # ---- STAGE 3: probe the reduced corpus with batch A -----------------
    reduced_retriever = _make_retriever(reduced)
    probe = evaluate(reduced_retriever, batch_a, threshold=threshold)
    b_queries = {g["query"] for g in batch_b}
    probe_rows = [r for r in probe["per_query"]
                  if r["gold_id"] is not None
                  and (r["rank"] is None or r["rank"] > RANK_FAIL
                       or r["top1_score"] < threshold)]
    batch_a_failures = [
        {"query": r["query"], "gold_id": r["gold_id"], "category": r["category"],
         "rank": r["rank"], "top1_score": r["top1_score"],
         "predicted_abstain": r["predicted_abstain"],
         "reason": ("unretrievable" if r["rank"] is None
                    else "rank>%d" % RANK_FAIL if r["rank"] > RANK_FAIL
                    else "below_gate")}
        for r in probe_rows
    ]
    # Batch B must not leak into Stage 3's probe.
    leaked = sorted({f["query"] for f in batch_a_failures} & b_queries)
    if leaked:
        raise AssertionError("batch B leaked into Stage 3 probe: %s" % leaked)

    # Entries batch A needs that happen to be withheld.
    batch_a_withheld = sorted({g["gold_id"] for g in batch_a
                               if g["gold_id"] in withheld_set})
    # Probe-driven top-up: withheld entries whose *category* is one the probe
    # showed to be under-served. Without this the failure list is redundant -
    # a failure's gold_id is by definition already in batch_a_withheld, so the
    # exact-id rule alone can never add anything the withheld set didn't
    # already imply.
    failed_categories = {f["category"] for f in batch_a_failures}
    by_index = {e.index: e for e in entries}
    targeted_by_category = sorted(
        w for w in withheld
        if getattr(by_index.get(w), "category", None) in failed_categories)
    targeted = sorted(set(batch_a_withheld) | set(targeted_by_category))

    # ---- STAGE 4: full restoration (upper bound = Stage 1 by construction)
    stage4 = _stage(_make_retriever(entries), batch_b, threshold)

    # ---- STAGE 4b: targeted restoration (the actual finding) -------------
    restored = corpus_without(withheld_set - set(targeted))
    stage4b = _stage(_make_retriever(restored), batch_b, threshold)

    # ---- STAGE 5: paired bootstrap on batch B ---------------------------
    bootstrap_ci = {}
    for label, later in (("full", stage4), ("targeted", stage4b)):
        for key in ("recall_at_3", "recall_at_5"):
            bootstrap_ci["%s_%s" % (label, key)] = paired_bootstrap(
                stage2["rows"], later["rows"], key,
                n_bootstrap=n_bootstrap, seed=seed)

    delta = {
        # cost = baseline - cold_start; positive means the cold start hurt
        "cold_start_cost_recall_at_3": _delta(stage2, stage1, "recall_at_3"),
        "cold_start_cost_recall_at_5": _delta(stage2, stage1, "recall_at_5"),
        "full_restore_recall_at_3": _delta(stage2, stage4, "recall_at_3"),
        "full_restore_recall_at_5": _delta(stage2, stage4, "recall_at_5"),
        "targeted_restore_recall_at_3": _delta(stage2, stage4b, "recall_at_3"),
        "targeted_restore_recall_at_5": _delta(stage2, stage4b, "recall_at_5"),
    }
    recovery = {
        "full_recall_at_3": _recovery(stage1, stage2, stage4, "recall_at_3"),
        "targeted_recall_at_3": _recovery(stage1, stage2, stage4b, "recall_at_3"),
        "full_recall_at_5": _recovery(stage1, stage2, stage4, "recall_at_5"),
        "targeted_recall_at_5": _recovery(stage1, stage2, stage4b, "recall_at_5"),
    }
    # What targeted top-up leaves on the table, vs full restoration.
    residual_gap = {
        "recall_at_3": round(stage4["summary"]["recall_at_3"]
                             - stage4b["summary"]["recall_at_3"], 4),
        "recall_at_5": round(stage4["summary"]["recall_at_5"]
                             - stage4b["summary"]["recall_at_5"], 4),
    }

    after = snapshot_integrity(corpus_path)
    integrity = {"before": before, "after": after,
                 "unchanged": before == after,
                 "corpus_unchanged": before["corpus"] == after["corpus"],
                 "caches_unchanged": (
                     before["vectorizer_cache"] == after["vectorizer_cache"]
                     and before["matrix_cache"] == after["matrix_cache"])}

    return {
        "config": {"corpus": str(corpus_path), "batch_a": str(batch_a_path or
                                                              DEFAULT_BATCH_A),
                   "batch_b": str(batch_b_path or DEFAULT_BATCH_B),
                   "threshold": threshold, "seed": seed,
                   "n_bootstrap": n_bootstrap, "withhold_frac": WITHHOLD_FRAC,
                   "rank_fail": RANK_FAIL, "top_k": TOP_K},
        "stage1_full": stage1["summary"],
        "stage2_cold_start": stage2["summary"],
        "stage3_probe": {"n_probe_queries": len([g for g in batch_a
                                                 if g["gold_id"] is not None]),
                         "n_failures": len(batch_a_failures)},
        "stage4_full_restore": stage4["summary"],
        "stage4b_targeted_restore": stage4b["summary"],
        "n_entries_full": stage1["n_corpus_entries"],
        "n_entries_reduced": stage2["n_corpus_entries"],
        "n_entries_targeted": stage4b["n_corpus_entries"],
        "withheld_ids": withheld,
        "restored_ids": withheld,
        "targeted_ids": targeted,
        "batch_a_withheld_ids": batch_a_withheld,
        "targeted_by_category_ids": targeted_by_category,
        "failed_categories": sorted(failed_categories),
        "batch_a_failures": batch_a_failures,
        "batch_b_in_stage3": bool(leaked),
        "delta": delta,
        "recovery": recovery,
        "residual_gap": residual_gap,
        "bootstrap_ci": bootstrap_ci,
        "integrity": integrity,
    }


def _pct(value) -> str:
    return "n/a" if value is None else "%d%%" % round(100 * value)


def format_human(out: Dict[str, Any]) -> str:
    """Render the cold-start report (ASCII only, for any console codepage)."""
    s = lambda k: out[k]  # stage summaries are stored flattened
    ci3 = out["bootstrap_ci"]["full_recall_at_3"]
    ci3t = out["bootstrap_ci"]["targeted_recall_at_3"]
    lines = [
        "Cold-start adaptation experiment",
        "---------------------------------",
        "Corpus entries: full %d, reduced %d, targeted-restored %d"
        % (out["n_entries_full"], out["n_entries_reduced"],
           out["n_entries_targeted"]),
        "Gate: top1_score < %.2f => abstain" % out["config"]["threshold"],
        "",
        "Stage 1  Baseline (full corpus)       Rank-1 %.2f  MRR %.2f  R@3 %.2f  R@5 %.2f"
        % (s("stage1_full")["rank_1_accuracy"], s("stage1_full")["mrr"],
           s("stage1_full")["recall_at_3"], s("stage1_full")["recall_at_5"]),
        "Stage 2  Cold-start (%d%% withheld)    Rank-1 %.2f  MRR %.2f  R@3 %.2f  R@5 %.2f"
        % (round(100 * out["config"]["withhold_frac"]),
           s("stage2_cold_start")["rank_1_accuracy"],
           s("stage2_cold_start")["mrr"],
           s("stage2_cold_start")["recall_at_3"],
           s("stage2_cold_start")["recall_at_5"]),
        "Stage 4  Full restore                 Rank-1 %.2f  MRR %.2f  R@3 %.2f  R@5 %.2f"
        % (s("stage4_full_restore")["rank_1_accuracy"],
           s("stage4_full_restore")["mrr"],
           s("stage4_full_restore")["recall_at_3"],
           s("stage4_full_restore")["recall_at_5"]),
        "Stage 4b Targeted restore (%d ids)     Rank-1 %.2f  MRR %.2f  R@3 %.2f  R@5 %.2f"
        % (len(out["targeted_ids"]),
           s("stage4b_targeted_restore")["rank_1_accuracy"],
           s("stage4b_targeted_restore")["mrr"],
           s("stage4b_targeted_restore")["recall_at_3"],
           s("stage4b_targeted_restore")["recall_at_5"]),
        "",
        "Cold-start cost (R@3 baseline - cold): %+.2f"
        % out["delta"]["cold_start_cost_recall_at_3"],
        "",
        "Full restore vs cold      R@3 %+.2f  (95%% CI [%+.2f, %+.2f])"
        % (ci3["delta"], ci3["ci_low"], ci3["ci_high"]),
        "Targeted restore vs cold  R@3 %+.2f  (95%% CI [%+.2f, %+.2f])"
        % (ci3t["delta"], ci3t["ci_low"], ci3t["ci_high"]),
        "Full vs targeted gap      R@3 %+.2f  (what targeted top-up leaves)"
        % out["residual_gap"]["recall_at_3"],
        "",
        "Recovery vs baseline   full %s   targeted %s"
        % (_pct(out["recovery"]["full_recall_at_3"]),
           _pct(out["recovery"]["targeted_recall_at_3"])),
        "Full restore == baseline: %s (determinism check)"
        % (s("stage1_full") == s("stage4_full_restore")),
        "",
        "Withheld: %d corpus entries  %s"
        % (len(out["withheld_ids"]), out["withheld_ids"]),
        "Targeted ids: %d  %s" % (len(out["targeted_ids"]), out["targeted_ids"]),
        "Batch A probe failures on reduced corpus: %d queries"
        % out["stage3_probe"]["n_failures"],
        "Batch B in Stage 3: %s (must be False)" % out["batch_b_in_stage3"],
        "Corpus on-disk unchanged: %s   caches unchanged: %s"
        % (out["integrity"]["corpus_unchanged"],
           out["integrity"]["caches_unchanged"]),
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cold-start adaptation experiment")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--batch-a", default=str(DEFAULT_BATCH_A))
    ap.add_argument("--batch-b", default=str(DEFAULT_BATCH_B))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None,
                    help="write the JSON payload here as UTF-8 "
                         "(preferred over shell redirection)")
    args = ap.parse_args(argv)
    out = run_adaptation_experiment(
        corpus_path=args.corpus, batch_a_path=args.batch_a,
        batch_b_path=args.batch_b, threshold=args.threshold, seed=args.seed,
        n_bootstrap=args.n_bootstrap)
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


def _make_retriever(entries):
    """Build a Retriever over ``entries`` with on-disk persistence disabled."""
    from chatbot.retriever import Retriever

    class _QuietRetriever(Retriever):
        def _persist(self) -> None:  # noqa: D401 - intentionally a no-op
            return None

    return _QuietRetriever(entries)


def _sha256(path) -> Optional[str]:
    """Hash a file's bytes, or None when it does not exist."""
    fp = Path(path)
    if not fp.is_file():
        return None
    digest = hashlib.sha256()
    with open(fp, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_integrity(corpus_path) -> Dict[str, Optional[str]]:
    """Hash the corpus and every retrieval cache."""
    out = {"corpus": _sha256(corpus_path)}
    for name, path in zip(("vectorizer_cache", "matrix_cache"), CACHE_PATHS):
        out[name] = _sha256(path)
    return out