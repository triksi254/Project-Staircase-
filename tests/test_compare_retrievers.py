"""Tests for evaluation.compare_retrievers (paired TF-IDF vs SBERT)."""
import pytest

from evaluation.compare_retrievers import (_paired_detail, _summarise_outcomes,
                                          compare_retrievers)
from chatbot.embeddings import available


def _row(query, gold_id, rank, raw=None, score=0.5):
    return {"query": query, "gold_id": gold_id, "category": "Fees & Funding",
            "expected_abstain": gold_id is None, "rank": rank,
            "raw_cosine": raw, "top1_score": score, "predicted_abstain": False,
            "correct": rank == 1, "recall_at_3": bool(rank and rank <= 3),
            "recall_at_5": bool(rank and rank <= 5), "false_abstention": False}


def test_paired_detail_classifies_wins_losses_ties():
    a = [_row("q1", 1, 1), _row("q2", 2, 3), _row("q3", 3, 2),
         _row("q4", 4, None)]
    b = [_row("q1", 1, 1), _row("q2", 2, 1), _row("q3", 3, 2),
         _row("q4", 4, 5)]
    detail = _paired_detail(a, b)
    assert [d["outcome"] for d in detail] == [
        "tie",        # both rank 1
        "sbert_win",  # 3 -> 1
        "tie",        # 2 == 2
        "sbert_win",  # not retrieved -> 5
    ]
    assert detail[1]["rank1_flip"] is True
    assert detail[0]["rank1_flip"] is False


def test_paired_detail_counts_tfidf_wins():
    a = [_row("q1", 1, 1), _row("q2", 2, 2)]
    b = [_row("q1", 1, 4), _row("q2", 2, None)]
    detail = _paired_detail(a, b)
    assert [d["outcome"] for d in detail] == ["tfidf_win", "tfidf_win"]
    counts = _summarise_outcomes(detail)
    assert counts["tfidf_win"] == 2 and counts["sbert_win"] == 0
    assert counts["rank1_flips"] == 1  # q1 only


def test_outcome_counts_sum_to_n():
    a = [_row("q%d" % i, i, i) for i in range(1, 5)]
    b = [_row("q%d" % i, i, 5 - i) for i in range(1, 5)]
    counts = _summarise_outcomes(_paired_detail(a, b))
    assert counts["sbert_win"] + counts["tfidf_win"] + counts["tie"] == counts["n"]


def test_paired_detail_rejects_row_order_divergence():
    a = [_row("q1", 1, 1), _row("q2", 2, 1)]
    b = [_row("q2", 2, 1), _row("q1", 1, 1)]
    with pytest.raises(AssertionError):
        _paired_detail(a, b)


@pytest.mark.skipif(not available(),
                    reason="sentence-transformers not installed")
def test_compare_produces_expected_keys_and_pairs_queries():
    out = compare_retrievers(thresholds=(0.40, 0.60), n_bootstrap=200)
    for key in ("gold", "n_queries", "n_answerable", "summary",
                "delta_sbert_minus_tfidf", "bootstrap", "outcomes",
                "per_query", "sweep"):
        assert key in out, "missing key %s" % key
    assert set(out["summary"]) == {"tfidf", "sbert"}
    assert set(out["bootstrap"]) == {"rank_1", "recall_at_3", "recall_at_5"}
    assert out["bootstrap"]["rank_1"]["n_bootstrap"] == 200
    # pairing: one detail row per answerable query, in gold order
    assert len(out["per_query"]) == out["n_answerable"]
    assert all(r["gold_id"] is not None for r in out["per_query"])
    counts = out["outcomes"]
    assert counts["sbert_win"] + counts["tfidf_win"] + counts["tie"] == counts["n"]
    # the bootstrap delta must equal the summary delta it bounds
    staged = (out["summary"]["sbert"]["rank_1_accuracy"]
              - out["summary"]["tfidf"]["rank_1_accuracy"])
    assert abs(out["bootstrap"]["rank_1"]["delta"] - staged) <= 0.001
    # both sweeps present and gate-invariant on retrieval metrics
    for kind in ("tfidf", "sbert"):
        rows = out["sweep"][kind]
        assert [r["threshold"] for r in rows] == [0.40, 0.60]
        assert len({r["rank_1_accuracy"] for r in rows}) == 1