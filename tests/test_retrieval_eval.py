"""Tests for evaluation.retrieval_eval (H1 gold-set scoring)."""
import json

from evaluation.retrieval_eval import (DEFAULT_GOLD, SWEEP_THRESHOLDS, evaluate,
                                       load_gold, threshold_sweep)


class _StubEntry:
    __slots__ = ("index", "question")

    def __init__(self, index, question=""):
        self.index = index
        self.question = question


class _StubRetriever:
    """Returns a fixed index order per query at a fixed top-1 score.

    Exposes no ``_vectorizer``/``_matrix``, so ``raw_cosine`` is None - which
    also checks the metrics treat a missing cosine as missing, not as zero.
    """

    def __init__(self, order=None, score=0.9):
        self.order = dict(order or {})
        self.score = float(score)
        self.entries = []

    def search(self, query, top_k=5):
        idxs = self.order.get(query)
        if idxs is None:
            return []
        out = []
        for pos, idx in enumerate(idxs[:max(1, top_k)]):
            out.append((_StubEntry(idx),
                        max(0.0, round(self.score - 0.1 * pos, 4))))
        return out


GOLD = [
    {"query": "q1", "gold_id": 3, "category": "Entry Requirements",
     "expected_abstain": False},
    {"query": "q2", "gold_id": 7, "category": "Fees & Funding",
     "expected_abstain": False},
    {"query": "q3", "gold_id": None, "category": "General Enquiries",
     "expected_abstain": True},
]

EXPECTED_KEYS = {
    "n_queries", "n_answerable", "n_abstain_expected", "rank_1_accuracy",
    "mrr", "recall_at_3", "recall_at_5", "mean_top1_score", "std_top1_score",
    "mean_top1_score_answerable", "mean_raw_cosine", "std_raw_cosine",
    "mean_raw_cosine_answerable", "abstention_precision", "abstention_recall",
    "false_abstention_rate", "n_predicted_abstain", "gold_ids_missing",
    "per_category", "per_query", "config",
}


def test_eval_returns_expected_keys():
    res = evaluate(_StubRetriever({"q1": [3], "q2": [7]}), GOLD)
    assert set(res) == EXPECTED_KEYS
    assert res["n_queries"] == 3
    assert res["n_answerable"] == 2
    assert res["n_abstain_expected"] == 1
    assert res["config"]["threshold"] == 0.60
    assert all(r["raw_cosine"] is None for r in res["per_query"])
    assert res["mean_raw_cosine"] == 0.0


def test_eval_on_stub_retriever_perfect_rank1():
    res = evaluate(_StubRetriever({"q1": [3], "q2": [7]}), GOLD)
    assert res["rank_1_accuracy"] == 1.0
    assert res["mrr"] == 1.0
    assert res["recall_at_3"] == 1.0
    assert res["recall_at_5"] == 1.0
    assert all(r["correct"] for r in res["per_query"]
               if r["gold_id"] is not None)


def test_eval_on_stub_retriever_zero_rank1():
    res = evaluate(_StubRetriever({"q1": [9, 3], "q2": [9, 7]}), GOLD)
    assert res["rank_1_accuracy"] == 0.0
    assert res["recall_at_3"] == 1.0
    assert res["mrr"] == 0.5


def test_abstain_precision_computed_correctly():
    """Always-abstain stub: recall 1.0, precision = TP / all-abstentions = 1/3.

    Precision is over ALL queries, so it cannot collapse into recall.
    """
    stub = _StubRetriever({"q1": [3], "q2": [7]}, score=0.01)
    res = evaluate(stub, GOLD, threshold=0.60)
    assert res["n_predicted_abstain"] == 3
    assert res["abstention_recall"] == 1.0
    assert res["abstention_precision"] == round(1 / 3, 4)
    assert res["false_abstention_rate"] == 1.0
    assert res["abstention_precision"] != res["abstention_recall"]


def test_no_false_abstention_when_gate_is_low():
    res = evaluate(_StubRetriever({"q1": [3], "q2": [7]}, score=0.9), GOLD,
                   threshold=0.20)
    assert res["false_abstention_rate"] == 0.0
    assert res["abstention_recall"] == 1.0
    assert res["abstention_precision"] == 1.0


def test_per_category_keys_match_input_categories():
    res = evaluate(_StubRetriever({"q1": [3], "q2": [7]}), GOLD)
    assert set(res["per_category"]) == {g["category"] for g in GOLD}
    assert res["per_category"]["General Enquiries"]["n_answerable"] == 0
    assert res["per_category"]["General Enquiries"]["rank_1"] == 0.0
    assert res["per_category"]["Entry Requirements"]["thin"] is True


def test_threshold_sweep_one_row_per_gate():
    rows = threshold_sweep(_StubRetriever({"q1": [3], "q2": [7]}, score=0.45),
                           GOLD)
    assert [r["threshold"] for r in rows] == list(SWEEP_THRESHOLDS)
    assert {r["rank_1_accuracy"] for r in rows} == {1.0}
    assert len({r["abstention_precision"] for r in rows}) > 1
    assert len({r["n_predicted_abstain"] for r in rows}) > 1


def test_load_gold_rejects_malformed_entries(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"query": "q", "gold_id": None,
                                "expected_abstain": False}]), encoding="utf-8")
    try:
        load_gold(bad)
    except ValueError as exc:
        assert "expected_abstain" in str(exc)
    else:
        raise AssertionError("null gold_id without abstain must be rejected")


def test_real_gold_set_loads_and_is_consistent():
    gold = load_gold(DEFAULT_GOLD)
    assert len(gold) == 39
    assert sum(1 for g in gold if g["expected_abstain"]) == 16
    assert sum(1 for g in gold if g["gold_id"] is not None) == 23