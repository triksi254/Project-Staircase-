"""Committed artifacts must agree with the code and with each other.

Two problems motivated these checks: the tagged retrieval artifacts were
computed with a TF-IDF keyword bonus of 0.05 while the code later moved to 0.10
(unrelated commit, artifacts not regenerated), and the leakage figures quoted in
the README were never pinned to a recomputation. The tagged files are frozen
evidence and are left alone; the ``*_corrected.json`` files are the regenerated
counterparts and must record the constants they were computed with.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts"


def _load(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# retrieval / adaptation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["compare_gold_a_corrected.json",
                                  "compare_gold_b_corrected.json"])
def test_corrected_comparison_records_the_constants_it_used(name):
    from chatbot.responder import EVALUATED_GATE
    from chatbot.retriever import KEYWORD_BONUS
    cfg = _load(name)["config"]
    assert cfg["keyword_bonus"] == KEYWORD_BONUS, (
        "artifact is stale: regenerate with evaluation.compare_retrievers")
    assert cfg["summary_gate"] == EVALUATED_GATE


@pytest.mark.parametrize("gold", ["a", "b"])
def test_corrected_comparison_did_not_change_any_rank_metric(gold):
    """Only the abstention-gate metrics may move (the keyword bonus feeds them)."""
    old, new = _load("compare_gold_%s.json" % gold), _load(
        "compare_gold_%s_corrected.json" % gold)
    for kind in ("tfidf", "sbert"):
        for key in ("rank_1_accuracy", "mrr", "recall_at_3", "recall_at_5",
                    "mean_raw_cosine", "mean_raw_cosine_answerable"):
            assert old["summary"][kind][key] == new["summary"][kind][key], (kind, key)
    assert old["bootstrap"] == new["bootstrap"]
    assert old["sweep"]["sbert"] == new["sweep"]["sbert"]      # no bonus in SBERT


def test_corrected_comparison_reports_the_h1_shares():
    for gold in ("a", "b"):
        h1 = _load("compare_gold_%s_corrected.json" % gold)["summary"]["sbert"]["h1"]
        assert 0.0 <= h1["share_raw_cosine_gt_0.80"] <= h1["share_raw_cosine_gt_0.60"] <= 1.0
        assert h1["n_answerable"] in (23, 31)


def test_corrected_adaptation_flags_only_retrieval_gaps():
    from chatbot.retriever import KEYWORD_BONUS
    d = _load("adaptation_results_corrected.json")
    probe = d["stage3_probe"]
    assert d["config"]["keyword_bonus"] == KEYWORD_BONUS
    assert probe["n_failures"] < probe["n_probe_queries"]
    assert probe["n_below_gate"] >= probe["n_failures"]
    assert {f["reason"] for f in d["batch_a_failures"]} <= {"unretrievable", "rank>3"}
    # every failing category actually contains a withheld entry
    corpus = json.loads((REPO / "data" / "faq_corpus.json")
                        .read_text(encoding="utf-8"))["faqs"]
    withheld_cats = {corpus[i]["category"] for i in d["withheld_ids"]}
    assert set(d["failed_categories"]) <= withheld_cats


def test_tagged_adaptation_artifact_is_preserved_as_evidence():
    old = _load("adaptation_results.json")
    assert old["stage3_probe"]["n_failures"] == old["stage3_probe"]["n_probe_queries"]
    assert "n_below_gate" not in old["stage3_probe"]      # untouched original


# --------------------------------------------------------------------------- #
# lead-level leakage figures quoted in the README
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_df():
    from leads.eval_augmentation import load_real_frame
    return load_real_frame()[0]


def test_repeated_lead_figures_match_the_readme(real_df):
    from leads.train_ml import align_schema, lead_groups
    g = lead_groups(align_schema(real_df, extra_cols=("crm_id",)))
    from collections import Counter
    counts = Counter(x for x in g if not x.startswith("__row"))
    repeated = {k: v for k, v in counts.items() if v > 1}
    assert len(g) == 861
    assert sum(repeated.values()) == 336                 # 39% of 861
    assert len(repeated) == 80


@pytest.mark.parametrize("seed,expected", [(1, 66), (7, 71), (42, 73)])
def test_row_level_split_leaks_the_documented_number_of_siblings(real_df, seed,
                                                                  expected):
    from leads.eval_augmentation import split_labeled_rows
    lo = split_labeled_rows(real_df, random_state=seed)["lead_overlap"]
    assert lo["n_holdout_rows"] == 173
    assert lo["n_holdout_rows_with_train_sibling"] == expected   # 38% / 41% / 42%


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_grouped_split_used_by_the_committed_results_has_zero_overlap(real_df, seed):
    from leads.eval_augmentation import split_labeled_rows
    sp = split_labeled_rows(real_df, random_state=seed, group_split=True)
    assert sp["lead_overlap"]["n_holdout_rows_with_train_sibling"] == 0
    assert sp["n_holdout"] in range(165, 181)            # ~20% of 861


def test_grouped_artifact_is_marked_and_covers_the_three_seeds():
    d = _load("eval_augmentation_grouped.json")
    assert d["config"]["group_split"] is True
    assert d["split"]["grouped_by_lead"] is True
    assert d["split"]["lead_overlap"]["n_holdout_rows_with_train_sibling"] == 0
    assert d["seed_robustness"]["seeds"] == [1, 7, 42]
    assert _load("eval_augmentation_grouped_seed_robustness.json")["config"][
        "group_split"] is True


def test_grouped_and_row_level_runs_share_the_tagged_row_level_baseline():
    """Guards the claim 'the default run is unchanged': its artifact is intact."""
    row = _load("eval_augmentation.json")
    assert row["config"].get("group_split") in (None, False)
    assert row["split"]["seed"] == 1 and row["split"]["n_holdout"] == 173
    # committed numbers used in the README / RESULTS_v2.md
    rob = row["seed_robustness"]["feature_sets"]["full"]["summary"]["v2"]["macro"]
    assert round(rob["mean"], 4) == 0.1074


# --------------------------------------------------------------------------- #
# classifier
# --------------------------------------------------------------------------- #
def test_classifier_cv_artifact_reproduces():
    from chatbot.classifier import cv_summary, load_training_data
    stored = _load("classifier_cv.json")
    texts, labels = load_training_data()
    fresh = cv_summary(texts, labels, k=5)
    assert fresh["mean_macro_f1"] == pytest.approx(stored["mean_macro_f1"], abs=1e-9)
    assert stored["n_documents"] == 98
