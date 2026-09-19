"""Tests for evaluation.adaptation_loop (cold-start adaptation experiment)."""
import hashlib
import json
from pathlib import Path

from evaluation.adaptation_loop import CACHE_PATHS, run_adaptation_experiment

REPO = Path(__file__).resolve().parent.parent


def _write_corpus(tmp_path, n=20):
    """Tiny corpus with distinctive vocabulary so rank-1 is deterministic."""
    faqs = [{"question": "What about topic%d alpha%d?" % (i, i),
             "answer": "Answer for topic%d alpha%d beta%d." % (i, i, i),
             "keywords": ["topic%d" % i], "category": "Program Details",
             "institution": "General"} for i in range(n)]
    fp = tmp_path / "corpus.json"
    fp.write_text(json.dumps({"faqs": faqs}), encoding="utf-8")
    return fp


def _write_gold(tmp_path, name, pairs):
    fp = tmp_path / name
    fp.write_text(json.dumps(pairs), encoding="utf-8")
    return fp


def _fixtures(tmp_path):
    """Batch B targets ids 0-9; batch A probes ids 5-13 with distinct wording.

    Phrasings are disjoint across the two sets on purpose: a shared query
    string would trip the batch-B-leak guard by construction.
    """
    corpus = _write_corpus(tmp_path)
    b = _write_gold(tmp_path, "b.json", [
        {"query": "topic%d alpha%d" % (i, i), "gold_id": i,
         "category": "Program Details", "expected_abstain": False}
        for i in range(10)])
    a = _write_gold(tmp_path, "a.json", [
        {"query": "please explain topic%d alpha%d" % (i, i), "gold_id": i,
         "category": "Program Details", "expected_abstain": False}
        for i in range(5, 14)])
    return corpus, a, b


def _run(tmp_path, **kw):
    corpus, a, b = _fixtures(tmp_path)
    return run_adaptation_experiment(corpus_path=corpus, batch_a_path=a,
                                     batch_b_path=b, **kw)


def _hashes(paths):
    return {str(p): (hashlib.sha256(p.read_bytes()).hexdigest()
                     if p.is_file() else None) for p in paths}


def test_experiment_produces_expected_keys(tmp_path):
    out = _run(tmp_path)
    for key in ("config", "stage1_full", "stage2_cold_start", "stage3_probe",
                "stage4_full_restore", "stage4b_targeted_restore",
                "withheld_ids", "restored_ids", "targeted_ids",
                "batch_a_failures", "delta", "recovery", "residual_gap",
                "bootstrap_ci", "integrity", "n_entries_full",
                "n_entries_reduced", "n_entries_targeted",
                "batch_b_in_stage3"):
        assert key in out, "missing key %s" % key
    assert out["config"]["n_bootstrap"] == 1000
    assert out["config"]["seed"] == 42


def test_batch_b_not_used_in_stage3(tmp_path):
    out = _run(tmp_path)
    b_queries = {g["query"] for g in
                 json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))}
    assert out["batch_b_in_stage3"] is False
    assert not (b_queries & {f["query"] for f in out["batch_a_failures"]})
    assert out["stage3_probe"]["n_probe_queries"] == 9


def test_corpus_on_disk_unchanged(tmp_path):
    corpus, a, b = _fixtures(tmp_path)
    watched = [corpus] + list(CACHE_PATHS)
    before = _hashes(watched)
    run_adaptation_experiment(corpus_path=corpus, batch_a_path=a,
                              batch_b_path=b)
    assert _hashes(watched) == before
    assert before[str(corpus)] is not None


def test_integrity_block_reports_corpus_and_caches(tmp_path):
    out = _run(tmp_path)
    assert out["integrity"]["corpus_unchanged"] is True
    assert out["integrity"]["caches_unchanged"] is True
    assert out["integrity"]["unchanged"] is True
    assert set(out["integrity"]["before"]) == {
        "corpus", "vectorizer_cache", "matrix_cache"}


def test_bootstrap_ci_has_1000_resamples(tmp_path):
    out = _run(tmp_path)
    assert set(out["bootstrap_ci"]) == {
        "full_recall_at_3", "full_recall_at_5",
        "targeted_recall_at_3", "targeted_recall_at_5"}
    for ci in out["bootstrap_ci"].values():
        assert ci["n_bootstrap"] == 1000
        assert ci["n"] == 10  # batch B in the synthetic fixture
        assert ci["ci_low"] <= ci["ci_high"]


def test_reproducible_with_seed(tmp_path):
    first = _run(tmp_path)
    second = _run(tmp_path)
    assert first["withheld_ids"] == second["withheld_ids"]
    assert first["targeted_ids"] == second["targeted_ids"]
    assert first["delta"] == second["delta"]
    assert first["recovery"] == second["recovery"]
    assert (first["stage2_cold_start"]["recall_at_3"]
            == second["stage2_cold_start"]["recall_at_3"])


def test_stage4_recall_geq_stage2_recall(tmp_path):
    out = _run(tmp_path)
    assert (out["stage4_full_restore"]["recall_at_3"]
            >= out["stage2_cold_start"]["recall_at_3"])
    assert (out["stage4_full_restore"]["recall_at_5"]
            >= out["stage2_cold_start"]["recall_at_5"])


def test_stage4_is_stage1_by_construction(tmp_path):
    """Full restoration returns the baseline state exactly - determinism check."""
    out = _run(tmp_path)
    assert out["stage1_full"] == out["stage4_full_restore"]
    assert out["recovery"]["full_recall_at_3"] == 1.0


def test_targeted_restoration_is_subset_of_full(tmp_path):
    out = _run(tmp_path)
    assert set(out["targeted_ids"]) <= set(out["withheld_ids"])
    assert out["n_entries_reduced"] <= out["n_entries_targeted"]
    assert out["n_entries_targeted"] <= out["n_entries_full"]
    assert out["residual_gap"]["recall_at_3"] >= 0.0


def test_withhold_fraction_is_thirty_percent(tmp_path):
    out = _run(tmp_path)
    # batch B targets 10 distinct entries; 30% rounds to 3
    assert len(out["withheld_ids"]) == 3
    assert out["n_entries_full"] - out["n_entries_reduced"] == 3


def test_cold_start_does_not_improve_recall(tmp_path):
    out = _run(tmp_path)
    assert (out["stage2_cold_start"]["recall_at_3"]
            <= out["stage1_full"]["recall_at_3"])
    assert out["delta"]["cold_start_cost_recall_at_3"] >= 0.0


def test_real_gold_sets_are_disjoint_on_query_strings():
    """The batch-B leak guard requires the two halves to share no query text."""
    a = json.loads((REPO / "evaluation" / "gold_queries.json")
                   .read_text(encoding="utf-8"))
    b = json.loads((REPO / "evaluation" / "gold_queries_b.json")
                   .read_text(encoding="utf-8"))
    overlap = {x["query"] for x in a} & {x["query"] for x in b}
    assert not overlap, "gold sets share query text: %s" % sorted(overlap)