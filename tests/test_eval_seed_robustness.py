"""Tests for the seed-robustness sweep in ``leads.eval_augmentation``.

The sweep re-runs the *whole* leakage-safe protocol once per seed (split ->
CounsellorLabelModel fit -> v1/v2 generation -> real-only/+v1/+v2/+distill), so
a seed change perturbs the split and the generators together.

``test_seed_robustness_runs`` drives the real ``main()`` with ``--seeds 1,42``
but a small ``--n``, ``--model logreg`` and ``--no-engagement``: the test has to
prove the wiring (two independent pipeline runs, per-seed rows, summary ranges,
artifact structure), not reproduce the headline numbers.
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leads.eval_augmentation import (  # noqa: E402
    DELTA_VARIANTS,
    FEATURE_SETS,
    _seed_verdict,
    main,
    parse_seeds,
    seed_robustness,
)


# --------------------------------------------------------------------------- #
# Unit: seed parsing
# --------------------------------------------------------------------------- #
def test_parse_seeds_accepts_comma_list_and_single_value():
    assert parse_seeds("1,7,42") == [1, 7, 42]
    assert parse_seeds(" 1 , 42 ") == [1, 42]
    # single value keeps the legacy single-seed command line working
    assert parse_seeds("42") == [42]


def test_parse_seeds_rejects_empty():
    with pytest.raises(ValueError):
        parse_seeds("")


def test_default_delta_variants_cover_v1_v2_distill():
    assert DELTA_VARIANTS == ("v1", "v2", "distill")
    assert "full" in FEATURE_SETS
# --------------------------------------------------------------------------- #
# Unit: aggregation into per-seed rows + summary ranges
# --------------------------------------------------------------------------- #
def _fake_generator(delta_macro: float, holdout_macro: float, hot_f1: float):
    """One ``generators[gen]`` block shaped like the live payload."""
    return {
        "headline_macro_f1": holdout_macro - 0.1,
        "generalization": {
            "real_plus_synthetic": {
                "macro_f1": holdout_macro,
                "per_class": {"Hot": {"f1": hot_f1}},
            },
            "delta_macro_f1": delta_macro,
            "delta_per_class_f1": {"Hot": hot_f1 - 0.4},
        },
    }


def _fake_pipeline_payload(deltas):
    """Mimic ``_pipeline_for_seed`` output for the ``full`` feature set."""
    return {
        "results": {
            "full": {
                "real_only": {
                    "holdout_macro_f1": 0.5,
                    "holdout_per_class": {"Hot": {"f1": 0.4}},
                },
                "generators": {
                    gen: _fake_generator(delta, 0.5 + delta, 0.4 + delta)
                    for gen, delta in deltas.items()
                },
            }
        }
    }


def test_seed_robustness_records_distinct_per_seed_deltas():
    per_seed = {
        1: _fake_pipeline_payload({"v1": 0.02, "v2": 0.12, "distill": 0.05}),
        42: _fake_pipeline_payload({"v1": 0.04, "v2": 0.30, "distill": 0.07}),
    }
    robust = seed_robustness(per_seed, ("full",))

    # top-level structure
    assert robust["seeds"] == [1, 42]
    assert robust["n_seeds"] == 2
    block = robust["feature_sets"]["full"]
    assert set(block) == {"rows", "summary"}

    # one row per seed, with the delta of every variant recorded separately
    assert [r["seed"] for r in block["rows"]] == [1, 42]
    for row in block["rows"]:
        for gen in DELTA_VARIANTS:
            assert "delta_%s_macro" % gen in row
            assert "delta_%s_hot" % gen in row
    assert block["rows"][0]["delta_v2_macro"] == pytest.approx(0.12)
    assert block["rows"][1]["delta_v2_macro"] == pytest.approx(0.30)
    assert (block["rows"][0]["delta_v2_macro"]
            != block["rows"][1]["delta_v2_macro"])

    # summary carries n / mean / min / max per variant and metric
    summary = block["summary"]
    for gen in DELTA_VARIANTS:
        for metric in ("macro", "hot"):
            rec = summary[gen][metric]
            assert set(rec) == {"n", "mean", "min", "max"}
            assert rec["n"] == 2
    v2 = summary["v2"]["macro"]
    assert v2["mean"] == pytest.approx(0.21)
    assert v2["min"] == pytest.approx(0.12)
    assert v2["max"] == pytest.approx(0.30)


def test_seed_verdict_tiers():
    def summary(lo, hi, mean):
        return {"v2": {"macro": {"n": 3, "min": lo, "max": hi, "mean": mean}}}

    assert _seed_verdict(summary(0.05, 0.20, 0.10))["tier"] == "ROBUST"
    assert _seed_verdict(summary(0.01, 0.20, 0.08))["tier"] == "PARTIALLY ROBUST"
    assert _seed_verdict(summary(-0.02, 0.20, 0.05))["tier"] == "SEED-SENSITIVE"
    assert _seed_verdict(summary(0.005, 0.01, 0.008))["tier"] == "INCONCLUSIVE"
    assert _seed_verdict({}) is None


# --------------------------------------------------------------------------- #
# Integration: the real CLI actually runs two pipelines, one per seed
# --------------------------------------------------------------------------- #
def test_seed_robustness_runs(tmp_path):
    """``--seeds 1,42`` must run two full pipelines and record both deltas.

    ``--n`` / ``--model logreg`` / ``--no-engagement`` keep the runtime small:
    the point is the wiring (two independent pipeline runs, one row per seed,
    summary ranges, both artifacts written), not the headline numbers.
    """
    out_md = tmp_path / "RESULTS_v2.md"
    art_dir = tmp_path / "artifacts"
    rc = main([
        "--seeds", "1,42",
        "--n", "150",
        "--model", "logreg",
        "--no-engagement",
        "--out", str(out_md),
        "--artifacts", str(art_dir),
    ])
    assert rc == 0

    payload_path = art_dir / "eval_augmentation.json"
    robust_path = art_dir / "eval_augmentation_seed_robustness.json"
    assert payload_path.exists(), "payload artifact must be written"
    assert robust_path.exists(), "seed-robustness artifact (--seeds with >1) must be written"

    robust = json.loads(robust_path.read_text(encoding="utf-8"))["seed_robustness"]

    # expected top-level structure
    assert set(robust) == {"seeds", "n_seeds", "feature_sets"}
    assert robust["seeds"] == [1, 42]
    assert robust["n_seeds"] == 2
    assert set(robust["feature_sets"]) == {"no-engagement"}

    block = robust["feature_sets"]["no-engagement"]
    assert set(block) == {"rows", "summary"}

    # one completed pipeline per seed, each recording every variant's delta
    assert [r["seed"] for r in block["rows"]] == [1, 42]
    for row in block["rows"]:
        assert row["real_only_macro"] is not None
        for gen in DELTA_VARIANTS:
            assert row["delta_%s_macro" % gen] is not None
            assert row["delta_%s_hot" % gen] is not None

    # the whole point of the sweep: the per-seed deltas must differ
    deltas = [r["delta_v2_macro"] for r in block["rows"]]
    assert len(set(deltas)) == 2, "seeds 1 and 42 produced identical deltas"

    # summary carries n / mean / min / max per variant and metric
    for gen in DELTA_VARIANTS:
        for metric in ("macro", "hot"):
            rec = block["summary"][gen][metric]
            assert set(rec) == {"n", "mean", "min", "max"}
            assert rec["n"] == 2
            assert rec["min"] <= rec["mean"] <= rec["max"]

    # the rendered report is written and carries the new tables
    assert out_md.exists()
    md = out_md.read_text(encoding="utf-8")
    assert "Table E - holdout macro F1" in md
    assert "Table E2 - holdout Hot F1" in md
    assert "Seed-robustness verdict" in md
