"""The generated RESULTS_v2.md must not contradict its own tables.

An earlier renderer hard-coded "+0.1188" (a seed-42 figure) into the mechanism
text and compared the *seed-1* delta with a pre-fix seed-42 number, so the
committed document said the leak had "partly inflated" a figure that its own
Table E reproduced to within 0.0004. These tests render from the committed
payload (no recomputation) and pin the corrected behaviour.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from leads.eval_augmentation import main, render_markdown

REPO = Path(__file__).resolve().parent.parent
PAYLOAD = REPO / "artifacts" / "eval_augmentation.json"


@pytest.fixture(scope="module")
def payload():
    return json.loads(PAYLOAD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def md(payload):
    return render_markdown(copy.deepcopy(payload))


def _delta(payload, fs):
    return payload["results"][fs]["generators"]["v2"]["generalization"][
        "delta_macro_f1"]


def test_mechanism_text_uses_this_runs_delta_not_a_hard_coded_one(payload, md):
    d = _delta(payload, "full")
    seed = payload["config"]["seed"]
    assert "The v2 delta of %+.4f (full feature set, seed %d)" % (d, seed) in md
    assert "gain of +0.1188" not in md
    assert "The v2 augmentation gain of" not in md


def test_verdict_does_not_claim_a_leak_the_seed_sweep_refutes(md):
    assert "partly inflated by the leak" not in md
    assert "no material effect" in md          # seed-42 row reproduces +0.1184


def test_verdict_leads_with_the_fair_no_engagement_configuration(payload, md):
    line = next(l for l in md.splitlines() if l.startswith("Verdict (seed"))
    assert line.index("no-engagement") < line.index("full-set")
    assert "not evidence that behavioural information transfers" in line
    overall = next(l for l in md.splitlines()
                   if l.startswith("Verdict (no-engagement configuration)"))
    assert "descriptive rules" in overall and "not significance tests" in overall
    assert "constant (0) for every real row" in overall


def test_control_caveat_is_stated(md):
    assert "not like-for-like" in md
    assert "in-sample" in md


def test_single_seed_tables_say_which_seed_they_are(payload, md):
    assert "Tables A-D show a single run: generator/split seed %d" % (
        payload["config"]["seed"]) in md


def test_residual_caveats_are_accurate_about_v1_and_the_pools(md):
    assert "hand-authored" in md
    assert "shape v1's *profile*" in md


def test_old_payload_without_lead_level_check_says_so(payload, md):
    assert "lead_overlap" not in payload["split"]      # committed before the check
    assert "not recorded in this payload" in md


def test_lead_level_overlap_is_reported_when_present(payload):
    p = copy.deepcopy(payload)
    p["split"]["grouped_by_lead"] = True
    p["split"]["lead_overlap"] = {"n_holdout_rows": 173,
                                  "n_holdout_rows_with_train_sibling": 0,
                                  "share": 0.0}
    p["split"]["row_split_lead_overlap"] = {
        "n_holdout_rows": 173, "n_holdout_rows_with_train_sibling": 66,
        "share": 0.3815}
    out = render_markdown(p)
    assert "lead-grouped" in out.splitlines()[0]
    assert "| holdout rows with a same-lead sibling in train | 0 of 173 (0%) |" in out
    assert "| ... under the row-level split (reference) | 66 of 173 (38%) |" in out
    assert "--group-split" in out


def test_render_only_recomputes_nothing_and_leaves_the_payload_alone(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "eval_augmentation.json").write_bytes(PAYLOAD.read_bytes())
    before = hashlib.sha256((art / "eval_augmentation.json").read_bytes()).hexdigest()
    out = tmp_path / "R.md"
    assert main(["--render-only", "--artifacts", str(art), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "without recomputation" in text
    assert hashlib.sha256((art / "eval_augmentation.json").read_bytes()
                          ).hexdigest() == before
    assert sorted(p.name for p in art.iterdir()) == ["eval_augmentation.json"]


def test_every_number_row_survives_the_rerender(payload, md):
    """Only prose changed: the committed tables' numbers are all still there."""
    committed = (REPO / "leads" / "RESULTS_v2.md").read_text(encoding="utf-8")
    rows = [l for l in committed.splitlines()
            if l.startswith("|") and any(c.isdigit() for c in l)]
    fresh = set(md.splitlines())
    assert rows and all(r in fresh for r in rows)
