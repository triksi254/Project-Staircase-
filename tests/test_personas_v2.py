"""Tests for the v2 inverted generator (leads.personas v2).

Frames are built in-memory so the tests need no ``data/processed`` files (that
directory is gitignored and absent in a fresh clone).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leads.personas import (  # noqa: E402
    GENERATORS,
    LABEL_NAMES,
    PERSONA_LABELS,
    PROFILE_FIELDS,
    SESSION_COLUMNS,
    CounsellorLabelModel,
    generate_session_v2,
    generate_sessions,
    generate_sessions_v2,
    main,
)
from leads.train_ml import ENGAGEMENT  # noqa: E402


PROFILE_BY_LABEL = {
    0: {"funding_clarity": 0, "destination_uk": 0, "has_course": 0,
        "has_intake": 0, "has_english_test": 0, "qual_level": 0,
        "passport_status": 1},
    1: {"funding_clarity": 2, "destination_uk": 1, "has_course": 1,
        "has_intake": 1, "has_english_test": 1, "qual_level": 3,
        "passport_status": 2},
    2: {"funding_clarity": 3, "destination_uk": 1, "has_course": 1,
        "has_intake": 1, "has_english_test": 1, "qual_level": 5,
        "passport_status": 2},
}


def _row(label, idx=0, **over):
    row = {
        "passport_status": 2, "qual_level": 4, "destination_uk": 1,
        "has_course": 1, "has_intake": 1, "study_gap_mentioned": 0,
        "previous_application_mentioned": 0, "funding_clarity": 3,
        "funding_method_present": 1, "has_english_test": 1,
        "english_band": 6.5, "note_word_count": 0,
        "label": label, "source": "real",
    }
    row.update(PROFILE_BY_LABEL.get(label, {}))
    row.update(over)
    return row


def _real_frame(n_per_class=20):
    return pd.DataFrame(
        [_row(lbl, idx=i) for lbl in (0, 1, 2) for i in range(n_per_class)]
    )


def _ambiguous_frame(n=60):
    """Identical profiles with cycling labels -> near-uniform ``predict_proba``."""
    return pd.DataFrame([
        _row(i % 3, funding_clarity=1, destination_uk=1, has_course=1,
             has_intake=1, has_english_test=1, qual_level=2,
             passport_status=2)
        for i in range(n)
    ])


def _label_model(frame, **kwargs):
    """Light test model (fewer trees -> fast, still deterministic)."""
    kwargs.setdefault("n_estimators", 60)
    kwargs.setdefault("random_state", 42)
    return CounsellorLabelModel(frame, **kwargs)



# --------------------------------------------------------------------------- #
# v1 regression guard (must remain byte-identical)
# --------------------------------------------------------------------------- #
V1_SNAPSHOT_FIELDS = [
    "session_id", "persona", "label", "funding_clarity",
    "passport_status", "message_count", "avg_delay_s",
    "session_word_count", "lead_score", "lead_label",
]

V1_SNAPSHOT = [
    ("synth-00000", "Hot", 2, 3, 2, 6, 12.9, 150, 0.76, "Hot"),
    ("synth-00001", "Warm", 1, 1, 2, 7, 162.8, 35, 0.4, "Warm"),
    ("synth-00002", "Hot", 2, 3, 2, 9, 43.6, 180, 0.82, "Hot"),
]


def test_v1_snapshot_unchanged():
    sessions, _ = generate_sessions(n=3, seed=42)
    got = [tuple(s[f] for f in V1_SNAPSHOT_FIELDS) for s in sessions]
    assert got == V1_SNAPSHOT


def test_v1_labels_come_from_persona_not_v2():
    assert GENERATORS == ("v1", "v2")
    assert LABEL_NAMES == {0: "Cold", 1: "Warm", 2: "Hot"}
    assert LABEL_NAMES[PERSONA_LABELS["Hot"]] == "Hot"


# --------------------------------------------------------------------------- #
# label model
# --------------------------------------------------------------------------- #
def test_label_model_excludes_engagement_features():
    model = _label_model(_real_frame(), random_state=42)
    for col in ENGAGEMENT:
        assert not any(f == col or f.startswith(col + "_")
                       for f in model.feature_names), col


def test_label_model_drops_unrated_rows():
    frame = _real_frame(n_per_class=5)
    extra = pd.DataFrame([_row(-1), _row(-1)])
    model = _label_model(pd.concat([frame, extra], ignore_index=True),
                                 random_state=42)
    assert model.n_train == 15


def test_label_model_requires_labels():
    frame = _real_frame().drop(columns=["label"])
    with pytest.raises(ValueError):
        _label_model(frame)


def test_label_model_proba_normalised():
    model = _label_model(_real_frame(), random_state=42)
    proba = model.label_proba(0)
    assert len(proba) == len(model.classes)
    assert abs(sum(proba) - 1.0) < 1e-9
# --------------------------------------------------------------------------- #
# v2 generator
# --------------------------------------------------------------------------- #
def test_v2_deterministic_same_seed():
    model_a = _label_model(_real_frame(), random_state=42)
    model_b = _label_model(_real_frame(), random_state=42)
    a, _ = generate_sessions_v2(n=40, seed=42, label_model=model_a)
    b, _ = generate_sessions_v2(n=40, seed=42, label_model=model_b)
    assert a == b


def test_v2_different_seeds_differ():
    model = _label_model(_real_frame(), random_state=42)
    a, _ = generate_sessions_v2(n=60, seed=1, label_model=model)
    b, _ = generate_sessions_v2(n=60, seed=2, label_model=model)
    assert [s["label"] for s in a] != [s["label"] for s in b] or \
        [s["message_count"] for s in a] != [s["message_count"] for s in b]


def test_v2_samples_from_proba_not_argmax():
    model = _label_model(_ambiguous_frame(), random_state=42)
    argmax = {int(v) for v in model.model.predict(model.X)}
    sessions, _ = generate_sessions_v2(n=400, seed=42, label_model=model)
    sampled = {s["label"] for s in sessions}
    # a single argmax class cannot explain a multi-class sampled set
    assert len(argmax) == 1
    assert len(sampled) >= 2
    assert sampled != argmax


def test_v2_engagement_conditioned_on_label():
    model = _label_model(_real_frame(), random_state=42)
    sessions, _ = generate_sessions_v2(n=800, seed=42, label_model=model)
    by_label = {
        lbl: [s["message_count"] for s in sessions if s["label"] == lbl]
        for lbl in (0, 2)
    }
    assert by_label[0] and by_label[2]
    mean_cold = sum(by_label[0]) / len(by_label[0])
    mean_hot = sum(by_label[2]) / len(by_label[2])
    assert mean_cold < mean_hot


def test_v2_schema_and_source_fields():
    model = _label_model(_real_frame(n_per_class=5), random_state=42)
    sessions, meta = generate_sessions_v2(n=25, seed=42, label_model=model)
    assert meta["generator"] == "v2"
    assert meta["label_model"]["excludes_engagement"] is True
    assert meta["label_model"]["n_train"] == 15
    for s in sessions:
        for col in SESSION_COLUMNS:
            if col == "question_categories":
                continue
            assert col in s, col
        for field in PROFILE_FIELDS:
            assert field in s
        assert s["label"] in (0, 1, 2)
        assert s["source_label"] in (0, 1, 2)
        assert 0 <= s["source_index"] < len(model.frame)


def test_v2_ignores_priors():
    model = _label_model(_real_frame(), random_state=42)
    plain, meta_plain = generate_sessions_v2(n=30, seed=42, label_model=model)
    priors = {"Cold": 0.34, "Warm": 0.33, "Hot": 0.33}
    weighted, meta = generate_sessions_v2(
        n=30, seed=42, label_model=model, priors=priors)
    assert plain == weighted
    assert meta["priors_ignored"] is True
    assert meta_plain["priors_ignored"] is False


def test_v2_invalid_n_raises():
    model = _label_model(_real_frame(), random_state=42)
    with pytest.raises(ValueError):
        generate_sessions_v2(n=0, label_model=model)


def test_generate_session_v2_single():
    import random
    model = _label_model(_real_frame(), random_state=42)
    s = generate_session_v2("synth-00000", random.Random(7), model)
    assert s["session_id"] == "synth-00000"
    assert s["persona"] == LABEL_NAMES[s["label"]]
    assert 0.0 <= s["lead_score"] <= 1.0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_v1_default_writes(tmp_path):
    rc = main(["--n", "12", "--seed", "42", "--out", str(tmp_path)])
    assert rc == 0
    csvs = list(tmp_path.glob("synthetic_sessions_*.csv"))
    assert len(csvs) == 1
    assert len(csvs[0].read_text(encoding="utf-8-sig").splitlines()) == 13


def test_cli_v2_generator_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("leads.train_ml.load_real",
                        lambda *a, **k: _real_frame(n_per_class=8))
    rc = main(["--generator", "v2", "--n", "12", "--seed", "42",
               "--out", str(tmp_path)])
    assert rc == 0
    csvs = list(tmp_path.glob("synthetic_sessions_*.csv"))
    assert len(csvs) == 1
    df = pd.read_csv(csvs[0], encoding="utf-8-sig")
    assert len(df) == 12
    assert set(df["label"].unique()) <= {0, 1, 2}


# --------------------------------------------------------------------------- #
# leakage safety
# --------------------------------------------------------------------------- #
def test_no_holdout_leakage():
    """v2 must never bootstrap from a real row that is in the holdout split.

    Two paths are checked: the explicitly fitted label model used by
    ``leads.eval_augmentation``, and the default internal fit of
    ``generate_sessions_v2`` (``--bootstrap-pool train``).
    """
    from leads.eval_augmentation import (
        fit_label_model,
        leakage_report,
        split_labeled_rows,
    )

    frame = _real_frame(n_per_class=25)
    split = split_labeled_rows(frame, random_state=42)
    assert split["n_train"] > 0 and split["n_holdout"] > 0
    holdout_ids = set(split["holdout_row_ids"])
    train_ids = set(split["train_row_ids"])
    # Sanity: the split is a partition of the labelled rows.
    assert holdout_ids.isdisjoint(train_ids)
    assert len(holdout_ids) + len(train_ids) == len(split["aligned"])

    # (1) Explicitly fitted model (the evaluation path).
    model = fit_label_model(split, bootstrap_pool="train", seed=42)
    sessions, meta = generate_sessions_v2(
        n=300, seed=42, label_model=model, bootstrap_pool="train")
    report = leakage_report(split, sessions)
    used = {s["source_row_id"] for s in sessions}
    assert used  # some real row was actually used
    assert report["n_holdout_rows_in_bootstrap_pool"] == 0
    assert used.isdisjoint(holdout_ids)
    assert used <= train_ids
    assert meta["bootstrap_pool"]["mode"] == "train"
    assert meta["bootstrap_pool"]["n_rows"] == split["n_train"]

    # (2) Default internal fit (``--bootstrap-pool train``).
    sessions2, meta2 = generate_sessions_v2(
        n=120, seed=42, real_frame=frame)
    used2 = {s["source_row_id"] for s in sessions2}
    assert used2
    assert used2.isdisjoint(holdout_ids)
    assert used2 <= train_ids
    assert meta2["bootstrap_pool"]["mode"] == "train"
    assert meta2["bootstrap_pool"]["n_rows"] == split["n_train"]




# --------------------------------------------------------------------------- #
# the protocol split follows --seed (it used to be pinned to 42)
# --------------------------------------------------------------------------- #
def test_v2_internal_split_follows_the_seed():
    """``--seed 7`` must bootstrap from seed-7's train split, as the eval does."""
    from leads.eval_augmentation import split_labeled_rows

    frame = _real_frame(n_per_class=25)
    sessions, _ = generate_sessions_v2(n=300, seed=7, real_frame=frame)
    used = {s["source_row_id"] for s in sessions}
    split7 = split_labeled_rows(frame, random_state=7)
    split42 = split_labeled_rows(frame, random_state=42)
    assert used <= set(split7["train_row_ids"])
    assert used.isdisjoint(split7["holdout_row_ids"])
    # premise: the seed-7 and seed-42 holdouts differ, so a pinned-42 pool would
    # have leaked seed-7 holdout rows
    assert set(split7["holdout_row_ids"]) != set(split42["holdout_row_ids"])


def test_v2_group_split_keeps_leads_apart():
    import pandas as pd
    frame = _real_frame(n_per_class=25).copy()
    frame["crm_id"] = [str(1000 + i // 2) for i in range(len(frame))]  # 2 forms / lead
    sessions, meta = generate_sessions_v2(n=200, seed=3, real_frame=frame,
                                          group_split=True)
    labelled = pd.DataFrame({"crm_id": frame["crm_id"]}).reset_index(drop=True)
    used_ids = {labelled.iloc[s["source_row_id"]]["crm_id"] for s in sessions}
    assert used_ids                       # some lead was sampled
    from leads.train_ml import lead_groups, split_real_indices
    fr = frame.reset_index(drop=True)
    _, hold = split_real_indices(fr, random_state=3, groups=lead_groups(fr))
    hold_ids = {labelled.iloc[i]["crm_id"] for i in hold}
    assert used_ids.isdisjoint(hold_ids)
