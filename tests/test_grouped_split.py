"""Lead-grouped splitting: the same CRM id must never sit on both sides.

39% of the labelled counsellor rows belong to a CRM id that occurs more than
once, and a row-level split put a same-lead sibling (always with the same
label) in train for 38-42% of holdout rows. ``split_real_indices(groups=...)``
is the opt-in remedy; the default split is unchanged so every tagged result
stays reproducible.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from leads.train_ml import (
    SOURCE,
    align_schema,
    experiment_cv,
    experiment_generalization,
    lead_groups,
    split_real_indices,
)


def _frame(n_leads=80, sizes=(1, 2, 3), n_blank=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_leads):
        label = int(rng.choice([0, 1, 2], p=[0.45, 0.45, 0.10]))
        for _ in range(int(rng.choice(sizes))):
            rows.append({"crm_id": str(100000 + i), "label": label})
    for _ in range(n_blank):
        rows.append({"crm_id": "", "label": int(rng.choice([0, 1]))})
    return pd.DataFrame(rows)


def _overlap(frame, tr, te):
    train_ids = set(frame.iloc[tr]["crm_id"]) - {""}
    return sum(1 for i in te if frame.iloc[i]["crm_id"] in train_ids)


# --------------------------------------------------------------------------- #
# lead_groups
# --------------------------------------------------------------------------- #
def test_lead_groups_blank_and_missing_ids_are_singletons():
    df = pd.DataFrame({"crm_id": ["7", "", None, np.nan, "7"], "label": [0] * 5})
    g = lead_groups(df)
    assert g[0] == g[4] == "7"
    assert len({g[1], g[2], g[3]}) == 3          # each blank is its own lead
    assert "7" not in (g[1], g[2], g[3])


def test_lead_groups_normalises_numeric_ids():
    df = pd.DataFrame({"crm_id": [375732.0, 375732, "375732"], "label": [0] * 3})
    assert lead_groups(df) == ["375732"] * 3


def test_lead_groups_without_an_id_column_is_row_level():
    df = pd.DataFrame({"label": [0, 1, 2]})
    assert len(set(lead_groups(df))) == 3


# --------------------------------------------------------------------------- #
# split_real_indices
# --------------------------------------------------------------------------- #
def test_row_level_split_leaks_repeated_leads():
    """Premise: on this frame the legacy split does put siblings in train."""
    df = _frame()
    tr, te = split_real_indices(df, random_state=1)
    assert _overlap(df, tr, te) > 0


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_grouped_split_has_no_lead_on_both_sides(seed):
    df = _frame()
    tr, te = split_real_indices(df, random_state=seed, groups=lead_groups(df))
    assert _overlap(df, tr, te) == 0
    assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(df)


def test_grouped_split_is_deterministic_and_seed_dependent():
    df = _frame()
    g = lead_groups(df)
    a = split_real_indices(df, random_state=3, groups=g)
    b = split_real_indices(df, random_state=3, groups=g)
    c = split_real_indices(df, random_state=4, groups=g)
    assert (a[1] == b[1]).all() and (a[0] == b[0]).all()
    assert not np.array_equal(a[1], c[1])


def test_grouped_split_size_and_stratification_are_close_to_the_row_split():
    df = _frame(n_leads=200)
    tr, te = split_real_indices(df, random_state=5, groups=lead_groups(df))
    assert 0.15 <= len(te) / len(df) <= 0.25
    assert set(df.iloc[te]["label"]) == set(df.iloc[tr]["label"]) == {0, 1, 2}
    share_all = (df["label"] == 2).mean()
    share_te = (df.iloc[te]["label"] == 2).mean()
    assert abs(share_all - share_te) < 0.06


def test_default_split_is_byte_identical_to_the_legacy_call():
    df = _frame()
    tr, te = split_real_indices(df, random_state=42)
    idx = np.arange(len(df))
    tr0, te0 = train_test_split(idx, test_size=0.2, stratify=df["label"],
                                random_state=42)
    assert (tr == tr0).all() and (te == te0).all()


# --------------------------------------------------------------------------- #
# threaded through the experiments
# --------------------------------------------------------------------------- #
def _real_like(seed, n_leads=70):
    """Feature frame shaped like the real corpus, with repeated CRM ids."""
    from leads.personas import generate_sessions
    sessions, _ = generate_sessions(n=n_leads * 2, seed=seed)
    df = pd.DataFrame(sessions)
    ids = []
    for i in range(len(df)):
        ids.append(str(200000 + i // 2))            # every lead appears twice
    df["crm_id"] = ids
    df["source"] = "real"
    for col in ("message_count", "avg_delay_s", "question_category_entropy",
                "visa_intent_mentioned", "returning_session",
                "session_word_count"):
        df[col] = np.nan                             # real rows have no engagement
    return df


def _synthetic(seed, n=150):
    from leads.personas import generate_sessions
    sessions, _ = generate_sessions(n=n, seed=seed)
    df = pd.DataFrame(sessions)
    df["source"] = "synthetic"
    return df


def test_align_schema_can_carry_the_lead_id_through():
    real, synth = _real_like(1), _synthetic(2)
    combined = align_schema(real, synth, extra_cols=("crm_id",))
    assert "crm_id" in combined.columns
    assert combined.loc[combined[SOURCE] == "synthetic", "crm_id"].isna().all()
    assert "crm_id" not in align_schema(real, synth).columns   # default unchanged


def test_generalization_holdout_is_lead_disjoint_when_grouped():
    combined = align_schema(_real_like(1), _synthetic(2), extra_cols=("crm_id",))
    res = experiment_generalization(combined, False, "rf", no_engagement=True,
                                    split_random_state=1, group_by_lead=True)
    real = combined[combined[SOURCE] == "real"].reset_index(drop=True)
    tr = real.iloc[res["real_train_index"]]["crm_id"]
    te = real.iloc[res["real_holdout_index"]]["crm_id"]
    assert set(tr).isdisjoint(set(te))
    # and the legacy path on the same data does leak (proves the flag matters)
    leaky = experiment_generalization(combined, False, "rf", no_engagement=True,
                                      split_random_state=1)
    tr0 = real.iloc[leaky["real_train_index"]]["crm_id"]
    te0 = real.iloc[leaky["real_holdout_index"]]["crm_id"]
    assert set(tr0) & set(te0)


def test_grouped_cv_never_splits_a_lead_across_folds():
    combined = align_schema(_real_like(3), None, extra_cols=("crm_id",))
    out = experiment_cv(combined, False, "rf", n_splits=3, n_repeats=1,
                        no_engagement=True, group_by_lead=True)
    assert out["grouped_by_lead"] is True
    assert out["n_splits"] == 3
    assert 0.0 <= out["macro_f1"]["mean"] <= 1.0
    assert experiment_cv(combined, False, "rf", n_splits=3, n_repeats=1,
                         no_engagement=True)["grouped_by_lead"] is False
