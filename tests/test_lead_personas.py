"""Tests for the Cold/Warm/Hot persona simulator (leads.personas)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leads.personas import (  # noqa: E402
    MEASURED_PRIORS,
    PERSONA_LABELS,
    SESSION_COLUMNS,
    generate_session,
    generate_sessions,
)


def _rng(seed=7):
    import random
    return random.Random(seed)


def test_deterministic_same_seed():
    a, _ = generate_sessions(n=50, seed=123)
    b, _ = generate_sessions(n=50, seed=123)
    assert [s["session_id"] for s in a] == [s["session_id"] for s in b]
    assert [s["lead_score"] for s in a] == [s["lead_score"] for s in b]
    assert [s["persona"] for s in a] == [s["persona"] for s in b]


def test_different_seeds_differ():
    a, _ = generate_sessions(n=50, seed=1)
    b, _ = generate_sessions(n=50, seed=2)
    assert [s["persona"] for s in a] != [s["persona"] for s in b]


def test_persona_scores_ordered_cold_lt_warm_lt_hot():
    sessions, _ = generate_sessions(n=600, seed=42)
    means = {}
    for persona in ("Cold", "Warm", "Hot"):
        sub = [s for s in sessions if s["persona"] == persona]
        assert sub, "no %s sessions sampled" % persona
        means[persona] = sum(s["lead_score"] for s in sub) / len(sub)
    assert means["Cold"] < means["Warm"] < means["Hot"]


def test_behaviour_monotonic_with_persona():
    sessions, _ = generate_sessions(n=600, seed=42)
    for field in ("message_count", "question_category_entropy",
                  "session_word_count"):
        means = {}
        for persona in ("Cold", "Warm", "Hot"):
            sub = [s for s in sessions if s["persona"] == persona]
            means[persona] = sum(s[field] for s in sub) / len(sub)
        assert means["Cold"] < means["Warm"] < means["Hot"], field
    delays = {}
    for persona in ("Cold", "Warm", "Hot"):
        sub = [s for s in sessions if s["persona"] == persona]
        delays[persona] = sum(s["avg_delay_s"] for s in sub) / len(sub)
    assert delays["Cold"] > delays["Warm"] > delays["Hot"]


def test_custom_priors_respected():
    sessions, meta = generate_sessions(
        n=300, seed=9,
        priors={"Cold": 0.1, "Warm": 0.3, "Hot": 0.6},
    )
    assert meta["persona_counts"]["Hot"] > meta["persona_counts"]["Cold"]
    assert sum(meta["persona_counts"].values()) == 300


def test_measured_priors_default():
    assert abs(sum(MEASURED_PRIORS.values()) - 1.0) < 1e-9
    assert MEASURED_PRIORS["Hot"] > MEASURED_PRIORS["Warm"] > MEASURED_PRIORS["Cold"]


def test_invalid_n_raises():
    import pytest
    with pytest.raises(ValueError):
        generate_sessions(n=0)


def test_single_session_helper_columns():
    s = generate_session("synth-00000", "Hot", _rng())
    assert s["session_id"] == "synth-00000"
    assert s["persona"] == "Hot"
    assert s["label"] == PERSONA_LABELS["Hot"] == 2
    assert 0.0 <= s["lead_score"] <= 1.0
    assert s["lead_label"] in ("Cold", "Warm", "Hot")
    for col in SESSION_COLUMNS:
        if col in ("question_categories",):
            continue
        assert col in s, col
