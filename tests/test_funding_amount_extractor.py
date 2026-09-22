"""extract_funding_amount + the funding_clarity it feeds into rubric_evidence().

Written red-first against the pre-change code (no ``extract_funding_amount``,
``rubric_evidence()`` never emits ``funding_clarity`` -- see
tests/test_funding_structural_test.py, which pins that older behaviour for a
*different* session and is untouched by this change).

Scope decision, pinned explicitly here: ``funding_clarity`` is real evidence
now (``leads.rubric.score_row`` sees a genuine tier, and so does the ML path,
since ``dashboard.app._score_turn`` passes the full ``rubric_evidence()`` dict
to ``predict_ml_proba``), but it is deliberately **not** added to
``LIVE_OBSERVABLE``: that would silently move the dashboard's displayed live
rule-score ceiling (0.7778, mass 0.18) and several tests already pinned to it
(``tests/test_live_path.py``). Expanding ``LIVE_OBSERVABLE`` is a separate,
larger decision left to the caller.
"""
import pytest

from chatbot.session_features import (
    LIVE_OBSERVABLE,
    SessionTracker,
    empty_evidence,
    extract_funding_amount,
    live_rule_breakdown,
)
from leads.rubric import score_row


def _session():
    from datetime import datetime
    return SessionTracker("LEAD-FUND-TEST", datetime(2026, 9, 22, 12, 0, 0))


def _turn(t, msg, category="Scholarships", offset=0):
    from datetime import datetime, timedelta
    t.add_turn(msg, {"category": category, "confidence": 0.8},
              datetime(2026, 9, 22, 12, 0, 0) + timedelta(seconds=30 * offset))


# --------------------------------------------------------------------------- #
# extract_funding_amount: currency patterns
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("I have £15,000 saved", 15000.0),
    ("I have £15000 saved", 15000.0),
    ("15,000 pounds is my budget", 15000.0),
    ("my budget is £20k", 20000.0),
    ("my budget is 20k", 20000.0),
    ("I can offer twenty thousand", 20000.0),
    ("I can offer fifteen hundred", 1500.0),
    ("£15,000-£20,000 is my range", 15000.0),          # range -> lower bound
    ("I can pay up to £20,000", 20000.0),               # "up to X" -> the bound
    ("£ 15,000", 15000.0),                              # space after symbol
    ("£15,000 POUNDS", 15000.0),                        # case-insensitive
    ("TWENTY THOUSAND pounds", 20000.0),
])
def test_extract_funding_amount_matches_the_documented_patterns(text, expected):
    assert extract_funding_amount(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", [
    "",
    "What is the total cost of a one-year masters at BCU?",   # "one" != an amount
    "Can I pay fees in three instalments over the year?",     # "three" != an amount
    "I have no source of funds right now",
    "What is the no visa no pay policy?",
    "How much is tuition at Aston for international students?",
])
def test_extract_funding_amount_returns_none_without_a_real_amount(text):
    assert extract_funding_amount(text) is None


# --------------------------------------------------------------------------- #
# rubric_evidence(): the four verify queries, run through one session
# --------------------------------------------------------------------------- #
def test_the_four_verify_queries_move_funding_clarity_as_specified():
    t = _session()

    _turn(t, "I can self-fund up to £20,000 but would prefer a partial "
             "scholarship", offset=0)
    ev = t.rubric_evidence()
    assert ev["funding_clarity"] == 3, "turn 1: amount >= 20000 -> tier 3"

    _turn(t, "My parents have £15,000 saved for my education", offset=1)
    ev = t.rubric_evidence()
    assert ev["funding_clarity"] == 2, (
        "turn 2: 10000 <= 15000 < 20000 -> tier 2 (each qualifying turn sets "
        "its own tier; it does not only ever increase)")

    _turn(t, "I have no source of funds right now", offset=2)
    ev = t.rubric_evidence()
    assert ev["funding_clarity"] == 2, "turn 3: method mentioned, no amount -> unchanged"

    _turn(t, "What is the no visa no pay policy?", offset=3)
    ev = t.rubric_evidence()
    assert ev["funding_clarity"] == 2, "turn 4: no method, no amount -> unchanged"


def test_a_bare_amount_without_a_funding_method_does_not_set_clarity():
    t = _session()
    _turn(t, "Tuition is £15,000 per year", offset=0)     # an amount, no method phrase
    assert t.rubric_evidence()["funding_clarity"] == 0


def test_a_bare_method_without_an_amount_never_moves_off_zero():
    t = _session()
    _turn(t, "I can self-fund my studies", offset=0)
    assert t.rubric_evidence()["funding_clarity"] == 0


def test_empty_session_defaults_funding_clarity_to_zero():
    assert empty_evidence()["funding_clarity"] == 0
    assert _session().rubric_evidence()["funding_clarity"] == 0


# --------------------------------------------------------------------------- #
# funding_clarity now has a real rubric contribution ...
# --------------------------------------------------------------------------- #
def test_funding_clarity_now_has_a_real_rubric_contribution():
    t = _session()
    _turn(t, "I can self-fund up to £20,000", offset=0)
    ev = t.rubric_evidence()
    line = next(c for c in score_row(dict(ev)).contributions
                if c["feature"] == "funding_clarity")
    assert line["contribution"] == pytest.approx(0.20)   # tier 3 (CLEAR) = full weight


def test_tier_1_is_a_documented_surprise_zero_contribution():
    """amount < 10000 -> funding_clarity = 1 = FUNDING_UNCLEAR, which
    leads.rubric._funding_contrib maps to 0.0 -- the same as never observing
    it. Not something this change alters (rubric weights are untouched); worth
    pinning so it is not mistaken for a bug later."""
    t = _session()
    _turn(t, "I can self-fund £5,000", offset=0)
    ev = t.rubric_evidence()
    assert ev["funding_clarity"] == 1
    line = next(c for c in score_row(dict(ev)).contributions
                if c["feature"] == "funding_clarity")
    assert line["contribution"] == 0.0


# --------------------------------------------------------------------------- #
# ... but the LIVE rule score (dashboard panel) is deliberately left alone
# --------------------------------------------------------------------------- #
def test_funding_clarity_is_not_in_live_observable_by_design():
    assert "funding_clarity" not in LIVE_OBSERVABLE
    assert set(LIVE_OBSERVABLE) == {"has_course", "has_intake", "english_test"}


def test_live_rule_score_ceiling_is_unchanged_by_a_funding_turn():
    t = _session()
    _turn(t, "I can self-fund up to £20,000", offset=0)
    bd = live_rule_breakdown(t.rubric_evidence())
    assert bd["mass"] == pytest.approx(0.18)              # unchanged: 0.05+0.05+0.08
    assert "funding_clarity" not in {l["feature"] for l in bd["lines"]}


# --------------------------------------------------------------------------- #
# widened funding-method phrasing ("can afford" / "can put towards")
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "I can afford £15,000 towards my degree",
    "I can put towards £15,000 for my studies",
])
def test_the_two_new_method_phrases_are_recognised(text):
    t = _session()
    _turn(t, text, offset=0)
    assert t.rubric_evidence()["funding_clarity"] == 2


def test_a_bare_amount_alone_still_does_not_set_clarity_via_the_new_phrases():
    """"can afford"/"can put towards" are added markers, not a relaxation of
    the amount+method requirement: an amount with neither still does nothing."""
    t = _session()
    _turn(t, "The course costs around £9,250 a year", offset=0)
    assert t.rubric_evidence()["funding_clarity"] == 0
