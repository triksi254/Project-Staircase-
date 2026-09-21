"""The multi-intent guard, so respond() can report *which* groups matched.

Both queries in the reported session escalated at displayed confidence 0.81 (gate
0.60). The guard counts keyword groups, and an institution name is one of the
groups, so "one question + a document + an institution" reads as three intents.
"""
import pytest

from chatbot import intent_guard as G

REPORTED = {
    "Do I need IELTS to study at Aston if I did KCSE English?":
        [(1, ["ielts"]), (2, ["kcse"]), (5, ["aston"])],
    "What IELTS score do I need for a masters at RGU?":
        [(1, ["ielts"]), (3, ["masters"]), (5, ["rgu"])],
}


@pytest.mark.parametrize("query,expected", list(REPORTED.items()))
def test_matched_groups_for_the_reported_queries(query, expected):
    assert G.matched_intent_groups(query) == expected
    assert G.is_multi_intent(query) is True


def test_an_institution_name_counts_as_an_intent_group():
    """Why 'both name a specific institution' matters: drop the institution and
    the same question is no longer multi-intent."""
    assert G.is_multi_intent("What IELTS score do I need for a masters?") is False
    assert G.is_multi_intent("What IELTS score do I need for a masters at RGU?") is True
    assert (5, ["rgu"]) in G.matched_intent_groups(
        "What IELTS score do I need for a masters at RGU?")


def test_the_threshold_is_more_than_two_groups():
    assert G.THRESHOLD == 2
    assert G.is_multi_intent("Do I need IELTS at Aston?") is False    # 2 groups


def test_stock_phrase_and_word_boundaries_are_respected():
    # keywords are reported in the group's own order ("intake" is listed first)
    assert G.matched_intent_groups("Of course, I want the September intake") == [
        (4, ["intake", "september"])]
    assert G.matched_intent_groups("will an upgrade help") == []


def test_no_match_is_an_empty_list():
    assert G.matched_intent_groups("hello there") == []


def test_only_keywords_that_matched_are_reported():
    (idx, kws), = G.matched_intent_groups("what is the deadline")
    assert idx == 4 and kws == ["deadline"]


def test_the_dashboard_uses_this_guard_not_a_copy():
    import dashboard.app as app
    assert app._is_multi_intent is G.is_multi_intent
    assert app._INTENT_GROUPS is G.INTENT_GROUPS
