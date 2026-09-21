"""The multi-intent guard: which keyword groups count as separate intents.

An institution name ("at Aston", "at RGU") names the *scope* of a question, not a
second question, so it is not an intent group. While it was one, "What IELTS
score do I need for a masters at RGU?" scored three groups (document + level +
institution) and abstained at a displayed confidence of 0.81 against a 0.60 gate.
"""
import pytest

from chatbot import intent_guard as G

#: single questions that name an institution: two topics, so not multi-intent
REPORTED = {
    "Do I need IELTS to study at Aston if I did KCSE English?":
        [(1, ["ielts"]), (2, ["kcse"])],
    "What IELTS score do I need for a masters at RGU?":
        [(1, ["ielts"]), (3, ["masters"])],
}
#: three distinct topics (passport, English test, programme): still multi-intent
COMPOUND = ("I have a Kenyan passport, IELTS 6.5, and want to study MSc Data "
            "Science at Aston")
INSTITUTIONS = ("aston", "bcu", "usw", "rgu", "herts", "salford", "uclan")


@pytest.mark.parametrize("query,expected", list(REPORTED.items()))
def test_a_single_question_naming_an_institution_is_not_multi_intent(query, expected):
    assert G.matched_intent_groups(query) == expected
    assert G.is_multi_intent(query) is False


def test_three_distinct_topics_are_still_multi_intent():
    assert G.matched_intent_groups(COMPOUND) == [
        (0, ["passport"]), (1, ["ielts"]), (3, ["data science"])]
    assert G.is_multi_intent(COMPOUND) is True


@pytest.mark.parametrize("name", INSTITUTIONS)
def test_an_institution_name_is_a_scope_qualifier_not_an_intent(name):
    base = "What IELTS score do I need for a masters"
    assert G.matched_intent_groups("%s at %s?" % (base, name.upper())) == \
        G.matched_intent_groups(base + "?")
    assert G.matched_intent_groups(name) == []


def test_no_intent_group_is_made_of_institution_names():
    for group in G.INTENT_GROUPS:
        assert not set(group) & set(INSTITUTIONS), group


@pytest.mark.parametrize("query", [COMPOUND, *REPORTED])
def test_naming_an_institution_never_changes_the_verdict(query):
    without = query.replace(" at Aston", "").replace(" at RGU", "")
    assert G.is_multi_intent(query) == G.is_multi_intent(without)
    assert G.matched_intent_groups(query) == G.matched_intent_groups(without)


def test_the_threshold_is_more_than_two_topics():
    assert G.THRESHOLD == 2
    assert G.is_multi_intent("Do I need IELTS for a masters?") is False        # 2
    assert G.is_multi_intent(
        "I need IELTS, a masters and the September intake") is True            # 3


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
