"""Single-source-of-truth checks for constants that used to be duplicated.

The review found the same quantity defined in several modules with different
values (abstention gate 0.15 vs 0.60, keyword bonus 0.05 vs 0.10, hybrid cuts
vs rubric cuts, alpha hard-coded four times, feature lists copied between
``train_ml`` and ``hybrid``). These tests fail loudly if they drift again.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# abstention gate
# --------------------------------------------------------------------------- #
def test_evaluated_gate_is_one_value_everywhere():
    from chatbot.responder import EVALUATED_GATE
    from evaluation import adaptation_loop, retrieval_eval
    assert EVALUATED_GATE == 0.60
    assert retrieval_eval.DEFAULT_THRESHOLD == EVALUATED_GATE
    assert adaptation_loop.DEFAULT_THRESHOLD == EVALUATED_GATE


def test_default_policy_gate_stays_low_for_the_tfidf_demo():
    """The CLI demo / TF-IDF default is a separate, documented operating point."""
    from chatbot.responder import DEFAULT_MIN_CONFIDENCE, EscalationPolicy
    assert EscalationPolicy().min_confidence == DEFAULT_MIN_CONFIDENCE == 0.15


# --------------------------------------------------------------------------- #
# keyword bonus
# --------------------------------------------------------------------------- #
def test_keyword_bonus_constant_matches_behaviour_and_docs():
    from sklearn.metrics.pairwise import cosine_similarity

    from chatbot import retriever
    from chatbot.retriever import KEYWORD_BONUS, FaqEntry, Retriever
    from evaluation import retrieval_eval

    assert KEYWORD_BONUS == 0.10
    assert ("%.2f" % KEYWORD_BONUS) in retrieval_eval.__doc__
    assert "0.05" not in retrieval_eval.__doc__

    r = Retriever([FaqEntry(question="Visa question", answer="visa answer here",
                            keywords=["visa"]),
                   FaqEntry(question="Fees", answer="tuition details",
                            keywords=["fees"])])
    entry, score = r.search("visa", top_k=1)[0]
    raw = float(cosine_similarity(r._vectorizer.transform(["visa"]),
                                  r._matrix)[0].max())
    # reported score = raw cosine + one keyword hit, exactly KEYWORD_BONUS
    assert score == pytest.approx(min(1.0, round(raw, 4) + KEYWORD_BONUS), abs=1e-4)
    assert "cosine" in retriever.__doc__.lower()


def test_retriever_construction_does_not_write_caches_by_default(tmp_path,
                                                                 monkeypatch):
    from chatbot import retriever
    from chatbot.retriever import FaqEntry, Retriever

    (tmp_path / "data").mkdir()
    monkeypatch.setattr(retriever, "PROJECT_ROOT", tmp_path)
    Retriever([FaqEntry(question="q one", answer="a one"),
               FaqEntry(question="q two", answer="a two")])
    assert list((tmp_path / "data").iterdir()) == []


def test_retriever_persistence_is_opt_in(tmp_path, monkeypatch):
    from chatbot import retriever
    from chatbot.retriever import FaqEntry, Retriever

    (tmp_path / "data").mkdir()
    monkeypatch.setattr(retriever, "PROJECT_ROOT", tmp_path)
    Retriever([FaqEntry(question="q one", answer="a one"),
               FaqEntry(question="q two", answer="a two")], persist=True)
    assert (tmp_path / "data" / "tfidf_vectorizer.pkl").is_file()


# --------------------------------------------------------------------------- #
# hybrid cuts, escalation and alpha
# --------------------------------------------------------------------------- #
def test_escalation_hot_threshold_is_the_label_cut():
    from chatbot.responder import EscalationPolicy
    from leads.hybrid import WARM_HOT_CUT, score_to_label
    assert EscalationPolicy().hot_threshold == WARM_HOT_CUT
    assert score_to_label(WARM_HOT_CUT) == "Hot"
    assert score_to_label(WARM_HOT_CUT - 1e-9) == "Warm"


def test_rubric_and_hybrid_labels_differ_only_in_two_documented_bands():
    """rubric cuts (0.35/0.68) vs hybrid tertiles: bound the disagreement."""
    from leads.hybrid import COLD_WARM_CUT, WARM_HOT_CUT, score_to_label
    from leads.rubric import COLD_HOT_THRESHOLD, WARM_HOT_THRESHOLD
    assert COLD_WARM_CUT < COLD_HOT_THRESHOLD and WARM_HOT_CUT < WARM_HOT_THRESHOLD

    def rubric_label(s):
        return ("Hot" if s >= WARM_HOT_THRESHOLD
                else "Cold" if s < COLD_HOT_THRESHOLD else "Warm")

    disagree = [s / 1000 for s in range(1001)
                if rubric_label(s / 1000) != score_to_label(s / 1000)]
    assert disagree, "cuts are expected to differ; if not, drop this test"
    assert all(COLD_WARM_CUT <= s < COLD_HOT_THRESHOLD
               or WARM_HOT_CUT <= s < WARM_HOT_THRESHOLD for s in disagree)


def test_alpha_has_one_default():
    from chatbot.responder import EscalationPolicy
    from chatbot.session_features import SessionTracker
    from leads.hybrid import DEFAULT_ALPHA, hybrid_score
    import inspect
    from datetime import datetime

    assert DEFAULT_ALPHA == 0.5
    assert EscalationPolicy().alpha == DEFAULT_ALPHA
    sig = inspect.signature(SessionTracker.hybrid)
    assert sig.parameters["alpha"].default == DEFAULT_ALPHA
    t = SessionTracker("L", datetime(2026, 9, 21))
    t.add_turn("hi", {"category": "English Language"}, datetime(2026, 9, 21, 1))
    assert t.hybrid([0.2, 0.3, 0.5]) == pytest.approx(
        hybrid_score(t.rule_score(), [0.2, 0.3, 0.5], DEFAULT_ALPHA))


def test_demo_default_alpha_reads_the_shipped_config():
    from chatbot.demo import _default_alpha
    cfg = json.loads((REPO / "artifacts" / "config.json")
                     .read_text(encoding="utf-8"))
    assert _default_alpha() == cfg["alpha"]


# --------------------------------------------------------------------------- #
# mirrored feature lists and label encodings
# --------------------------------------------------------------------------- #
def test_hybrid_mirrors_train_ml_feature_lists():
    from leads import hybrid, train_ml
    assert hybrid.NUMERIC == train_ml.NUMERIC
    assert hybrid.CATEGORICAL == train_ml.CATEGORICAL


def test_known_cat_values_use_the_training_encoding():
    """Categoricals are int-coded in leads.features; names must match."""
    from leads import features, hybrid
    assert hybrid.KNOWN_CAT_VALUES["passport_status"] == ["0", "1", "2"]
    assert hybrid.KNOWN_CAT_VALUES["qual_level"] == [
        str(i) for i in range(len(features.QUAL_LEVEL_ORDER))]
    for feat in ("destination_uk", "has_course", "has_intake",
                 "study_gap_mentioned", "previous_application_mentioned"):
        assert hybrid.KNOWN_CAT_VALUES[feat] == ["0", "1"]


def test_category_taxonomy_is_identical_across_modules():
    from chatbot import classifier, session_features
    from leads import personas
    corpus = json.loads((REPO / "data" / "faq_corpus.json")
                        .read_text(encoding="utf-8"))["faqs"]
    in_corpus = {e["category"] for e in corpus}
    sets = [set(classifier.CATEGORIES), set(session_features.KNOWN_CATEGORIES),
            set(personas.QUESTION_CATEGORIES)]
    assert sets[0] == sets[1] == sets[2] == in_corpus


# --------------------------------------------------------------------------- #
# responder API used by the dashboard
# --------------------------------------------------------------------------- #
class _R:
    class _E:
        question, answer, category, institution = "Q?", "A.", "Program Details", "General"
    entries = [_E()]

    def search(self, query, top_k=3):
        return [(self._E(), 0.99)]


def test_respond_force_abstain_escalates_without_a_fake_query():
    from chatbot.responder import ABSTAIN_MESSAGE, respond
    out = respond("a perfectly good question", _R(), rule_score=0.1,
                  force_abstain=True)
    assert out["answer"] == ABSTAIN_MESSAGE
    assert out["cited_question"] is None
    assert out["escalate"] is True and out["priority"] == "normal"
    assert out["query"] == "a perfectly good question"


def test_respond_without_force_abstain_is_unchanged():
    from chatbot.responder import respond
    out = respond("a perfectly good question", _R(), rule_score=0.1)
    assert out["cited_question"] == "Q?" and out["escalate"] is False


@pytest.mark.parametrize("label,abstained,expected", [
    ("Hot", False, (True, "high")), ("Hot", True, (True, "high")),
    ("Warm", True, (True, "normal")), ("Cold", True, (True, "normal")),
    ("Warm", False, (False, "none")), ("Cold", False, (False, "none")),
])
def test_decide_escalation_matrix(label, abstained, expected):
    from chatbot.responder import decide_escalation
    assert decide_escalation(label, abstained) == expected
