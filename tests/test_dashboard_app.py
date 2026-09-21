"""End-to-end: run the real ``dashboard/app.py`` headlessly with Streamlit's AppTest.

The dashboard previously had no automated test at all; its defects (silent
TF-IDF fallback, stale escalation, gibberish abstention) were invisible to the
suite. These tests execute the actual script, chat input included.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
LABELS = {"Cold", "Warm", "Hot"}


def _start():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def _captions(at):
    return [c.value for c in at.caption]


def _metric(at):
    return at.metric[0]


def test_app_starts_empty_without_exceptions():
    at = _start()
    assert not at.exception, [e.value for e in at.exception]
    assert _metric(at).value == "0.000"
    assert _metric(at).delta == "Awaiting first message"
    assert any(c.startswith("Retrieval: ") for c in _captions(at))
    assert any("Provisional live scoring" in c for c in _captions(at))
    assert not any("Rule-only live scoring" in c for c in _captions(at))


def test_retrieval_backend_is_shown_and_matches_the_gate():
    at = _start()
    line = next(c for c in _captions(at) if c.startswith("Retrieval: "))
    from chatbot.embeddings import available
    from chatbot.responder import DEFAULT_MIN_CONFIDENCE, EVALUATED_GATE
    if available():
        assert line == "Retrieval: sbert · abstains below %.2f" % EVALUATED_GATE
        assert not at.warning
    else:
        assert line == "Retrieval: tfidf · abstains below %.2f" % DEFAULT_MIN_CONFIDENCE
        assert at.warning


def test_a_chat_turn_scores_the_lead_and_records_the_decision():
    at = _start()
    at.chat_input[0].set_value("Do I need IELTS to study in the UK?").run()
    assert not at.exception, [e.value for e in at.exception]

    log = at.session_state["turns_log"]
    assert len(log) == 1
    turn = log[0]
    for key in ("escalate", "priority", "lead_label", "lead_score",
                "retrieval_backend"):
        assert key in turn, "transcript is missing %r" % key
    assert turn["lead_label"] in LABELS
    assert turn["priority"] in {"none", "normal", "high"}
    assert turn["category"]                       # the answer/abstention has a category

    assert _metric(at).value != "0.000"           # panel left the empty state
    assert _metric(at).delta in LABELS
    assert _metric(at).delta == turn["lead_label"]


def test_english_question_registers_as_observable_evidence():
    """Asking about English earns the chat-observable 'english_test' credit."""
    at = _start()
    at.chat_input[0].set_value("What IELTS score do I need?").run()
    assert not at.exception
    text = " ".join(w.value for w in at.markdown) + " ".join(
        w.value for w in at.text)
    assert "english_test" in text or at.session_state["turns_log"][0][
        "category"] in {"English Language", "Entry Requirements"}
    # Unobservable fields must not appear as rubric lines in the live panel.
    assert "passport_status" not in text
    assert "qual_level" not in text


def test_multi_intent_query_abstains_and_escalates_on_the_real_text():
    at = _start()
    q = ("I have a passport, IELTS 6.5 and a KCSE C+, and want data science "
         "at Aston in September")
    at.chat_input[0].set_value(q).run()
    assert not at.exception
    turn = at.session_state["turns_log"][0]
    assert turn["user_msg"] == q
    assert turn["escalate"] is True
    assert "flagged your question for a counsellor" in turn["assistant_text"]


@pytest.mark.parametrize("q", [
    "Do I need IELTS to study at Aston if I did KCSE English?",
    "What IELTS score do I need for a masters at RGU?",
])
def test_a_single_question_naming_an_institution_is_answered_not_escalated(q):
    """Reported session: both escalated at a displayed confidence of 0.81 (gate
    0.60) because the institution counted as a third intent."""
    at = _start()
    at.chat_input[0].set_value(q).run()
    assert not at.exception, [e.value for e in at.exception]
    turn = at.session_state["turns_log"][0]
    assert turn["user_msg"] == q
    assert turn["escalate"] is False
    assert "flagged your question for a counsellor" not in turn["assistant_text"]


def test_close_lead_archives_the_transcript_with_decisions():
    at = _start()
    at.chat_input[0].set_value("Do I need IELTS to study in the UK?").run()
    at.button[0].click().run()                    # "Close Lead & Start New"
    assert not at.exception
    archive = at.session_state["leads_archive"]
    assert len(archive) == 1
    assert archive[0]["turns_log"][0]["lead_label"] in LABELS
    assert _metric(at).value == "0.000"           # fresh lead


# --------------------------------------------------------------------------- #
# nothing stray on the page (Streamlit "magic" renders bare expressions)
# --------------------------------------------------------------------------- #
def _walk(node):
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _walk(child)


def _all_text(at):
    return "\n".join(v for v in (getattr(el, "value", None) for el in _walk(at.main))
                     if isinstance(v, str))


def _top_level_markdown(at):
    return [c.value for c in at.main.children.values()
            if type(c).__name__ == "Markdown"]


def test_no_stray_model_repr_is_rendered_on_the_page():
    """A bare ``retriever.model`` expression used to render the whole
    SentenceTransformer repr into the app (Streamlit magic)."""
    at = _start()
    assert "SentenceTransformer(" not in _all_text(at)
    assert not [m for m in _top_level_markdown(at) if m.lstrip().startswith("```")], \
        "a code block was rendered at page level"


def test_page_stays_clean_after_a_chat_turn():
    at = _start()
    at.chat_input[0].set_value("Do I need IELTS to study in the UK?").run()
    assert not at.exception
    assert "SentenceTransformer(" not in _all_text(at)


# --------------------------------------------------------------------------- #
# the rubric panel must add up to the rule score it sits next to
# --------------------------------------------------------------------------- #
def test_rubric_panel_shows_the_arithmetic_behind_the_rule_score():
    from datetime import datetime

    from chatbot.session_features import SessionTracker, live_rule_breakdown

    at = _start()
    at.chat_input[0].set_value("Do I need IELTS to study in the UK?").run()
    assert not at.exception
    tr = SessionTracker("x", datetime.fromisoformat(
        at.session_state["tracker_state"]["started_at"]))
    for lg in at.session_state["turns_log"]:
        tr.add_turn(lg["user_msg"], {"category": lg["category"],
                                     "confidence": lg["confidence"]},
                    datetime.fromisoformat(lg["ts"]))
    bd = live_rule_breakdown(tr.rubric_evidence())
    text = _all_text(at)
    assert "= %.3f ÷ %.3f = %.3f" % (bd["raw"], bd["mass"], bd["score"]) in text
    assert "Rule score: **%.3f**" % bd["score"] in text
    assert "(None)" not in text                    # no raw None in the panel
