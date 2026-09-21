"""Regression tests for dashboard/app.py logic (no Streamlit runtime needed).

The dashboard had no tests at all; the review found that it silently used
TF-IDF (bad import swallowed by ``except ImportError``), scored the lead
*before* recording the turn, forced abstention with a nonsense query and
matched intent keywords as substrings ("of course" -> a course intent).
"""
from datetime import datetime, timezone

import pytest

import dashboard.app as app
from chatbot.session_features import SessionTracker

UTC = timezone.utc


class _Entry:
    def __init__(self, question="How do I apply?", category="Application Process",
                 institution="General"):
        self.question = question
        self.answer = "Apply through the portal."
        self.category = category
        self.institution = institution
        self.keywords = []
        self.index = 0


class _Retriever:
    """Records every query it is asked, always returns one confident hit."""

    def __init__(self, score=0.95, category="Application Process"):
        self.entries = [_Entry(category=category)]
        self.score = score
        self.queries = []

    def search(self, query, top_k=3):
        self.queries.append(query)
        return [(self.entries[0], self.score)]


def _tracker_with(*cats):
    t = SessionTracker("LEAD-TEST", datetime(2026, 9, 21, tzinfo=UTC))
    for i, c in enumerate(cats):
        t.add_turn("word " * 10, {"category": c, "confidence": 0.9},
                   datetime(2026, 9, 21, 12, i, tzinfo=UTC))
    return t


# --------------------------------------------------------------------------- #
# multi-intent guard
# --------------------------------------------------------------------------- #
def test_multi_intent_is_not_triggered_by_substrings():
    assert app._is_multi_intent(
        "Of course, can you tell me about the September intake for Aston?"
    ) is False
    assert app._is_multi_intent(
        "Will an upgrade help with the Aston September intake?") is False


def test_multi_intent_still_catches_compound_queries():
    assert app._is_multi_intent(
        "I have a passport, IELTS 6.5 and a KCSE C+, and want data science "
        "at Aston in September") is True


def test_single_intent_query_is_not_multi_intent():
    assert app._is_multi_intent("Do I need IELTS?") is False


def test_multi_intent_abstains_on_the_real_query_not_a_nonsense_string():
    q = ("I have a passport, IELTS 6.5 and a KCSE C+, and want data science "
         "at Aston in September")
    r = _Retriever(score=0.99)
    out = app._answer_query(q, _tracker_with(), r)
    assert r.queries == [q], "the visitor's query must be what is searched"
    assert out["cited_question"] is None
    assert out["escalate"] is True
    assert out["query"] == q


# --------------------------------------------------------------------------- #
# retriever selection and gate
# --------------------------------------------------------------------------- #
def test_get_retriever_prefers_sbert_when_installed():
    from chatbot.embeddings import available
    if not available():
        pytest.skip("sentence-transformers not installed")
    r = app._get_retriever()
    assert type(r).__name__ == "SbertRetriever", (
        "dashboard silently fell back to TF-IDF")
    assert app._backend_of(r) == "sbert"
    hits = r.search("Do I need IELTS to study in the UK?", top_k=1)
    assert hits and hits[0][1] > 0.3


def test_get_retriever_falls_back_to_tfidf_when_sbert_cannot_start(monkeypatch):
    import chatbot.embeddings as E

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(E, "SbertRetriever", _Boom)
    r = app._get_retriever()
    assert type(r).__name__ == "Retriever"
    assert app._backend_of(r) == "tfidf"


def test_gate_is_specific_to_the_retrieval_backend():
    from chatbot.responder import EVALUATED_GATE, EscalationPolicy
    assert app._gate_for("sbert") == EVALUATED_GATE == 0.60
    assert app._gate_for("tfidf") == EscalationPolicy().min_confidence


# --------------------------------------------------------------------------- #
# per-turn scoring order and escalation
# --------------------------------------------------------------------------- #
def test_escalation_uses_the_lead_score_including_the_current_turn(monkeypatch):
    """One more intent-bearing question can turn a Warm lead Hot *now*."""
    import leads.hybrid as H
    monkeypatch.setattr(H, "predict_ml_proba",
                        lambda ev, engagement=None, config=None: [0.0, 0.0, 1.0])
    t = _tracker_with("Entry Requirements")
    before = app._score_turn(t)
    assert before["label"] == "Warm"           # premise: not Hot yet

    out = app._answer_query("how do I apply?", t,
                            _Retriever(category="Application Process"))
    assert out["lead_label"] == "Hot"
    assert out["escalate"] is True
    assert out["priority"] == "high"


def test_low_confidence_still_escalates_with_normal_priority(monkeypatch):
    import leads.hybrid as H
    monkeypatch.setattr(H, "predict_ml_proba",
                        lambda ev, engagement=None, config=None: None)
    out = app._answer_query("blah", _tracker_with(), _Retriever(score=0.01))
    assert out["cited_question"] is None
    assert out["escalate"] is True and out["priority"] == "normal"


def test_answer_records_gate_and_backend_for_the_transcript(monkeypatch):
    import leads.hybrid as H
    monkeypatch.setattr(H, "predict_ml_proba",
                        lambda ev, engagement=None, config=None: None)
    out = app._answer_query("how do I apply?", _tracker_with(), _Retriever())
    assert out["retrieval_backend"] == "tfidf"     # fake retriever: no SBERT
    assert out["gate"] == app._gate_for("tfidf")


# --------------------------------------------------------------------------- #
# exports carry the decision, not just the text
# --------------------------------------------------------------------------- #
def _entry_with_decision():
    return {
        "lead_id": "LEAD-20260921-0001",
        "final_label": "Hot", "final_hybrid_score": 0.78,
        "final_rule_score": 0.56,
        "chat": [{"role": "user", "text": "how do I apply?"},
                 {"role": "assistant", "text": "Apply through the portal."}],
        "turns_log": [{
            "user_msg": "how do I apply?", "category": "Application Process",
            "confidence": 0.9, "assistant_text": "Apply through the portal.",
            "ts": "2026-09-21T12:00:00+00:00",
            "escalate": True, "priority": "high",
            "lead_label": "Hot", "lead_score": 0.78,
            "retrieval_backend": "sbert",
        }],
    }


def test_transcript_rows_expose_escalation_and_lead_scores():
    row = app.transcript_rows(_entry_with_decision())[0]
    assert row["escalate"] is True
    assert row["priority"] == "high"
    assert row["lead_label"] == "Hot"
    assert row["lead_score"] == 0.78
    assert row["retrieval_backend"] == "sbert"


def test_csv_and_markdown_exports_include_the_decision_columns():
    entry = _entry_with_decision()
    header = app.transcript_to_csv(entry).splitlines()[0].split(",")
    for col in ("escalate", "priority", "lead_label", "lead_score",
                "retrieval_backend"):
        assert col in header
    md = app.transcript_to_markdown(entry)
    assert "escalat" in md.lower() and "Hot" in md


def test_old_entries_without_decision_fields_still_export():
    entry = {"lead_id": "L", "chat": [], "turns_log": [{
        "user_msg": "hi", "category": "General Enquiries", "confidence": 0.5,
        "assistant_text": "hello", "ts": "2026-09-21T12:00:00+00:00"}]}
    row = app.transcript_rows(entry)[0]
    assert row["escalate"] == "" and row["lead_label"] == ""
    assert "hello" in app.transcript_to_csv(entry)


# --------------------------------------------------------------------------- #
# terminal noise: Streamlit's file watcher probing transformers' lazy modules
# --------------------------------------------------------------------------- #
def _watcher_record(msg):
    import logging
    return logging.LogRecord("streamlit.watcher.local_sources_watcher",
                             logging.WARNING, __file__, 1, msg, None, None)


def test_watcher_noise_filter_drops_only_the_transformers_probe_errors():
    """With SBERT loaded, the watcher logged ~400 tracebacks (torchvision is not
    installed and transformers' lazy aliases raise on ``hasattr(m, '__path__')``)."""
    import logging
    app._quiet_watcher_noise()
    app._quiet_watcher_noise()                        # idempotent
    lg = logging.getLogger("streamlit.watcher.local_sources_watcher")
    assert sum(isinstance(f, app._WatcherNoiseFilter) for f in lg.filters) == 1

    noisy = _watcher_record("Examining the path of "
                            "transformers.models.aria.image_processing_aria_fast raised:")
    assert not lg.filter(noisy)
    # a genuine problem with the app's own modules must still be reported
    ours = _watcher_record("Examining the path of chatbot.retriever raised:")
    assert lg.filter(ours)
    other = _watcher_record("Something else from the watcher")
    assert lg.filter(other)


# --------------------------------------------------------------------------- #
# browser-console noise: the category chart's Vega-Lite spec
# --------------------------------------------------------------------------- #
class _FakeSt:
    def __init__(self):
        self.charts, self.written = [], []

    def altair_chart(self, chart, **kw):
        self.charts.append(chart)

    def write(self, value):
        self.written.append(value)


def test_category_chart_spec_has_no_scale_bindings_and_no_discrete_height():
    """Both are Vega-Lite console warnings on Streamlit's chart:

    * ``st.bar_chart`` adds zoom/pan *scale bindings* (``params``), unsupported on
      a categorical axis;
    * a discrete (``{"step": n}``) height clashes with Streamlit's ``fit-y``
      autosize ("Dropping fit-y because spec has discrete height").
    """
    fake = _FakeSt()
    app._category_chart(fake, {"English Language": 2, "Fees & Funding": 1,
                               "Visa & Immigration": 1})
    spec = fake.charts[0].to_dict()
    assert "params" not in spec
    assert isinstance(spec["height"], (int, float)), spec["height"]
    assert spec["width"] == "container"
    assert spec["mark"]["type"] == "bar" if isinstance(spec["mark"], dict) \
        else spec["mark"] == "bar"


def test_category_chart_height_grows_with_the_number_of_categories():
    a, b = _FakeSt(), _FakeSt()
    app._category_chart(a, {"A": 1})
    app._category_chart(b, {c: 1 for c in "ABCDEFGH"})
    assert b.charts[0].to_dict()["height"] > a.charts[0].to_dict()["height"]
