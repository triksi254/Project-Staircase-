"""respond() must say *why* it abstained / escalated.

A session where every query escalated could not be explained from the UI: the
footer shows the retrieval confidence, but an abstention can also be forced by
the dashboard's multi-intent guard (``force_abstain``) regardless of confidence
(e.g. displayed confidence 0.81 against a 0.60 gate). ``respond`` therefore logs,
at DEBUG, the exact value compared with ``min_confidence`` and every condition
that fired.
"""
import logging

import pytest

from chatbot.responder import EscalationPolicy, escalation_reasons, respond


class _Entry:
    question, answer, category, institution = "Q?", "A.", "Program Details", "General"


class _Retriever:
    def __init__(self, score, has_hits=True):
        self.score, self.has_hits = score, has_hits
        self.entries = [_Entry()]

    def search(self, query, top_k=3):
        return [(_Entry(), self.score)] if self.has_hits else []


def _call(caplog, score, *, force=False, rule=0.0, has_hits=True, gate=0.60):
    with caplog.at_level(logging.DEBUG, logger="chatbot.responder"):
        out = respond("what IELTS at RGU?", _Retriever(score, has_hits),
                      rule_score=rule, force_abstain=force,
                      policy=EscalationPolicy(min_confidence=gate, alpha=1.0))
    recs = [r for r in caplog.records
            if r.name == "chatbot.responder" and r.levelno == logging.DEBUG]
    assert len(recs) == 1, "exactly one debug line per respond() call"
    return out, recs[0].getMessage()


@pytest.mark.parametrize("score,kw,reason", [
    (0.81, {"force": True}, "multi_intent_guard"),         # high score, guard fires
    (0.35, {}, "low_confidence"),                           # below the gate
    (0.0, {"has_hits": False}, "no_retrieval_hits"),
    (0.95, {"rule": 1.0}, "hot_lead"),
    (0.95, {}, "none"),
    (0.35, {"force": True}, "multi_intent_guard+low_confidence"),
])
def test_debug_line_names_the_condition_that_fired(caplog, score, kw, reason):
    _, msg = _call(caplog, score, **kw)
    assert "escalation_reason=%s" % reason in msg


def test_debug_line_reports_the_exact_compared_value_and_inputs(caplog):
    _, msg = _call(caplog, 0.8123, force=True)
    assert "query='what IELTS at RGU?'" in msg
    assert "top1_score=0.8123" in msg
    assert "min_confidence=0.6" in msg
    assert "multi_intent_flag=True" in msg


def test_a_confident_answer_is_not_blamed_on_the_gate(caplog):
    """The reported case: confidence 0.81 >= gate 0.60 yet abstained."""
    out, msg = _call(caplog, 0.81, force=True)
    assert out["abstained"] is True and out["escalate"] is True
    assert "low_confidence" not in msg and "multi_intent_guard" in msg


def test_logging_does_not_change_the_decision(caplog):
    quiet = respond("q", _Retriever(0.9), rule_score=0.2,
                    policy=EscalationPolicy(min_confidence=0.6, alpha=1.0))
    out, _ = _call(caplog, 0.9, rule=0.2)
    assert {k: out[k] for k in quiet if k != "query"} == \
           {k: quiet[k] for k in quiet if k != "query"}


def test_escalation_reasons_helper_is_pure():
    assert escalation_reasons(force_abstain=False, has_hits=True, confidence=0.7,
                              min_confidence=0.6, label="Cold") == ["none"]
    assert escalation_reasons(force_abstain=True, has_hits=True, confidence=0.2,
                              min_confidence=0.6, label="Hot") == [
        "multi_intent_guard", "low_confidence", "hot_lead"]
