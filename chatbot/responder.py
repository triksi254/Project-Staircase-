"""Responder: grounded answers + lead-aware counsellor escalation.

Policy: answer from the top-ranked FAQ (with its corpus question cited so
replies stay grounded); retrieval confidence below ``min_confidence``
yields an abstention + escalation instead of a guess. Hot leads always
escalate with priority so a counsellor follows up, even when the answer
itself is confident.

Two signals are deliberately separate and must not be conflated:

* ``confidence`` is the **retrieval** score: the top-1 score returned by
  ``search``. For the TF-IDF ``Retriever`` that is the raw cosine *plus* a
  ``KEYWORD_BONUS`` per matched corpus keyword (not a pure cosine); for the
  Sentence-BERT retriever it is the raw cosine. It is compared against
  ``EscalationPolicy.min_confidence`` on every path and drives abstention vs
  grounded answer.
* ``lead_score`` / ``lead_label`` is the **lead** score: ``hybrid_score``
  of the rule and ML components, mapped to Cold/Warm/Hot by
  ``score_to_label``. It drives counsellor escalation, not abstention.

Consequence: a confident-looking answer may still be an abstention if the
retrieval similarity is below the gate, and a well-grounded answer may still
escalate if the lead is Hot. ``min_confidence`` therefore gates retrieval
quality only -- it is *not* a threshold on the lead score.

Retrieval is deliberately corpus-wide: ``respond`` never pre-filters or
re-ranks the retriever by institution, so the top-1 hit can come from any
corpus entry (a query naming a university does not scope the search) and the
``min_confidence`` gate stays the only answer/abstain decision. A caller that
knows a query must not be answered (the dashboard's multi-intent guard) passes
``force_abstain=True`` instead of substituting a different query.

Hot classification is delegated to ``score_to_label`` (tertile cut,
``leads.hybrid.WARM_HOT_CUT``). ``EscalationPolicy.hot_threshold`` is that same
constant, kept for callers, serialisation and display; a test pins the
equality so the two cannot drift.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from chatbot.retriever import FaqEntry, Retriever
from leads.hybrid import (
    DEFAULT_ALPHA,
    WARM_HOT_CUT,
    hybrid_score,
    score_to_label,
)

_LOG = logging.getLogger(__name__)

ABSTAIN_MESSAGE = (
    "I don't have a verified answer for that yet. "
    "I've flagged your question for a counsellor who will follow up."
)

#: Gate of the TF-IDF demo / CLI default (``chatbot.demo``): permissive, so the
#: keyword-boosted TF-IDF score can clear it.
DEFAULT_MIN_CONFIDENCE = 0.15

#: Operating point evaluated in ``evaluation/`` (H1 study) and used by the
#: dashboard with Sentence-BERT. It is meaningful on the SBERT scale (raw
#: cosine). Applied to TF-IDF scores it withholds 90-100% of answerable gold
#: queries (``artifacts/compare_gold_*.json``), so a TF-IDF fallback uses
#: ``DEFAULT_MIN_CONFIDENCE`` instead. One definition; ``retrieval_eval`` and
#: ``adaptation_loop`` import it.
EVALUATED_GATE = 0.60


@dataclass
class EscalationPolicy:
    """Thresholds for abstention and counsellor handoff."""
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    hot_threshold: float = WARM_HOT_CUT
    alpha: float = DEFAULT_ALPHA


def decide_escalation(label: str, abstained: bool) -> Tuple[bool, str]:
    """``(escalate, priority)``: Hot always escalates; abstentions escalate."""
    escalate = bool(abstained) or label == "Hot"
    priority = "high" if label == "Hot" else ("normal" if escalate else "none")
    return escalate, priority


def escalation_reasons(*, force_abstain: bool, has_hits: bool, confidence: float,
                        min_confidence: float, label: str) -> List[str]:
    """Every condition that makes ``respond`` abstain or escalate (``["none"]`` if
    there is none). Pure; used for the debug line, not for the decision itself.

    * ``multi_intent_guard`` -- the caller forced an abstention (``force_abstain``),
      whatever the retrieval score;
    * ``no_retrieval_hits`` -- the retriever returned nothing;
    * ``low_confidence`` -- top-1 score < ``min_confidence``;
    * ``hot_lead`` -- the lead label is Hot (escalates even when answering).
    """
    reasons: List[str] = []
    if force_abstain:
        reasons.append("multi_intent_guard")
    if not has_hits:
        reasons.append("no_retrieval_hits")
    elif confidence < min_confidence:
        reasons.append("low_confidence")
    if label == "Hot":
        reasons.append("hot_lead")
    return reasons or ["none"]


def respond(
    query: str,
    retriever: Retriever,
    rule_score: float = 0.5,
    ml_score: Optional[float] = None,
    policy: Optional[EscalationPolicy] = None,
    top_k: int = 3,
    force_abstain: bool = False,
) -> Dict[str, Any]:
    """Answer ``query``; return dict with answer + routing signals.

    Retrieval is corpus-wide: ``retriever.search`` ranks the **full** corpus
    (no institution scoping/filtering before ranking) and the resulting top-1
    score is then gated by ``min_confidence``. A query therefore either gets
    the grounded top-1 FAQ with its question cited, or abstains + escalates;
    naming a university in the query does not change the ranking.
    ``force_abstain`` abstains regardless of the score (the retrieval result is
    still reported so the decision is auditable).
    """
    pol = policy or EscalationPolicy()
    lead = hybrid_score(rule_score, ml_score, alpha=pol.alpha)
    label = score_to_label(lead)

    hits = retriever.search(query, top_k=top_k)
    confidence = hits[0][1] if hits else 0.0

    abstained = bool(force_abstain) or (not hits) or (confidence < pol.min_confidence)
    escalate, priority = decide_escalation(label, abstained)
    # ``top1_score`` is the exact value compared with ``min_confidence``;
    # ``multi_intent_flag`` is ``force_abstain`` (how the dashboard's multi-intent
    # guard signals it). Enable with logging.getLogger("chatbot.responder").
    _LOG.debug(
        "respond: query=%r top1_score=%r min_confidence=%r multi_intent_flag=%s "
        "escalation_reason=%s",
        query, confidence, pol.min_confidence, bool(force_abstain),
        "+".join(escalation_reasons(
            force_abstain=bool(force_abstain), has_hits=bool(hits),
            confidence=confidence, min_confidence=pol.min_confidence,
            label=label)))
    if abstained:
        answer = ABSTAIN_MESSAGE
        cited: Optional[str] = None
        category: Optional[str] = None
        try:
            from chatbot.classifier import predict as _predict
            category = _predict(query)
        except Exception:
            category = "General Enquiries"
    else:
        top = hits[0][0]
        answer = f"{top.answer} (Source: '{top.question}')"
        cited = top.question
        category = top.category or "General Enquiries"
    return {
        "query": query,
        "answer": answer,
        "cited_question": cited,
        "category": category,
        "institution": hits[0][0].institution if hits else "General",
        "confidence": confidence,
        "lead_score": lead,
        "lead_label": label,
        "escalate": escalate,
        "priority": priority,
        "abstained": abstained,
        "candidates": [{"question": e.question, "score": s,
                        "category": e.category,
                        "institution": e.institution} for e, s in hits],
    }
