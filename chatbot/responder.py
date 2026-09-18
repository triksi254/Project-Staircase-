"""Responder: grounded answers + lead-aware counsellor escalation.

Policy: answer from the top-ranked FAQ (with its corpus question cited so
replies stay grounded); retrieval confidence below ``min_confidence``
yields an abstention + escalation instead of a guess. Hot leads
(hybrid_score >= ``hot_threshold``) always escalate with priority so a
counsellor follows up, even when the answer itself is confident.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from chatbot.retriever import FaqEntry, Retriever
from leads.hybrid import hybrid_score

ABSTAIN_MESSAGE = (
    "I don't have a verified answer for that yet. "
    "I've flagged your question for a counsellor who will follow up."
)


@dataclass
class EscalationPolicy:
    """Thresholds for abstention and counsellor handoff."""
    min_confidence: float = 0.15
    hot_threshold: float = 2.0 / 3.0
    alpha: float = 0.5


def respond(
    query: str,
    retriever: Retriever,
    rule_score: float = 0.5,
    ml_score: Optional[float] = None,
    policy: Optional[EscalationPolicy] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Answer ``query``; return dict with answer + routing signals."""
    pol = policy or EscalationPolicy()
    lead = hybrid_score(rule_score, ml_score, alpha=pol.alpha)
    label = "Cold" if lead < 1.0 / 3.0 else ("Warm" if lead < 2.0 / 3.0 else "Hot")
    hits = retriever.search(query, top_k=top_k)
    confidence = hits[0][1] if hits else 0.0
    escalate = (not hits) or (confidence < pol.min_confidence) or (lead >= pol.hot_threshold)
    priority = "high" if lead >= pol.hot_threshold else ("normal" if escalate else "none")
    if not hits or confidence < pol.min_confidence:
        answer = ABSTAIN_MESSAGE
        cited: Optional[str] = None
    else:
        top = hits[0][0]
        answer = f"{top.answer} (Source: '{top.question}')"
        cited = top.question
    return {
        "query": query,
        "answer": answer,
        "cited_question": cited,
        "confidence": confidence,
        "lead_score": lead,
        "lead_label": label,
        "escalate": escalate,
        "priority": priority,
        "candidates": [{"question": e.question, "score": s} for e, s in hits],
    }
