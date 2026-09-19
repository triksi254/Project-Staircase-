"""Responder: grounded answers + lead-aware counsellor escalation.

Policy: answer from the top-ranked FAQ (with its corpus question cited so
replies stay grounded); retrieval confidence below ``min_confidence``
yields an abstention + escalation instead of a guess. Hot leads always
escalate with priority so a counsellor follows up, even when the answer
itself is confident.

Two signals are deliberately separate and must not be conflated:

* ``confidence`` is the **retrieval** score: the top-1 similarity returned
  by ``Retriever.search`` (TF-IDF cosine in [0, 1]). It is compared against
  ``EscalationPolicy.min_confidence`` on every path, including when a top-1
  hit exists, and drives abstention vs grounded answer.
* ``lead_score`` / ``lead_label`` is the **lead** score: ``hybrid_score``
  of the rule and ML components, mapped to Cold/Warm/Hot by
  ``score_to_label``. It drives counsellor escalation, not abstention.

Consequence: a confident-looking answer may still be an abstention if the
retrieval similarity is below the gate, and a well-grounded answer may still
escalate if the lead is Hot. ``min_confidence`` therefore gates retrieval
quality only -- it is *not* a threshold on the lead score.

Hot classification is delegated to ``score_to_label`` (tertile cut, i.e.
``lead_score >= 2/3``). ``EscalationPolicy.hot_threshold`` is kept for
callers, serialisation, and display; to avoid silent divergence it must
equal that tertile cut. ``respond`` reads ``min_confidence`` and ``alpha``
from the policy but derives Hot/Warm/Cold from ``score_to_label``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from chatbot.retriever import FaqEntry, Retriever
from leads.hybrid import hybrid_score, score_to_label

ABSTAIN_MESSAGE = (
    "I don't have a verified answer for that yet. "
    "I've flagged your question for a counsellor who will follow up."
)

#: Institution names in the corpus (besides "General"). A query that names
#: one of these scopes the answer: a top-1 FAQ from a *different* named
#: institution is a scope mismatch and must abstain rather than present
#: another university's facts as the answer. "General" FAQs answer anyone.
_SCOPED_INSTITUTIONS = frozenset({
    "aston", "bcu", "herts", "rgu", "salford", "uclan", "usw",
})


def _query_institution(query: str) -> Optional[str]:
    """Lowercased corpus institution named in ``query``, else None."""
    lowered = f" {query.lower()} "
    for name in sorted(_SCOPED_INSTITUTIONS):
        if f" {name} " in lowered:
            return name
    return None


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
    label = score_to_label(lead)
    hits = retriever.search(query, top_k=top_k)
    confidence = hits[0][1] if hits else 0.0
    # Institution scope: a query naming a corpus institution (e.g. "USW")
    # must not be answered with another university's FAQ (e.g. RGU facts
    # presented as USW facts). "General" FAQs answer anyone; only a named
    # institution mismatched against a *different* named institution
    # abstains. Checked before the confidence gate so a high-confidence
    # wrong-university hit still abstains.
    scope = _query_institution(query)
    scope_mismatch = bool(
        hits and scope is not None
        and hits[0][0].institution.lower() not in ("general", scope))
    # Escalation aligns exactly with score_to_label: Hot <=> lead >= 2/3.
    escalate = ((not hits) or scope_mismatch
                or (confidence < pol.min_confidence) or (label == "Hot"))
    priority = "high" if label == "Hot" else ("normal" if escalate else "none")
    if not hits or scope_mismatch or confidence < pol.min_confidence:
        answer = ABSTAIN_MESSAGE
        cited = None
        if scope_mismatch:
            category = hits[0][0].category or "General Enquiries"
        else:
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
        "candidates": [{"question": e.question, "score": s,
                        "category": e.category,
                        "institution": e.institution} for e, s in hits],
    }
