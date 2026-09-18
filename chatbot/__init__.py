"""Chatbot retrieval layer for Project-Staircase.

Keyword + TF-IDF retrieval over the grounded FAQ corpus
(``data/faq_corpus.json``, 98 pairs: question / answer / keywords),
plus a responder that couples retrieval confidence with the hybrid
lead score (``leads.hybrid``) to decide counsellor escalation.
"""
from chatbot.responder import (
    EscalationPolicy,
    respond,
)
from chatbot.retriever import (
    FaqEntry,
    Retriever,
    load_corpus,
)

__all__ = [
    "EscalationPolicy",
    "FaqEntry",
    "Retriever",
    "load_corpus",
    "respond",
]
