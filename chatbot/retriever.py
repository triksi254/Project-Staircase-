"""TF-IDF + keyword retrieval over the grounded FAQ corpus.

Corpus entries carry only question / answer / keywords (no category or
institution fields), so topic signals come from the text itself. Uses
sklearn TfidfVectorizer when available, else a pure-stdlib token-overlap
fallback. Never raises on missing corpus: returns [] hits instead.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "faq_corpus.json"

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset({"the", "a", "an", "and", "or", "of", "to", "in", "for",
                   "is", "are", "do", "does", "what", "how", "can", "i", "my"})


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens minus a small stopword set."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class FaqEntry:
    """One grounded FAQ pair plus its retrieval document text."""
    question: str
    answer: str
    keywords: List[str] = field(default_factory=list)
    index: int = 0

    @property
    def document(self) -> str:
        return f"{self.question} {self.answer} {' '.join(self.keywords)}"


def load_corpus(path=None) -> List[FaqEntry]:
    """Load data/faq_corpus.json; warn + return [] when missing/broken."""
    fp = Path(path) if path is not None else DEFAULT_CORPUS
    if not fp.is_file():
        logger.warning("retriever: corpus missing at %s — no answers available", fp)
        return []
    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("retriever: corpus unreadable at %s (%s)", fp, exc)
        return []
    items = raw.get("faqs", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        logger.warning("retriever: corpus at %s has no list — no answers", fp)
        return []
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        q, a = str(item.get("question", "")), str(item.get("answer", ""))
        if not q or not a:
            continue
        kw = item.get("keywords", [])
        out.append(FaqEntry(question=q, answer=a,
                            keywords=[str(k) for k in kw if k], index=i))
    if not out:
        logger.warning("retriever: corpus at %s yielded 0 usable FAQs", fp)
    return out

class Retriever:
    """Rank corpus entries for a query. ``search`` returns (entry, score)."""

    def __init__(self, entries: List[FaqEntry]) -> None:
        self.entries = list(entries)
        self._vectorizer: Any = None
        self._matrix: Any = None
        if self.entries:
            self._build_index()

    def _build_index(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            logger.warning("retriever: sklearn missing — keyword fallback active")
            return
        try:
            docs = [e.document for e in self.entries]
            self._vectorizer = TfidfVectorizer(tokenizer=tokenize,
                                               lowercase=False, ngram_range=(1, 2),
                                               token_pattern=None)
            self._matrix = self._vectorizer.fit_transform(docs)
        except ValueError as exc:
            logger.warning("retriever: TF-IDF build failed (%s) — fallback", exc)
            self._vectorizer, self._matrix = None, None

    def _fallback_scores(self, query: str) -> List[Tuple[FaqEntry, float]]:
        qtok = set(tokenize(query))
        scored = []
        for e in self.entries:
            dtok = set(tokenize(e.document))
            overlap = len(qtok & dtok)
            kw_hit = sum(1 for k in e.keywords if k.lower() in query.lower())
            score = overlap / max(1, len(qtok)) + 0.5 * kw_hit
            scored.append((e, round(float(score), 4)))
        scored.sort(key=lambda p: -p[1])
        return scored

    def search(self, query: str, top_k: int = 3) -> List[Tuple[FaqEntry, float]]:
        """Top-k (entry, score in [0, 1]); [] for empty query/corpus."""
        if not query or not query.strip() or not self.entries:
            return []
        if self._vectorizer is None or self._matrix is None:
            return self._fallback_scores(query)[:max(1, top_k)]
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            qv = self._vectorizer.transform([query])
            sims = cosine_similarity(qv, self._matrix)[0]
        except ValueError as exc:
            logger.warning("retriever: query transform failed (%s)", exc)
            return self._fallback_scores(query)[:max(1, top_k)]
        order = sorted(range(len(sims)), key=lambda i: -sims[i])[:max(1, top_k)]
        out = []
        for i in order:
            s = round(float(max(0.0, min(1.0, sims[i]))), 4)
            bonus = sum(0.05 for k in self.entries[i].keywords
                        if k.lower() in query.lower())
            out.append((self.entries[i], round(min(1.0, s + bonus), 4)))
        return out

    def confidence(self, query: str) -> float:
        """Top-1 score or 0.0 when nothing matches."""
        hits = self.search(query, top_k=1)
        return hits[0][1] if hits else 0.0