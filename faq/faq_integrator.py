"""FAQ integration: enrich scraped records with relevant FAQ answers.

Uses a hybrid matching approach:
  1. Keyword overlap scoring across question, answer and configured keywords.
  2. Basic text-similarity (token overlap / character n-gram cosine-like scoring)
     when keyword matches are insufficient.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from config.settings import FAQ_CORPUS_PATH, Settings, get_settings
from models.schemas import FAQEntry


class FAQIntegrator:
    """Cross-references scraped records with the FAQ corpus."""

    def __init__(self, settings: Optional[Settings] = None, corpus_path: Optional[Path] = None) -> None:
        self.settings = settings or get_settings()
        self.corpus_path = corpus_path or FAQ_CORPUS_PATH
        self.corpus: List[FAQEntry] = self._load_corpus()
        logger.info("Loaded {} FAQ entries", len(self.corpus))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def enrich_records(self, records: List[BaseModel], data_type: str) -> List[BaseModel]:
        """Enrich a list of Pydantic records with related FAQs."""
        if not self.settings.INTEGRATE_FAQ or not self.corpus:
            return records

        for record in records:
            related = self.find_related_faqs(record, data_type)
            if related:
                record.related_faqs = related
        return records

    def find_related_faqs(
        self, record: BaseModel, data_type: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Return the top-k related FAQs for a single record."""
        if not self.corpus:
            return []

        text = self._record_text(record, data_type)
        if not text:
            return []

        scored: List[tuple[float, FAQEntry]] = []
        for faq in self.corpus:
            relevance = self._similarity(text, faq)
            if relevance >= self.settings.FAQ_RELEVANCE_THRESHOLD:
                scored.append((relevance, faq))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [
            {
                "question": faq.question,
                "answer": faq.answer,
                "relevance_score": round(score, 4),
            }
            for score, faq in scored[:top_k]
        ]
        return results

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _load_corpus(self) -> List[FAQEntry]:
        """Load FAQ entries from the JSON corpus file."""
        if not self.corpus_path.exists():
            logger.warning("FAQ corpus not found at {}", self.corpus_path)
            return []
        try:
            raw = json.loads(self.corpus_path.read_text(encoding="utf-8"))
            entries = raw if isinstance(raw, list) else raw.get("faqs", [])
            return [FAQEntry(**e) for e in entries]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load FAQ corpus: {}", exc)
            return []

    def _record_text(self, record: BaseModel, data_type: str) -> str:
        """Build a searchable text blob from a record."""
        if data_type == "accommodation":
            parts = [
                record.title,
                record.description,
                getattr(record, "location", None),
                getattr(record, "room_type", None),
            ]
        else:
            parts = [
                record.title,
                record.description,
                getattr(record, "subject_area", None),
                getattr(record, "level", None),
                getattr(record, "study_type", None),
                getattr(record, "duration", None),
            ]
        return " ".join(str(p) for p in parts if p)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Split text into lowercase alphanumeric tokens."""
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _similarity(self, text: str, faq: FAQEntry) -> float:
        """Compute a hybrid relevance score between text and a FAQ entry.

        Scores are normalized to 0..1. Keyword coverage is the strongest
        signal; question/answer token overlap and character-level fuzzy
        similarity add secondary evidence.
        """
        text_tokens = self._tokenize(text)
        if not text_tokens:
            return 0.0
        faq_q_tokens = self._tokenize(faq.question)
        faq_a_tokens = self._tokenize(faq.answer)
        kw_tokens = self._tokenize(" ".join(faq.keywords))

        # 1) Keyword coverage: fraction of the FAQ's keywords found in text
        matched_kws = text_tokens & kw_tokens
        kw_score = len(matched_kws) / max(1, len(kw_tokens))

        # 2) Question token overlap: fraction of question tokens present in text
        q_overlap = len(text_tokens & faq_q_tokens) / max(1, len(faq_q_tokens))

        # 3) Answer token overlap
        a_overlap = len(text_tokens & faq_a_tokens) / max(1, len(faq_a_tokens))

        # 4) Character-level fuzzy similarity to the question (handles typos/word forms)
        char_sim_q = SequenceMatcher(None, text.lower()[:2000], faq.question.lower()).ratio()

        # Weighted combination (keyword match is the strongest indicator)
        score = (
            0.45 * kw_score
            + 0.25 * q_overlap
            + 0.15 * a_overlap
            + 0.15 * char_sim_q
        )
        return score

