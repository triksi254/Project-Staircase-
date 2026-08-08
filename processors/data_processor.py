"""Data cleaning, validation and normalization of scraped records."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, ValidationError

from models.schemas import Accommodation, Course, QualificationLevel, StudyMode


@dataclass
class ProcessResult:
    """Result of processing raw scraped data."""

    processed: List[BaseModel] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    dropped: int = 0

    @property
    def count(self) -> int:
        return len(self.processed)


class DataProcessor:
    """Clean and validate raw scraped records into Pydantic models."""

    def __init__(self, validate: bool = True, institution: Optional[str] = None, data_type: Optional[str] = None) -> None:
        self.validate = validate
        self.institution = institution
        self.data_type = data_type

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(
        self, raw_items: List[Dict[str, Any]], data_type: str, institution: Optional[str] = None
    ) -> ProcessResult:
        """Process raw items for a given data type."""
        if data_type == "accommodation":
            return self.process_accommodation(raw_items, institution=institution)
        return self.process_courses(raw_items, institution=institution)

    def process_courses(
        self, raw_items: List[Dict[str, Any]], institution: Optional[str] = None
    ) -> ProcessResult:
        """Validate raw course records into Course models."""
        result = ProcessResult()
        inst = institution or self.institution
        for idx, item in enumerate(raw_items):
            try:
                cleaned = self._clean_course(item, inst)
                if not cleaned.get("title") and not cleaned.get("url"):
                    result.dropped += 1
                    result.errors.append({"index": idx, "type": "dropped", "reason": "no title or url"})
                    continue
                if self.validate:
                    model = Course(**cleaned)
                else:
                    model = Course.model_construct(**cleaned)
                result.processed.append(model)
            except ValidationError as exc:
                msg = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                logger.warning("Course validation error (item {}): {}", idx, msg)
                result.errors.append(
                    {"index": idx, "type": "validation", "message": msg, "raw": item}
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Course processing error (item {}): {}", idx, exc)
                result.errors.append({"index": idx, "type": "processing", "message": str(exc)})
        logger.info(
            "Processed {} courses ({} valid, {} errors, {} dropped)",
            len(raw_items),
            len(result.processed),
            len(result.errors),
            result.dropped,
        )
        return result

    def process_accommodation(
        self, raw_items: List[Dict[str, Any]], institution: Optional[str] = None
    ) -> ProcessResult:
        """Validate raw accommodation records into Accommodation models."""
        result = ProcessResult()
        inst = institution or self.institution
        for idx, item in enumerate(raw_items):
            try:
                cleaned = self._clean_accommodation(item, inst)
                if not cleaned.get("title") and not cleaned.get("url"):
                    result.dropped += 1
                    result.errors.append({"index": idx, "type": "dropped", "reason": "no title or url"})
                    continue
                if self.validate:
                    model = Accommodation(**cleaned)
                else:
                    model = Accommodation.model_construct(**cleaned)
                result.processed.append(model)
            except ValidationError as exc:
                msg = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                logger.warning("Accommodation validation error (item {}): {}", idx, msg)
                result.errors.append(
                    {"index": idx, "type": "validation", "message": msg, "raw": item}
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Accommodation processing error (item {}): {}", idx, exc)
                result.errors.append({"index": idx, "type": "processing", "message": str(exc)})
        logger.info(
            "Processed {} accommodation records ({} valid, {} errors, {} dropped)",
            len(raw_items),
            len(result.processed),
            len(result.errors),
            result.dropped,
        )
        return result

    # ------------------------------------------------------------------ #
    # Cleaning
    # ------------------------------------------------------------------ #
    def _clean_course(self, item: Dict[str, Any], institution: Optional[str]) -> Dict[str, Any]:
        """Clean/normalize a raw course dict."""
        item = {k: v for k, v in item.items() if v is not None and v != ""}

        title = self._clean_text(item.get("title"))
        url = self._clean_text(item.get("url"))
        description = self._clean_text(item.get("description"))

        level = item.get("level")
        level = self._normalize_level(level)
        study_type = item.get("study_type")
        study_type = self._normalize_study_type(study_type)

        fees = self._clean_fees(item.get("fees"))
        duration = self._clean_text(item.get("duration"))
        subject_area = self._clean_text(item.get("subject_area"))
        entry_requirements = self._clean_text(item.get("entry_requirements"))

        return {
            "title": title,
            "url": url,
            "description": description,
            "level": level,
            "duration": duration,
            "study_type": study_type,
            "fees": fees,
            "subject_area": subject_area,
            "entry_requirements": entry_requirements,
            "institution": institution or self.institution,
            "data_type": "courses",
        }

    def _clean_accommodation(self, item: Dict[str, Any], institution: Optional[str]) -> Dict[str, Any]:
        """Clean/normalize a raw accommodation dict."""
        item = {k: v for k, v in item.items() if v is not None and v != ""}

        title = self._clean_text(item.get("title"))
        url = self._clean_text(item.get("url"))
        description = self._clean_text(item.get("description"))
        location = self._clean_text(item.get("location"))
        price = self._clean_text(item.get("price"))
        room_type = self._clean_text(item.get("room_type"))
        available_from = self._clean_text(item.get("available_from"))

        price_per_week = self._parse_price(price)

        return {
            "title": title,
            "url": url,
            "description": description,
            "location": location,
            "price": price,
            "price_per_week": price_per_week,
            "room_type": room_type,
            "available_from": available_from,
            "institution": institution or self.institution,
            "data_type": "accommodation",
        }

    # ------------------------------------------------------------------ #
    # Normalization helpers
    # ------------------------------------------------------------------ #
    def _normalize_level(self, value: Any) -> Optional[str]:
        """Normalize a qualification level string."""
        if not value:
            return None
        normalized = QualificationLevel.normalize(str(value))
        return normalized.value if normalized else self._clean_text(str(value))

    def _normalize_study_type(self, value: Any) -> Optional[str]:
        """Normalize a study mode string."""
        if not value:
            return None
        normalized = StudyMode.normalize(str(value))
        return normalized.value if normalized else self._clean_text(str(value))

    @staticmethod
    def _clean_text(value: Any) -> Optional[str]:
        """Normalize whitespace in a text value."""
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text or None

    @staticmethod
    def _clean_fees(value: Any) -> Optional[str]:
        """Clean a fees string (remove stray whitespace/newlines)."""
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        # Collapse repeated currency separators
        text = re.sub(r"([£$€])\s*([0-9])", r"\1\2", text)
        return text or None

    @staticmethod
    def _parse_price(value: Any) -> Optional[float]:
        """Parse a currency string like '£150 / week' into a float."""
        if value is None:
            return None
        m = re.search(r"[\d,]+(?:\.\d+)?", str(value))
        if not m:
            return None
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            return None

