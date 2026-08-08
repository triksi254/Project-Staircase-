"""Pydantic data models and enumerations for scraped institutional data."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class QualificationLevel(str, Enum):
    """Normalized qualification levels."""

    FOUNDATION = "foundation"
    UNDERGRADUATE = "undergraduate"
    BACHELOR = "bachelor"
    MASTER = "master"
    POSTGRADUATE = "postgraduate"
    PHD = "phd"
    DIPLOMA = "diploma"
    CERTIFICATE = "certificate"
    OTHER = "other"

    @classmethod
    def normalize(cls, value: Optional[str]) -> Optional["QualificationLevel"]:
        """Map a raw string to a normalized qualification level."""
        if not value:
            return None
        v = str(value).strip().lower()
        mapping = {
            "foundation": cls.FOUNDATION,
            "foundation degree": cls.FOUNDATION,
            "undergraduate": cls.UNDERGRADUATE,
            "bachelor": cls.BACHELOR,
            "bsc": cls.BACHELOR,
            "ba": cls.BACHELOR,
            "beng": cls.BACHELOR,
            "bachelor of science": cls.BACHELOR,
            "bachelor of arts": cls.BACHELOR,
            "master": cls.MASTER,
            "msc": cls.MASTER,
            "ma": cls.MASTER,
            "meng": cls.MASTER,
            "mba": cls.MASTER,
            "master of science": cls.MASTER,
            "master of arts": cls.MASTER,
            "postgraduate": cls.POSTGRADUATE,
            "pg": cls.POSTGRADUATE,
            "postgrad": cls.POSTGRADUATE,
            "phd": cls.PHD,
            "dphil": cls.PHD,
            "doctorate": cls.PHD,
            "doctor of philosophy": cls.PHD,
            "diploma": cls.DIPLOMA,
            "certificate": cls.CERTIFICATE,
            "cert": cls.CERTIFICATE,
        }
        # Try exact match first
        if v in mapping:
            return mapping[v]
        # Try stripping "in ...", "of ...", "degree"
        cleaned = re.sub(r"\b(in|of|with honours|honours)\b.*$", "", v).strip()
        if cleaned in mapping:
            return mapping[cleaned]
        return None


class StudyMode(str, Enum):
    """Normalized study modes."""

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    DISTANCE = "distance_learning"
    ONLINE = "online"
    SANDWICH = "sandwich"
    OTHER = "other"

    @classmethod
    def normalize(cls, value: Optional[str]) -> Optional["StudyMode"]:
        """Map a raw string to a normalized study mode."""
        if not value:
            return None
        v = str(value).strip().lower()
        mapping = {
            "full time": cls.FULL_TIME,
            "full-time": cls.FULL_TIME,
            "fulltime": cls.FULL_TIME,
            "ft": cls.FULL_TIME,
            "full_time": cls.FULL_TIME,
            "part time": cls.PART_TIME,
            "part-time": cls.PART_TIME,
            "parttime": cls.PART_TIME,
            "pt": cls.PART_TIME,
            "part_time": cls.PART_TIME,
            "distance": cls.DISTANCE,
            "distance learning": cls.DISTANCE,
            "distance_learning": cls.DISTANCE,
            "online": cls.ONLINE,
            "e-learning": cls.ONLINE,
            "sandwich": cls.SANDWICH,
            "sandwich course": cls.SANDWICH,
            "sandwich_course": cls.SANDWICH,
            "other": cls.OTHER,
        }
        return mapping.get(v, None)


class FAQEntry(BaseModel):
    """A single FAQ question/answer pair used for enrichment."""

    question: str
    answer: str
    keywords: List[str] = Field(default_factory=list)
    relevance_score: Optional[float] = None


class BaseRecord(BaseModel):
    """Common fields shared by all scraped records."""

    model_config = {"extra": "allow"}

    title: str
    url: Optional[str] = None
    description: Optional[str] = None
    institution: Optional[str] = None
    data_type: Optional[str] = None
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    related_faqs: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class Course(BaseRecord):
    """A university course record."""

    level: Optional[Union[QualificationLevel, str]] = None
    duration: Optional[str] = None
    study_type: Optional[Union[StudyMode, str]] = None
    fees: Optional[str] = None
    subject_area: Optional[str] = None
    entry_requirements: Optional[str] = None

    @field_validator("level", mode="before")
    @classmethod
    def _normalize_level(cls, v: Any) -> Any:
        if isinstance(v, str):
            return QualificationLevel.normalize(v) or v
        return v

    @field_validator("study_type", mode="before")
    @classmethod
    def _normalize_study_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            return StudyMode.normalize(v) or v
        return v


class Accommodation(BaseRecord):
    """A university accommodation / housing record."""

    location: Optional[str] = None
    price: Optional[str] = None
    price_per_week: Optional[float] = None
    room_type: Optional[str] = None
    available_from: Optional[str] = None

