"""Unit tests for data processing and FAQ integration."""
from __future__ import annotations

import pytest

from faq.faq_integrator import FAQIntegrator
from models.schemas import Accommodation, Course, QualificationLevel, StudyMode
from processors.data_processor import DataProcessor

SAMPLE_COURSES = [
    {
        "title": "BSc Computer Science",
        "url": "https://www.aston.ac.uk/study/courses/bsc-computer-science",
        "description": "A comprehensive degree in computer science.",
        "level": "BSc",
        "duration": "3 years",
        "study_type": "Full-time",
        "fees": "£9,250",
    },
    {
        "title": "MSc Data Science",
        "url": "https://www.aston.ac.uk/study/courses/msc-data-science",
        "description": "Advanced data science programme.",
        "level": "MSc",
        "duration": "1 year",
        "study_type": "Part-time",
        "fees": "£12,000",
    },
]

SAMPLE_ACCOMMODATION = [
    {
        "title": "Lakeside Halls",
        "url": "https://www.aston.ac.uk/accommodation/lakeside",
        "location": "City Centre",
        "price": "£150 per week",
        "room_type": "En-suite",
    }
]


@pytest.fixture
def processor():
    return DataProcessor(validate=True)


@pytest.fixture
def faq_integrator():
    # Point at the real corpus file in the repo
    from config.settings import FAQ_CORPUS_PATH

    return FAQIntegrator(corpus_path=FAQ_CORPUS_PATH)


def test_process_courses_valid(processor):
    result = processor.process_courses(SAMPLE_COURSES, institution="aston_university")
    assert result.count == 2
    assert result.dropped == 0
    assert all(isinstance(m, Course) for m in result.processed)


def test_process_courses_level_normalization(processor):
    result = processor.process_courses(SAMPLE_COURSES, institution="aston_university")
    first = result.processed[0]
    assert first.level == QualificationLevel.BACHELOR
    second = result.processed[1]
    assert second.level == QualificationLevel.MASTER


def test_process_courses_study_type_normalization(processor):
    result = processor.process_courses(SAMPLE_COURSES, institution="aston_university")
    first = result.processed[0]
    assert first.study_type == StudyMode.FULL_TIME


def test_process_courses_fees_preserved(processor):
    result = processor.process_courses(SAMPLE_COURSES, institution="aston_university")
    assert result.processed[0].fees == "£9,250"


def test_process_courses_drop_blank(processor):
    result = processor.process_courses(
        [{"description": "no title or url here"}], institution="aston_university"
    )
    assert result.count == 0
    assert result.dropped == 1


def test_process_accommodation(processor):
    result = processor.process_accommodation(SAMPLE_ACCOMMODATION, institution="aston_university")
    assert result.count == 1
    item = result.processed[0]
    assert isinstance(item, Accommodation)
    assert item.price_per_week == 150.0


def test_parse_price():
    processor = DataProcessor()
    assert processor._parse_price("£150 per week") == 150.0
    assert processor._parse_price("$1,200 /month") == 1200.0
    assert processor._parse_price("Price on request") is None


def test_faq_integration_returns_related(faq_integrator):
    course = Course(
        title="BSc Computer Science",
        description="This course covers computing, software and fees information.",
        level="bachelor",
        duration="3 years",
    )
    related = faq_integrator.find_related_faqs(course, "courses", top_k=3)
    assert len(related) > 0
    assert all("relevance_score" in r for r in related)


def test_faq_integrator_enrich_records(faq_integrator):
    records = [
        Course(
            title="MSc Data Science",
            description="Postgraduate course on data science and machine learning.",
            level="master",
            duration="1 year",
        )
    ]
    enriched = faq_integrator.enrich_records(records, "courses")
    assert len(enriched[0].related_faqs) > 0


def test_faq_no_match_returns_empty(faq_integrator):
    course = Course(title="Quantum Xylophone Studies", description="zzz zzz zzz zzz zzz")
    related = faq_integrator.find_related_faqs(course, "courses", top_k=3)
    assert related == []

