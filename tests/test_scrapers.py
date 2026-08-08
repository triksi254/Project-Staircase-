"""Unit tests for scrapers."""
from __future__ import annotations

import pytest

from scrapers.base_scraper import BaseScraper
from scrapers.html_scraper import HTMLScraper
from scrapers.smart_scraper import SmartScraper

SAMPLE_HTML = """
<html>
<body>
  <div class="course-item">
    <h2 class="course-title">BSc Computer Science</h2>
    <a class="course-link" href="/study/courses/bsc-computer-science">View</a>
    <div class="course-description">A comprehensive degree in computer science.</div>
    <div class="qualification-level">BSc</div>
    <div class="course-duration">3 years</div>
    <div class="study-mode">Full-time</div>
    <div class="fees-amount">£9,250</div>
  </div>
  <div class="course-item">
    <h2 class="course-title">MSc Data Science</h2>
    <a class="course-link" href="/study/courses/msc-data-science">View</a>
    <div class="course-description">Advanced data science.</div>
    <div class="qualification-level">MSc</div>
    <div class="course-duration">1 year</div>
    <div class="study-mode">Full-time</div>
    <div class="fees-amount">£12,000</div>
  </div>
</body>
</html>
"""


class _TestScraper(BaseScraper):
    """Minimal BaseScraper subclass for testing extract_items."""

    SCRAPER_NAME = "test"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Provide selectors directly (bypass YAML)
        self.selectors_cfg = {
            "container": ".course-item",
            "fields": {
                "title": "h2.course-title::text",
                "url": "a.course-link::attr(href)",
                "description": ".course-description::text",
                "level": ".qualification-level::text",
                "duration": ".course-duration::text",
                "study_type": ".study-mode::text",
                "fees": ".fees-amount::text",
            },
        }

    def fetch_page(self, url):
        return SAMPLE_HTML

    def close(self):
        pass


@pytest.fixture
def sample_scraper():
    return _TestScraper("aston_university", "courses", settings=None)


def test_extract_items_returns_expected_count(sample_scraper):
    items = sample_scraper.extract_items(SAMPLE_HTML, "https://www.aston.ac.uk/courses")
    assert len(items) == 2


def test_extract_items_field_values(sample_scraper):
    items = sample_scraper.extract_items(SAMPLE_HTML, "https://www.aston.ac.uk/courses")
    first = items[0]
    assert first["title"] == "BSc Computer Science"
    assert first["level"] == "BSc"
    assert first["fees"] == "£9,250"


def test_extract_items_resolves_relative_urls(sample_scraper):
    items = sample_scraper.extract_items(SAMPLE_HTML, "https://www.aston.ac.uk/study/courses")
    first = items[0]
    assert first["url"] == "https://www.aston.ac.uk/study/courses/bsc-computer-science"


def test_extract_items_blank_container_skipped():
    html = """
    <html><body>
      <div class="course-item">
        <h2 class="course-title"></h2>
        <a class="course-link" href=""></a>
      </div>
    </body></html>
    """
    scraper = _TestScraper("aston_university", "courses")
    items = scraper.extract_items(html, "https://www.aston.ac.uk")
    assert items == []


def test_build_page_url():
    scraper = _TestScraper("aston_university", "courses")
    scraper.pagination_param = "page"
    url = scraper._build_page_url("https://www.aston.ac.uk/courses", 2)
    assert url == "https://www.aston.ac.uk/courses?page=2"


def test_build_page_url_with_existing_query():
    scraper = _TestScraper("aston_university", "courses")
    scraper.pagination_param = "p"
    url = scraper._build_page_url("https://www.aston.ac.uk/courses?cat=cs", 3)
    assert url == "https://www.aston.ac.uk/courses?cat=cs&p=3"


def test_has_next_page_detects_rel_next():
    scraper = _TestScraper("aston_university", "courses")
    scraper.paginated = True
    html = '<a rel="next" href="/courses?page=2">Next</a>'
    assert scraper._has_next_page(html, 1) is True


def test_has_next_page_no_next():
    scraper = _TestScraper("aston_university", "courses")
    scraper.paginated = True
    assert scraper._has_next_page("<html></html>", 1) is False


def test_html_scraper_name():
    scraper = HTMLScraper("aston_university", "courses")
    assert scraper.SCRAPER_NAME == "html"
    scraper.close()


def test_smart_scraper_force_html_uses_html_scraper():
    smart = SmartScraper("aston_university", "courses", force_html=True)
    resolved = smart.resolve_scraper()
    assert resolved.SCRAPER_NAME == "html"
    resolved.close()


def test_smart_scraper_force_js_uses_js_scraper():
    smart = SmartScraper("aston_university", "courses", force_js=True)
    resolved = smart.resolve_scraper()
    assert resolved.SCRAPER_NAME == "javascript"
    resolved.close()


def test_smart_scraper_scores_js_high_for_spa():
    smart = SmartScraper("aston_university", "courses")
    spa_html = """
    <html><head>
      <script src="/static/js/main.abc123.js"></script>
    </head>
    <body>
      <div id="root"><!-- dynamic content --></div>
    </body></html>
    """
    js_score = smart._score_js(spa_html)
    html_score = smart._score_html(spa_html)
    assert js_score > html_score


def test_smart_scraper_scores_html_high_for_static():
    smart = SmartScraper("aston_university", "courses")
    static_html = """
    <html><body>
      <h1>Courses</h1>
      <div class="course-item"><h2>BSc Computer Science</h2>
      <p>This is a comprehensive description of the computer science course.</p></div>
    </body></html>
    """
    js_score = smart._score_js(static_html)
    html_score = smart._score_html(static_html)
    assert html_score > js_score

