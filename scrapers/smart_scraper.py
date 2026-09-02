"""Smart scraper that routes to the appropriate scraper based on JS detection."""
from __future__ import annotations

import re
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger

from config.settings import Settings, get_settings
from scrapers.base_scraper import ScrapeMetadata
from scrapers.html_scraper import HTMLScraper
from scrapers.js_scraper import JavaScriptScraper

# Indicators that a page requires JavaScript rendering
JS_INDICATORS = {
    "frameworks": [
        r"<div[^>]*id=[\"']root[\"']",
        r"<div[^>]*id=[\"']app[\"']",
        r"<div[^>]*id=[\"']__next[\"']",
        r"<div[^>]*id=[\"']__nuxt[\"']",
        r"id=[\"']gatsby-[^\"']*[\"']",
        r"ng-version=",
        r"data-reactroot",
        r"__NUXT__",
        r"__NEXT_DATA__",
        r"window\.__INITIAL_STATE__",
        r"webpackRuntime",
        r"wpReact",
    ],
    "script_src": [
        r"\.jsx\?",
        r"\.tsx\?",
        r"react\.js",
        r"vue\.js",
        r"angular\.js",
        r"main\.[a-f0-9]+\.js",
        r"chunk-vendors",
    ],
    "sparse_content": [
        # NOTE: HTML comments (`<!-- ... -->`) are intentionally NOT treated as
        # a JS indicator - they are common in server-rendered templates too, so
        # they are a weak and misleading signal.
        r"<noscript>",
    ],
}

# Indicators that a page is simple static HTML
HTML_INDICATORS = [
    r"<h1[^>]*>",
    r"<h2[^>]*>",
    r"<article",
    r"<li[^>]*class=[\"']course",
    r"<div[^>]*class=[\"'][^\"']*course",
    r"<table",
]


class SmartScraper:
    """Routes to HTML or JavaScript scraper based on content analysis.

    Strategy:
      1. If `force_html` or `force_js` is set, use it directly.
      2. Otherwise check YAML config `type` hint.
      3. Probe the page: fetch initial HTML and score JS vs HTML indicators.
         If JS score > HTML score, use the JS scraper; otherwise the HTML scraper.
    """

    def __init__(
        self,
        institution: str,
        data_type: str,
        force_html: bool = False,
        force_js: bool = False,
        delay: Optional[float] = None,
        max_pages: Optional[int] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.institution = institution
        self.data_type = data_type
        self.force_html = force_html
        self.force_js = force_js
        self.delay = delay
        self.max_pages = max_pages
        self.settings = settings or get_settings()
        self.scraper = None
        self.metadata = ScrapeMetadata(institution=institution, data_type=data_type)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def scrape(self, url: Optional[str] = None) -> List[dict]:
        """Scrape the institution/data-type, choosing the best scraper."""
        if self.scraper is not None:
            # Close any scraper created by an earlier call (avoids leaking a
            # Playwright browser when scrape() is invoked more than once).
            self.scraper.close()
            self.scraper = None
        resolved = self.resolve_scraper(url)
        logger.info(
            "Using {} scraper for {}/{} (force_html={}, force_js={})",
            resolved.SCRAPER_NAME,
            self.institution,
            self.data_type,
            self.force_html,
            self.force_js,
        )
        items = resolved.scrape(url=url)
        self.scraper = resolved
        self.metadata = resolved.metadata
        return items

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def resolve_scraper(self, url: Optional[str] = None) -> object:
        """Determine and return the appropriate scraper instance."""
        if self.force_html:
            return self._make_html_scraper()
        if self.force_js:
            return self._make_js_scraper()

        config_type = self._config_type()
        if config_type == "javascript":
            logger.info("Config declares {} as javascript", self.data_type)
            return self._make_js_scraper()
        if config_type == "html":
            return self._make_html_scraper()

        if not self.settings.USE_PLAYWRIGHT:
            logger.warning("Playwright disabled by settings; using HTML scraper")
            return self._make_html_scraper()

        # Probe the page
        probe_url = url or self._probe_url()
        if probe_url:
            html, status = self._probe(probe_url)
            if html:
                js_score = self._score_js(html)
                html_score = self._score_html(html)
                logger.info(
                    "Probe {}/{}: js_score={}, html_score={}",
                    self.institution,
                    self.data_type,
                    js_score,
                    html_score,
                )
                if js_score > html_score:
                    logger.info("Page appears to require JavaScript")
                    return self._make_js_scraper()
        logger.info("Page appears to be static HTML")
        return self._make_html_scraper()

    def _make_html_scraper(self) -> HTMLScraper:
        return HTMLScraper(
            self.institution,
            self.data_type,
            delay=self.delay,
            max_pages=self.max_pages,
            settings=self.settings,
        )

    def _make_js_scraper(self) -> JavaScriptScraper:
        return JavaScriptScraper(
            self.institution,
            self.data_type,
            delay=self.delay,
            max_pages=self.max_pages,
            settings=self.settings,
        )

    def _config_type(self) -> Optional[str]:
        """Return the configured 'type' for the data source, if any."""
        from config.settings import get_institutions

        inst = get_institutions().get(self.institution, {})
        source = inst.get("data_sources", {}).get(self.data_type, {})
        return source.get("type")

    def _probe_url(self) -> Optional[str]:
        """Determine the URL to probe (first page of the data source)."""
        from config.settings import get_institutions

        inst = get_institutions().get(self.institution, {})
        base_url = inst.get("base_url", "")
        source = inst.get("data_sources", {}).get(self.data_type, {})
        return source.get("url") or base_url or None

    def _probe(self, url: str) -> tuple[Optional[str], Optional[int]]:
        """Fetch the initial HTML of a page for analysis."""
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": self.settings.USER_AGENT},
                timeout=self.settings.DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.text, resp.status_code
        except Exception as exc:  # noqa: BLE001
            logger.warning("Probe failed for {}: {}", url, exc)
            return None, None

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _score_js(self, html: str) -> int:
        """Score how likely the page requires JavaScript."""
        score = 0
        low = html[:50000].lower()

        for pattern in JS_INDICATORS["frameworks"]:
            if re.search(pattern, html, re.IGNORECASE):
                score += 3
        for pattern in JS_INDICATORS["script_src"]:
            if re.search(pattern, low, re.IGNORECASE):
                score += 1

        # Sparse content check: tiny visible text relative to markup
        text_len = len(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
        if len(html) > 2000 and text_len < max(200, int(len(html) * 0.03)):
            score += 3

        # Framework indicators from scripts
        if re.search(r"__NEXT_DATA__|__NUXT__|ng-version|data-reactroot|window\.__INITIAL_STATE__", html):
            score += 4

        return score

    def _score_html(self, html: str) -> int:
        """Score how likely the page is simple static HTML."""
        score = 0
        for pattern in HTML_INDICATORS:
            if re.search(pattern, html, re.IGNORECASE):
                score += 1

        text_len = len(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
        if len(html) > 500 and text_len > len(html) * 0.10:
            score += 2
        return score

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Close the underlying scraper if created."""
        if self.scraper is not None:
            self.scraper.close()

    def __enter__(self) -> "SmartScraper":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

