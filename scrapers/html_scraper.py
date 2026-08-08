"""HTML scraper using requests + BeautifulSoup for static sites."""
from __future__ import annotations

from typing import Optional

import requests
from loguru import logger

from config.settings import Settings
from scrapers.base_scraper import BaseScraper


class HTMLScraper(BaseScraper):
    """Scraper for static HTML pages."""

    SCRAPER_NAME = "html"

    def fetch_page(self, url: str) -> str:
        """Fetch a page over HTTP using requests."""
        self._rate_limit()
        logger.debug("HTTP GET {}", url)
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text

    def close(self) -> None:
        """Close the underlying session."""
        try:
            self.session.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing session: {}", exc)

