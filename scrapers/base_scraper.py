"""Abstract base class for scrapers.

Handles common concerns:
  - Pagination (configurable params and page limits)
  - Retry with exponential backoff
  - Rate limiting (delay between requests)
  - CSS selector extraction with pseudo-selectors (::text, ::attr, ::html, ::all)
  - Metadata collection
"""
from __future__ import annotations

import abc
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from loguru import logger

from config.settings import Settings, get_settings, get_institutions, get_selectors

# Pseudo-selector patterns
PSEUDO_TEXT = re.compile(r"^(.*)::text$")
PSEUDO_ATTR = re.compile(r"^(.*)::attr\(([^)]+)\)$")
PSEUDO_HTML = re.compile(r"^(.*)::html$")
PSEUDO_ALL = re.compile(r"^(.*)::all$")


@dataclass
class ScrapeMetadata:
    """Metadata about a scrape run."""

    institution: str
    data_type: str
    scraper_used: str = ""
    started_at: str = ""
    completed_at: str = ""
    total_items: int = 0
    total_errors: int = 0
    pages_scraped: int = 0
    final_url: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "institution": self.institution,
            "data_type": self.data_type,
            "scraper_used": self.scraper_used,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_items": self.total_items,
            "total_errors": self.total_errors,
            "pages_scraped": self.pages_scraped,
            "final_url": self.final_url,
            **self.extra,
        }


class BaseScraper(abc.ABC):
    """Base scraper implementing shared logic."""

    SCRAPER_NAME = "base"

    def __init__(
        self,
        institution: str,
        data_type: str,
        delay: Optional[float] = None,
        max_pages: Optional[int] = None,
        timeout: Optional[int] = None,
        settings: Optional[Settings] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.institution = institution
        self.data_type = data_type
        self.delay = delay or self.settings.DEFAULT_DELAY
        self.timeout = timeout or self.settings.DEFAULT_TIMEOUT

        self.institution_cfg = get_institutions().get(institution, {})
        self.source_cfg = self.institution_cfg.get("data_sources", {}).get(data_type, {})
        self.selectors_cfg = get_selectors().get(institution, {}).get(data_type, {})

        self.base_url = self.institution_cfg.get("base_url", "")
        self.source_url = self.source_cfg.get("url", "")
        self.paginated = self.source_cfg.get("paginated", False)
        self.pagination_param = self.source_cfg.get("pagination_param", "page")
        self.max_pages = max_pages or self.source_cfg.get("max_pages") or self.settings.DEFAULT_MAX_PAGES

        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.settings.USER_AGENT,
                "Accept-Language": "en-GB,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

        self.metadata = ScrapeMetadata(
            institution=institution,
            data_type=data_type,
            scraper_used=self.SCRAPER_NAME,
        )
        self._items: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []
        self._visited_urls: Set[str] = set()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def scrape(self, url: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run the full scraping pipeline and return raw items."""
        from datetime import datetime, timezone

        self.metadata.started_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Starting {} scrape for {} / {}",
            self.SCRAPER_NAME,
            self.institution,
            self.data_type,
        )
        try:
            start_url = url or self.source_url
            if not start_url:
                raise ValueError(
                    f"No source URL configured for {self.institution}/{self.data_type}"
                )
            self._scrape_pages(start_url)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scrape failed: {}", exc)
            self._errors.append({"type": "fatal", "message": str(exc)})
        finally:
            self.metadata.completed_at = datetime.now(timezone.utc).isoformat()
            self.metadata.total_items = len(self._items)
            self.metadata.total_errors = len(self._errors)
            self.metadata.pages_scraped = len(self._visited_urls)

        logger.info(
            "Finished: {} items, {} errors across {} pages",
            self.metadata.total_items,
            self.metadata.total_errors,
            self.metadata.pages_scraped,
        )
        return self._items

    # ------------------------------------------------------------------ #
    # Abstract hooks
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def fetch_page(self, url: str) -> str:
        """Fetch and return the HTML content of a page."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources."""

    # ------------------------------------------------------------------ #
    # Page fetching with pagination & retry
    # ------------------------------------------------------------------ #
    def _scrape_pages(self, start_url: str) -> None:
        """Fetch pages (respecting pagination config) and extract items."""
        page_number = 1
        while True:
            if page_number > self.max_pages:
                logger.warning("Reached max pages ({}) for {}", self.max_pages, self.institution)
                break

            url = self._build_page_url(start_url, page_number) if page_number > 1 else start_url
            if url in self._visited_urls:
                break
            self._visited_urls.add(url)

            html = self._fetch_with_retry(url)
            if html is None:
                break

            logger.debug("Scraping page {}: {}", page_number, url)
            items = self.extract_items(html, url)
            self._items.extend(items)
            self.metadata.final_url = url

            if not self.paginated:
                break

            has_next = self._has_next_page(html, page_number)
            if not has_next:
                logger.info("No more pages after page {}", page_number)
                break

            page_number += 1
            self._rate_limit()

    def _fetch_with_retry(self, url: str) -> Optional[str]:
        """Fetch a page with retries and exponential backoff."""
        attempt = 0
        while attempt < self.settings.MAX_RETRIES:
            try:
                return self.fetch_page(url)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 429:
                    # Rate limited: back off longer
                    backoff = self.settings.RETRY_BACKOFF * (2 ** attempt) * 2
                    logger.warning("Rate limited (429). Backing off for {:.1f}s", backoff)
                elif status and 500 <= status < 600:
                    backoff = self.settings.RETRY_BACKOFF * (2 ** attempt)
                    logger.warning("Server error {} on {}. Retrying in {:.1f}s", status, url, backoff)
                else:
                    raise
                attempt += 1
                time.sleep(backoff)
            except requests.RequestException as exc:
                backoff = self.settings.RETRY_BACKOFF * (2 ** attempt)
                logger.warning("Request error on {}: {}. Retry {} in {:.1f}s", url, exc, attempt, backoff)
                attempt += 1
                if attempt >= self.settings.MAX_RETRIES:
                    self._errors.append({"type": "request", "url": url, "message": str(exc)})
                    return None
                time.sleep(backoff)
        return None

    def _rate_limit(self) -> None:
        """Sleep between requests to be respectful of the server."""
        if self.delay and self.delay > 0:
            time.sleep(self.delay)

    def _build_page_url(self, base: str, page_number: int) -> str:
        """Append the pagination parameter to a URL."""
        if page_number <= 1:
            return base
        parsed = urlparse(base)
        query = parse_qs(parsed.query, keep_blank_values=True)
        query[self.pagination_param] = [str(page_number)]
        new_query = urlencode(query, doseq=True)
        return urljoin(base, f"{parsed.path}?{new_query}")

    # ------------------------------------------------------------------ #
    # Content extraction
    # ------------------------------------------------------------------ #
    def extract_items(self, html: str, url: str) -> List[Dict[str, Any]]:
        """Extract items from HTML using configured CSS selectors."""
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        container_sel = self.selectors_cfg.get("container")
        if not container_sel:
            logger.warning("No container selector configured for {}/{}", self.institution, self.data_type)
            return []

        blocks = self._select_all(soup, container_sel)
        fields_cfg = self.selectors_cfg.get("fields", {})
        items: List[Dict[str, Any]] = []
        errors = 0

        for block in blocks:
            item: Dict[str, Any] = {}
            for field_name, selector in fields_cfg.items():
                try:
                    item[field_name] = self._extract_field(block, selector)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Field '{}' extraction failed: {}", field_name, exc)
                    item[field_name] = None
            # Resolve relative URLs against the page URL
            if item.get("url") and isinstance(item["url"], str) and not item["url"].startswith(("http://", "https://")):
                item["url"] = urljoin(url, item["url"])
            items.append(item)

        # Count records that are effectively empty (nothing useful extracted)
        for item in items:
            if not item.get("title") and not item.get("url"):
                errors += 1
        if errors:
            logger.warning("{} empty/blank items skipped on {}", errors, url)
            items = [it for it in items if it.get("title") or it.get("url")]

        self._errors.extend(
            {"type": "extraction", "url": url, "message": f"{errors} blank items"} for _ in range(errors)
        )
        return items

    def _extract_field(self, block: Tag, selector: str) -> Any:
        """Extract a single field from a container using a pseudo-selector.

        Supports comma-separated CSS alternatives; the first that yields a
        non-empty result wins.
        """
        selector = selector.strip()
        # ::all - return list of all matches
        m = PSEUDO_ALL.match(selector)
        if m:
            base_sel = m.group(1).strip()
            return [self._extract_node(node) for node in self._select_all(block, base_sel)]

        # Split on top-level commas into candidate selectors
        candidates = [c.strip() for c in selector.split(",") if c.strip()]
        for candidate in candidates:
            result = self._extract_field_single(block, candidate)
            if result not in (None, "", [], {}):
                return result
        return None

    def _extract_field_single(self, block: Tag, selector: str) -> Any:
        """Extract a single field using one (non-comma) selector."""
        # ::text
        m = PSEUDO_TEXT.match(selector)
        if m:
            base_sel = m.group(1).strip()
            node = self._select_one(block, base_sel)
            return self._clean_text(node.get_text(" ", strip=True)) if node else None

        # ::attr(name)
        m = PSEUDO_ATTR.match(selector)
        if m:
            base_sel, attr = m.group(1).strip(), m.group(2)
            node = self._select_one(block, base_sel)
            if node and node.has_attr(attr):
                return node[attr].strip()
            return None

        # ::html
        m = PSEUDO_HTML.match(selector)
        if m:
            base_sel = m.group(1).strip()
            node = self._select_one(block, base_sel)
            return str(node) if node else None

        # Plain selector: try text by default, then attribute 'href' for <a>
        node = self._select_one(block, selector)
        if node is None:
            return None
        if node.name in ("a", "link") and node.has_attr("href"):
            return node["href"].strip()
        if node.name == "img":
            return node.get("src", "").strip() or None
        return self._clean_text(node.get_text(" ", strip=True))

    def _select_one(self, element: Tag, selector: str) -> Optional[Tag]:
        """Select the first element matching a CSS selector."""
        try:
            return element.select_one(selector)
        except Exception:  # noqa: BLE001
            logger.warning("Invalid CSS selector: {}", selector)
            return None

    def _select_all(self, element: Tag, selector: str) -> List[Tag]:
        """Select all elements matching a CSS selector."""
        try:
            return element.select(selector)
        except Exception:  # noqa: BLE001
            logger.warning("Invalid CSS selector: {}", selector)
            return []

    @staticmethod
    def _clean_text(value: Optional[str]) -> Optional[str]:
        """Normalize whitespace in extracted text."""
        if value is None:
            return None
        text = re.sub(r"\s+", " ", value).strip()
        return text or None

    # ------------------------------------------------------------------ #
    # Pagination detection
    # ------------------------------------------------------------------ #
    def _has_next_page(self, html: str, current_page: int) -> bool:
        """Detect if a next page exists.

        Checks common pagination patterns in the HTML: next links, page numbers.
        """
        if not self.paginated:
            return False
        soup = BeautifulSoup(html, "html.parser")

        # Look for a next-page link
        next_selectors = [
            "a[rel='next']",
            "a.next",
            "a.next-page",
            "li.next a",
            "a[aria-label='Next']",
            "a[aria-label='Next page']",
        ]
        for sel in next_selectors:
            link = soup.select_one(sel)
            if link:
                return True

        # Look for pagination numbers greater than current page
        pagination = soup.select_one(".pagination, .pager, nav[aria-label*='pagination' i], nav[aria-label*='Pagination']")
        if pagination:
            numbers = set()
            for a in pagination.select("a"):
                try:
                    numbers.add(int(a.get_text(strip=True)))
                except ValueError:
                    continue
            if any(n > current_page for n in numbers):
                return True
        return False

    def __enter__(self) -> "BaseScraper":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

