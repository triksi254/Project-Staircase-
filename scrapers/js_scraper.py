"""JavaScript scraper using Playwright for dynamic sites."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import DATA_DIR, Settings
from scrapers.base_scraper import BaseScraper


class JavaScriptScraper(BaseScraper):
    """Scraper for JavaScript-rendered pages using Playwright.

    Launches a headless Chromium, navigates to the page, waits for dynamic
    content (either a configured wait selector or a fixed timeout), then
    captures the fully rendered HTML.
    """

    SCRAPER_NAME = "javascript"

    def __init__(self, *args, wait_selector: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Prefer explicit argument, then YAML config
        self.wait_selector = wait_selector or self.source_cfg.get("wait_selector")
        self._browser = None
        self._context = None

    def _get_browser(self):
        """Lazily launch a Playwright browser and return (playwright, browser, context, page)."""
        if self._browser is not None:
            return self._browser

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            ) from exc

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=self.settings.USER_AGENT,
            viewport={"width": 1366, "height": 900},
            locale="en-GB",
        )
        page = context.new_page()
        self._browser = (playwright, browser, context, page)
        return self._browser

    def fetch_page(self, url: str) -> str:
        """Navigate to the URL, wait for content, return rendered HTML."""
        _, browser, context, page = self._get_browser()

        try:
            logger.debug("Playwright navigating to {}", url)
            response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)

            if response and response.status == 429:
                # Treat as rate-limit; BaseScraper retry logic will handle
                from requests.exceptions import HTTPError

                err = HTTPError(f"Rate limited: {response.status}")
                err.response = response
                raise err

            if self.wait_selector:
                try:
                    page.wait_for_selector(
                        self.wait_selector,
                        timeout=self.settings.WAIT_SELECTOR_TIMEOUT,
                    )
                    logger.debug("Wait selector '{}' satisfied", self.wait_selector)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Wait selector '{}' timed out: {}", self.wait_selector, exc)
                    self._capture_screenshot(page, url, prefix="timeout")
                    time.sleep(2)  # give JS a final chance
            else:
                # Fixed wait for dynamic content
                time.sleep(2)

            html = page.content()
            return html

        except Exception:
            try:
                self._capture_screenshot(page, url)
            except Exception:  # noqa: BLE001
                pass
            raise

    def _capture_screenshot(self, page, url: str, prefix: str = "error") -> None:
        """Capture a screenshot on JS errors/timeouts for debugging."""
        try:
            screenshots_dir = Path(DATA_DIR) / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            import hashlib
            import re

            slug = re.sub(r"[^a-zA-Z0-9]+", "_", url)[:60]
            digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
            filename = f"{prefix}_{self.institution}_{self.data_type}_{slug}_{digest}.png"
            path = screenshots_dir / filename
            page.screenshot(path=str(path))
            logger.info("Screenshot saved to {}", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not capture screenshot: {}", exc)

    def close(self) -> None:
        """Close the browser context and stop Playwright."""
        if self._browser is None:
            return
        _, browser, context, _ = self._browser
        try:
            context.close()
            browser.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing browser: {}", exc)
        try:
            self._browser[0].stop()
        except Exception:  # noqa: BLE001
            pass
        self._browser = None

