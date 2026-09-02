"""Shared Playwright page-interaction helpers for the requirements scrapers.

Small, site-agnostic helpers (browser/page creation and accordion expansion)
used by ``scrape_all_universities.py`` and ``scrape_fix_remaining.py`` so the
Kenyan requirements scrapers don't each re-implement the same Playwright
boilerplate with slightly different results.
"""
from __future__ import annotations

from typing import Optional, Tuple

from loguru import logger

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "Chrome/125.0 Safari/537.36"
)

ACCORDION_SELECTORS = [
    "button[aria-expanded]",
    ".accordion-trigger",
    ".accordion-header button",
    "[data-toggle='collapse']",
    ".toggle",
    ".collapse-toggle",
    "button.collapsed",
    "button[data-bs-toggle='collapse']",
    "details summary",
    ".accordion-item button",
    "button[aria-controls]",
    "button:has(span)",
]


def launch_browser(
    playwright,
    user_agent: str = DEFAULT_USER_AGENT,
    viewport: Optional[dict] = None,
) -> Tuple[object, object, object]:
    """Launch a headless Chromium and return ``(browser, context, page)``.

    Callers remain responsible for closing the browser/context (e.g. in a
    ``finally`` block), matching the existing scrapers' lifecycle.
    """
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent=user_agent,
        viewport=viewport or {"width": 1366, "height": 900},
    )
    page = context.new_page()
    return browser, context, page


def click_accordions(page, wait_ms: int = 500) -> int:
    """Expand collapsed accordion/expandable triggers to reveal hidden content.

    Best-effort: selector/click failures are skipped silently. Only triggers
    whose ``aria-expanded`` is ``"false"`` or unset are clicked, and each
    distinct trigger (by inner HTML) is expanded at most once.
    """
    clicked = set()
    for selector in ACCORDION_SELECTORS:
        try:
            buttons = page.query_selector_all(selector)
        except Exception:  # noqa: BLE001 - non-standard selector
            continue
        for btn in buttons:
            try:
                outer = btn.inner_html()
                if outer in clicked:
                    continue
                expanded = btn.get_attribute("aria-expanded")
                if expanded == "false" or expanded is None:
                    btn.click()
                    clicked.add(outer)
                    page.wait_for_timeout(wait_ms)
            except Exception:  # noqa: BLE001 - click may detach the node
                continue
    logger.info("  Clicked {} unique accordion elements", len(clicked))
    return len(clicked)