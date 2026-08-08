"""Scrapers package for the University Web Scraper."""
from scrapers.base_scraper import BaseScraper
from scrapers.html_scraper import HTMLScraper
from scrapers.js_scraper import JavaScriptScraper
from scrapers.smart_scraper import SmartScraper

__all__ = ["BaseScraper", "HTMLScraper", "JavaScriptScraper", "SmartScraper"]

