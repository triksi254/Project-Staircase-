"""Global configuration for the scraper.

Loads settings from environment variables (via .env) and YAML config files.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings

# Project root is two levels up from this file: config/settings.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root if present
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
FAQ_CORPUS_PATH = DATA_DIR / "faq_corpus.json"


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "env_file_encoding": "utf-8", "extra": "ignore"}

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default=str(LOGS_DIR / "scraper.log"))

    # Database
    DATABASE_URL: str = Field(default="sqlite:///./university_data.db")

    # Behavior
    USE_PLAYWRIGHT: bool = Field(default=True)
    VALIDATE_DATA: bool = Field(default=True)
    INTEGRATE_FAQ: bool = Field(default=True)

    # Scraper defaults
    DEFAULT_DELAY: float = Field(default=1.5)
    DEFAULT_MAX_PAGES: int = Field(default=10)
    DEFAULT_TIMEOUT: int = Field(default=30)
    WAIT_SELECTOR_TIMEOUT: int = Field(default=15000)
    MAX_RETRIES: int = Field(default=3)
    RETRY_BACKOFF: float = Field(default=2.0)

    # FAQ
    FAQ_RELEVANCE_THRESHOLD: float = Field(default=0.10)

    # Headers
    USER_AGENT: str = Field(
        default="UniversityScraper/1.0 (+https://github.com/example/university-scraper; educational research)"
    )

    def ensure_directories(self) -> None:
        """Create required runtime directories."""
        for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, LOGS_DIR):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, returning an empty dict if missing."""
    if not path.exists():
        logger.warning("Config file not found: {}", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


@lru_cache
def get_institutions() -> Dict[str, Any]:
    """Return the institutions configuration dictionary."""
    return load_yaml(CONFIG_DIR / "institutions.yaml").get("institutions", {})


@lru_cache
def get_selectors() -> Dict[str, Any]:
    """Return the selectors configuration dictionary."""
    return load_yaml(CONFIG_DIR / "selectors.yaml")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings

