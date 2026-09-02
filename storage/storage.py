"""Storage manager: exports scraped data to JSON, CSV, SQLite and PostgreSQL."""
from __future__ import annotations

import csv
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel

from config.settings import PROCESSED_DIR, RAW_DIR, Settings, get_settings
from scrapers.base_scraper import ScrapeMetadata


class JSONExporter:
    """Export data to JSON files."""

    @staticmethod
    def export(
        institution: str,
        data_type: str,
        data: List[Union[dict, BaseModel]],
        metadata: Dict[str, Any],
        directory: Path,
        data_kind: str = "processed",
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{institution}_{data_type}_{data_kind}_{ts}.json"
        path = directory / filename

        records = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in data
        ]
        payload = {"metadata": metadata, "data": records}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Exported {} records to JSON: {}", len(records), path)
        return path


class CSVExporter:
    """Export data to CSV files (flattening nested fields)."""

    @staticmethod
    def export(
        institution: str,
        data_type: str,
        data: List[Union[dict, BaseModel]],
        metadata: Dict[str, Any],
        directory: Path,
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{institution}_{data_type}_{ts}.csv"
        path = directory / filename

        records = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in data
        ]
        if not records:
            # Write header-only file with metadata
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh)
                writer.writerow(["No data scraped"])
            logger.info("No records to export to CSV; wrote placeholder: {}", path)
            return path

        # Flatten each record; collect all keys
        flat_records: List[Dict[str, Any]] = []
        all_keys: set[str] = set()
        for rec in records:
            flat = {}
            for key, value in rec.items():
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(value, ensure_ascii=False)
                elif value is None:
                    flat[key] = ""
                else:
                    flat[key] = str(value)
                all_keys.add(key)
            flat_records.append(flat)

        fieldnames = sorted(all_keys)
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_records)
        logger.info("Exported {} records to CSV: {}", len(flat_records), path)
        return path


class StorageManager:
    """High-level facade combining JSON/CSV export and database storage.

    Matches the programmatic usage documented in the README:

        storage = StorageManager()
        storage.save_results("aston_university", "courses", data, metadata,
                             formats=["json", "csv", "database"])
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def save_results(
        self,
        institution: str,
        data_type: str,
        data: List[Union[dict, BaseModel]],
        metadata: Union[ScrapeMetadata, Dict[str, Any]],
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Save results to the requested formats and return saved file paths/stats."""
        formats = formats or ["json"]
        meta = metadata.to_dict() if hasattr(metadata, "to_dict") else metadata

        saved_files: List[str] = []
        stats: Dict[str, Any] = {}

        if "json" in formats:
            path = JSONExporter.export(institution, data_type, data, meta, PROCESSED_DIR)
            saved_files.append(str(path))
        if "csv" in formats:
            path = CSVExporter.export(institution, data_type, data, meta, PROCESSED_DIR)
            saved_files.append(str(path))
        if "database" in formats:
            db = DatabaseStorage(self.settings.DATABASE_URL)
            db.save_records(institution, data_type, data)
            stats = db.get_statistics()

        return {"saved_files": saved_files, "database_stats": stats}


class DatabaseStorage:
    """Persist records into SQLite or PostgreSQL."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql://") or self.database_url.startswith("postgres://")

    def _connect(self):
        if self.is_postgres():
            try:
                import psycopg2  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL storage requested but psycopg2 is not installed. "
                    "Install with: pip install psycopg2-binary"
                ) from exc
            return psycopg2.connect(self.database_url)
        # SQLite: strip sqlite:/// prefix
        db_path = self.database_url.replace("sqlite:///", "", 1)
        if db_path == "":  # empty means :memory:
            db_path = ":memory:"
        return sqlite3.connect(db_path)

    def save_records(
        self,
        institution: str,
        data_type: str,
        records: List[Union[dict, BaseModel]],
    ) -> None:
        """Insert/upsert records into the target database."""
        conn = self._connect()
        try:
            table = self._table_name(institution, data_type)
            self._create_table(conn, table, data_type)
            self._insert_records(conn, table, records)
            count = self._count_rows(conn, table)
            logger.info("Database '{}' now has {} rows", table, count)
        finally:
            conn.close()

    def get_statistics(self) -> Dict[str, int]:
        """Return database statistics: total courses, accommodation, institutions."""
        conn = self._connect()
        stats: Dict[str, int] = {"courses": 0, "accommodation": 0, "institutions": 0}
        try:
            cursor = conn.cursor()
            tables = self._list_tables(cursor)
            for table in tables:
                parts = table.rsplit("_", 1)
                if len(parts) == 2 and parts[1] in ("courses", "accommodation"):
                    cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                    stats[parts[1]] += cursor.fetchone()[0]
            cursor.execute(self._distinct_institutions_sql(cursor))
            stats["institutions"] = cursor.fetchone()[0]
        finally:
            conn.close()
        return stats

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _table_name(self, institution: str, data_type: str) -> str:
        return f"{institution}_{data_type}"

    def _create_table(self, conn, table: str, data_type: str) -> None:
        cursor = conn.cursor()
        if self.is_postgres():
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS "{table}" (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT UNIQUE,
                    description TEXT,
                    institution TEXT,
                    data_type TEXT,
                    scraped_at TIMESTAMPTZ DEFAULT NOW(),
                    related_faqs JSONB DEFAULT '[]'::jsonb,
                    extra JSONB DEFAULT '{{}}'::jsonb
                )"""
            )
        else:
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS "{table}" (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    url TEXT UNIQUE,
                    description TEXT,
                    institution TEXT,
                    data_type TEXT,
                    scraped_at TEXT,
                    related_faqs TEXT,
                    extra TEXT
                )"""
            )
        conn.commit()

    def _insert_records(self, conn, table: str, records: List[Union[dict, BaseModel]]) -> None:
        cursor = conn.cursor()
        for rec in records:
            item = rec.model_dump(mode="json") if isinstance(rec, BaseModel) else rec
            related_faqs = json.dumps(item.get("related_faqs", []), ensure_ascii=False)
            extra_keys = [
                k
                for k in item
                if k
                not in {
                    "title",
                    "url",
                    "description",
                    "institution",
                    "data_type",
                    "scraped_at",
                    "related_faqs",
                }
            ]
            extra = json.dumps({k: item[k] for k in extra_keys if item[k] is not None}, ensure_ascii=False)
            scraped_at = item.get("scraped_at") or datetime.now().isoformat()

            if self.is_postgres():
                cursor.execute(
                    f"""INSERT INTO "{table}"
                        (title, url, description, institution, data_type, scraped_at, related_faqs, extra)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO NOTHING""",
                    (
                        item.get("title"),
                        item.get("url"),
                        item.get("description"),
                        item.get("institution"),
                        item.get("data_type"),
                        scraped_at,
                        related_faqs,
                        extra,
                    ),
                )
            else:
                cursor.execute(
                    f"""INSERT OR IGNORE INTO "{table}"
                        (title, url, description, institution, data_type, scraped_at, related_faqs, extra)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item.get("title"),
                        item.get("url"),
                        item.get("description"),
                        item.get("institution"),
                        item.get("data_type"),
                        scraped_at,
                        related_faqs,
                        extra,
                    ),
                )
        conn.commit()

    def _count_rows(self, conn, table: str) -> int:
        cursor = conn.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        return int(cursor.fetchone()[0])

    def _list_tables(self, cursor) -> List[str]:
        if self.is_postgres():
            cursor.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        else:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        return [row[0] for row in cursor.fetchall()]

    def _distinct_institutions_sql(self, cursor) -> str:
        if self.is_postgres():
            return "SELECT COUNT(DISTINCT institution) FROM (SELECT institution FROM public.courses UNION SELECT institution FROM public.accommodation) t"
        # For SQLite we union all institution columns across *_courses / *_accommodation tables
        tables = self._list_tables(cursor)
        parts = []
        for t in tables:
            if t.endswith("_courses") or t.endswith("_accommodation"):
                parts.append(f'SELECT institution FROM "{t}"')
        if not parts:
            return "SELECT 0"
        return "SELECT COUNT(DISTINCT institution) FROM (" + " UNION ".join(parts) + ")"

