from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def request_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SQLiteCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        statement = (
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            "cache_key TEXT PRIMARY KEY, "
            "response TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        with self._connect() as connection:
            connection.execute(statement)

    def get(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT response FROM llm_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        return None if row is None else str(row[0])

    def put(self, key: str, response: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        statement = (
            "INSERT INTO llm_cache(cache_key, response, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(cache_key) DO UPDATE SET "
            "response = excluded.response, created_at = excluded.created_at"
        )
        with self._connect() as connection:
            connection.execute(statement, (key, response, now))
