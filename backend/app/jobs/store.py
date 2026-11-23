"""Historique des générations, persisté dans SQLite (un fichier, aucun serveur)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'generate',
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    params TEXT NOT NULL,
    result TEXT,
    error TEXT,
    upload_id TEXT,
    parent_id TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at DESC);
"""

JSON_FIELDS = ("params", "result")
STATUSES = ("queued", "running", "done", "failed", "cancelled")


class JobStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _row(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        for f in JSON_FIELDS:
            d[f] = json.loads(d[f]) if d[f] else None
        return d

    def create(self, job_id: str, name: str, params: dict, kind: str = "generate", upload_id: str | None = None,
               parent_id: str | None = None) -> dict:
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, name, kind, status, params, upload_id, parent_id, created_at, message)"
                " VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, 'En attente')",
                (job_id, name, kind, json.dumps(params), upload_id, parent_id, time.time()),
            )
        return self.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        for f in JSON_FIELDS:
            if f in fields and fields[f] is not None:
                fields[f] = json.dumps(fields[f])
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row(row)

    def list(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self._row(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])

    def ids_with_status(self, *statuses: str) -> list[str]:
        q = ",".join("?" * len(statuses))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id FROM jobs WHERE status IN ({q}) ORDER BY created_at", statuses
            ).fetchall()
        return [r[0] for r in rows]

    def oldest_finished_beyond(self, keep: int) -> list[str]:
        """Identifiants des générations terminées au-delà des ``keep`` plus récentes."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE status IN ('done','failed','cancelled')"
                " ORDER BY created_at DESC LIMIT -1 OFFSET ?",
                (keep,),
            ).fetchall()
        return [r[0] for r in rows]

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def close(self) -> None:
        self._conn.close()
