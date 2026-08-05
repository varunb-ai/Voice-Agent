"""Storage backends shared by the Database Agent.

One small interface — upsert / get / all — implemented twice:

  * PostgresBackend — real psycopg connection to DATABASE_URL.
  * JsonBackend     — files under data/db/<table>.json, used automatically when
                      Postgres isn't reachable so the whole pipeline still runs.

Both use the same deterministic TEXT primary keys, so switching from JSON to
Postgres later changes nothing in the agents that call this.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from core.config import settings

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "agents" / "database" / "schema.sql"
_JSON_DIR = Path("data/db")


def slug(*parts: str) -> str:
    """Deterministic id from natural keys, e.g. slug('Apollo','Dallas') -> 'apollo::dallas'."""
    cleaned = []
    for p in parts:
        if not p:
            continue
        s = re.sub(r"[^a-z0-9]+", "-", p.strip().lower()).strip("-")
        if s:
            cleaned.append(s)
    return "::".join(cleaned)


# ── Postgres backend ─────────────────────────────────────────────────────────

class PostgresBackend:
    name = "postgres"

    def __init__(self, conn):
        self._conn = conn

    def init_schema(self) -> None:
        ddl = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._conn, self._conn.cursor() as cur:
            cur.execute(ddl)

    def upsert(self, table: str, pk_col: str, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        placeholders = ", ".join(f"%({c})s" for c in cols)
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != pk_col)
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT ({pk_col}) DO UPDATE SET {updates}"
            if updates
            else f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
                 f"ON CONFLICT ({pk_col}) DO NOTHING"
        )
        prepared = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                    for k, v in row.items()}
        with self._conn, self._conn.cursor() as cur:
            cur.execute(sql, prepared)

    def all(self, table: str) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table}")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def count(self, table: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()


# ── JSON fallback backend ────────────────────────────────────────────────────

class JsonBackend:
    name = "json"

    def __init__(self, root: Path = _JSON_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def init_schema(self) -> None:
        for table in ("hospitals", "branches", "doctors", "calls", "emails"):
            f = self.root / f"{table}.json"
            if not f.exists():
                f.write_text("{}", encoding="utf-8")

    def _path(self, table: str) -> Path:
        return self.root / f"{table}.json"

    def _load(self, table: str) -> dict[str, dict]:
        p = self._path(table)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def upsert(self, table: str, pk_col: str, row: dict[str, Any]) -> None:
        data = self._load(table)
        data[row[pk_col]] = {**data.get(row[pk_col], {}), **row}
        self._path(table).write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def all(self, table: str) -> list[dict]:
        return list(self._load(table).values())

    def count(self, table: str) -> int:
        return len(self._load(table))

    def close(self) -> None:
        pass


# ── Selection ────────────────────────────────────────────────────────────────

def get_backend():
    """Return a Postgres backend if reachable, else the JSON fallback."""
    try:
        import psycopg
        conn = psycopg.connect(settings.database_url, connect_timeout=3)
        return PostgresBackend(conn)
    except Exception:
        return JsonBackend()
