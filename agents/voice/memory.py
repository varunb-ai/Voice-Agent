"""Layer 5 — Memory.

Current-call memory lives in Redis (fast, ephemeral, good for real-time). If Redis
isn't running we transparently fall back to an in-process dict so the brain stays
testable offline. Long-term memory is PostgreSQL (the Database Agent), not here.

The memory holds the evolving fact sheet for one call:
    { doctor, specialization, hospital, branch, city, notes, transcript[] }
"""
from __future__ import annotations

import json
from typing import Any, Optional

from core.config import settings


class CallMemory:
    """Per-call scratchpad keyed by a call id. Redis-backed, dict fallback."""

    def __init__(self, call_id: str, ttl_seconds: int = 3600):
        self.call_id = call_id
        self.key = f"call:{call_id}"
        self.ttl = ttl_seconds
        self._redis = self._connect()
        self._local: dict[str, Any] = {}      # used only when Redis is unavailable

    @staticmethod
    def _connect():
        try:
            import redis  # imported lazily; optional dependency at runtime
            client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception:
            return None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def _load(self) -> dict[str, Any]:
        if self._redis is not None:
            raw = self._redis.get(self.key)
            return json.loads(raw) if raw else {}
        return dict(self._local)

    def _store(self, data: dict[str, Any]) -> None:
        if self._redis is not None:
            self._redis.set(self.key, json.dumps(data), ex=self.ttl)
        else:
            self._local = data

    def update(self, **fields: Any) -> dict[str, Any]:
        """Merge new facts into the call sheet (ignores Nones)."""
        data = self._load()
        for k, v in fields.items():
            if v is not None:
                data[k] = v
        self._store(data)
        return data

    def append_turn(self, role: str, text: str) -> None:
        data = self._load()
        data.setdefault("transcript", []).append({"role": role, "text": text})
        self._store(data)

    def get(self, field: str, default: Any = None) -> Any:
        return self._load().get(field, default)

    def snapshot(self) -> dict[str, Any]:
        return self._load()

    def clear(self) -> None:
        if self._redis is not None:
            self._redis.delete(self.key)
        self._local = {}
