"""
Widget runtime service — canonical state management for live UI widgets.

Provides:
- WidgetRecord: immutable snapshot of a widget's state
- WidgetStore (Protocol): abstract storage interface
- RedisWidgetStore: cross-process shared store for real MCP + HTTP flows
- InMemoryWidgetStore: isolated unit-test store
- WidgetConnectionManager: per-widget WebSocket fan-out
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from fastapi import WebSocket

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_WIDGET_STATE_BYTES = 256 * 1024  # 256 KB
DEFAULT_WIDGET_TTL_SECONDS = 3600  # 1 hour
# A Redis write that keeps losing its WATCH race gives up instead of spinning.
_WATCH_RETRY_LIMIT = 10


class WidgetStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# WidgetRecord
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WidgetRecord:
    widget_id: str
    session_id: str
    title: str | None
    state: dict[str, Any]
    status: WidgetStatus
    version: int
    created_at: float
    updated_at: float
    expires_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "widget_id": self.widget_id,
            "session_id": self.session_id,
            "title": self.title,
            "state": self.state,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }


def _kept_changing(widget_id: str) -> ValueError:
    return ValueError(
        f"Widget {widget_id} kept changing during the write "
        f"({_WATCH_RETRY_LIMIT} attempts); try again"
    )


def _validate_state_size(state: dict[str, Any]) -> None:
    serialized = json.dumps(state, separators=(",", ":"))
    if len(serialized.encode()) > MAX_WIDGET_STATE_BYTES:
        raise ValueError(f"Widget state exceeds maximum size of {MAX_WIDGET_STATE_BYTES} bytes")


# ---------------------------------------------------------------------------
# WidgetStore protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class WidgetStore(Protocol):
    async def create(
        self,
        session_id: str,
        initial_state: dict[str, Any],
        title: str | None = None,
        ttl_seconds: int = DEFAULT_WIDGET_TTL_SECONDS,
    ) -> WidgetRecord: ...

    async def get(self, widget_id: str) -> WidgetRecord | None: ...

    async def update(
        self,
        widget_id: str,
        state: dict[str, Any],
        expected_version: int | None = None,
    ) -> WidgetRecord: ...

    async def patch(
        self,
        widget_id: str,
        patch: dict[str, Any],
    ) -> WidgetRecord: ...

    async def close(self, widget_id: str) -> WidgetRecord: ...

    async def list_by_session(self, session_id: str) -> list[WidgetRecord]: ...

    async def restore(
        self,
        *,
        widget_id: str,
        session_id: str,
        state: dict[str, Any],
        title: str | None = None,
        status: str = WidgetStatus.ACTIVE.value,
        version: int = 1,
        ttl_seconds: int = DEFAULT_WIDGET_TTL_SECONDS,
    ) -> WidgetRecord: ...


# ---------------------------------------------------------------------------
# In-memory store (tests only)
# ---------------------------------------------------------------------------
class InMemoryWidgetStore:
    """Thread-safe in-memory widget store for isolated unit tests."""

    def __init__(self) -> None:
        self._widgets: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        session_id: str,
        initial_state: dict[str, Any],
        title: str | None = None,
        ttl_seconds: int = DEFAULT_WIDGET_TTL_SECONDS,
    ) -> WidgetRecord:
        _validate_state_size(initial_state)
        now = time.time()
        widget_id = str(uuid.uuid4())
        record_data = {
            "widget_id": widget_id,
            "session_id": session_id,
            "title": title,
            "state": initial_state,
            "status": WidgetStatus.ACTIVE.value,
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "expires_at": now + ttl_seconds,
            "ttl_seconds": ttl_seconds,
        }
        async with self._lock:
            self._widgets[widget_id] = record_data
        return self._to_record(record_data)

    async def get(self, widget_id: str) -> WidgetRecord | None:
        async with self._lock:
            data = self._widgets.get(widget_id)
        if data is None:
            return None
        if data["expires_at"] < time.time():
            async with self._lock:
                self._widgets.pop(widget_id, None)
            return None
        return self._to_record(data)

    async def update(
        self,
        widget_id: str,
        state: dict[str, Any],
        expected_version: int | None = None,
    ) -> WidgetRecord:
        _validate_state_size(state)
        async with self._lock:
            data = self._widgets.get(widget_id)
            if data is None:
                raise KeyError(f"Widget {widget_id} not found")
            if data["status"] == WidgetStatus.CLOSED.value:
                raise ValueError(f"Widget {widget_id} is closed")
            if expected_version is not None and data["version"] != expected_version:
                raise ValueError(
                    f"Version mismatch: expected {expected_version}, current {data['version']}"
                )
            data["state"] = state
            data["version"] += 1
            data["updated_at"] = time.time()
            data["expires_at"] = data["updated_at"] + int(
                data.get("ttl_seconds", DEFAULT_WIDGET_TTL_SECONDS)
            )
        return self._to_record(data)

    async def patch(
        self,
        widget_id: str,
        patch: dict[str, Any],
    ) -> WidgetRecord:
        async with self._lock:
            data = self._widgets.get(widget_id)
            if data is None:
                raise KeyError(f"Widget {widget_id} not found")
            if data["status"] == WidgetStatus.CLOSED.value:
                raise ValueError(f"Widget {widget_id} is closed")
            merged = {**data["state"], **patch}
            _validate_state_size(merged)
            data["state"] = merged
            data["version"] += 1
            data["updated_at"] = time.time()
            data["expires_at"] = data["updated_at"] + int(
                data.get("ttl_seconds", DEFAULT_WIDGET_TTL_SECONDS)
            )
        return self._to_record(data)

    async def close(self, widget_id: str) -> WidgetRecord:
        async with self._lock:
            data = self._widgets.get(widget_id)
            if data is None:
                raise KeyError(f"Widget {widget_id} not found")
            data["status"] = WidgetStatus.CLOSED.value
            data["version"] += 1
            data["updated_at"] = time.time()
            data["expires_at"] = data["updated_at"] + int(
                data.get("ttl_seconds", DEFAULT_WIDGET_TTL_SECONDS)
            )
        return self._to_record(data)

    async def list_by_session(self, session_id: str) -> list[WidgetRecord]:
        now = time.time()
        results: list[WidgetRecord] = []
        async with self._lock:
            for data in self._widgets.values():
                if data["session_id"] == session_id and data["expires_at"] >= now:
                    results.append(self._to_record(data))
        return results

    async def restore(
        self,
        *,
        widget_id: str,
        session_id: str,
        state: dict[str, Any],
        title: str | None = None,
        status: str = WidgetStatus.ACTIVE.value,
        version: int = 1,
        ttl_seconds: int = DEFAULT_WIDGET_TTL_SECONDS,
    ) -> WidgetRecord:
        _validate_state_size(state)
        now = time.time()
        record_data = {
            "widget_id": widget_id,
            "session_id": session_id,
            "title": title,
            "state": state,
            "status": WidgetStatus(status).value,
            "version": max(1, int(version or 1)),
            "created_at": now,
            "updated_at": now,
            "expires_at": now + ttl_seconds,
            "ttl_seconds": ttl_seconds,
        }
        async with self._lock:
            self._widgets[widget_id] = record_data
        return self._to_record(record_data)

    @staticmethod
    def _to_record(data: dict[str, Any]) -> WidgetRecord:
        return WidgetRecord(
            widget_id=data["widget_id"],
            session_id=data["session_id"],
            title=data["title"],
            state=data["state"],
            status=WidgetStatus(data["status"]),
            version=data["version"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            expires_at=data["expires_at"],
        )


# ---------------------------------------------------------------------------
# Redis-backed store
# ---------------------------------------------------------------------------
class RedisWidgetStore:
    """Cross-process widget store backed by Redis."""

    KEY_PREFIX = "widget:"
    SESSION_INDEX_PREFIX = "widget_session:"

    def __init__(self, redis_url: str | None = None) -> None:
        import redis as _redis

        url = redis_url or settings.redis_url or settings.celery_broker_url
        if not url:
            raise RuntimeError("No Redis URL configured for widget runtime")
        self._redis = _redis.from_url(url, decode_responses=True)
        self._watch_error = _redis.exceptions.WatchError

    def _key(self, widget_id: str) -> str:
        return f"{self.KEY_PREFIX}{widget_id}"

    def _session_key(self, session_id: str) -> str:
        return f"{self.SESSION_INDEX_PREFIX}{session_id}"

    async def create(
        self,
        session_id: str,
        initial_state: dict[str, Any],
        title: str | None = None,
        ttl_seconds: int = DEFAULT_WIDGET_TTL_SECONDS,
    ) -> WidgetRecord:
        _validate_state_size(initial_state)
        now = time.time()
        widget_id = str(uuid.uuid4())
        record_data = {
            "widget_id": widget_id,
            "session_id": session_id,
            "title": title or "",
            "state": json.dumps(initial_state, separators=(",", ":")),
            "status": WidgetStatus.ACTIVE.value,
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "expires_at": now + ttl_seconds,
            "ttl_seconds": ttl_seconds,
        }
        key = self._key(widget_id)
        pipe = self._redis.pipeline()
        pipe.hset(key, mapping=record_data)
        pipe.expire(key, ttl_seconds)
        pipe.sadd(self._session_key(session_id), widget_id)
        pipe.expire(self._session_key(session_id), ttl_seconds)
        await asyncio.to_thread(pipe.execute)
        return self._to_record(record_data)

    async def get(self, widget_id: str) -> WidgetRecord | None:
        data = await asyncio.to_thread(self._redis.hgetall, self._key(widget_id))
        if not data:
            return None
        return self._to_record(data)

    async def update(
        self,
        widget_id: str,
        state: dict[str, Any],
        expected_version: int | None = None,
    ) -> WidgetRecord:
        _validate_state_size(state)
        key = self._key(widget_id)

        def _do_update():
            with self._redis.pipeline() as pipe:
                for _attempt in range(_WATCH_RETRY_LIMIT):
                    try:
                        pipe.watch(key)
                        data = pipe.hgetall(key)
                        if not data:
                            raise KeyError(f"Widget {widget_id} not found")
                        if data.get("status") == WidgetStatus.CLOSED.value:
                            raise ValueError(f"Widget {widget_id} is closed")
                        current_version = int(data.get("version", 0))
                        if expected_version is not None and current_version != expected_version:
                            raise ValueError(
                                f"Version mismatch: expected {expected_version}, "
                                f"current {current_version}"
                            )
                        new_version = current_version + 1
                        now = time.time()
                        ttl_seconds = int(data.get("ttl_seconds", DEFAULT_WIDGET_TTL_SECONDS))
                        session_key = self._session_key(str(data.get("session_id", "")))
                        pipe.multi()
                        pipe.hset(
                            key,
                            mapping={
                                "state": json.dumps(state, separators=(",", ":")),
                                "version": new_version,
                                "updated_at": now,
                                "expires_at": now + ttl_seconds,
                            },
                        )
                        pipe.expire(key, ttl_seconds)
                        if session_key != self._session_key(""):
                            pipe.expire(session_key, ttl_seconds)
                        pipe.execute()
                        data["state"] = json.dumps(state, separators=(",", ":"))
                        data["version"] = new_version
                        data["updated_at"] = now
                        data["expires_at"] = now + ttl_seconds
                        return self._to_record(data)
                    except self._watch_error:
                        continue
            raise _kept_changing(widget_id)

        return await asyncio.to_thread(_do_update)

    async def patch(
        self,
        widget_id: str,
        patch: dict[str, Any],
    ) -> WidgetRecord:
        key = self._key(widget_id)

        def _do_patch():
            with self._redis.pipeline() as pipe:
                for _attempt in range(_WATCH_RETRY_LIMIT):
                    try:
                        pipe.watch(key)
                        data = pipe.hgetall(key)
                        if not data:
                            raise KeyError(f"Widget {widget_id} not found")
                        if data.get("status") == WidgetStatus.CLOSED.value:
                            raise ValueError(f"Widget {widget_id} is closed")
                        current_state = json.loads(data.get("state", "{}"))
                        merged = {**current_state, **patch}
                        _validate_state_size(merged)
                        new_version = int(data.get("version", 0)) + 1
                        now = time.time()
                        ttl_seconds = int(data.get("ttl_seconds", DEFAULT_WIDGET_TTL_SECONDS))
                        session_key = self._session_key(str(data.get("session_id", "")))
                        pipe.multi()
                        pipe.hset(
                            key,
                            mapping={
                                "state": json.dumps(merged, separators=(",", ":")),
                                "version": new_version,
                                "updated_at": now,
                                "expires_at": now + ttl_seconds,
                            },
                        )
                        pipe.expire(key, ttl_seconds)
                        if session_key != self._session_key(""):
                            pipe.expire(session_key, ttl_seconds)
                        pipe.execute()
                        data["state"] = json.dumps(merged, separators=(",", ":"))
                        data["version"] = new_version
                        data["updated_at"] = now
                        data["expires_at"] = now + ttl_seconds
                        return self._to_record(data)
                    except self._watch_error:
                        continue
            raise _kept_changing(widget_id)

        return await asyncio.to_thread(_do_patch)

    async def close(self, widget_id: str) -> WidgetRecord:
        key = self._key(widget_id)

        def _do_close():
            with self._redis.pipeline() as pipe:
                for _attempt in range(_WATCH_RETRY_LIMIT):
                    try:
                        pipe.watch(key)
                        data = pipe.hgetall(key)
                        if not data:
                            raise KeyError(f"Widget {widget_id} not found")
                        now = time.time()
                        new_version = int(data.get("version", 0)) + 1
                        ttl_seconds = int(data.get("ttl_seconds", DEFAULT_WIDGET_TTL_SECONDS))
                        session_key = self._session_key(str(data.get("session_id", "")))
                        pipe.multi()
                        pipe.hset(
                            key,
                            mapping={
                                "status": WidgetStatus.CLOSED.value,
                                "version": new_version,
                                "updated_at": now,
                                "expires_at": now + ttl_seconds,
                            },
                        )
                        pipe.expire(key, ttl_seconds)
                        if session_key != self._session_key(""):
                            pipe.expire(session_key, ttl_seconds)
                        pipe.execute()
                        data["status"] = WidgetStatus.CLOSED.value
                        data["version"] = new_version
                        data["updated_at"] = now
                        data["expires_at"] = now + ttl_seconds
                        return self._to_record(data)
                    except self._watch_error:
                        continue
            raise _kept_changing(widget_id)

        return await asyncio.to_thread(_do_close)

    async def list_by_session(self, session_id: str) -> list[WidgetRecord]:
        def _do_list():
            widget_ids = self._redis.smembers(self._session_key(session_id))
            results = []
            for wid in widget_ids:
                data = self._redis.hgetall(self._key(wid))
                if data:
                    results.append(self._to_record(data))
            return results

        return await asyncio.to_thread(_do_list)

    async def restore(
        self,
        *,
        widget_id: str,
        session_id: str,
        state: dict[str, Any],
        title: str | None = None,
        status: str = WidgetStatus.ACTIVE.value,
        version: int = 1,
        ttl_seconds: int = DEFAULT_WIDGET_TTL_SECONDS,
    ) -> WidgetRecord:
        _validate_state_size(state)
        now = time.time()
        record_data = {
            "widget_id": widget_id,
            "session_id": session_id,
            "title": title or "",
            "state": json.dumps(state, separators=(",", ":")),
            "status": WidgetStatus(status).value,
            "version": max(1, int(version or 1)),
            "created_at": now,
            "updated_at": now,
            "expires_at": now + ttl_seconds,
            "ttl_seconds": ttl_seconds,
        }
        key = self._key(widget_id)
        pipe = self._redis.pipeline()
        pipe.hset(key, mapping=record_data)
        pipe.expire(key, ttl_seconds)
        pipe.sadd(self._session_key(session_id), widget_id)
        pipe.expire(self._session_key(session_id), ttl_seconds)
        await asyncio.to_thread(pipe.execute)
        return self._to_record(record_data)

    @staticmethod
    def _to_record(data: dict[str, Any]) -> WidgetRecord:
        state_raw = data.get("state", "{}")
        state = json.loads(state_raw) if isinstance(state_raw, str) else state_raw
        return WidgetRecord(
            widget_id=str(data["widget_id"]),
            session_id=str(data["session_id"]),
            title=data.get("title") or None,
            state=state,
            status=WidgetStatus(data.get("status", "active")),
            version=int(data.get("version", 1)),
            created_at=float(data.get("created_at", 0)),
            updated_at=float(data.get("updated_at", 0)),
            expires_at=float(data.get("expires_at", 0)),
        )


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class WidgetConnectionManager:
    """Manages per-widget WebSocket connections and fan-out."""

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, widget_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            if widget_id not in self._connections:
                self._connections[widget_id] = set()
            self._connections[widget_id].add(websocket)

    async def disconnect(self, widget_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            conns = self._connections.get(widget_id)
            if conns:
                conns.discard(websocket)
                if not conns:
                    del self._connections[widget_id]

    async def broadcast(self, widget_id: str, event: dict[str, Any]) -> None:
        async with self._lock:
            conns = list(self._connections.get(widget_id, []))
        message = json.dumps(event)
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                conns_set = self._connections.get(widget_id)
                if conns_set:
                    for ws in dead:
                        conns_set.discard(ws)
                    if not conns_set:
                        del self._connections[widget_id]


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_widget_store: WidgetStore | None = None
_widget_connection_manager: WidgetConnectionManager | None = None


def get_widget_store() -> WidgetStore:
    global _widget_store
    if _widget_store is None:
        try:
            _widget_store = RedisWidgetStore()
            logger.info("Widget store: using Redis backend")
        except Exception:
            logger.warning("Widget store: Redis unavailable, falling back to in-memory (test only)")
            _widget_store = InMemoryWidgetStore()
    return _widget_store


def get_widget_connection_manager() -> WidgetConnectionManager:
    global _widget_connection_manager
    if _widget_connection_manager is None:
        _widget_connection_manager = WidgetConnectionManager()
    return _widget_connection_manager
