"""Shared store and dispatcher for client-side runtime sessions.

Supports two backends:
- Redis for production-safe multi-worker / multi-instance coordination
- In-memory fallback for local development and tests when Redis is unavailable
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

try:
    from redis import Redis as SyncRedis
    from redis.asyncio import Redis as AsyncRedis
except ImportError:  # pragma: no cover - exercised in environments without redis installed
    SyncRedis = None
    AsyncRedis = None

from app.core.config import settings
from app.schemas.runtime_protocol import (
    RuntimeErrorContext,
    ToolDispatchRequest,
    ToolDispatchResult,
)

logger = logging.getLogger(__name__)

_SESSION_KEY_PREFIX = "client-runtime:session"
_USER_DEVICES_KEY_PREFIX = "client-runtime:user-devices"
_ISSUED_SESSION_KEY_PREFIX = "client-runtime:issued-session"
_REQUEST_QUEUE_KEY_PREFIX = "client-runtime:request-queue"
_RESULT_QUEUE_KEY_PREFIX = "client-runtime:result-queue"
_PENDING_REQUESTS_KEY_PREFIX = "client-runtime:pending-requests"
_REQUEST_DEVICE_KEY_PREFIX = "client-runtime:request-device"
_RESULT_TTL_SECONDS = 300


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _session_ttl_seconds() -> int:
    return max(90, settings.client_runtime_heartbeat_interval_seconds * 3)


def _issued_session_ttl_seconds() -> int:
    return max(120, settings.client_runtime_ws_timeout_seconds * 2)


def _runtime_redis_url() -> str:
    explicit = (getattr(settings, "redis_url", "") or "").strip()
    candidate = explicit or (getattr(settings, "celery_broker_url", "") or "").strip()
    if not candidate:
        return ""

    scheme = urlparse(candidate).scheme.lower()
    if scheme in {"redis", "rediss", "unix"}:
        return candidate
    return ""


def _request_tracking_ttl_seconds(timeout_seconds: int | None = None) -> int:
    requested_timeout = int(timeout_seconds or settings.client_runtime_ws_timeout_seconds or 0)
    return max(_session_ttl_seconds(), requested_timeout + _RESULT_TTL_SECONDS)


def _disconnect_result(request_id: str, reason: str) -> ToolDispatchResult:
    return ToolDispatchResult(
        request_id=request_id,
        success=False,
        error=reason,
        error_context=RuntimeErrorContext(
            message=reason,
            code="DEVICE_DISCONNECTED",
        ),
        execution_time_ms=0,
    )


def _decode_session_payload(payload: str | None) -> DeviceSessionRecord | None:
    if not payload:
        return None

    try:
        return DeviceSessionRecord.from_payload(json.loads(payload))
    except Exception as exc:
        logger.warning("Failed to decode client runtime session payload: %s", exc)
        return None


@dataclass
class DeviceSessionRecord:
    """Serializable view of an active client runtime session."""

    device_id: UUID
    session_id: str
    user_id: UUID
    connected_at: datetime = field(default_factory=_utc_now)
    last_heartbeat: datetime = field(default_factory=_utc_now)
    tool_catalog: dict[str, Any] = field(default_factory=dict)
    skill_catalog: dict[str, Any] = field(default_factory=dict)
    tool_catalog_version: int = 0
    skill_catalog_version: int = 0
    tool_catalog_updated_at: datetime | None = None
    skill_catalog_updated_at: datetime | None = None

    def is_alive(self) -> bool:
        timeout = settings.client_runtime_heartbeat_interval_seconds * 2
        elapsed = (_utc_now() - self.last_heartbeat).total_seconds()
        return elapsed < timeout

    def update_heartbeat(self) -> None:
        self.last_heartbeat = _utc_now()

    def update_tool_catalog(self, catalog: dict[str, Any]) -> None:
        self.tool_catalog = catalog
        self.tool_catalog_version += 1
        self.tool_catalog_updated_at = _utc_now()

    def update_skill_catalog(self, catalog: dict[str, Any]) -> None:
        self.skill_catalog = catalog
        self.skill_catalog_version += 1
        self.skill_catalog_updated_at = _utc_now()

    def get_tool_cache_key(self) -> tuple[str, str, str, int]:
        return (
            str(self.user_id),
            str(self.device_id),
            self.session_id,
            self.tool_catalog_version,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "device_id": str(self.device_id),
            "session_id": self.session_id,
            "user_id": str(self.user_id),
            "connected_at": self.connected_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "tool_catalog": self.tool_catalog,
            "skill_catalog": self.skill_catalog,
            "tool_catalog_version": self.tool_catalog_version,
            "skill_catalog_version": self.skill_catalog_version,
            "tool_catalog_updated_at": (
                self.tool_catalog_updated_at.isoformat() if self.tool_catalog_updated_at else None
            ),
            "skill_catalog_updated_at": (
                self.skill_catalog_updated_at.isoformat() if self.skill_catalog_updated_at else None
            ),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DeviceSessionRecord:
        return cls(
            device_id=UUID(str(payload["device_id"])),
            session_id=str(payload["session_id"]),
            user_id=UUID(str(payload["user_id"])),
            connected_at=_coerce_datetime(payload.get("connected_at")) or _utc_now(),
            last_heartbeat=_coerce_datetime(payload.get("last_heartbeat")) or _utc_now(),
            tool_catalog=dict(payload.get("tool_catalog") or {}),
            skill_catalog=dict(payload.get("skill_catalog") or {}),
            tool_catalog_version=int(payload.get("tool_catalog_version") or 0),
            skill_catalog_version=int(payload.get("skill_catalog_version") or 0),
            tool_catalog_updated_at=_coerce_datetime(payload.get("tool_catalog_updated_at")),
            skill_catalog_updated_at=_coerce_datetime(payload.get("skill_catalog_updated_at")),
        )


class BaseClientRuntimeStore:
    """Store interface for runtime session metadata and dispatch queues."""

    async def issue_runtime_session_id(self, device_id: UUID, session_id: str) -> None:
        raise NotImplementedError

    async def consume_runtime_session_id(self, device_id: UUID, session_id: str) -> bool:
        raise NotImplementedError

    async def put_session(self, session: DeviceSessionRecord) -> None:
        raise NotImplementedError

    async def delete_session(self, device_id: UUID) -> DeviceSessionRecord | None:
        raise NotImplementedError

    async def update_heartbeat(self, device_id: UUID) -> DeviceSessionRecord | None:
        raise NotImplementedError

    async def update_tool_catalog(
        self,
        device_id: UUID,
        catalog: dict[str, Any],
    ) -> DeviceSessionRecord | None:
        raise NotImplementedError

    async def update_skill_catalog(
        self,
        device_id: UUID,
        catalog: dict[str, Any],
    ) -> DeviceSessionRecord | None:
        raise NotImplementedError

    def get_session(self, device_id: UUID) -> DeviceSessionRecord | None:
        raise NotImplementedError

    def list_sessions_for_user(self, user_id: UUID) -> list[DeviceSessionRecord]:
        raise NotImplementedError

    async def dispatch_request(
        self,
        session: DeviceSessionRecord,
        request: ToolDispatchRequest,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def get_next_request(
        self,
        device_id: UUID,
        *,
        timeout_seconds: int = 1,
    ) -> ToolDispatchRequest | None:
        raise NotImplementedError

    async def publish_result(self, result: ToolDispatchResult) -> None:
        raise NotImplementedError

    async def fail_pending_requests(self, device_id: UUID, reason: str) -> None:
        raise NotImplementedError

    async def cleanup_stale_sessions(self) -> int:
        return 0

    async def close(self) -> None:
        return None


class InMemoryClientRuntimeStore(BaseClientRuntimeStore):
    """Single-process fallback store for development and tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._issued_session_ids: dict[str, tuple[str, float]] = {}
        self._sessions: dict[str, DeviceSessionRecord] = {}
        self._user_devices: dict[str, set[str]] = {}
        self._request_queues: dict[str, asyncio.Queue[ToolDispatchRequest]] = {}
        self._result_futures: dict[str, asyncio.Future[ToolDispatchResult]] = {}
        self._request_devices: dict[str, str] = {}

    @staticmethod
    def _device_key(device_id: UUID) -> str:
        return str(device_id)

    def _get_queue(self, device_id: UUID) -> asyncio.Queue[ToolDispatchRequest]:
        key = self._device_key(device_id)
        with self._lock:
            queue = self._request_queues.get(key)
            if queue is None:
                queue = asyncio.Queue()
                self._request_queues[key] = queue
            return queue

    async def issue_runtime_session_id(self, device_id: UUID, session_id: str) -> None:
        expires_at = time.monotonic() + _issued_session_ttl_seconds()
        with self._lock:
            self._issued_session_ids[self._device_key(device_id)] = (session_id, expires_at)

    async def consume_runtime_session_id(self, device_id: UUID, session_id: str) -> bool:
        with self._lock:
            stored = self._issued_session_ids.pop(self._device_key(device_id), None)
        if stored is None:
            return False
        expected, expires_at = stored
        if expires_at < time.monotonic():
            return False
        return expected == session_id

    async def put_session(self, session: DeviceSessionRecord) -> None:
        key = self._device_key(session.device_id)
        with self._lock:
            self._sessions[key] = session
            self._user_devices.setdefault(str(session.user_id), set()).add(key)
            self._request_queues.setdefault(key, asyncio.Queue())

    async def delete_session(self, device_id: UUID) -> DeviceSessionRecord | None:
        key = self._device_key(device_id)
        with self._lock:
            session = self._sessions.pop(key, None)
            self._issued_session_ids.pop(key, None)
            self._request_queues.pop(key, None)
            stale_request_ids = [
                request_id
                for request_id, bound_device_key in self._request_devices.items()
                if bound_device_key == key
            ]
            for request_id in stale_request_ids:
                self._request_devices.pop(request_id, None)
                self._result_futures.pop(request_id, None)
            if session is not None:
                devices = self._user_devices.get(str(session.user_id))
                if devices is not None:
                    devices.discard(key)
                    if not devices:
                        self._user_devices.pop(str(session.user_id), None)
        return session

    async def update_heartbeat(self, device_id: UUID) -> DeviceSessionRecord | None:
        session = self.get_session(device_id)
        if session is None:
            return None
        session.update_heartbeat()
        return session

    async def update_tool_catalog(
        self,
        device_id: UUID,
        catalog: dict[str, Any],
    ) -> DeviceSessionRecord | None:
        session = self.get_session(device_id)
        if session is None:
            return None
        session.update_tool_catalog(catalog)
        return session

    async def update_skill_catalog(
        self,
        device_id: UUID,
        catalog: dict[str, Any],
    ) -> DeviceSessionRecord | None:
        session = self.get_session(device_id)
        if session is None:
            return None
        session.update_skill_catalog(catalog)
        return session

    def get_session(self, device_id: UUID) -> DeviceSessionRecord | None:
        key = self._device_key(device_id)
        with self._lock:
            session = self._sessions.get(key)
        if session is None:
            return None
        return session if session.is_alive() else None

    def list_sessions_for_user(self, user_id: UUID) -> list[DeviceSessionRecord]:
        user_key = str(user_id)
        with self._lock:
            device_keys = list(self._user_devices.get(user_key, set()))
        sessions: list[DeviceSessionRecord] = []
        for device_key in device_keys:
            try:
                device_id = UUID(device_key)
            except Exception:
                continue
            session = self.get_session(device_id)
            if session is None:
                with self._lock:
                    devices = self._user_devices.get(user_key)
                    if devices is not None:
                        devices.discard(device_key)
                continue
            sessions.append(session)
        return sessions

    async def dispatch_request(
        self,
        session: DeviceSessionRecord,
        request: ToolDispatchRequest,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ToolDispatchResult] = loop.create_future()
        device_key = self._device_key(session.device_id)

        with self._lock:
            self._result_futures[request.request_id] = future
            self._request_devices[request.request_id] = device_key

        await self._get_queue(session.device_id).put(request)

        try:
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
        finally:
            with self._lock:
                self._result_futures.pop(request.request_id, None)
                self._request_devices.pop(request.request_id, None)

        return result.model_dump(mode="json")

    async def get_next_request(
        self,
        device_id: UUID,
        *,
        timeout_seconds: int = 1,
    ) -> ToolDispatchRequest | None:
        queue = self._get_queue(device_id)
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None

    async def publish_result(self, result: ToolDispatchResult) -> None:
        with self._lock:
            future = self._result_futures.get(result.request_id)
        if future is not None and not future.done():
            future.set_result(result)

    async def fail_pending_requests(self, device_id: UUID, reason: str) -> None:
        device_key = self._device_key(device_id)
        error_result = RuntimeErrorContext(message=reason, code="DEVICE_DISCONNECTED")

        with self._lock:
            request_ids = [
                request_id
                for request_id, bound_device_key in self._request_devices.items()
                if bound_device_key == device_key
            ]
            futures = {
                request_id: self._result_futures.get(request_id) for request_id in request_ids
            }

        for request_id, future in futures.items():
            if future is None or future.done():
                continue
            future.set_result(
                ToolDispatchResult(
                    request_id=request_id,
                    success=False,
                    error=reason,
                    error_context=error_result,
                    execution_time_ms=0,
                )
            )

        queue = self._get_queue(device_id)
        while True:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self.publish_result(
                ToolDispatchResult(
                    request_id=request.request_id,
                    success=False,
                    error=reason,
                    error_context=error_result,
                    execution_time_ms=0,
                )
            )

    async def cleanup_stale_sessions(self) -> int:
        with self._lock:
            stale_device_keys = [
                device_key
                for device_key, session in self._sessions.items()
                if not session.is_alive()
            ]

        cleaned = 0
        for device_key in stale_device_keys:
            try:
                device_id = UUID(device_key)
            except Exception:
                continue
            await self.fail_pending_requests(device_id, "Client runtime heartbeat expired.")
            await self.delete_session(device_id)
            cleaned += 1
        return cleaned


class RedisClientRuntimeStore(BaseClientRuntimeStore):
    """Redis-backed store for multi-worker / multi-instance runtime coordination."""

    def __init__(self, redis_url: str) -> None:
        if SyncRedis is None or AsyncRedis is None:
            raise RuntimeError("The 'redis' package is required for the Redis runtime store.")

        self._redis_url = redis_url
        self._sync = SyncRedis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
            health_check_interval=30,
        )
        self._sync.ping()
        self._async = AsyncRedis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
            health_check_interval=30,
        )

    @staticmethod
    def _session_key(device_id: UUID) -> str:
        return f"{_SESSION_KEY_PREFIX}:{device_id}"

    @staticmethod
    def _user_devices_key(user_id: UUID) -> str:
        return f"{_USER_DEVICES_KEY_PREFIX}:{user_id}"

    @staticmethod
    def _issued_session_key(device_id: UUID) -> str:
        return f"{_ISSUED_SESSION_KEY_PREFIX}:{device_id}"

    @staticmethod
    def _request_queue_key(device_id: UUID) -> str:
        return f"{_REQUEST_QUEUE_KEY_PREFIX}:{device_id}"

    @staticmethod
    def _result_queue_key(request_id: str) -> str:
        return f"{_RESULT_QUEUE_KEY_PREFIX}:{request_id}"

    @staticmethod
    def _pending_requests_key(device_id: UUID) -> str:
        return f"{_PENDING_REQUESTS_KEY_PREFIX}:{device_id}"

    @staticmethod
    def _pending_requests_key_for_value(device_id: str) -> str:
        return f"{_PENDING_REQUESTS_KEY_PREFIX}:{device_id}"

    @staticmethod
    def _request_device_key(request_id: str) -> str:
        return f"{_REQUEST_DEVICE_KEY_PREFIX}:{request_id}"

    async def issue_runtime_session_id(self, device_id: UUID, session_id: str) -> None:
        await self._async.set(
            self._issued_session_key(device_id),
            session_id,
            ex=_issued_session_ttl_seconds(),
        )

    async def consume_runtime_session_id(self, device_id: UUID, session_id: str) -> bool:
        key = self._issued_session_key(device_id)
        try:
            stored = await self._async.getdel(key)
        except AttributeError:
            pipeline = self._async.pipeline()
            await pipeline.get(key)
            await pipeline.delete(key)
            stored, _ = await pipeline.execute()
        return str(stored or "") == session_id

    async def put_session(self, session: DeviceSessionRecord) -> None:
        payload = json.dumps(session.to_payload(), default=str)
        await self._async.set(
            self._session_key(session.device_id),
            payload,
            ex=_session_ttl_seconds(),
        )
        await self._async.sadd(self._user_devices_key(session.user_id), str(session.device_id))

    async def delete_session(self, device_id: UUID) -> DeviceSessionRecord | None:
        session = await self._get_session_async_raw(device_id)
        pending_request_ids = await self._async.smembers(self._pending_requests_key(device_id))
        keys_to_delete = [
            self._session_key(device_id),
            self._issued_session_key(device_id),
            self._request_queue_key(device_id),
            self._pending_requests_key(device_id),
        ]
        keys_to_delete.extend(
            self._request_device_key(request_id) for request_id in pending_request_ids
        )
        await self._async.delete(*keys_to_delete)
        if session is not None:
            await self._async.srem(self._user_devices_key(session.user_id), str(device_id))
        return session

    async def update_heartbeat(self, device_id: UUID) -> DeviceSessionRecord | None:
        session = await self._get_session_async(device_id)
        if session is None:
            return None
        session.update_heartbeat()
        await self.put_session(session)
        return session

    async def update_tool_catalog(
        self,
        device_id: UUID,
        catalog: dict[str, Any],
    ) -> DeviceSessionRecord | None:
        session = await self._get_session_async(device_id)
        if session is None:
            return None
        session.update_tool_catalog(catalog)
        await self.put_session(session)
        return session

    async def update_skill_catalog(
        self,
        device_id: UUID,
        catalog: dict[str, Any],
    ) -> DeviceSessionRecord | None:
        session = await self._get_session_async(device_id)
        if session is None:
            return None
        session.update_skill_catalog(catalog)
        await self.put_session(session)
        return session

    def get_session(self, device_id: UUID) -> DeviceSessionRecord | None:
        session = self._get_session_sync_raw(device_id)
        if session is None:
            return None
        if session.is_alive():
            return session
        self._sync_expire_stale_session(session)
        return None

    def list_sessions_for_user(self, user_id: UUID) -> list[DeviceSessionRecord]:
        device_ids = self._sync.smembers(self._user_devices_key(user_id))
        sessions: list[DeviceSessionRecord] = []
        for raw_device_id in device_ids:
            try:
                device_id = UUID(str(raw_device_id))
            except Exception:
                continue
            session = self.get_session(device_id)
            if session is None:
                self._sync.srem(self._user_devices_key(user_id), str(raw_device_id))
                continue
            sessions.append(session)
        return sessions

    async def dispatch_request(
        self,
        session: DeviceSessionRecord,
        request: ToolDispatchRequest,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        tracking_ttl = _request_tracking_ttl_seconds(timeout_seconds)
        pipeline = self._async.pipeline()
        pipeline.sadd(self._pending_requests_key(session.device_id), request.request_id)
        pipeline.expire(self._pending_requests_key(session.device_id), tracking_ttl)
        pipeline.set(
            self._request_device_key(request.request_id),
            str(session.device_id),
            ex=tracking_ttl,
        )
        pipeline.rpush(
            self._request_queue_key(session.device_id),
            request.model_dump_json(),
        )
        await pipeline.execute()

        try:
            item = await self._async.blpop(
                self._result_queue_key(request.request_id),
                timeout=timeout_seconds,
            )
            if item is None:
                raise TimeoutError(
                    f"Timed out waiting for client runtime result after {timeout_seconds}s"
                )
            _, payload = item
            parsed = ToolDispatchResult.model_validate_json(payload)
            return parsed.model_dump(mode="json")
        finally:
            await self._clear_pending_request(request.request_id)

    async def get_next_request(
        self,
        device_id: UUID,
        *,
        timeout_seconds: int = 1,
    ) -> ToolDispatchRequest | None:
        item = await self._async.blpop(
            self._request_queue_key(device_id),
            timeout=timeout_seconds,
        )
        if item is None:
            return None
        _, payload = item
        return ToolDispatchRequest.model_validate_json(payload)

    async def publish_result(self, result: ToolDispatchResult) -> None:
        key = self._result_queue_key(result.request_id)
        request_device_key = self._request_device_key(result.request_id)
        device_id_value = await self._async.get(request_device_key)

        pipeline = self._async.pipeline()
        pipeline.rpush(key, result.model_dump_json())
        pipeline.expire(key, _RESULT_TTL_SECONDS)
        if device_id_value:
            pipeline.srem(
                self._pending_requests_key_for_value(str(device_id_value)),
                result.request_id,
            )
        pipeline.delete(request_device_key)
        await pipeline.execute()

    async def fail_pending_requests(self, device_id: UUID, reason: str) -> None:
        request_ids = await self._async.smembers(self._pending_requests_key(device_id))
        for request_id in request_ids:
            await self.publish_result(_disconnect_result(str(request_id), reason))
        await self._async.delete(self._pending_requests_key(device_id))

    async def cleanup_stale_sessions(self) -> int:
        cleaned = 0
        async for session_key in self._async.scan_iter(match=f"{_SESSION_KEY_PREFIX}:*"):
            payload = await self._async.get(session_key)
            session = _decode_session_payload(payload)
            if session is None or session.is_alive():
                continue
            await self.fail_pending_requests(session.device_id, "Client runtime heartbeat expired.")
            await self.delete_session(session.device_id)
            cleaned += 1
        return cleaned

    async def close(self) -> None:
        await self._async.aclose()
        self._sync.close()

    async def _get_session_async(self, device_id: UUID) -> DeviceSessionRecord | None:
        session = await self._get_session_async_raw(device_id)
        if session is None:
            return None
        if session.is_alive():
            return session
        await self.fail_pending_requests(device_id, "Client runtime heartbeat expired.")
        await self.delete_session(device_id)
        return None

    def _get_session_sync_raw(self, device_id: UUID) -> DeviceSessionRecord | None:
        return _decode_session_payload(self._sync.get(self._session_key(device_id)))

    async def _get_session_async_raw(self, device_id: UUID) -> DeviceSessionRecord | None:
        return _decode_session_payload(await self._async.get(self._session_key(device_id)))

    async def _clear_pending_request(self, request_id: str) -> None:
        request_device_key = self._request_device_key(request_id)
        device_id_value = await self._async.get(request_device_key)

        pipeline = self._async.pipeline()
        if device_id_value:
            pipeline.srem(
                self._pending_requests_key_for_value(str(device_id_value)),
                request_id,
            )
        pipeline.delete(request_device_key, self._result_queue_key(request_id))
        await pipeline.execute()

    def _sync_expire_stale_session(self, session: DeviceSessionRecord) -> None:
        self._sync_fail_pending_requests(
            session.device_id,
            "Client runtime heartbeat expired.",
        )
        self._sync_delete_session(session)

    def _sync_fail_pending_requests(self, device_id: UUID, reason: str) -> None:
        request_ids = self._sync.smembers(self._pending_requests_key(device_id))
        if not request_ids:
            return

        pipeline = self._sync.pipeline()
        for request_id in request_ids:
            result = _disconnect_result(str(request_id), reason)
            pipeline.rpush(self._result_queue_key(str(request_id)), result.model_dump_json())
            pipeline.expire(self._result_queue_key(str(request_id)), _RESULT_TTL_SECONDS)
            pipeline.delete(self._request_device_key(str(request_id)))
        pipeline.delete(self._pending_requests_key(device_id))
        pipeline.execute()

    def _sync_delete_session(self, session: DeviceSessionRecord) -> None:
        self._sync.delete(
            self._session_key(session.device_id),
            self._issued_session_key(session.device_id),
            self._request_queue_key(session.device_id),
            self._pending_requests_key(session.device_id),
        )
        self._sync.srem(self._user_devices_key(session.user_id), str(session.device_id))


_store: BaseClientRuntimeStore | None = None
_store_lock = threading.Lock()


def get_client_runtime_store() -> BaseClientRuntimeStore:
    """Return the shared runtime store, falling back to in-memory if Redis is unavailable."""
    global _store
    if _store is not None:
        return _store

    with _store_lock:
        if _store is not None:
            return _store

        redis_url = _runtime_redis_url()
        if redis_url:
            try:
                _store = RedisClientRuntimeStore(redis_url)
                logger.info("Client runtime store initialized with Redis backend")
                return _store
            except Exception as exc:
                logger.warning(
                    "Client runtime store falling back to in-memory backend because Redis "
                    "is unavailable: %s",
                    exc,
                )

        _store = InMemoryClientRuntimeStore()
        logger.info("Client runtime store initialized with in-memory backend")
        return _store


async def close_client_runtime_store() -> None:
    """Close the shared runtime store backend if needed."""
    global _store
    if _store is None:
        return
    await _store.close()
    _store = None


def reset_client_runtime_store() -> None:
    """Testing hook to reset the shared runtime store singleton."""
    global _store
    _store = None
