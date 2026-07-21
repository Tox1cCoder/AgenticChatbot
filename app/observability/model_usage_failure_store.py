"""Content-free shared persistence-failure counters for model-usage health."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True)
class FailureStoreSnapshot:
    count: int
    available: bool


class ModelUsageFailureStore(Protocol):
    def record_failure(self) -> bool: ...

    def recent_failure_count(self, *, window_seconds: float) -> FailureStoreSnapshot: ...


class UnavailableModelUsageFailureStore:
    """Explicit unavailable implementation; never claims deployment health."""

    def record_failure(self) -> bool:
        return False

    def recent_failure_count(self, *, window_seconds: float) -> FailureStoreSnapshot:
        return FailureStoreSnapshot(count=0, available=False)


class RedisModelUsageFailureStore:
    """Redis UTC-minute counters with fixed content-free keys and bounded TTL."""

    _KEY_PREFIX = "model_usage:persistence_failures:v1"
    _MAX_READ_BUCKETS = 61

    def __init__(
        self,
        client: Any,
        *,
        ttl_seconds: int,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if ttl_seconds < 60:
            raise ValueError("ttl_seconds must be at least 60")
        self._client = client
        self._ttl_seconds = int(ttl_seconds)
        self._clock = clock

    def _minute_number(self) -> int:
        return int(self._clock().astimezone(timezone.utc).timestamp() // 60)

    @classmethod
    def _key(cls, minute_number: int) -> str:
        return f"{cls._KEY_PREFIX}:{minute_number}"

    def record_failure(self) -> bool:
        try:
            key = self._key(self._minute_number())
            pipeline = self._client.pipeline(transaction=True)
            pipeline.incr(key)
            pipeline.expire(key, self._ttl_seconds)
            pipeline.execute()
            return True
        except Exception:
            return False

    def recent_failure_count(self, *, window_seconds: float) -> FailureStoreSnapshot:
        window = max(1.0, float(window_seconds))
        max_buckets = max(1, math.ceil(self._ttl_seconds / 60))
        requested_buckets = min(max_buckets, math.ceil(window / 60) + 1)
        bucket_count = min(requested_buckets, self._MAX_READ_BUCKETS)
        complete = requested_buckets <= self._MAX_READ_BUCKETS
        current = self._minute_number()
        keys = [self._key(current - offset) for offset in range(bucket_count)]
        try:
            values = self._client.mget(keys)
            return FailureStoreSnapshot(
                count=sum(int(value or 0) for value in values),
                available=complete,
            )
        except Exception:
            return FailureStoreSnapshot(count=0, available=False)


def create_redis_failure_store(
    redis_url: str,
    *,
    ttl_seconds: int,
    timeout_seconds: float,
) -> ModelUsageFailureStore:
    """Build a bounded-time Redis store without opening a connection eagerly."""
    if not (redis_url or "").strip():
        return UnavailableModelUsageFailureStore()
    try:
        import redis

        client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=max(0.05, float(timeout_seconds)),
            socket_timeout=max(0.05, float(timeout_seconds)),
        )
        return RedisModelUsageFailureStore(client, ttl_seconds=ttl_seconds)
    except Exception:
        return UnavailableModelUsageFailureStore()
