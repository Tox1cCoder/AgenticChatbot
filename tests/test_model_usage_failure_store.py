from __future__ import annotations

from datetime import datetime, timezone

from app.observability.model_usage import ModelUsageHealthService, ModelUsageMetrics
from app.observability.model_usage_failure_store import (
    FailureStoreSnapshot,
    RedisModelUsageFailureStore,
)


class SharedFailureStore:
    def __init__(self, *, available: bool = True, fail_writes: bool = False):
        self.count = 0
        self.available = available
        self.fail_writes = fail_writes

    def record_failure(self) -> bool:
        if self.fail_writes:
            raise ConnectionError("must be swallowed")
        self.count += 1
        return self.available

    def recent_failure_count(self, *, window_seconds: float) -> FailureStoreSnapshot:
        return FailureStoreSnapshot(count=self.count, available=self.available)


class SnapshotRepository:
    def get_model_usage_health_snapshot(self, *, now, lookback_minutes):
        minute = datetime(2026, 7, 21, 12, 29, tzinfo=timezone.utc)
        return {
            "raw_event_count": 1,
            "rollup_request_count": 1,
            "unattributed_event_count": 0,
            "latest_event_minute": minute,
            "latest_rollup_minute": minute,
        }


def health(metrics):
    return ModelUsageHealthService(
        SnapshotRepository(),
        metrics=metrics,
        clock=lambda: datetime(2026, 7, 21, 12, 31, tzinfo=timezone.utc),
    ).get_health()


def test_failure_recorded_by_one_instance_degrades_another_instance():
    store = SharedFailureStore()
    writer = ModelUsageMetrics(failure_store=store)
    reader = ModelUsageMetrics(failure_store=store)

    writer.record_persistence("retry_enqueued", failure_class="ConnectionError")
    result = health(reader)

    assert result["status"] == "degraded"
    assert result["persistence_failure_count"] == 1
    assert result["persistence_failure_scope"] == "deployment"
    assert result["persistence_failure_store_available"] is True


def test_unavailable_shared_store_never_claims_global_health():
    result = health(ModelUsageMetrics(failure_store=SharedFailureStore(available=False)))

    assert result["status"] == "degraded"
    assert result["persistence_failure_scope"] == "unavailable"
    assert result["persistence_failure_store_available"] is False


def test_shared_store_failure_is_best_effort_and_never_escapes_recorder_metric():
    metrics = ModelUsageMetrics(failure_store=SharedFailureStore(fail_writes=True))

    metrics.record_persistence("dropped", failure_class="OperationalError")

    assert metrics.recent_persistence_failure_count() == 1


def test_redis_store_uses_content_free_minute_bucket_and_ttl():
    values = {}
    transaction_modes = []

    class Pipeline:
        def incr(self, key):
            values[key] = values.get(key, 0) + 1

        def expire(self, key, ttl):
            values[f"ttl:{key}"] = ttl

        def execute(self):
            return []

    class Redis:
        def pipeline(self, *, transaction):
            transaction_modes.append(transaction)
            return Pipeline()

        def mget(self, keys):
            return [values.get(key) for key in keys]

    now = datetime(2026, 7, 21, 12, 34, tzinfo=timezone.utc)
    store = RedisModelUsageFailureStore(Redis(), ttl_seconds=900, clock=lambda: now)

    assert store.record_failure() is True
    result = store.recent_failure_count(window_seconds=300)

    assert result == FailureStoreSnapshot(count=1, available=True)
    assert transaction_modes == [True]
    bucket_keys = [key for key in values if not key.startswith("ttl:")]
    assert bucket_keys == [f"model_usage:persistence_failures:v1:{int(now.timestamp() // 60)}"]
    assert values[f"ttl:{bucket_keys[0]}"] == 900


def test_redis_store_caps_direct_large_window_reads_and_marks_them_incomplete():
    captured_keys = []

    class Redis:
        def mget(self, keys):
            captured_keys.extend(keys)
            return [0] * len(keys)

    store = RedisModelUsageFailureStore(
        Redis(),
        ttl_seconds=86_400,
        clock=lambda: datetime(2026, 7, 21, 12, 34, tzinfo=timezone.utc),
    )

    result = store.recent_failure_count(window_seconds=10_000_000)

    assert len(captured_keys) == 61
    assert result.available is False
