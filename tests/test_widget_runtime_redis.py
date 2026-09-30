"""The Redis widget store never hands two writes the same version number.

Every write reads the widget, bumps its version and writes it back. Without
WATCH, a write that lands between another write's read and its write-back is
overwritten with the same version number, so clients that sync by version miss
a change. Runs against a real Redis; skips without one.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import redis

from app.services.widget_runtime import (
    _WATCH_RETRY_LIMIT,
    RedisWidgetStore,
    WidgetStatus,
)


def _redis_url() -> str:
    url = os.getenv("TEST_REDIS_URL") or ""
    if not url.strip():
        from app.core.config import settings

        url = str(getattr(settings, "redis_url", "") or "")
    if not url.strip():
        pytest.skip("no Redis URL available")
    return url


@pytest.fixture
def stores() -> Iterator[tuple[RedisWidgetStore, redis.Redis]]:
    url = _redis_url()
    try:
        racer = redis.from_url(url, decode_responses=True)
        racer.ping()
    except Exception as exc:  # noqa: BLE001 - any connection failure means skip
        pytest.skip(f"Redis unavailable: {exc}")
    store = RedisWidgetStore(url)
    created_keys: list[str] = []
    store.created_keys = created_keys
    try:
        yield store, racer
    finally:
        if created_keys:
            racer.delete(*created_keys)
        racer.close()
        store._redis.close()


async def _widget(store: RedisWidgetStore) -> tuple[str, str]:
    record = await store.create("session-race", {"count": 0})
    key = store._key(record.widget_id)
    store.created_keys.extend([key, store._session_key("session-race")])
    return record.widget_id, key


def _another_write_lands_after_each_read(monkeypatch, key: str, racer: redis.Redis, times: int):
    """Commit a competing version bump right after the store reads ``key``."""
    original = redis.Redis.hgetall
    remaining = {"writes": times}

    def hgetall(self, name):
        data = original(self, name)
        if name == key and remaining["writes"] > 0:
            remaining["writes"] -= 1
            racer.hincrby(key, "version", 1)
        return data

    monkeypatch.setattr(redis.Redis, "hgetall", hgetall)


async def test_close_racing_another_write_does_not_reuse_its_version(stores, monkeypatch):
    store, racer = stores
    widget_id, key = await _widget(store)
    _another_write_lands_after_each_read(monkeypatch, key, racer, times=1)

    closed = await store.close(widget_id)
    # Read into a local first: a failing assert would otherwise print the
    # client's repr, which includes the Redis password.
    stored_version = int(racer.hget(key, "version"))

    # create -> 1, the competing write -> 2, the close -> 3.
    assert stored_version == 3
    assert closed.version == 3
    assert closed.status == WidgetStatus.CLOSED


@pytest.mark.parametrize("operation", ["close", "update", "patch"])
async def test_a_write_that_keeps_losing_the_race_gives_up(stores, monkeypatch, operation):
    store, racer = stores
    widget_id, key = await _widget(store)
    _another_write_lands_after_each_read(monkeypatch, key, racer, times=_WATCH_RETRY_LIMIT + 5)
    write = {
        "close": lambda: store.close(widget_id),
        "update": lambda: store.update(widget_id, {"count": 1}),
        "patch": lambda: store.patch(widget_id, {"count": 1}),
    }[operation]

    with pytest.raises(ValueError, match="kept changing"):
        await write()
    stored_status = racer.hget(key, "status")

    assert stored_status == WidgetStatus.ACTIVE.value
