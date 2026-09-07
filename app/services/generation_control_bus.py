"""Deliver a Stop decision to whichever worker owns the generation.

The HTTP request asking for a Stop rarely lands on the process streaming the
answer. So the durable transition in ``generations`` is the decision, and this
bus is only how the owning worker finds out — which is why a publish with no
subscriber is success, not an error, and why nothing here waits for an
acknowledgement. A worker that never hears the signal still finds
``stop_requested`` on its next durable check.

The payload is a schema version, a generation id and a lifecycle version.
Nothing else: a subscriber that needs more reads the row it already has
permission to read. Keeping user content off this channel means a misconfigured
Redis cannot become a data leak.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

logger = logging.getLogger(__name__)

__all__ = [
    "GenerationControlBus",
    "build_generation_control_bus",
    "InMemoryGenerationControlBus",
    "RedisGenerationControlBus",
    "StopSignal",
    "StopHandler",
    "decode_stop_signal",
    "encode_stop_signal",
]

#: Bumped only for an incompatible payload change. A subscriber that does not
#: recognise the version drops the message rather than guessing at its fields.
STOP_SIGNAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StopSignal:
    """A request to stop one generation, fenced by the version it was seen at."""

    generation_id: UUID
    version: int


StopHandler = Callable[[StopSignal], Awaitable[None]]


class GenerationControlBus(Protocol):
    """How a Stop reaches the worker holding the generation."""

    async def publish_stop(self, generation_id: UUID, version: int) -> None: ...

    async def subscribe(self, handler: StopHandler) -> None: ...

    async def close(self) -> None: ...


def encode_stop_signal(signal: StopSignal) -> str:
    return json.dumps(
        {
            "schema_version": STOP_SIGNAL_SCHEMA_VERSION,
            "generation_id": str(signal.generation_id),
            "version": int(signal.version),
        }
    )


def decode_stop_signal(payload: Any) -> StopSignal | None:
    """Parse one message, or ``None`` for anything unusable.

    Never raises. The caller is a subscriber loop, and that loop is how every
    future Stop arrives — letting one malformed message escape would disable
    cancellation for the whole process until it restarts.
    """
    try:
        if isinstance(payload, bytes | bytearray):
            payload = payload.decode("utf-8")
        document = json.loads(payload)
        if not isinstance(document, dict):
            return None
        if document.get("schema_version") != STOP_SIGNAL_SCHEMA_VERSION:
            return None
        version = int(document["version"])
        if version < 1:
            return None
        return StopSignal(generation_id=UUID(str(document["generation_id"])), version=version)
    except Exception:
        logger.debug("Dropped an unusable generation stop signal")
        return None


class InMemoryGenerationControlBus:
    """Single-process bus. Deterministic, and the only one tests need.

    Delivery is awaited rather than dispatched to a task, so a test that
    publishes has observed every handler by the time the call returns. That is
    also honest about the single-process case: there is no network to wait for.
    """

    def __init__(self) -> None:
        self._handlers: list[StopHandler] = []
        self._closed = False

    async def publish_stop(self, generation_id: UUID, version: int) -> None:
        if self._closed:
            return
        signal = StopSignal(generation_id=generation_id, version=int(version))
        for handler in list(self._handlers):
            await _deliver(handler, signal)

    async def subscribe(self, handler: StopHandler) -> None:
        if not self._closed:
            self._handlers.append(handler)

    async def close(self) -> None:
        self._closed = True
        self._handlers.clear()


class RedisGenerationControlBus:
    """Cross-process bus over one Redis pub/sub channel.

    The connection is opened lazily on first use rather than at construction:
    the container builds this during startup, and a Redis that is briefly
    unreachable then must not prevent the application from serving requests
    that never touch Stop.
    """

    def __init__(
        self,
        *,
        redis_url: str,
        channel: str,
        reconnect_backoff_seconds: float = 1.0,
        max_reconnect_backoff_seconds: float = 30.0,
    ) -> None:
        self._redis_url = str(redis_url or "")
        self._channel = str(channel or "generation:stop")
        self._backoff = max(0.05, float(reconnect_backoff_seconds))
        self._max_backoff = max(self._backoff, float(max_reconnect_backoff_seconds))
        self._client: Any | None = None
        self._listener: asyncio.Task[None] | None = None
        self._handlers: list[StopHandler] = []
        self._closed = False

    async def publish_stop(self, generation_id: UUID, version: int) -> None:
        """Best effort. The durable transition already happened."""
        if self._closed:
            return
        client = await self._ensure_client()
        if client is None:
            return
        signal = StopSignal(generation_id=generation_id, version=int(version))
        try:
            await client.publish(self._channel, encode_stop_signal(signal))
        except Exception as exc:
            # Not raised to the caller: Stop is already durable, and failing the
            # HTTP request would tell the user nothing happened when it did.
            logger.warning("Could not publish a stop signal: %s", type(exc).__name__)

    async def subscribe(self, handler: StopHandler) -> None:
        if self._closed:
            return
        self._handlers.append(handler)
        if self._listener is None:
            self._listener = asyncio.create_task(self._listen_forever())

    async def close(self) -> None:
        self._closed = True
        self._handlers.clear()
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.cancel()
            with _suppress_cancelled():
                await listener
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                logger.debug("Redis control bus close was not clean")

    async def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        if not self._redis_url.strip():
            return None
        try:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self._redis_url, decode_responses=True)
        except Exception as exc:
            logger.warning("Redis control bus unavailable: %s", type(exc).__name__)
            return None
        return self._client

    async def _listen_forever(self) -> None:
        """Re-subscribe with backoff. A dropped connection must not end Stop."""
        delay = self._backoff
        while not self._closed:
            try:
                await self._listen_once()
                delay = self._backoff
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Generation stop subscriber reconnecting after %s", type(exc).__name__
                )
            if self._closed:
                return
            await asyncio.sleep(delay)
            delay = min(self._max_backoff, delay * 2)

    async def _listen_once(self) -> None:
        client = await self._ensure_client()
        if client is None:
            raise RuntimeError("redis client unavailable")
        pubsub = client.pubsub()
        await pubsub.subscribe(self._channel)
        try:
            async for message in pubsub.listen():
                if self._closed:
                    return
                if (message or {}).get("type") != "message":
                    continue
                signal = decode_stop_signal(message.get("data"))
                if signal is None:
                    continue
                for handler in list(self._handlers):
                    await _deliver(handler, signal)
        finally:
            with _suppress_exceptions():
                await pubsub.aclose()


async def _deliver(handler: StopHandler, signal: StopSignal) -> None:
    """One handler's failure must not silence the others."""
    try:
        await handler(signal)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "A generation stop handler failed for %s: %s", signal.generation_id, type(exc).__name__
        )


def _suppress_cancelled():
    from contextlib import suppress

    return suppress(asyncio.CancelledError)


def _suppress_exceptions():
    from contextlib import suppress

    return suppress(Exception)


def build_generation_control_bus(
    *,
    redis_url: str | None,
    channel: str,
    reconnect_backoff_seconds: float,
    max_reconnect_backoff_seconds: float,
) -> GenerationControlBus:
    """Pick a bus from configuration.

    Without a Redis URL there is only one process worth coordinating, so the
    in-memory bus is the honest choice rather than a degraded Redis one. A
    single-worker deployment then still gets immediate local cancellation.
    """
    if not str(redis_url or "").strip():
        logger.info("Generation stop signals stay in-process: no redis_url configured")
        return InMemoryGenerationControlBus()
    return RedisGenerationControlBus(
        redis_url=str(redis_url),
        channel=channel,
        reconnect_backoff_seconds=reconnect_backoff_seconds,
        max_reconnect_backoff_seconds=max_reconnect_backoff_seconds,
    )
