"""A Stop crossing the Redis hop between two independent workers.

This is the one part of "Stop is durable, distributed, idempotent" that was
verified by construction rather than by experiment. Every other bus test uses
``InMemoryGenerationControlBus``, which awaits its handlers in-process and so
proves nothing about the network.

The buses here each open their **own** Redis connection and never share
anything in Python. Redis is indifferent to whether the two ends are in one
process or two: what it delivers between two separate connections on a channel
is what it delivers between two workers. Sharing one bus, or one client, would
demonstrate only that a list dispatches to itself.

Why the distributed hop is a latency optimisation and not the correctness
mechanism, asserted here too: the durable transition happens *before* the
publication, so a subscriber that is down when the signal is sent still finds
`stop_requested` on the row. That ordering is why `publish_stop` swallows its
own failures instead of failing the HTTP request.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from app.services.generation_control_bus import (
    STOP_SIGNAL_SCHEMA_VERSION,
    RedisGenerationControlBus,
    StopSignal,
    decode_stop_signal,
    encode_stop_signal,
)

pytestmark = pytest.mark.selector_event_loop

_DELIVERY_TIMEOUT_SECONDS = 5.0


def _redis_url() -> str:
    url = os.getenv("TEST_REDIS_URL") or ""
    if not url.strip():
        from app.core.config import settings

        url = str(getattr(settings, "redis_url", "") or "")
    if not url.strip():
        pytest.skip("no Redis URL available for the control-bus integration test")
    return url


@pytest.fixture
def channel() -> str:
    """A channel per test, so one test's signal cannot reach another's."""
    return f"test:generation:stop:{uuid.uuid4().hex}"


def _bus(channel: str) -> RedisGenerationControlBus:
    return RedisGenerationControlBus(
        redis_url=_redis_url(),
        channel=channel,
        reconnect_backoff_seconds=0.05,
        max_reconnect_backoff_seconds=0.2,
    )


async def _await_signal(received: asyncio.Queue) -> StopSignal:
    return await asyncio.wait_for(received.get(), timeout=_DELIVERY_TIMEOUT_SECONDS)


async def _subscribed(bus: RedisGenerationControlBus) -> asyncio.Queue:
    """Subscribe, and wait until the listener is actually attached.

    Redis pub/sub has no backlog: a message published before the subscription
    is established is simply gone. Without this wait the tests would be racing
    the listener task and would flake in exactly the direction that hides a
    real failure.
    """
    received: asyncio.Queue = asyncio.Queue()

    async def handler(signal: StopSignal) -> None:
        await received.put(signal)

    await bus.subscribe(handler)
    for _ in range(100):
        if getattr(bus, "_client", None) is not None:  # noqa: SLF001 - readiness probe
            break
        await asyncio.sleep(0.02)
    # The client existing is necessary but not sufficient; give the SUBSCRIBE
    # itself a moment to land on the server.
    await asyncio.sleep(0.15)
    return received


# ----------------------------------------------------------------------
# the hop itself
# ----------------------------------------------------------------------


async def test_a_stop_published_by_one_worker_reaches_another(channel: str) -> None:
    """The claim that was previously unexercised."""
    publisher, subscriber = _bus(channel), _bus(channel)
    generation_id = uuid.uuid4()
    try:
        received = await _subscribed(subscriber)

        await publisher.publish_stop(generation_id, 7)
        signal = await _await_signal(received)
    finally:
        await asyncio.gather(publisher.close(), subscriber.close())

    assert signal.generation_id == generation_id
    assert signal.version == 7


async def test_the_publisher_and_subscriber_share_no_python_state(channel: str) -> None:
    """Otherwise this would be testing a list, not Redis."""
    publisher, subscriber = _bus(channel), _bus(channel)
    try:
        received = await _subscribed(subscriber)
        assert publisher is not subscriber
        assert publisher._handlers == []  # noqa: SLF001 - the point of the test

        await publisher.publish_stop(uuid.uuid4(), 1)
        await _await_signal(received)

        # The publisher's own client is a different object than the
        # subscriber's, so the delivery above went over the wire.
        assert publisher._client is not subscriber._client  # noqa: SLF001
    finally:
        await asyncio.gather(publisher.close(), subscriber.close())


async def test_every_subscribed_worker_receives_the_signal(channel: str) -> None:
    """Fan-out, because the publisher does not know which worker owns the turn.

    A Stop is broadcast and each worker checks whether the generation is one of
    its own; addressing it would require a worker registry the lifecycle
    deliberately does not keep.
    """
    publisher = _bus(channel)
    first, second = _bus(channel), _bus(channel)
    generation_id = uuid.uuid4()
    try:
        first_received = await _subscribed(first)
        second_received = await _subscribed(second)

        await publisher.publish_stop(generation_id, 3)

        assert (await _await_signal(first_received)).generation_id == generation_id
        assert (await _await_signal(second_received)).generation_id == generation_id
    finally:
        await asyncio.gather(publisher.close(), first.close(), second.close())


async def test_a_signal_carries_no_content_only_identity_and_fence(channel: str) -> None:
    """The channel is shared infrastructure; nothing private may ride on it."""
    publisher, subscriber = _bus(channel), _bus(channel)
    try:
        received = await _subscribed(subscriber)
        await publisher.publish_stop(uuid.uuid4(), 2)
        signal = await _await_signal(received)
    finally:
        await asyncio.gather(publisher.close(), subscriber.close())

    # The dataclass is the whole payload surface: two fields, both opaque.
    assert set(vars(signal)) == {"generation_id", "version"}


# ----------------------------------------------------------------------
# what a broken hop must not break
# ----------------------------------------------------------------------


async def test_a_malformed_message_does_not_kill_the_subscriber(channel: str) -> None:
    """The loop is how every *future* Stop arrives.

    One unusable message escaping it would disable cancellation for the whole
    process until restart, so the loop drops what it cannot read and continues.
    """
    publisher, subscriber = _bus(channel), _bus(channel)
    generation_id = uuid.uuid4()
    try:
        received = await _subscribed(subscriber)

        # Straight onto the channel, bypassing `publish_stop`'s encoder.
        raw = await publisher._ensure_client()  # noqa: SLF001 - deliberate bad input
        await raw.publish(channel, "this is not a stop signal")
        await raw.publish(channel, '{"schema_version": 999}')

        await publisher.publish_stop(generation_id, 4)
        signal = await _await_signal(received)
    finally:
        await asyncio.gather(publisher.close(), subscriber.close())

    assert signal.generation_id == generation_id


async def test_publishing_with_no_subscriber_is_not_an_error(channel: str) -> None:
    """The durable transition already happened.

    Raising here would fail the HTTP request and tell the user nothing had
    happened, when the row already says `stop_requested` and the owning worker
    will find it on its next check.
    """
    publisher = _bus(channel)
    try:
        await publisher.publish_stop(uuid.uuid4(), 1)
    finally:
        await publisher.close()


async def test_a_closed_bus_publishes_nothing(channel: str) -> None:
    publisher, subscriber = _bus(channel), _bus(channel)
    try:
        received = await _subscribed(subscriber)
        await publisher.close()

        await publisher.publish_stop(uuid.uuid4(), 1)
        await asyncio.sleep(0.2)

        assert received.empty()
    finally:
        await subscriber.close()


async def test_a_closed_subscriber_stops_receiving(channel: str) -> None:
    publisher, subscriber = _bus(channel), _bus(channel)
    try:
        received = await _subscribed(subscriber)
        await subscriber.close()

        await publisher.publish_stop(uuid.uuid4(), 1)
        await asyncio.sleep(0.2)

        assert received.empty()
    finally:
        await publisher.close()


# ----------------------------------------------------------------------
# the wire format, over the real transport
# ----------------------------------------------------------------------


async def test_the_encoder_and_decoder_agree_over_the_wire(channel: str) -> None:
    """A round trip through Redis, not just through the two functions.

    `decode_responses=True` means the subscriber sees `str` where a differently
    configured client would see `bytes`; the decoder handles both, and this is
    the path that proves the configured one works.
    """
    publisher, subscriber = _bus(channel), _bus(channel)
    signal = StopSignal(generation_id=uuid.uuid4(), version=11)
    try:
        received = await _subscribed(subscriber)
        raw = await publisher._ensure_client()  # noqa: SLF001 - exercising the encoder
        await raw.publish(channel, encode_stop_signal(signal))

        delivered = await _await_signal(received)
    finally:
        await asyncio.gather(publisher.close(), subscriber.close())

    assert delivered == signal
    assert decode_stop_signal(encode_stop_signal(signal)) == signal


def test_the_schema_version_is_the_one_the_wire_carries() -> None:
    """A bump here is a compatibility event for every deployed worker."""
    assert STOP_SIGNAL_SCHEMA_VERSION == 1
