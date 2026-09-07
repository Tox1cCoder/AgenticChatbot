"""Stop has to reach the worker that owns the generation, wherever it is.

The HTTP request that asks for a Stop rarely lands on the process streaming the
answer. A process-local registry can only cancel what it happens to hold, so
the durable transition is the decision and this bus is how the owning worker
learns about it.

What crosses the wire is deliberately almost nothing: a schema version, a
generation id and the lifecycle version. No prompt, no partial answer, no user
or conversation id. A subscriber that needs more reads the row it already has
permission to read — the bus is a wake-up, not a data channel.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.services.generation_control_bus import (
    InMemoryGenerationControlBus,
    StopSignal,
    decode_stop_signal,
    encode_stop_signal,
)


async def test_a_subscriber_receives_a_published_stop():
    bus = InMemoryGenerationControlBus()
    received: list[StopSignal] = []

    async def handler(signal: StopSignal) -> None:
        received.append(signal)

    await bus.subscribe(handler)
    generation_id = uuid4()

    await bus.publish_stop(generation_id, 4)

    assert received == [StopSignal(generation_id=generation_id, version=4)]


async def test_every_subscriber_in_this_process_is_notified():
    """Workers subscribe once each; a single delivery would drop one."""
    bus = InMemoryGenerationControlBus()
    first: list[StopSignal] = []
    second: list[StopSignal] = []

    await bus.subscribe(lambda signal: _append(first, signal))
    await bus.subscribe(lambda signal: _append(second, signal))
    await bus.publish_stop(uuid4(), 1)

    assert len(first) == 1
    assert len(second) == 1


async def test_a_failing_handler_does_not_stop_the_others():
    """One worker's bug must not silence a Stop for every other worker."""
    bus = InMemoryGenerationControlBus()
    delivered: list[StopSignal] = []

    async def broken(_signal: StopSignal) -> None:
        raise RuntimeError("subscriber is broken")

    await bus.subscribe(broken)
    await bus.subscribe(lambda signal: _append(delivered, signal))

    await bus.publish_stop(uuid4(), 1)

    assert len(delivered) == 1


async def test_publishing_with_no_subscriber_is_not_an_error():
    """The owning worker may be in another process, or already gone."""
    bus = InMemoryGenerationControlBus()

    await bus.publish_stop(uuid4(), 1)


async def test_a_closed_bus_delivers_nothing_further():
    bus = InMemoryGenerationControlBus()
    received: list[StopSignal] = []
    await bus.subscribe(lambda signal: _append(received, signal))

    await bus.close()
    await bus.publish_stop(uuid4(), 1)

    assert received == []


async def test_close_is_idempotent():
    bus = InMemoryGenerationControlBus()

    await bus.close()
    await bus.close()


# ----------------------------------------------------------------------
# wire format
# ----------------------------------------------------------------------


def test_the_payload_carries_only_the_three_fields_it_needs():
    """A Stop signal is a wake-up. Anything more is a leak waiting to happen."""
    signal = StopSignal(generation_id=uuid4(), version=7)

    payload = json.loads(encode_stop_signal(signal))

    assert set(payload) == {"schema_version", "generation_id", "version"}
    assert payload["schema_version"] == 1


def test_a_signal_round_trips_through_the_wire_format():
    signal = StopSignal(generation_id=uuid4(), version=7)

    assert decode_stop_signal(encode_stop_signal(signal)) == signal


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "not json",
        "{}",
        '{"schema_version": 1}',
        '{"schema_version": 1, "generation_id": "not-a-uuid", "version": 1}',
        '{"schema_version": 99, "generation_id": "%s", "version": 1}',
        '{"schema_version": 1, "generation_id": "%s", "version": -1}',
    ],
)
def test_an_unusable_payload_is_dropped_rather_than_raised(payload: str):
    """A malformed message must not kill the subscriber loop.

    The loop is how every future Stop arrives. Raising there would mean one bad
    publisher permanently disables cancellation for the whole process.
    """
    if "%s" in payload:
        payload = payload % uuid4()

    assert decode_stop_signal(payload) is None


def test_a_future_schema_version_is_ignored_not_guessed():
    """Forward compatibility by refusal: a v2 field set is not a v1 signal."""
    signal = StopSignal(generation_id=uuid4(), version=1)
    payload = json.loads(encode_stop_signal(signal))
    payload["schema_version"] = 2

    assert decode_stop_signal(json.dumps(payload)) is None


def _append(sink: list, signal: StopSignal):
    async def _run() -> None:
        sink.append(signal)

    return _run()

