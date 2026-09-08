"""The same turn, projected through both public transports, compared.

Parity is the point of the canonical event layer, and it is the kind of
property that decays quietly: an adapter gains a field, or drops one, and
nothing fails until a user reports that Stop works in one client and not the
other. So these tests do not read the two adapters and agree they look similar
— they normalize both projections to a transport-independent shape and compare
them as values.

What must match is the *lifecycle*: which generation events were published, in
what order, carrying which fields. What must not is the wire vocabulary —
``token`` versus ``text-delta``, a ``[DONE]`` sentinel or none — and the
normalizer drops exactly that.

One scenario is deliberately about an absence: a stream that simply stops
producing events publishes no lifecycle event at all, through either transport.
A closed socket is transport recovery, not a decision about the turn.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import V3StreamEvent, make_event
from app.services.event_streaming.internal_sse import legacy_event_from_v3

GENERATION_ID = "33333333-3333-3333-3333-333333333333"
CONTINUATION_ID = "44444444-4444-4444-4444-444444444444"
ASSISTANT_MESSAGE_ID = "55555555-5555-5555-5555-555555555555"

#: Internal SSE name -> the transport-independent phase. The AI SDK reports the
#: phase directly on its ``data-generation`` part, so both sides normalize to
#: the same vocabulary.
_PHASE_BY_LEGACY_TYPE = {
    "generation_start": "start",
    "generation_status": "status",
    "continuation_available": "continuation_available",
}


def _lifecycle_via_internal_sse(events: list[V3StreamEvent]) -> list[dict[str, Any]]:
    """The lifecycle projection a Streamlit client sees."""
    projected = []
    for event in events:
        payload = legacy_event_from_v3(event)
        if payload is None:
            continue
        phase = _PHASE_BY_LEGACY_TYPE.get(str(payload.get("type")))
        if phase is None:
            continue
        projected.append({"phase": phase, **{k: v for k, v in payload.items() if k != "type"}})
    return projected


def _lifecycle_via_ai_sdk(events: list[V3StreamEvent]) -> list[dict[str, Any]]:
    """The lifecycle projection an AI SDK client sees."""

    async def source():
        for event in events:
            yield event

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m1", text_id="t1", reasoning_id="r1"),
        heartbeat_interval_seconds=30.0,
    )

    async def collect():
        return [chunk async for chunk in adapter.iter_sse()]

    parts = [
        json.loads(chunk[len("data: ") :].strip())
        for chunk in asyncio.run(collect())
        if chunk.startswith("data: ") and chunk[len("data: ") :].strip() != "[DONE]"
    ]
    return [dict(part["data"]) for part in parts if part.get("type") == "data-generation"]


def _assert_parity(events: list[V3StreamEvent]) -> list[dict[str, Any]]:
    """Both transports publish the same lifecycle, and return it."""
    internal = _lifecycle_via_internal_sse(events)
    ai_sdk = _lifecycle_via_ai_sdk(events)
    assert internal == ai_sdk, (
        "the transports disagree about the lifecycle\n"
        f"internal SSE: {internal}\n"
        f"AI SDK:       {ai_sdk}"
    )
    return internal


# ----------------------------------------------------------------------
# scripted scenarios
# ----------------------------------------------------------------------


def _start(status: str = "running", version: int = 2) -> V3StreamEvent:
    return make_event(
        "run_start",
        sequence=1,
        data={
            "generation_id": GENERATION_ID,
            "logical_turn_id": "turn-1",
            "status": status,
            "version": version,
            "execution_epoch": 0,
            "continuation_id": None,
            "continuation_available": False,
            "continuation_block_reason": None,
            "assistant_message_id": None,
            "terminal_reason": None,
        },
    )


def _status(status: str, *, version: int, **overrides: Any) -> V3StreamEvent:
    data: dict[str, Any] = {
        "generation_id": GENERATION_ID,
        "status": status,
        "version": version,
    }
    data.update(overrides)
    return make_event("generation_status", sequence=9, data=data)


def _pause(*, version: int, epoch: int = 0) -> V3StreamEvent:
    return make_event(
        "continuation_available",
        sequence=5,
        data={
            "type": "execution_budget_exhausted",
            "generation_id": GENERATION_ID,
            "logical_turn_id": "turn-1",
            "status": "continuable",
            "version": version,
            "execution_epoch": epoch,
            "continuation_id": CONTINUATION_ID,
            "continuation_available": True,
            "continuation_block_reason": None,
            "assistant_message_id": ASSISTANT_MESSAGE_ID,
            "terminal_reason": None,
            # Internal-only; neither transport may publish these.
            "validated_content": "SECRET-PARTIAL-TEXT",
            "budget": {"exhausted_by": "tool_calls"},
            "thread_id": "wf2:conv:turn-1",
            "active_agent_id": "search_agent",
        },
    )


def _tokens(*texts: str) -> list[V3StreamEvent]:
    return [make_event("message_delta", sequence=2, data={"text": text}) for text in texts]


def _complete() -> V3StreamEvent:
    return make_event("complete", sequence=99, data={"message": {"id": ASSISTANT_MESSAGE_ID}})


# ----------------------------------------------------------------------
# parity, scenario by scenario
# ----------------------------------------------------------------------


def test_normal_completion_has_parity():
    lifecycle = _assert_parity([_start(), *_tokens("hello ", "world"), _complete()])

    assert [item["phase"] for item in lifecycle] == ["start"]


def test_a_soft_limit_continuation_has_parity():
    lifecycle = _assert_parity([_start(), *_tokens("partial"), _pause(version=4)])

    assert [item["phase"] for item in lifecycle] == ["start", "continuation_available"]
    assert lifecycle[-1]["continuation_id"] == CONTINUATION_ID
    assert lifecycle[-1]["continuation_available"] is True


def test_a_stop_during_a_provider_wait_has_parity():
    """No tokens yet: the turn was interrupted before it produced anything."""
    lifecycle = _assert_parity(
        [_start(), _status("stop_requested", version=3), _status("stopped", version=4)]
    )

    assert [item["phase"] for item in lifecycle] == ["start", "status", "status"]
    assert [item["status"] for item in lifecycle] == ["running", "stop_requested", "stopped"]


def test_a_stop_that_times_out_has_parity():
    """It stays ``stop_requested``, and both transports say so.

    A timeout is a pending state, not a failure and not a stop. Reporting
    ``stopped`` on one transport and ``stop_requested`` on the other is exactly
    the divergence this file exists to prevent.
    """
    lifecycle = _assert_parity(
        [_start(), *_tokens("half an "), _status("stop_requested", version=3)]
    )

    assert lifecycle[-1]["status"] == "stop_requested"


def test_a_stop_while_continuable_has_parity():
    lifecycle = _assert_parity(
        [
            _start(),
            _pause(version=4),
            _status(
                "completed_partial",
                version=5,
                continuation_available=False,
                continuation_id=None,
                terminal_reason="stopped_partial",
            ),
        ]
    )

    assert [item["phase"] for item in lifecycle] == [
        "start",
        "continuation_available",
        "status",
    ]
    assert lifecycle[-1]["continuation_available"] is False


def test_a_continue_after_a_pause_has_parity():
    lifecycle = _assert_parity(
        [
            _start(status="continuing", version=5),
            *_tokens("the rest of the answer"),
            _complete(),
        ]
    )

    assert [item["phase"] for item in lifecycle] == ["start"]
    assert lifecycle[0]["status"] == "continuing"


def test_a_repeated_continue_has_parity():
    """The second Continue is refused, and the refusal is not a lifecycle event.

    An error must not masquerade as a status change on either transport: a
    client that treated it as one would show the turn as having moved.
    """
    events = [
        _start(status="continuing", version=5),
        make_event(
            "error",
            sequence=2,
            data={
                "error": "the continuation has already been used",
                "error_code": "continuation_unavailable",
            },
        ),
    ]

    lifecycle = _assert_parity(events)

    assert [item["phase"] for item in lifecycle] == ["start"]


def test_a_disconnect_publishes_no_lifecycle_event_on_either_transport():
    """The absence is the assertion.

    A stream that stops producing events has not thereby stopped the turn, and
    neither adapter may invent a status from the socket closing.
    """
    lifecycle = _assert_parity([_start(), *_tokens("half an answ")])

    assert [item["phase"] for item in lifecycle] == ["start"]
    assert lifecycle[0]["status"] == "running"


def test_a_blocked_mutation_outcome_has_parity():
    """An unknown mutation receipt blocks continuation, with its reason."""
    lifecycle = _assert_parity(
        [
            _start(),
            _status(
                "stopped",
                version=4,
                continuation_available=False,
                continuation_id=None,
                continuation_block_reason="mutation_outcome_unknown",
                assistant_message_id=ASSISTANT_MESSAGE_ID,
            ),
        ]
    )

    assert lifecycle[-1]["continuation_available"] is False
    assert lifecycle[-1]["continuation_block_reason"] == "mutation_outcome_unknown"


# ----------------------------------------------------------------------
# what neither transport may publish
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "leaked", ["SECRET-PARTIAL-TEXT", "wf2:conv:turn-1", "exhausted_by", "search_agent"]
)
def test_neither_transport_publishes_the_pause_internals(leaked):
    events = [_start(), _pause(version=4)]

    internal = json.dumps(_lifecycle_via_internal_sse(events))
    ai_sdk = json.dumps(_lifecycle_via_ai_sdk(events))

    assert leaked not in internal
    assert leaked not in ai_sdk


def test_both_transports_publish_the_fence_a_command_needs():
    """R5. A client with no version cannot issue a fenced command at all."""
    lifecycle = _assert_parity([_start(version=7)])

    assert lifecycle[0]["version"] == 7


def test_the_two_transports_publish_the_identical_field_set():
    """Compared as sets, so a field added to one adapter alone fails here."""
    events = [_start(), _pause(version=4), _status("completed_partial", version=5)]

    internal = _lifecycle_via_internal_sse(events)
    ai_sdk = _lifecycle_via_ai_sdk(events)

    assert [set(item) for item in internal] == [set(item) for item in ai_sdk]


# ----------------------------------------------------------------------
# the wire vocabulary is allowed to differ, and does
# ----------------------------------------------------------------------


def test_the_transports_still_differ_where_they_are_meant_to():
    """A guard on the normalizer itself.

    If the two projections were identical at the wire level, this file would be
    comparing something trivial rather than a real parity property.
    """

    async def source():
        for event in [_start(), *_tokens("hi")]:
            yield event

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m1", text_id="t1", reasoning_id="r1"),
        heartbeat_interval_seconds=30.0,
    )

    async def collect():
        return [chunk async for chunk in adapter.iter_sse()]

    ai_sdk_raw = "".join(asyncio.run(collect()))
    internal_raw = json.dumps([legacy_event_from_v3(event) for event in [_start(), *_tokens("hi")]])

    assert "text-delta" in ai_sdk_raw
    assert "data: [DONE]" in ai_sdk_raw
    assert '"token"' in internal_raw
    assert "[DONE]" not in internal_raw
