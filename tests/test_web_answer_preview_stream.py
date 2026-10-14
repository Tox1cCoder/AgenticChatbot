"""Web answer drafts reach the UI without becoming authoritative answer text."""

import json

import pytest
from langchain_core.messages import ToolMessage

from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event
from app.services.event_streaming.graph_public_projection import (
    GraphPublicStreamProjector,
    StreamProjectionContext,
)


def _projector():
    return GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda **kwargs: iter(()),
        suppress_internal_stream_chunks=True,
    )


def test_web_answer_chunks_are_previewed_and_a_repair_replaces_the_draft():
    projector = _projector()
    context = StreamProjectionContext()
    raw = [
        make_event("tool_execution_end", sequence=1, tool_name="web_search", data={"output": {}}),
        make_event("message_delta", sequence=2, run_id="draft", data={"text": "First "}),
        make_event("message_delta", sequence=3, run_id="draft", data={"text": "draft"}),
        make_event("message_delta", sequence=4, run_id="repair", data={"text": "Cited "}),
        make_event("message_delta", sequence=5, run_id="repair", data={"text": "answer"}),
    ]

    public = [event for item in raw for event in projector.map_event(item, context)]
    previews = [event.data for event in public if event.type == "answer_preview"]

    assert previews == [
        {"operation": "replace", "text": "First "},
        {"operation": "append", "text": "draft"},
        {"operation": "replace", "text": "Cited "},
        {"operation": "append", "text": "answer"},
    ]
    assert all(event.type != "message_delta" for event in public)


def test_repair_with_same_prefix_replaces_the_whole_draft():
    projector = _projector()
    context = StreamProjectionContext()
    raw = [
        make_event("tool_execution_end", sequence=1, tool_name="web_search", data={"output": {}}),
        make_event("message_delta", sequence=2, run_id="draft", data={"text": "Hello"}),
        make_event("message_delta", sequence=3, run_id="repair", data={"text": "Hello there"}),
    ]
    public = [event for item in raw for event in projector.map_event(item, context)]
    assert [event.data for event in public if event.type == "answer_preview"] == [
        {"operation": "replace", "text": "Hello"},
        {"operation": "replace", "text": "Hello there"},
    ]


def test_web_draft_keeps_repeated_tokens():
    projector = _projector()
    context = StreamProjectionContext()
    raw = [
        make_event("tool_execution_end", sequence=1, tool_name="web_search", data={"output": {}}),
        *(
            make_event("message_delta", sequence=i, run_id="draft", data={"text": text})
            for i, text in enumerate(["ha", "ha", "!"], start=2)
        ),
    ]
    public = [event for item in raw for event in projector.map_event(item, context)]
    assert [event.data for event in public if event.type == "answer_preview"] == [
        {"operation": "replace", "text": "ha"},
        {"operation": "append", "text": "ha"},
        {"operation": "append", "text": "!"},
    ]


def test_tuple_path_repair_run_replaces_the_draft():
    """The tuple fallback carries run identity on the chunk id; a repair run
    must replace the draft rather than append to it."""
    from langchain_core.messages import AIMessageChunk

    projector = _projector()
    context = StreamProjectionContext()

    def chunk(text, run):
        return make_event(
            "state_snapshot",
            sequence=0,
            data={
                "kind": "messages_tuple",
                "chunk": AIMessageChunk(content=text, id=run),
                "metadata": {"langgraph_node": "search_agent"},
            },
        )

    raw = [
        make_event("tool_execution_end", sequence=1, tool_name="web_search", data={"output": {}}),
        chunk("Hello", "lc_run--draft"),
        chunk(" world", "lc_run--draft"),
        chunk("Hello", "lc_run--repair"),
        chunk(" there", "lc_run--repair"),
    ]
    public = [event for item in raw for event in projector.map_event(item, context)]
    assert [event.data for event in public if event.type == "answer_preview"] == [
        {"operation": "replace", "text": "Hello"},
        {"operation": "append", "text": " world"},
        {"operation": "replace", "text": "Hello"},
        {"operation": "append", "text": " there"},
    ]


def test_web_answer_does_not_preview_before_search_completes():
    projector = _projector()
    context = StreamProjectionContext(requires_web=True)
    public = list(
        projector.map_event(
            make_event("message_delta", sequence=1, run_id="draft", data={"text": "Internal"}),
            context,
        )
    )
    assert public == []


def test_legacy_web_tool_completion_enables_answer_preview():
    projector = GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda **kwargs: iter(
            ({"tool_call_id": "web-1", "name": "web_search", "result": "Results"},)
        ),
        suppress_internal_stream_chunks=True,
    )
    context = StreamProjectionContext(requires_web=True)
    update = make_event(
        "state_snapshot",
        sequence=1,
        data={
            "kind": "updates_tuple",
            "node_state": {"messages": [ToolMessage(content="Results", tool_call_id="web-1")]},
        },
    )
    list(projector.map_event(update, context))
    public = list(
        projector.map_event(
            make_event("message_delta", sequence=2, run_id="answer", data={"text": "Hello"}),
            context,
        )
    )
    assert [event.data for event in public if event.type == "answer_preview"] == [
        {"operation": "replace", "text": "Hello"}
    ]


@pytest.mark.asyncio
async def test_ai_sdk_preview_is_transient_and_final_answer_remains_authoritative():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Uncited draft"}
        )
        yield make_event("message_delta", sequence=2, data={"text": "Grounded final answer"})
        yield make_event("complete", sequence=3, data={"message": {"id": "m-1"}})

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    previews = [part for part in payloads if part.get("type") == "data-answer-preview"]
    final_text = [part["delta"] for part in payloads if part.get("type") == "text-delta"]
    assert previews == [
        {
            "type": "data-answer-preview",
            "data": {"operation": "replace", "text": "Uncited draft"},
            "transient": True,
        },
        {"type": "data-answer-preview", "data": {"operation": "clear"}, "transient": True},
    ]
    assert final_text == ["Grounded final answer"]


@pytest.mark.asyncio
async def test_ai_sdk_honors_explicit_preview_clear():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Draft"}
        )
        yield make_event("answer_preview", sequence=2, data={"operation": "clear"})
        yield make_event(
            "answer_preview", sequence=3, data={"operation": "replace", "text": "New draft"}
        )
        yield make_event("complete", sequence=4, data={"message": {"id": "m-1"}})

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert [
        part.get("data", {}).get("operation")
        for part in payloads
        if part.get("type") == "data-answer-preview"
    ] == ["replace", "clear", "replace", "clear"]


@pytest.mark.asyncio
async def test_ai_sdk_clears_preview_when_stream_source_fails():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Draft"}
        )
        raise RuntimeError("provider failure")

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    assert [
        part.get("data", {}).get("operation")
        for part in payloads
        if part.get("type") == "data-answer-preview"
    ] == ["replace", "clear"]


@pytest.mark.asyncio
async def test_opted_in_web_answer_streams_text_deltas_and_reconciles_final_message():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Draft "}
        )
        yield make_event(
            "answer_preview", sequence=2, data={"operation": "append", "text": "answer"}
        )
        yield make_event("message_delta", sequence=3, data={"text": "Grounded final answer"})
        yield make_event(
            "complete",
            sequence=4,
            data={"message": {"id": "m-1", "content": "Grounded final answer"}},
        )

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(
            message_id="m-1",
            text_id="t-1",
            reasoning_id="r-1",
            stream_web_answer_v1=True,
        ),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    assert [part["delta"] for part in payloads if part.get("type") == "text-delta"] == [
        "Draft ",
        "answer",
    ]
    assert [part["data"] for part in payloads if part.get("type") == "data-answer-reconcile"] == [
        {"messageId": "m-1", "content": "Grounded final answer"}
    ]
    types = [part.get("type") for part in payloads]
    assert types.index("text-delta") < types.index("data-answer-reconcile") < types.index("finish")


@pytest.mark.asyncio
async def test_opted_in_web_answer_repair_is_sent_as_replacement_draft():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Uncited"}
        )
        yield make_event(
            "answer_preview", sequence=2, data={"operation": "replace", "text": "Cited "}
        )
        yield make_event(
            "answer_preview", sequence=3, data={"operation": "append", "text": "reply"}
        )
        yield make_event("message_delta", sequence=4, data={"text": "Cited [1](https://a.test)"})
        yield make_event(
            "complete",
            sequence=5,
            data={"message": {"id": "m-1", "content": "Cited [1](https://a.test)"}},
        )

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(
            message_id="m-1",
            text_id="t-1",
            reasoning_id="r-1",
            stream_web_answer_v1=True,
        ),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    assert [part["delta"] for part in payloads if part.get("type") == "text-delta"] == [
        "Uncited"
    ]
    assert [part["data"] for part in payloads if part.get("type") == "data-answer-preview"] == [
        {"operation": "replace", "text": "Cited "},
        {"operation": "append", "text": "reply"},
        {"operation": "clear"},  # the active repair never survives reconciliation
    ]
    assert [
        part["data"]["content"]
        for part in payloads
        if part.get("type") == "data-answer-reconcile"
    ] == [
        "Cited [1](https://a.test)"
    ]


@pytest.mark.asyncio
async def test_opted_in_web_answer_error_clears_streamed_draft():
    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Draft"}
        )
        yield make_event("error", sequence=2, data={"error": "Generation failed"})

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(
            message_id="m-1",
            text_id="t-1",
            reasoning_id="r-1",
            stream_web_answer_v1=True,
        ),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    assert [part["data"] for part in payloads if part.get("type") == "data-answer-reconcile"] == [
        {"messageId": "m-1", "content": ""}
    ]


# -- deterministic draft replacement (opted-in clients) -----------------------

_ANSWER_EVENT_TYPES = {"text-delta", "data-answer-preview", "data-answer-reconcile"}


async def _wire_payloads(source, **state_kwargs):
    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1", **state_kwargs),
    )
    return [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _answer_events(payloads):
    return [part for part in payloads if part.get("type") in _ANSWER_EVENT_TYPES]


def _reconciled(payloads):
    return [
        part["data"]["content"]
        for part in payloads
        if part.get("type") == "data-answer-reconcile"
    ]


def _preview(operation, text=None):
    data = {"operation": operation}
    if text is not None:
        data["text"] = text
    return make_event("answer_preview", sequence=0, data=data)


def _complete(content):
    return make_event("complete", sequence=0, data={"message": {"id": "m-1", "content": content}})


def _visible_answer(payloads) -> str:
    """The FE reducer contract: authoritative ?? repair ?? SDK draft."""
    sdk_draft = ""
    repair = None
    authoritative = None
    for part in payloads:
        kind = part.get("type")
        if kind == "text-delta":
            sdk_draft += part["delta"]
        elif kind == "data-answer-preview" and authoritative is None:
            operation = part["data"]["operation"]
            if operation == "replace":
                repair = part["data"]["text"]
            elif operation == "append":
                repair = (repair or "") + part["data"]["text"]
            else:
                repair = None
        elif kind == "data-answer-reconcile":
            authoritative = part["data"]["content"]
            repair = None
    if authoritative is not None:
        return authoritative
    return repair if repair is not None else sdk_draft


@pytest.mark.asyncio
async def test_opted_in_draft_repair_and_completion_follow_the_exact_sequence():
    async def source():
        yield _preview("replace", "Initial answer ")
        yield _preview("replace", "Repaired answer")
        yield make_event("message_delta", sequence=0, data={"text": "Final cited answer"})
        yield _complete("Final cited answer")

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)
    answer_events = _answer_events(payloads)

    assert [part["type"] for part in answer_events] == [
        "text-delta",  # initial draft only
        "data-answer-preview",  # repair replaces the display
        "data-answer-preview",  # explicit clear of the active repair
        "data-answer-reconcile",  # persisted final content, even if empty
    ]
    assert answer_events[2]["data"] == {"operation": "clear"}
    assert _reconciled(payloads) == ["Final cited answer"]
    assert "Final cited answer" not in [p.get("delta") for p in payloads]


@pytest.mark.asyncio
async def test_first_draft_without_repair_reconciles_once():
    async def source():
        yield _preview("replace", "Draft ")
        yield _preview("append", "text")
        yield _complete("Draft text")

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)

    assert [part["type"] for part in _answer_events(payloads)] == [
        "text-delta",
        "text-delta",
        "data-answer-reconcile",
    ]
    assert _reconciled(payloads) == ["Draft text"]


@pytest.mark.asyncio
async def test_multiple_repairs_end_with_one_clear_and_one_reconcile():
    async def source():
        yield _preview("replace", "Draft")
        yield _preview("replace", "Repair one")
        yield _preview("replace", "Repair two")
        yield _preview("append", " continues")
        yield _complete("Final")

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)
    previews = [p["data"] for p in payloads if p["type"] == "data-answer-preview"]

    assert previews == [
        {"operation": "replace", "text": "Repair one"},
        {"operation": "replace", "text": "Repair two"},
        {"operation": "append", "text": " continues"},
        {"operation": "clear"},
    ]
    assert _reconciled(payloads) == ["Final"]


@pytest.mark.asyncio
async def test_repair_with_identical_prefix_replaces_rather_than_appends():
    async def source():
        yield _preview("replace", "Hello")
        yield _preview("replace", "Hello")
        yield _preview("append", " there")
        yield _complete("Hello there")

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)

    assert [p["delta"] for p in payloads if p["type"] == "text-delta"] == ["Hello"]
    assert [p["data"] for p in payloads if p["type"] == "data-answer-preview"] == [
        {"operation": "replace", "text": "Hello"},
        {"operation": "append", "text": " there"},
        {"operation": "clear"},
    ]
    assert _visible_answer(payloads) == "Hello there"


@pytest.mark.asyncio
async def test_explicit_clear_then_repair_clears_the_new_repair_at_the_end():
    async def source():
        yield _preview("replace", "Draft")
        yield _preview("clear")
        yield _preview("replace", "Repair")
        yield _complete("Final")

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)

    assert [p["data"].get("operation") for p in payloads if p["type"] == "data-answer-preview"] == [
        "clear",
        "replace",
        "clear",
    ]
    assert _reconciled(payloads) == ["Final"]


@pytest.mark.asyncio
@pytest.mark.parametrize("final", ["Draft", "Different final", ""])
async def test_final_content_is_reconciled_exactly_once_never_appended(final):
    async def source():
        yield _preview("replace", "Draft")
        yield make_event("message_delta", sequence=0, data={"text": final})
        yield _complete(final)

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)

    assert _reconciled(payloads) == [final]
    assert [p["delta"] for p in payloads if p["type"] == "text-delta"] == ["Draft"]
    assert _visible_answer(payloads) == final


@pytest.mark.asyncio
async def test_source_exception_during_repair_clears_and_discards_before_the_error():
    async def source():
        yield _preview("replace", "Draft")
        yield _preview("replace", "Repair")
        raise RuntimeError("provider failure")

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)
    types = [p["type"] for p in payloads]

    assert _reconciled(payloads) == [""]
    clear_index = max(
        i for i, p in enumerate(payloads) if p.get("data", {}).get("operation") == "clear"
    )
    assert clear_index < types.index("data-answer-reconcile") < types.index("error")
    assert _visible_answer(payloads) == ""


@pytest.mark.asyncio
async def test_interrupt_during_repair_reconciles_to_the_persisted_message():
    async def source():
        yield _preview("replace", "Draft")
        yield _preview("replace", "Repair")
        yield make_event(
            "interrupt",
            sequence=0,
            data={
                "thread_id": "thread-1",
                "interrupt": {"interrupt_id": "int-1"},
                "message": {"id": "m-1", "content": "Approval needed"},
            },
        )

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)
    types = [p["type"] for p in payloads]

    assert _reconciled(payloads) == ["Approval needed"]
    assert types.index("data-answer-reconcile") < types.index("data-interrupt")
    assert [p["data"]["operation"] for p in payloads if p["type"] == "data-answer-preview"] == [
        "replace",
        "clear",
    ]


@pytest.mark.asyncio
async def test_validated_pause_reconciles_to_the_saved_partial():
    async def source():
        yield _preview("replace", "Unvalidated draft")
        yield make_event(
            "message_end",
            sequence=0,
            message_id="m-1",
            data={"message": {"id": "m-1", "content": "Validated partial"}},
        )
        yield make_event(
            "continuation_available",
            sequence=0,
            data={"generation_id": "g-1", "continuation_id": "c-1", "version": 3},
        )

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)

    assert _reconciled(payloads) == ["Validated partial"]
    assert _visible_answer(payloads) == "Validated partial"


@pytest.mark.asyncio
async def test_validated_pause_without_capability_delivers_the_saved_partial_as_text():
    async def source():
        yield _preview("replace", "Unvalidated draft")
        yield make_event(
            "message_end",
            sequence=0,
            message_id="m-1",
            data={"message": {"id": "m-1", "content": "Validated partial"}},
        )

    payloads = await _wire_payloads(source)

    assert [p["delta"] for p in payloads if p["type"] == "text-delta"] == ["Validated partial"]
    assert [p["data"]["operation"] for p in payloads if p["type"] == "data-answer-preview"] == [
        "replace",
        "clear",
    ]
    assert _reconciled(payloads) == []


@pytest.mark.asyncio
async def test_stream_ending_without_saved_content_discards_the_draft():
    async def source():
        yield _preview("replace", "Draft")
        yield _preview("replace", "Repair")

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)

    assert _reconciled(payloads) == [""]
    assert _visible_answer(payloads) == ""


@pytest.mark.asyncio
async def test_late_preview_after_reconciliation_is_ignored():
    async def source():
        yield _preview("replace", "Draft")
        yield make_event(
            "message_end",
            sequence=0,
            message_id="m-1",
            data={"message": {"id": "m-1", "content": "Saved"}},
        )
        yield _preview("replace", "Late repair")

    payloads = await _wire_payloads(source, stream_web_answer_v1=True)

    assert _reconciled(payloads) == ["Saved"]
    assert not any(
        p.get("data", {}).get("text") == "Late repair"
        for p in payloads
        if p["type"] == "data-answer-preview"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rich_capable", [True, False])
async def test_post_stream_answer_matches_projected_history(rich_capable):
    from app.services.event_streaming.ai_sdk_projection import (
        project_ai_sdk_message_for_capability,
    )

    persisted = {
        "id": "m-1",
        "content": "Intro [1](https://a.test)\n<!--rich:image:img-1-->\nMore.",
        "message_metadata": {},
    }

    async def source():
        yield _preview("replace", "Intro [[source:S1]]")
        yield _preview("replace", "Intro [1](https://a.test)")
        yield make_event("complete", sequence=0, data={"message": persisted})

    payloads = await _wire_payloads(
        source, stream_web_answer_v1=True, inline_rich_response_v1=rich_capable
    )
    history = project_ai_sdk_message_for_capability(
        dict(persisted), inline_rich_response_v1=rich_capable
    )

    assert _visible_answer(payloads) == history["content"]
