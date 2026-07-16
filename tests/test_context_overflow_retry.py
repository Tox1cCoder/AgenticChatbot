import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.ai.context_overflow import (
    invoke_with_context_overflow_retry,
    is_context_overflow_error,
    prepare_aggressive_context_retry,
)


def test_detects_common_context_limit_errors():
    assert is_context_overflow_error(Exception("maximum context length exceeded"))
    assert is_context_overflow_error(Exception("input token limit exceeded"))
    assert is_context_overflow_error(Exception("context window is too small"))
    assert not is_context_overflow_error(Exception("network timeout"))


def test_aggressive_retry_compacts_large_tool_output():
    messages = [
        ToolMessage(
            content="x" * 200,
            tool_call_id="call-1",
            name="search_documents",
        )
    ]

    compacted = prepare_aggressive_context_retry(messages, tool_preview_chars=40)

    assert len(compacted) == 1
    assert isinstance(compacted[0], ToolMessage)
    assert len(compacted[0].content) < 120
    assert "Tool output compacted for context retry" in compacted[0].content
    assert "call-1" in compacted[0].content


def test_aggressive_retry_reduces_old_turns_and_tool_previews_without_splitting() -> None:
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="old question"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "search", "args": {}}]),
        ToolMessage(content="x" * 500, tool_call_id="c1", name="search"),
        AIMessage(content="old answer"),
        HumanMessage(content="recent question"),
        AIMessage(content="recent answer"),
        HumanMessage(content="current"),
    ]

    reduced = prepare_aggressive_context_retry(messages, tool_preview_chars=30)

    assert isinstance(reduced[0], SystemMessage)
    assert isinstance(reduced[-1], HumanMessage)
    assert reduced[-1].content == "current"
    assert all("c1" not in str(getattr(message, "tool_call_id", "")) for message in reduced)
    assert any(message.content == "recent question" for message in reduced)


def test_aggressive_retry_does_not_slice_through_group_after_incomplete_prefix() -> None:
    messages = [
        SystemMessage(content="system"),
        ToolMessage(content="orphan result", tool_call_id="orphan"),
        HumanMessage(content="older user"),
        AIMessage(content="older answer"),
        HumanMessage(content="current user"),
    ]

    reduced = prepare_aggressive_context_retry(messages, tool_preview_chars=100)

    assert [(message.type, message.content) for message in reduced] == [
        ("system", "system"),
        ("tool", "orphan result"),
        ("human", "current user"),
    ]


@pytest.mark.asyncio
async def test_provider_overflow_gets_exactly_one_aggressive_retry() -> None:
    calls = []

    async def invoke(messages):
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError("maximum context length exceeded")
        return "ok"

    result = await invoke_with_context_overflow_retry(
        invoke,
        [SystemMessage(content="system"), HumanMessage(content="current")],
        enabled=True,
        tool_preview_chars=20,
    )

    assert result == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_repeated_provider_overflow_is_surfaced_without_looping() -> None:
    calls = 0

    async def invoke(_messages):
        nonlocal calls
        calls += 1
        raise RuntimeError("maximum context length exceeded")

    with pytest.raises(RuntimeError, match="maximum context length"):
        await invoke_with_context_overflow_retry(
            invoke,
            [SystemMessage(content="system"), HumanMessage(content="current")],
            enabled=True,
            tool_preview_chars=20,
        )

    assert calls == 2
