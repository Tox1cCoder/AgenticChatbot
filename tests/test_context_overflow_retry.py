from app.ai.context_overflow import is_context_overflow_error, compact_tool_messages_for_retry
from langchain_core.messages import ToolMessage


def test_detects_common_context_limit_errors():
    assert is_context_overflow_error(Exception("maximum context length exceeded"))
    assert is_context_overflow_error(Exception("input token limit exceeded"))
    assert is_context_overflow_error(Exception("context window is too small"))
    assert not is_context_overflow_error(Exception("network timeout"))


def test_compact_tool_messages_replaces_large_tool_output():
    messages = [
        ToolMessage(
            content="x" * 200,
            tool_call_id="call-1",
            name="search_documents",
        )
    ]

    compacted = compact_tool_messages_for_retry(messages, max_chars=40)

    assert len(compacted) == 1
    assert isinstance(compacted[0], ToolMessage)
    assert len(compacted[0].content) < 120
    assert "Tool output compacted for context retry" in compacted[0].content
    assert "call-1" in compacted[0].content
