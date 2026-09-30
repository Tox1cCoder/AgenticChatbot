"""An agent's error response is persisted, so it names the failure, not its text.

``AgentResponse.error`` and ``metadata["error"]`` reach the message record and
the stream. Provider exception text can carry URLs, request ids and key
fragments, so it belongs in the log; the stored value is a stable code when the
exception declares one, else the exception's type name.
"""

from __future__ import annotations

import pytest

from app.ai.agents.chat_agent import ChatAgent
from app.ai.schemas import AgentMessage, MessageRole


def _chat_agent() -> ChatAgent:
    agent = ChatAgent.__new__(ChatAgent)
    agent.model_name = "test-model"
    return agent


@pytest.mark.asyncio
async def test_chat_agent_failure_stores_the_type_not_the_text(monkeypatch):
    agent = _chat_agent()

    async def _failing_invoke(*_args, **_kwargs):
        raise ValueError("bad gateway from https://internal.example/?token=private-value")

    monkeypatch.setattr(agent, "invoke_model", _failing_invoke)

    response = await agent.process_message(
        AgentMessage(role=MessageRole.USER, content="hi", metadata={}),
        conversation_id="conv-1",
    )

    assert response.error == "ValueError"
    assert response.metadata["error"] == "ValueError"
    assert response.message.content == (
        "I'm sorry, but I encountered an error processing your request."
    )
    assert "private-value" not in response.model_dump_json()


def test_an_exception_with_a_stable_code_stores_the_code():
    class _CodedError(RuntimeError):
        code = "provider_rate_limited"

    response = _chat_agent()._build_error_response(
        message="I encountered an error processing your request.",
        conversation_id="conv-1",
        error=_CodedError("429 for account acct-private"),
    )

    assert response.error == "provider_rate_limited"
    assert "acct-private" not in response.model_dump_json()


def test_a_plain_string_error_code_is_kept_as_given():
    response = _chat_agent()._build_error_response(
        message="the request is too large.",
        conversation_id="conv-1",
        error="context_budget_fixed_input_exceeded",
    )

    assert response.error == "context_budget_fixed_input_exceeded"
    assert response.metadata["error"] == "context_budget_fixed_input_exceeded"
