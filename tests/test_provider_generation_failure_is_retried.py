"""A provider that says it failed did not succeed at producing nothing.

Reported as ``Error: No response generated`` with
``finish_reason=MALFORMED_FUNCTION_CALL``: the model spent its whole output on
thinking (the usage row reads ``out=1462, reasoning=1462``) and the function
call it then began was rejected by the provider's own parser. Nothing came back
-- no text, no tool calls -- and the turn failed ``empty_public_content``.

``MALFORMED_FUNCTION_CALL`` is not an answer. It is Gemini stating that this
generation failed, on an HTTP 200. The retry loop already existed and already
had backoff; it only ever treated a raised exception as failure, so a 200 that
declares its own failure was counted as success and returned as an empty
response.

Six faithful replays of the failing turn all succeeded, which is what a
transient generation failure looks like -- so the fix is to let the existing
loop see it, not to change what is sent.

Deliberately narrow. Only ``MALFORMED_FUNCTION_CALL`` is retried, and only when
nothing usable came back:

* ``SAFETY``/``RECITATION``/``PROHIBITED_CONTENT`` are decisions, not failures;
  an identical retry would be refused identically and the user is owed the real
  reason.
* ``MAX_TOKENS`` would truncate again the same way.
* A malformed call that still carried text or a usable tool call is a partial
  success -- retrying would throw away work the turn can use.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.ai.agents.chat_agent import ChatAgent


class _Provider:
    """Returns the scripted responses in order, recording each attempt."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.attempts = 0

    async def ainvoke(self, messages, config=None):
        self.attempts += 1
        response = self._responses[min(self.attempts - 1, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


def _malformed() -> AIMessage:
    return AIMessage(content="", response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"})


def _good() -> AIMessage:
    return AIMessage(content="here is the answer", response_metadata={"finish_reason": "STOP"})


def _refused(reason: str) -> AIMessage:
    return AIMessage(content="", response_metadata={"finish_reason": reason})


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr("app.ai.agents.base_agent.settings.provider_retry_attempts", 3)
    monkeypatch.setattr("app.ai.agents.base_agent.settings.provider_retry_delay_seconds", 0)
    return ChatAgent()


@pytest.mark.asyncio
async def test_a_malformed_function_call_is_retried(agent):
    provider = _Provider([_malformed(), _good()])

    response = await agent._ainvoke_with_retries(provider, [])

    assert provider.attempts == 2, "the failed generation was counted as a success"
    assert response.text == "here is the answer"


@pytest.mark.asyncio
async def test_retries_are_bounded_by_the_existing_attempt_budget(agent):
    provider = _Provider([_malformed()])

    response = await agent._ainvoke_with_retries(provider, [])

    assert provider.attempts == 3
    # The last response is still returned: the turn's own diagnostics report
    # the finish reason, rather than this raising something new.
    assert response.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL"


@pytest.mark.asyncio
async def test_a_good_response_is_never_retried(agent):
    provider = _Provider([_good()])

    await agent._ainvoke_with_retries(provider, [])

    assert provider.attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "MAX_TOKENS"])
async def test_a_refusal_or_a_truncation_is_not_retried(agent, reason):
    """These are outcomes. Repeating the same request repeats the outcome."""
    provider = _Provider([_refused(reason)])

    await agent._ainvoke_with_retries(provider, [])

    assert provider.attempts == 1


@pytest.mark.asyncio
async def test_a_malformed_call_that_still_produced_work_is_kept(agent):
    """Partial success is not failure -- the turn can route the tool call."""
    usable = AIMessage(
        content="",
        tool_calls=[{"name": "write_todos", "args": {}, "id": "c1", "type": "tool_call"}],
        response_metadata={"finish_reason": "MALFORMED_FUNCTION_CALL"},
    )
    provider = _Provider([usable])

    await agent._ainvoke_with_retries(provider, [])

    assert provider.attempts == 1
