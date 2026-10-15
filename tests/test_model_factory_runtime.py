from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from app.ai.model_factory import ModelFactory
from app.core.runtime_modeling import ResolvedRuntimeModelConfig


def _runtime(
    *,
    provider: str = "openai",
    model: str = "gpt-5.6-luna",
    reasoning_effort: str | None = None,
) -> ResolvedRuntimeModelConfig:
    return ResolvedRuntimeModelConfig(
        agent_key="search",
        provider=provider,
        model=model,
        temperature=0.7,
        api_key="test-key",
        key_source="user",
        source="default",
        reasoning_effort=reasoning_effort,
    )


@pytest.fixture
def captured_openai(monkeypatch):
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("app.ai.model_factory.ChatOpenAI", constructor)
    return captured


def test_default_gpt_56_reasoning_uses_responses_api(captured_openai):
    ModelFactory.create_model_from_runtime(_runtime())

    assert captured_openai["use_responses_api"] is True
    # A summary is requested even with no explicit effort: the Responses API
    # returns reasoning text only when asked, and without it the turn shows no
    # thinking at all.
    assert captured_openai["reasoning"] == {"summary": "auto"}


async def test_gpt_61_sol_default_reasoning_and_tools_use_responses_on_the_wire(monkeypatch):
    """The real LangChain/OpenAI clients must not send tools to Chat Completions."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path != "/v1/responses":
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Function tools with reasoning_effort require /v1/responses",
                        "type": "invalid_request_error",
                        "param": "reasoning_effort",
                        "code": None,
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-6.1-sol",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Ready.", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        )

    @tool
    def lookup(query: str) -> str:
        """Look up information."""
        return query

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ModelFactory.create_model_from_runtime(
            _runtime(model="gpt-6.1-sol"),
            http_async_client=client,
        )
        response = await model.bind_tools([lookup]).ainvoke([HumanMessage(content="Hello")])

    assert len(requests) == 1
    assert requests[0].url.path == "/v1/responses"
    payload = json.loads(requests[0].content)
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["name"] == "lookup"
    assert payload["reasoning"] == {"summary": "auto"}
    assert "Ready." in str(response.content)


def test_explicit_openai_reasoning_uses_responses_api_and_preserves_effort(captured_openai):
    ModelFactory.create_model_from_runtime(_runtime(reasoning_effort="high"))

    assert captured_openai["use_responses_api"] is True
    assert captured_openai["reasoning"] == {"effort": "high", "summary": "auto"}


def test_explicit_none_keeps_tool_compatible_chat_completions(captured_openai):
    ModelFactory.create_model_from_runtime(_runtime(reasoning_effort="none"))

    assert captured_openai.get("use_responses_api") is not True
    assert captured_openai["reasoning_effort"] == "none"


def test_non_reasoning_openai_model_does_not_force_responses_api(captured_openai):
    ModelFactory.create_model_from_runtime(_runtime(model="gpt-4o"))

    assert captured_openai.get("use_responses_api") is not True


def test_provider_default_none_does_not_force_responses_api(captured_openai):
    ModelFactory.create_model_from_runtime(_runtime(model="gpt-5.4"))

    assert captured_openai.get("use_responses_api") is not True


def test_gemini_runtime_reasoning_reaches_provider_constructor(monkeypatch):
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    # The subclass is what this path constructs; patching the plain class
    # would patch a name the factory no longer calls.
    monkeypatch.setattr(
        "app.ai.model_factory.ReasoningNormalizedChatGoogleGenerativeAI", constructor
    )

    ModelFactory.create_model_from_runtime(
        _runtime(
            provider="gemini",
            model="gemini-3-flash-preview",
            reasoning_effort="high",
        )
    )

    assert captured["thinking_level"] == "high"
