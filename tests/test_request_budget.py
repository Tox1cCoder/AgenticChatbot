from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.ai.agents.base_agent import BaseAgent
from app.ai.request_budget import (
    BudgetConfig,
    RequestBudgetService,
    RequestEnvelope,
)
from app.ai.schemas import AgentMessage, AgentType, MessageRole
from app.ai.token_counter import TokenCounter
from app.core.config import settings
from app.core.runtime_modeling import ResolvedRuntimeModelConfig


class WeightedCounter:
    def estimate_request(self, *, messages, tools, attachments, **_kwargs):
        message_tokens = sum(_weight(message) for message in messages)
        tool_tokens = sum(int(tool.get("tokens", 0)) for tool in tools or ())
        attachment_tokens = sum(int(item.get("tokens", 0)) for item in attachments or ())
        return SimpleNamespace(
            input_tokens=message_tokens + tool_tokens + attachment_tokens,
            strategy="weighted",
            source="local",
        )


class NativeAwareCounter(WeightedCounter):
    async def count_request(self, **_kwargs):
        return SimpleNamespace(
            input_tokens=75,
            strategy="gemini:native_count",
            source="provider",
        )


def _weight(message) -> int:
    if isinstance(message, dict):
        return int(message.get("tokens", 0))
    return int(getattr(message, "additional_kwargs", {}).get("tokens", 0))


def _message(role: str, tokens: int, content: str = "x") -> dict:
    return {"role": role, "content": content, "tokens": tokens}


def _envelope(*, system=10, history=(), current=10, tools=(), attachments=()):
    return RequestEnvelope(
        provider="gemini",
        model="gemini-2.5-flash",
        system_messages=(_message("system", system),),
        history_messages=tuple(history),
        current_messages=(_message("user", current),),
        tools=tuple(tools),
        attachments=tuple(attachments),
    )


def _config(**overrides):
    values = dict(
        max_input_tokens=120,
        reserved_output_tokens=10,
        safety_margin_tokens=10,
        soft_ratio=0.70,
        hard_ratio=0.85,
        emergency_timeout_seconds=0.05,
    )
    values.update(overrides)
    return BudgetConfig(**values)


@pytest.mark.asyncio
async def test_complete_input_accounting_and_below_soft_proceeds() -> None:
    service = RequestBudgetService(WeightedCounter())
    envelope = _envelope(
        system=10,
        history=[_message("user", 10), _message("assistant", 10)],
        current=10,
        tools=[{"tokens": 10}],
        attachments=[{"tokens": 10}],
    )

    result = await service.preflight(envelope, _config())

    assert result.available_input_tokens == 100
    assert result.input_tokens == 60
    assert result.usage_ratio == 0.60
    assert result.action == "proceed"
    assert result.evidence_token_allowance == 25


@pytest.mark.asyncio
async def test_current_question_and_evidence_group_are_fixed_input() -> None:
    service = RequestBudgetService(WeightedCounter())
    current = (
        HumanMessage(content="question", additional_kwargs={"tokens": 20}),
        AIMessage(
            content="",
            tool_calls=[{"id": "e1", "name": "search_documents", "args": {}}],
            additional_kwargs={"tokens": 20},
        ),
        ToolMessage(
            content="bounded evidence",
            tool_call_id="e1",
            additional_kwargs={"tokens": 20},
        ),
    )

    result = await service.preflight(
        RequestEnvelope(
            provider="gemini",
            model="gemini-2.5-flash",
            system_messages=(_message("system", 10),),
            history_messages=(
                _message("user", 30, "old"),
                _message("assistant", 30, "old answer"),
            ),
            current_messages=current,
        ),
        _config(),
    )

    assert result.action == "reduced"
    assert result.envelope.history_messages == ()
    assert result.envelope.current_messages == current
    assert result.input_tokens == 70
    assert result.evidence_token_allowance == 15


@pytest.mark.asyncio
async def test_near_boundary_uses_authoritative_provider_count() -> None:
    service = RequestBudgetService(NativeAwareCounter())
    envelope = _envelope(
        system=20,
        history=[_message("user", 15), _message("assistant", 15)],
        current=15,
    )

    result = await service.preflight(envelope, _config())

    assert result.action == "durable_requested"
    assert result.input_tokens == 75
    assert result.count_strategy == "gemini:native_count"


@pytest.mark.asyncio
async def test_evidence_allowance_requests_authoritative_count_below_soft_boundary() -> None:
    service = RequestBudgetService(NativeAwareCounter())
    envelope = RequestEnvelope(
        provider="gemini",
        model="gemini-2.5-flash",
        system_messages=(_message("system", 5),),
        history_messages=(),
        current_messages=(_message("user", 5),),
        authoritative_allowance=True,
    )

    result = await service.preflight(envelope, _config())

    assert result.input_tokens == 75
    assert result.evidence_token_allowance == 10
    assert result.count_strategy == "gemini:native_count"


@pytest.mark.asyncio
async def test_evidence_allowance_reserves_actual_assistant_and_all_tool_wrappers() -> None:
    service = RequestBudgetService(WeightedCounter())
    result = await service.preflight(
        RequestEnvelope(
            provider="gemini",
            model="gemini-2.5-flash",
            system_messages=(_message("system", 10),),
            history_messages=(),
            current_messages=(_message("user", 10),),
            authoritative_allowance=True,
        ),
        _config(),
    )
    assistant = AIMessage(
        content="searching",
        tool_calls=[
            {"id": "s1", "name": "search_documents", "args": {}},
            {"id": "s2", "name": "search_documents", "args": {}},
        ],
        additional_kwargs={"tokens": 11},
    )

    reserved = await service.reserve_tool_result_envelopes(
        result,
        assistant_message=assistant,
        tool_messages=(
            ToolMessage(content="", tool_call_id="s1", additional_kwargs={"tokens": 7}),
            ToolMessage(content="", tool_call_id="s2", additional_kwargs={"tokens": 7}),
        ),
    )

    assert reserved.input_tokens == 45
    assert reserved.evidence_token_allowance == 40


@pytest.mark.asyncio
async def test_gemini_native_count_uses_provider_request_structure_without_blocking() -> None:
    sdk_calls: list[dict] = []
    prepared_calls: list[dict] = []
    sync_calls: list[str] = []

    class AsyncModels:
        async def count_tokens(self, **kwargs):
            sdk_calls.append(kwargs)
            await asyncio.sleep(0)
            return SimpleNamespace(total_tokens=37)

    class FakeGeminiModel:
        async_client = SimpleNamespace(models=AsyncModels())

        def get_num_tokens(self, text: str) -> int:
            sync_calls.append(text)
            return 999

        def _prepare_request(self, messages, *, tools=None, **_kwargs):
            prepared_calls.append({"messages": messages, "tools": tools})
            return {
                "model": "models/gemini-2.5-flash",
                "contents": ("provider-user-content", "provider-function-response"),
                "config": SimpleNamespace(
                    system_instruction="provider-system-instruction",
                    tools=("provider-function-declarations",),
                ),
            }

    messages = [
        SystemMessage(content="system"),
        HumanMessage(
            content=[
                {"type": "text", "text": "inspect"},
                {"type": "image_url", "image_url": "data:image/png;base64,AAAA"},
            ]
        ),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "inspect", "args": {}}],
        ),
        ToolMessage(content="done", tool_call_id="call-1", name="inspect"),
    ]
    tools = [{"name": "inspect", "description": "Inspect an image"}]
    counter = BaseAgent._token_counter_for_model("gemini", FakeGeminiModel())
    heartbeat_seen = asyncio.Event()

    async def heartbeat() -> None:
        heartbeat_seen.set()

    asyncio.create_task(heartbeat())

    result = await counter.count_request(
        provider="gemini",
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools,
        authoritative=True,
    )

    assert heartbeat_seen.is_set(), "native counting must yield instead of blocking the event loop"
    assert sync_calls == []
    assert result.input_tokens == 37
    assert prepared_calls == [{"messages": messages, "tools": tools}]
    assert sdk_calls == [
        {
            "model": "models/gemini-2.5-flash",
            "contents": ("provider-user-content", "provider-function-response"),
            "config": {
                "system_instruction": "provider-system-instruction",
                "tools": ("provider-function-declarations",),
            },
        }
    ]


@pytest.mark.asyncio
async def test_hard_budget_reduction_bounds_native_reconciliation_calls() -> None:
    class CountingNativeCounter(WeightedCounter):
        def __init__(self) -> None:
            self.native_calls = 0

        async def count_request(self, **kwargs):
            self.native_calls += 1
            await asyncio.sleep(0)
            return self.estimate_request(**kwargs)

    counter = CountingNativeCounter()
    history = [
        message
        for turn in range(10)
        for message in (
            _message("user", 12, f"question {turn}"),
            _message("assistant", 12, f"answer {turn}"),
        )
    ]
    envelope = _envelope(system=10, history=history, current=10)

    result = await RequestBudgetService(counter).preflight(
        envelope,
        _config(max_input_tokens=200, reserved_output_tokens=0, safety_margin_tokens=0),
    )

    assert result.input_tokens <= result.hard_input_tokens
    assert counter.native_calls <= 2


@pytest.mark.asyncio
async def test_soft_to_hard_requests_durable_compaction_and_proceeds() -> None:
    requested = []
    service = RequestBudgetService(WeightedCounter())
    envelope = _envelope(history=[_message("user", 25), _message("assistant", 25)])

    result = await service.preflight(
        envelope,
        _config(),
        durable_request=lambda: requested.append("requested"),
    )

    assert result.input_tokens == 70
    assert result.action == "durable_requested"
    assert result.durable_requested is True
    assert requested == ["requested"]


@pytest.mark.asyncio
async def test_hard_budget_runs_bounded_compaction_then_recounts() -> None:
    calls = []
    service = RequestBudgetService(WeightedCounter())
    history = [
        _message("user", 35, "old"),
        _message("assistant", 35, "old answer"),
        _message("user", 5, "recent"),
        _message("assistant", 5, "recent answer"),
    ]

    async def compact(messages):
        calls.append(tuple(messages))
        return (_message("memory", 20, "compacted"), *messages[-2:])

    result = await service.preflight(
        _envelope(history=history),
        _config(),
        emergency_compact=compact,
    )

    assert calls
    assert result.action == "emergency_compacted"
    assert result.emergency_compacted is True
    assert result.input_tokens == 50


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "failure"])
async def test_compaction_timeout_or_failure_removes_oldest_complete_turns(mode) -> None:
    service = RequestBudgetService(WeightedCounter())
    history = [
        _message("user", 30, "turn one"),
        _message("assistant", 30, "answer one"),
        _message("user", 20, "turn two"),
        _message("assistant", 20, "answer two"),
    ]

    async def compact(_messages):
        if mode == "timeout":
            await asyncio.sleep(0.05)
        raise RuntimeError("provider failure with transcript details")

    result = await service.preflight(
        _envelope(history=history),
        _config(emergency_timeout_seconds=0.001),
        emergency_compact=compact,
    )

    assert result.action == "reduced"
    assert [message["content"] for message in result.envelope.history_messages] == [
        "turn two",
        "answer two",
    ]
    assert result.removed_groups == 1


@pytest.mark.asyncio
async def test_reduction_never_splits_tool_call_result_or_user_assistant_turn() -> None:
    service = RequestBudgetService(WeightedCounter())
    tool_turn = [
        _message("user", 20, "question"),
        {"role": "assistant", "content": "", "tokens": 20, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "result", "tokens": 20, "tool_call_id": "c1"},
        _message("assistant", 20, "final"),
    ]
    recent = [_message("user", 10, "recent"), _message("assistant", 10, "recent answer")]

    result = await service.preflight(
        _envelope(history=[*tool_turn, *recent]),
        _config(),
    )

    assert result.action == "reduced"
    remaining = result.envelope.history_messages
    assert [message["content"] for message in remaining] == ["recent", "recent answer"]
    assert not any(message.get("tool_call_id") == "c1" for message in remaining)


@pytest.mark.asyncio
async def test_reduction_skips_incomplete_prefix_and_removes_complete_old_turn() -> None:
    service = RequestBudgetService(WeightedCounter())
    history = [
        {"role": "tool", "content": "orphan", "tokens": 10, "tool_call_id": "orphan"},
        _message("user", 30, "old question"),
        _message("assistant", 30, "old answer"),
        _message("user", 10, "recent question"),
        _message("assistant", 10, "recent answer"),
    ]

    result = await service.preflight(_envelope(history=history), _config())

    assert result.action == "reduced"
    assert [message["content"] for message in result.envelope.history_messages] == [
        "orphan",
        "recent question",
        "recent answer",
    ]


@pytest.mark.asyncio
async def test_fixed_input_overflow_returns_specific_safe_error() -> None:
    service = RequestBudgetService(WeightedCounter())

    result = await service.preflight(
        _envelope(system=70, current=30, tools=[{"tokens": 10}]),
        _config(),
    )

    assert result.action == "error"
    assert result.error_code == "context_budget_fixed_input_exceeded"
    assert "transcript details" not in result.error_code


class _BoundaryAgent(BaseAgent):
    def _init_gemini(self) -> None:
        self.gemini_client = None
        self.langchain_model = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "budget_boundary_agent"

    def _get_base_system_prompt(self) -> str:
        return "Answer the current request."


class _CapturingModel:
    def __init__(self) -> None:
        self.calls: list[list] = []

    async def ainvoke(self, messages, _config=None):
        self.calls.append(list(messages))
        return AIMessage(content="within budget")


class _OverflowModel:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _messages, _config=None):
        self.calls += 1
        raise RuntimeError("maximum context length exceeded with private provider data")


def _runtime_with_limit(max_input_tokens: int) -> ResolvedRuntimeModelConfig:
    return ResolvedRuntimeModelConfig(
        agent_key="chat",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0,
        api_key=None,
        key_source="none",
        source="test",
        context_window={
            "max_input_tokens": max_input_tokens,
            "context_window_tokens": max_input_tokens,
            "max_output_tokens": 100,
            "known": True,
            "source": "test",
        },
    )


def _configure_boundary_agent(monkeypatch, *, max_input_tokens: int):
    agent = _BoundaryAgent(agent_config_key="chat")
    model = _CapturingModel()
    monkeypatch.setattr(settings, "conversation_summary_default_reserved_output_tokens", 100)
    monkeypatch.setattr(settings, "conversation_summary_safety_margin_tokens", 50)
    monkeypatch.setattr(settings, "conversation_summary_soft_context_ratio", 0.70)
    monkeypatch.setattr(settings, "conversation_summary_hard_context_ratio", 0.85)
    monkeypatch.setattr(settings, "conversation_summary_timeout_seconds", 1)
    monkeypatch.setattr(agent, "_init_tools", AsyncMock())
    monkeypatch.setattr(
        agent,
        "_resolve_runtime_model_config",
        lambda *_args, **_kwargs: _runtime_with_limit(max_input_tokens),
    )
    monkeypatch.setattr(
        agent,
        "_create_langchain_model_from_runtime",
        lambda *_args, **_kwargs: (model, False),
    )
    monkeypatch.setattr(
        agent,
        "_get_llm_with_tools",
        Mock(side_effect=AssertionError("tools disabled")),
    )
    monkeypatch.setattr(
        agent,
        "_get_tools_for_binding",
        Mock(side_effect=AssertionError("tools disabled")),
    )
    return agent, model


class _GeminiCapturingModel:
    """A langchain-model-shaped fake that is also a Gemini SDK counter target.

    ``ainvoke`` lets it serve as the model the agent actually calls;
    ``_prepare_request``/``async_client`` let ``BaseAgent._token_counter_for_model``
    recognize it as Gemini and register a native counter against it.
    """

    def __init__(self) -> None:
        self.calls: list[list] = []
        self.native_calls: list[dict] = []

    async def ainvoke(self, messages, _config=None):
        self.calls.append(list(messages))
        return AIMessage(content="within budget")

    async def _count_tokens(self, **kwargs):
        self.native_calls.append(kwargs)
        return SimpleNamespace(total_tokens=5)

    @property
    def async_client(self):
        return SimpleNamespace(models=SimpleNamespace(count_tokens=self._count_tokens))

    def _prepare_request(self, messages, *, tools=None, **_kwargs):
        del messages, tools
        return {
            "model": "models/gemini-2.5-flash",
            "contents": (),
            "config": SimpleNamespace(system_instruction="", tools=()),
        }


def _gemini_runtime_with_limit(max_input_tokens: int) -> ResolvedRuntimeModelConfig:
    return ResolvedRuntimeModelConfig(
        agent_key="chat",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0,
        api_key=None,
        key_source="none",
        source="test",
        context_window={
            "max_input_tokens": max_input_tokens,
            "context_window_tokens": max_input_tokens,
            "max_output_tokens": 100,
            "known": True,
            "source": "test",
        },
    )


@pytest.mark.asyncio
async def test_non_rag_preflight_uses_provider_counter_not_local_estimate(monkeypatch) -> None:
    """Item 1: the non-RAG chat preflight must reconcile Gemini's local byte-count
    estimate against the provider's own counter, the same way rag_agent.py:1054
    already does, instead of silently trusting the (now 3x larger) local bound.
    """
    agent = _BoundaryAgent(agent_config_key="chat")
    model = _GeminiCapturingModel()
    monkeypatch.setattr(settings, "conversation_summary_default_reserved_output_tokens", 10)
    monkeypatch.setattr(settings, "conversation_summary_safety_margin_tokens", 10)
    monkeypatch.setattr(settings, "conversation_summary_soft_context_ratio", 0.70)
    monkeypatch.setattr(settings, "conversation_summary_hard_context_ratio", 0.85)
    monkeypatch.setattr(settings, "conversation_summary_timeout_seconds", 1)
    monkeypatch.setattr(agent, "_init_tools", AsyncMock())
    monkeypatch.setattr(
        agent,
        "_resolve_runtime_model_config",
        lambda *_args, **_kwargs: _gemini_runtime_with_limit(6_000),
    )
    monkeypatch.setattr(
        agent,
        "_create_langchain_model_from_runtime",
        lambda *_args, **_kwargs: (model, False),
    )
    monkeypatch.setattr(
        agent,
        "_get_llm_with_tools",
        Mock(side_effect=AssertionError("tools disabled")),
    )
    monkeypatch.setattr(
        agent,
        "_get_tools_for_binding",
        Mock(side_effect=AssertionError("tools disabled")),
    )

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="current question " * 100)],
        conversation_history=[],
        persona=None,
        disable_tools=True,
    )

    assert response.message.content == "within budget"
    assert model.native_calls, (
        "preflight must escalate to Gemini's native count_tokens once the local "
        "estimate crosses the escalation ratio, not rely solely on the local "
        "utf8-byte upper bound"
    )


@pytest.mark.asyncio
async def test_base_agent_reduces_before_emitting_provider_request(monkeypatch) -> None:
    agent, model = _configure_boundary_agent(monkeypatch, max_input_tokens=1_000)
    old_text = "oldest persisted detail " * 450
    history = [
        AgentMessage(role=MessageRole.USER, content=old_text),
        AgentMessage(role=MessageRole.ASSISTANT, content=old_text),
        AgentMessage(role=MessageRole.USER, content="recent question"),
        AgentMessage(role=MessageRole.ASSISTANT, content="recent answer"),
    ]

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="current question")],
        conversation_history=history,
        persona=None,
        disable_tools=True,
    )

    assert response.message.content == "within budget"
    assert len(model.calls) == 1
    emitted = model.calls[0]
    counted = TokenCounter().estimate_request(
        provider="openai",
        model="gpt-4o-mini",
        messages=emitted,
    )
    assert counted.input_tokens <= int((1_000 - 100 - 50) * 0.85)
    assert all("oldest persisted detail" not in str(message.content) for message in emitted)
    assert response.metadata["request_budget"]["action"] == "reduced"


@pytest.mark.asyncio
async def test_base_agent_persists_public_thinking_under_trace_summary_key(monkeypatch) -> None:
    agent, model = _configure_boundary_agent(monkeypatch, max_input_tokens=10_000)

    async def response_with_public_thinking(_messages, _config=None):
        return AIMessage(
            content=[
                {"type": "thinking", "thinking": "Checking constraints."},
                {"type": "text", "text": "Final answer"},
            ]
        )

    model.ainvoke = response_with_public_thinking

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="current question")],
        conversation_history=[],
        persona=None,
        disable_tools=True,
    )

    assert response.metadata["thinking_summary"] == "Checking constraints."
    assert "thinking" not in response.metadata


@pytest.mark.asyncio
async def test_base_preflight_builds_default_durable_and_emergency_callbacks(monkeypatch) -> None:
    agent, _model = _configure_boundary_agent(monkeypatch, max_input_tokens=1_000)
    events = []

    def build_callbacks(conversation_id, user_id):
        events.append(("built", conversation_id, user_id))

        def durable():
            events.append(("durable",))

        async def emergency(_history):
            events.append(("emergency",))
            return [HumanMessage(content="compacted memory")]

        return durable, emergency

    monkeypatch.setattr(agent, "_build_compaction_callbacks", build_callbacks)

    result = await agent._preflight_model_request(
        _runtime_with_limit(1_000),
        system_messages=[HumanMessage(content="system")],
        history_messages=[
            HumanMessage(content="old history " * 600),
            AIMessage(content="old answer " * 600),
        ],
        current_messages=[HumanMessage(content="current")],
        conversation_id="11111111-1111-1111-1111-111111111111",
        user_id="22222222-2222-2222-2222-222222222222",
    )

    assert result.action == "emergency_compacted"
    assert events == [
        (
            "built",
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
        ),
        ("durable",),
        ("emergency",),
    ]


@pytest.mark.asyncio
async def test_base_agent_never_emits_unreducible_fixed_request(monkeypatch) -> None:
    agent, model = _configure_boundary_agent(monkeypatch, max_input_tokens=1_000)

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="required current content " * 1_000)],
        conversation_history=[],
        persona=None,
        disable_tools=True,
    )

    assert model.calls == []
    assert response.error == "context_budget_fixed_input_exceeded"
    assert "too large" in response.message.content.lower()


@pytest.mark.asyncio
async def test_base_agent_surfaces_second_overflow_after_exactly_one_retry(
    monkeypatch,
) -> None:
    agent, _model = _configure_boundary_agent(monkeypatch, max_input_tokens=10_000)
    overflow_model = _OverflowModel()
    monkeypatch.setattr(
        agent,
        "_create_langchain_model_from_runtime",
        lambda *_args, **_kwargs: (overflow_model, False),
    )

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="current question")],
        conversation_history=[],
        persona=None,
        disable_tools=True,
    )

    assert overflow_model.calls == 2
    assert response.error == "provider_context_overflow"
    assert "private provider data" not in response.model_dump_json()
