"""Task 7 — per-user attribution of model usage across the workflow.

These tests verify that every real provider attempt made by the workflow, the
router, title generation, and suggestion generation is recorded exactly once,
attributed to the correct user/conversation/request-message via the bound
``UsageContext``, and grouped under one ``operation_id`` per logical
invocation. A later tool-loop invocation (a new call to
``invoke_model_with_history``) must start a new operation, and deterministic
short-circuits that make no provider call must record zero events.

The tests wire a real :class:`ModelUsageRecorder` to an in-memory spy
repository so they can assert on the exact :class:`RecordEventCommand`s the
recorder would persist — the same seam the recorder's own tests use.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from prometheus_client import CollectorRegistry

from app.ai.agents.base_agent import BaseAgent
from app.ai.agents.router import Router
from app.ai.schemas import AgentMessage, AgentType, MessageRole
from app.core.config import settings
from app.core.runtime_modeling import ResolvedRuntimeModelConfig, RuntimeFallbackConfig
from app.observability.model_usage import ModelUsageMetrics
from app.repositories.model_usage import RecordEventCommand, RecordResult
from app.services.event_streaming.events import make_event
from app.usage import (
    UsageContext,
    bind_usage_context,
    current_usage_context,
)
from app.usage.recorder import ModelUsageRecorder

# --------------------------------------------------------------------------
# Spies and doubles
# --------------------------------------------------------------------------


class SpyRepository:
    """In-memory double capturing every recorded command."""

    def __init__(self) -> None:
        self.commands: list[RecordEventCommand] = []

    def record_event(self, command: RecordEventCommand) -> RecordResult:
        self.commands.append(command)
        return RecordResult(inserted=True, event_id=uuid4())


def make_recorder() -> tuple[ModelUsageRecorder, SpyRepository]:
    repo = SpyRepository()
    recorder = ModelUsageRecorder(
        repository=repo,
        enqueue_failed_write=lambda payload: None,
        metrics=ModelUsageMetrics(registry=CollectorRegistry()),
    )
    return recorder, repo


class FakeLLM:
    """Async model double whose per-call behaviour is scripted."""

    def __init__(self, behaviors: list[tuple[str, object]]) -> None:
        self._behaviors = list(behaviors)
        self.calls = 0

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        kind, payload = self._behaviors.pop(0)
        if kind == "raise":
            raise payload
        return payload


class _FakeGenResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, response=None, raises: Exception | None = None) -> None:
        self._response = response
        self._raises = raises
        self.calls = 0

    def generate_content(self, **_kwargs):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._response


class FakeGeminiClient:
    def __init__(self, response=None, raises: Exception | None = None) -> None:
        self.models = _FakeModels(response=response, raises=raises)


class StubAgent(BaseAgent):
    """Concrete BaseAgent whose provider setup is stubbed out per-test."""

    agent_type = AgentType.CHAT
    agent_id = "chat_agent"

    def _init_gemini(self) -> None:
        self.gemini_client = None
        self.langchain_model = None

    def _get_base_system_prompt(self) -> str:
        return "system"


def make_agent(
    *,
    recorder: ModelUsageRecorder | None,
    fake_llm: FakeLLM,
    runtime_config: ResolvedRuntimeModelConfig,
) -> StubAgent:
    agent = StubAgent(recorder=recorder)

    async def _init_tools() -> None:
        return None

    async def _preflight(*_args, **_kwargs):
        return None

    agent._init_tools = _init_tools  # type: ignore[method-assign]
    agent._resolve_runtime_model_config = (  # type: ignore[method-assign]
        lambda user_id, model_request=None: runtime_config
    )
    agent._create_langchain_model_from_runtime = (  # type: ignore[method-assign]
        lambda rc, **_kwargs: (fake_llm, False)
    )
    agent._get_llm_with_tools = lambda llm, **_kwargs: llm  # type: ignore[method-assign]
    agent._get_tools_for_binding = lambda **_kwargs: []  # type: ignore[method-assign]
    agent._preflight_model_request = _preflight  # type: ignore[method-assign]
    agent._build_system_prompt = lambda *a, **k: "system"  # type: ignore[method-assign]
    return agent


def gemini_runtime(*, with_fallback: bool = False) -> ResolvedRuntimeModelConfig:
    fallback = (
        RuntimeFallbackConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=1.0,
            api_key="fallback-key",
            key_source="test",
        )
        if with_fallback
        else None
    )
    return ResolvedRuntimeModelConfig(
        agent_key="chat",
        provider="gemini",
        model="gemini-3-flash-preview",
        temperature=1.0,
        api_key="primary-key",
        key_source="test",
        source="test",
        fallback_config=fallback,
    )


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(settings, "provider_retry_delay_seconds", 0.0)
    monkeypatch.setattr(settings, "provider_retry_attempts", 3)


# --------------------------------------------------------------------------
# Step 3 — provider-attempt wrapping in BaseAgent._ainvoke_with_retries
# --------------------------------------------------------------------------


async def test_generic_retries_record_one_attempt_each_under_one_operation():
    recorder, repo = make_recorder()
    fake_llm = FakeLLM(
        [
            ("raise", RuntimeError("boom-1")),
            ("raise", RuntimeError("boom-2")),
            ("ok", AIMessage(content="ok")),
        ]
    )
    agent = make_agent(recorder=recorder, fake_llm=fake_llm, runtime_config=gemini_runtime())

    from app.usage import begin_usage_operation

    with (
        bind_usage_context(UsageContext(user_id=uuid4(), operation="workflow")),
        begin_usage_operation() as op,
    ):
        response = await agent._ainvoke_with_retries(
            fake_llm,
            [HumanMessage(content="hi")],
            operation=op,
            provider="gemini",
            model="gemini-3-flash-preview",
        )

    assert isinstance(response, AIMessage)
    assert fake_llm.calls == 3
    assert len(repo.commands) == 3
    assert {c.operation_id for c in repo.commands} == {op.operation_id}
    assert [c.attempt for c in repo.commands] == [1, 2, 3]
    assert [c.status for c in repo.commands] == ["error", "error", "success"]
    assert all(c.provider == "gemini" for c in repo.commands)
    assert all(c.model == "gemini-3-flash-preview" for c in repo.commands)
    assert all(c.context.operation == "workflow" for c in repo.commands)


async def test_cancelled_provider_call_records_cancelled_attempt_and_reraises():
    recorder, repo = make_recorder()
    fake_llm = FakeLLM([("raise", asyncio.CancelledError())])
    agent = make_agent(recorder=recorder, fake_llm=fake_llm, runtime_config=gemini_runtime())

    from app.usage import begin_usage_operation

    with (
        bind_usage_context(UsageContext(user_id=uuid4(), operation="workflow")),
        begin_usage_operation() as op,
        pytest.raises(asyncio.CancelledError),
    ):
        await agent._ainvoke_with_retries(
            fake_llm,
            [HumanMessage(content="hi")],
            operation=op,
            provider="gemini",
            model="gemini-3-flash-preview",
        )

    assert len(repo.commands) == 1
    assert repo.commands[0].status == "cancelled"


async def test_recorder_absent_behaves_exactly_as_today():
    fake_llm = FakeLLM([("ok", AIMessage(content="ok"))])
    agent = make_agent(recorder=None, fake_llm=fake_llm, runtime_config=gemini_runtime())

    # No recorder and no operation: identical to the pre-instrumentation path.
    response = await agent._ainvoke_with_retries(fake_llm, [HumanMessage(content="hi")])

    assert isinstance(response, AIMessage)
    assert fake_llm.calls == 1


# --------------------------------------------------------------------------
# Step 3 — one operation spans context-overflow retry and provider fallback
# --------------------------------------------------------------------------


async def _run_invoke(agent: StubAgent, *, user_id: UUID, conversation_id: UUID, request_id: UUID):
    context = UsageContext(
        user_id=user_id,
        conversation_id=conversation_id,
        request_message_id=request_id,
        operation="workflow",
    )
    with bind_usage_context(context):
        return await agent.invoke_model_with_history(
            messages=[HumanMessage(content="hi")],
            conversation_history=[],
            persona=None,
            conversation_id=str(conversation_id),
            user_id=str(user_id),
        )


async def test_context_overflow_retry_shares_operation_with_first_attempt(monkeypatch):
    monkeypatch.setattr(settings, "context_overflow_retry_enabled", True)
    recorder, repo = make_recorder()
    fake_llm = FakeLLM(
        [
            ("raise", RuntimeError("maximum context length exceeded")),
            ("ok", AIMessage(content="ok")),
        ]
    )
    agent = make_agent(recorder=recorder, fake_llm=fake_llm, runtime_config=gemini_runtime())

    user_id, conversation_id, request_id = uuid4(), uuid4(), uuid4()
    await _run_invoke(
        agent, user_id=user_id, conversation_id=conversation_id, request_id=request_id
    )

    assert len(repo.commands) == 2
    assert len({c.operation_id for c in repo.commands}) == 1
    assert [c.attempt for c in repo.commands] == [1, 2]
    assert [c.status for c in repo.commands] == ["error", "success"]
    assert all(c.provider == "gemini" for c in repo.commands)
    # Attribution flows from the bound context, not from arguments to the recorder.
    for command in repo.commands:
        assert command.context.user_id == user_id
        assert command.context.conversation_id == conversation_id
        assert command.context.request_message_id == request_id
        assert command.context.operation == "workflow"
        assert command.context.agent_id == "chat_agent"


async def test_provider_fallback_records_under_shared_operation_with_new_provider(monkeypatch):
    monkeypatch.setattr(settings, "context_overflow_retry_enabled", True)
    recorder, repo = make_recorder()
    fake_llm = FakeLLM(
        [
            ("raise", RuntimeError("primary down 1")),
            ("raise", RuntimeError("primary down 2")),
            ("raise", RuntimeError("primary down 3")),
            ("ok", AIMessage(content="fallback ok")),
        ]
    )
    agent = make_agent(
        recorder=recorder,
        fake_llm=fake_llm,
        runtime_config=gemini_runtime(with_fallback=True),
    )

    await _run_invoke(agent, user_id=uuid4(), conversation_id=uuid4(), request_id=uuid4())

    assert len(repo.commands) == 4
    assert len({c.operation_id for c in repo.commands}) == 1
    assert [c.attempt for c in repo.commands] == [1, 2, 3, 4]
    assert [c.provider for c in repo.commands] == ["gemini", "gemini", "gemini", "openai"]
    assert [c.model for c in repo.commands] == [
        "gemini-3-flash-preview",
        "gemini-3-flash-preview",
        "gemini-3-flash-preview",
        "gpt-4o-mini",
    ]
    assert [c.status for c in repo.commands] == ["error", "error", "error", "success"]


async def test_later_tool_loop_invocation_starts_a_new_operation():
    recorder, repo = make_recorder()
    fake_llm = FakeLLM([("ok", AIMessage(content="one")), ("ok", AIMessage(content="two"))])
    agent = make_agent(recorder=recorder, fake_llm=fake_llm, runtime_config=gemini_runtime())

    user_id, conversation_id, request_id = uuid4(), uuid4(), uuid4()
    await _run_invoke(
        agent, user_id=user_id, conversation_id=conversation_id, request_id=request_id
    )
    await _run_invoke(
        agent, user_id=user_id, conversation_id=conversation_id, request_id=request_id
    )

    assert len(repo.commands) == 2
    assert len({c.operation_id for c in repo.commands}) == 2
    # Each fresh invocation restarts attempt numbering at 1.
    assert [c.attempt for c in repo.commands] == [1, 1]


async def test_concurrent_users_keep_separate_attribution():
    recorder, repo = make_recorder()

    async def one_user() -> UUID:
        user_id = uuid4()
        fake_llm = FakeLLM([("ok", AIMessage(content="ok"))])
        agent = make_agent(recorder=recorder, fake_llm=fake_llm, runtime_config=gemini_runtime())
        await _run_invoke(agent, user_id=user_id, conversation_id=uuid4(), request_id=uuid4())
        return user_id

    user_ids = await asyncio.gather(*(one_user() for _ in range(6)))

    recorded_users = {c.context.user_id for c in repo.commands}
    assert recorded_users == set(user_ids)
    assert len(repo.commands) == 6


# --------------------------------------------------------------------------
# Step 5 — router instrumentation
# --------------------------------------------------------------------------


async def _make_router(monkeypatch, recorder, *, client) -> Router:
    monkeypatch.setattr(Router, "_init_gemini", lambda self: None)
    router = Router(recorder=recorder)
    router.gemini_client = client
    return router


async def test_router_records_one_success_attempt(monkeypatch):
    recorder, repo = make_recorder()
    router = await _make_router(
        monkeypatch, recorder, client=FakeGeminiClient(response=_FakeGenResponse("search_agent"))
    )

    with bind_usage_context(UsageContext(user_id=uuid4(), operation="workflow")):
        result = await router.route_message(
            AgentMessage(role=MessageRole.USER, content="find the news"),
            ["chat_agent", "search_agent"],
        )

    assert result == "search_agent"
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.provider == "gemini"
    assert command.model == router.model_name
    assert command.context.operation == "router"
    assert command.context.agent_id == "router"
    assert command.status == "success"


async def test_router_provider_error_records_error_attempt_and_defaults(monkeypatch):
    recorder, repo = make_recorder()
    router = await _make_router(
        monkeypatch, recorder, client=FakeGeminiClient(raises=RuntimeError("gemini down"))
    )

    with bind_usage_context(UsageContext(user_id=uuid4(), operation="workflow")):
        result = await router.route_message(
            AgentMessage(role=MessageRole.USER, content="find the news"),
            ["chat_agent", "search_agent"],
        )

    assert result == "chat_agent"
    assert len(repo.commands) == 1
    assert repo.commands[0].status == "error"
    assert repo.commands[0].context.operation == "router"


async def test_router_deterministic_shortcircuits_record_zero_events(monkeypatch):
    recorder, repo = make_recorder()

    # No available agents at all: returns immediately, no provider call.
    router = await _make_router(
        monkeypatch, recorder, client=FakeGeminiClient(response=_FakeGenResponse("x"))
    )
    with bind_usage_context(UsageContext(user_id=uuid4(), operation="workflow")):
        result = await router.route_message(AgentMessage(role=MessageRole.USER, content="hi"), [])
    assert result == "chat_agent"

    # No client configured: deterministic fallback, still no provider call.
    router_no_client = await _make_router(monkeypatch, recorder, client=None)
    with bind_usage_context(UsageContext(user_id=uuid4(), operation="workflow")):
        assert (
            await router_no_client.route_message(
                AgentMessage(role=MessageRole.USER, content="hi"),
                ["chat_agent", "search_agent"],
            )
            == "chat_agent"
        )

    assert repo.commands == []


# --------------------------------------------------------------------------
# Step 4 — title generation instrumentation (AIService method)
# --------------------------------------------------------------------------


def _title_service(recorder):
    from app.services.ai_service import AIService

    service = AIService.__new__(AIService)
    service.workflow = SimpleNamespace(model_usage_recorder=recorder)
    return service


async def test_generate_title_records_one_attempt(monkeypatch):
    recorder, repo = make_recorder()
    service = _title_service(recorder)

    fake_llm = FakeLLM([("ok", AIMessage(content="A Concise Title"))])
    monkeypatch.setattr(
        "app.ai.agent_config.create_langchain_model",
        lambda **_kwargs: fake_llm,
    )

    user_id, conversation_id = uuid4(), uuid4()
    title = await service.generate_conversation_title(
        "Tell me about the history of Rome",
        user_id=user_id,
        conversation_id=conversation_id,
    )

    assert title == "A Concise Title"
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.provider == "gemini"
    assert command.context.operation == "title_generation"
    assert command.context.agent_id == "title_generator"
    assert command.context.user_id == user_id
    assert command.context.conversation_id == conversation_id
    assert command.status == "success"


async def test_generate_title_records_error_attempt_but_still_falls_back(monkeypatch):
    recorder, repo = make_recorder()
    service = _title_service(recorder)

    fake_llm = FakeLLM([("raise", RuntimeError("title model down"))])
    monkeypatch.setattr(
        "app.ai.agent_config.create_langchain_model",
        lambda **_kwargs: fake_llm,
    )

    long_message = "x" * 80
    title = await service.generate_conversation_title(long_message, user_id=uuid4())

    # Fallback to a truncated title still happens even though the call failed.
    assert title.endswith("...")
    assert len(repo.commands) == 1
    assert repo.commands[0].status == "error"
    assert repo.commands[0].context.operation == "title_generation"


async def test_generate_title_without_recorder_still_works(monkeypatch):
    service = _title_service(None)
    fake_llm = FakeLLM([("ok", AIMessage(content="No Recorder Title"))])
    monkeypatch.setattr(
        "app.ai.agent_config.create_langchain_model",
        lambda **_kwargs: fake_llm,
    )

    title = await service.generate_conversation_title("hello", user_id=uuid4())
    assert title == "No Recorder Title"


# --------------------------------------------------------------------------
# Step 4 — suggestion generation instrumentation
# --------------------------------------------------------------------------


async def test_suggestions_record_one_attempt_and_skip_on_cache_hit():
    from app.ai.suggestion_generator import SuggestionGenerator

    recorder, repo = make_recorder()
    generator = SuggestionGenerator.__new__(SuggestionGenerator)
    generator.model_name = "gemini-3-flash-preview"
    generator._suggestion_cache = {}
    generator.client = FakeGeminiClient(response=_FakeGenResponse('["What next?", "Why?"]'))

    usage_context = UsageContext(
        user_id=uuid4(),
        conversation_id=uuid4(),
        request_message_id=uuid4(),
        operation="suggestions",
        agent_id="suggestion_generator",
    )

    first = await generator.generate_suggestions(
        "user query",
        "assistant response",
        usage_context=usage_context,
        recorder=recorder,
    )
    assert first == ["What next?", "Why?"]
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.provider == "gemini"
    assert command.context.operation == "suggestions"
    assert command.context.agent_id == "suggestion_generator"

    # Identical inputs hit the in-memory cache: no second provider call recorded.
    second = await generator.generate_suggestions(
        "user query",
        "assistant response",
        usage_context=usage_context,
        recorder=recorder,
    )
    assert second == ["What next?", "Why?"]
    assert len(repo.commands) == 1


async def test_suggestions_without_recorder_behave_as_today():
    from app.ai.suggestion_generator import SuggestionGenerator

    generator = SuggestionGenerator.__new__(SuggestionGenerator)
    generator.model_name = "gemini-3-flash-preview"
    generator._suggestion_cache = {}
    generator.client = FakeGeminiClient(response=_FakeGenResponse("[]"))

    result = await generator.generate_suggestions("q", "r")
    assert result == []


# --------------------------------------------------------------------------
# Step 2 — AI service boundary binds attribution context
# --------------------------------------------------------------------------


class CtxCapturingWorkflow:
    def __init__(self) -> None:
        self.captured: UsageContext | None = None
        self.model_usage_recorder = None

    async def execute_request(self, _ai_request):
        self.captured = current_usage_context()
        return None

    async def execute_request_stream(self, _ai_request):
        self.captured = current_usage_context()
        yield make_event("complete", sequence=0, data={"response": None})

    async def resume(self, **_kwargs):
        self.captured = current_usage_context()
        return None

    async def resume_with_decisions_stream(self, **_kwargs):
        self.captured = current_usage_context()
        yield make_event("complete", sequence=0, data={"response": None})


def _ai_service(workflow, *, checkpointer=None):
    from app.services.ai_service import AIService

    service = AIService.__new__(AIService)
    service.workflow = workflow
    service.checkpointer = checkpointer
    service.conversation_repository = None
    service._prepare_request = lambda request: request
    service._to_ai_request = lambda request: request
    service._to_ai_decisions = lambda decisions: decisions
    return service


def _workflow_request(*, user_id, conversation_id, user_message_id, thread_id="thread-x"):
    from app.schemas.workflow import WorkflowExecutionRequest

    return WorkflowExecutionRequest(
        message="hello",
        conversation_id=str(conversation_id),
        user_id=str(user_id),
        user_message_id=str(user_message_id),
        thread_id=thread_id,
    )


async def test_execute_request_binds_workflow_usage_context():
    workflow = CtxCapturingWorkflow()
    service = _ai_service(workflow)
    user_id, conversation_id, user_message_id = uuid4(), uuid4(), uuid4()

    await service.execute_request(
        _workflow_request(
            user_id=user_id, conversation_id=conversation_id, user_message_id=user_message_id
        )
    )

    captured = workflow.captured
    assert captured is not None
    assert captured.user_id == user_id
    assert captured.conversation_id == conversation_id
    assert captured.request_message_id == user_message_id
    assert captured.correlation_id == "thread-x"
    assert captured.operation == "workflow"


async def test_execute_request_stream_binds_context_for_whole_body():
    workflow = CtxCapturingWorkflow()
    service = _ai_service(workflow)
    user_id, conversation_id, user_message_id = uuid4(), uuid4(), uuid4()

    events = [
        event
        async for event in service.execute_request_stream(
            _workflow_request(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message_id=user_message_id,
            )
        )
    ]

    assert events  # stream produced at least the terminal complete event
    captured = workflow.captured
    assert captured.user_id == user_id
    assert captured.request_message_id == user_message_id
    assert captured.operation == "workflow"


async def test_execute_request_stream_can_close_from_another_task_context():
    closed = asyncio.Event()
    captured: list[UsageContext] = []

    async def source(_request):
        try:
            captured.append(current_usage_context())
            yield make_event("message_delta", sequence=1, data={"content": "hello"})
            await asyncio.Event().wait()
        finally:
            captured.append(current_usage_context())
            closed.set()

    workflow = SimpleNamespace(execute_request_stream=source, model_usage_recorder=None)
    service = _ai_service(workflow)
    user_id, conversation_id, user_message_id = uuid4(), uuid4(), uuid4()
    stream = service.execute_request_stream(
        _workflow_request(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
        )
    )

    assert (await anext(stream)).type == "message_delta"
    await asyncio.create_task(stream.aclose())
    await asyncio.wait_for(closed.wait(), timeout=0.1)

    assert [context.operation for context in captured] == ["workflow", "workflow"]
    assert current_usage_context().operation == "unknown"


async def test_resume_workflow_rebuilds_ownership_from_arguments():
    workflow = CtxCapturingWorkflow()
    service = _ai_service(workflow, checkpointer=object())
    user_id, conversation_id = uuid4(), uuid4()

    await service.resume_workflow(conversation_id=conversation_id, user_id=user_id)

    captured = workflow.captured
    assert captured.user_id == user_id
    assert captured.conversation_id == conversation_id
    assert captured.operation == "workflow"
    # The reserved assistant id is never used as a request-message FK on resume.
    assert captured.request_message_id is None


async def test_resume_interrupted_stream_binds_context_from_authenticated_args():
    workflow = CtxCapturingWorkflow()
    service = _ai_service(workflow, checkpointer=object())
    user_id, conversation_id = uuid4(), uuid4()

    events = [
        event
        async for event in service.resume_interrupted_execution_stream(
            thread_id=str(conversation_id),
            decisions=[],
            user_id=user_id,
            conversation_id=conversation_id,
        )
    ]

    assert events
    captured = workflow.captured
    assert captured.user_id == user_id
    assert captured.conversation_id == conversation_id
    assert captured.operation == "workflow"


async def test_resume_interrupted_stream_can_close_from_another_task_context():
    closed = asyncio.Event()
    captured: list[UsageContext] = []

    async def source(**_kwargs):
        try:
            captured.append(current_usage_context())
            yield make_event("message_delta", sequence=1, data={"content": "hello"})
            await asyncio.Event().wait()
        finally:
            captured.append(current_usage_context())
            closed.set()

    workflow = SimpleNamespace(resume_with_decisions_stream=source, model_usage_recorder=None)
    service = _ai_service(workflow, checkpointer=object())
    user_id, conversation_id = uuid4(), uuid4()
    stream = service.resume_interrupted_execution_stream(
        thread_id=str(conversation_id),
        decisions=[],
        user_id=user_id,
        conversation_id=conversation_id,
    )

    assert (await anext(stream)).type == "message_delta"
    await asyncio.create_task(stream.aclose())
    await asyncio.wait_for(closed.wait(), timeout=0.1)

    assert [context.operation for context in captured] == ["workflow", "workflow"]
    assert current_usage_context().operation == "unknown"


async def test_execute_request_survives_non_uuid_identifiers():
    workflow = CtxCapturingWorkflow()
    service = _ai_service(workflow)
    from app.schemas.workflow import WorkflowExecutionRequest

    # Non-UUID identifiers must not crash the boundary; they bind as None.
    request = WorkflowExecutionRequest(
        message="hello",
        conversation_id="not-a-uuid",
        user_id="also-not-a-uuid",
        thread_id="thread-y",
    )

    await service.execute_request(request)

    captured = workflow.captured
    assert captured is not None
    assert captured.user_id is None
    assert captured.conversation_id is None
    assert captured.operation == "workflow"
