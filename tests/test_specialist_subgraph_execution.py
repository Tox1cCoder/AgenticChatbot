"""End-to-end proof that a standard specialist runs inside a real subgraph.

These drive a genuine ``create_agent`` graph with a deterministic fake chat
model. What they establish is that the framework loop — not a bespoke one —
produces the answer, runs tools, honours the call limits, and hands a
server-owned outcome back to the parent graph.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.ai.schemas import AgentType
from app.ai.workflow.contracts import ResponseOutcome, WorkerTask
from app.ai.workflow.specialists import (
    SpecialistDefinition,
    SpecialistFactory,
    SpecialistRequest,
)

pytestmark = pytest.mark.usefixtures("disable_langsmith_tracing")


@pytest.fixture
def disable_langsmith_tracing(monkeypatch):
    """Deterministic subgraph tests never emit traces."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed sequence of AI messages, tool calls included.

    ``GenericFakeChatModel`` cannot bind tools, and every interesting case here
    needs a tool-calling turn, so this fake implements the small surface
    ``create_agent`` actually uses.
    """

    responses: list[AIMessage] = []
    call_count: int = 0
    bound_tools: list = []

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _next(self) -> AIMessage:
        if self.call_count >= len(self.responses):
            raise AssertionError("scripted model exhausted: the loop ran longer than scripted")
        message = self.responses[self.call_count]
        self.call_count += 1
        return message

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next())])


def scripted_model(messages: list[AIMessage]) -> ScriptedChatModel:
    """A deterministic chat model that replays a fixed sequence of AI messages."""
    return ScriptedChatModel(responses=list(messages))


def _resolver():
    return SimpleNamespace(
        resolve_runtime_config=lambda *a, **k: SimpleNamespace(
            agent_key="chat",
            provider="gemini",
            model="gemini-3-flash-preview",
            temperature=1.0,
            api_key="key",
            key_source="user",
            source="default",
            warnings=[],
            capabilities={},
            fallback_config=None,
        )
    )


def _factory(model, tools=None, **overrides):
    definition = SpecialistDefinition(
        agent_id="chat_agent",
        agent_type=AgentType.CHAT,
        model_config_key="chat",
        system_prompt_factory=lambda request: "You are a helpful assistant.",
        tool_factory=lambda request: list(tools or []),
        output_policy_ids=("public_content",),
    )
    payload = {
        "definitions": {"chat_agent": definition},
        "runtime_model_resolver": _resolver(),
        "model_factory": SimpleNamespace(create_model_from_runtime=lambda config, **kw: model),
        "usage_recorder": None,
        "settings": SimpleNamespace(
            generation_hard_model_calls_per_epoch=4,
            generation_soft_model_calls_per_epoch=3,
            generation_hard_tool_calls_per_epoch=4,
            generation_soft_tool_calls_per_epoch=3,
        ),
    }
    payload.update(overrides)
    return SpecialistFactory(**payload)


def _request(**overrides):
    payload = {
        "agent_id": "chat_agent",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "persona": None,
        "model_request": None,
        "messages": [HumanMessage(content="what is 2+2?")],
        "history": [],
        "state": {},
    }
    payload.update(overrides)
    return SpecialistRequest(**payload)


async def test_specialist_answers_through_a_real_create_agent_subgraph():
    model = scripted_model([AIMessage(content="It is 4.")])
    outcome = await _factory(model).invoke(_request())

    assert isinstance(outcome, ResponseOutcome)
    assert outcome.agent_id == "chat_agent"
    assert outcome.response.message.content == "It is 4."
    assert outcome.provenance.output_policy_ids == ("public_content",)


async def test_specialist_runs_a_tool_and_then_answers():
    calls: list[dict] = []

    @tool
    def lookup_price(symbol: str) -> str:
        """Look up a ticker price."""
        calls.append({"symbol": symbol})
        return "42"

    model = scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "lookup_price", "args": {"symbol": "ACME"}}],
            ),
            AIMessage(content="ACME trades at 42."),
        ]
    )

    outcome = await _factory(model, tools=[lookup_price]).invoke(_request())

    assert calls == [{"symbol": "ACME"}]
    assert outcome.response.message.content == "ACME trades at 42."


async def test_intermediate_tool_turn_is_private_not_the_public_answer():
    @tool
    def noop() -> str:
        """Do nothing."""
        return "done"

    model = scripted_model(
        [
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "noop", "args": {}}]),
            AIMessage(content="All finished."),
        ]
    )
    outcome = await _factory(model, tools=[noop]).invoke(_request())

    assert outcome.response.message.content == "All finished."
    # The tool-calling turn and the ToolMessage stay in private provenance.
    private_types = {type(message).__name__ for message in outcome.provenance.private_messages}
    assert "ToolMessage" in private_types


async def test_a_hard_limit_becomes_a_server_owned_partial_not_an_error():
    """The framework ceiling means the soft budget failed to reserve an answer.

    Reporting ``agent_execution_limit`` to a client throws away everything the
    turn gathered and says nothing it can act on. What is still recoverable at
    this boundary is the artifacts and images the tool pipeline recorded plus
    the counters -- not the model's text, which died with the exception -- so
    the fallback says so plainly and stays continuable.
    """

    @tool
    def spin() -> str:
        """Always asks to be called again."""
        return "again"

    model = scripted_model(
        [
            AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "spin", "args": {}}])
            for i in range(10)
        ]
    )
    factory = _factory(
        model,
        tools=[spin],
        settings=SimpleNamespace(
            generation_hard_model_calls_per_epoch=2,
            generation_soft_model_calls_per_epoch=1,
            generation_hard_tool_calls_per_epoch=10,
            generation_soft_tool_calls_per_epoch=9,
        ),
    )

    outcome = await factory.invoke(_request())
    budget = outcome.response.metadata["execution_budget"]

    assert budget["exhausted_by"] == "hard_limit"
    assert budget["forced_synthesis"] is True
    assert outcome.response.message.content


async def test_the_hard_limit_partial_carries_what_the_pipeline_recorded():
    @tool
    def spin() -> str:
        """Always asks to be called again."""
        return "again"

    model = scripted_model(
        [
            AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "spin", "args": {}}])
            for i in range(10)
        ]
    )
    factory = _factory(
        model,
        tools=[spin],
        settings=SimpleNamespace(
            generation_hard_model_calls_per_epoch=2,
            generation_soft_model_calls_per_epoch=1,
            generation_hard_tool_calls_per_epoch=10,
            generation_soft_tool_calls_per_epoch=9,
        ),
    )

    outcome = await factory.invoke(_request())

    assert outcome.agent_id == "chat_agent"
    assert outcome.provenance.output_policy_ids


def _worker_task(task_id: str = "t1", agent_id: str = "chat_agent", **overrides) -> WorkerTask:
    """The server-owned identity a dispatched worker carries."""
    payload = {
        "dispatch_id": "d1",
        "task_id": task_id,
        "position": 0,
        "objective": f"do {task_id}",
        "agent_id": agent_id,
    }
    payload.update(overrides)
    return WorkerTask(**payload)


def _spinning_worker_factory():
    """A worker whose model never stops asking for the same tool.

    Its soft rung is one call below its hard rung, so the ceiling fires on the
    very call the soft budget reserved for an answer -- the case a delegated
    worker has to survive without taking the parent turn down with it.
    """

    @tool
    def spin() -> str:
        """Always asks to be called again."""
        return "again"

    model = scripted_model(
        [
            AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "spin", "args": {}}])
            for i in range(10)
        ]
    )
    return _factory(
        model,
        tools=[spin],
        settings=SimpleNamespace(
            generation_hard_model_calls_per_epoch=2,
            generation_soft_model_calls_per_epoch=1,
            generation_hard_tool_calls_per_epoch=10,
            generation_soft_tool_calls_per_epoch=9,
        ),
    )


async def test_a_worker_execution_limit_returns_a_partial_not_a_failure():
    """R1: only the top-level turn pauses; a worker hands its parent a partial.

    Reporting ``failed``/``agent_execution_limit`` here threw away everything
    the worker gathered *and* told the synthesizing parent to disregard it.
    ``partial`` is the honest status: the evidence is real, the work is not
    finished, and the parent decides what that is worth.
    """
    result = await _spinning_worker_factory().invoke_worker(_request(), task=_worker_task())

    assert result.status == "partial"
    assert result.content
    # `error_code` stays paired with `failed`. A partial is not an error, and
    # populating it here would render as one wherever a worker end is shown.
    assert result.error_code is None


async def test_a_worker_partial_keeps_the_identity_of_its_dispatched_task():
    """No dispatched task may be orphaned, whatever status answers it."""
    task = _worker_task(task_id="t9", agent_id="chat_agent", position=3)

    result = await _spinning_worker_factory().invoke_worker(_request(), task=task)

    assert (result.dispatch_id, result.task_id, result.position) == ("d1", "t9", 3)
    assert result.agent_id == "chat_agent"


async def test_a_worker_partial_carries_what_its_tool_pipeline_recorded():
    """The evidence is the whole point of not reporting a bare failure.

    ``_failed_worker`` sets no artifacts and no images, so the old mapping
    discarded every record the worker's tools had already written -- and
    ``build_planning_outcome`` aggregates those across results regardless of
    status, so they were lost from the parent's synthesis too.
    """
    factory = _spinning_worker_factory()
    recorded = [{"kind": "web_result", "url": "https://example.test/a"}]
    factory_build = factory._build

    async def build_with_artifacts(definition, request):
        agent, tool_execution, accountant = await factory_build(definition, request)
        tool_execution.artifacts.extend(recorded)
        return agent, tool_execution, accountant

    factory._build = build_with_artifacts

    result = await factory.invoke_worker(_request(), task=_worker_task())

    assert result.status == "partial"
    assert list(result.artifacts) == recorded


async def test_a_worker_partial_still_records_the_execution_limit_metric():
    """The rung firing is an operational signal even when the turn survives.

    Downgrading the status must not also downgrade the telemetry: a rising
    limit rate is how a soft rung set too close to its hard rung is noticed.
    """
    from app.observability.routing import get_routing_metrics_recorder

    recorder = get_routing_metrics_recorder()
    recorder.reset()
    try:
        await _spinning_worker_factory().invoke_worker(_request(), task=_worker_task())
        limit_counters = {
            key: value
            for key, value in recorder.counters.items()
            if key.startswith("execution.limit.")
        }
    finally:
        recorder.reset()

    assert limit_counters, "the execution-limit counter did not fire for a worker partial"


async def test_usage_is_recorded_for_every_model_turn_in_the_loop():
    recorded: list[tuple[str, str]] = []

    class Recorder:
        async def record_one_async_attempt(self, *, call, provider, model, operation):
            recorded.append((provider, model))
            return await call()

    @tool
    def noop() -> str:
        """Do nothing."""
        return "done"

    model = scripted_model(
        [
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "noop", "args": {}}]),
            AIMessage(content="Finished."),
        ]
    )
    await _factory(model, tools=[noop], usage_recorder=Recorder()).invoke(_request())

    assert recorded == [
        ("gemini", "gemini-3-flash-preview"),
        ("gemini", "gemini-3-flash-preview"),
    ]


async def test_history_is_sent_but_not_returned_as_produced_output():
    model = scripted_model([AIMessage(content="Answer.")])
    outcome = await _factory(model).invoke(
        _request(history=[HumanMessage(content="earlier"), AIMessage(content="earlier reply")])
    )

    assert outcome.response.message.content == "Answer."
    contents = [getattr(m, "content", "") for m in outcome.provenance.private_messages]
    assert "earlier reply" not in contents


async def test_the_reserved_answer_call_is_made_with_no_tools(monkeypatch):
    """R3, asserted after the whole assembled stack has run.

    The budget middleware cannot strip ``request.tools`` -- the execution
    middleware re-offers them after it -- so suppression happens in the tool
    factory, which that same middleware re-consults on every model call.

    The assertion is on the binding sequence rather than on the fake's
    ``bound_tools``, because ``create_agent`` calls ``bind_tools`` only when the
    tool list is non-empty and ``bind`` otherwise (``langchain/agents/factory``).
    So ``bind`` on the third call *is* the tool-free answer call -- and a fake
    whose ``bind_tools`` returns ``self`` would report the stale second binding
    forever.
    """

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return "evidence"

    model = scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {"query": "a"}, "id": "c1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {"query": "b"}, "id": "c2"}],
            ),
            AIMessage(content="final answer"),
        ]
    )
    sequence: list[str] = []
    bind_tools = type(model).bind_tools

    def spy_bind_tools(self, tools, **kwargs):
        sequence.append(f"bind_tools:{len(list(tools))}")
        return bind_tools(self, tools, **kwargs)

    def spy_bind(self, **kwargs):
        sequence.append("bind")
        return self

    monkeypatch.setattr(type(model), "bind_tools", spy_bind_tools)
    monkeypatch.setattr(type(model), "bind", spy_bind, raising=False)
    factory = _factory(
        model,
        tools=[lookup],
        settings=SimpleNamespace(
            generation_soft_tool_calls_per_epoch=1,
            generation_hard_tool_calls_per_epoch=3,
            generation_soft_model_calls_per_epoch=5,
            generation_hard_model_calls_per_epoch=6,
        ),
    )

    outcome = await factory.invoke(_request())

    assert sequence == ["bind_tools:1", "bind_tools:1", "bind"]
    assert outcome.response.message.content == "final answer"


async def test_the_outcome_reports_why_the_turn_stopped_gathering():
    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return "evidence"

    model = scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {"query": "a"}, "id": "c1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {"query": "b"}, "id": "c2"}],
            ),
            AIMessage(content="partial answer"),
        ]
    )
    factory = _factory(
        model,
        tools=[lookup],
        settings=SimpleNamespace(
            generation_soft_tool_calls_per_epoch=1,
            generation_hard_tool_calls_per_epoch=3,
            generation_soft_model_calls_per_epoch=5,
            generation_hard_model_calls_per_epoch=6,
        ),
    )

    outcome = await factory.invoke(_request())
    budget = outcome.response.metadata["execution_budget"]

    assert budget["exhausted_by"] == "tool_calls"
    assert budget["forced_synthesis"] is True
    assert budget["tool_calls"] == 1


async def test_a_turn_within_its_budget_reports_no_exhaustion():
    model = scripted_model([AIMessage(content="direct answer")])
    factory = _factory(model)

    outcome = await factory.invoke(_request())
    budget = outcome.response.metadata["execution_budget"]

    assert budget["exhausted_by"] is None
    assert budget["forced_synthesis"] is False
