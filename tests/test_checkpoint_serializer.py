from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.ai.checkpoint as checkpoint_module
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import (
    AgentTransition,
    OutcomeProvenance,
    PendingTransition,
    PlanningDispatch,
    ResponseOutcome,
    RoutingDecision,
    TurnIdentity,
    WorkerResult,
    WorkerTask,
    WorkflowError,
)

_CHECKPOINT_TYPES = (
    AgentType,
    MessageRole,
    AgentResponse,
    AgentMessage,
    TurnIdentity,
    RoutingDecision,
    AgentTransition,
    PendingTransition,
    OutcomeProvenance,
    ResponseOutcome,
    WorkerTask,
    WorkerResult,
    PlanningDispatch,
    WorkflowError,
)


def _expected_json_allowlist() -> list[tuple[str, ...]]:
    return [(*symbol.__module__.split("."), symbol.__name__) for symbol in _CHECKPOINT_TYPES]


def _expected_msgpack_allowlist() -> list[tuple[str, str]]:
    return [(symbol.__module__, symbol.__name__) for symbol in _CHECKPOINT_TYPES]


def test_build_checkpoint_serializer_passes_both_allowlists():
    """Both are required. A type missing from either comes back as a dict,
    and a control-plane contract that degrades to a dict is no longer typed
    when it is read back out of the checkpoint."""
    captured: dict[str, object] = {}

    class FakeSerializer:
        def __init__(self, *, allowed_json_modules=None, allowed_msgpack_modules=None):
            captured["allowed_json_modules"] = allowed_json_modules
            captured["allowed_msgpack_modules"] = allowed_msgpack_modules

    original = checkpoint_module.JsonPlusSerializer
    checkpoint_module.JsonPlusSerializer = FakeSerializer
    try:
        serializer = checkpoint_module._build_checkpoint_serializer()
    finally:
        checkpoint_module.JsonPlusSerializer = original

    assert isinstance(serializer, FakeSerializer)
    assert list(captured["allowed_json_modules"]) == _expected_json_allowlist()
    assert list(captured["allowed_msgpack_modules"]) == _expected_msgpack_allowlist()


def test_build_checkpoint_serializer_rejects_legacy_msgpack_api(monkeypatch):
    class LegacySerializer:
        def __init__(self, *, allowed_json_modules=None):
            self.allowed_json_modules = allowed_json_modules

    monkeypatch.setattr(checkpoint_module, "JsonPlusSerializer", LegacySerializer)

    with pytest.raises(
        RuntimeError,
        match=r"langgraph-checkpoint>=4\.1\.1",
    ):
        checkpoint_module._build_checkpoint_serializer()


@pytest.mark.asyncio
async def test_checkpoint_setup_validates_serializer_before_opening_pool(monkeypatch):
    events: list[str] = []

    def reject_legacy_serializer():
        events.append("serializer")
        raise RuntimeError("unsupported checkpoint serializer")

    class UnexpectedPool:
        def __init__(self, *args, **kwargs):
            events.append("pool")

    monkeypatch.setattr(
        checkpoint_module,
        "_build_checkpoint_serializer",
        reject_legacy_serializer,
    )
    monkeypatch.setattr(checkpoint_module, "AsyncConnectionPool", UnexpectedPool)
    manager = checkpoint_module.CheckpointManager(
        db_url="postgresql://user:pass@localhost/db",
        settings=SimpleNamespace(
            checkpoint_schema="public",
            checkpoint_pool_min_size=1,
            checkpoint_pool_max_size=2,
        ),
    )

    with pytest.raises(RuntimeError, match="unsupported checkpoint serializer"):
        await manager.setup()

    assert events == ["serializer"]
    assert manager._pool is None


@pytest.mark.asyncio
async def test_checkpoint_manager_delete_thread_delegates_to_async_saver():
    manager = checkpoint_module.CheckpointManager(
        db_url="postgresql://user:pass@localhost/db",
        settings=SimpleNamespace(checkpoint_schema="public"),
    )
    manager._initialized = True
    calls: list[str] = []

    class FakeCheckpointer:
        async def adelete_thread(self, thread_id: str) -> None:
            calls.append(thread_id)

    manager.checkpointer = FakeCheckpointer()

    deleted = await manager.delete_thread("thread-1")

    assert deleted is True
    assert calls == ["thread-1"]


@pytest.mark.asyncio
async def test_checkpoint_manager_delete_thread_falls_back_to_pool_sql():
    manager = checkpoint_module.CheckpointManager(
        db_url="postgresql://user:pass@localhost/db",
        settings=SimpleNamespace(checkpoint_schema="public"),
    )
    manager._initialized = True
    manager.checkpointer = SimpleNamespace()
    executed: list[tuple[str, tuple[str]]] = []

    class FakeConnection:
        async def execute(self, statement: str, params: tuple[str]) -> None:
            executed.append((statement, params))

    class FakePoolConnection:
        async def __aenter__(self):
            return FakeConnection()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class FakePool:
        def connection(self):
            return FakePoolConnection()

    manager._pool = FakePool()

    deleted = await manager.delete_thread("thread-1")

    assert deleted is True
    assert executed == [
        ('DELETE FROM "public"."checkpoint_writes" WHERE thread_id = %s', ("thread-1",)),
        ('DELETE FROM "public"."checkpoint_blobs" WHERE thread_id = %s', ("thread-1",)),
        ('DELETE FROM "public"."checkpoints" WHERE thread_id = %s', ("thread-1",)),
    ]


def _routed_checkpoint_state() -> dict[str, object]:
    from langchain_core.messages import AIMessage, ToolMessage

    from app.ai.workflow.contracts import (
        AgentTransition,
        OutcomeProvenance,
        PendingTransition,
        PlanningDispatch,
        ResponseOutcome,
        RoutingDecision,
        TurnIdentity,
        WorkerResult,
        WorkerTask,
        WorkflowError,
    )

    return {
        "turn_identity": TurnIdentity(
            request_id="request-1",
            turn_id="message-1",
            checkpoint_thread_id="routing-v2:conversation-1:message-1",
        ),
        "routing_decision": RoutingDecision(
            agent_id="search_agent", confidence=0.91, reason="needs current sources"
        ),
        "agent_history": [
            AgentTransition(from_agent_id=None, to_agent_id="chat_agent", source="router"),
            AgentTransition(
                from_agent_id="chat_agent",
                to_agent_id="search_agent",
                source="handoff",
                tool_call_id="call-1",
            ),
        ],
        "pending_transition": PendingTransition(
            from_agent_id="chat_agent",
            to_agent_id="search_agent",
            tool_call_id="call-1",
            tool_message_id="handoff:call-1",
            reason="needs current sources",
        ),
        "agent_outcome": ResponseOutcome(
            agent_id="search_agent",
            response=AgentResponse(
                agent_type=AgentType.SEARCH,
                agent_id="search_agent",
                message=AgentMessage(role=MessageRole.ASSISTANT, content="answer"),
            ),
            provenance=OutcomeProvenance(
                output_policy_ids=("public_content",),
                evidence=({"evidence_id": "E1"},),
                private_messages=(
                    AIMessage(content="private", id="private-1"),
                    ToolMessage(content="ok", tool_call_id="call-1", id="handoff:call-1"),
                ),
            ),
        ),
        "planning_dispatch": PlanningDispatch(
            dispatch_id="dispatch-1",
            tool_call_id="call-dispatch-1",
            wave=1,
            tasks=(
                WorkerTask(
                    dispatch_id="dispatch-1",
                    task_id="t1",
                    position=0,
                    objective="retrieve the policy",
                    agent_id="rag_agent",
                    allowed_tool_ids=("rag::search",),
                ),
            ),
        ),
        "worker_results": [
            WorkerResult(
                dispatch_id="dispatch-1",
                task_id="t1",
                position=0,
                agent_id="rag_agent",
                status="failed",
                error_code="worker_timeout",
            )
        ],
        "workflow_error": WorkflowError(
            code="routing_timeout",
            retriable=True,
            request_id="request-1",
            details={"attempts": 2},
        ),
    }


def test_checkpoint_serializer_round_trips_v2_contract_types():
    from app.ai.workflow.contracts import (
        AgentTransition,
        PendingTransition,
        PlanningDispatch,
        ResponseOutcome,
        RoutingDecision,
        TurnIdentity,
        WorkerResult,
        WorkerTask,
        WorkflowError,
    )

    serializer = checkpoint_module._build_checkpoint_serializer()
    state = _routed_checkpoint_state()

    restored = serializer.loads_typed(serializer.dumps_typed(state))

    assert isinstance(restored["turn_identity"], TurnIdentity)
    assert isinstance(restored["routing_decision"], RoutingDecision)
    assert isinstance(restored["agent_history"][0], AgentTransition)
    assert isinstance(restored["pending_transition"], PendingTransition)
    assert isinstance(restored["agent_outcome"], ResponseOutcome)
    assert isinstance(restored["worker_results"][0], WorkerResult)
    assert isinstance(restored["planning_dispatch"], PlanningDispatch)
    assert isinstance(restored["planning_dispatch"].tasks[0], WorkerTask)
    assert isinstance(restored["workflow_error"], WorkflowError)


def test_checkpoint_round_trip_preserves_nested_typed_contract_details():
    serializer = checkpoint_module._build_checkpoint_serializer()
    state = _routed_checkpoint_state()

    restored = serializer.loads_typed(serializer.dumps_typed(state))

    outcome = restored["agent_outcome"]
    assert isinstance(outcome, ResponseOutcome)
    assert isinstance(outcome.response, AgentResponse)
    assert outcome.provenance.output_policy_ids == ("public_content",)
    assert outcome.provenance.evidence == ({"evidence_id": "E1"},)
    assert [type(message).__name__ for message in outcome.provenance.private_messages] == [
        "AIMessage",
        "ToolMessage",
    ]
    assert restored["workflow_error"].details == {"attempts": 2}
    assert restored["agent_history"][1].tool_call_id == "call-1"


def test_checkpointed_contract_types_stay_frozen_after_restore():
    import pydantic

    serializer = checkpoint_module._build_checkpoint_serializer()
    restored = serializer.loads_typed(serializer.dumps_typed(_routed_checkpoint_state()))

    with pytest.raises(pydantic.ValidationError):
        restored["routing_decision"].agent_id = "chat_agent"
