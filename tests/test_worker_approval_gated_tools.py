"""A Planning worker cannot call a tool that needs human approval.

Workers run inside ``asyncio.gather``, not as framework-managed tasks. So a
worker cannot use LangGraph's ``interrupt()``: the ``GraphInterrupt`` would
propagate through the gather and cancel every sibling worker, destroying their
in-flight work. That is why the worker path fabricated an ``awaiting_approval``
result instead — the design forbids exactly that, and it was the only thing
that worked.

Fabricating it is worse than it looks. There is no resume, so the worker starts
over: re-running searches, re-spending its evidence budget, and repeating any
side effect it already performed. Meanwhile Planning synthesizes a public
answer from a worker that never finished.

So the tool is refused instead, with bounded model-visible feedback, and the
worker keeps going. It fails closed — the side effect never happens — it keeps
the work already done, and it lets the model try another route. Approval stays
where it can actually be honoured: the top-level agent, which runs as a real
graph node and can interrupt.

This is step 1 of three. Steps 2 and 3 are the ``Send`` migration (which makes
workers framework-managed tasks) and then real ``interrupt()`` support. Until
those land, a worker refusing is the only behaviour that neither loses work nor
performs an unapproved side effect.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def _tool(name: str, *, mutation: bool = False):
    return SimpleNamespace(
        name=name,
        metadata={"tool_origin": "server_mcp", "server_name": "srv", "mutation": mutation},
    )


def _agent_with_calls(*calls: dict, final_text: str = "done"):
    """A worker agent that asks for ``calls`` once, then answers."""
    responses = [
        AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT, content="", tool_calls=list(calls)
            ),
            metadata={"provider": "test", "model": "test-model"},
        ),
        AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=final_text),
            metadata={},
        ),
    ]

    async def process_message(_message, _conversation_id):
        return responses.pop(0)

    return SimpleNamespace(
        process_message=process_message,
        agent_config_key="rag",
        tool_state_key="rag",
        agent_type=AgentType.RAG,
    )


def _workflow(agent):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.rag_agent = agent
    workflow.agents = {"rag_agent": agent}
    return workflow


def _install_tools(monkeypatch, executed: list[str], tools: dict):
    """Bind ``tools`` for the worker and record what actually executes."""
    import app.ai.graph as graph_module

    async def fake_tool_map(*_args, **_kwargs):
        return tools

    async def fake_execute(**kwargs):
        outputs = []
        for call in kwargs["tool_calls"]:
            executed.append(str(call.get("name")))
            outputs.append(
                {"tool_call_id": call.get("id"), "name": call.get("name"), "content": "ran"}
            )
        return outputs, [], []

    import app.ai.workflow.tool_loop as tool_loop_module

    monkeypatch.setattr(graph_module, "ensure_agent_tool_map", fake_tool_map)
    monkeypatch.setattr(graph_module, "execute_tool_calls", fake_execute)
    # The refusal helper lives in tool_loop and resolves provenance through its
    # own reference. Gating by name works without a tool_map; the mutation floor
    # needs the tool's metadata, so this one matters.
    monkeypatch.setattr(tool_loop_module, "ensure_agent_tool_map", fake_tool_map)


async def _run(workflow, *, policy=None):
    context = {"hitl_policy": policy} if policy else {}
    return await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="do the thing",
        parent_state={
            "conversation_id": str(UUID(int=9)),
            "user_id": "owner",
            "device_id": "device-1",
            "context": context,
            "messages": [],
        },
    )


_GATED_POLICY = {"master_enabled": True, "global_tools": ["send_email"]}


# ----------------------------------------------------------------------
# the gated call is refused, not obeyed and not abandoned
# ----------------------------------------------------------------------


async def test_a_worker_never_executes_an_approval_gated_tool(monkeypatch):
    """Fails closed: the side effect must not happen without approval."""
    executed: list[str] = []
    _install_tools(monkeypatch, executed, {"send_email": _tool("send_email")})
    workflow = _workflow(
        _agent_with_calls({"id": "c1", "name": "send_email", "args": {"to": "a@b.c"}})
    )

    await _run(workflow, policy=_GATED_POLICY)

    assert "send_email" not in executed


async def test_a_worker_does_not_abandon_the_turn_for_approval(monkeypatch):
    """The old behaviour fabricated a status the design forbids."""
    executed: list[str] = []
    _install_tools(monkeypatch, executed, {"send_email": _tool("send_email")})
    workflow = _workflow(
        _agent_with_calls(
            {"id": "c1", "name": "send_email", "args": {}}, final_text="finished anyway"
        )
    )

    response = await _run(workflow, policy=_GATED_POLICY)

    metadata = response.metadata or {}
    assert metadata.get("requires_approval") is not True
    assert metadata.get("pause_reason") != "awaiting_approval"
    assert response.message.content == "finished anyway", (
        "the worker discarded its remaining work instead of continuing"
    )


async def test_the_refusal_is_visible_to_the_model(monkeypatch):
    """A silent drop teaches the model nothing and it retries forever."""
    executed: list[str] = []
    _install_tools(monkeypatch, executed, {"send_email": _tool("send_email")})
    workflow = _workflow(_agent_with_calls({"id": "c1", "name": "send_email", "args": {}}))

    response = await _run(workflow, policy=_GATED_POLICY)

    refused = [
        artifact
        for artifact in (response.tool_artifacts or [])
        if artifact.get("tool_call_id") == "c1"
    ]
    assert refused, "the refused call produced no artifact"
    text = str(refused[0].get("output") or refused[0].get("output_text") or "")
    assert "approval" in text.lower()


async def test_an_ungated_call_in_the_same_batch_still_runs(monkeypatch):
    """Refusal is per call, not per batch."""
    executed: list[str] = []
    _install_tools(
        monkeypatch,
        executed,
        {"send_email": _tool("send_email"), "read_docs": _tool("read_docs")},
    )
    workflow = _workflow(
        _agent_with_calls(
            {"id": "c1", "name": "send_email", "args": {}},
            {"id": "c2", "name": "read_docs", "args": {}},
        )
    )

    await _run(workflow, policy=_GATED_POLICY)

    assert executed == ["read_docs"]


async def test_an_auto_gated_mutation_is_refused_too(monkeypatch):
    """A mutation is gated by the policy floor, with no explicit entry."""
    executed: list[str] = []
    _install_tools(monkeypatch, executed, {"delete_thing": _tool("delete_thing", mutation=True)})
    workflow = _workflow(_agent_with_calls({"id": "c1", "name": "delete_thing", "args": {}}))

    await _run(workflow, policy={"master_enabled": True})

    assert executed == []


async def test_nothing_is_refused_when_approval_is_disabled(monkeypatch):
    """With HITL off, a worker is no more restricted than the policy says."""
    executed: list[str] = []
    _install_tools(monkeypatch, executed, {"send_email": _tool("send_email")})
    workflow = _workflow(_agent_with_calls({"id": "c1", "name": "send_email", "args": {}}))

    await _run(workflow, policy={"master_enabled": False, "global_tools": ["send_email"]})

    assert executed == ["send_email"]


async def test_an_ordinary_worker_call_is_untouched(monkeypatch):
    executed: list[str] = []
    _install_tools(monkeypatch, executed, {"read_docs": _tool("read_docs")})
    workflow = _workflow(_agent_with_calls({"id": "c1", "name": "read_docs", "args": {}}))

    response = await _run(workflow, policy=_GATED_POLICY)

    assert executed == ["read_docs"]
    assert (response.metadata or {}).get("requires_approval") is not True


# ----------------------------------------------------------------------
# approval still works where it can be honoured
# ----------------------------------------------------------------------


def test_the_top_level_graph_still_routes_to_approval():
    """Only workers are restricted. The top-level node runs as a real graph
    node, so it can interrupt and must keep doing so."""
    import inspect

    source = inspect.getsource(MultiAgentWorkflow._should_call_tools)
    assert 'return "approval"' in source


def test_no_worker_path_fabricates_an_awaiting_approval_status():
    """The status the design forbids must not come back."""
    import pathlib
    import re

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    graph = (repo_root / "app" / "ai" / "graph.py").read_text(encoding="utf-8")
    worker_region = graph[graph.index("async def _run_agent_in_isolated_context") :]
    assert not re.search(r'"pause_reason"\]\s*=\s*"awaiting_approval"', worker_region), (
        "a worker is fabricating awaiting_approval again — it has no resume, so "
        "the worker restarts and repeats any side effect it already performed"
    )
