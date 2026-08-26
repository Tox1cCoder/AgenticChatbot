"""Phase 4 (Task 4.1-4.3): RAG must produce a final no-tools synthesis pass
when the agentic tool-loop budget is exhausted.

Behaviour under test:
* ``_should_continue_rag`` routes back to ``rag_agent`` once more after the
  budget is reached, marking the context so RAG knows to disable tool binding.
* If the final pass also returns tool calls, the graph routes to ``end`` to
  avoid an infinite loop.
* ``RAGAgent._process_message_agentic`` honours the
  ``rag_force_final_response`` flag by invoking ``_invoke_agentic_rag_model``
  with ``disable_tools=True``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.agents.rag_agent import RAGAgent
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.core.config import settings


def _make_workflow():
    return MultiAgentWorkflow.__new__(MultiAgentWorkflow)


@pytest.mark.asyncio
async def test_rag_handoff_routes_to_search_agent_in_same_turn(monkeypatch):
    """RAG must transfer immediately instead of consuming its local loop budget."""
    import app.ai.workflow.rag_loop as rag_loop

    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {
        "rag_agent": SimpleNamespace(agent_config_key="rag"),
        "search_agent": object(),
    }

    async def no_approval(*args, **kwargs):
        return False

    async def tool_map(*args, **kwargs):
        return {"hand_off": SimpleNamespace(name="hand_off")}

    async def execute_tools(**kwargs):
        return (
            [
                {
                    "tool_call_id": "handoff-1",
                    "name": "hand_off",
                    "content": '{"hand_off":"search_agent"}',
                }
            ],
            [],
            [],
        )

    workflow._needs_approval = no_approval
    workflow._update_tool_error_streak = lambda *args, **kwargs: None
    monkeypatch.setattr(rag_loop, "ensure_agent_tool_map", tool_map)
    monkeypatch.setattr(rag_loop, "execute_tool_calls", execute_tools)

    state = {
        "active_agent_id": "rag_agent",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "handoff-1", "name": "hand_off", "args": {}}],
            )
        ],
        "context": {},
        "custom_agents": {},
    }

    await workflow._rag_tools_node(state)

    assert state["active_agent_id"] == "search_agent"
    assert workflow._should_continue_rag(state) == "search_agent"


@pytest.mark.asyncio
async def test_refused_handoff_feedback_reaches_the_tool_message(monkeypatch):
    """Delegation is interpreted on the dicts that become ToolMessages.

    ``_apply_hand_off_if_present`` rewrites a refused ``hand_off`` output in
    place. If evidence budgeting interposes a copy, the refusal is written to the
    copy and the model still reads canonical handoff JSON — it believes the
    delegation happened and re-issues it forever.
    """
    import app.ai.workflow.rag_loop as rag_loop

    workflow = _make_workflow()
    workflow.rag_agent = SimpleNamespace(
        _take_evidence_token_counter=lambda descriptor, **_kwargs: SimpleNamespace(
            count_text=lambda **kwargs: SimpleNamespace(
                tokens=len(kwargs["text"]), strategy="characters"
            )
        )
    )
    workflow.agents = {
        "rag_agent": SimpleNamespace(agent_config_key="rag"),
        "search_agent": object(),
    }

    async def no_approval(*_args, **_kwargs):
        return False

    async def tool_map(*_args, **_kwargs):
        return {"hand_off": SimpleNamespace(name="hand_off")}

    async def execute_tools(**_kwargs):
        return (
            [
                {
                    "tool_call_id": "handoff-1",
                    "name": "hand_off",
                    "content": '{"hand_off":"unreachable_agent"}',
                }
            ],
            [
                {
                    "tool_call_id": "handoff-1",
                    "tool": "hand_off",
                    "status": "success",
                    "output": '{"hand_off":"unreachable_agent"}',
                }
            ],
            [],
        )

    workflow._needs_approval = no_approval
    workflow._update_tool_error_streak = lambda *_args, **_kwargs: None
    monkeypatch.setattr(rag_loop, "ensure_agent_tool_map", tool_map)
    monkeypatch.setattr(rag_loop, "execute_tool_calls", execute_tools)

    state = {
        "active_agent_id": "rag_agent",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "handoff-1", "name": "hand_off", "args": {}}],
            )
        ],
        "context": {},
        "custom_agents": {},
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={"request_budget": {"evidence_token_allowance": 4}},
        ),
    }

    await workflow._rag_tools_node(state)

    assert state["active_agent_id"] == "rag_agent"
    tool_message = state["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content.startswith("Hand-off refused:")
    assert "unreachable_agent" in tool_message.content


def test_rag_budget_routes_to_final_no_tool_pass(monkeypatch):
    monkeypatch.setattr(
        "app.ai.graph.settings.agentic_max_iterations",
        2,
        raising=False,
    )

    workflow = _make_workflow()
    state = {
        "messages": [ToolMessage(content="chunk text", tool_call_id="call-1")],
        "context": {"agentic_rag_iteration": 2},
    }

    route = workflow._should_continue_rag(state)

    assert route == "rag_agent", (
        "Budget exhausted with a tool result available — RAG should run "
        "one final no-tools synthesis pass instead of ending immediately."
    )
    assert state["context"].get("rag_force_final_response") is True
    assert state["context"].get("rag_tool_budget_notice"), (
        "A budget notice must be set so the RAG agent can append it to the prompt"
    )


def test_rag_budget_already_forced_routes_to_end_to_avoid_loop(monkeypatch):
    """If the final pass still emits tool calls, the second visit must end."""
    monkeypatch.setattr(
        "app.ai.graph.settings.agentic_max_iterations",
        2,
        raising=False,
    )

    workflow = _make_workflow()
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "call-2", "name": "search_documents", "args": {}}],
            )
        ],
        "context": {
            "agentic_rag_iteration": 3,
            "rag_force_final_response": True,
            "rag_tool_budget_notice": "budget reached",
        },
    }

    route = workflow._should_continue_rag(state)
    assert route == "end", (
        "rag_force_final_response=True means we already gave the model the "
        "final pass; if it still asks for tools, the graph must end."
    )


def test_forced_final_assistant_tool_calls_do_not_route_to_rag_tools():
    workflow = _make_workflow()
    discarded: list[dict] = []
    workflow.rag_agent = SimpleNamespace(
        _discard_evidence_token_counter=lambda descriptor: discarded.append(descriptor)
    )
    descriptor = {
        "reference": "forced-final-counter",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "fallback": "deterministic_local_conservative",
    }
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "call-2", "name": "search_documents", "args": {}}],
            )
        ],
        "context": {"rag_force_final_response": True},
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={"evidence_tokenization": descriptor},
        ),
    }

    route = workflow._should_call_rag_tools(state)

    assert route == "end"
    assert "evidence_tokenization" not in state["response"].metadata
    assert discarded == [descriptor]


def test_rag_budget_not_yet_reached_continues_normally(monkeypatch):
    monkeypatch.setattr(
        "app.ai.graph.settings.agentic_max_iterations",
        5,
        raising=False,
    )

    workflow = _make_workflow()
    state = {
        "messages": [ToolMessage(content="x", tool_call_id="call-1")],
        "context": {"agentic_rag_iteration": 2},
    }

    route = workflow._should_continue_rag(state)
    assert route == "rag_agent"
    assert "rag_force_final_response" not in state["context"]


def test_process_message_agentic_disables_tools_when_force_final_response_flag_set():
    """RAG must disable tool binding when graph forwards rag_force_final_response."""
    agent = object.__new__(RAGAgent)
    agent.settings = type("S", (), {"agentic_preview_chars": 500})()
    agent.agentic_max_iterations = 5
    agent.tools = []
    agent.mcp_manager = None
    agent._tools_generation_seen = -1

    captured = {}

    async def fake_invoke(**kwargs):
        captured.update(kwargs)
        from app.ai.schemas import AgentResponse, AgentType

        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final answer"),
            metadata={"agentic_mode": True},
        )

    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    runtime_config = ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": False},
        fallback_config=None,
        warnings=[],
        provider_fallback=None,
        is_custom_model=False,
    )

    agent._invoke_agentic_rag_model = fake_invoke
    agent._resolve_runtime_model_config = lambda *a, **kw: runtime_config
    agent._create_fallback_runtime_config = lambda *a, **kw: None
    agent._build_skills_suffix = lambda **kw: ""
    agent._get_tools_for_binding = lambda **kw: []

    msg = AgentMessage(
        role=MessageRole.USER,
        content="Summarise the document.",
        metadata={
            "original_query": "Summarise the document.",
            "rag_force_final_response": True,
            "rag_tool_budget_notice": "Tool budget reached. Synthesise now.",
        },
    )

    response = asyncio.run(agent._process_message_agentic(msg, "conv-1"))

    assert captured.get("disable_tools") is True, (
        f"_invoke_agentic_rag_model must be called with disable_tools=True; "
        f"got {captured.get('disable_tools')}"
    )

    # The system prompt must include the budget notice.
    system_msg = captured["messages"][0]
    rendered = system_msg.content if hasattr(system_msg, "content") else str(system_msg)
    assert "TOOL BUDGET NOTICE" in rendered
    assert "Tool budget reached. Synthesise now." in rendered
    assert "Use the search_documents tool" not in captured["messages"][-1].content

    assert response.metadata.get("rag_force_final_response") is True
    assert response.message.content == "final answer"


def test_rag_document_tool_results_are_recorded_as_response_artifacts(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}

    async def fake_execute_search_documents_action(**kwargs):
        action = kwargs["tool_args"]["action"]
        if action == "search_chunks":
            return "SEARCH RESULTS:\n\n[1] report.pdf\nchunk evidence", action, {}
        if action == "read_document":
            return "DOCUMENT CONTENT (doc-1):\n\nfull document text", action, {}
        raise AssertionError(f"Unexpected action: {action}")

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )

    state = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {},
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "chunk-call",
                        "name": "search_documents",
                        "args": {"action": "search_chunks", "query": "revenue"},
                    },
                    {
                        "id": "document-call",
                        "name": "search_documents",
                        "args": {"action": "read_document", "document_id": "doc-1"},
                    },
                ],
            )
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final"),
            metadata={},
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    artifacts = state["context"].get("tool_artifacts")
    assert artifacts and len(artifacts) == 2
    assert artifacts[0]["tool_call_id"] == "chunk-call"
    assert artifacts[0]["tool"] == "search_documents"
    assert artifacts[0]["args"]["action"] == "search_chunks"
    assert "chunk evidence" in artifacts[0]["output"]
    assert artifacts[1]["tool_call_id"] == "document-call"
    assert artifacts[1]["args"]["action"] == "read_document"
    assert "full document text" in artifacts[1]["output"]

    recovered = workflow._recover_terminal_response(state)
    assert recovered is not None
    assert recovered.tool_artifacts == artifacts


def test_rag_search_passes_authoritative_allowance_and_persists_pack(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}
    captured = {}

    async def fake_execute_search_documents_action(**kwargs):
        captured.update(kwargs)
        pack = {
            "records": [{"evidence_id": "E1", "content": "bounded"}],
            "evidence_ids": ["E1"],
            "token_count": 7,
            "omitted_count": 0,
            "truncated_count": 0,
            "count_strategy": "test",
        }
        return (
            "BEGIN UNTRUSTED EVIDENCE E1\nbounded\nEND UNTRUSTED EVIDENCE E1",
            "search_chunks",
            pack,
        )

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.apply_tool_output_offload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("bounded evidence serialization must remain the ToolMessage")
        ),
    )
    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {},
        "messages": [
            HumanMessage(content="What is revenue?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search-1",
                        "name": "search_documents",
                        "args": {"action": "search_chunks", "query": "revenue"},
                    }
                ],
            ),
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "request_budget": {"evidence_token_allowance": 321},
            },
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    assert captured["question"] == "What is revenue?"
    assert captured["evidence_max_tokens"] == 321
    assert captured["evidence_provider"] == "gemini"
    assert captured["evidence_model"] == "gemini-2.5-flash"
    tool_message = state["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content.startswith("BEGIN UNTRUSTED EVIDENCE E1")
    artifact = state["context"]["tool_artifacts"][0]
    assert artifact["rag_evidence"]["records"][0]["evidence_id"] == "E1"


def test_rag_zero_allowance_does_not_fall_back_to_independent_budget(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}
    captured = {}

    async def fake_execute_search_documents_action(**kwargs):
        captured.update(kwargs)
        return "", "search_chunks", {
            "records": [],
            "evidence_ids": [],
            "token_count": 0,
            "omitted_count": 1,
            "truncated_count": 0,
        }

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {},
        "messages": [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search-1",
                        "name": "search_documents",
                        "args": {"action": "search_chunks", "query": "question"},
                    }
                ],
            ),
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={"request_budget": {"evidence_token_allowance": 0}},
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    assert captured["evidence_max_tokens"] == 0
    assert state["messages"][-1].content == ""


def test_rag_search_calls_share_one_cumulative_evidence_allowance(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}
    allowances: list[int] = []

    async def fake_execute_search_documents_action(**kwargs):
        allowance = kwargs["evidence_max_tokens"]
        allowances.append(allowance)
        used = min(60, allowance)
        return "bounded", "search_chunks", {
            "records": [{"evidence_id": f"E{len(allowances)}"}],
            "evidence_ids": [f"E{len(allowances)}"],
            "token_count": used,
            "omitted_count": 0,
            "truncated_count": 0,
        }

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {},
        "messages": [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "s1", "name": "search_documents", "args": {}},
                    {"id": "s2", "name": "search_documents", "args": {}},
                ],
            ),
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={"request_budget": {"evidence_token_allowance": 100}},
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    assert allowances == [100, 40]


def test_mixed_rag_actions_charge_non_pack_content_before_later_search(monkeypatch):
    workflow = _make_workflow()
    search_allowances: list[int] = []

    async def fake_execute_search_documents_action(**kwargs):
        action = kwargs["tool_args"]["action"]
        if action == "list_documents":
            return "one two three four", action, {"documents": []}
        search_allowances.append(kwargs["evidence_max_tokens"])
        return "pack", action, {
            "records": [{"evidence_id": "E1"}],
            "evidence_ids": ["E1"],
            "token_count": 1,
            "omitted_count": 0,
            "truncated_count": 0,
        }

    class FourWordCounter:
        def count_text(self, **kwargs):
            return SimpleNamespace(tokens=len(kwargs["text"].split()), strategy="words")

    workflow.rag_agent = SimpleNamespace(
        _take_evidence_token_counter=lambda descriptor, **_kwargs: FourWordCounter()
    )
    workflow.agents = {}

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {},
        "messages": [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "l1", "name": "search_documents", "args": {"action": "list_documents"}},
                    {"id": "s1", "name": "search_documents", "args": {"action": "search_chunks"}},
                ],
            ),
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={
                "request_budget": {"evidence_token_allowance": 100},
                "evidence_tokenization": {
                    "reference": "test-counter",
                    "provider": "gemini",
                    "model": "gemini-2.5-flash",
                    "fallback": "deterministic_local_conservative",
                },
            },
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    assert search_allowances == [96]


def test_oversized_last_non_pack_result_is_omitted_before_tool_message_append(monkeypatch):
    import json

    workflow = _make_workflow()

    class CharacterCounter:
        def count_text(self, **kwargs):
            return SimpleNamespace(tokens=len(kwargs["text"]), strategy="characters")

    workflow.rag_agent = SimpleNamespace(
        _take_evidence_token_counter=lambda descriptor, **_kwargs: CharacterCounter()
    )
    workflow.agents = {}
    oversized = json.dumps(
        {"rows": [{"id": index, "value": "x" * 20} for index in range(20)]},
        separators=(",", ":"),
    )

    async def fake_execute_search_documents_action(**kwargs):
        action = kwargs["tool_args"]["action"]
        if action == "search_chunks":
            return "bounded pack", action, {
                "records": [{"evidence_id": "E1"}],
                "evidence_ids": ["E1"],
                "token_count": 20,
                "omitted_count": 0,
                "truncated_count": 0,
            }
        return oversized, action, {"documents": []}

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )
    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {},
        "messages": [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search-first",
                        "name": "search_documents",
                        "args": {"action": "search_chunks", "query": "question"},
                    },
                    {
                        "id": "list-last",
                        "name": "search_documents",
                        "args": {"action": "list_documents"},
                    },
                ],
            ),
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={
                "request_budget": {"evidence_token_allowance": 100},
                "evidence_tokenization": {
                    "reference": "character-counter",
                    "provider": "test",
                    "model": "test",
                    "fallback": "deterministic_local_conservative",
                },
            },
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    last_result = state["messages"][-1].content
    assert len(last_result) <= 80
    assert last_result != oversized
    # A blank ToolMessage is indistinguishable from an empty-but-successful
    # result, so the bounded replacement is always the explicit marker.
    assert json.loads(last_result)["reason"] == "context_budget"
    artifacts = state["context"]["tool_artifacts"]
    omitted = [a for a in artifacts if a.get("tool_call_id") == "list-last"]
    assert omitted and omitted[0]["model_output_omitted_reason"] == "context_budget"


def test_missing_authoritative_allowance_keeps_model_visible_tool_text(monkeypatch):
    """No propagated allowance must not be read as a zero-token allowance.

    ``evidence_token_allowance`` is absent whenever the request budget could not
    run (unresolvable context window, non-preflighted provider). Treating that
    as ``0`` would blank every tool result and starve the loop of its own
    evidence, so non-pack content is charged but not bounded in that case.
    """
    workflow = _make_workflow()
    workflow.rag_agent = SimpleNamespace(
        _take_evidence_token_counter=lambda descriptor, **_kwargs: SimpleNamespace(
            count_text=lambda **kwargs: SimpleNamespace(
                tokens=len(kwargs["text"]), strategy="characters"
            )
        )
    )
    workflow.agents = {}

    async def fake_execute_search_documents_action(**kwargs):
        return "AVAILABLE DOCUMENTS: report.pdf", kwargs["tool_args"]["action"], {}

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )
    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {},
        "messages": [
            HumanMessage(content="question"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "list-only",
                        "name": "search_documents",
                        "args": {"action": "list_documents"},
                    }
                ],
            ),
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={},
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    assert state["messages"][-1].content == "AVAILABLE DOCUMENTS: report.pdf"
    assert "model_output_omitted" not in state["context"]["tool_artifacts"][0]


@pytest.mark.asyncio
async def test_rag_agent_preserves_current_assistant_tool_group_without_synthetic_human_text():
    agent = object.__new__(RAGAgent)
    agent.settings = type("S", (), {"agentic_preview_chars": 500})()
    agent.agentic_max_iterations = 5
    agent.tools = [SimpleNamespace(name="search_documents")]
    agent.mcp_manager = None
    agent._tools_generation_seen = -1
    captured = {}

    async def fake_invoke(**kwargs):
        captured.update(kwargs)
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="answer"),
            metadata={},
        )

    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    runtime_config = ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": False},
        fallback_config=None,
        warnings=[],
        provider_fallback=None,
        is_custom_model=False,
    )
    agent._invoke_agentic_rag_model = fake_invoke
    agent._resolve_runtime_model_config = lambda *a, **kw: runtime_config
    agent._create_fallback_runtime_config = lambda *a, **kw: None
    agent._build_skills_suffix = lambda **kw: ""
    agent._get_tools_for_binding = lambda **kw: []

    tool_call = AIMessage(
        content="",
        tool_calls=[{"id": "search-1", "name": "search_documents", "args": {}}],
    )
    evidence = ToolMessage(
        content="BEGIN UNTRUSTED EVIDENCE E1\nbounded\nEND UNTRUSTED EVIDENCE E1",
        tool_call_id="search-1",
        name="search_documents",
    )
    msg = AgentMessage(
        role=MessageRole.USER,
        content="What is revenue?",
        metadata={
            "original_query": "What is revenue?",
            "rag_tool_messages": [tool_call, evidence],
        },
    )

    response = await agent._process_message_agentic(msg, "conv-1")

    assert response.message.content == "answer"
    emitted = captured["messages"]
    assert isinstance(emitted[-3], HumanMessage)
    assert isinstance(emitted[-2], AIMessage)
    assert isinstance(emitted[-1], ToolMessage)
    assert all("Previous Tool Results" not in str(item.content) for item in emitted)


def test_rag_action_named_tool_call_is_canonicalized_to_search_documents(monkeypatch):
    workflow = _make_workflow()
    workflow.rag_agent = object()
    workflow.agents = {}

    async def fake_execute_search_documents_action(**kwargs):
        assert kwargs["tool_args"] == {
            "document_id": "doc-1",
            "pattern": "gabapentin",
            "action": "grep_document",
        }
        return "MATCHES for 'gabapentin' in document:\n\n1. gabapentin", "grep_document", {}

    async def fail_execute_tool_calls(**_kwargs):
        raise AssertionError("RAG document actions must not use generic deferred tool execution")

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr("app.ai.workflow.rag_loop.execute_tool_calls", fail_execute_tool_calls)

    state = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {},
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "grep-call",
                        "name": "grep_document",
                        "args": {"document_id": "doc-1", "pattern": "gabapentin"},
                    }
                ],
            )
        ],
        "response": AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final"),
            metadata={},
        ),
    }

    asyncio.run(workflow._rag_tools_node(state))

    artifacts = state["context"].get("tool_artifacts")
    assert artifacts and len(artifacts) == 1
    assert artifacts[0]["tool"] == "search_documents"
    assert artifacts[0]["args"]["action"] == "grep_document"
    assert "gabapentin" in artifacts[0]["output"]

    tool_message = state["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.name == "search_documents"


@pytest.mark.asyncio
async def test_rag_tools_node_tracks_document_tool_error_streak(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    graph.rag_agent = object()
    graph.agents = {}

    async def fake_execute_search_documents_action(**kwargs):
        return "Error: Unknown action: nope", "nope", {}

    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.workflow.rag_loop.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    state = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {},
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "rag-call-1",
                        "name": "search_documents",
                        "args": {"action": "nope"},
                    }
                ],
            )
        ],
    }

    await graph._rag_tools_node(state)

    streak = state["context"]["tool_error_streak"]
    assert streak["count"] == 1
    assert streak["signature"]["tool"] == "search_documents"


def test_rag_repeated_error_forces_final_response(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {
        "context": {
            "tool_error_streak": {
                "count": 2,
                "limit": 2,
                "signature": {
                    "tool": "search_documents",
                    "error_type": "validation",
                    "args": '{"action":"not_a_real_action"}',
                },
            }
        }
    }

    decision = graph._should_continue_rag(state)

    assert decision == "rag_agent"
    assert state["context"]["rag_force_final_response"] is True
    assert "repeated tool errors" in state["context"]["rag_tool_budget_notice"].lower()


@pytest.mark.asyncio
async def test_execute_search_documents_action_returns_compact_error_for_unknown_action():
    import json
    from types import SimpleNamespace

    from app.ai.rag_tool_actions import execute_search_documents_action

    result, action, evidence = await execute_search_documents_action(
        rag_agent=SimpleNamespace(),
        conversation_id="conv-1",
        user_id="user-1",
        tool_args={"action": "not_a_real_action"},
        context={},
        max_agentic_images=3,
    )

    payload = json.loads(result)
    assert action == "not_a_real_action"
    assert evidence == {}
    assert payload == {
        "status": "error",
        "error_type": "validation",
        "retryable": False,
        "message": "search_documents rejected the requested action.",
        "hint": "Use one of the supported document exploration actions from the tool schema.",
    }


@pytest.mark.asyncio
async def test_list_documents_returns_one_server_bounded_page_with_total_count():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.ai.rag_tool_actions import execute_search_documents_action

    documents_page = {
        "documents": [
            {"document_id": "doc-1", "filename": "one.pdf", "chunk_count": 2},
        ],
        "total": 76,
        "page": 3,
        "page_size": 25,
    }
    rag_agent = SimpleNamespace(
        list_conversation_documents=AsyncMock(return_value=documents_page)
    )

    result, action, evidence = await execute_search_documents_action(
        rag_agent=rag_agent,
        conversation_id="conv-1",
        user_id="user-1",
        tool_args={"action": "list_documents", "page": 3, "page_size": 999},
        context={},
        max_agentic_images=3,
    )

    assert action == "list_documents"
    assert "Page 3" in result
    assert evidence["pagination"] == {
        "page": 3,
        "page_size": 25,
        "total": 76,
        "next_page": 4,
    }
    rag_agent.list_conversation_documents.assert_awaited_once_with(
        "conv-1",
        user_id="user-1",
        page=3,
        page_size=25,
    )


@pytest.mark.asyncio
async def test_list_documents_empty_page_still_returns_pagination():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.ai.rag_tool_actions import execute_search_documents_action

    rag_agent = SimpleNamespace(
        list_conversation_documents=AsyncMock(
            return_value={
                "documents": [],
                "total": 7,
                "page": 2,
                "page_size": 5,
            }
        )
    )

    result, _, evidence = await execute_search_documents_action(
        rag_agent=rag_agent,
        conversation_id="conv-1",
        user_id="user-1",
        tool_args={"action": "list_documents", "page": 2, "page_size": 5},
        context={},
        max_agentic_images=3,
    )

    assert "Page 2" in result
    assert "0 of 7 documents" in result
    assert evidence["pagination"] == {
        "page": 2,
        "page_size": 5,
        "total": 7,
        "next_page": None,
    }


def _grounded_state(*, evidence_id: str = "E1", filename: str = "report.pdf"):
    """One RAG turn whose single search call produced one server-owned record."""
    from uuid import UUID

    artifact = {
        "tool_call_id": "search-1",
        "tool": "search_documents",
        "args": {"action": "search_chunks", "query": "revenue"},
        "output": f"BEGIN UNTRUSTED EVIDENCE {evidence_id}",
        "error": None,
        "status": "success",
        "rag_evidence": {
            "records": [
                {
                    "evidence_id": evidence_id,
                    "document_id": str(UUID(int=1)),
                    "chunk_id": str(UUID(int=11)),
                    "image_id": None,
                    "filename": filename,
                    "page_start": 3,
                    "page_end": 3,
                    "section_path": ["Results"],
                    "modality": "text",
                    "content": "Revenue rose to 10 million in FY24.",
                }
            ],
            "evidence_ids": [evidence_id],
            "token_count": 19,
            "omitted_count": 0,
            "truncated_count": 0,
            "count_strategy": "test:fixture",
        },
    }
    return {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {"tool_artifacts": [artifact], "agentic_rag_iteration": 1},
        "messages": [
            HumanMessage(content="What was revenue?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search-1",
                        "name": "search_documents",
                        "args": {"action": "search_chunks", "query": "revenue"},
                    }
                ],
            ),
            ToolMessage(
                content=f"BEGIN UNTRUSTED EVIDENCE {evidence_id}",
                tool_call_id="search-1",
                name="search_documents",
            ),
        ],
    }


def _grounded_workflow(*, final_text: str, regenerated_answer=None):
    """A workflow whose RAG agent returns ``final_text`` as its final response."""
    calls: dict[str, object] = {"regenerations": []}

    async def process_message(_message, _conversation_id, **_kwargs):
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=final_text),
            metadata={"agentic_mode": True},
        )

    async def regenerate_grounded_answer(*, reason_codes, **kwargs):
        calls["regenerations"].append((tuple(reason_codes), kwargs.get("question")))
        return regenerated_answer

    workflow = _make_workflow()
    workflow.rag_agent = SimpleNamespace(
        process_message=process_message,
        regenerate_grounded_answer=regenerate_grounded_answer,
    )

    async def history(*_args, **_kwargs):
        return []

    workflow._get_conversation_history = history
    workflow._get_state_attachments = lambda _state: []
    return workflow, calls


def test_an_uncited_answer_is_not_published_as_written(monkeypatch):
    """There is no shadow mode: a below-coverage answer is acted on, not logged.

    Previously this same input was recorded as ``would_abstain`` and published
    unchanged. Mandatory grounding means the turn now spends its one
    regeneration and then abstains rather than shipping the uncited claim.
    """
    monkeypatch.setattr(settings, "enable_citation_verification", True, raising=False)
    state = _grounded_state()
    workflow, calls = _grounded_workflow(final_text="Revenue rose to 10 million.")

    asyncio.run(workflow._rag_node(state))

    grounded = state["response"].metadata["grounded_answer"]
    assert grounded["mode"] == "enforced"
    assert grounded["outcome"] != "would_abstain"
    assert grounded["valid"] is False
    assert grounded["evidence_id_count"] == 1
    assert calls["regenerations"], "enforcement must spend its one regeneration"


def test_enforced_grounded_gate_appends_server_rendered_sources(monkeypatch):
    monkeypatch.setattr(settings, "enable_citation_verification", True, raising=False)
    state = _grounded_state()
    workflow, calls = _grounded_workflow(
        final_text="Revenue rose to 10 million [E1] [Source: forged.pdf, Page 99].",
    )

    asyncio.run(workflow._rag_node(state))

    content = state["response"].message.content
    assert "Revenue rose to 10 million [E1]." in content
    assert "forged.pdf" not in content
    assert '[E1] "report.pdf" page 3' in content
    assert calls["regenerations"] == []
    assert state["response"].metadata["grounded_answer"]["outcome"] == "accepted"


def test_enforced_grounded_gate_regenerates_once_then_abstains(monkeypatch):
    from app.services.rag_grounding import GroundedAnswer, GroundedClaim

    monkeypatch.setattr(settings, "enable_citation_verification", True, raising=False)
    state = _grounded_state()
    workflow, calls = _grounded_workflow(
        final_text="Revenue rose to 10 million [E9].",
        regenerated_answer=GroundedAnswer(
            claims=[GroundedClaim(text="Revenue rose to 10 million.", evidence_ids=("E8",))]
        ),
    )

    asyncio.run(workflow._rag_node(state))

    metadata = state["response"].metadata
    assert calls["regenerations"] == [(("unknown_evidence_id",), "What was revenue?")]
    assert metadata["grounded_answer"]["outcome"] == "abstained"
    assert metadata["grounded_answer"]["regenerated"] is True
    assert "E9" not in state["response"].message.content
    assert "report.pdf" not in state["response"].message.content
    assert "evidence_tokenization" not in metadata
    assert "_evidence_token_counter" not in metadata
    assert all(
        isinstance(value, (bool, int, float, str, list))
        for value in metadata["grounded_answer"].values()
    ), "gate metadata must stay msgpack-safe for checkpointed state"


def test_grounded_gate_ignores_evidence_from_other_tool_calls(monkeypatch):
    """Only the current turn's pack authorizes a citation."""
    monkeypatch.setattr(settings, "enable_citation_verification", True, raising=False)
    state = _grounded_state()
    stale = dict(state["context"]["tool_artifacts"][0])
    stale["tool_call_id"] = "search-from-an-earlier-turn"
    state["context"]["tool_artifacts"] = [stale]
    workflow, _calls = _grounded_workflow(final_text="Revenue rose to 10 million [E1].")

    asyncio.run(workflow._rag_node(state))

    grounded = state["response"].metadata["grounded_answer"]
    assert grounded["evidence_id_count"] == 0
    assert grounded["reason_codes"] == ["unknown_evidence_id", "answer_without_evidence"]


def test_grounded_gate_leaves_failed_rag_turns_reporting_their_own_error(monkeypatch):
    """An error response is not an answer: replacing it would hide the failure."""
    monkeypatch.setattr(settings, "enable_citation_verification", True, raising=False)
    state = _grounded_state()
    workflow, calls = _grounded_workflow(final_text="unused")

    async def failing_process_message(_message, _conversation_id, **_kwargs):
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="Error during document exploration: provider timeout",
            ),
            metadata={"agentic_mode": True, "error": "provider timeout"},
            error="provider timeout",
        )

    workflow.rag_agent.process_message = failing_process_message

    asyncio.run(workflow._rag_node(state))

    assert (
        state["response"].message.content
        == "Error during document exploration: provider timeout"
    )
    assert "grounded_answer" not in state["response"].metadata
    assert calls["regenerations"] == []


def test_grounding_runs_even_with_citation_verification_turned_off(monkeypatch):
    """No setting can skip validation on a RAG answer.

    ``enable_citation_verification`` is retained for operational visibility;
    it is not an off switch, because a RAG turn that publishes unvalidated
    claims is the failure grounding exists to prevent.
    """
    monkeypatch.setattr(settings, "enable_citation_verification", False, raising=False)
    state = _grounded_state()
    workflow, _calls = _grounded_workflow(final_text="Revenue rose to 10 million.")

    asyncio.run(workflow._rag_node(state))

    assert "grounded_answer" in state["response"].metadata
    assert state["response"].metadata["grounded_answer"]["mode"] == "enforced"


def test_enforced_regeneration_preserves_the_regenerated_markdown_structure(monkeypatch):
    """Round-1 finding 3: a regenerated answer must render from its own raw
    text, not a space-joined run-on paragraph built from its claims."""
    from app.services.rag_grounding import parse_grounded_answer

    monkeypatch.setattr(settings, "enable_citation_verification", True, raising=False)
    state = _grounded_state()
    regenerated_text = "Revenue rose [E1].\n\n- Costs fell [E1]\n- Margins widened [E1]"
    workflow, calls = _grounded_workflow(
        final_text="Revenue rose to 10 million [E9].",
        regenerated_answer=parse_grounded_answer(regenerated_text),
    )

    asyncio.run(workflow._rag_node(state))

    content = state["response"].message.content
    assert calls["regenerations"] == [(("unknown_evidence_id",), "What was revenue?")]
    assert state["response"].metadata["grounded_answer"]["outcome"] == "regenerated"
    assert "\n\n- Costs fell [E1]\n- Margins widened [E1]" in content, (
        "regenerated markdown structure must survive rendering, not be "
        "flattened into one space-joined paragraph"
    )


def test_grounded_gate_reports_ambiguous_evidence_id_count(monkeypatch):
    """Round-1 finding 5: an id reused for two different records this turn
    must be visible in the shadow metadata, not silently dropped."""
    from uuid import UUID

    monkeypatch.setattr(settings, "enable_citation_verification", True, raising=False)

    def _artifact(tool_call_id: str, filename: str) -> dict:
        return {
            "tool_call_id": tool_call_id,
            "tool": "search_documents",
            "args": {"action": "search_chunks", "query": "revenue"},
            "output": "BEGIN UNTRUSTED EVIDENCE E1",
            "error": None,
            "status": "success",
            "rag_evidence": {
                "records": [
                    {
                        "evidence_id": "E1",
                        "document_id": str(UUID(int=1)),
                        "chunk_id": str(UUID(int=11)),
                        "image_id": None,
                        "filename": filename,
                        "page_start": 3,
                        "page_end": 3,
                        "section_path": ["Results"],
                        "modality": "text",
                        "content": "Revenue rose to 10 million in FY24.",
                    }
                ],
                "evidence_ids": ["E1"],
                "token_count": 19,
                "omitted_count": 0,
                "truncated_count": 0,
                "count_strategy": "test:fixture",
            },
        }

    state = {
        "conversation_id": "conv-1",
        "user_id": "owner",
        "context": {
            "tool_artifacts": [
                _artifact("search-1", "report.pdf"),
                _artifact("search-2", "different.pdf"),
            ],
            "agentic_rag_iteration": 1,
        },
        "messages": [
            HumanMessage(content="What was revenue?"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "search-1", "name": "search_documents", "args": {}},
                    {"id": "search-2", "name": "search_documents", "args": {}},
                ],
            ),
            ToolMessage(
                content="BEGIN UNTRUSTED EVIDENCE E1",
                tool_call_id="search-1",
                name="search_documents",
            ),
            ToolMessage(
                content="BEGIN UNTRUSTED EVIDENCE E1",
                tool_call_id="search-2",
                name="search_documents",
            ),
        ],
    }
    workflow, _calls = _grounded_workflow(final_text="Revenue rose to 10 million [E1].")

    asyncio.run(workflow._rag_node(state))

    shadow = state["response"].metadata["grounded_answer"]
    assert shadow["ambiguous_evidence_id_count"] == 1
    assert shadow["evidence_id_count"] == 0, "the colliding id must not be citable either"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["search_chunks", "view_images"])
@pytest.mark.parametrize(
    ("conversation_id", "user_id"),
    [(None, None), ("conv-1", None), (None, "user-1")],
)
async def test_retrieval_actions_reject_missing_server_scope_without_querying(
    action, conversation_id, user_id
):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.ai.rag_tool_actions import execute_search_documents_action

    rag_agent = SimpleNamespace(
        _search=AsyncMock(),
        get_document_images=AsyncMock(),
    )
    tool_args = {"action": action}
    if action == "search_chunks":
        tool_args["query"] = "revenue"
    else:
        tool_args["document_id"] = "doc-1"

    result, normalized_action, evidence = await execute_search_documents_action(
        rag_agent=rag_agent,
        conversation_id=conversation_id,
        user_id=user_id,
        tool_args=tool_args,
        context={},
        max_agentic_images=3,
    )

    payload = json.loads(result)
    assert normalized_action == action
    assert evidence == {}
    assert payload["error_type"] == "validation"
    assert "authenticated user and conversation context" in payload["message"]
    rag_agent._search.assert_not_awaited()
    rag_agent.get_document_images.assert_not_awaited()
