"""A Planning RAG worker's answer is grounded by the same gate as a top-level one.

This is the duplication the routing-v2 design named first: "the planning worker
contains a second RAG loop that does not invoke the graph RAG grounding gate."

The consequence was concrete. A worker ran its own ``search_documents`` loop and
returned an ``AgentResponse`` straight to Planning, so its citations were never
checked against the evidence that worker actually retrieved. Planning then
synthesized those citations into a public answer. An id naming nothing —
invented, or carried over from another turn — reached the reader wearing the
same brackets as a real one.

Grounding a worker is not a second policy. These assert it is the *same* gate,
because two implementations is how one path quietly stops validating.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def _worker_agent(final_text: str, *, cite: str = "E1"):
    """A RAG agent that searches once, then answers with ``final_text``."""
    responses = [
        AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    {
                        "id": "search-1",
                        "name": "search_documents",
                        "args": {"action": "search_chunks", "query": "revenue"},
                    }
                ],
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


def _install_retrieval(monkeypatch, *evidence_ids: str):
    async def fake_action(**_kwargs):
        return (
            "BEGIN UNTRUSTED EVIDENCE\nrevenue rose\nEND UNTRUSTED EVIDENCE",
            "search_chunks",
            {
                "records": [
                    {
                        "evidence_id": evidence_id,
                        "document_id": str(UUID(int=1)),
                        "filename": "report.pdf",
                        "content": "revenue rose",
                    }
                    for evidence_id in evidence_ids
                ],
                "evidence_ids": list(evidence_ids),
                "token_count": 5,
                "omitted_count": 0,
                "truncated_count": 0,
            },
        )

    monkeypatch.setattr("app.ai.graph.execute_search_documents_action", fake_action)
    monkeypatch.setattr(
        "app.ai.graph.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )


async def _run(workflow):
    return await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="What was revenue?",
        parent_state={
            "conversation_id": str(UUID(int=9)),
            "user_id": "owner",
            "device_id": "device-1",
            "context": {},
            "messages": [],
        },
    )


# ----------------------------------------------------------------------
# the gate runs
# ----------------------------------------------------------------------


async def test_a_worker_answer_is_validated_before_it_reaches_planning(monkeypatch):
    _install_retrieval(monkeypatch, "E1")
    response = await _run(_workflow(_worker_agent("Revenue rose [E1].")))

    assert "grounded_answer" in (response.metadata or {}), (
        "the worker returned an answer Planning will synthesize, and nothing "
        "validated it"
    )


async def test_a_worker_citation_to_unretrieved_evidence_is_neutralized(monkeypatch):
    """The bug this closes: an unchecked citation reaching the reader."""
    _install_retrieval(monkeypatch, "E1")
    response = await _run(_workflow(_worker_agent("Revenue rose [E9].")))

    assert "E9" not in response.message.content
    assert "Revenue rose" in response.message.content


async def test_a_worker_citation_the_worker_retrieved_survives(monkeypatch):
    _install_retrieval(monkeypatch, "E1")
    response = await _run(_workflow(_worker_agent("Revenue rose [E1].")))

    assert "[E1]" in response.message.content


async def test_a_worker_answer_carries_the_server_source_list(monkeypatch):
    _install_retrieval(monkeypatch, "E1")
    response = await _run(_workflow(_worker_agent("Revenue rose [E1].")))

    assert "report.pdf" in response.message.content


async def test_a_worker_is_graded_only_against_its_own_evidence(monkeypatch):
    """A worker's ids come from its own retrieval, not the parent turn's.

    Worker context is local by design; grading against a wider pool would let
    one worker authorize another's citation.
    """
    _install_retrieval(monkeypatch, "E2")
    response = await _run(_workflow(_worker_agent("Revenue rose [E1].")))

    assert "E1" not in response.message.content


# ----------------------------------------------------------------------
# it is the same gate, not a second one
# ----------------------------------------------------------------------


async def test_the_worker_uses_the_shared_gate_helper(monkeypatch):
    """Reject a parallel implementation, which is how one path stops checking."""
    _install_retrieval(monkeypatch, "E1")
    workflow = _workflow(_worker_agent("Revenue rose [E1]."))
    calls: list[str] = []

    original = MultiAgentWorkflow._apply_grounded_answer_gate

    async def spy(self, state, response):
        calls.append(str(response.message.content))
        return await original(self, state, response)

    monkeypatch.setattr(MultiAgentWorkflow, "_apply_grounded_answer_gate", spy)

    await _run(workflow)

    assert calls, "the worker did not route its answer through the shared gate"


async def test_a_failed_worker_answer_is_not_ground_into_something_else(monkeypatch):
    """An error reports itself. Grounding it would hide the failure."""
    _install_retrieval(monkeypatch, "E1")

    async def process_message(_message, _conversation_id):
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={},
            error="provider_unavailable",
        )

    agent = SimpleNamespace(
        process_message=process_message,
        agent_config_key="rag",
        tool_state_key="rag",
        agent_type=AgentType.RAG,
    )
    response = await _run(_workflow(agent))

    assert response.error == "provider_unavailable"
    assert "grounded_answer" not in (response.metadata or {})


async def test_a_worker_that_never_retrieved_publishes_no_citations(monkeypatch):
    """With nothing retrieved, every citation is unresolvable."""

    async def process_message(_message, _conversation_id):
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT, content="Revenue rose [E1]."
            ),
            metadata={},
        )

    agent = SimpleNamespace(
        process_message=process_message,
        agent_config_key="rag",
        tool_state_key="rag",
        agent_type=AgentType.RAG,
    )
    response = await _run(_workflow(agent))

    assert "E1" not in response.message.content
    assert "Revenue rose" in response.message.content


@pytest.mark.parametrize("agent_name", ["search_agent", "chat_agent"])
async def test_a_non_rag_worker_is_not_grounded(monkeypatch, agent_name):
    """Only RAG answers carry evidence, so only they have citations to check."""

    async def invoke_model_with_history(**_kwargs):
        return AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id=agent_name,
            message=AgentMessage(role=MessageRole.ASSISTANT, content="Plain answer [E1]."),
            metadata={},
        )

    agent = SimpleNamespace(
        invoke_model_with_history=invoke_model_with_history,
        agent_config_key=agent_name,
        tool_state_key=agent_name,
        agent_type=AgentType.CHAT,
    )
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agents = {agent_name: agent}

    response = await workflow._run_agent_in_isolated_context(
        agent_name=agent_name,
        task_prompt="hi",
        parent_state={
            "conversation_id": str(UUID(int=9)),
            "user_id": "owner",
            "context": {},
            "messages": [],
        },
    )

    assert "grounded_answer" not in (response.metadata or {})


# ----------------------------------------------------------------------
# the other half: the synthesis that publishes worker results
# ----------------------------------------------------------------------
#
# Grounding every worker is not enough. Planning reads their results and writes
# the public answer itself, so it can introduce a citation no worker ever
# retrieved. The design says a public synthesis carrying RAG evidence is
# validated before publication; without it, the stream filter would drop an
# invented marker while the stored answer kept it, and the reader and the
# database would disagree.


def _planning_workflow(final_text: str, monkeypatch):
    from app.ai.workflow import planning_loop

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    async def invoke_model_with_history(**_kwargs):
        return AgentResponse(
            agent_type=AgentType.PLANNING,
            agent_id="planning_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=final_text),
            metadata={},
        )

    workflow.planning_agent = SimpleNamespace(
        invoke_model_with_history=invoke_model_with_history
    )

    async def fake_ensure_map(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(planning_loop, "ensure_agent_tool_map", fake_ensure_map)
    return workflow


def _planning_state(*evidence_ids: str):
    from langchain_core.messages import AIMessage, HumanMessage

    return {
        "conversation_id": str(UUID(int=9)),
        "user_id": "owner",
        "messages": [
            HumanMessage(content="summarize the report"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "dispatch-1", "name": "dispatch_subagents", "args": {}}
                ],
            ),
        ],
        "context": {
            "tool_artifacts": [
                {
                    "tool_call_id": "dispatch-1",
                    "rag_evidence": {
                        "records": [
                            {
                                "evidence_id": evidence_id,
                                "document_id": str(UUID(int=1)),
                                "filename": "report.pdf",
                                "content": "revenue rose",
                            }
                            for evidence_id in evidence_ids
                        ],
                        "evidence_ids": list(evidence_ids),
                        "token_count": 5,
                        "omitted_count": 0,
                        "truncated_count": 0,
                    },
                }
            ]
        },
        "todos": [],
    }


async def test_a_planning_synthesis_is_grounded_before_publication(monkeypatch):
    workflow = _planning_workflow("Revenue rose [E1].", monkeypatch)
    calls: list[str] = []
    original = MultiAgentWorkflow._apply_grounded_answer_gate

    async def spy(self, state, response):
        calls.append(str(response.message.content))
        return await original(self, state, response)

    monkeypatch.setattr(MultiAgentWorkflow, "_apply_grounded_answer_gate", spy)

    await workflow._planning_node(_planning_state("E1"))

    assert calls, "the planning synthesis was published without grounding"


async def test_a_planning_synthesis_cannot_invent_a_citation(monkeypatch):
    """The stored answer must say what the streamed one said."""
    workflow = _planning_workflow("Revenue rose [E9].", monkeypatch)

    state = await workflow._planning_node(_planning_state("E1"))

    assert "E9" not in state["response"].message.content
    assert "Revenue rose" in state["response"].message.content


async def test_a_planning_synthesis_keeps_a_worker_backed_citation(monkeypatch):
    workflow = _planning_workflow("Revenue rose [E1].", monkeypatch)

    state = await workflow._planning_node(_planning_state("E1"))

    assert "[E1]" in state["response"].message.content


async def test_a_tool_calling_planning_step_is_not_grounded(monkeypatch):
    """Only a final answer is published; an intermediate step has no citations
    to check and grounding it would rewrite a tool-calling message."""
    from app.ai.workflow import planning_loop

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    async def invoke_model_with_history(**_kwargs):
        return AgentResponse(
            agent_type=AgentType.PLANNING,
            agent_id="planning_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{"id": "t1", "name": "dispatch_subagents", "args": {}}],
            ),
            metadata={},
        )

    workflow.planning_agent = SimpleNamespace(
        invoke_model_with_history=invoke_model_with_history
    )

    async def fake_ensure_map(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(planning_loop, "ensure_agent_tool_map", fake_ensure_map)

    calls: list[str] = []
    original = MultiAgentWorkflow._apply_grounded_answer_gate

    async def spy(self, state, response):
        calls.append(str(response.message.content))
        return await original(self, state, response)

    monkeypatch.setattr(MultiAgentWorkflow, "_apply_grounded_answer_gate", spy)

    await workflow._planning_node(_planning_state("E1"))

    assert calls == []


# ----------------------------------------------------------------------
# what this did and did not fix
# ----------------------------------------------------------------------
#
# The design's first complaint was "the planning worker contains a second RAG
# loop that does not invoke the graph RAG grounding gate". Two things were
# wrong there, and only one of them is fixed.
#
# Fixed: the grounding policy was duplicated by omission — one RAG path
# validated, the other did not. Every RAG answer now passes the same gate,
# whichever loop produced it.
#
# Not fixed: the *execution* is still duplicated. Two separate
# search_documents loops remain, with their own budget accounting, artifact
# handling, and iteration limits. Collapsing them is the Task 7 cutover, which
# needs the production RAG runtime that rag_execution.py was never given.


def test_the_two_rag_loops_are_still_separate_implementations():
    """Records the half that remains, so a green suite cannot imply otherwise.

    Delete this when the loops collapse into one — not by loosening it.
    """
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    graph = (repo_root / "app" / "ai" / "graph.py").read_text(encoding="utf-8")
    rag_loop = (repo_root / "app" / "ai" / "workflow" / "rag_loop.py").read_text(
        encoding="utf-8"
    )

    assert "execute_search_documents_action" in graph
    assert "execute_search_documents_action" in rag_loop, (
        "the two search loops merged — remove this test and the ledger entry "
        "in test_routing_legacy_removal.py"
    )


def test_every_rag_answer_path_shares_one_gate():
    """One grounding implementation, called from each place an answer is made."""
    import pathlib
    import re

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    callers = {
        path.name
        for path in sorted((repo_root / "app").rglob("*.py"))
        if re.search(r"self\._apply_grounded_answer_gate\(", path.read_text(encoding="utf-8"))
    }

    assert callers == {"graph.py", "planning_loop.py", "rag_loop.py"}, (
        f"a RAG answer path was added or removed without review: {sorted(callers)}"
    )
