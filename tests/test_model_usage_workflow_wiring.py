"""Workflow wiring contract test for the model-usage recorder (Task 6).

Constructs a real ``MultiAgentWorkflow`` via ``create_workflow`` with a
sentinel recorder object (no container, no globals) and asserts constructor
injection carries it to every consumer: the router, each built-in
``BaseAgent``, and a lazily built ``CustomAgent`` from the custom-agents mixin
path (``app/ai/workflow/custom_agents.py``). Mirrors the collaborator mocking
used in ``tests/test_graph_refactor_contract.py`` -- no live DB or Qdrant.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.ai.graph import create_workflow

_CUSTOM_AGENT_RUNTIME_ID = "custom_agent:11111111-1111-1111-1111-111111111111"


class _SentinelRecorder:
    """Distinct identity so a stray default/None couldn't accidentally pass."""


def _build_workflow(recorder):
    return create_workflow(
        qdrant_client=MagicMock(),
        embedding_service=MagicMock(),
        model_usage_recorder=recorder,
    )


def test_recorder_reaches_the_routing_service():
    recorder = _SentinelRecorder()
    workflow = _build_workflow(recorder)
    assert workflow.routing_service._usage_recorder is recorder


def test_recorder_reaches_every_built_in_agent():
    recorder = _SentinelRecorder()
    workflow = _build_workflow(recorder)
    assert workflow.agents
    for agent in workflow.agents.values():
        assert agent.recorder is recorder


def test_recorder_reaches_lazily_built_custom_agent():
    recorder = _SentinelRecorder()
    workflow = _build_workflow(recorder)
    state = {
        "custom_agents": {
            _CUSTOM_AGENT_RUNTIME_ID: {
                "id": "11111111-1111-1111-1111-111111111111",
                "runtime_agent_id": _CUSTOM_AGENT_RUNTIME_ID,
                "name": "Test Custom Agent",
                "description": "A test custom agent.",
                "prompt": "You are a test agent.",
            }
        }
    }

    custom_agent = workflow._build_custom_agent(state, _CUSTOM_AGENT_RUNTIME_ID)

    assert custom_agent is not None
    assert custom_agent.recorder is recorder
