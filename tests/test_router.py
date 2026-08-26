"""Contract tests for the temporary ``Router`` compatibility adapter.

The adapter exists only until the routing-v2 cutover removes it. What it must
*not* do is the point of these tests: no provider SDK client, no free-text
parsing, no keyword or explicit-name matching, and no ``chat_agent`` fallback.
Semantic routing behavior is covered by ``tests/test_routing_service.py``.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest

from app.ai.agents.router import Router
from app.ai.schemas import AgentMessage, MessageRole
from app.ai.workflow.contracts import RoutingDecision, WorkflowError, WorkflowRoutingException

ROUTER_SOURCE = pathlib.Path("app/ai/agents/router.py").read_text(encoding="utf-8")


class FakeRoutingService:
    def __init__(self, *, decision=None, error=None):
        self.decision = decision
        self.error = error
        self.calls = 0
        self.last_inventory = None
        self.last_context = None

    async def route(self, context, inventory, *, user_id, model_request, request_id):
        self.calls += 1
        self.last_inventory = inventory
        self.last_context = context
        if self.error is not None:
            raise self.error
        return self.decision


def _message(content: str = "what changed today") -> AgentMessage:
    return AgentMessage(
        role=MessageRole.USER,
        content=content,
        metadata={"user_id": None, "device_id": "device-1", "persona": None},
    )


def _router(service) -> Router:
    return Router(recorder=None, routing_service=service)


async def test_adapter_returns_the_service_decision():
    service = FakeRoutingService(
        decision=RoutingDecision(agent_id="search_agent", confidence=0.9, reason="current info")
    )
    agent_id = await _router(service).route_message(
        _message(), ["chat_agent", "search_agent"], has_documents=False
    )
    assert agent_id == "search_agent"
    assert service.calls == 1


async def test_adapter_passes_custom_agents_as_inventory_targets():
    service = FakeRoutingService(
        decision=RoutingDecision(agent_id="custom_agent:alpha", confidence=0.8, reason="attached")
    )
    agent_id = await _router(service).route_message(
        _message(),
        ["chat_agent", "custom_agent:alpha"],
        custom_agent_descriptors=[
            {"runtime_agent_id": "custom_agent:alpha", "name": "Alpha", "description": "alpha"}
        ],
    )
    assert agent_id == "custom_agent:alpha"
    assert service.last_inventory.get("custom_agent:alpha") is not None


async def test_adapter_carries_state_as_context_not_preselection():
    service = FakeRoutingService(
        decision=RoutingDecision(agent_id="chat_agent", confidence=0.5, reason="general")
    )
    await _router(service).route_message(
        _message(),
        ["chat_agent", "canvas_agent", "planning_agent", "rag_agent"],
        has_documents=True,
        planning_mode_enabled=True,
        has_existing_plan=True,
        active_canvas={"artifact_id": "canvas:main", "revision": 3, "title": "Site"},
    )

    context = service.last_context
    assert context.planning.planning_mode_enabled is True
    assert context.planning.has_existing_plan is True
    assert context.active_canvas.title == "Site"
    # Planning state and an active canvas are context. They did not preselect.
    assert service.calls == 1


async def test_adapter_never_falls_back_to_chat_on_failure():
    error = WorkflowRoutingException(
        WorkflowError(code="routing_provider_unavailable", retriable=True, request_id="request-1")
    )
    service = FakeRoutingService(error=error)

    with pytest.raises(WorkflowRoutingException) as exc:
        await _router(service).route_message(_message(), ["chat_agent", "search_agent"])

    assert exc.value.error.code == "routing_provider_unavailable"


def test_router_module_has_no_provider_sdk_client():
    assert "google.genai" not in ROUTER_SOURCE
    assert "from google import genai" not in ROUTER_SOURCE
    assert "genai.Client" not in ROUTER_SOURCE


def test_router_module_has_no_free_text_parser_or_keyword_matcher():
    assert "_extract_agent_name" not in ROUTER_SOURCE
    assert "_match_explicit_custom_agent" not in ROUTER_SOURCE
    assert "tokenize_text" not in ROUTER_SOURCE
    tree = ast.parse(ROUTER_SOURCE)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "re" not in imported


def test_router_module_never_returns_a_hard_coded_agent():
    tree = ast.parse(ROUTER_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            assert node.value.value != "chat_agent"


def test_router_constructs_without_provider_credentials(monkeypatch):
    """Construction must not require or probe a credential."""
    router = Router(recorder=None, routing_service=FakeRoutingService())
    assert isinstance(router.service, FakeRoutingService)
    assert isinstance(router.model_name, str)


def test_structured_router_prompt_is_owned_by_the_routing_module():
    from app.ai.workflow.routing import ROUTER_SYSTEM_PROMPT

    assert "structured" in ROUTER_SYSTEM_PROMPT.lower()
    assert "untrusted" in ROUTER_SYSTEM_PROMPT.lower()
    # The instruction must not restate a phrase-matching rule.
    assert "keyword" in ROUTER_SYSTEM_PROMPT.lower()


def test_router_no_longer_exposes_stickiness_helpers():
    from app.ai.workflow import custom_agents as custom_agents_module

    source = pathlib.Path(custom_agents_module.__file__).read_text(encoding="utf-8")
    assert "_sticky_custom_agent" not in source
    assert "_match_explicit_custom_agent" not in source


def test_router_adapter_is_a_thin_delegation_layer():
    """Guard against the adapter growing routing logic of its own."""
    tree = ast.parse(ROUTER_SOURCE)
    router_class = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "Router"
    )
    method_names = {
        node.name
        for node in router_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert method_names <= {"__init__", "route_message", "service"}


def test_router_adapter_signature_is_unchanged_for_the_legacy_graph():
    import inspect

    parameters = inspect.signature(Router.route_message).parameters
    for name in (
        "message",
        "available_agents",
        "has_documents",
        "planning_mode_enabled",
        "has_existing_plan",
        "custom_agent_descriptors",
        "active_canvas",
    ):
        assert name in parameters


def test_legacy_router_settings_still_resolve():
    from app.core.config import settings

    assert settings.router_model
    assert settings.router_provider == "gemini"
    assert isinstance(SimpleNamespace(model=settings.router_model).model, str)
