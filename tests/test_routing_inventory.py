from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ai.workflow.inventory import (
    AgentDescriptor,
    RoutingInventory,
    build_routing_inventory,
    inventory_version,
)


def _descriptor(agent_id: str, **overrides) -> AgentDescriptor:
    payload = {
        "agent_id": agent_id,
        "display_name": agent_id.replace("_", " ").title(),
        "capability_description": f"capability for {agent_id}",
        "enabled": True,
        "attached": True,
        "kind": "custom" if agent_id.startswith("custom_agent:") else "base",
    }
    payload.update(overrides)
    return AgentDescriptor(**payload)


def test_agent_descriptor_is_frozen_and_closed():
    descriptor = _descriptor("chat_agent")
    with pytest.raises(ValidationError):
        descriptor.enabled = False
    with pytest.raises(ValidationError):
        AgentDescriptor(
            agent_id="chat_agent",
            display_name="Chat",
            capability_description="c",
            enabled=True,
            kind="base",
            selected_agent="legacy",
        )
    with pytest.raises(ValidationError):
        _descriptor("chat_agent", kind="sticky")


def test_inventory_version_is_stable_and_order_independent():
    descriptor_a = _descriptor("chat_agent")
    descriptor_b = _descriptor("search_agent")
    assert inventory_version([descriptor_b, descriptor_a]) == inventory_version(
        [descriptor_a, descriptor_b]
    )


def test_inventory_version_changes_when_capabilities_change():
    baseline = inventory_version([_descriptor("chat_agent")])
    changed = inventory_version([_descriptor("chat_agent", capability_description="different")])
    disabled = inventory_version([_descriptor("chat_agent", enabled=False)])
    assert baseline != changed
    assert baseline != disabled


def test_inventory_resolves_base_and_custom_node_names():
    inventory = RoutingInventory.from_descriptors(
        [_descriptor("chat_agent"), _descriptor("custom_agent:alpha")]
    )
    assert inventory.resolve_node("chat_agent") == "chat_agent"
    assert inventory.resolve_node("custom_agent:alpha") == "custom_agent"
    with pytest.raises(KeyError):
        inventory.resolve_node("missing_agent")


def test_inventory_lookup_reports_enablement_and_attachment():
    inventory = RoutingInventory.from_descriptors(
        [
            _descriptor("chat_agent"),
            _descriptor("canvas_agent", enabled=False),
            _descriptor("custom_agent:alpha", attached=False),
        ]
    )
    assert inventory.get("chat_agent") is not None
    assert inventory.get("nope") is None
    assert inventory.is_routable("chat_agent") is True
    assert inventory.is_routable("canvas_agent") is False
    assert inventory.is_routable("custom_agent:alpha") is False
    assert inventory.is_routable("missing_agent") is False


def test_inventory_rejects_duplicate_agent_ids():
    with pytest.raises(ValueError):
        RoutingInventory.from_descriptors([_descriptor("chat_agent"), _descriptor("chat_agent")])


def test_inventory_agents_are_sorted_by_agent_id():
    inventory = RoutingInventory.from_descriptors(
        [_descriptor("search_agent"), _descriptor("chat_agent")]
    )
    assert [descriptor.agent_id for descriptor in inventory.agents] == [
        "chat_agent",
        "search_agent",
    ]


def test_build_routing_inventory_uses_registry_and_attached_custom_agents():
    inventory = build_routing_inventory(
        base_agent_ids=["chat_agent", "search_agent"],
        custom_agents={
            "custom_agent:alpha": {
                "runtime_agent_id": "custom_agent:alpha",
                "name": "Alpha",
                "description": "handles alpha work",
            }
        },
    )
    agent_ids = [descriptor.agent_id for descriptor in inventory.agents]
    assert agent_ids == ["chat_agent", "custom_agent:alpha", "search_agent"]

    custom = inventory.get("custom_agent:alpha")
    assert custom is not None
    assert custom.kind == "custom"
    assert custom.display_name == "Alpha"
    assert custom.capability_description == "handles alpha work"

    base = inventory.get("chat_agent")
    assert base is not None
    assert base.kind == "base"
    assert base.capability_description


def test_build_routing_inventory_ignores_unattached_custom_entries():
    inventory = build_routing_inventory(
        base_agent_ids=["chat_agent"],
        custom_agents={"custom_agent:alpha": {"runtime_agent_id": "custom_agent:alpha"}},
        attached_custom_agent_ids={"custom_agent:beta"},
    )
    assert inventory.get("custom_agent:alpha") is None


def test_build_routing_inventory_bounds_custom_agent_count():
    custom_agents = {
        f"custom_agent:{index}": {
            "runtime_agent_id": f"custom_agent:{index}",
            "name": f"Agent {index}",
            "description": "d",
        }
        for index in range(40)
    }
    inventory = build_routing_inventory(
        base_agent_ids=["chat_agent"],
        custom_agents=custom_agents,
        max_custom_agents=5,
    )
    custom_ids = [d.agent_id for d in inventory.agents if d.kind == "custom"]
    assert len(custom_ids) == 5


def test_inventory_module_does_not_read_user_text():
    """The inventory is a registry projection; it must not inspect messages."""
    import ast
    import pathlib

    source = pathlib.Path("app/ai/workflow/inventory.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

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

    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "lower" not in called_attributes
    assert "tokenize_text" not in {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
