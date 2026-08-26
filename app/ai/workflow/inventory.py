"""Live specialist inventory for routing and transition validation.

The inventory is a pure projection of the agent registry plus the
authenticated attached-custom-agent map. It answers only control-plane
questions — does this agent exist, is it enabled, is it attached, which graph
node runs it — and never inspects user text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ai.agent_metadata import BASE_AGENT_CAPABILITIES, BASE_AGENT_DISPLAY_NAMES

__all__ = [
    "CUSTOM_AGENT_NODE",
    "CUSTOM_AGENT_PREFIX",
    "AgentDescriptor",
    "RoutingInventory",
    "build_routing_inventory",
    "inventory_version",
]

CUSTOM_AGENT_PREFIX = "custom_agent:"
CUSTOM_AGENT_NODE = "custom_agent"

AgentKind = Literal["base", "custom"]


class AgentDescriptor(BaseModel):
    """One routable specialist as the router sees it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1, max_length=160)
    display_name: str = Field(min_length=1, max_length=200)
    capability_description: str = Field(default="", max_length=2000)
    enabled: bool = True
    attached: bool = True
    kind: AgentKind


def _canonical_payload(descriptors: Iterable[AgentDescriptor]) -> str:
    ordered = sorted(descriptors, key=lambda descriptor: descriptor.agent_id)
    parts = [
        "\x1f".join(
            (
                descriptor.agent_id,
                descriptor.kind,
                descriptor.display_name,
                descriptor.capability_description,
                "1" if descriptor.enabled else "0",
                "1" if descriptor.attached else "0",
            )
        )
        for descriptor in ordered
    ]
    return "\x1e".join(parts)


def inventory_version(descriptors: Iterable[AgentDescriptor]) -> str:
    """Stable, order-independent hash of the routable inventory."""
    payload = _canonical_payload(descriptors)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RoutingInventory(BaseModel):
    """Immutable snapshot of every routable specialist for one request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1, max_length=128)
    agents: tuple[AgentDescriptor, ...] = ()

    @classmethod
    def from_descriptors(cls, descriptors: Iterable[AgentDescriptor]) -> RoutingInventory:
        ordered = sorted(descriptors, key=lambda descriptor: descriptor.agent_id)
        seen: set[str] = set()
        for descriptor in ordered:
            if descriptor.agent_id in seen:
                raise ValueError(f"duplicate agent_id in inventory: {descriptor.agent_id!r}")
            seen.add(descriptor.agent_id)
        return cls(version=inventory_version(ordered), agents=tuple(ordered))

    def get(self, agent_id: str | None) -> AgentDescriptor | None:
        if not agent_id:
            return None
        for descriptor in self.agents:
            if descriptor.agent_id == agent_id:
                return descriptor
        return None

    def is_routable(self, agent_id: str | None) -> bool:
        """A target is routable only when it exists, is enabled, and is attached."""
        descriptor = self.get(agent_id)
        return bool(descriptor and descriptor.enabled and descriptor.attached)

    def resolve_node(self, agent_id: str) -> str:
        """Map an agent ID to its graph node name.

        Custom agents share one parametrized wrapper node, so the agent ID is
        never assumed to be a node name.
        """
        descriptor = self.get(agent_id)
        if descriptor is None:
            raise KeyError(f"unknown agent_id: {agent_id!r}")
        return CUSTOM_AGENT_NODE if descriptor.kind == "custom" else descriptor.agent_id

    def routable_ids(self) -> tuple[str, ...]:
        return tuple(
            descriptor.agent_id
            for descriptor in self.agents
            if descriptor.enabled and descriptor.attached
        )

    def reachable_from(self, agent_id: str | None) -> tuple[AgentDescriptor, ...]:
        """Handoff targets available to ``agent_id`` (everything but itself)."""
        return tuple(
            descriptor
            for descriptor in self.agents
            if descriptor.enabled and descriptor.attached and descriptor.agent_id != agent_id
        )


def _base_descriptor(agent_id: str, *, enabled: bool = True) -> AgentDescriptor:
    return AgentDescriptor(
        agent_id=agent_id,
        display_name=BASE_AGENT_DISPLAY_NAMES.get(agent_id, agent_id.replace("_", " ").title()),
        capability_description=BASE_AGENT_CAPABILITIES.get(agent_id, ""),
        enabled=enabled,
        attached=True,
        kind="base",
    )


def _custom_descriptor(runtime_agent_id: str, entry: Mapping[str, Any]) -> AgentDescriptor:
    name = entry.get("name") or runtime_agent_id
    description = entry.get("description") or ""
    return AgentDescriptor(
        agent_id=runtime_agent_id,
        display_name=str(name)[:200] or runtime_agent_id,
        capability_description=str(description)[:2000],
        enabled=bool(entry.get("enabled", True)),
        attached=True,
        kind="custom",
    )


def build_routing_inventory(
    *,
    base_agent_ids: Sequence[str],
    custom_agents: Mapping[str, Any] | None,
    attached_custom_agent_ids: set[str] | None = None,
    disabled_agent_ids: set[str] | None = None,
    max_custom_agents: int | None = None,
) -> RoutingInventory:
    """Build the live inventory from the registry and attached custom agents.

    ``custom_agents`` is the authenticated attachment map. When
    ``attached_custom_agent_ids`` is supplied it further restricts the map so a
    stale state entry can never make an unattached agent routable.
    """
    disabled = disabled_agent_ids or set()
    descriptors: list[AgentDescriptor] = [
        _base_descriptor(agent_id, enabled=agent_id not in disabled)
        for agent_id in base_agent_ids
        if agent_id and not agent_id.startswith(CUSTOM_AGENT_PREFIX)
    ]

    custom_entries: list[tuple[str, Mapping[str, Any]]] = []
    for runtime_id, entry in (custom_agents or {}).items():
        if not isinstance(entry, Mapping):
            continue
        resolved_id = str(entry.get("runtime_agent_id") or runtime_id)
        if not resolved_id.startswith(CUSTOM_AGENT_PREFIX):
            continue
        if attached_custom_agent_ids is not None and resolved_id not in attached_custom_agent_ids:
            continue
        custom_entries.append((resolved_id, entry))

    custom_entries.sort(key=lambda item: (int(item[1].get("agent_order", 0) or 0), item[0]))
    if max_custom_agents is not None and max_custom_agents >= 0:
        custom_entries = custom_entries[:max_custom_agents]

    descriptors.extend(
        _custom_descriptor(resolved_id, entry) for resolved_id, entry in custom_entries
    )
    return RoutingInventory.from_descriptors(descriptors)
