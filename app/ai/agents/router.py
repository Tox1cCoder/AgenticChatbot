"""Temporary compatibility adapter over :class:`RoutingService`.

The pre-v2 graph still imports this module, so it stays until Task 11 of the
routing-v2 cutover deletes it. It deliberately contains **no** provider SDK
client, no free-text parsing, no keyword or explicit-name matching, and no
``chat_agent`` fallback: routing failures raise the typed
``WorkflowRoutingException`` instead of silently selecting an agent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...core.config import settings
from ..model_factory import ModelFactory
from ..schemas import AgentMessage
from ..time_context import build_runtime_time_context_block
from ..workflow.inventory import build_routing_inventory
from ..workflow.routing import (
    RoutingContextBuilder,
    RoutingContextRequest,
    RoutingDecisionValidator,
    RoutingService,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...usage.recorder import ModelUsageRecorder


class Router:
    """Adapter that maps the legacy ``route_message`` call onto ``RoutingService``."""

    def __init__(
        self,
        recorder: ModelUsageRecorder | None = None,
        *,
        routing_service: RoutingService | None = None,
        runtime_model_resolver: Any = None,
    ):
        self.model_name = settings.router_model
        self.recorder = recorder
        self._service = routing_service or RoutingService(
            runtime_model_resolver=runtime_model_resolver,
            model_factory=ModelFactory,
            settings=settings,
            validator=RoutingDecisionValidator(),
            usage_recorder=recorder,
        )
        self._context_builder = RoutingContextBuilder(
            history_provider=None,
            document_repository=None,
            settings=settings,
        )

    @property
    def service(self) -> RoutingService:
        return self._service

    async def route_message(
        self,
        message: AgentMessage,
        available_agents: list[str],
        has_documents: bool = False,
        planning_mode_enabled: bool = False,
        has_existing_plan: bool = False,
        custom_agent_descriptors: list[dict] | None = None,
        active_canvas: dict | None = None,
        request_id: str | None = None,
    ) -> str:
        """Route one turn through ``RoutingService``.

        Raises ``WorkflowRoutingException`` when routing cannot complete. It
        never returns a guessed agent.
        """
        metadata = message.metadata or {}
        custom_agents = {
            str(descriptor.get("runtime_agent_id")): descriptor
            for descriptor in (custom_agent_descriptors or [])
            if descriptor.get("runtime_agent_id")
        }
        inventory = build_routing_inventory(
            base_agent_ids=[
                agent_id
                for agent_id in available_agents
                if not str(agent_id).startswith("custom_agent:")
            ],
            custom_agents=custom_agents,
        )

        context = await self._context_builder.build(
            RoutingContextRequest(
                message=message.content or "",
                inventory=inventory,
                user_id=metadata.get("user_id"),
                device_id=metadata.get("device_id"),
                persona=metadata.get("persona"),
                active_canvas=active_canvas,
                planning={
                    "planning_mode_enabled": planning_mode_enabled,
                    "has_existing_plan": has_existing_plan,
                },
                # Server-owned clock context. Trusted data, never an instruction.
                runtime_time=build_runtime_time_context_block().strip() or None,
                locale=metadata.get("locale"),
            )
        )

        decision = await self._service.route(
            context,
            inventory,
            user_id=metadata.get("user_id"),
            model_request=metadata.get("model_request"),
            request_id=request_id or str(metadata.get("request_id") or "legacy-router"),
        )
        return decision.agent_id
