"""Inter-agent delegation as a LangGraph parent command.

The model chooses a target through the tool's schema. The tool then records a
*pending transition* in parent state and appends the matching ``ToolMessage``,
so the tool-call history stays valid and the orchestrator never has to parse a
control decision back out of a tool result.

Accepting or refusing the transition is not this tool's job — that belongs to
the single resolver in ``app.ai.workflow.transitions``.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from app.ai.workflow.contracts import PendingTransition

__all__ = [
    "HAND_OFF_TOOL_NAME",
    "HandOffInput",
    "create_hand_off_tool",
    "handoff_message_id",
]

HAND_OFF_TOOL_NAME = "hand_off"
TRANSITION_RESOLVER_NODE = "resolve_transition"

_HAND_OFF_DOC = (
    "Hand the conversation off to a different specialist agent.\n\n"
    "Use this tool when the current request, or a distinct part of the request, "
    "is better suited to another listed target. Do NOT delegate if no listed "
    "target is better suited.\n\n"
    "Args:\n"
    "    target_agent: The agent id to delegate to.{targets}\n"
    "    reason: A short factual reason for the delegation.\n"
)


def handoff_message_id(tool_call_id: str) -> str:
    """The deterministic id of a handoff's paired ``ToolMessage``.

    Both the request marker and any refusal use this id, so the message reducer
    replaces the marker instead of leaving two messages paired to one call.
    """
    return f"handoff:{tool_call_id}"


class HandOffInput(BaseModel):
    """What the model chooses. ``tool_call_id`` is injected, never model-authored."""

    model_config = ConfigDict(extra="forbid")

    target_agent: str = Field(..., description="The agent id to delegate to.")
    reason: str = Field(
        default="delegating to a better-suited specialist",
        description="Short factual reason for the delegation.",
        max_length=500,
    )
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(default="")


def create_hand_off_tool(
    *,
    source_agent_id: str,
    allowed_targets: list[str] | None = None,
    target_descriptions: dict[str, str] | None = None,
) -> StructuredTool:
    """Build the live handoff tool for one specialist.

    The description lists only targets reachable from ``source_agent_id``. The
    schema still accepts a free-form string so dynamic custom runtime ids work;
    the resolver re-validates the chosen target against the live inventory.
    """
    targets = list(allowed_targets or [])
    descriptions = target_descriptions or {}
    if targets:
        lines = [
            f"      - {target}" + (f": {descriptions[target]}" if descriptions.get(target) else "")
            for target in targets
        ]
        targets_block = " Valid targets:\n" + "\n".join(lines)
    else:
        targets_block = ""

    def _hand_off(
        target_agent: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        reason: str = "delegating to a better-suited specialist",
    ) -> Command:
        message_id = handoff_message_id(tool_call_id)
        return Command(
            graph=Command.PARENT,
            update={
                "pending_transition": PendingTransition(
                    from_agent_id=source_agent_id,
                    to_agent_id=target_agent,
                    tool_call_id=tool_call_id,
                    tool_message_id=message_id,
                    reason=(reason or "delegating to a better-suited specialist")[:500],
                ),
                "messages": [
                    ToolMessage(
                        id=message_id,
                        content="handoff_requested",
                        name=HAND_OFF_TOOL_NAME,
                        tool_call_id=tool_call_id,
                    )
                ],
            },
            goto=TRANSITION_RESOLVER_NODE,
        )

    return StructuredTool.from_function(
        func=_hand_off,
        name=HAND_OFF_TOOL_NAME,
        description=_HAND_OFF_DOC.format(targets=targets_block),
        args_schema=HandOffInput,
        # This tool's result is a control decision, not a value. The framework
        # tool node knows how to turn it into a parent command; the product
        # tool pipeline would render it as text and the turn would carry on
        # with the wrong agent. The marker is what tells them apart.
        metadata={"tool_origin": "internal", "returns_control_command": True},
    )
