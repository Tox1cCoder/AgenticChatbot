"""
Inter-agent delegation tool.

Allows any agent to hand off the current conversation to a more suitable agent
by calling the ``hand_off`` tool.  The graph's ``_tool_node`` detects the
structured output and re-routes execution to the target agent.

Targets are no longer a fixed ``Literal``: a per-run dynamic tool can be built
with ``create_hand_off_tool`` so attached custom agents (``custom_agent:<uuid>``)
become valid delegation targets. The graph validates every target at execution
time against base agents + attached custom agents and returns a structured tool
error for unknown/unattached targets.
"""

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

_HAND_OFF_DOC = (
    "Hand the conversation off to a different specialist agent.\n\n"
    "Use this tool when the current request, or a distinct part of the request, "
    "is better suited to another listed target. Do NOT delegate if no listed "
    "target is better suited.\n\n"
    "Args:\n"
    "    target_agent: The agent id to delegate to.{targets}\n\n"
    "Returns: a JSON object consumed by the orchestrator to re-route execution."
)


class HandOffInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_agent: str = Field(..., description="The agent id to delegate to.")


def _hand_off_impl(target_agent: str) -> str:
    return json.dumps({"hand_off": target_agent})


def create_hand_off_tool(
    allowed_targets: list[str] | None = None,
    target_descriptions: dict[str, str] | None = None,
) -> StructuredTool:
    """Build a hand_off tool whose description lists the valid targets.

    The schema accepts a free-form ``target_agent`` string so dynamic custom
    runtime ids work; the graph re-validates the target at execution time.
    """
    targets = list(allowed_targets or [])
    descriptions = target_descriptions or {}
    if targets:
        lines = []
        for target in targets:
            desc = descriptions.get(target)
            lines.append(f"      - {target}" + (f": {desc}" if desc else ""))
        targets_block = " Valid targets:\n" + "\n".join(lines)
    else:
        targets_block = ""

    return StructuredTool.from_function(
        func=_hand_off_impl,
        name="hand_off",
        description=_HAND_OFF_DOC.format(targets=targets_block),
        args_schema=HandOffInput,
    )
