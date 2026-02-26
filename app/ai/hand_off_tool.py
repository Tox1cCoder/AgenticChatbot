"""
Inter-agent delegation tool.

Allows any agent to hand off the current conversation to a more suitable agent
by calling the ``hand_off`` tool.  The graph's ``_tool_node`` detects the
structured output and re-routes execution to the target agent.
"""

import json
from typing import Literal

from langchain_core.tools import tool


# Canonical list of agents that can be delegation targets.
DELEGATABLE_AGENTS = (
    "chat_agent",
    "rag_agent",
    "search_agent",
    "image_generator_agent",
    "planning_agent",
    "canvas_agent",
)

# Maximum number of inter-agent delegations per user turn.
MAX_DELEGATION_DEPTH = 3


@tool
def hand_off(
    target_agent: Literal[
        "chat_agent",
        "rag_agent",
        "search_agent",
        "image_generator_agent",
        "planning_agent",
        "canvas_agent",
    ],
    reason: str,
) -> str:
    """Hand the conversation off to a different specialist agent.

    Use this tool ONLY when the current request clearly falls outside your
    expertise and another agent is better suited.  Do NOT delegate if you can
    handle the request yourself.

    Args:
        target_agent: The agent to delegate to.
        reason: A brief explanation of *why* delegation is appropriate.

    Returns:
        JSON object consumed by the orchestrator to re-route execution.
    """
    return json.dumps({"hand_off": target_agent, "reason": reason})
