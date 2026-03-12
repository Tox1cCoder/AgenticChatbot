"""
Skills Tool – progressive disclosure for the skills system.

Instead of dumping every active skill's full Markdown body into the system
prompt, only skill *names* and *descriptions* are injected.  The LLM can
then call ``activate_skill`` to load the full instructions for any skill
it deems relevant to the current conversation.

This mirrors the deferred-tool-loading pattern used for MCP tools:
small summary upfront, full content on demand.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .skills_registry import get_skills_registry

logger = logging.getLogger(__name__)


# ── Pydantic schemas ────────────────────────────────────────────────


class ActivateSkillInput(BaseModel):
    """Input schema for the activate_skill tool."""

    skill_name: str = Field(
        description=(
            "The exact name of the skill to load (as shown in the "
            "Available Skills list in your system prompt)."
        ),
    )


# ── Tool factory ────────────────────────────────────────────────────


def create_activate_skill_tool():
    """
    Create the ``activate_skill`` tool that the LLM can call to load
    the full instructions for a specific skill.

    The tool is self-executing: it reads directly from the
    :class:`SkillsRegistry` singleton and returns the Markdown body.
    No dedicated graph node is required — the standard ``_tool_node``
    execution path handles it.
    """

    @tool(args_schema=ActivateSkillInput)
    def activate_skill(skill_name: str) -> str:
        """Load the full instructions for a skill by name.

        Use this when you determine that a skill's guidelines are relevant
        to the current user request.  The returned text contains the
        complete instructions you should follow for that skill.
        """
        try:
            registry = get_skills_registry()
            skill = registry.get_skill(skill_name)
        except KeyError:
            available = [s.name for s in get_skills_registry().get_active_skills()]
            return (
                f"Error: skill '{skill_name}' not found. "
                f"Available skills: {', '.join(available) if available else '(none)'}"
            )

        if not skill.enabled:
            return (
                f"Error: skill '{skill_name}' exists but is currently disabled. "
                "Only enabled skills can be activated."
            )

        # Return the full Markdown body
        return f"── Skill: {skill.name} ──\n\n{skill.content}\n\n── End Skill: {skill.name} ──"

    return activate_skill
