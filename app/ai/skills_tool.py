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

import json
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.client_device_service import ClientDeviceService

from .skill_resolver import (
    get_available_skill_summaries as get_resolved_skill_summaries,
)
from .skill_resolver import (
    get_bound_device_session,
)
from .skill_resolver import (
    resolve_skill_reference as resolve_runtime_skill_reference,
)
from .tool_context import get_tool_context

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


def _format_runtime_skill_error(response: dict[str, Any]) -> str:
    error_context = response.get("error_context")
    if isinstance(error_context, dict):
        message = str(
            error_context.get("message")
            or response.get("error")
            or "Unknown client-side skill error"
        )
        code = error_context.get("code")
        detail = error_context.get("detail")
        extras: list[str] = []
        if code:
            extras.append(f"code={code}")
        if detail not in (None, "", {}):
            extras.append("detail=" + json.dumps(detail, indent=2, ensure_ascii=False, default=str))
        if extras:
            return f"{message} ({'; '.join(extras)})"
        return message

    error_message = response.get("error")
    if isinstance(error_message, str) and error_message:
        return error_message
    return "Unknown client-side skill error"


def get_available_skill_summaries(
    *,
    user_id: str | None,
    device_id: str | None,
    allowed_skill_refs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return prompt-safe skill summaries for the active execution scope.

    ``allowed_skill_refs`` (custom agents) restricts the result to selected
    skills; None preserves base-agent behavior (all skills visible).
    """
    return get_resolved_skill_summaries(
        user_id=user_id, device_id=device_id, allowed_skill_refs=allowed_skill_refs
    )


# ── Tool factory ────────────────────────────────────────────────────


def create_activate_skill_tool(
    *,
    user_id: str | None,
    device_id: str | None,
    allowed_skill_refs: list[dict[str, Any]] | None = None,
):
    """
    Create the ``activate_skill`` tool that the LLM can call to load
    the full instructions for a specific skill.

    Skill content is fetched from the currently connected client device
    over the runtime bridge so the canonical backend does not need direct
    access to client-local skill files.

    When ``allowed_skill_refs`` is provided (custom agents), only skills in
    that allowlist can be resolved and activated; any other name is rejected.
    """

    session = get_bound_device_session(user_id=user_id, device_id=device_id)
    bound_user_id = str(session.user_id) if session is not None else str(user_id or "")
    bound_device_id = str(session.device_id) if session is not None else str(device_id or "")
    bound_session_id = session.session_id if session is not None else None

    @tool(args_schema=ActivateSkillInput)
    async def activate_skill(skill_name: str) -> str:
        """Load the full instructions for a skill by name.

        Use this when you determine that a skill's guidelines are relevant
        to the current user request.  The returned text contains the
        complete instructions you should follow for that skill.
        """
        resolved_skill, resolution_error = resolve_runtime_skill_reference(
            skill_name=skill_name,
            user_id=bound_user_id,
            device_id=bound_device_id,
            allowed_skill_refs=allowed_skill_refs,
        )
        if resolution_error:
            return resolution_error
        if resolved_skill is None:
            return "Error: skill resolution failed."

        ctx = get_tool_context()
        context_device_id = str(ctx.device_id or bound_device_id or "")
        if bound_device_id and context_device_id and context_device_id != bound_device_id:
            return (
                "Error: client-side skill activation was requested for a different device session "
                "than the active run."
            )

        session = get_bound_device_session(user_id=bound_user_id, device_id=bound_device_id)
        if session is None:
            return "Error: client-side skills are not available because the device is disconnected."

        expected_session_id = resolved_skill.bound_session_id or bound_session_id
        if expected_session_id and session.session_id != expected_session_id:
            return (
                "Error: the client device session changed after skills were bound. "
                "Retry from the active device session."
            )

        response = await ClientDeviceService.dispatch_tool_call(
            user_id=bound_user_id,
            device_id=bound_device_id,
            tool_name="activate_skill",
            qualified_tool_id="client_skill::activate",
            arguments={"skill_name": resolved_skill.name},
            timeout_seconds=settings.client_runtime_ws_timeout_seconds,
            bound_session_id=expected_session_id,
        )

        if not response.get("success", False):
            return f"Error: {_format_runtime_skill_error(response)}"

        result = response.get("result")
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and isinstance(result.get("content"), str):
            return str(result["content"])
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)

    return activate_skill
