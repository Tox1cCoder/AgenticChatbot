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
from collections import Counter
from typing import Any
from uuid import UUID, uuid4

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.client_device_service import ClientDeviceService

from .skills_registry import get_skills_registry
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


def _get_device_session(*, user_id: str | None, device_id: str | None):
    if not user_id or not device_id:
        return None

    try:
        device_uuid = UUID(str(device_id))
    except Exception:
        logger.warning("Invalid device_id passed to client skill lookup: %s", device_id)
        return None

    session = ClientDeviceService.lookup_active_session(device_uuid)
    if session is None or str(session.user_id) != str(user_id):
        return None

    return session


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


def _get_server_skill_summaries() -> list[dict[str, Any]]:
    try:
        active_skills = get_skills_registry().get_active_skills()
    except Exception as exc:
        logger.warning("Failed to load server-side skills: %s", exc)
        return []

    return [
        {
            "name": skill.name,
            "lookup_name": skill.name,
            "description": skill.description,
            "category": None,
            "tags": [],
            "content_length": len(skill.content) if skill.content else 0,
            "source": "server",
        }
        for skill in active_skills
    ]


def _get_client_skill_summaries(
    *,
    user_id: str | None,
    device_id: str | None,
) -> list[dict[str, Any]]:
    session = _get_device_session(user_id=user_id, device_id=device_id)
    if session is None or not session.skill_catalog:
        return []

    raw_skills = session.skill_catalog.get("skills", [])
    if not isinstance(raw_skills, list):
        return []

    summaries: list[dict[str, Any]] = []
    for raw_skill in raw_skills:
        if not isinstance(raw_skill, dict):
            continue

        name = str(raw_skill.get("name") or "").strip()
        if not name or not bool(raw_skill.get("enabled", True)):
            continue

        summaries.append(
            {
                "name": name,
                "lookup_name": name,
                "description": str(raw_skill.get("description") or "").strip(),
                "category": raw_skill.get("category"),
                "tags": raw_skill.get("tags") or [],
                "content_length": raw_skill.get("content_length"),
                "source": "client",
            }
        )

    return summaries


def _apply_lookup_names(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(summary.get("name") or "") for summary in summaries)
    normalized: list[dict[str, Any]] = []

    for summary in summaries:
        entry = dict(summary)
        name = str(entry.get("name") or "").strip()
        source = str(entry.get("source") or "server").strip().lower()
        lookup_name = name
        if name and counts[name] > 1:
            lookup_name = f"{source}:{name}"
        entry["lookup_name"] = lookup_name
        entry["source"] = source
        normalized.append(entry)

    return normalized


def get_available_skill_summaries(
    *,
    user_id: str | None,
    device_id: str | None,
) -> list[dict[str, Any]]:
    summaries = _get_server_skill_summaries()
    summaries.extend(_get_client_skill_summaries(user_id=user_id, device_id=device_id))
    normalized = _apply_lookup_names(summaries)
    return sorted(normalized, key=lambda item: item["lookup_name"].lower())


def _resolve_skill_reference(
    *,
    skill_name: str,
    user_id: str | None,
    device_id: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    available_skills = get_available_skill_summaries(user_id=user_id, device_id=device_id)
    available_names = [
        str(item.get("lookup_name") or item.get("name") or "") for item in available_skills
    ]

    exact_lookup_match = next(
        (
            item
            for item in available_skills
            if str(item.get("lookup_name") or item.get("name") or "") == skill_name
        ),
        None,
    )
    if exact_lookup_match is not None:
        return exact_lookup_match, None

    raw_matches = [
        item for item in available_skills if str(item.get("name") or "").strip() == skill_name
    ]
    if len(raw_matches) == 1:
        return raw_matches[0], None

    if len(raw_matches) > 1:
        options = ", ".join(
            str(item.get("lookup_name") or item.get("name") or "") for item in raw_matches
        )
        return None, f"Error: skill '{skill_name}' is ambiguous. Use one of: {options}"

    return (
        None,
        f"Error: skill '{skill_name}' not found. "
        f"Available skills: {', '.join(available_names) if available_names else '(none)'}",
    )


# ── Tool factory ────────────────────────────────────────────────────


def create_activate_skill_tool(
    *,
    user_id: str | None,
    device_id: str | None,
):
    """
    Create the ``activate_skill`` tool that the LLM can call to load
    the full instructions for a specific skill.

    Skill content is fetched from the currently connected client device
    over the runtime bridge so the canonical backend does not need direct
    access to client-local skill files. Server-local skills continue to
    load directly from the canonical backend registry.
    """

    session = _get_device_session(user_id=user_id, device_id=device_id)
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
        resolved_skill, resolution_error = _resolve_skill_reference(
            skill_name=skill_name,
            user_id=bound_user_id,
            device_id=bound_device_id,
        )
        if resolution_error:
            return resolution_error
        if resolved_skill is None:
            return "Error: skill resolution failed."

        if str(resolved_skill.get("source") or "server") == "server":
            try:
                skill = get_skills_registry().get_skill(str(resolved_skill["name"]))
            except KeyError:
                return (
                    f"Error: skill '{resolved_skill['name']}' is no longer available on the server."
                )

            if not skill.enabled:
                return (
                    f"Error: skill '{resolved_skill['name']}' exists but is currently disabled. "
                    "Only enabled skills can be activated."
                )

            return f"── Skill: {skill.name} ──\n\n{skill.content}\n\n── End Skill: {skill.name} ──"

        ctx = get_tool_context()
        context_device_id = str(ctx.device_id or bound_device_id or "")
        if bound_device_id and context_device_id and context_device_id != bound_device_id:
            return (
                "Error: client-side skill activation was requested for a different device session "
                "than the active run."
            )

        session = _get_device_session(user_id=bound_user_id, device_id=bound_device_id)
        if session is None:
            return "Error: client-side skills are not available because the device is disconnected."

        if bound_session_id and session.session_id != bound_session_id:
            return (
                "Error: the client device session changed after skills were bound. "
                "Retry from the active device session."
            )

        gateway = session.websocket
        if gateway is None or not hasattr(gateway, "dispatch_tool_call"):
            return "Error: client runtime gateway is not available for skill activation."

        response = await gateway.dispatch_tool_call(
            request_id=str(uuid4()),
            tool_name="activate_skill",
            qualified_tool_id="native::activate_skill",
            arguments={"skill_name": str(resolved_skill["name"])},
            timeout_seconds=settings.client_runtime_ws_timeout_seconds,
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
