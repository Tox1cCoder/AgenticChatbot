from __future__ import annotations

from typing import Any

BASE_AGENT_DISPLAY_NAMES = {
    "chat_agent": "Chat Agent",
    "rag_agent": "RAG Agent",
    "search_agent": "Search Agent",
    "image_generator_agent": "Image Generator Agent",
    "planning_agent": "Planning Agent",
    "canvas_agent": "Canvas Agent",
}

# Short capability blurbs for base agents, surfaced to a delegating agent so it
# can recognise which specialist to hand off to when a request — or a part of it
# — falls outside its own toolset. Kept terse and aligned with the router's
# agent descriptions (app/ai/prompts.py::ROUTER_SYSTEM_PROMPT).
BASE_AGENT_CAPABILITIES = {
    "chat_agent": (
        "General conversation, explanations, advice, Q&A, coding help, and "
        "tool-backed work in external integrations, accounts, and apps."
    ),
    "rag_agent": (
        "Questions about uploaded documents: document analysis and summaries of uploaded content."
    ),
    "search_agent": (
        "Current events, news, recent and time-sensitive information, web search, "
        "and fact-checking."
    ),
    "image_generator_agent": (
        "Creating raster/pixel images, drawings, and illustrations (non-code visual generation)."
    ),
    "planning_agent": ("Creating, editing, viewing, or executing multi-step task plans."),
    "canvas_agent": (
        "Authoring standalone browser artifacts: websites, web apps, interactive "
        "components, games, visualizations, SVG, and HTML/CSS/JS."
    ),
}


def base_agent_capability(agent_id: str | None) -> str | None:
    """Return a short capability blurb for a base agent id, if known."""
    if not isinstance(agent_id, str):
        return None
    return BASE_AGENT_CAPABILITIES.get(agent_id)


def is_custom_agent_id(agent_id: str | None) -> bool:
    return isinstance(agent_id, str) and agent_id.startswith("custom_agent:")


def _fallback_name(agent_id: str) -> str:
    return agent_id.replace("_", " ").title()


def agent_identity(
    agent_id: str | None,
    custom_agents: dict[str, Any] | None,
    *,
    fallback_name: str | None = None,
    custom_agent_id: str | None = None,
) -> dict[str, Any] | None:
    if not agent_id:
        return None

    custom_agents = custom_agents if isinstance(custom_agents, dict) else {}
    entry = custom_agents.get(agent_id)
    if isinstance(entry, dict):
        resolved_custom_id = entry.get("id") or custom_agent_id
        if not resolved_custom_id and ":" in agent_id:
            resolved_custom_id = agent_id.split(":", 1)[1]
        return {
            "id": agent_id,
            "kind": "custom",
            "name": str(entry.get("name") or fallback_name or agent_id),
            "custom_agent_id": str(resolved_custom_id) if resolved_custom_id else None,
        }

    if is_custom_agent_id(agent_id):
        resolved_custom_id = custom_agent_id or agent_id.split(":", 1)[1]
        return {
            "id": agent_id,
            "kind": "custom",
            "name": fallback_name or agent_id,
            "custom_agent_id": str(resolved_custom_id) if resolved_custom_id else None,
        }

    return {
        "id": agent_id,
        "kind": "base",
        "name": BASE_AGENT_DISPLAY_NAMES.get(agent_id, fallback_name or _fallback_name(agent_id)),
        "custom_agent_id": None,
    }


def attach_agent_metadata(
    metadata: dict[str, Any],
    *,
    response_agent_id: str | None,
    selected_agent_id: str | None,
    custom_agents: dict[str, Any] | None,
) -> None:
    compat_runtime_id = metadata.get("runtime_agent_id")
    compat_name = metadata.get("custom_agent_name")
    compat_custom_id = metadata.get("custom_agent_id")

    if response_agent_id:
        source = "response"
        agent_id = response_agent_id
    elif isinstance(compat_runtime_id, str) and compat_runtime_id:
        source = "response"
        agent_id = compat_runtime_id
    else:
        source = "selected_agent"
        agent_id = selected_agent_id

    identity = agent_identity(
        agent_id,
        custom_agents,
        fallback_name=compat_name if isinstance(compat_name, str) else None,
        custom_agent_id=compat_custom_id if isinstance(compat_custom_id, str) else None,
    )
    if identity:
        metadata["agent"] = {**identity, "source": source}


def normalize_handoff_metadata(context: dict[str, Any] | None) -> dict[str, Any] | None:
    context = context if isinstance(context, dict) else {}
    handoff = context.get("handoff")
    if not isinstance(handoff, dict) or not handoff.get("target_agent"):
        return None
    return {
        "from_agent_id": handoff.get("source_agent"),
        "to_agent_id": handoff.get("target_agent"),
        "tool_call_id": handoff.get("tool_call_id"),
    }


def normalize_subagent_metadata(
    raw_results: Any,
    custom_agents: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_results, list):
        return []

    normalized: list[dict[str, Any]] = []
    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        agent_id = entry.get("agent_id") or entry.get("agent")
        entry_name = entry.get("agent_name")
        entry_custom_id = entry.get("custom_agent_id")
        identity = agent_identity(
            str(agent_id) if agent_id else None,
            custom_agents,
            fallback_name=entry_name if isinstance(entry_name, str) else None,
            custom_agent_id=entry_custom_id if isinstance(entry_custom_id, str) else None,
        )
        if not identity:
            continue
        item = {
            "id": str(entry.get("id") or entry.get("worker_id") or ""),
            "agent": identity["id"],
            "agent_name": identity["name"],
            "agent_kind": identity["kind"],
            "custom_agent_id": identity.get("custom_agent_id"),
            "status": str(entry.get("status") or "unknown"),
            "summary": str(entry.get("summary") or ""),
        }
        # Durable per-worker record: keep the dispatcher's activity fields so
        # persisted messages render the same detail as the live panel.
        if isinstance(entry.get("thinking"), str):
            item["thinking"] = entry["thinking"]
        if isinstance(entry.get("error"), str):
            item["error"] = entry["error"]
        if isinstance(entry.get("elapsed_ms"), (int, float)):
            item["elapsed_ms"] = int(entry["elapsed_ms"])
        related = entry.get("related_todo_ids")
        if isinstance(related, list) and related:
            item["related_todo_ids"] = [str(v) for v in related]
        for model_key in ("requested_model", "resolved_model"):
            if isinstance(entry.get(model_key), dict):
                item[model_key] = entry[model_key]
        normalized.append({k: v for k, v in item.items() if v not in (None, "")})
    return normalized
