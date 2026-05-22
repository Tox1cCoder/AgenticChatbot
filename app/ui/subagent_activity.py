from __future__ import annotations

import json
from typing import Any

_TERMINAL_BLOCKED_STATUSES = {"failed", "timeout", "requires_approval"}
_AGGREGATE_STATUSES = {"completed", "partial", "failed", "running"}
_DISPATCH_SUBAGENTS_TOOL = "dispatch_subagents"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_structured_dispatch_payload(artifact: dict[str, Any]) -> dict[str, Any] | None:
    render = artifact.get("render")
    if isinstance(render, dict):
        structured = render.get("structured_content") or render.get("structuredContent")
        if isinstance(structured, dict) and isinstance(structured.get("results"), list):
            return structured

    for key in ("result", "output", "tool_output"):
        payload = _parse_json_object(artifact.get(key))
        if payload and isinstance(payload.get("results"), list):
            return payload

    return None


def _normalize_model_info(entry: Any) -> dict[str, Any] | None:
    """Pick the subset of model fields the UI cares about.

    Strips API keys, runtime config objects, and any keys other than
    ``provider``/``model``/``config_source``/``reasoning_effort``/``temperature``
    plus the safe nested ``context_window`` usage contract.
    """
    if not isinstance(entry, dict):
        return None
    snapshot: dict[str, Any] = {}
    for key in ("provider", "model", "config_source", "reasoning_effort", "temperature"):
        value = entry.get(key)
        if value not in (None, ""):
            snapshot[key] = value
    context_window = entry.get("context_window")
    if isinstance(context_window, dict):
        snapshot["context_window"] = dict(context_window)
    return snapshot or None


def _normalize_result(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None

    worker_id = str(entry.get("id") or entry.get("worker_id") or "").strip()
    agent = str(entry.get("agent") or "unknown_agent").strip() or "unknown_agent"
    status = str(entry.get("status") or "unknown").strip().lower() or "unknown"
    summary = str(entry.get("summary") or "").strip()

    normalized: dict[str, Any] = {
        "id": worker_id or f"{agent}:{status}",
        "agent": agent,
        "status": status,
        "summary": summary,
        "related_todo_ids": [
            str(todo_id)
            for todo_id in _as_list(entry.get("related_todo_ids"))
            if str(todo_id).strip()
        ],
    }

    elapsed_ms = entry.get("elapsed_ms")
    if isinstance(elapsed_ms, (int, float)) and elapsed_ms >= 0:
        normalized["elapsed_ms"] = int(elapsed_ms)

    error = entry.get("error")
    if isinstance(error, str) and error.strip():
        normalized["error"] = error.strip()

    artifacts = entry.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        normalized["artifacts"] = artifacts

    requested_model = _normalize_model_info(entry.get("requested_model"))
    if requested_model:
        normalized["requested_model"] = requested_model

    resolved_model = _normalize_model_info(entry.get("resolved_model"))
    if resolved_model:
        normalized["resolved_model"] = resolved_model

    return normalized


def _normalize_pending_task(entry: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None

    worker_id = str(entry.get("id") or entry.get("worker_id") or f"worker-{index}").strip()
    agent = str(entry.get("agent") or "unknown_agent").strip() or "unknown_agent"

    summary_candidates = (
        entry.get("summary"),
        entry.get("task"),
        entry.get("description"),
        entry.get("instructions"),
        entry.get("expected_output"),
    )
    summary = next(
        (
            str(candidate).strip()
            for candidate in summary_candidates
            if isinstance(candidate, str) and candidate.strip()
        ),
        "Waiting for worker result.",
    )

    pending: dict[str, Any] = {
        "id": worker_id or f"worker-{index}",
        "agent": agent,
        "status": "running",
        "summary": summary,
        "related_todo_ids": [
            str(todo_id)
            for todo_id in _as_list(entry.get("related_todo_ids"))
            if str(todo_id).strip()
        ],
    }

    # Surface the requested model now so the user sees which model the
    # supervisor assigned to each running worker before results arrive. The
    # resolved model only appears once the worker returns.
    requested_model = _normalize_model_info(entry.get("model_override"))
    if requested_model:
        pending["requested_model"] = requested_model

    return pending


def _dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for result in results:
        key = (str(result.get("id") or ""), str(result.get("agent") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _append_unique_text(values: list[str], candidate: Any) -> None:
    if not isinstance(candidate, str):
        return
    text = candidate.strip()
    if text and text not in values:
        values.append(text)


def _build_activity_view(
    *,
    results: list[dict[str, Any]],
    rationales: list[str],
    dispatch_statuses: list[str],
) -> dict[str, Any] | None:
    results = _dedupe_results(results)
    if not results:
        return None

    completed = sum(1 for item in results if item.get("status") == "completed")
    running = sum(1 for item in results if item.get("status") == "running")
    timeout = sum(1 for item in results if item.get("status") == "timeout")
    requires_approval = sum(1 for item in results if item.get("status") == "requires_approval")
    failed = sum(1 for item in results if item.get("status") in _TERMINAL_BLOCKED_STATUSES)
    total = len(results)

    if dispatch_statuses:
        status = dispatch_statuses[-1]
    elif running:
        status = "running"
    elif completed == total:
        status = "completed"
    elif completed == 0:
        status = "failed"
    else:
        status = "partial"

    return {
        "status": status,
        "total": total,
        "completed": completed,
        "failed": failed,
        "timeout": timeout,
        "requires_approval": requires_approval,
        "running": running,
        "rationales": rationales,
        "results": results,
    }


def build_subagent_activity_view(message_metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a compact UI view model for Planning subagent activity.

    The preferred source is explicit Planning response metadata
    (``subagent_results`` / ``subagent_dispatches``). The artifact fallback
    keeps older responses renderable when only the ``dispatch_subagents`` tool
    output was persisted.
    """
    if not isinstance(message_metadata, dict):
        return None

    results: list[dict[str, Any]] = []
    rationales: list[str] = []
    dispatch_statuses: list[str] = []

    for dispatch in _as_list(message_metadata.get("subagent_dispatches")):
        if not isinstance(dispatch, dict):
            continue
        _append_unique_text(rationales, dispatch.get("rationale"))
        status = str(dispatch.get("status") or "").strip().lower()
        if status in _AGGREGATE_STATUSES:
            dispatch_statuses.append(status)

    for entry in _as_list(message_metadata.get("subagent_results")):
        normalized = _normalize_result(entry)
        if normalized:
            results.append(normalized)

    for artifact in _as_list(message_metadata.get("tool_artifacts")):
        if not isinstance(artifact, dict):
            continue
        tool_name = artifact.get("tool") or artifact.get("tool_name")
        render = artifact.get("render")
        render_type = render.get("type") if isinstance(render, dict) else None
        if tool_name != "dispatch_subagents" and render_type != "subagent_dispatch":
            continue

        payload = _extract_structured_dispatch_payload(artifact)
        if not payload:
            continue

        _append_unique_text(rationales, payload.get("rationale"))
        status = str(payload.get("status") or "").strip().lower()
        if status in _AGGREGATE_STATUSES:
            dispatch_statuses.append(status)

        for entry in _as_list(payload.get("results")):
            normalized = _normalize_result(entry)
            if normalized:
                results.append(normalized)

    worker_artifacts_map = message_metadata.get("subagent_worker_artifacts")
    if isinstance(worker_artifacts_map, dict) and worker_artifacts_map and results:
        for result in results:
            worker_id = str(result.get("id") or "")
            if not worker_id:
                continue
            artifacts = worker_artifacts_map.get(worker_id)
            if isinstance(artifacts, list) and artifacts:
                # Don't clobber inline artifacts (e.g. legacy payloads); merge new
                # ones so the UI shows every observed worker tool invocation.
                existing = _as_list(result.get("artifacts"))
                seen_ids = {
                    art.get("tool_call_id")
                    for art in existing
                    if isinstance(art, dict) and art.get("tool_call_id")
                }
                for artifact in artifacts:
                    if not isinstance(artifact, dict):
                        continue
                    tc_id = artifact.get("tool_call_id")
                    if tc_id and tc_id in seen_ids:
                        continue
                    existing.append(artifact)
                    if tc_id:
                        seen_ids.add(tc_id)
                result["artifacts"] = existing

    return _build_activity_view(
        results=results,
        rationales=rationales,
        dispatch_statuses=dispatch_statuses,
    )


def build_live_subagent_activity_view(
    tool_event: dict[str, Any] | None,
    *,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build or update a live subagent activity view from a streaming tool event."""
    if not isinstance(tool_event, dict):
        return previous

    if tool_event.get("type") == "node_complete" and tool_event.get("node") == "planning_agent":
        for tool_call in _as_list(tool_event.get("tool_calls")):
            if not isinstance(tool_call, dict):
                continue
            if tool_call.get("name") != _DISPATCH_SUBAGENTS_TOOL:
                continue
            return build_live_subagent_activity_view(
                {
                    "type": "tool",
                    "name": _DISPATCH_SUBAGENTS_TOOL,
                    "phase": "start",
                    "tool_call_id": tool_call.get("id"),
                    "args": tool_call.get("args"),
                },
                previous=previous,
            )

    tool_name = tool_event.get("name") or tool_event.get("tool") or tool_event.get("tool_name")
    if tool_name != _DISPATCH_SUBAGENTS_TOOL:
        return previous

    phase = str(tool_event.get("phase") or tool_event.get("status") or "").strip().lower()
    if phase == "start":
        args = _parse_json_object(tool_event.get("args")) or {}
        tasks = _as_list(args.get("tasks"))
        results = [
            normalized
            for index, task in enumerate(tasks, start=1)
            if (normalized := _normalize_pending_task(task, index)) is not None
        ]
        if not results:
            return previous

        rationales: list[str] = []
        _append_unique_text(rationales, args.get("rationale"))

        return _build_activity_view(
            results=results,
            rationales=rationales,
            dispatch_statuses=["running"],
        )

    if phase == "end":
        view = build_subagent_activity_view(
            {
                "tool_artifacts": [
                    {
                        "tool": _DISPATCH_SUBAGENTS_TOOL,
                        "render": tool_event.get("render"),
                        "result": tool_event.get("result"),
                        "output": tool_event.get("result"),
                    }
                ]
            }
        )
        return view or previous

    return previous
