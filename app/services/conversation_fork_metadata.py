"""Snapshot policy for messages copied into a forked conversation.

A fork copies the source transcript into new rows owned by the child
conversation. Each copied row must render the same history -- text, images,
canvas, citations, reasoning, attribution -- while carrying nothing that can
resume, approve, continue, or reconnect to execution owned by the source.

The policy is an allowlist built from an inventory of the metadata producers
(``build_bot_metadata`` and the rich-item finalizer in
``app/core/response_constants.py``, ``MessageService``'s continuation, stop,
interrupt, and planning fields, the agents' ``response.metadata``) and their
consumers (``demo.py``, ``app/ui``, the AI SDK projection, history and canvas
readers). Unknown keys are dropped rather than copied: metadata is an open
dict that producers extend without review, and a new key is far more likely to
be a runtime handle than display data. A display field missing from the
allowlist costs a cosmetic gap in copied history; a resume handle copied by
default would hand the child a capability over the source's execution.
"""

from __future__ import annotations

import copy
import json
from typing import Any
from uuid import UUID

from app.core.rich_response import (
    RichItemType,
    replace_inline_rich_references,
    validate_rich_references,
)

FORK_HISTORY_KEY = "fork_history"
FORK_CUTOFF_ELIGIBLE_KEY = "fork_cutoff_eligible"

#: Copied verbatim. Each is read by a renderer, the AI SDK projection, prompt
#: history, or canvas/agent continuity, and none can drive execution.
_DISPLAY_KEYS: frozenset[str] = frozenset(
    {
        # Persona and agent identity (routing continuity reads ``agent``).
        "persona_used",
        "agent",
        "runtime_agent_id",
        "custom_agent_id",
        "custom_agent_name",
        # Model attribution and the context-window indicator.
        "provider",
        "model",
        "reasoning_effort",
        "provider_fallback",
        "custom_model_override",
        "context_window",
        # Reasoning and tool/subagent activity panels.
        "thinking_summary",
        "reasoning_summary",
        "subagent_results",
        "subagent_worker_artifacts",
        # Sources and citations.
        "web_sources",
        "web_sources_version",
        "web_grounding_warnings",
        "documents_cited",
        "citations",
        # Media references. Owner validation happens in the fork repository.
        "images",
        "attachments",
        "rich_items_version",
        # Canvas: the latest revision is rebuilt from child-scoped rows.
        "canvas_artifact",
        "canvas_update",
        # Historical partial/stop attribution, inert without the ids below.
        "partial",
        "continued",
        "stopped",
        "stop_reason",
        # Keeps a substantive paused row hidden exactly as in the source. Its
        # ``interrupt``/``thread_id`` resume handles are not copied.
        "paused",
        "pause_reason",
        # Display text for error rows and follow-up suggestions.
        "error",
        "message",
        "suggested_questions",
    }
)

#: Tools whose artifacts anchor a live widget. ``app/api/widgets.py`` recovers
#: a widget's conversation and state from these outputs, so a copy would make
#: the child a second owner of the source's widget session.
_WIDGET_TOOL_NAMES = frozenset(
    {"widget_create", "widget_update", "widget_get_state", "widget_close"}
)

#: Rich-item types that open a live connection when rendered.
_EXECUTABLE_RICH_ITEM_TYPES = frozenset({RichItemType.live_widget.value})

_WIDGET_LABEL_FALLBACK = "Interactive widget"
_WIDGET_LABEL_MAX_CHARS = 120


def snapshot_fork_message(
    *,
    content: str,
    metadata: dict[str, Any] | None,
    message_id_map: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    """Return ``(content, metadata)`` for an inert copy of one source message.

    Pure: no I/O, and neither input is mutated; the returned metadata shares no
    mutable state with ``metadata``. ``message_id_map`` maps source message
    ids to child message ids; it is the only source of ids in the output.

    Every result carries ``fork_history=True``. The helper never sets or infers
    ``fork_cutoff_eligible`` (the repository computes it from the source's
    durable lifecycle), but a boolean already present on the input -- a row
    copied by an earlier fork -- is preserved so a nested fork can read it.
    """
    source = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
    # Iterating the source (not the set) keeps the source's key order, so the
    # serialized snapshot does not depend on the process's hash seed.
    snapshot = {key: value for key, value in source.items() if key in _DISPLAY_KEYS}

    reply_to = _mapped_message_id(source.get("reply_to_user_message_id"), message_id_map)
    if reply_to is not None:
        snapshot["reply_to_user_message_id"] = reply_to

    tool_artifacts = _non_widget_tool_artifacts(source.get("tool_artifacts"))
    if tool_artifacts:
        snapshot["tool_artifacts"] = tool_artifacts

    retired_widgets = _retired_widget_labels(source)
    if isinstance(source.get("rich_items"), list):
        snapshot["rich_items"] = [
            item for item in source["rich_items"] if not _is_executable_rich_item(item)
        ]

    new_content = _replace_widget_markers(
        content,
        retired_widgets,
        append_unplaced=not _is_hidden_pause_row(source),
    )

    if "rich_reference_warnings" in source:
        snapshot["rich_reference_warnings"] = validate_rich_references(
            new_content, snapshot.get("rich_items") or []
        )

    eligible = source.get(FORK_CUTOFF_ELIGIBLE_KEY)
    if isinstance(eligible, bool):
        snapshot[FORK_CUTOFF_ELIGIBLE_KEY] = eligible

    snapshot[FORK_HISTORY_KEY] = True
    return new_content, snapshot


def _mapped_message_id(value: Any, message_id_map: dict[str, str]) -> str | None:
    """Map one recognized message reference; ``None`` when it was not copied."""
    if not isinstance(value, str) or not value:
        return None
    if value in message_id_map:
        return message_id_map[value]
    try:
        canonical = str(UUID(value))
    except ValueError:
        return None
    return message_id_map.get(canonical)


def _non_widget_tool_artifacts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [
        artifact
        for artifact in raw
        if isinstance(artifact, dict)
        and (artifact.get("tool") or artifact.get("tool_name")) not in _WIDGET_TOOL_NAMES
    ]


def _is_executable_rich_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") in _EXECUTABLE_RICH_ITEM_TYPES


def _retired_widget_labels(source: dict[str, Any]) -> dict[str, str]:
    """Marker id -> display text for every live widget the copy drops.

    Collects both shapes: v1 ``rich_items`` descriptors (marker id is the item
    id) and legacy ``live_widgets`` entries (marker id ``widget:<widget_id>``,
    the id ``_widget_rich_item_from_live_widget`` assigns). Insertion order is
    source order, so appended labels are deterministic.
    """
    labels: dict[str, str] = {}
    for item in source.get("rich_items") or []:
        if _is_executable_rich_item(item) and isinstance(item.get("id"), str):
            labels[item["id"]] = _widget_label(item.get("title"))
    for widget in source.get("live_widgets") or []:
        if not isinstance(widget, dict) or not widget.get("widget_id"):
            continue
        labels.setdefault(f"widget:{widget['widget_id']}", _widget_label(widget.get("title")))
    return labels


def _widget_label(title: Any) -> str:
    text = " ".join(str(title).split()) if isinstance(title, str) else ""
    # Angle brackets are dropped so a title can never spell a new marker.
    text = text.replace("<", "").replace(">", "")[:_WIDGET_LABEL_MAX_CHARS].strip()
    name = json.dumps(text, ensure_ascii=False) if text else _WIDGET_LABEL_FALLBACK.lower()
    return f"[Interactive widget {name} is not available in a forked conversation]"


def _replace_widget_markers(content: str, labels: dict[str, str], *, append_unplaced: bool) -> str:
    """Put each retired widget's label where its marker was.

    A widget with no marker was rendered after the body in the source
    (``inline_or_append``), so its label is appended there instead. Hidden
    paused rows are never shown, and adding text to an empty one would turn a
    placeholder into a visible transcript turn, so they get no appended label.
    """
    if not labels:
        return content
    replaced, placed = replace_inline_rich_references(content, labels)
    unplaced = list(dict.fromkeys(labels[key] for key in labels if key not in placed))
    if not unplaced or not append_unplaced:
        return replaced
    body = replaced.rstrip()
    return "\n\n".join([body, *unplaced]) if body else "\n\n".join(unplaced)


def _is_hidden_pause_row(source: dict[str, Any]) -> bool:
    """Mirror of the source UI's rule: paused/interrupt rows are not shown."""
    return source.get("paused") is True or bool(source.get("interrupt"))


__all__ = [
    "FORK_CUTOFF_ELIGIBLE_KEY",
    "FORK_HISTORY_KEY",
    "snapshot_fork_message",
]
