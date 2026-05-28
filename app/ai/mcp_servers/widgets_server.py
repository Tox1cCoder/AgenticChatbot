"""
Widgets MCP server — live UI widget tools for in-chat visual aids.

Runs as a stdio MCP server process. Uses the shared Redis-backed widget
runtime so that both this process and the main API server can read/write
the same widget state.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from typing import Any

# Ensure the project root is on sys.path so `app.*` imports resolve
# when this file is launched as a subprocess by MCPManager.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


from mcp.server.fastmcp import FastMCP

from app.services.widget_quality import assess_widget_state
from app.services.widget_runtime import get_widget_store

mcp = FastMCP("widgets")


def _parse_widget_state(raw: str, *, field: str = "initial_state") -> Any:
    """Parse a widget state JSON string with a clear error on failure.

    Falls back to ``ast.literal_eval`` for Python-style literals
    (``True``/``False``/``None``/single-quoted keys) so a single common
    formatting slip from the model doesn't lose the whole widget creation.
    On true malformation, raise a ``ValueError`` whose message includes the
    offending snippet so the model can self-correct on the next turn.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            value = None
        if value is not None and isinstance(value, (dict, list)):
            return value
        snippet_start = max(0, exc.pos - 30)
        snippet_end = min(len(raw), exc.pos + 30)
        snippet = raw[snippet_start:snippet_end].replace("\n", " ").replace("\r", " ")
        raise ValueError(
            f"{field} must be a valid JSON string "
            f"({exc.msg} at character {exc.pos}). "
            f"Context: ...{snippet}... "
            "Use double-quoted keys and strings, lowercase true/false/null, "
            "no trailing commas, and escape any embedded quotes."
        ) from None


@mcp.tool()
async def widget_create(
    session_id: str,
    widget_type: str,
    initial_state: str,
    title: str = "",
) -> str:
    """Create a new live widget that appears inside the chat conversation.

    Use this to show structured, interactive content such as:
    - A live comparison table
    - A chart or dashboard summarising results
    - A structured form for collecting user choices
    - A sortable/filterable data view
    - A compact concept explainer for comparisons, taxonomies, timelines, or decision guides
    - A compact bespoke micro-app rendered in a sandboxed iframe when the built-in widget types are too rigid

    Treat widgets like inline visuals in an article: provide a `presentation` block
    (title, caption, axis labels, units, annotations) and place the widget marker
    near the paragraph it supports. Soft guidance is returned in the response as
    `quality_guidance` when the widget is accepted but could be improved.

    `initial_state` must be a valid JSON string — double-quoted keys and strings,
    lowercase `true`/`false`/`null`, no trailing commas, embedded quotes escaped
    as `\\"`. If the parser rejects the input, the error message includes the
    offending snippet so you can fix and retry on the next turn.

    Objective validation rejects widgets that are clearly low-value, including:
    - chart with fewer than two labels or non-numeric datasets
    - line/area chart without an ordered/time/sequence `x_kind`
    - donut/pie with negative values or multiple unrelated series
    - empty dashboard with no panels or fallback metrics
    - table with no rows/columns and no `empty_state`
    - html widget with empty content or height outside 260..960

    Args:
        session_id: The conversation ID this widget belongs to.
        widget_type: Widget kind — e.g. "table", "chart", "dashboard", "form", "list", or "html".
            Use the structured kinds for durable data widgets that should be patchable and easy to
            read back on later turns. Use "html" only when you need a more bespoke in-chat UI that
            behaves more like a compact Canvas artifact.
        initial_state: JSON string with the widget's initial data/configuration.
            Preferred state conventions:
            - table: {"columns": ["Column"], "rows": [["Value"]], "presentation": {"caption": "..."}}
            - chart (article-style): {
                "chart_type": "bar",
                "labels": ["A", "B"],
                "datasets": [{"label": "Score", "data": [1, 2]}],
                "presentation": {
                    "title": "Score by option",
                    "caption": "B leads A by roughly 2x.",
                    "x_label": "Option",
                    "y_label": "Score",
                    "unit": "points",
                    "annotations": [{"label": "Best", "series": "Score", "point_index": 1}]
                }
              }
            - line chart: include `"presentation": {"x_kind": "time"}` (or "ordered"/"sequence")
            - dashboard: {"panels": [{"type": "metric", "title": "Total", "value": 42}, {"type": "chart", "title": "Trend", "data": {...}}]}
            - form: {"fields": [{"key": "topic", "label": "Topic", "type": "text"}], "values": {"topic": "Widgets"}}
            - list: {"items": [{"id": "a", "label": "Option A", "description": "Why it matters"}], "selection": "a"}
            - html micro-app: {
                "html": "<!doctype html>...self-contained responsive HTML/CSS/JS...",
                "height": 540,
                "caption": "Optional short note shown above the iframe"
              }
            - interactive views: {
                "controls": [{"key": "metric", "label": "Metric", "type": "segmented", "options": [{"value": "revenue", "label": "Revenue"}, {"value": "profit", "label": "Profit"}], "value": "revenue"}],
                "control_values": {"metric": "revenue"},
                "views": {
                    "metric=revenue": {"chart_type": "bar", "labels": ["Jan", "Feb"], "datasets": [{"label": "Revenue", "data": [12, 18]}]},
                    "metric=profit": {"chart_type": "line", "labels": ["Jan", "Feb"], "datasets": [{"label": "Profit", "data": [3, 5]}], "presentation": {"x_kind": "time"}}
                }
              }
            - alternative interactive shape: {
                "controls": [...],
                "control_values": {...},
                "variants": [{"match": {"metric": "revenue"}, "state": {...}}, {"match": {"metric": "profit"}, "state": {...}}]
              }
            - assistant actions: {
                "actions": [{
                    "key": "explain_current_state",
                    "label": "Explain current state",
                    "type": "assistant_message",
                    "message_template": "Explain the widget state for damping={{control_values.damping}} in the context of the current answer."
                }]
              }
            Prefer putting reusable controls at the top level and per-view chart/table payloads in `views` or `variants`.
            For dashboards, prefer layered, polished explainer layouts over a single raw chart:
            combine hero metrics, supporting charts, ranked lists, and decision notes.
            For html widgets:
            - keep the experience compact and in-chat, not a full standalone website
            - prefer self-contained HTML with inline CSS/JS
            - make it responsive and visually polished
            - avoid external auth assumptions or cross-window dependencies
        title: Optional short human-readable title for the widget.

    Returns:
        JSON object describing the created widget (widget_id, version, etc.), and a
        `quality_guidance` list of soft recommendations when present.
    """
    store = get_widget_store()
    state = _parse_widget_state(initial_state, field="initial_state")
    quality = assess_widget_state(widget_type, state)
    if not quality.allowed:
        raise ValueError("; ".join(quality.messages))
    record = await store.create(
        session_id=session_id,
        widget_type=widget_type,
        initial_state=state,
        title=title or None,
    )
    payload = record.to_dict()
    if quality.soft_messages:
        payload["quality_guidance"] = list(quality.soft_messages)
    return json.dumps(payload, default=str)


@mcp.tool()
async def widget_update(
    widget_id: str,
    state: str,
    version: int = 0,
) -> str:
    """Replace the full state of an existing live widget.

    Use this when the widget's data needs to change — for example after
    fetching new search results or recalculating a dashboard. Keep the
    overall state shape consistent with the original widget when possible
    so the frontend can re-render smoothly. For interactive widgets, preserve
    top-level `controls` / `control_values` / `views` or `variants` keys unless
    you intentionally want to reset the available user interactions.

    Args:
        widget_id: The widget to update (returned by widget_create).
        state: JSON string with the new complete widget state.
        version: Expected current version for optimistic concurrency (0 = skip check).

    Returns:
        JSON object with the updated widget record.
    """
    store = get_widget_store()
    new_state = _parse_widget_state(state, field="state")
    existing = await store.get(widget_id)
    if existing is None:
        raise KeyError(f"Widget {widget_id} not found")
    quality = assess_widget_state(existing.widget_type, new_state)
    if not quality.allowed:
        raise ValueError("; ".join(quality.messages))
    record = await store.update(
        widget_id=widget_id,
        state=new_state,
        expected_version=version if version > 0 else None,
    )
    payload = record.to_dict()
    if quality.soft_messages:
        payload["quality_guidance"] = list(quality.soft_messages)
    return json.dumps(payload, default=str)


@mcp.tool()
async def widget_get_state(widget_id: str) -> str:
    """Read back the current state of a live widget.

    Call this on a follow-up turn to see what data the widget currently
    holds — including any changes the user may have made via the UI.

    Args:
        widget_id: The widget to inspect.

    Returns:
        JSON object with the widget's current record, or an error if not found.
    """
    store = get_widget_store()
    record = await store.get(widget_id)
    if record is None:
        return json.dumps({"error": f"Widget {widget_id} not found"})
    return json.dumps(record.to_dict(), default=str)


@mcp.tool()
async def widget_close(widget_id: str) -> str:
    """Close a live widget so it is no longer interactive.

    The widget will still be visible in message history but will stop
    accepting updates or user interactions.

    Args:
        widget_id: The widget to close.

    Returns:
        JSON object confirming the widget is closed.
    """
    store = get_widget_store()
    record = await store.close(widget_id)
    return json.dumps(record.to_dict(), default=str)


@mcp.tool()
async def session_list_widgets(session_id: str) -> str:
    """List all live widgets in a conversation session.

    Args:
        session_id: The conversation/session ID.

    Returns:
        JSON array of widget records for this session.
    """
    store = get_widget_store()
    records = await store.list_by_session(session_id)
    return json.dumps([r.to_dict() for r in records], default=str)


if __name__ == "__main__":
    mcp.run(transport="stdio")
