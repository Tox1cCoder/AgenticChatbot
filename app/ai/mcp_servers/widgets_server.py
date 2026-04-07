"""
Widgets MCP server — live UI widget tools for in-chat visual aids.

Runs as a stdio MCP server process. Uses the shared Redis-backed widget
runtime so that both this process and the main API server can read/write
the same widget state.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the project root is on sys.path so `app.*` imports resolve
# when this file is launched as a subprocess by MCPManager.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


from mcp.server.fastmcp import FastMCP

from app.services.widget_runtime import get_widget_store

mcp = FastMCP("widgets")


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

    Args:
        session_id: The conversation ID this widget belongs to.
        widget_type: Widget kind — e.g. "table", "chart", "dashboard", "form", "list", or "html".
            Use the structured kinds for durable data widgets that should be patchable and easy to
            read back on later turns. Use "html" only when you need a more bespoke in-chat UI that
            behaves more like a compact Canvas artifact.
        initial_state: JSON string with the widget's initial data/configuration.
            Preferred state conventions:
            - table: {"columns": ["Column"], "rows": [["Value"]]}
            - chart: {"chart_type": "bar", "labels": ["A", "B"], "datasets": [{"label": "Score", "data": [1, 2]}]}
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
                    "metric=profit": {"chart_type": "line", "labels": ["Jan", "Feb"], "datasets": [{"label": "Profit", "data": [3, 5]}]}
                }
              }
            - alternative interactive shape: {
                "controls": [...],
                "control_values": {...},
                "variants": [{"match": {"metric": "revenue"}, "state": {...}}, {"match": {"metric": "profit"}, "state": {...}}]
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
        JSON object describing the created widget (widget_id, version, etc.).
    """
    store = get_widget_store()
    state = json.loads(initial_state)
    record = await store.create(
        session_id=session_id,
        widget_type=widget_type,
        initial_state=state,
        title=title or None,
    )
    return json.dumps(record.to_dict(), default=str)


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
    new_state = json.loads(state)
    record = await store.update(
        widget_id=widget_id,
        state=new_state,
        expected_version=version if version > 0 else None,
    )
    return json.dumps(record.to_dict(), default=str)


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
