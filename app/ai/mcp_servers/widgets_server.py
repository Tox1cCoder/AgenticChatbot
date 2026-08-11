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
from typing import Annotated, Any

# Ensure the project root is on sys.path so `app.*` imports resolve
# when this file is launched as a subprocess by MCPManager.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


from mcp.server.fastmcp import FastMCP  # noqa: E402
from pydantic import BeforeValidator, WithJsonSchema  # noqa: E402

from app.services.widget_contract import (  # noqa: E402
    coerce_widget_state_object,
    validate_html_widget_state,
)
from app.services.widget_runtime import get_widget_store  # noqa: E402

mcp = FastMCP("widgets")


def _coerce_initial_state(raw: Any) -> Any:
    return coerce_widget_state_object(raw, field="initial_state")


def _coerce_state(raw: Any) -> Any:
    return coerce_widget_state_object(raw, field="state")


# The advertised schema stays object-only, so the model is still told to send
# one native object. The BeforeValidator repairs the common slip — a stringified
# object, usually pretty-printed JSON with raw newlines inside ``html`` — which
# pydantic would otherwise reject before the tool body ever runs.
_OBJECT_SCHEMA = WithJsonSchema({"type": "object", "additionalProperties": True})
InitialWidgetState = Annotated[
    dict[str, Any], BeforeValidator(_coerce_initial_state), _OBJECT_SCHEMA
]
WidgetState = Annotated[dict[str, Any], BeforeValidator(_coerce_state), _OBJECT_SCHEMA]


@mcp.tool()
async def widget_create(
    session_id: str,
    initial_state: InitialWidgetState,
    title: str = "",
) -> str:
    """Create a live HTML widget that appears inside the chat conversation.

    A live widget is a self-contained, sandboxed iframe micro-app — the one
    supported experience is HTML. Reach for it when a concept is easier to
    *show* than to describe: motion, changing variables, systems, physics, math,
    processes, or any "show how it works" explanation. Build something the reader
    can poke at:

    - animation or manipulable visual state
    - sliders or controls for the key parameters
    - live numeric readouts
    - canvas, SVG, or DOM diagrams/graphs when useful
    - labels and captions in the user's language
    - responsive inline CSS, with no auth assumptions and no cross-window
      requirements

    Vanilla JavaScript covers most widgets. When the subject genuinely calls for
    more, load one focused library from a public CDN with a plain
    `<script src="...">` tag — for example Matter.js for rigid-body physics,
    Three.js for a 3D scene, D3 for axes and scales over a dataset, Math.js for
    symbolic or matrix math, Anime.js for tweened motion. Prefer one library over
    three, and prefer a hand-drawn canvas over a library that only saves you a
    few lines.

    The iframe is sandboxed without same-origin privileges, so the document has
    no localStorage, sessionStorage, cookies, or access to the surrounding app,
    and it cannot call this app's APIs. Anything it needs must be inline or
    CDN-hosted. Guard startup — a `window.onerror` handler or a check that the
    library's global is defined — so a CDN that fails to answer leaves a readable
    message rather than an empty box.

    Place the widget's `<!--rich:widget:<id>-->` marker near the paragraph it
    supports so it reads like an inline figure in an article.

    Pass `initial_state` as one native object. Do not wrap it in Markdown or
    additional prose.

    The state contract is minimal and strict:

        {
          "html": "<!doctype html>...self-contained responsive HTML/CSS/JS...",
          "height": 620,
          "caption": "Optional short caption"
        }

    Use the literal keys `html` and `height`; aliases such as `document`,
    `content`, `srcdoc`, `min_height`, or `minHeight` are not accepted. `height`
    must be a number between 260 and 960. Creation fails with a clear error for an
    non-object state, empty html, or a missing, non-numeric, or out-of-range
    height.

    Example — "explain simple harmonic motion" (label in Vietnamese when asked in
    Vietnamese): animate the oscillator position `x(t)`, draw a time graph of
    displacement, expose sliders for amplitude, angular frequency, and phase, add
    pause/reset controls, and show live values for time and displacement — all
    inside the single `html` document.

    Args:
        session_id: The conversation ID this widget belongs to.
        initial_state: Object with `{"html": "...", "height": 620,
            "caption": "..."}`. `caption` is optional. The whole experience —
            controls, animation, graphs — lives inside the `html` document.
        title: Optional short human-readable title for the widget.

    Returns:
        JSON object describing the created widget (widget_id, version, etc.).
    """
    store = get_widget_store()
    # Coerced again here (idempotent for a dict) because direct in-process
    # callers bypass the argument validator above.
    state = coerce_widget_state_object(initial_state, field="initial_state")
    validate_html_widget_state(state)
    record = await store.create(
        session_id=session_id,
        initial_state=state,
        title=title or None,
    )
    return json.dumps(record.to_dict(), default=str)


@mcp.tool()
async def widget_update(
    widget_id: str,
    state: WidgetState,
    version: int = 0,
) -> str:
    """Replace the full state of an existing live HTML widget.

    Use this when the micro-app's data needs to change — for example after
    fetching new numbers to re-render inside the iframe. Pass the complete new
    state (`html`, `height`, and optionally `caption`); the widget keeps its
    original identity. The same minimal contract as `widget_create` applies.

    Args:
        widget_id: The widget to update (returned by widget_create).
        state: Object with the new complete widget state
            (`{"html": "...", "height": 620, "caption": "..."}`).
        version: Expected current version for optimistic concurrency (0 = skip check).

    Returns:
        JSON object with the updated widget record.
    """
    store = get_widget_store()
    # Coerced again here (idempotent for a dict) because direct in-process
    # callers bypass the argument validator above.
    new_state = coerce_widget_state_object(state, field="state")
    existing = await store.get(widget_id)
    if existing is None:
        raise KeyError(f"Widget {widget_id} not found")
    validate_html_widget_state(new_state)
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
