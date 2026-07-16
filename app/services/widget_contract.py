"""HTML-only live widget contract.

Live widgets have one supported type: ``html``. A widget renders as a
self-contained, sandboxed iframe built from ``state.html``. The contract is
intentionally minimal — it validates shape, not editorial quality:

- widget state must be a JSON object
- ``html`` content must be non-empty
- ``height`` must be present, numeric, and within ``MIN_WIDGET_HEIGHT`` ..
  ``MAX_WIDGET_HEIGHT``

No state-field aliases (``document``/``content``/``srcdoc``/``min_height``/
``minHeight``) and no widget-type aliases (``iframe``/``micro_app``) are part of
the contract. New widgets must use ``html`` and the ``html``/``height``/
``caption`` state shape.

This module also hosts the action-template helpers used by the widget action
endpoint, which survives the HTML-only migration.
"""

from __future__ import annotations

import re
from typing import Any

SUPPORTED_WIDGET_TYPE = "html"
MIN_WIDGET_HEIGHT = 260
MAX_WIDGET_HEIGHT = 960


def assert_supported_widget_type(widget_type: str) -> None:
    """Raise ``ValueError`` unless ``widget_type`` is exactly ``html``.

    Rejects the removed structured types (``table``/``chart``/``dashboard``/
    ``form``/``list``), former aliases (``iframe``/``micro_app``), and any
    unknown type with a clear, model-readable message.
    """
    normalized = str(widget_type or "").strip().lower()
    if normalized != SUPPORTED_WIDGET_TYPE:
        raise ValueError(
            f"unsupported widget type {widget_type!r}; "
            f"the only supported live widget type is {SUPPORTED_WIDGET_TYPE!r}. "
            "Create a self-contained HTML micro-app in initial_state.html."
        )


def validate_html_widget_state(state: Any) -> None:
    """Validate an HTML widget state against the minimal contract.

    Raises ``ValueError`` with a clear, model-readable message for each failure:
    non-object state, missing/empty html content, missing height, non-numeric
    height, or height outside the accepted iframe range.
    """
    if not isinstance(state, dict):
        raise ValueError("widget state must be a JSON object")

    html = str(state.get("html") or "").strip()
    if not html:
        raise ValueError("html widget requires non-empty html content in state.html")

    if "height" not in state or state.get("height") is None:
        raise ValueError("html widget requires a numeric height in state.height")

    height = state.get("height")
    if isinstance(height, bool):
        raise ValueError("html widget height must be numeric")
    try:
        numeric_height = float(height)
    except (TypeError, ValueError):
        raise ValueError("html widget height must be numeric") from None

    if not (MIN_WIDGET_HEIGHT <= numeric_height <= MAX_WIDGET_HEIGHT):
        raise ValueError(
            f"html widget height must be between {MIN_WIDGET_HEIGHT} and {MAX_WIDGET_HEIGHT}"
        )


# ---------------------------------------------------------------------------
# Action template resolution
# ---------------------------------------------------------------------------
_ACTION_TOKEN_RE = re.compile(r"{{\s*([A-Za-z0-9_.]+)\s*}}")


def _lookup_path(source: dict[str, Any], path: str) -> Any:
    current: Any = source
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return ""
        current = current[part]
    return current


def render_action_template(
    template: str,
    state: dict[str, Any],
    input_values: dict[str, Any],
) -> str:
    scope = {
        "state": state,
        "presentation": state.get("presentation")
        if isinstance(state.get("presentation"), dict)
        else {},
        "control_values": state.get("control_values")
        if isinstance(state.get("control_values"), dict)
        else {},
        "input_values": input_values,
    }

    def replace(match: re.Match[str]) -> str:
        value = _lookup_path(scope, match.group(1))
        return str(value)

    return _ACTION_TOKEN_RE.sub(replace, template).strip()


def resolve_widget_action_message(
    state: dict[str, Any],
    action_key: str,
    input_values: dict[str, Any] | None = None,
) -> str:
    actions = state.get("actions") if isinstance(state, dict) else None
    if not isinstance(actions, list):
        raise KeyError(f"Widget action {action_key} not found")
    action = next(
        (
            item
            for item in actions
            if isinstance(item, dict) and str(item.get("key") or "") == action_key
        ),
        None,
    )
    if not action:
        raise KeyError(f"Widget action {action_key} not found")
    if str(action.get("type") or "") != "assistant_message":
        raise ValueError(f"Widget action {action_key} is not an assistant action")
    template = str(action.get("message_template") or "").strip()
    if not template:
        raise ValueError(f"Widget action {action_key} has no message template")
    return render_action_template(template, state, input_values or {})
