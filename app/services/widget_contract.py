"""Live interactive HTML experience contract.

Each widget renders as a self-contained, sandboxed iframe built from
``state.html``. The contract is
intentionally minimal — it validates shape, not editorial quality:

- widget state must be a JSON object
- ``html`` content must be non-empty
- ``height`` must be present, numeric, and within ``MIN_WIDGET_HEIGHT`` ..
  ``MAX_WIDGET_HEIGHT``

No state-field aliases (``document``/``content``/``srcdoc``/``min_height``/
``minHeight``) are part of the contract.

This module also hosts the action-template helpers used by the widget action
endpoint, which survives the HTML-only migration.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

MIN_WIDGET_HEIGHT = 260
MAX_WIDGET_HEIGHT = 960

_FENCE_RE = re.compile(r"^\s*```(?:json|javascript|js)?\s*\n(?P<body>.*?)\n?\s*```\s*$", re.S)


def _strip_markdown_fence(raw: str) -> str:
    match = _FENCE_RE.match(raw)
    return match.group("body") if match else raw


def coerce_widget_state_object(raw: Any, *, field: str = "initial_state") -> dict[str, Any]:
    """Return widget state as a native object, tolerating a stringified one.

    The tool schema advertises an object, but models still serialize this
    argument — most often as pretty-printed JSON whose ``html`` value carries
    raw, unescaped newlines. Strict ``json.loads`` rejects that with "Invalid
    control character", so the recovery ladder is:

    1. already a dict -> use it;
    2. strip a Markdown code fence the model wrapped it in;
    3. strict JSON;
    4. JSON with ``strict=False`` (allows literal control characters inside
       string values, which is exactly the observed failure);
    5. the first complete JSON object when the model kept writing past the
       closing brace — it sometimes appends the *remaining tool arguments*
       (``{...},session_id:"conv-1"``), which strict JSON reports as "Extra
       data". The trailing text is discarded: ``session_id`` is rebound from
       the active conversation anyway and ``title`` is optional;
    6. ``ast.literal_eval`` for Python-style literals (single quotes,
       ``True``/``None``).

    On true malformation, raise a ``ValueError`` naming the field and the
    offending position so the model can self-correct on the next turn.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field} must be an object.")

    candidate = _strip_markdown_fence(raw).strip()
    decode_error: json.JSONDecodeError | None = None

    for strict in (True, False):
        try:
            parsed = json.loads(candidate, strict=strict)
        except json.JSONDecodeError as exc:
            decode_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"{field} must be an object, not a {type(parsed).__name__}.")

    for strict in (True, False):
        try:
            prefix, end = json.JSONDecoder(strict=strict).raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(prefix, dict):
            raise ValueError(f"{field} must be an object, not a {type(prefix).__name__}.")
        discarded = candidate[end:].strip()
        logger.warning(
            "Recovered %s from a serialized object with %d trailing characters; "
            "discarded remainder starts with %r",
            field,
            len(discarded),
            discarded[:40],
        )
        return prefix

    try:
        literal = ast.literal_eval(candidate)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        literal = None
    if isinstance(literal, dict):
        return literal

    position = decode_error.pos if decode_error is not None else 0
    message = decode_error.msg if decode_error is not None else "could not be parsed"
    snippet_start = max(0, position - 30)
    snippet = candidate[snippet_start : position + 30].replace("\n", " ").replace("\r", " ")
    raise ValueError(
        f"{field} must be one JSON object ({message} at character {position}). "
        f"Context: ...{snippet}... "
        "Prefer sending it as a native object. If you serialize it, use "
        "double-quoted keys and strings, lowercase true/false/null, no trailing "
        "commas, and escape embedded quotes and newlines."
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
