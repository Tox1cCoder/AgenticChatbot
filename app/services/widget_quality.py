"""Widget quality assessment — objective validation for live widget state.

Hard failures prevent widgets from being stored; soft messages are returned to
the model as guidance without user-visible warnings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WidgetQualityResult:
    allowed: bool
    messages: list[str] = field(default_factory=list)
    soft_messages: list[str] = field(default_factory=list)


def assess_widget_state(widget_type: str, state: Any) -> WidgetQualityResult:
    normalized_type = str(widget_type or "").strip().lower()
    if normalized_type in {"html", "iframe", "micro_app"}:
        return _assess_html(state)
    if not isinstance(state, dict):
        return WidgetQualityResult(False, ["structured widget state must be a JSON object"])
    if normalized_type == "chart":
        return _assess_chart(state)
    if normalized_type == "dashboard":
        return _assess_dashboard(state)
    if normalized_type == "table":
        return _assess_table(state)
    if normalized_type in {"form", "list"}:
        return WidgetQualityResult(True)
    return WidgetQualityResult(True)


def _assess_html(state: Any) -> WidgetQualityResult:
    if not isinstance(state, dict):
        return WidgetQualityResult(False, ["html widget state must be a JSON object"])
    html = str(state.get("html") or state.get("document") or state.get("content") or "").strip()
    if not html:
        return WidgetQualityResult(False, ["html widgets require non-empty html content"])
    height = state.get("height", 560)
    try:
        numeric_height = int(height)
    except (TypeError, ValueError):
        return WidgetQualityResult(False, ["html widget height must be numeric"])
    if numeric_height < 260 or numeric_height > 960:
        return WidgetQualityResult(False, ["html widget height must be between 260 and 960"])
    return WidgetQualityResult(True)


def _numeric_values(dataset: dict[str, Any]) -> list[float]:
    values = dataset.get("data") or dataset.get("values") or []
    if not isinstance(values, list):
        return []
    numeric: list[float] = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return []
    return numeric


def _assess_chart(state: dict[str, Any]) -> WidgetQualityResult:
    labels = state.get("labels")
    datasets = state.get("datasets")
    if not isinstance(labels, list) or len(labels) < 2:
        return WidgetQualityResult(False, ["chart requires at least two labels"])
    if not isinstance(datasets, list) or not datasets:
        return WidgetQualityResult(False, ["chart requires at least one dataset"])
    numeric_series = [
        _numeric_values(dataset)
        for dataset in datasets
        if isinstance(dataset, dict)
    ]
    if not numeric_series or any(len(series) != len(labels) for series in numeric_series):
        return WidgetQualityResult(
            False, ["chart datasets must contain numeric values for every label"]
        )

    chart_type = str(state.get("chart_type") or state.get("chartType") or "bar").lower()
    if "donut" in chart_type or "pie" in chart_type:
        if len(numeric_series) != 1:
            return WidgetQualityResult(
                False, ["donut charts require a single part-to-whole series"]
            )
        if any(value < 0 for value in numeric_series[0]):
            return WidgetQualityResult(False, ["donut charts require non-negative values"])
    if "line" in chart_type or "area" in chart_type:
        presentation = state.get("presentation") if isinstance(state.get("presentation"), dict) else {}
        axis_kind = str(presentation.get("x_kind") or state.get("x_kind") or "").lower()
        if axis_kind not in {"ordered", "time", "sequence"}:
            return WidgetQualityResult(
                False,
                ["line and area charts require x_kind ordered, time, or sequence"],
            )

    soft_messages: list[str] = []
    presentation = state.get("presentation") if isinstance(state.get("presentation"), dict) else {}
    if not presentation.get("caption"):
        soft_messages.append("presentation.caption improves article-style readability")
    if not presentation.get("x_label") or not presentation.get("y_label"):
        soft_messages.append("axis labels improve chart readability")
    return WidgetQualityResult(True, soft_messages=soft_messages)


def _assess_dashboard(state: dict[str, Any]) -> WidgetQualityResult:
    panels = state.get("panels")
    if isinstance(panels, list) and panels:
        return WidgetQualityResult(True)
    fallback_metrics = [
        value
        for key, value in state.items()
        if key != "panels" and isinstance(value, (str, int, float, bool))
    ]
    if fallback_metrics:
        return WidgetQualityResult(True)
    return WidgetQualityResult(False, ["dashboard requires panels or fallback metrics"])


def _assess_table(state: dict[str, Any]) -> WidgetQualityResult:
    rows = state.get("rows")
    columns = state.get("columns") or state.get("cols")
    if isinstance(rows, list) and rows and isinstance(columns, list) and columns:
        return WidgetQualityResult(True)
    empty_state = state.get("empty_state")
    if isinstance(empty_state, str) and empty_state.strip():
        return WidgetQualityResult(True)
    return WidgetQualityResult(False, ["table requires rows with columns or a clear empty_state"])


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
        "presentation": state.get("presentation") if isinstance(state.get("presentation"), dict) else {},
        "control_values": state.get("control_values") if isinstance(state.get("control_values"), dict) else {},
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
