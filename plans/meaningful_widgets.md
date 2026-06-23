# Meaningful Live Widgets Implementation Plan

> **SUPERSEDED (2026-06-22):** This historical plan describes the structured widget system
> (`table`/`chart`/`dashboard`/`form`/`list`, `widget_quality.py`, `quality_guidance`) that has
> been removed. Live widgets are now HTML-only — see
> [`docs/superpowers/specs/2026-06-22-html-only-live-widgets-design.md`](../docs/superpowers/specs/2026-06-22-html-only-live-widgets-design.md)
> and [`live-widgets-frontend-integration.md`](live-widgets-frontend-integration.md). Kept for history only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make live widgets read like article-quality inline visuals that explain the conversation, avoid low-value raw chart output, and behave consistently in both the AI SDK frontend path and the Streamlit `demo.py` path.

**Architecture:** Treat widget state as the shared contract. The backend should guide and validate widget creation before low-value widgets are stored, while renderers use the same state conventions for captions, annotations, controls, hover values, and assistant-triggering actions.

**Tech Stack:** FastAPI, Pydantic, Redis-backed widget runtime, MCP `widgets` server, AI SDK SSE rich-items contract, Streamlit `components.v1.html`, pytest.

---

## Current Codebase Findings

- Widget creation already exists in [app/ai/mcp_servers/widgets_server.py](app/ai/mcp_servers/widgets_server.py). Agents create live widgets through `widget_create`, `widget_update`, `widget_get_state`, and related MCP tools.
- Runtime state already exists in [app/services/widget_runtime.py](app/services/widget_runtime.py). It stores widget state, accepts shallow `user_state_patch` updates, and hydrates widgets over `/widgets/{id}/connect`.
- Inline rich response placement already exists through `rich_items` and markers such as `<!--rich:widget:<id>-->`.
- AI SDK clients already receive transient widget records via `data-rich-items` and final records via `messageMetadata.rich_items`.
- Streamlit already renders widgets in [demo.py](demo.py), including table search/sort, chart type toggles, series toggles, controls, forms, lists, dashboards, HTML iframes, and WebSocket patching.
- The main gaps are generation quality, a stronger state contract, chart hover values, article-like presentation, action semantics, and documentation/tests that keep AI SDK renderers aligned with Streamlit.

## Product Direction

- Do not add visible "why this chart" or "reason for this widget" boilerplate.
- Prevent low-value widgets at generation time through prompt/tool guidance plus objective validation.
- Make widgets feel like inline visuals in an article or paper: clear titles, axis labels, units, captions, annotations, meaningful controls, and placement near the relevant text.
- Allow `html` micro-widgets when structured widgets cannot express the concept well, especially for simulations or conceptual diagrams.
- Support two interaction modes:
  - local interaction: update only widget state, such as sliders, filters, chart type, hidden series, selected option, or form values
  - assistant action: selected widget controls can generate a normal follow-up chat turn using the current widget state

## Approaches Considered

### Approach A: Prompt-Only Guidance

Update prompts and MCP tool descriptions so models choose better widgets.

Tradeoffs:
- Lowest implementation cost.
- Does not protect against malformed or low-value widgets when the model ignores guidance.
- Does not guarantee AI SDK and Streamlit consistency.

### Approach B: Shared Contract, Quality Gate, Renderer Upgrade

Add a shared widget quality module, strengthen widget state conventions, update prompts/tool docs, and upgrade Streamlit plus AI SDK integration guidance around the same contract.

Tradeoffs:
- Best balance of quality and maintainability.
- Allows objective hard rejection for malformed widgets while leaving subjective editorial quality to generation guidance.
- Keeps `html` micro-widgets available without making every widget an opaque iframe.

Recommendation: use Approach B.

### Approach C: HTML-First Micro-Apps

Push most meaningful visuals into `widget_type="html"` and treat structured widgets as legacy.

Tradeoffs:
- Richest possible visuals.
- Harder to keep AI SDK and Streamlit behavior consistent.
- Loses patchable structured state for tables, charts, forms, and follow-up agent reads.
- Increases security and accessibility risk.

## Shared Widget State Contract

All frontends should render from the same state shape when present. Existing simple shapes remain valid, but new widgets should prefer this enriched structure.

```json
{
  "presentation": {
    "title": "Phase-space intuition",
    "caption": "The same oscillator becomes easier to read when position and velocity are viewed together.",
    "layout": "article",
    "x_label": "Position",
    "y_label": "Velocity",
    "unit": "normalized",
    "annotations": [
      {
        "label": "Stable orbit",
        "series": "Trajectory",
        "point_index": 8
      }
    ]
  },
  "chart_type": "line",
  "labels": ["t0", "t1", "t2"],
  "datasets": [
    { "label": "Trajectory", "data": [0.1, 0.4, 0.9] }
  ],
  "controls": [
    {
      "key": "damping",
      "label": "Damping",
      "type": "slider",
      "min": 0,
      "max": 1,
      "step": 0.05,
      "value": 0.2
    }
  ],
  "control_values": {
    "damping": 0.2
  },
  "views": {
    "damping=0.2": {
      "chart_type": "line",
      "labels": ["t0", "t1", "t2"],
      "datasets": [
        { "label": "Trajectory", "data": [0.1, 0.4, 0.9] }
      ]
    }
  },
  "actions": [
    {
      "key": "explain_current_state",
      "label": "Explain current state",
      "type": "assistant_message",
      "message_template": "Explain the widget state for damping={{control_values.damping}} in the context of the current answer."
    }
  ]
}
```

Rules:
- `presentation` is article-style context, not a visible justification field.
- `controls` and `control_values` remain the canonical local interaction model.
- `views` and `variants` remain supported for scenario switching.
- `actions` define interactions that become normal chat messages.
- `html` widgets use the same `presentation`, `controls`, `control_values`, and `actions` wrapper when useful, with the rendered app in `html`.

## Objective Low-Value Criteria

The backend should reject or flag only objective failures. Subjective editorial choices stay in prompt guidance.

Hard failures:
- Empty or non-object widget state for structured widget types.
- `chart` without usable labels and numeric datasets.
- `chart` with fewer than two comparable points unless it is embedded in a dashboard with other context.
- `line` or `area` chart with labels that are neither ordered nor explicitly marked as ordered/time-like.
- `pie` or `donut` chart with negative values, multiple unrelated series, or no part-to-whole structure.
- `dashboard` with no panels and no fallback metrics.
- `table` with no columns/rows and no clear empty state.
- `html` widget without non-empty `html` content or with an out-of-range height.

Soft issues:
- Missing `presentation.caption`, axis labels, or units.
- Ambiguous chart type when `bar` would communicate comparison better.
- Interaction missing where controls would naturally help compare scenarios.

Hard failures should prevent `widget_create` from storing a widget. Soft issues should be returned to the model in the tool result as guidance, without showing warnings to the user.

## File Structure

- Create `app/services/widget_quality.py`
  - Owns objective validation, state normalization, chart-kind checks, and action resolution helpers.
- Modify `app/ai/mcp_servers/widgets_server.py`
  - Uses `widget_quality` before create/update.
  - Expands tool descriptions with article-style state examples.
- Modify `app/ai/prompts.py`
  - Adds concise guidance that widgets must clarify the answer and should use article-like captions, labels, annotations, and controls.
- Modify `app/api/widgets.py`
  - Adds widget action resolution endpoint.
  - Extends WebSocket handling for action acknowledgements only when needed.
- Modify `client_backend/api/proxy.py`
  - Proxies the widget action endpoint just like widget connection minting.
- Modify `demo.py`
  - Renders `presentation` consistently.
  - Adds chart hover value tooltips.
  - Adds action buttons for `actions`.
  - Lets action buttons resolve a chat message and then stream the response through `/messages/stream`.
- Update `plans/live-widgets-frontend-integration.md`
  - Documents the shared state contract for AI SDK clients.
- Update `README.md`
  - Adds the meaningful-widget contract summary and test commands.
- Create tests:
  - `tests/test_widget_quality.py`
  - `tests/test_widget_actions_api.py`
  - `tests/test_demo_meaningful_widgets.py`
  - `tests/client_backend/test_widget_action_proxy.py`

## Task 1: Widget Quality Module ✅ DONE 2026-05-28

**Files:**
- Create: `app/services/widget_quality.py`
- Test: `tests/test_widget_quality.py`

**Verification:** `python -m pytest tests/test_widget_quality.py -q` → 19 passed.

**Design notes:**
- Module also hosts `render_action_template` / `resolve_widget_action_message` (Task 4) so all widget-quality concerns live in one place. The Task 4 plan asks us to *add* those helpers to the same module, so they live here from the start; Task 4 only exposes them through the API surface.
- Donut/pie are treated as the same family (chart type substring match).
- Line/area charts require explicit `x_kind` of `ordered`, `time`, or `sequence` either in `presentation` or at top level.

- [x] Write tests for chart validation.

```python
from app.services.widget_quality import assess_widget_state


def test_chart_requires_labels_and_numeric_datasets():
    result = assess_widget_state("chart", {"chart_type": "bar", "labels": [], "datasets": []})

    assert result.allowed is False
    assert "chart requires at least two labels" in result.messages


def test_donut_rejects_negative_values():
    result = assess_widget_state(
        "chart",
        {
            "chart_type": "donut",
            "labels": ["A", "B"],
            "datasets": [{"label": "Share", "data": [5, -1]}],
        },
    )

    assert result.allowed is False
    assert "donut charts require non-negative values" in result.messages
```

- [x] Add the quality module.

```python
from __future__ import annotations

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
        return WidgetQualityResult(False, ["chart datasets must contain numeric values for every label"])

    chart_type = str(state.get("chart_type") or state.get("chartType") or "bar").lower()
    if "donut" in chart_type or "pie" in chart_type:
        if len(numeric_series) != 1:
            return WidgetQualityResult(False, ["donut charts require a single part-to-whole series"])
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
```

- [x] Run the focused tests.

Run: `python -m pytest tests/test_widget_quality.py -q`

Expected: all new tests pass after implementation. **Actual: 19 passed.**

## Task 2: Enforce Objective Quality in Widget Tools ✅ DONE 2026-05-28

**Files:**
- Modify: `app/ai/mcp_servers/widgets_server.py`
- Modify: `tests/test_widget_runtime.py`

**Verification:** `python -m pytest tests/test_widget_runtime.py -q` → 64 passed.

**Design notes:**
- Soft messages are returned inside the JSON payload as `quality_guidance: [...]` rather than appended as free text. The original plan snippet appended free text after `json.dumps`, which would have broken `extract_live_widgets_from_artifacts` in `app/core/response_constants.py` since that function calls `json.loads(raw_output)` on the tool output. Embedding inside the JSON keeps the model able to read guidance while preserving downstream parsing.
- `widget_update` looks up the existing record first to recover `widget_type` (the tool only receives `widget_id` and the new state), then validates the new state under that type.
- `widget_update` raises `KeyError` if the widget was never created — matches the underlying store behavior and avoids leaking guidance for non-existent widgets.

- [x] Add tests that invalid widgets are not stored.

```python
import json

import pytest


@pytest.mark.asyncio
async def test_widget_create_rejects_empty_chart(monkeypatch):
    from app.ai.mcp_servers import widgets_server
    from app.services.widget_runtime import InMemoryWidgetStore
    import app.services.widget_runtime as widget_runtime

    store = InMemoryWidgetStore()
    monkeypatch.setattr(widget_runtime, "_widget_store", store)

    with pytest.raises(ValueError, match="chart requires at least two labels"):
        await widgets_server.widget_create(
            session_id="conv-1",
            widget_type="chart",
            initial_state=json.dumps({"chart_type": "bar", "labels": [], "datasets": []}),
            title="Empty",
        )
```

- [x] Call `assess_widget_state()` inside `widget_create` and `widget_update`.

```python
from app.services.widget_quality import assess_widget_state


quality = assess_widget_state(widget_type, state)
if not quality.allowed:
    raise ValueError("; ".join(quality.messages))
```

- [x] Preserve existing successful widget behavior.

Run: `python -m pytest tests/test_widget_runtime.py -q`

Expected: existing widget runtime tests pass, plus the new rejection test. **Actual: 64 passed.**

## Task 3: Improve Generation Guidance ✅ DONE 2026-05-28

**Files:**
- Modify: `app/ai/prompts.py`
- Modify: `app/ai/mcp_servers/widgets_server.py`
- Test: `tests/test_widget_runtime.py`

**Verification:** `python -m pytest tests/test_widget_runtime.py::TestPromptUpdates -q` → 3 passed.

**Design notes:**
- The prompt explicitly mentions `<!--rich:widget:<id>-->` marker placement near the supporting paragraph and the canonical `presentation` block (title/caption/x_label/y_label/unit/annotations).
- The line-chart `x_kind` requirement is surfaced in both the prompt and the docstring, since enforcement is now objective (Task 2).
- The `widget_create` docstring now lists the objective rejection rules so the model can predict what will fail before calling the tool.

- [x] Update prompt guidance around widget creation.

Add guidance with these rules:
- Create a widget only when it improves comprehension of the current answer.
- Use `dashboard` for mixed metrics, timelines, comparisons, and layered explanations.
- Use `html` for compact simulations or concept visuals that structured widgets cannot express.
- Choose chart type from data semantics: bar for comparison, line/area for ordered sequences, donut only for part-to-whole.
- Include article-like `presentation` context, labels, units, and annotations when useful.
- Put the widget marker near the paragraph it supports.
- Keep the text answer useful without the widget.

- [x] Update `widget_create` docstring examples to include `presentation`, annotations, controls, and actions.

- [x] Extend prompt tests.

```python
def test_prompt_mentions_article_style_widget_quality():
    from app.ai.prompts import CHAT_SYSTEM_PROMPT

    prompt = CHAT_SYSTEM_PROMPT.lower()
    assert "article" in prompt
    assert "annotations" in prompt
    assert "choose chart type" in prompt
```

Run: `python -m pytest tests/test_widget_runtime.py::TestPromptUpdates -q`

Expected: prompt guidance tests pass. **Actual: 3 passed.**

## Task 4: Add Widget Action Resolution ✅ DONE 2026-05-28

**Files:**
- Modify: `app/services/widget_quality.py`
- Modify: `app/api/widgets.py`
- Modify: `client_backend/api/proxy.py`
- Create: `tests/test_widget_actions_api.py`
- Create: `tests/client_backend/test_widget_action_proxy.py`

**Verification:** `python -m pytest tests/test_widget_actions_api.py tests/client_backend/test_widget_action_proxy.py -q` → 9 passed.

**Design notes:**
- Used `payload: WidgetActionRequest | None = Body(default=None)` instead of `default_factory=WidgetActionRequest` because the file uses `from __future__ import annotations` and Pydantic v2 cannot resolve the forward-ref class in a `default_factory` until the class is fully built. The endpoint normalizes `None` → empty request body internally.
- Called `WidgetActionRequest.model_rebuild()` after class definition to force schema resolution under the future-annotations regime.
- `last_action` is recorded via `store.patch` (shallow merge), preserving widget version semantics. Errors in writing `last_action` are swallowed and logged so the action endpoint still succeeds — the rendered message is the primary contract.
- 404 strips quote characters from `KeyError(...)` strings (Python wraps the message in single quotes for `str(KeyError(...))`).

- [x] Add action message resolution helper.

```python
import re


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
```

- [x] Add `POST /widgets/{widget_id}/actions/{action_key}`.

Request:

```json
{
  "input_values": {
    "note": "optional user input"
  },
  "state_patch": {
    "control_values": {
      "damping": 0.4
    }
  }
}
```

Response:

```json
{
  "widget_id": "w-1",
  "session_id": "conversation-id",
  "action_key": "explain_current_state",
  "content": "Explain the widget state for damping=0.4 in the context of the current answer."
}
```

Behavior:
- Authenticate with the normal bearer token.
- Reuse widget ownership validation from `widget_connection`.
- Apply `state_patch` before resolving the message when provided.
- Store `last_action` in widget state so `widget_get_state` can inspect user intent on a later turn.
- Do not run the assistant in this endpoint. It only resolves the message. Frontends submit the returned message through their normal stream path.

- [x] Proxy the action endpoint through `client_backend/api/proxy.py`.

```python
@router.post("/widgets/{widget_id}/actions/{action_key}")
async def proxy_widget_action(
    widget_id: str,
    action_key: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/widgets/{widget_id}/actions/{action_key}",
    )
```

- [x] Add API tests for success, missing action, denied access, and state patching.

Run: `python -m pytest tests/test_widget_actions_api.py tests/client_backend/test_widget_action_proxy.py -q`

Expected: action resolution works through both server and client backend. **Actual: 9 passed.**

## Task 5: Upgrade Streamlit Widget Rendering ✅ DONE 2026-05-28

**Files:**
- Modify: `demo.py`
- Create: `tests/test_demo_meaningful_widgets.py`

**Verification:** `python -m pytest tests/test_demo_meaningful_widgets.py tests/test_demo_plan_widget.py -q` → 18 passed.

**Design notes:**
- Added a fixed-position `.lw-tooltip` element at the iframe root (outside `.lw-card`) so it can float above the body without clipping.
- Tooltip payload format is the pipe-delimited string `series|label|value`. The display layer renders the series as bold heading, label as the row, and value as the data line. Donut slices use `"Total||value"` because they have no series dimension.
- `data-tooltip` attached to: bar `<rect>`s (line 3974), line/area `<circle>` points (line 4012 region), donut ring + each legend row, with matching `aria-label` and SVG `<title>` for screen-reader fallback.
- Action buttons only render when the action declares `type: "assistant_message"` to match the action contract enforced server-side. Buttons live in `.lw-actions` below the chart body so the visual hierarchy stays consistent across widget types.
- `runWidgetAction` POSTs to `/widgets/{id}/actions/{key}` with the *current* `control_values` as `state_patch` (mirrors what the user sees), receives the rendered `content`, then calls `/messages/stream` with `inlineRichResponseV1: true`. The streamed response appears in the compact `.lw-action-result` region; after stream completion we append "— saved to conversation".
- Presentation rendering is scoped to the widget body block, NOT the head — the head still shows the explicit widget title (preserves the existing inline-mode chrome reduction policy). If the body payload includes `presentation.title` and the head has no title yet, we copy it over.
- Annotations render as yellow pill badges above the chart surface; only annotations with at least one of `label`/`series` are shown.

**Bugfix 2026-05-28 (post-implementation):** `streamAssistantResponse` originally used `chunk.split("\n")` in the Python triple-quoted JS template. Python interprets `\n` inside `"""..."""` as a real newline, so the rendered template arrived in the browser with a raw `0x0A` byte inside a JS `"..."` string literal — a JS parse error. The whole IIFE then failed to parse, which meant `connect()` was never invoked and the widget stayed at the initial "Connecting widget…" placeholder with no `/widgets/{id}/connection` request hitting the server. Fix: double the backslash (`"\\n"` in Python source → `\n` in rendered JS). Added `test_live_widget_component_js_strings_have_no_raw_newlines` as a regression guard that walks the rendered JS character-by-character and fails if any `"..."` literal contains an unescaped newline.

- [x] Render `presentation` as article-style context.

Implementation targets in `_build_live_widget_component_html()`:
- Use `presentation.title` and widget title with graceful fallback.
- Render `presentation.caption` below the visual, not as a heavy card.
- Render `presentation.x_label`, `presentation.y_label`, and `presentation.unit` near chart axes or metadata.
- Render `presentation.annotations` as labels, callouts, or pinned markers when the target can be resolved.
- In inline mode, reduce attachment chrome so the widget feels embedded in the answer.

- [x] Add chart hover values.

Implementation targets:
- Add a `.lw-tooltip` element to the iframe HTML.
- Add hit targets to bars, line points, and donut slices with label, series, and formatted value.
- On pointer move, show tooltip near the pointer.
- On pointer leave, hide tooltip.
- Add `aria-label` and SVG `<title>` fallback for accessibility.

- [x] Render `actions` as assistant action buttons.

Implementation targets:
- Render action buttons below local controls.
- On click, call `POST /widgets/{widget_id}/actions/{action_key}` with current `control_values` and any action input values.
- Then call existing `/messages/stream` with the returned `content`, `conversationId`, role `user`, and `inlineRichResponseV1: true`.
- Render the streamed response inside a compact action result region in the iframe.
- Show that the response is saved to the conversation after the stream completes.

- [x] Add tests that inspect generated component HTML.

```python
def test_live_widget_component_contains_hover_tooltip(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    markup = demo._build_live_widget_component_html(
        {
            "widget_id": "w-1",
            "session_id": "conv-1",
            "widget_type": "chart",
            "title": "Growth",
            "status": "active",
            "version": 1,
            "connection_endpoint": "/widgets/w-1/connection",
        },
        "token",
    )

    assert "lw-tooltip" in markup
    assert "data-tooltip" in markup
```

Run: `python -m pytest tests/test_demo_meaningful_widgets.py tests/test_demo_plan_widget.py -q`

Expected: existing Streamlit widget behavior remains green and new presentation/action hooks are present. **Actual: 18 passed.**

## Task 6: Document the AI SDK Renderer Contract ✅ DONE 2026-05-28

**Files:**
- Modify: `plans/live-widgets-frontend-integration.md`
- Modify: `README.md`

**Verification:** Manual review — both docs updated. Cross-references between README → frontend-integration plan are linked.

**Design notes:**
- Appended a new "§ 11 Meaningful Widgets — Shared State Contract" section to `live-widgets-frontend-integration.md` rather than rewriting the existing migration notes. The existing inline-rich-response migration note (§ 11 before, now superseded numbering by the new section) remains intact above it; this keeps the historical timeline readable.
- README adds a "Meaningful Widgets contract" subsection under Live Widgets with the canonical test commands and a single link to the deeper frontend doc. Avoids duplicating the JSON envelope in two places.
- Added the action endpoint row to the Widgets API table so the Postman/curl readers see it alongside `/connection`.

- [x] Add an AI SDK renderer contract section.

Required frontend behavior:
- Opt into `inlineRichResponseV1`.
- Read transient widgets from `data-rich-items`.
- Read final widgets from `messageMetadata.rich_items`.
- Hydrate widget state through `POST /widgets/{id}/connection` and the widget WebSocket.
- Render `presentation`, controls, views/variants, chart hover values, and actions the same way Streamlit does.
- For action buttons, call `POST /widgets/{id}/actions/{action_key}` and submit the returned `content` through the normal AI SDK chat stream.

- [x] Add a TypeScript reference snippet.

```ts
async function runWidgetAction(widgetId: string, actionKey: string, statePatch: unknown) {
  const response = await fetch(`/widgets/${widgetId}/actions/${actionKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state_patch: statePatch }),
  });
  if (!response.ok) throw new Error(`Widget action failed: ${response.status}`);
  const payload = await response.json();
  sendMessage({ text: payload.content });
}
```

- [x] Note that no raw widget state is added to `rich_items`; state still arrives over the widget WebSocket.

## Task 7: Renderer Consistency Tests ✅ DONE 2026-05-28

**Files:**
- Modify: `tests/test_rich_response_streaming.py`
- Modify: `tests/test_widgets_api.py`
- Modify: `tests/test_demo_rich_response.py`

**Verification:** `python -m pytest tests/test_rich_response_streaming.py tests/test_widgets_api.py tests/test_demo_rich_response.py -q` → 34 passed.

**Design notes:**
- The compactness regression in `test_rich_response_streaming.py` now asserts the *negative* — even when a `live_widget` dict contains a state blob with `presentation`/`actions`, `_widget_rich_item_from_live_widget` strips it. This protects against accidental future leaks of widget state into the streamed `rich_items`.
- The widget recovery test seeds an enriched `tool_artifacts.args.initial_state` with `presentation`, `controls`, `control_values`, and `actions` and asserts they are preserved end-to-end through `_extract_widget_snapshot_from_metadata` + `WidgetStore.restore`. This catches future changes to the snapshot extractor that might silently filter unknown keys.
- The Streamlit inline test confirms that `build_rich_response_view` does not duplicate widget chrome at the marker — the renderer hydrates over the WebSocket and the inline segment carries only the compact mount metadata.

- [x] Add regression tests that widget rich-item records stay compact.

```python
def test_live_widget_rich_item_does_not_embed_state():
    from app.core.response_constants import _widget_rich_item_from_live_widget

    item = _widget_rich_item_from_live_widget(
        {
            "widget_id": "w-1",
            "session_id": "conv-1",
            "widget_type": "chart",
            "title": "Meaningful Chart",
            "status": "active",
            "version": 1,
            "state": {"labels": ["A", "B"]},
        }
    )

    assert "state" not in item["payload"]
```

- [x] Add tests for historical widget recovery with enriched state.

Use `tool_artifacts.args.initial_state` containing `presentation`, `controls`, and `actions`, then assert `/widgets/{id}/connection` restores the widget and preserves those keys.

- [x] Add a Streamlit inline test that article chrome is not duplicated for inline widgets.

Run: `python -m pytest tests/test_rich_response_streaming.py tests/test_widgets_api.py tests/test_demo_rich_response.py -q`

Expected: rich response placement, streaming upserts, and widget restoration stay compatible. **Actual: 34 passed.**

## Task 8: End-to-End Verification ✅ DONE 2026-05-28

**Files:**
- No new files required.

**Verification summary:**
| Batch | Command | Result |
|-------|---------|--------|
| Widget backend | `pytest tests/test_widget_quality.py tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_widget_actions_api.py -q` | **98 passed** |
| Streamlit renderer | `pytest tests/test_demo_meaningful_widgets.py tests/test_demo_plan_widget.py tests/test_demo_rich_response.py -q` | **31 passed** |
| AI SDK / rich response | `pytest tests/test_rich_response_streaming.py tests/test_rich_response_metadata.py tests/test_message_history_pipeline.py -q` | **32 passed** |
| Client-backend proxy | `pytest tests/client_backend/test_widget_action_proxy.py tests/client_backend/test_server_api.py -q` | **6 passed** |

Total: **167 tests passing**, zero regressions across the existing widget runtime, rich-response, message history, and proxy suites.

- [x] Run widget-specific backend tests.

Run:

```bash
python -m pytest tests/test_widget_quality.py tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_widget_actions_api.py -q
```

Expected: all widget backend tests pass.

- [x] Run Streamlit renderer tests.

Run:

```bash
python -m pytest tests/test_demo_meaningful_widgets.py tests/test_demo_plan_widget.py tests/test_demo_rich_response.py -q
```

Expected: Streamlit widget rendering, inline placement, hover/action HTML hooks, and legacy behavior pass.

- [x] Run AI SDK/rich-response tests.

Run:

```bash
python -m pytest tests/test_rich_response_streaming.py tests/test_rich_response_metadata.py tests/test_message_history_pipeline.py -q
```

Expected: AI SDK `data-rich-items`, final `messageMetadata.rich_items`, and marker behavior pass.

- [x] Run client-backend proxy tests.

Run:

```bash
python -m pytest tests/client_backend/test_widget_action_proxy.py tests/client_backend/test_server_api.py -q
```

Expected: widget connection and action requests proxy through the local backend without changing device-runtime behavior.

## Acceptance Criteria

- Agents no longer create bare, low-value chart widgets when the data does not support the chart.
- Objective malformed widgets are rejected before they are stored.
- Article-like widget state supports presentation context, captions, labels, units, annotations, controls, and actions without requiring a visible "why this chart" field.
- Streamlit charts expose hover values for bars, line points, and donut slices.
- AI SDK clients and Streamlit render the same shared state conventions.
- Widget actions can become normal assistant turns through the existing chat stream path.
- `html` widgets remain supported as compact in-chat micro-widgets for simulations and concept visuals.
- Existing widget token, WebSocket, rich-items, message history, and sidecar proxy behavior remain backward compatible.

## Out of Scope

- Building a full React renderer inside this repository.
- Replacing the existing widget WebSocket protocol.
- Adding long-lived widget state tables in Postgres.
- Making every widget an `html` iframe.
- Showing visible rationale text such as "why this chart was created."
