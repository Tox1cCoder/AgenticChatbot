# MCP Tool UI Backend Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve MCP and MCP-App style render metadata in the backend so frontend clients can render tool results as tables, charts, cards, widgets, or app iframes instead of falling back to JSON, and update the in-repo Streamlit demo so the new contract is visible during local testing.

**Architecture:** Add a backend-only normalization layer that converts raw tool results into two outputs: compact model-facing text for `ToolMessage` and rich UI-facing render metadata for stream events and persisted `tool_artifacts`. Keep existing widget behavior and generic JSON fallback compatible while adding a stable contract for future frontend renderers.

**Tech Stack:** Python 3.10+, FastAPI, LangGraph, LangChain tool messages, MCP tool results, Vercel AI SDK UI Message Stream, pytest.

---

## Scope

This plan changes the backend plus the in-repo `demo.py` Streamlit surface. It does not build the separate production frontend renderer, but it gives that frontend enough structured data to build one.

Backend responsibilities:

- Preserve structured MCP result fields instead of converting every result to `str(result)`.
- Keep `ToolMessage.content` compact and safe for the model.
- Persist rich UI metadata in `message_metadata.tool_artifacts`.
- Emit rich UI metadata in live AI SDK `tool-output-available` events.
- Keep the existing `live_widgets` path working.
- Keep unknown tool results renderable as plain text or JSON.

Frontend responsibilities after this plan:

- Read `tool_artifacts[].render`.
- Read AI SDK `tool-output-available.render`.
- Choose a renderer by `render.type`.
- Fall back to JSON/text when the render type is unknown.

Streamlit demo responsibilities after this plan:

- Render `tool_artifacts[].render` when present.
- Show table/chart/resource/text/error previews instead of raw JSON for common render types.
- Show MCP/App template metadata clearly when the demo cannot mount the app iframe.
- Keep raw JSON available in an expander for debugging.

## Current Backend Behavior

Important current files:

- `app/ai/tool_execution.py`
  - Executes tools.
  - Calls `extract_content_from_result(result)`.
  - Converts the result to `str(result)`.
  - Builds `tool_artifacts`.
- `app/ai/utils.py`
  - Contains `extract_content_from_result()`, `format_tool_result()`, and `make_json_safe()`.
- `app/ai/graph.py`
  - Turns tool outputs into `ToolMessage` objects.
  - Streams `tool_start` and `tool_end` events.
- `app/services/ai_service.py`
  - Converts graph `tool_start` / `tool_end` events into canonical `tool` events.
- `app/services/stream_events.py`
  - Builds canonical tool event payloads.
- `app/api/ai_sdk.py`
  - Converts canonical tool events into Vercel AI SDK events.
  - Emits `tool-input-start`, `tool-input-available`, and `tool-output-available`.
- `app/core/response_constants.py`
  - Derives `live_widgets` from widget tool artifacts.
- `tests/test_widget_runtime.py`
  - Existing coverage for widget artifact behavior.
- `tests/client_backend/test_sse_keepalive.py`
  - Existing coverage for AI SDK stream response behavior.

Main gap:

- Generic MCP results are flattened before UI metadata can survive. This makes external tools look like JSON even when the tool returned structured content or an app/widget template URI.

## Target Backend Contract

Every successful tool artifact should keep the existing fields and add `render`.

```json
{
  "tool_call_id": "call_123",
  "tool": "canva_create_design",
  "args": {"prompt": "quarterly roadmap deck"},
  "output": "Created presentation: Quarterly Roadmap",
  "error": null,
  "status": "success",
  "render": {
    "version": 1,
    "type": "mcp_app",
    "title": "Quarterly Roadmap",
    "model_content": "Created presentation: Quarterly Roadmap",
    "text": "Created presentation: Quarterly Roadmap",
    "structured_content": {
      "presentation_id": "deck_123",
      "slides": [
        {"title": "Overview"},
        {"title": "Timeline"}
      ]
    },
    "content": [
      {"type": "text", "text": "Created presentation: Quarterly Roadmap"}
    ],
    "resources": [
      {
        "uri": "ui://canva/presentation-viewer.html",
        "mime_type": "text/html",
        "title": "Presentation viewer"
      }
    ],
    "template_uri": "ui://canva/presentation-viewer.html",
    "ui_meta": {
      "openai/outputTemplate": "ui://canva/presentation-viewer.html"
    }
  }
}
```

Allowed `render.type` values for this backend pass:

- `mcp_app`: an MCP/App result advertises a UI template URI.
- `live_widget`: existing in-repo `widget_create` / `widget_update` result.
- `chart`: structured result has chart-like keys.
- `table`: structured result has table-like keys.
- `image`: content/resource contains image media.
- `resource`: result contains MCP resources but no app template.
- `json`: structured result exists but no richer type is inferred.
- `text`: plain text result.
- `error`: failed tool result.

The model-facing `ToolMessage.content` must use `render.model_content`, not the full UI payload.

## File Structure

Create:

- `app/ai/tool_result_rendering.py`
  - Owns result normalization and render type inference.
  - Does not call tools.
  - Does not know about FastAPI or Streamlit.

- `tests/test_tool_result_rendering.py`
  - Unit tests for normalization, render type inference, app template extraction, and text fallback.

Modify:

- `app/ai/tool_execution.py`
  - Use `normalize_tool_result_for_rendering()` before creating outputs and artifacts.
  - Extend `build_tool_artifact()` to accept `render`.

- `app/ai/graph.py`
  - Preserve `render` in state context keyed by `tool_call_id`.
  - Attach `render` to streamed `tool_end` events when available.

- `app/services/stream_events.py`
  - Allow canonical tool events to carry `render`.

- `app/services/ai_service.py`
  - Forward `render` from graph events into canonical tool events.

- `app/api/ai_sdk.py`
  - Include `render` in `tool-output-available` events.

- `demo.py`
  - Render persisted `tool_artifacts[].render` in message history and tool trace panels.
  - Keep raw JSON fallback and existing live widget/canvas renderers.

- `tests/test_widget_runtime.py`
  - Add compatibility checks that widget artifacts still derive `live_widgets`.

- `tests/client_backend/test_sse_keepalive.py`
  - Add AI SDK stream test coverage for `tool-output-available.render`.

Do not modify:

- external frontend repo
- `client_backend/` proxy behavior unless a test proves it strips the new field
- existing unrelated local changes in `README.md`, `app/ai/text_normalization.py`, or `app/ai/tool_search_scoring.py`

---

### Task 1: Add Tool Result Normalizer Tests

**Files:**

- Create: `tests/test_tool_result_rendering.py`

- [x] **Step 1: Create failing tests for MCP/App result normalization**

Add this test file:

```python
from __future__ import annotations

from app.ai.tool_result_rendering import normalize_tool_result_for_rendering


def test_normalizes_apps_sdk_template_result():
    raw_result = {
        "content": [
            {"type": "text", "text": "Created presentation: Quarterly Roadmap"}
        ],
        "structuredContent": {
            "presentation_id": "deck_123",
            "slides": [{"title": "Overview"}, {"title": "Timeline"}],
        },
        "_meta": {
            "openai/outputTemplate": "ui://canva/presentation-viewer.html",
            "title": "Quarterly Roadmap",
        },
    }

    normalized = normalize_tool_result_for_rendering(
        raw_result,
        tool_name="canva_create_presentation",
    )

    assert normalized.model_content == "Created presentation: Quarterly Roadmap"
    assert normalized.output_preview == "Created presentation: Quarterly Roadmap"
    assert normalized.render["version"] == 1
    assert normalized.render["type"] == "mcp_app"
    assert normalized.render["template_uri"] == "ui://canva/presentation-viewer.html"
    assert normalized.render["structured_content"]["presentation_id"] == "deck_123"
    assert normalized.render["ui_meta"]["openai/outputTemplate"] == (
        "ui://canva/presentation-viewer.html"
    )


def test_normalizes_snake_case_structured_content_and_resource_uri():
    raw_result = {
        "content": [{"type": "text", "text": "Dashboard ready"}],
        "structured_content": {"chart_type": "bar", "labels": ["Q1"], "datasets": []},
        "_meta": {"ui": {"resourceUri": "ui://charts/dashboard.html"}},
    }

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="chart_tool")

    assert normalized.render["type"] == "mcp_app"
    assert normalized.render["template_uri"] == "ui://charts/dashboard.html"
    assert normalized.render["structured_content"]["chart_type"] == "bar"


def test_infers_chart_without_template_uri():
    raw_result = {
        "chart_type": "line",
        "labels": ["Mon", "Tue"],
        "datasets": [{"label": "Visits", "data": [10, 12]}],
    }

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="chart_tool")

    assert normalized.model_content.startswith("{")
    assert normalized.render["type"] == "chart"
    assert normalized.render["structured_content"]["labels"] == ["Mon", "Tue"]


def test_infers_table_without_template_uri():
    raw_result = {
        "columns": ["Name", "Score"],
        "rows": [["Ada", 99], ["Grace", 98]],
    }

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="table_tool")

    assert normalized.render["type"] == "table"
    assert normalized.render["structured_content"]["rows"][0] == ["Ada", 99]


def test_unwraps_langchain_text_content_blocks():
    raw_result = [{"type": "text", "text": "The answer is 42."}]

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="answer_tool")

    assert normalized.model_content == "The answer is 42."
    assert normalized.output_preview == "The answer is 42."
    assert normalized.render["type"] == "text"
    assert normalized.render["content"] == [{"type": "text", "text": "The answer is 42."}]


def test_preserves_image_content_as_image_render():
    raw_result = {
        "content": [
            {
                "type": "image",
                "mimeType": "image/png",
                "data": "iVBORw0KGgo=",
            }
        ]
    }

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="image_tool")

    assert normalized.model_content == "[image/png image]"
    assert normalized.render["type"] == "image"
    assert normalized.render["content"][0]["type"] == "image"


def test_error_normalization_uses_error_render_type():
    normalized = normalize_tool_result_for_rendering(
        "Error: permission denied",
        tool_name="dangerous_tool",
        error="permission denied",
    )

    assert normalized.model_content == "Error: permission denied"
    assert normalized.render["type"] == "error"
    assert normalized.render["error"] == "permission denied"
```

- [x] **Step 2: Run tests and verify they fail because the module does not exist**

Run:

```powershell
pytest tests/test_tool_result_rendering.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'app.ai.tool_result_rendering'
```

---

### Task 2: Implement Backend Tool Result Normalizer

**Files:**

- Create: `app/ai/tool_result_rendering.py`
- Test: `tests/test_tool_result_rendering.py`

- [x] **Step 1: Add the normalizer module**

Create `app/ai/tool_result_rendering.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.ai.utils import make_json_safe


_WIDGET_TOOLS = {"widget_create", "widget_update"}


@dataclass(frozen=True)
class NormalizedToolRender:
    model_content: str
    output_preview: str
    render: dict[str, Any]


def normalize_tool_result_for_rendering(
    result: Any,
    *,
    tool_name: str,
    error: str | None = None,
) -> NormalizedToolRender:
    safe_result = make_json_safe(result)
    content_blocks = _extract_content_blocks(safe_result)
    structured_content = _extract_structured_content(safe_result)
    ui_meta = _extract_ui_meta(safe_result)
    resources = _extract_resources(safe_result)
    template_uri = _extract_template_uri(ui_meta, resources)
    text = _extract_text(content_blocks)

    if structured_content is None:
        structured_content = _infer_structured_content_from_result(safe_result, content_blocks)

    if not text:
        text = _build_model_content(
            safe_result=safe_result,
            structured_content=structured_content,
            content_blocks=content_blocks,
            error=error,
        )

    render_type = _infer_render_type(
        tool_name=tool_name,
        error=error,
        template_uri=template_uri,
        structured_content=structured_content,
        content_blocks=content_blocks,
        resources=resources,
        text=text,
    )

    render: dict[str, Any] = {
        "version": 1,
        "type": render_type,
        "model_content": text,
        "text": text,
    }

    title = _extract_title(safe_result, ui_meta)
    if title:
        render["title"] = title
    if structured_content is not None:
        render["structured_content"] = structured_content
    if content_blocks:
        render["content"] = content_blocks
    if resources:
        render["resources"] = resources
    if template_uri:
        render["template_uri"] = template_uri
    if ui_meta:
        render["ui_meta"] = ui_meta
    if error:
        render["error"] = str(error)

    return NormalizedToolRender(
        model_content=text,
        output_preview=_preview_text(text),
        render=render,
    )


def _extract_content_blocks(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            return [make_json_safe(item) for item in content if isinstance(item, dict)]
        if isinstance(content, dict):
            return [make_json_safe(content)]

    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        typed_blocks = [
            make_json_safe(item)
            for item in value
            if isinstance(item, dict) and isinstance(item.get("type"), str)
        ]
        if typed_blocks:
            return typed_blocks

    return []


def _extract_structured_content(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    for key in ("structuredContent", "structured_content"):
        candidate = value.get(key)
        if candidate not in (None, "", [], {}):
            return make_json_safe(candidate)
    return None


def _extract_ui_meta(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    meta = value.get("_meta") or value.get("meta")
    return make_json_safe(meta) if isinstance(meta, dict) else {}


def _extract_resources(value: Any) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []

    def add_resource(candidate: Any) -> None:
        if not isinstance(candidate, dict):
            return
        uri = candidate.get("uri") or candidate.get("url")
        if not uri:
            return
        resource = {
            "uri": str(uri),
            "mime_type": str(
                candidate.get("mimeType")
                or candidate.get("mime_type")
                or candidate.get("contentType")
                or ""
            ),
        }
        title = candidate.get("title") or candidate.get("name")
        if title:
            resource["title"] = str(title)
        resources.append(resource)

    if isinstance(value, dict):
        for key in ("resources", "resource"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                for item in candidate:
                    add_resource(item)
            else:
                add_resource(candidate)

    for block in _extract_content_blocks(value):
        if block.get("type") in {"resource", "resource_link", "resourceLink"}:
            add_resource(block.get("resource") or block)

    return resources


def _extract_template_uri(ui_meta: dict[str, Any], resources: list[dict[str, Any]]) -> str | None:
    for key in ("openai/outputTemplate", "outputTemplate", "ui/resourceUri"):
        value = ui_meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    ui_value = ui_meta.get("ui")
    if isinstance(ui_value, dict):
        for key in ("resourceUri", "resource_uri", "uri"):
            value = ui_value.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for resource in resources:
        uri = resource.get("uri")
        mime_type = str(resource.get("mime_type") or "")
        if isinstance(uri, str) and uri.startswith("ui://"):
            return uri
        if isinstance(uri, str) and mime_type == "text/html":
            return uri

    return None


def _extract_text(content_blocks: list[dict[str, Any]]) -> str:
    text_parts: list[str] = []
    media_parts: list[str] = []

    for block in content_blocks:
        block_type = str(block.get("type") or "").lower()
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block_type == "image":
            mime = block.get("mimeType") or block.get("mime_type") or "image"
            media_parts.append(f"[{mime} image]")
        elif block_type == "audio":
            mime = block.get("mimeType") or block.get("mime_type") or "audio"
            media_parts.append(f"[{mime} audio]")
        elif block_type in {"resource", "resource_link", "resourcelink"}:
            uri = block.get("uri") or block.get("url")
            if uri:
                media_parts.append(f"[resource: {uri}]")

    joined = "\n".join(part.strip() for part in text_parts if part.strip()).strip()
    if joined:
        return joined
    return "\n".join(media_parts).strip()


def _infer_structured_content_from_result(value: Any, content_blocks: list[dict[str, Any]]) -> Any:
    if isinstance(value, dict):
        metadata_keys = {"content", "_meta", "meta", "resources", "resource"}
        remaining = {k: v for k, v in value.items() if k not in metadata_keys}
        return remaining or None

    if isinstance(value, list) and not content_blocks:
        return value

    return None


def _build_model_content(
    *,
    safe_result: Any,
    structured_content: Any,
    content_blocks: list[dict[str, Any]],
    error: str | None,
) -> str:
    if error:
        return f"Error: {error}"
    if structured_content is not None:
        return json.dumps(structured_content, ensure_ascii=False, default=str)
    if content_blocks:
        return json.dumps(content_blocks, ensure_ascii=False, default=str)
    if isinstance(safe_result, str):
        return safe_result
    return json.dumps(safe_result, ensure_ascii=False, default=str)


def _infer_render_type(
    *,
    tool_name: str,
    error: str | None,
    template_uri: str | None,
    structured_content: Any,
    content_blocks: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    text: str,
) -> str:
    if error:
        return "error"
    if template_uri:
        return "mcp_app"
    if tool_name in _WIDGET_TOOLS:
        return "live_widget"
    if any(str(block.get("type") or "").lower() == "image" for block in content_blocks):
        return "image"
    if _looks_like_chart(structured_content):
        return "chart"
    if _looks_like_table(structured_content):
        return "table"
    if resources:
        return "resource"
    if structured_content is not None:
        return "json"
    return "text" if text else "json"


def _looks_like_chart(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    return bool({"chart_type", "chartType"} & keys) and "labels" in keys and "datasets" in keys


def _looks_like_table(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("columns"), list) and isinstance(value.get("rows"), list)


def _extract_title(value: Any, ui_meta: dict[str, Any]) -> str | None:
    for candidate in (
        ui_meta.get("title"),
        value.get("title") if isinstance(value, dict) else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _preview_text(value: str, max_chars: int = 1000) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()
```

- [x] **Step 2: Run normalizer tests**

Run:

```powershell
pytest tests/test_tool_result_rendering.py -q
```

Expected:

```text
7 passed
```

---

### Task 3: Preserve Render Metadata in Tool Artifacts

**Files:**

- Modify: `app/ai/tool_execution.py`
- Modify: `tests/test_widget_runtime.py`
- Test: `tests/test_widget_runtime.py::TestBuildToolArtifactWidgets`

- [x] **Step 1: Add failing artifact tests**

Append these tests to `TestBuildToolArtifactWidgets` in `tests/test_widget_runtime.py`:

```python
    def test_tool_artifact_preserves_render_payload(self):
        render = {
            "version": 1,
            "type": "mcp_app",
            "template_uri": "ui://canva/presentation-viewer.html",
            "structured_content": {"presentation_id": "deck_123"},
        }

        artifact = build_tool_artifact(
            tool_call_id="tc-render",
            tool_name="canva_create_presentation",
            tool_args={"prompt": "roadmap"},
            output_text="Created presentation",
            error=None,
            render=render,
        )

        assert artifact["output"] == "Created presentation"
        assert artifact["render"] == render

    def test_error_tool_artifact_preserves_error_render_payload(self):
        render = {
            "version": 1,
            "type": "error",
            "model_content": "Error: permission denied",
            "error": "permission denied",
        }

        artifact = build_tool_artifact(
            tool_call_id="tc-error",
            tool_name="dangerous_tool",
            tool_args={},
            output_text="Error: permission denied",
            error="permission denied",
            render=render,
        )

        assert artifact["status"] == "error"
        assert artifact["render"]["type"] == "error"
```

- [x] **Step 2: Run focused tests and verify they fail on unexpected keyword**

Run:

```powershell
pytest tests/test_widget_runtime.py::TestBuildToolArtifactWidgets -q
```

Expected:

```text
TypeError: build_tool_artifact() got an unexpected keyword argument 'render'
```

- [x] **Step 3: Extend `build_tool_artifact()`**

In `app/ai/tool_execution.py`, change the function signature:

```python
def build_tool_artifact(
    *,
    tool_call_id: str | None,
    tool_name: str,
    tool_args: Any,
    output_text: str | None,
    error: str | None,
    status: str | None = None,
    max_output_chars: int = 1000,
    render: dict[str, Any] | None = None,
) -> dict[str, Any]:
```

Inside the function, after the existing `output` assignment block, add:

```python
    if render is not None:
        artifact["render"] = render
```

Keep `_compact_widget_artifact_output()` unchanged so `live_widgets` extraction stays compatible.

- [x] **Step 4: Run focused tests**

Run:

```powershell
pytest tests/test_widget_runtime.py::TestBuildToolArtifactWidgets -q
```

Expected:

```text
4 passed
```

---

### Task 4: Use Normalized Render Results During Tool Execution

**Files:**

- Modify: `app/ai/tool_execution.py`
- Create or modify: `tests/test_tool_execution_rendering.py`

- [x] **Step 1: Add failing execution tests**

Create `tests/test_tool_execution_rendering.py`:

```python
from __future__ import annotations

import pytest

from app.ai.tool_execution import execute_tool_calls


class _FakeTool:
    name = "canva_create_presentation"

    async def ainvoke(self, args):
        return {
            "content": [{"type": "text", "text": "Created presentation"}],
            "structuredContent": {"presentation_id": "deck_123"},
            "_meta": {"openai/outputTemplate": "ui://canva/presentation-viewer.html"},
        }


class _FailingTool:
    name = "failing_tool"

    async def ainvoke(self, args):
        raise ValueError("permission denied")


@pytest.mark.asyncio
async def test_execute_tool_calls_returns_model_text_and_render_artifact():
    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[
            {
                "id": "tool-call-1",
                "name": "canva_create_presentation",
                "args": {"prompt": "roadmap"},
            }
        ],
        tool_map={"canva_create_presentation": _FakeTool()},
    )

    assert outputs == [
        {
            "tool_call_id": "tool-call-1",
            "name": "canva_create_presentation",
            "content": "Created presentation",
            "render": artifacts[0]["render"],
        }
    ]
    assert artifacts[0]["output"] == "Created presentation"
    assert artifacts[0]["render"]["type"] == "mcp_app"
    assert artifacts[0]["render"]["template_uri"] == "ui://canva/presentation-viewer.html"
    assert artifacts[0]["render"]["structured_content"]["presentation_id"] == "deck_123"
    assert images == []


@pytest.mark.asyncio
async def test_execute_tool_calls_preserves_error_render_artifact():
    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[{"id": "tool-call-err", "name": "failing_tool", "args": {}}],
        tool_map={"failing_tool": _FailingTool()},
    )

    assert outputs[0]["content"] == "Error: permission denied"
    assert outputs[0]["render"]["type"] == "error"
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["render"]["type"] == "error"
    assert artifacts[0]["render"]["error"] == "permission denied"
    assert images == []
```

- [x] **Step 2: Run tests and verify they fail because `render` is missing**

Run:

```powershell
pytest tests/test_tool_execution_rendering.py -q
```

Expected:

```text
KeyError: 'render'
```

- [x] **Step 3: Integrate `normalize_tool_result_for_rendering()` in successful execution paths**

In `app/ai/tool_execution.py`, add the import:

```python
from .tool_result_rendering import normalize_tool_result_for_rendering
```

In the main success path, replace:

```python
            result = await invoke_tool(tool, tool_args)
            result = extract_content_from_result(result)
            result_text = str(result)

            outputs.append({"tool_call_id": tool_id, "name": tool_name, "content": result_text})
```

with:

```python
            result = await invoke_tool(tool, tool_args)
            result = extract_content_from_result(result)
            normalized_result = normalize_tool_result_for_rendering(
                result,
                tool_name=tool_name,
            )
            result_text = normalized_result.model_content

            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": result_text,
                    "render": normalized_result.render,
                }
            )
```

In the matching `build_tool_artifact()` call, pass:

```python
                    render=normalized_result.render,
```

Apply the same pattern in the reconnect retry success path.

- [x] **Step 4: Integrate render metadata in error paths**

For the generic exception path, before appending `outputs`, add:

```python
            normalized_result = normalize_tool_result_for_rendering(
                error_msg,
                tool_name=tool_name,
                error=str(exc),
            )
```

Then append:

```python
            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": normalized_result.model_content,
                    "render": normalized_result.render,
                }
            )
```

Pass `render=normalized_result.render` to `build_tool_artifact()`.

Repeat the same shape for:

- missing tool errors
- device binding errors
- MCP reconnect failure errors

- [x] **Step 5: Run focused execution tests**

Run:

```powershell
pytest tests/test_tool_execution_rendering.py -q
```

Expected:

```text
2 passed
```

- [x] **Step 6: Run existing widget and recovery tests**

Run:

```powershell
pytest tests/test_widget_runtime.py tests/test_tool_execution_recovery.py -q
```

Expected:

```text
all selected tests pass
```

---

### Task 5: Persist Render Metadata in Graph State and Stream Tool End Events

**Files:**

- Modify: `app/ai/graph.py`
- Test: `tests/test_tool_execution_rendering.py`

- [x] **Step 1: Add unit test for render lookup helper**

Append to `tests/test_tool_execution_rendering.py`:

```python
from app.ai.graph import MultiAgentWorkflow


def test_lookup_tool_render_payload_from_state_context():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state_values = {
        "context": {
            "tool_render_results": {
                "tool-call-1": {
                    "version": 1,
                    "type": "mcp_app",
                    "template_uri": "ui://canva/presentation-viewer.html",
                }
            }
        }
    }

    render = workflow._lookup_tool_render_payload(state_values, "tool-call-1")

    assert render["type"] == "mcp_app"
    assert render["template_uri"] == "ui://canva/presentation-viewer.html"
```

- [x] **Step 2: Run the new test and verify helper is missing**

Run:

```powershell
pytest tests/test_tool_execution_rendering.py::test_lookup_tool_render_payload_from_state_context -q
```

Expected:

```text
AttributeError: 'MultiAgentWorkflow' object has no attribute '_lookup_tool_render_payload'
```

- [x] **Step 3: Store render results by tool call ID**

In `app/ai/graph.py`, update `_apply_tool_outputs_to_state()`.

After the existing `if tool_artifacts:` block, add:

```python
        render_results = dict(context.get("tool_render_results", {}))
        for output in tool_outputs:
            tool_call_id = output.get("tool_call_id")
            render = output.get("render")
            if tool_call_id and isinstance(render, dict):
                render_results[str(tool_call_id)] = make_json_safe(render)
        if render_results:
            context["tool_render_results"] = render_results
```

In the RAG tools path where `context["tool_artifacts"]` is extended manually, add the same `tool_render_results` update after `tool_outputs` is built.

- [x] **Step 4: Add render lookup helper**

Add this method to `MultiAgentWorkflow`:

```python
    @staticmethod
    def _lookup_tool_render_payload(
        state_values: dict[str, Any] | None,
        tool_call_id: Any,
    ) -> dict[str, Any] | None:
        if not state_values or not tool_call_id:
            return None
        context = state_values.get("context")
        if not isinstance(context, dict):
            return None
        render_results = context.get("tool_render_results")
        if not isinstance(render_results, dict):
            return None
        render = render_results.get(str(tool_call_id))
        return render if isinstance(render, dict) else None
```

- [x] **Step 5: Attach render to streamed `tool_end` events**

In both stream loops in `app/ai/graph.py`, replace each `ToolMessage` event block shaped like:

```python
                                            yield {
                                                "type": "tool_end",
                                                "name": getattr(last_msg, "name", "unknown"),
                                                "tool_call_id": getattr(
                                                    last_msg, "tool_call_id", None
                                                ),
                                                "result": make_json_safe(last_msg.content),
                                            }
```

with:

```python
                                            tool_call_id = getattr(
                                                last_msg, "tool_call_id", None
                                            )
                                            event_payload = {
                                                "type": "tool_end",
                                                "name": getattr(last_msg, "name", "unknown"),
                                                "tool_call_id": tool_call_id,
                                                "result": make_json_safe(last_msg.content),
                                            }
                                            render_payload = self._lookup_tool_render_payload(
                                                last_state_values,
                                                tool_call_id,
                                            ) or self._lookup_tool_render_payload(
                                                node_state if isinstance(node_state, dict) else None,
                                                tool_call_id,
                                            )
                                            if render_payload:
                                                event_payload["render"] = render_payload
                                            yield event_payload
```

Use the same local variable names in both occurrences to keep behavior consistent across stream implementations.

- [x] **Step 6: Run graph-related focused tests**

Run:

```powershell
pytest tests/test_tool_execution_rendering.py tests/test_client_tool_isolation.py::test_execute_agent_tool_calls_persists_deferred_snapshot_to_state_context -q
```

Expected:

```text
all selected tests pass
```

---

### Task 6: Forward Render Metadata Through Canonical Stream Events

**Files:**

- Modify: `app/services/stream_events.py`
- Modify: `app/services/ai_service.py`
- Test: `tests/test_tool_execution_rendering.py`

- [x] **Step 1: Add test for canonical tool event render field**

Append to `tests/test_tool_execution_rendering.py`:

```python
from app.services.stream_events import build_canonical_tool_event


def test_canonical_tool_event_preserves_render_payload():
    render = {
        "version": 1,
        "type": "mcp_app",
        "template_uri": "ui://canva/presentation-viewer.html",
    }

    event = build_canonical_tool_event(
        phase="end",
        name="canva_create_presentation",
        tool_call_id="tool-call-1",
        result="Created presentation",
        render=render,
    )

    assert event["result"] == "Created presentation"
    assert event["render"] == render
```

- [x] **Step 2: Run the test and verify failure**

Run:

```powershell
pytest tests/test_tool_execution_rendering.py::test_canonical_tool_event_preserves_render_payload -q
```

Expected:

```text
TypeError: build_canonical_tool_event() got an unexpected keyword argument 'render'
```

- [x] **Step 3: Extend canonical event builder**

In `app/services/stream_events.py`, change the function signature:

```python
def build_canonical_tool_event(
    *,
    phase: Any,
    name: str | None,
    tool_call_id: Any,
    args: Any = None,
    result: Any = None,
    duration_ms: int | None = None,
    state: str | None = None,
    render: dict[str, Any] | None = None,
) -> dict[str, Any]:
```

Before returning `payload`, add:

```python
    if render is not None:
        payload["render"] = render
```

- [x] **Step 4: Forward render from AI service**

In `app/services/ai_service.py`, inside the `event_type == "tool_end"` branch, pass:

```python
                    render=make_json_safe(event.get("render")),
```

to `build_canonical_tool_event()`.

Do not pass `render` for tool start events because render metadata is only known after execution.

- [x] **Step 5: Run focused tests**

Run:

```powershell
pytest tests/test_tool_execution_rendering.py::test_canonical_tool_event_preserves_render_payload -q
```

Expected:

```text
1 passed
```

---

### Task 7: Include Render Metadata in AI SDK Tool Output Events

**Files:**

- Modify: `app/api/ai_sdk.py`
- Modify: `tests/client_backend/test_sse_keepalive.py`

- [x] **Step 1: Add failing AI SDK stream test**

Append this test to `tests/client_backend/test_sse_keepalive.py`:

```python
@_skip_server
@pytest.mark.asyncio
async def test_ai_sdk_tool_output_event_preserves_render_payload():
    async def tool_event_source():
        yield {
            "type": "tool",
            "phase": "start",
            "name": "canva_create_presentation",
            "tool_call_id": "tool-call-1",
            "args": {"prompt": "roadmap"},
        }
        yield {
            "type": "tool",
            "phase": "end",
            "name": "canva_create_presentation",
            "tool_call_id": "tool-call-1",
            "result": "Created presentation",
            "render": {
                "version": 1,
                "type": "mcp_app",
                "template_uri": "ui://canva/presentation-viewer.html",
            },
        }
        yield {
            "type": "complete",
            "response": {
                "content": "Created presentation",
                "message_metadata": {},
            },
        }

    state = StreamState(message_id="msg-render", text_id="txt-render", reasoning_id="rsn-render")

    response = _build_ui_message_stream_response(tool_event_source, state)
    collected: list[str] = []
    async for chunk in response.body_iterator:
        collected.append(chunk)

    events = [
        line[6:]
        for line in "".join(collected).split("\n")
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]

    tool_output_events = [
        json.loads(event)
        for event in events
        if '"tool-output-available"' in event
    ]

    assert len(tool_output_events) == 1
    assert tool_output_events[0]["output"] == "Created presentation"
    assert tool_output_events[0]["render"]["type"] == "mcp_app"
    assert tool_output_events[0]["render"]["template_uri"] == (
        "ui://canva/presentation-viewer.html"
    )
```

If `json` is not already imported in `tests/client_backend/test_sse_keepalive.py`, add:

```python
import json
```

- [x] **Step 2: Run focused test and verify `render` is missing**

Run:

```powershell
pytest tests/client_backend/test_sse_keepalive.py::test_ai_sdk_tool_output_event_preserves_render_payload -q
```

Expected:

```text
KeyError: 'render'
```

- [x] **Step 3: Add render to AI SDK tool output events**

In `app/api/ai_sdk.py`, update `ToolEventHandler._handle_tool_end()`.

Replace:

```python
        yield _sse(
            {
                "type": "tool-output-available",
                "toolCallId": tool_call_id,
                "output": _coerce_json_object(_clean_tool_output(output)),
            }
        )
```

with:

```python
        payload = {
            "type": "tool-output-available",
            "toolCallId": tool_call_id,
            "output": _coerce_json_object(_clean_tool_output(output)),
        }
        render = event.get("render")
        if isinstance(render, dict):
            payload["render"] = _coerce_json_object(_clean_tool_output(render))
        yield _sse(payload)
```

- [x] **Step 4: Run focused AI SDK test**

Run:

```powershell
pytest tests/client_backend/test_sse_keepalive.py::test_ai_sdk_tool_output_event_preserves_render_payload -q
```

Expected:

```text
1 passed
```

---

### Task 8: Preserve Render Metadata on Completion Message Metadata

**Files:**

- Modify: `app/core/response_constants.py`
- Modify: `tests/test_widget_runtime.py`

- [x] **Step 1: Add compatibility test for render metadata in bot metadata**

Append to `TestBuildBotMetadataWidgets` in `tests/test_widget_runtime.py`:

```python
    def test_preserves_render_payload_in_tool_artifacts(self):
        render = {
            "version": 1,
            "type": "mcp_app",
            "template_uri": "ui://canva/presentation-viewer.html",
        }

        class FakeResponse:
            metadata = {}
            tool_artifacts = [
                {
                    "tool": "canva_create_presentation",
                    "output": "Created presentation",
                    "status": "success",
                    "render": render,
                }
            ]

        metadata = build_bot_metadata(FakeResponse())

        assert metadata["tool_artifacts"][0]["render"] == render
        assert "live_widgets" not in metadata
```

- [x] **Step 2: Run test**

Run:

```powershell
pytest tests/test_widget_runtime.py::TestBuildBotMetadataWidgets::test_preserves_render_payload_in_tool_artifacts -q
```

Expected:

```text
1 passed
```

If it fails, fix only `build_bot_metadata()` so it copies `response.tool_artifacts` without dropping unknown keys. Do not special-case Canva or any external MCP tool name.

---

### Task 9: Update Streamlit Demo Tool Rendering

**Files:**

- Modify: `demo.py`

- [x] **Step 1: Add demo render helpers**

Add these helpers near `render_tool_result_payload()` in `demo.py`:

```python
def _get_render_structured_content(render: dict[str, Any]) -> Any:
    if not isinstance(render, dict):
        return None
    return render.get("structured_content") or render.get("structuredContent")


def _render_table_tool_result(render: dict[str, Any]) -> bool:
    structured = _get_render_structured_content(render)
    if not isinstance(structured, dict):
        return False

    columns = structured.get("columns")
    rows = structured.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return False

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(row)
        elif isinstance(row, list):
            normalized_rows.append(
                {
                    str(column): row[index] if index < len(row) else None
                    for index, column in enumerate(columns)
                }
            )

    if not normalized_rows:
        st.caption("Table result contained no rows.")
        return True

    st.dataframe(normalized_rows, width="stretch", hide_index=True)
    return True


def _render_chart_tool_result(render: dict[str, Any]) -> bool:
    structured = _get_render_structured_content(render)
    if not isinstance(structured, dict):
        return False

    labels = structured.get("labels")
    datasets = structured.get("datasets")
    if not isinstance(labels, list) or not isinstance(datasets, list):
        return False

    chart_rows: list[dict[str, Any]] = []
    for label_index, label in enumerate(labels):
        row: dict[str, Any] = {"label": label}
        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            name = str(dataset.get("label") or f"Series {len(row)}")
            data = dataset.get("data")
            if isinstance(data, list) and label_index < len(data):
                row[name] = data[label_index]
        chart_rows.append(row)

    if not chart_rows:
        st.caption("Chart result contained no plottable values.")
        return True

    chart_type = str(
        structured.get("chart_type") or structured.get("chartType") or "line"
    ).lower()
    chart_data = {row["label"]: {k: v for k, v in row.items() if k != "label"} for row in chart_rows}
    if chart_type == "bar":
        st.bar_chart(chart_data)
    elif chart_type == "area":
        st.area_chart(chart_data)
    else:
        st.line_chart(chart_data)

    with st.expander("Chart data", expanded=False):
        st.dataframe(chart_rows, width="stretch", hide_index=True)
    return True


def _render_resource_tool_result(render: dict[str, Any]) -> bool:
    resources = render.get("resources")
    if not isinstance(resources, list) or not resources:
        return False

    for resource in resources:
        if not isinstance(resource, dict):
            continue
        uri = str(resource.get("uri") or "")
        title = str(resource.get("title") or uri or "Resource")
        mime_type = str(resource.get("mime_type") or resource.get("mimeType") or "")
        if uri.startswith(("http://", "https://")):
            st.link_button(title, uri, width="stretch")
        else:
            st.markdown(f"**{title}**")
            st.code(uri, language="text")
        if mime_type:
            st.caption(mime_type)
    return True


def _render_mcp_app_tool_result(render: dict[str, Any]) -> bool:
    template_uri = render.get("template_uri") or render.get("templateUri")
    if not isinstance(template_uri, str) or not template_uri.strip():
        return False

    title = render.get("title") or "MCP app view"
    st.info(
        "This tool returned an MCP app template. The Streamlit demo cannot mount "
        "arbitrary MCP app iframes yet, but the frontend can use this URI to render it."
    )
    st.markdown(f"**{html.escape(str(title))}**", unsafe_allow_html=True)
    st.code(template_uri, language="text")

    structured = _get_render_structured_content(render)
    if structured not in (None, "", [], {}):
        with st.expander("Structured app data", expanded=False):
            render_tool_result_payload(structured, use_expander=False)
    return True


def render_tool_render_payload(render: Any, fallback_output: Any = None) -> bool:
    if not isinstance(render, dict):
        return False

    render_type = str(render.get("type") or "").lower()
    title = render.get("title")
    if isinstance(title, str) and title.strip():
        st.markdown(f"**{html.escape(title.strip())}**", unsafe_allow_html=True)

    if render_type == "error":
        error = render.get("error") or render.get("text") or fallback_output
        st.error(str(error or "Tool execution failed."))
        return True

    if render_type == "mcp_app":
        rendered = _render_mcp_app_tool_result(render)
    elif render_type == "table":
        rendered = _render_table_tool_result(render)
    elif render_type == "chart":
        rendered = _render_chart_tool_result(render)
    elif render_type in {"resource", "image"}:
        rendered = _render_resource_tool_result(render)
    elif render_type == "text":
        text = render.get("text") or fallback_output
        if text not in (None, ""):
            st.markdown(str(text))
            rendered = True
        else:
            rendered = False
    else:
        structured = _get_render_structured_content(render)
        if structured not in (None, "", [], {}):
            render_tool_result_payload(structured, use_expander=False)
            rendered = True
        else:
            rendered = False

    with st.expander("Raw render metadata", expanded=False):
        render_tool_result_payload(render, use_expander=False)

    return rendered
```

- [x] **Step 2: Feed render metadata into trace items**

In `_build_tool_trace_items_from_artifacts()`, add `render` to each returned tool item:

```python
                "render": artifact.get("render"),
```

Place it next to `"result": artifact.get("output")`.

- [x] **Step 3: Render rich results in the trace panel**

In `_render_trace_tool_card()`, replace:

```python
    result = tool_item.get("result")
    if result not in (None, ""):
        _render_trace_preview_block(
            "Result Preview",
            result,
            f"Full output for {tool_name}",
        )
```

with:

```python
    result = tool_item.get("result")
    render = tool_item.get("render")
    if isinstance(render, dict):
        st.markdown(
            '<div class="trace-preview-label">Result Preview</div>',
            unsafe_allow_html=True,
        )
        if not render_tool_render_payload(render, fallback_output=result):
            _render_trace_preview_block(
                "Result Preview",
                result,
                f"Full output for {tool_name}",
            )
    elif result not in (None, ""):
        _render_trace_preview_block(
            "Result Preview",
            result,
            f"Full output for {tool_name}",
        )
```

Keep the existing running-tool branch unchanged.

- [x] **Step 4: Render rich results in the older tool artifact expander**

In `render_tool_artifacts()`, replace the current output block:

```python
            output = artifact.get("output")
            if output is not None:
                st.markdown("**Output:**")
                if isinstance(output, (dict, list)):
                    render_json_output(output, label=f"{tool_name} Output", expanded=False)
                else:
                    st.code(str(output), language="text")
```

with:

```python
            output = artifact.get("output")
            render = artifact.get("render")
            if output is not None or isinstance(render, dict):
                st.markdown("**Output:**")
                if isinstance(render, dict) and render_tool_render_payload(
                    render,
                    fallback_output=output,
                ):
                    pass
                elif isinstance(output, (dict, list)):
                    render_json_output(output, label=f"{tool_name} Output", expanded=False)
                elif output is not None:
                    st.code(str(output), language="text")
```

- [x] **Step 5: Keep fallback behavior for direct MCP tool execution**

In `render_tools_tab()`, leave the existing `render_tool_result_payload(payload)` call for direct `/mcp/tools/{tool}/execute` results. That endpoint does not yet return `tool_artifacts[].render`; it should continue to display raw payloads for tool debugging.

- [x] **Step 6: Run syntax verification**

Run:

```powershell
python -m py_compile demo.py
```

Expected:

```text
exit code 0 with no output
```

- [x] **Step 7: Run the Streamlit demo for visual verification**

Run:

```powershell
streamlit run demo.py
```

Expected:

```text
Local URL: http://localhost:8501
```

Manual checks:

- A persisted tool artifact with `render.type == "table"` displays as a dataframe.
- A persisted tool artifact with `render.type == "chart"` displays as a Streamlit chart plus chart-data expander.
- A persisted tool artifact with `render.type == "mcp_app"` displays the template URI and structured app data, not only raw JSON.
- Raw render metadata remains available in an expander.
- Existing `live_widgets` cards still render separately through `render_live_widgets()`.

---

### Task 10: Add Backend Contract Documentation

**Files:**

- Modify: `README.md`

- [x] **Step 1: Add backend contract section**

Add this section under the existing MCP or AI SDK documentation:

```markdown
### Tool result rendering contract

The backend preserves rich tool render metadata in two places:

- Persisted assistant message metadata: `messageMetadata.tool_artifacts[].render`
- AI SDK streams: `tool-output-available.render`

The model-facing tool message remains compact text. Frontends should render from
`render` when present and fall back to `output` when it is absent.

Supported backend render types:

- `mcp_app`: MCP/App result with a UI template URI such as `_meta["openai/outputTemplate"]`
- `live_widget`: in-repo live widget created by the `widgets` MCP server
- `chart`: structured chart payload
- `table`: structured table payload
- `image`: image content block
- `resource`: MCP resource without an app template
- `json`: structured payload without a richer type
- `text`: plain text payload
- `error`: failed tool result

The frontend owns component rendering. Unknown render types must fall back to JSON or text.
```

- [x] **Step 2: Confirm docs mention no frontend implementation**

Run:

```powershell
rg -n "Tool result rendering contract|tool-output-available.render|messageMetadata.tool_artifacts\\[\\]\\.render" README.md
```

Expected:

```text
README.md:<line>:### Tool result rendering contract
README.md:<line>:- Persisted assistant message metadata: `messageMetadata.tool_artifacts[].render`
README.md:<line>:- AI SDK streams: `tool-output-available.render`
```

---

### Task 11: Regression Verification

**Files:**

- No code changes in this task.

- [x] **Step 1: Run focused backend test set**

Run:

```powershell
pytest tests/test_tool_result_rendering.py tests/test_tool_execution_rendering.py tests/test_widget_runtime.py tests/client_backend/test_sse_keepalive.py -q
```

Expected:

```text
all selected tests pass
```

- [x] **Step 2: Run MCP and tool search regression tests**

Run:

```powershell
pytest tests/test_tool_execution_recovery.py tests/test_unified_tool_search.py tests/test_tool_search_scoring.py tests/test_tool_search_prompt_guidance.py -q
```

Expected:

```text
all selected tests pass
```

- [x] **Step 3: Run full test suite if environment dependencies are available**

Run:

```powershell
pytest -q
```

Expected:

```text
all tests pass
```

If optional services such as Qdrant, Redis, or provider SDK credentials are missing, record the exact skipped or failed dependency in the final implementation note and include the focused test results from Steps 1 and 2.

- [x] **Step 4: Run demo syntax verification**

Run:

```powershell
python -m py_compile demo.py
```

Expected:

```text
exit code 0 with no output
```

---

## Frontend Handoff Contract

After this backend plan is implemented, the frontend should read:

- live stream: `tool-output-available.render`
- message history: `message.messageMetadata.tool_artifacts[].render`

Recommended frontend renderer dispatch:

```ts
switch (render.type) {
  case "mcp_app":
    return <McpAppFrame templateUri={render.template_uri} render={render} />;
  case "live_widget":
    return <LiveWidgetCard artifact={artifact} render={render} />;
  case "chart":
    return <ChartRenderer data={render.structured_content} />;
  case "table":
    return <TableRenderer data={render.structured_content} />;
  case "image":
    return <ImageRenderer content={render.content} resources={render.resources} />;
  case "resource":
    return <ResourceList resources={render.resources} />;
  case "text":
    return <TextBlock>{render.text}</TextBlock>;
  case "error":
    return <ToolError render={render} />;
  default:
    return <JsonViewer value={render.structured_content ?? artifact.output} />;
}
```

Backend implementation must not depend on this frontend code. This snippet documents how the contract is intended to be consumed.

## Demo Preview Contract

The Streamlit demo is a local verification surface for the backend contract. It should consume the same persisted `tool_artifacts[].render` payload the external frontend will consume, but it should not try to become a full MCP Apps host. For `mcp_app`, the demo shows the template URI and structured app data; the production frontend can later mount the actual iframe/app bridge.

## Implementation Notes (2026-04-22)

### Status
All 11 tasks completed. All plan-authored tests pass. Pre-existing test failures on the branch were NOT introduced by this work (confirmed by re-running them against a clean git stash of this branch).

### Focused test results (Task 11 Step 1 set)
- `tests/test_tool_result_rendering.py` — 7 passed
- `tests/test_tool_execution_rendering.py` — 4 passed
- `tests/test_widget_runtime.py` — 66 passed
- `tests/client_backend/test_sse_keepalive.py` — 5 passed, 1 pre-existing failure (`test_ai_sdk_stream_emits_heartbeats_during_slow_source` — references removed module attribute `_AI_SDK_HEARTBEAT_INTERVAL_SECONDS`)

### Regression test set (Task 11 Step 2)
- `tests/test_tool_execution_recovery.py` — passed (test updated to include new `render` key in outputs — see Design Decisions)
- `tests/test_unified_tool_search.py` — passed
- `tests/test_tool_search_scoring.py` — 1 pre-existing failure (`test_build_query_tokens_filters_stopwords`, from the recent text-normalization refactor)
- `tests/test_tool_search_prompt_guidance.py` — passed

### Full suite (Task 11 Step 3)
`pytest -q` → 251 passed, 7 failed. All 7 failures confirmed pre-existing on clean branch state (pre-existing bugs in checkpoint serializer, client tool isolation / deferred snapshot, graph streaming summarization, heartbeat constant, and token filter).

### Design Decisions

1. **`outputs[].render` is an additive field.** Existing callers that read `content` continue to work unchanged. Only consumers that opt in see `render`. Tests asserting exact-dict equality on `outputs` were updated to reference `artifacts[0]["render"]` (one case in `test_tool_execution_recovery.py`) rather than invent a stub, so the assertion documents the real contract.

2. **Render type inference is purely structural.** `_infer_render_type` runs through: error → template → widget-name → image-block → chart-keys → table-keys → resources → structured → text. No external MCP tool name is hardcoded. The `live_widget` branch stays scoped to the two in-repo widget tools (`widget_create`, `widget_update`).

3. **`_WIDGET_TOOLS` duplicated between `tool_execution.py` (`_WIDGET_ARTIFACT_TOOLS`) and `tool_result_rendering.py`.** Kept separate intentionally: `tool_execution` handles artifact compaction (size concern), `tool_result_rendering` handles render-type inference (UI concern). The two concerns may diverge later; keeping them separate avoids coupling.

4. **`_lookup_tool_render_payload` is `@staticmethod`.** The test instantiates `MultiAgentWorkflow.__new__(...)` without running `__init__`, so the helper cannot reference instance state. Staticmethod also matches the usage — it only reads from the state dict passed in.

5. **Render lookup falls back from `last_state_values` to `node_state`.** Some graph nodes yield `ToolMessage`s before `last_state_values` is populated for that node. Checking `node_state` covers that window without leaking into other node boundaries.

6. **`build_bot_metadata` already preserves `render`.** The existing `list(response.tool_artifacts)` copies dicts by reference — unknown keys pass through. Task 8 only needed a test to lock that behavior in; no code change required. Noted because the plan said "fix only `build_bot_metadata()`" if the test fails — it didn't fail.

7. **Error paths normalize via `normalize_tool_result_for_rendering(error_msg, tool_name=..., error=...)`.** This produces a consistent `render.type == "error"` and keeps model-facing content as `"Error: <reason>"`. Same shape in all four error branches (missing tool-name, missing tool, device binding, reconnect failure, generic exception).

8. **Streamlit demo Step 7 (manual visual verification) skipped.** Cannot launch interactive Streamlit in this environment. `python -m py_compile demo.py` succeeds, confirming no syntax regressions.

### Dependencies Installed
The C:\Python314 environment was missing several modules required to import `app.ai.graph` and `app.api.ai_sdk`. Installed: `qdrant-client`, `sentence_transformers`, `langchain_openai`, `langchain`, `langchain_community`, `celery`. These are pre-existing deps of the project, not new ones introduced by this work.

### Files Changed
- Created: `app/ai/tool_result_rendering.py`, `tests/test_tool_result_rendering.py`, `tests/test_tool_execution_rendering.py`
- Modified: `app/ai/tool_execution.py`, `app/ai/graph.py`, `app/services/stream_events.py`, `app/services/ai_service.py`, `app/api/ai_sdk.py`, `demo.py`, `README.md`, `tests/test_widget_runtime.py`, `tests/test_tool_execution_recovery.py`, `tests/client_backend/test_sse_keepalive.py`

---

## Implementation Notes (2026-04-23) — Post-Review Fixes

A second pass audit surfaced five real bugs in the initial implementation; all five are fixed and covered by new tests.

### Fix 1 (HIGH) — MCP content flattened before normalization
`extract_content_from_result(result)` was being called before `normalize_tool_result_for_rendering`. For MCP results with mixed content blocks (e.g. `[text_block, image_block]`), that helper unwrapped text blocks to bare strings, producing a mixed list that the normalizer could not recognize as content blocks — the image result degraded to `render.type == "json"` with no `render.content`.

- Removed both `extract_content_from_result(result)` calls in `app/ai/tool_execution.py` (main success path and reconnect-retry success path).
- Removed the now-unused import.
- Enhanced `app/ai/tool_result_rendering.py` with a `_is_content_block_dict()` check so a bare `{type: "text", text: "x"}` dict is also treated as a single-block content list (previously only handled by the removed helper).

### Fix 2 (HIGH) — AI SDK stream corrupted render payloads
`_clean_tool_output` is designed for model/tool string output; applying it to `render` collapsed `{type: "text", ...}` to a bare string and stripped content arrays down to single strings. That broke the frontend contract.

- `app/api/ai_sdk.py`: render now passes through directly to SSE — no `_clean_tool_output` wrapping. Render is already JSON-safe (piped through `make_json_safe` upstream).

### Fix 3 (MEDIUM) — Error message bugs
Three related issues:

1. Callers passed already-prefixed `error="Error: Tool X not found"` to the normalizer, which re-prefixed → `"Error: Error: Tool X not found"`.
2. `render["error"]` stored the same prefixed string.
3. Reconnect failure path passed `error=str(session_exc)` (low-level), discarding the actionable `error_msg`.

Fixes in `app/ai/tool_result_rendering.py`:
- New `_clean_error_text()` strips a leading `Error:` before use; called from both `_build_model_content` and the `render["error"]` assignment.
- `_build_model_content` is now idempotent with respect to the `Error:` prefix.

Fix in `app/ai/tool_execution.py`:
- Reconnect-failure branch now passes the actionable reason as `error=reason`, preserving "MCP session lost for tool X. Reconnection failed. Please try again." as the model- and user-facing text.

### Fix 4 (MEDIUM) — RAG non-search path dropped render
In `app/ai/graph.py` (`_execute_rag_tools`-style loop), `outputs` from `execute_tool_calls` carried `render`, but only `output["content"]` was stashed in `non_search_outputs_by_id` and rebuilt without `render`. Live `tool_end` events for the RAG path and `context["tool_render_results"]` got nothing.

- `non_search_outputs_by_id` now stores the full output dict.
- Rejection feedback entries are wrapped in `{"content": feedback}` for uniformity.
- The rebuilt `tool_outputs` entry copies `render` through when present.

### Fix 5 (MEDIUM) — Unbounded render payload size
A base64 image MCP result could drop multi-megabyte `render.content[*].data` into every checkpoint, every persisted message, and every SSE event.

Added two size guards in `app/ai/tool_result_rendering.py`:
- `_redact_large_inline_data(blocks)`: redacts `data` / `base64` fields in content blocks over 64 KB, leaving `_truncated: true` and `_original_size: N` markers.
- `_cap_structured_content(value)`: replaces structured_content over 128 KB with a truncation marker.

Both are transparent for typical small payloads and integrated into the main `normalize_tool_result_for_rendering` flow.

### New tests
- `test_error_normalization_does_not_double_prefix`
- `test_mixed_text_and_image_content_blocks_produce_image_render`
- `test_bare_text_content_block_dict_is_unwrapped_to_text`
- `test_large_inline_image_data_is_redacted`
- `test_execute_tool_calls_preserves_mcp_image_content_blocks`
- `test_execute_tool_calls_missing_tool_does_not_double_prefix_error`
- `test_ai_sdk_render_payload_preserves_structure_and_text_type`

### Verification
- Focused rendering suite: 78 passed.
- Full server suite: 209 passed, 6 failed (all 6 are the pre-existing failures documented in the 2026-04-22 note).
- `client_backend` suite: 56 passed, 1 pre-existing failure (`test_ai_sdk_stream_emits_heartbeats_during_slow_source`).

### Files changed in this pass
- `app/ai/tool_result_rendering.py` (normalizer)
- `app/ai/tool_execution.py` (success paths + error-prefix cleanup + reconnect reason)
- `app/api/ai_sdk.py` (render passthrough)
- `app/ai/graph.py` (RAG non-search path)
- `tests/test_tool_result_rendering.py` (4 new tests)
- `tests/test_tool_execution_rendering.py` (2 new tests + `_MixedContentTool`/`_MissingNameTool` fixtures)
- `tests/client_backend/test_sse_keepalive.py` (1 new test)

---

## Self-Review Checklist

- Every backend path keeps compact model text separate from UI metadata.
- Existing `output` remains present for old clients.
- Existing `live_widgets` extraction still works.
- AI SDK stream clients get `render` during live tool execution.
- Message history clients get `render` after completion.
- `demo.py` renders common `render.type` values and keeps raw metadata available.
- `demo.py` does not hardcode Canva or any external MCP tool name.
- No external MCP tool is hardcoded by name.
- Unknown results still render as `json` or `text`.
- Error results include `render.type == "error"`.
- Tests cover OpenAI Apps SDK `_meta["openai/outputTemplate"]`.
- Tests cover snake-case `structured_content` and nested `_meta.ui.resourceUri`.
