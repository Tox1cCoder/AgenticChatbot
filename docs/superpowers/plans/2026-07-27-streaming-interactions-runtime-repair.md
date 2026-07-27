# Streaming, Interactions, and Runtime Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `widget_type` from live interactions, fix their Streamlit renderer, retain terminal execution traces, eliminate cross-context stream cleanup errors, and verify protected image delivery after restarting stale services.

**Architecture:** A live interaction is defined by validated HTML state rather than a renderer discriminator. Canonical stream events remain shared by Streamlit and AI SDK; terminal message metadata carries trace history, while usage context wraps each workflow-source advance and close rather than an outward yield. Generated images continue through the authenticated sidecar proxy already present in the checkout.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, Redis, FastMCP/LangChain, Streamlit, Vercel AI SDK UI Message Stream, pytest, pytest-asyncio, Ruff.

---

## File Map

- `app/ai/mcp_servers/widgets_server.py`: typeless `widget_create` schema.
- `app/services/widget_contract.py`: HTML-state validation only.
- `app/services/widget_runtime.py`: typeless records and stores; ignore legacy Redis keys.
- `app/api/widgets.py`: typeless recovery, connection, action, and WebSocket payloads.
- `app/ai/tool_execution.py`, `app/core/response_constants.py`, `app/core/rich_response.py`, `app/core/rich_placement.py`: typeless tool/rich/persisted metadata.
- `demo.py`: single HTML renderer and terminal trace reconciliation.
- `app/services/ai_service.py`: context-bound workflow-source iteration.
- `tests/test_widget_runtime.py`, `tests/test_widget_contract.py`, `tests/test_widgets_api.py`: runtime/API contract.
- `tests/test_demo_meaningful_widgets.py`, `tests/test_demo_rich_response.py`, `tests/test_demo_live_ui_state.py`: renderer and terminal UI.
- `tests/test_message_service_event_streaming.py`: trace persistence.
- `tests/test_model_usage_workflow_instrumentation.py`, `tests/test_internal_sse_stream_contract.py`, `tests/test_ai_sdk_v6_stream_contract.py`: context lifecycle through both transports.
- Current rich-response tests and active frontend contract docs: remove the field from live-interaction fixtures and examples while retaining explicit legacy-read coverage.

---

### Task 1: Define the Typeless Interaction Contract in RED Tests

**Files:**
- Modify: `tests/test_widget_runtime.py`
- Modify: `tests/test_widget_contract.py`
- Modify: `tests/test_widgets_api.py`

- [ ] **Step 1: Add a failing MCP schema test**

```python
def test_widget_create_schema_has_no_widget_type():
    from app.ai.mcp_servers.widgets_server import mcp

    tools = asyncio.run(mcp.list_tools())
    create_tool = next(tool for tool in tools if tool.name == "widget_create")

    assert "widget_type" not in create_tool.inputSchema["properties"]
    assert "widget_type" not in create_tool.inputSchema.get("required", [])
    assert create_tool.inputSchema["required"] == ["session_id", "initial_state"]
```

- [ ] **Step 2: Add a failing tool creation test**

```python
@pytest.mark.asyncio
async def test_widget_create_accepts_html_state_without_a_type(monkeypatch):
    import app.services.widget_runtime as widget_runtime
    from app.ai.mcp_servers import widgets_server

    store = InMemoryWidgetStore()
    monkeypatch.setattr(widget_runtime, "_widget_store", store)
    result = await widgets_server.widget_create(
        session_id="conv-1",
        initial_state=json.dumps(_VALID_HTML_STATE),
        title="Orbit simulator",
    )

    payload = json.loads(result)
    assert "widget_type" not in payload
    assert payload["status"] == "active"
    assert (await store.list_by_session("conv-1"))[0].state == _VALID_HTML_STATE
```

- [ ] **Step 3: Add failing record and legacy-read tests**

```python
@pytest.mark.asyncio
async def test_widget_record_metadata_omit_widget_type():
    record = await InMemoryWidgetStore().create(
        session_id="conv-1",
        initial_state=_VALID_HTML_STATE,
        title="Demo",
    )
    assert "widget_type" not in record.to_dict()
    assert "widget_type" not in record.to_live_widget_metadata()


def test_redis_reader_ignores_legacy_widget_type():
    record = RedisWidgetStore._to_record(
        {
            "widget_id": "legacy",
            "session_id": "conv-1",
            "widget_type": "html",
            "title": "Legacy",
            "state": json.dumps(_VALID_HTML_STATE),
            "status": "active",
            "version": "1",
            "created_at": "1",
            "updated_at": "1",
            "expires_at": "9999999999",
        }
    )
    assert record.state == _VALID_HTML_STATE
    assert "widget_type" not in record.to_dict()
```

- [ ] **Step 4: Make the connection response assertion typeless**

In `test_widget_connection_mints_token_and_ws_url`, create the record with
`session_id`, valid HTML state, and title, then assert:

```python
assert response.status_code == 200
assert "widget_type" not in response.json()
assert response.json()["widget_id"] == created.widget_id
```

- [ ] **Step 5: Run RED**

```powershell
python -m pytest -q `
  tests/test_widget_runtime.py tests/test_widget_contract.py tests/test_widgets_api.py
```

Expected: FAIL because the active schema and stores require and emit `widget_type`.

- [ ] **Step 6: Commit RED tests**

```powershell
git add tests/test_widget_runtime.py tests/test_widget_contract.py tests/test_widgets_api.py
git commit -m "test: define typeless interaction contract"
```

---

### Task 2: Remove the Type Discriminator from Tool and Runtime

**Files:**
- Modify: `app/ai/mcp_servers/widgets_server.py`
- Modify: `app/services/widget_contract.py`
- Modify: `app/services/widget_runtime.py`
- Test: `tests/test_widget_runtime.py`
- Test: `tests/test_widget_contract.py`
- Test: `tests/test_widgets_api.py`

- [ ] **Step 1: Delete `SUPPORTED_WIDGET_TYPE` and `assert_supported_widget_type`**

Retain `validate_html_widget_state` as the sole content contract:

```python
def validate_html_widget_state(state: Any) -> None:
    if not isinstance(state, dict):
        raise ValueError("widget state must be a JSON object")
    html_content = state.get("html")
    if not isinstance(html_content, str) or not html_content.strip():
        raise ValueError("widget state must contain non-empty html content")
    height = state.get("height")
    if isinstance(height, bool) or not isinstance(height, (int, float)):
        raise ValueError("widget height must be numeric")
    if not MIN_HTML_WIDGET_HEIGHT <= float(height) <= MAX_HTML_WIDGET_HEIGHT:
        raise ValueError(
            f"widget height must be between {MIN_HTML_WIDGET_HEIGHT} and "
            f"{MAX_HTML_WIDGET_HEIGHT}"
        )
    if state.get("caption") is not None and not isinstance(state["caption"], str):
        raise ValueError("widget caption must be a string when provided")
```

- [ ] **Step 2: Change the MCP tool signature**

```python
@mcp.tool()
async def widget_create(
    session_id: str,
    initial_state: str,
    title: str = "",
) -> str:
    """Create a sandboxed interactive HTML experience in the conversation."""
    store = get_widget_store()
    state = _parse_widget_state(initial_state, field="initial_state")
    validate_html_widget_state(state)
    record = await store.create(
        session_id=session_id,
        initial_state=state,
        title=title or None,
    )
    return json.dumps(record.to_dict(), default=str)
```

Remove the type assertion from `widget_update`; validate the replacement state.

- [ ] **Step 3: Change the record and store interfaces**

`WidgetRecord` becomes:

```python
@dataclass(frozen=True)
class WidgetRecord:
    widget_id: str
    session_id: str
    title: str | None
    state: dict[str, Any]
    status: WidgetStatus
    version: int
    created_at: float
    updated_at: float
    expires_at: float
```

Make `WidgetStore.create/restore`, `InMemoryWidgetStore.create/restore`, and
`RedisWidgetStore.create/restore` accept no type argument. Remove the key from
new mappings and serializers. Both `_to_record` functions ignore any legacy key.

- [ ] **Step 4: Run GREEN**

Run Task 1's test command. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/ai/mcp_servers/widgets_server.py app/services/widget_contract.py `
  app/services/widget_runtime.py tests/test_widget_runtime.py `
  tests/test_widget_contract.py tests/test_widgets_api.py
git commit -m "fix: remove widget type from interaction runtime"
```

---

### Task 3: Remove the Field from API and Persisted/Rich Metadata

**Files:**
- Modify: `app/api/widgets.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/core/response_constants.py`
- Modify: `app/core/rich_response.py`
- Modify: `app/core/rich_placement.py`
- Modify: `tests/test_widgets_api.py`
- Modify: `tests/test_rich_response_contract.py`
- Modify: `tests/test_rich_response_metadata.py`
- Modify: `tests/test_rich_response_sources.py`
- Modify: `tests/test_rich_response_streaming.py`
- Modify: `tests/test_rich_placement.py`
- Modify: `tests/test_message_history_pipeline.py`

- [ ] **Step 1: Add a failing metadata test**

Create a successful `widget_create` artifact whose args contain only
`session_id` and `initial_state`, and whose output contains no type. Assert:

```python
metadata = build_bot_metadata(response)
assert "widget_type" not in metadata["live_widgets"][0]
rich_widget = next(i for i in metadata["rich_items"] if i["type"] == "live_widget")
assert "widget_type" not in rich_widget["payload"]
```

Run the three primary modules and confirm RED:

```powershell
python -m pytest -q `
  tests/test_widgets_api.py tests/test_rich_response_contract.py `
  tests/test_rich_response_metadata.py
```

- [ ] **Step 2: Simplify API recovery**

In `_recover_widget_snapshot_from_messages`, validate recovered state with
`validate_html_widget_state` and return:

```python
{
    "widget_id": widget_id,
    "session_id": session_id,
    "title": title,
    "state": latest_state,
    "status": status,
    "version": version,
}
```

Remove `SUPPORTED_WIDGET_TYPE`, `record.widget_type`, and type gates. Restore the
record without a type. Connection/action/WebSocket responses omit the field.

- [ ] **Step 3: Simplify metadata builders**

`extract_live_widgets_from_artifacts` emits:

```python
{
    "widget_id": widget_id,
    "session_id": parsed.get("session_id", ""),
    "title": parsed.get("title"),
    "status": parsed.get("status", "active"),
    "version": parsed.get("version", 1),
    "connection_endpoint": f"/widgets/{widget_id}/connection",
}
```

Remove `widget_type` from `_widget_rich_item_from_live_widget`,
`LiveWidgetPayload`, live-widget tool candidates, compact outputs, and placement
text. A legacy input key may be present but must never be re-emitted.

- [ ] **Step 4: Update active fixtures and run GREEN**

```powershell
python -m pytest -q `
  tests/test_widgets_api.py tests/test_rich_response_contract.py `
  tests/test_rich_response_metadata.py tests/test_rich_response_sources.py `
  tests/test_rich_response_streaming.py tests/test_rich_placement.py `
  tests/test_message_history_pipeline.py
```

Expected: PASS. Retain one explicit legacy fixture to prove the key is ignored.

- [ ] **Step 5: Commit**

```powershell
git add app/api/widgets.py app/ai/tool_execution.py app/core/response_constants.py `
  app/core/rich_response.py app/core/rich_placement.py tests/test_widgets_api.py `
  tests/test_rich_response_contract.py tests/test_rich_response_metadata.py `
  tests/test_rich_response_sources.py tests/test_rich_response_streaming.py `
  tests/test_rich_placement.py tests/test_message_history_pipeline.py
git commit -m "fix: publish typeless interaction metadata"
```

---

### Task 4: Fix the Streamlit Interaction Renderer

**Files:**
- Modify: `demo.py`
- Modify: `tests/test_demo_meaningful_widgets.py`
- Modify: `tests/test_demo_rich_response.py`

- [ ] **Step 1: Add a failing renderer contract test**

```python
def test_live_widget_component_is_typeless_and_sandboxed():
    output = demo._build_live_widget_component_html(
        {
            "widget_id": "w-1",
            "title": "Orbit simulator",
            "status": "active",
            "version": 1,
        },
        "token",
    )
    assert "widget_type" not in output
    assert "widgetType" not in output
    assert "renderHtmlWidget(state.data)" in output
    assert 'sandbox="allow-scripts allow-forms allow-modals allow-downloads"' in output
    assert 'referrerpolicy="no-referrer"' in output
```

Run and confirm RED:

```powershell
python -m pytest -q `
  tests/test_demo_meaningful_widgets.py tests/test_demo_rich_response.py
```

- [ ] **Step 2: Remove the renderer discriminator**

Remove the field from `_build_live_widget_component_html` configuration. Replace
the JavaScript title/subtitle logic with:

```javascript
el.title.textContent = cfg.widget.title || "Interactive Experience";
el.sub.textContent = cfg.widget.widget_id || "pending";
```

Keep `renderBody()` unconditional on `renderHtmlWidget(state.data)`. In
`render_live_widgets`, use:

```python
title = str(widget.get("title") or f"Interactive Experience {index + 1}")
meta_suffix = f"v{version} · {widget_id[:16]}…"
```

Remove the type from details JSON and type-derived labels. Preserve sandbox,
authentication, reconnection, height clamping, and lazy mounting.

- [ ] **Step 3: Run GREEN and commit**

Run Task 4's command. Expected: PASS.

```powershell
git add demo.py tests/test_demo_meaningful_widgets.py tests/test_demo_rich_response.py
git commit -m "fix: render typeless interactive experiences"
```

---

### Task 5: Retain the Execution Trace after Completion

**Files:**
- Modify: `demo.py`
- Modify: `tests/test_demo_live_ui_state.py`
- Modify: `tests/test_message_service_event_streaming.py`

- [ ] **Step 1: Add a failing terminal reconciliation test**

Seed `stream_trace_items` with thinking plus an errored `widget_create`, call the
new wished-for helper, and assert:

```python
merged = demo._reconcile_terminal_trace({"id": "m-1", "messageMetadata": {}})
metadata = demo.get_message_metadata(merged)
assert metadata["thinking_summary"] == "Checked the interaction tool."
assert metadata["tool_artifacts"][0]["tool"] == "widget_create"
assert metadata["tool_artifacts"][0]["status"] == "error"
```

- [ ] **Step 2: Add a backend persistence regression**

Drive `tool_call_available`, `tool_execution_end`, and `complete` through
`MessageService.create_message_stream`. Assert the response passed to
`_persist_completed_workflow_response` contains the matching artifact, args,
output, error status, and render payload.

- [ ] **Step 3: Run RED**

```powershell
python -m pytest -q `
  tests/test_demo_live_ui_state.py tests/test_message_service_event_streaming.py
```

Expected: the frontend helper test FAILS. If backend persistence is already
GREEN, keep it as a lock and avoid unnecessary backend changes.

- [ ] **Step 4: Implement `_reconcile_terminal_trace`**

The helper must copy the terminal message, preserve its metadata key casing,
fill a missing `thinking_summary`, and merge tool artifacts by
`(tool_call_id, tool)` without overwriting richer persisted artifacts. Map live
trace fields as follows:

```python
artifact = {
    "tool_call_id": item.get("tool_call_id"),
    "tool": item.get("name") or "unknown",
    "args": item.get("args"),
    "output": item.get("result"),
    "error": item.get("error"),
    "status": item.get("state") or "unknown",
    "render": item.get("render"),
}
```

Call the helper in normal and resume `complete` handlers before renderer
finalization. After the normal authoritative reload, merge the exact final
message by ID back into `st.session_state.messages` so a response cannot lose
richer streamed metadata during the immediate rerender.

- [ ] **Step 5: Run GREEN and commit**

Run Task 5's command. Expected: PASS.

```powershell
git add demo.py tests/test_demo_live_ui_state.py tests/test_message_service_event_streaming.py
git commit -m "fix: retain execution trace after stream completion"
```

---

### Task 6: Own Usage Context at the Workflow-Source Boundary

**Files:**
- Modify: `app/services/ai_service.py`
- Modify: `tests/test_model_usage_workflow_instrumentation.py`
- Modify: `tests/test_internal_sse_stream_contract.py`
- Modify: `tests/test_ai_sdk_v6_stream_contract.py`

- [ ] **Step 1: Add a failing cross-task close test**

Create a workflow source that asserts workflow usage context in its body and
finally block. Consume one event, then run `stream.aclose()` in a new task:

```python
assert (await anext(stream)).type == "message_delta"
await asyncio.create_task(stream.aclose())
await asyncio.wait_for(closed.wait(), timeout=0.1)
assert current_usage_context().operation == "unknown"
```

Repeat for `resume_interrupted_execution_stream`.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q `
  tests/test_model_usage_workflow_instrumentation.py `
  -k "close_from_another_task_context"
```

Expected: FAIL with the reported token/different-Context cleanup error.

- [ ] **Step 3: Add the context-bound iterator**

```python
@staticmethod
async def _iterate_in_usage_context(workflow_stream, usage_context: UsageContext):
    iterator = workflow_stream.__aiter__()
    try:
        while True:
            try:
                with bind_usage_context(usage_context):
                    event = await anext(iterator)
            except StopAsyncIteration:
                break
            yield event
    finally:
        aclose = getattr(iterator, "aclose", None)
        if callable(aclose):
            with bind_usage_context(usage_context):
                await aclose()
```

In both streaming service methods, create the workflow source, wrap it with
`_iterate_in_usage_context`, and pass that wrapper to `_map_workflow_stream`.
Remove the outer `with bind_usage_context(...)` that currently spans outward
yields. Do not catch/suppress `ValueError`.

- [ ] **Step 4: Cover both transports**

Add internal SSE and AI SDK tests that consume one event and close early. Capture
the loop exception handler and assert no unretrieved-task or context-reset error.
Retain the existing AI SDK producer-owner tests.

- [ ] **Step 5: Run GREEN and commit**

```powershell
python -m pytest -q `
  tests/test_model_usage_workflow_instrumentation.py `
  tests/test_internal_sse_stream_contract.py `
  tests/test_ai_sdk_v6_stream_contract.py tests/test_message_stream_errors.py
```

Expected: PASS with no async cleanup warnings.

```powershell
git add app/services/ai_service.py tests/test_model_usage_workflow_instrumentation.py `
  tests/test_internal_sse_stream_contract.py tests/test_ai_sdk_v6_stream_contract.py
git commit -m "fix: own usage context at stream source boundary"
```

---

### Task 7: Update Prompts, Active Contracts, and Remaining Fixtures

**Files:**
- Modify: `app/ai/prompts.py`
- Modify: `README.md`
- Modify: `plans/live-widgets-frontend-integration.md`
- Modify: `plans/AI_SDK_FE_CONTRACT.md`
- Modify: `tests/test_widget_docs_html_only.py`
- Modify: `tests/test_rich_response_prompt_inventory.py`
- Modify: other tests returned by `rg -l "widget_type" tests`

- [ ] **Step 1: Change documentation tests to require the typeless example**

The active docs must show:

```json
{
  "session_id": "<conversation-id>",
  "initial_state": "{\"html\":\"<!doctype html>...\",\"height\":620}",
  "title": "Orbit simulator"
}
```

Assert active interaction sections do not instruct clients/models to send
`widget_type`. Historical plans may retain it only when clearly marked legacy.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q `
  tests/test_widget_docs_html_only.py tests/test_rich_response_prompt_inventory.py
```

- [ ] **Step 3: Update prompt and docs**

Replace the model instruction with a direct `initial_state` HTML/height/caption
contract. Document that every new interaction uses the sandboxed HTML renderer
and legacy type keys are ignored.

- [ ] **Step 4: Update remaining active fixtures**

```powershell
$widgetTests = rg -l "widget_type" tests
python -m pytest -q $widgetTests
```

Expected: PASS. Any remaining occurrence is an explicit legacy-read fixture or an
unrelated static widget format.

- [ ] **Step 5: Commit**

```powershell
git add app/ai/prompts.py README.md plans/live-widgets-frontend-integration.md `
  plans/AI_SDK_FE_CONTRACT.md tests/test_widget_docs_html_only.py `
  tests/test_rich_response_prompt_inventory.py
git commit -m "docs: publish typeless interaction contract"
```

If Step 4 changed other active fixture modules, inspect `git status --short` and
add each intended test path explicitly before this commit.

---

### Task 8: Verify Protected Images and Restart Stale Processes

**Files:**
- Verify: `client_backend/api/chat_images.py`
- Verify: `client_backend/main.py`
- Verify: `tests/client_backend/test_image_stream_proxy.py`
- Verify: `tests/test_demo_image_reference_rendering.py`
- Verify: `scripts/verify_image_streaming_contract.py`

- [ ] **Step 1: Run media regressions**

```powershell
python -m pytest -q `
  tests/client_backend/test_image_stream_proxy.py `
  tests/test_demo_image_reference_rendering.py tests/test_chat_images_api.py `
  tests/test_ai_sdk_v6_stream_contract.py
python `
  scripts/verify_image_streaming_contract.py
```

Expected: PASS. Do not add another proxy unless a RED regression proves a code
gap; current live 404 evidence points to the pre-route sidecar process.

- [ ] **Step 2: Resolve exact process targets**

```powershell
$serviceProcesses = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -match 'python\s+-m\s+(app\.main|client_backend\s+run)' -or
  $_.CommandLine -match 'streamlit(\.exe)?"?\s+run\s+demo\.py'
}
$serviceProcesses | Select-Object ProcessId, ExecutablePath, CommandLine
```

Validate exactly the three processes rooted in this workspace, then stop only
those IDs.

- [ ] **Step 3: Restart hidden from this workspace**

```powershell
$python = (Get-Command python).Source
Start-Process $python -ArgumentList '-m','app.main' `
  -WorkingDirectory (Get-Location) -WindowStyle Hidden
Start-Process $python -ArgumentList '-m','client_backend','run' `
  -WorkingDirectory (Get-Location) -WindowStyle Hidden
Start-Process $python -ArgumentList '-m','streamlit','run','demo.py' `
  -WorkingDirectory (Get-Location) -WindowStyle Hidden
```

Poll ports 8000, 8100, and 8501 with a bounded deadline.

- [ ] **Step 4: Probe both sidecar aliases**

```powershell
$probeId = '83b47a74-4c10-4856-b70e-ed0fec918a4e'
foreach ($path in @("/chat-images/$probeId", "/api/chat-images/$probeId")) {
  $response = Invoke-WebRequest -Uri "http://127.0.0.1:8100$path" `
    -SkipHttpErrorCheck -TimeoutSec 5
  [PSCustomObject]@{Path=$path; Status=[int]$response.StatusCode}
}
```

Expected without a local session: 401/403, never router-level 404. Verify one
authenticated image in the UI without printing its token or bytes.

---

### Task 9: Full Verification

**Files:**
- Verify: all modified files

- [ ] **Step 1: Run focused suites**

```powershell
python -m pytest -q `
  tests/test_widget_runtime.py tests/test_widget_contract.py tests/test_widgets_api.py `
  tests/test_demo_meaningful_widgets.py tests/test_demo_rich_response.py `
  tests/test_demo_live_ui_state.py tests/test_message_service_event_streaming.py `
  tests/test_model_usage_workflow_instrumentation.py `
  tests/test_internal_sse_stream_contract.py tests/test_ai_sdk_v6_stream_contract.py `
  tests/client_backend/test_image_stream_proxy.py `
  tests/test_demo_image_reference_rendering.py
```

- [ ] **Step 2: Run Ruff on changed Python files**

```powershell
python -m ruff check `
  app/ai/mcp_servers/widgets_server.py app/services/widget_contract.py `
  app/services/widget_runtime.py app/api/widgets.py app/ai/tool_execution.py `
  app/core/response_constants.py app/core/rich_response.py `
  app/core/rich_placement.py app/services/ai_service.py demo.py tests
```

- [ ] **Step 3: Run the full suite**

```powershell
python -m pytest -q
```

Expected: PASS with no unretrieved-task or context-token warnings.

- [ ] **Step 4: Inspect final state**

```powershell
git status --short
git diff --check
git log -12 --oneline
```

Confirm no secrets, generated image bytes, tokens, unrelated refactors, or
accidental historical-plan rewrites.

- [ ] **Step 5: Perform live acceptance checks**

In Streamlit, create an interactive simulation without a type argument, open the
sandboxed interaction, confirm the trace remains after completion, and generate
an image that remains visible after reload. Through AI SDK, confirm tool
input/output parts, typeless interaction metadata, and the protected final image
reference.
