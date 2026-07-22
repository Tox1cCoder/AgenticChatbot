# Canvas Agent Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Canvas follow-up turns edit the latest durable artifact while preventing an edit from mutating the widget channel.

**Architecture:** Persisted assistant message metadata remains the append-only canvas revision store. A focused snapshot reader supplies bounded routing context and full source only to Canvas. Canvas edit mode applies a centralized mutation-tool deny policy at binding, discovery, and execution boundaries.

**Tech Stack:** Python 3.11, FastAPI service layer, SQLAlchemy/PostgreSQL JSONB, LangChain/LangGraph, pytest/pytest-asyncio.

---

## File Map

- Create `app/ai/canvas_state.py`: validated canvas snapshot, descriptor, revision metadata, and edit-channel tool policy.
- Modify `app/repositories/message.py`: query canvas-bearing assistant rows and the latest assistant row.
- Modify `app/ai/history.py`: expose the latest valid conversation-scoped canvas snapshot.
- Modify `app/ai/graph.py`: hydrate bounded canvas context, apply canvas continuity, and pass full source to Canvas.
- Modify `app/ai/agents/router.py`: add a compact active-canvas hint without keyword routing.
- Modify `app/ai/agents/canvas_agent.py`: inject source, create revisions, and protect valid artifacts from bad edits.
- Modify `app/ai/agents/base_agent.py`: accept per-invocation excluded tools.
- Modify `app/ai/deferred_tool_binding.py` and `app/ai/tool_search_tool.py`: keep denied tools out of binding and discovery.
- Modify `app/ai/workflow/tool_loop.py`: reject stale denied mutation calls before execution.
- Modify `app/ai/schemas.py`: type bounded canvas routing/edit context.
- Modify `app/core/rich_response.py` and `app/core/response_constants.py`: expose optional revision/update semantics.
- Modify focused tests in `tests/test_canvas_agent.py`, `tests/test_history_provider.py`, `tests/test_router.py`, `tests/test_custom_agent_stickiness.py`, `tests/test_widget_runtime.py`, and `tests/test_rich_response_metadata.py`.
- Modify `plans/AI_SDK_FE_CONTRACT.md` and `README.md`: document additive canvas revision fields.

### Task 1: Durable Canvas Snapshot

**Files:**
- Create: `app/ai/canvas_state.py`
- Modify: `app/repositories/message.py`
- Modify: `app/ai/history.py`
- Test: `tests/test_history_provider.py`

- [ ] **Step 1: Write failing snapshot tests**

Add tests that construct assistant rows with canvas metadata and assert that the provider returns the newest valid same-conversation snapshot, normalizes legacy records to `canvas:main` revision `1`, skips malformed/deleted records, reports whether it is the latest assistant response, and can retrieve it independently of normal history trimming.

```python
snapshot = asyncio.run(
    provider.get_latest_canvas_artifact(conversation_id=conversation_id, user_id=user_id)
)
assert snapshot.artifact_id == "canvas:main"
assert snapshot.revision == 1
assert snapshot.content == "<html>latest</html>"
assert snapshot.is_latest_assistant is True
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_history_provider.py -q`

Expected: FAIL because `get_latest_canvas_artifact` and repository candidate queries do not exist.

- [ ] **Step 3: Implement snapshot parsing and lookup**

Create an immutable `CanvasArtifactSnapshot` with `descriptor()` and `to_artifact()` helpers. Add repository methods that query non-deleted assistant rows ordered by sequence, and let `ConversationHistoryProvider` choose the newest valid canvas candidate and compare it with the latest assistant row.

```python
@dataclass(frozen=True)
class CanvasArtifactSnapshot:
    artifact_id: str
    revision: int
    content: str
    language: str
    title: str
    message_id: str
    sequence: int
    is_latest_assistant: bool = False
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_history_provider.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/ai/canvas_state.py app/repositories/message.py app/ai/history.py tests/test_history_provider.py
git commit -m "feat: recover durable canvas snapshots"
```

### Task 2: Canvas Routing Continuity

**Files:**
- Modify: `app/ai/graph.py`
- Modify: `app/ai/agents/router.py`
- Modify: `app/ai/schemas.py`
- Test: `tests/test_custom_agent_stickiness.py`
- Test: `tests/test_router.py`

- [ ] **Step 1: Write failing routing tests**

Cover immediate canvas continuity without invoking the router, stale canvas state falling back to normal routing, planning precedence, and compact active-canvas context in the semantic router prompt.

```python
out = await workflow._route_node(state)
assert out["selected_agent"] == "canvas_agent"
assert router.calls == 0
assert out["context"]["canvas_edit_mode"] is True
```

- [ ] **Step 2: Run the routing tests and verify RED**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_custom_agent_stickiness.py tests/test_router.py -q`

Expected: FAIL because routing does not hydrate canvas state or support canvas continuity.

- [ ] **Step 3: Implement bounded routing context**

Add `_get_active_canvas_snapshot()` with defensive logging. Store only `snapshot.descriptor()` and `canvas_edit_mode` in graph context. Apply canvas continuity after planning precedence and before the custom-agent/router paths. Extend Router with an optional bounded descriptor and append one short context line.

```python
if active_canvas and active_canvas.is_latest_assistant:
    state["selected_agent"] = "canvas_agent"
    context["canvas_edit_mode"] = True
    self._record_agent_invocation(state, "canvas_agent", via="canvas_continuity")
    return state
```

- [ ] **Step 4: Run routing tests and verify GREEN**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_custom_agent_stickiness.py tests/test_router.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/ai/graph.py app/ai/agents/router.py app/ai/schemas.py tests/test_custom_agent_stickiness.py tests/test_router.py
git commit -m "fix: keep canvas follow-ups on active artifact"
```

### Task 3: Source-Aware Canvas Revisions

**Files:**
- Modify: `app/ai/agents/canvas_agent.py`
- Modify: `app/ai/graph.py`
- Test: `tests/test_canvas_agent.py`

- [ ] **Step 1: Write failing agent tests**

Patch the base invocation to capture model-facing messages and return controlled responses. Assert full source injection, stable identity, revision increments, create/update status, identical-output no-op, and no replacement artifact for missing or truncated edits.

```python
assert "<body>ORIGINAL</body>" in captured_messages[-2].content
assert response.metadata["canvas_artifact"]["revision"] == 2
assert response.metadata["canvas_update"]["status"] == "updated"
```

- [ ] **Step 2: Run the canvas tests and verify RED**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_canvas_agent.py -q`

Expected: FAIL because previous source is not injected and revision/failure metadata is absent.

- [ ] **Step 3: Implement source injection and protected revisioning**

Accept `previous_artifact: CanvasArtifactSnapshot | None`, insert a delimited untrusted-source message before the current user request, and classify response output:

```python
if previous_artifact and (artifact is None or artifact.get("truncated")):
    response.metadata["canvas_update"] = previous_artifact.update_status(
        "failed", reason="missing_artifact" if artifact is None else "truncated_output"
    )
    return response
```

Valid changed output becomes the next `canvas:main` revision. Identical output reports `unchanged` without publishing a replacement artifact. First-time truncated generation preserves the legacy warning behavior.

- [ ] **Step 4: Pass the snapshot from the canvas node**

Fetch the full snapshot in `_canvas_node` and pass it to `CanvasAgent.invoke_model_with_history`. Keep source out of graph context and other agent prompts.

- [ ] **Step 5: Run canvas and graph tests and verify GREEN**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_canvas_agent.py tests/test_message_history_pipeline.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add app/ai/agents/canvas_agent.py app/ai/graph.py tests/test_canvas_agent.py tests/test_message_history_pipeline.py
git commit -m "fix: edit the persisted canvas revision"
```

### Task 4: Turn-Scoped Widget Mutation Guard

**Files:**
- Modify: `app/ai/canvas_state.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/deferred_tool_binding.py`
- Modify: `app/ai/tool_search_tool.py`
- Modify: `app/ai/workflow/tool_loop.py`
- Test: `tests/test_widget_runtime.py`
- Test: `tests/test_tool_search_accuracy.py`

- [ ] **Step 1: Write failing channel-policy tests**

Assert that Canvas can discover widget mutations outside edit mode, edit-mode binding omits `widget_create`/`widget_update`, tool search filters them before autoload, stale deferred calls are rejected before tool execution, and read-only widget tools remain available.

```python
assert "widget_create" not in edit_tool_names
assert "widget_update" not in edit_tool_names
assert "widget_get_state" in edit_tool_names
```

- [ ] **Step 2: Run focused policy tests and verify RED**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_widget_runtime.py tests/test_tool_search_accuracy.py -q`

Expected: FAIL because exclusion is currently static, incomplete under deferred discovery, and not turn-scoped.

- [ ] **Step 3: Centralize mutation capability policy**

Define `CANVAS_EDIT_DENIED_TOOL_NAMES = frozenset({"widget_create", "widget_update"})` in `canvas_state.py`. Remove Canvas from the permanent widget-exclusion set while leaving widget pins restricted to Chat/RAG/Search.

- [ ] **Step 4: Apply exclusions to binding and discovery**

Thread optional `excluded_tool_names` through BaseAgent binding and `build_deferred_tool_list`. Add a denylist to the scoped `tool_search` closure and filter public/internal results before autoload.

```python
filtered = [
    item for item in results
    if item.get("tool_name") not in excluded and item.get("call_name") not in excluded
]
```

- [ ] **Step 5: Add the execution backstop**

In the generic tool loop, partition calls denied by the active canvas-edit policy, materialize rejected ToolMessages/artifacts for them, and execute only allowed calls. Never invoke or recover a denied tool.

- [ ] **Step 6: Run focused policy tests and verify GREEN**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_widget_runtime.py tests/test_tool_search_accuracy.py tests/test_tool_execution_policy.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add app/ai/canvas_state.py app/ai/agents/base_agent.py app/ai/deferred_tool_binding.py app/ai/tool_search_tool.py app/ai/workflow/tool_loop.py tests/test_widget_runtime.py tests/test_tool_search_accuracy.py
git commit -m "fix: separate canvas edits from widget mutations"
```

### Task 5: Public Canvas Revision Contract

**Files:**
- Modify: `app/core/rich_response.py`
- Modify: `app/core/response_constants.py`
- Test: `tests/test_rich_response_metadata.py`
- Modify: `plans/AI_SDK_FE_CONTRACT.md`
- Modify: `README.md`

- [ ] **Step 1: Write failing metadata tests**

Assert that optional `revision` and `operation` flow from legacy canvas metadata into the strict AI SDK rich payload while legacy records without them remain valid.

```python
assert canvas["id"] == "canvas:main"
assert canvas["payload"]["revision"] == 2
assert canvas["payload"]["operation"] == "update"
```

- [ ] **Step 2: Run metadata tests and verify RED**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_rich_response_metadata.py -q`

Expected: FAIL because strict `CanvasPayload` drops/rejects revision fields.

- [ ] **Step 3: Implement additive payload fields and documentation**

Add optional validated revision/operation fields, project them in `_canvas_rich_item_from_artifact`, and update the two contract documents. Keep `canvas:main`, content, language, and title unchanged.

- [ ] **Step 4: Run metadata tests and verify GREEN**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_rich_response_metadata.py tests/test_ai_sdk_v6_stream_contract.py tests/test_ai_sdk_context_window.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/core/rich_response.py app/core/response_constants.py tests/test_rich_response_metadata.py plans/AI_SDK_FE_CONTRACT.md README.md
git commit -m "feat: publish canvas revision metadata"
```

### Task 6: Full Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run focused canvas regression suite**

Run:

```powershell
C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest tests/test_canvas_agent.py tests/test_history_provider.py tests/test_router.py tests/test_custom_agent_stickiness.py tests/test_widget_runtime.py tests/test_tool_search_accuracy.py tests/test_rich_response_metadata.py -q
```

Expected: PASS with zero failures.

- [ ] **Step 2: Run formatting and lint checks**

Run:

```powershell
C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m ruff check app/ai/canvas_state.py app/ai/history.py app/ai/graph.py app/ai/agents/canvas_agent.py app/ai/agents/router.py app/ai/agents/base_agent.py app/ai/deferred_tool_binding.py app/ai/tool_search_tool.py app/ai/workflow/tool_loop.py app/repositories/message.py app/core/rich_response.py app/core/response_constants.py tests/test_canvas_agent.py tests/test_history_provider.py tests/test_router.py tests/test_custom_agent_stickiness.py tests/test_widget_runtime.py tests/test_tool_search_accuracy.py tests/test_rich_response_metadata.py
C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m ruff format --check app/ai/canvas_state.py app/ai/history.py app/ai/graph.py app/ai/agents/canvas_agent.py app/ai/agents/router.py app/ai/agents/base_agent.py app/ai/deferred_tool_binding.py app/ai/tool_search_tool.py app/ai/workflow/tool_loop.py app/repositories/message.py app/core/rich_response.py app/core/response_constants.py tests/test_canvas_agent.py tests/test_history_provider.py tests/test_router.py tests/test_custom_agent_stickiness.py tests/test_widget_runtime.py tests/test_tool_search_accuracy.py tests/test_rich_response_metadata.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Run the full test suite**

Run: `C:\Users\ADMIN\miniconda3\envs\agents\python.exe -m pytest -q`

Expected: PASS with zero failures.

- [ ] **Step 4: Inspect the final diff and commit verification adjustments**

```powershell
git diff --check
git status --short
```

If verification required any code adjustment, rerun the affected checks and commit only those changes with `git commit -m "test: verify canvas continuity"`.
