# Unresolved Rich-Marker Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent invented rich-item IDs from reaching users while keeping rich-item placement dynamic and inventory-driven.

**Architecture:** Remove marker protocol text from the always-on media prompt and retain it only in the dynamic inventory guidance. Reconcile final body markers against the authoritative `rich_items` registry before persistence, then make finalized Streamlit rendering silently omit unresolved legacy markers while preserving active-stream pending placeholders.

**Tech Stack:** Python 3.14, Pydantic, pytest, pytest-asyncio, Streamlit.

## Global Constraints

- Do not hard-code widget IDs, image IDs, subjects, providers, or domains.
- The current turn's bounded rich-item inventory is the only model-facing source of valid IDs.
- Preserve valid markers and marker examples in fenced, indented, and inline code.
- Do not rewrite existing database rows.
- Do not change selected-only image persistence, auto-placement, or remote-image externalization.
- Use `--confcutdir=tests/client_backend` for focused tests in this checkout because the root `tests/conftest.py` declares a loop-factory hook unsupported by the installed pytest-asyncio plugin.

---

### Task 1: Reconcile Final Markers Against the Authoritative Registry

**Files:**
- Modify: `tests/test_rich_response_metadata.py`
- Modify: `app/core/response_constants.py`

**Interfaces:**
- Consumes: `parse_inline_rich_references(markdown: str) -> list[str]`, `remove_inline_rich_reference(markdown: str, item_id: str) -> str`, and finalized `metadata["rich_items"]`.
- Produces: `reconcile_final_rich_references(content: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]`.

- [ ] **Step 1: Write failing reconciliation tests**

Add tests that call the wished-for function through the module so the missing function fails inside the test:

```python
from app.core import response_constants


def test_reconcile_final_rich_references_removes_invented_marker():
    content = "Before\n\n<!--rich:widget:invented-->\n\nAfter"
    metadata = {
        "rich_items_version": 1,
        "rich_items": [],
        "rich_reference_warnings": [
            {"code": "unknown_rich_item", "id": "widget:invented"}
        ],
    }

    cleaned, updated = response_constants.reconcile_final_rich_references(
        content, metadata
    )

    assert cleaned == "Before\n\nAfter"
    assert updated["rich_reference_warnings"] == []


def test_reconcile_preserves_valid_markers_and_code_examples():
    valid_id = "widget:w-1"
    content = (
        f"<!--rich:{valid_id}-->\n\n"
        "`<!--rich:widget:inline-example-->`\n\n"
        "```html\n<!--rich:widget:fenced-example-->\n```\n\n"
        "    <!--rich:widget:indented-example-->"
    )
    metadata = {
        "rich_items_version": 1,
        "rich_items": [{"id": valid_id, "type": "live_widget"}],
        "rich_reference_warnings": [],
    }

    cleaned, updated = response_constants.reconcile_final_rich_references(
        content, metadata
    )

    assert cleaned == content
    assert updated is metadata
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.\.conda\python.exe -m pytest -q --confcutdir=tests\client_backend tests\test_rich_response_metadata.py -k reconcile
```

Expected: FAIL because `reconcile_final_rich_references` does not exist.

- [ ] **Step 3: Implement the minimal pure reconciliation helper**

In `app/core/response_constants.py`, import `remove_inline_rich_reference` and add:

```python
def reconcile_final_rich_references(
    content: str,
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if metadata.get("rich_items_version") != RICH_ITEMS_VERSION:
        return content, metadata

    known_ids = {
        str(item.get("id"))
        for item in metadata.get("rich_items") or []
        if isinstance(item, dict) and item.get("id")
    }
    missing_ids = list(
        dict.fromkeys(
            item_id
            for item_id in parse_inline_rich_references(content)
            if item_id not in known_ids
        )
    )
    if not missing_ids:
        return content, metadata

    cleaned = content
    for item_id in missing_ids:
        cleaned = remove_inline_rich_reference(cleaned, item_id)

    updated = dict(metadata)
    updated["rich_reference_warnings"] = validate_rich_references(
        cleaned, updated.get("rich_items") or []
    )
    _logger.warning(
        "Unresolved rich markers removed code=unknown_rich_item count=%d ids=%s",
        len(missing_ids),
        missing_ids,
    )
    return cleaned, updated
```

- [ ] **Step 4: Run the focused metadata and parser tests**

Run:

```powershell
.\.conda\python.exe -m pytest -q --confcutdir=tests\client_backend tests\test_rich_response_metadata.py tests\test_rich_response_contract.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add app/core/response_constants.py tests/test_rich_response_metadata.py
git commit -m "fix: reconcile unresolved rich markers"
```

### Task 2: Apply Reconciliation to Every Persistence Path

**Files:**
- Modify: `tests/test_message_service_web_image_externalization.py`
- Modify: `app/services/message_service.py`

**Interfaces:**
- Consumes: `reconcile_final_rich_references(content, metadata)` from Task 1.
- Produces: normal completion and resume persistence that store cleaned content and matching metadata.

- [ ] **Step 1: Write failing normal-completion and resume tests**

Add the following tests beside the existing persistence-path tests:

```python
def _invented_marker_metadata():
    return {
        "rich_items_version": 1,
        "rich_items": [],
        "rich_reference_warnings": [
            {"code": "unknown_rich_item", "id": "widget:invented"}
        ],
    }


@pytest.mark.asyncio
async def test_completed_workflow_removes_invented_marker_before_persist(monkeypatch):
    service = _service(AsyncMock())
    service.chat_image_service = None
    service.task_plan_service = None
    service._sync_response_plan_state = Mock(return_value=False)
    service._acreate_bot_response_message = AsyncMock(return_value="persisted")
    body = "Before\n\n<!--rich:widget:invented-->\n\nAfter"
    _patch_response_builders(monkeypatch, body, _invented_marker_metadata())

    await service._persist_completed_workflow_response(
        conversation_id=uuid4(),
        user_id=uuid4(),
        bot_response=SimpleNamespace(metadata={}),
        sanitized_persona=None,
        workflow_request=None,
    )

    persisted = service._acreate_bot_response_message.await_args.kwargs
    assert persisted["content"] == "Before\n\nAfter"
    assert persisted["metadata"]["rich_reference_warnings"] == []


@pytest.mark.asyncio
async def test_resume_workflow_removes_invented_marker_before_persist(monkeypatch):
    service = _service(AsyncMock())
    service.conversation_validation_utils = SimpleNamespace(
        validate_conversation_access=lambda *_args: None
    )
    response = SimpleNamespace(metadata={})
    service.ai_service = SimpleNamespace(resume_workflow=AsyncMock(return_value=response))
    service._sync_response_plan_state = Mock(return_value=False)
    service._acreate_bot_response_message = AsyncMock(return_value="persisted")
    service._compact_checkpoint_after_persist = AsyncMock()
    body = "Before\n\n<!--rich:widget:invented-->\n\nAfter"
    _patch_response_builders(monkeypatch, body, _invented_marker_metadata())

    await service.resume_workflow(uuid4(), uuid4())

    persisted = service._acreate_bot_response_message.await_args.kwargs
    assert persisted["content"] == "Before\n\nAfter"
    assert persisted["metadata"]["rich_reference_warnings"] == []
```

- [ ] **Step 2: Run both tests and verify RED**

Run:

```powershell
.\.conda\python.exe -m pytest -q --confcutdir=tests\client_backend tests\test_message_service_web_image_externalization.py -k "invented_marker"
```

Expected: FAIL because both paths still persist the marker.

- [ ] **Step 3: Wire the helper once after metadata construction in both paths**

Import the helper from `app.core.response_constants`, then add the same boundary call immediately after `build_bot_metadata(...)` in `resume_workflow()` and `_persist_completed_workflow_response()`:

```python
bot_response_content, bot_metadata = reconcile_final_rich_references(
    bot_response_content,
    bot_metadata,
)
```

Keep `_externalize_remote_rich_images()` after reconciliation.

- [ ] **Step 4: Run message-service regression tests**

Run:

```powershell
.\.conda\python.exe -m pytest -q --confcutdir=tests\client_backend tests\test_message_service_web_image_externalization.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add app/services/message_service.py tests/test_message_service_web_image_externalization.py
git commit -m "fix: clean rich markers before persistence"
```

### Task 3: Remove the Unconditional Protocol Prompt and Final Unavailable Caption

**Files:**
- Modify: `tests/test_rich_response_prompt_inventory.py`
- Modify: `tests/test_demo_plan_widget.py`
- Modify: `tests/test_demo_stream_rendering.py`
- Modify: `app/ai/prompts.py`
- Modify: `demo.py`

**Interfaces:**
- Consumes: dynamic `build_rich_response_guidance()` output and internal `RichSegment(kind="unavailable")`.
- Produces: always-on prompts without marker syntax; dynamic prompts with exact inventory IDs; silent finalized rendering with active-stream pending placeholders unchanged.

- [ ] **Step 1: Write failing prompt and rendering tests**

Add a prompt boundary test:

```python
def test_always_on_media_guidance_does_not_expose_rich_marker_protocol():
    from app.ai.prompts import MEDIA_CAPABILITY_SNIPPET

    assert "<!--rich:" not in MEDIA_CAPABILITY_SNIPPET
```

Add a persisted-render test and a finalized-stream test:

```python
def test_persisted_unresolved_rich_segment_renders_silently(monkeypatch):
    from app.ui.rich_response import RichSegment

    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    captions = []
    streamlit_stub.caption = captions.append

    demo._render_rich_segments(
        [RichSegment(kind="unavailable", item_id="widget:invented")],
        message_metadata={},
        message_key="historic",
        auto_mount=False,
    )

    assert captions == []


def test_live_renderer_hides_unresolved_marker_after_completion(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    events = []

    class MarkdownSlot:
        def markdown(self, _text):
            return None

    class RootPlaceholder:
        def container(self):
            return nullcontext()

    streamlit_stub.empty = lambda: MarkdownSlot()
    streamlit_stub.caption = lambda text: events.append(("caption", text))
    monkeypatch.setattr(
        demo,
        "_render_pending_rich_placeholder",
        lambda item_id: events.append(("pending", item_id)),
    )
    renderer = demo._StreamingRichResponseRenderer(
        RootPlaceholder(), message_key="active-response"
    )
    body = "Before\n\n<!--rich:widget:invented-->\n\nAfter"
    renderer.append_text(body)
    assert events == [("pending", "widget:invented")]

    events.clear()
    renderer.finalize(
        {
            "content": body,
            "metadata": {"rich_items_version": 1, "rich_items": []},
        }
    )
    assert events == []
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.\.conda\python.exe -m pytest -q --confcutdir=tests\client_backend tests\test_rich_response_prompt_inventory.py tests\test_demo_plan_widget.py tests\test_demo_stream_rendering.py -k "marker_protocol or unresolved or unavailable"
```

Expected: FAIL on the unconditional prompt sentence and both finalized caption paths.

- [ ] **Step 3: Remove only the problem-causing prompt and captions**

Delete the marker-protocol sentence from `MEDIA_CAPABILITY_SNIPPET`. Keep
`INLINE_RICH_RESPONSE_SUFFIX` and `build_rich_response_guidance()` unchanged so
syntax and exact-ID rules still arrive dynamically with a real inventory.

In `_render_rich_segments()`, replace the unavailable caption branch with a
silent `continue`. In `_StreamingRichResponseRenderer._rebuild()`, retain
`_render_pending_rich_placeholder()` only when `_finalized` is false and emit
nothing when it is true.

- [ ] **Step 4: Run prompt, view-model, and renderer regressions**

Run:

```powershell
.\.conda\python.exe -m pytest -q --confcutdir=tests\client_backend tests\test_rich_response_prompt_inventory.py tests\test_demo_rich_response.py tests\test_demo_plan_widget.py tests\test_demo_stream_rendering.py
```

Expected: PASS, including the existing pending-placeholder test.

- [ ] **Step 5: Commit Task 3**

```powershell
git add app/ai/prompts.py demo.py tests/test_rich_response_prompt_inventory.py tests/test_demo_plan_widget.py tests/test_demo_stream_rendering.py
git commit -m "fix: hide unresolved rich markers"
```

### Task 4: Update Current Contracts and Verify the Integrated Fix

**Files:**
- Modify: `README.md`
- Modify: `docs/frontend/rich-image-rendering.md`
- Modify: `app/ui/rich_response.py`

**Interfaces:**
- Consumes: the final behavior from Tasks 1-3.
- Produces: current documentation and docstrings matching the new contract.

- [ ] **Step 1: Update current documentation**

Change current-contract prose to state:

- marker syntax is supplied only with a current-turn inventory;
- unresolved new markers are removed before persistence;
- unresolved historical markers render silently;
- active streams may show a temporary pending placeholder; and
- validation never mounts arbitrary input or crashes rendering.

Do not rewrite historical plans or the approved design record.

- [ ] **Step 2: Update `RichSegment` and view-builder docstrings**

Describe `unavailable` as an internal unresolved/pending state that final
renderers omit, rather than a required visible unavailable block.

- [ ] **Step 3: Run formatting and focused verification**

Run:

```powershell
.\.conda\python.exe -m ruff check app/core/response_constants.py app/services/message_service.py app/ai/prompts.py app/ui/rich_response.py demo.py tests/test_rich_response_metadata.py tests/test_message_service_web_image_externalization.py tests/test_rich_response_prompt_inventory.py tests/test_demo_plan_widget.py tests/test_demo_stream_rendering.py
.\.conda\python.exe -m pytest -q --confcutdir=tests\client_backend tests\test_rich_response_contract.py tests\test_rich_response_metadata.py tests\test_rich_response_prompt_inventory.py tests\test_rich_placement.py tests\test_demo_rich_response.py tests\test_demo_plan_widget.py tests\test_demo_stream_rendering.py tests\test_message_service_web_image_externalization.py tests\test_rich_response_streaming.py
git diff --check
```

Expected: all commands exit 0 with no failures or lint errors.

- [ ] **Step 4: Inspect the final diff for scope**

Confirm there are no hard-coded runtime IDs, database migrations, unrelated
refactors, or changes to image selection/externalization behavior.

- [ ] **Step 5: Commit Task 4**

```powershell
git add README.md docs/frontend/rich-image-rendering.md app/ui/rich_response.py
git commit -m "docs: update unresolved rich marker contract"
```
