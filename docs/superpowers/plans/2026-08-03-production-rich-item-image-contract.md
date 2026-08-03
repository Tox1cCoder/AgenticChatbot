# Production Rich-Item Image Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make image rich items the single final authority for Streamlit and rich-capable AI SDK clients while preserving selected-image compatibility for non-rich clients and publishing one FE contract.

**Architecture:** Persisted markers plus `metadata.rich_items` remain canonical. AI SDK adapters compute selected file parts from the original message before capability stripping, suppress them for rich-capable v1 clients, and retain them for non-rich clients. Streamlit renders the same registry while preserving failed group cells and validating protected media.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, AI SDK v6 wire protocol, Streamlit, pytest, Ruff, Markdown.

## Global Constraints

- `image` and `image_group` remain first-class `RichItem` variants beside widgets, canvas artifacts, and tool renders.
- Final markers plus final `metadata.rich_items` are the only durable authority for rich-capable clients.
- Rich-capable means `inlineRichResponseV1: true` (or its documented snake-case alias) and an enabled rollout setting.
- Rich-capable Streamlit and AI SDK paths render no duplicate image file parts.
- Non-rich AI SDK paths strip markers/rich metadata but retain exactly the finalized selected images as file parts.
- All `/web-images` and `/chat-images` canonical and `/api` aliases remain authenticated.
- `source_url` is attribution only and `provenance.original_image_url` is diagnostic only; neither is a rendering fallback.
- Media failures never fail or replace the text answer.
- Add no database migration, dependency, feature flag, parallel image registry, or remote CSP exception.
- Preserve the unrelated untracked `superpowers-main.zip`.

## File map

- `app/services/event_streaming/ai_sdk_projection.py`: shared capability-aware image projection.
- `app/services/event_streaming/ai_sdk_v6.py`: streaming projection.
- `app/api/ai_sdk.py`: history projection.
- `demo.py`: authenticated Streamlit media resolution.
- `app/ui/rich_response.py`: stable image-group failure rendering.
- `plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md`: sole normative FE contract.
- Focused tests under `tests/` and `tests/client_backend/`.

---

### Task 1: Capability-aware selected-image projection

**Files:**
- Modify: `app/services/event_streaming/ai_sdk_projection.py:289-355`
- Test: `tests/test_ai_sdk_context_window.py:380-435`

**Interfaces:**
- Consumes: original assistant messages and `is_v1_rich_items_message()`.
- Produces: `visible_image_file_parts(message, *, is_v1=None, inline_rich_response_v1=False)` and `attach_image_parts_to_message(message, *, image_parts=None, is_v1=None, inline_rich_response_v1=False)`.
- An explicit `image_parts=[]` attaches nothing; `None` derives through `visible_image_file_parts()`.

- [ ] **Step 1: Write failing capability-boundary tests**

Add beside the existing group/file projection tests:

```python
def _v1_selected_image_message():
    return {
        "content": "Intro\n\n<!--rich:image:tool:c1:0-->",
        "metadata": {
            "rich_items_version": 1,
            "rich_items": [{
                "id": "image:tool:c1:0",
                "type": "image",
                "display_policy": "inline_only",
                "alt_text": "Selected",
                "payload": {
                    "url": "/web-images/11111111-1111-4111-8111-111111111111",
                    "mime_type": "image/jpeg",
                },
            }],
        },
    }


def test_rich_capable_v1_message_has_no_image_file_parts():
    assert visible_image_file_parts(
        _v1_selected_image_message(),
        is_v1=True,
        inline_rich_response_v1=True,
    ) == []


def test_non_rich_v1_message_keeps_selected_image_file_parts():
    assert visible_image_file_parts(
        _v1_selected_image_message(),
        is_v1=True,
        inline_rich_response_v1=False,
    ) == [{
        "url": "/web-images/11111111-1111-4111-8111-111111111111",
        "mediaType": "image/jpeg",
    }]


def test_explicit_empty_image_parts_prevents_rederivation():
    projected = attach_image_parts_to_message(
        _v1_selected_image_message(), image_parts=[], is_v1=True
    )
    assert all(p.get("type") != "file" for p in projected.get("parts") or [])
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ai_sdk_context_window.py -k "rich_capable_v1_message or non_rich_v1_message or explicit_empty_image_parts" -v
```

Expected: FAIL because the helper signatures do not yet accept the new arguments.

- [ ] **Step 3: Implement the minimal helper behavior**

```python
def visible_image_file_parts(
    message: dict[str, Any],
    *,
    is_v1: bool | None = None,
    inline_rich_response_v1: bool = False,
) -> list[dict[str, str]]:
    metadata = find_message_metadata(message)
    if is_v1 is None:
        is_v1 = is_v1_rich_items_message(metadata)
    if is_v1:
        if inline_rich_response_v1:
            return []
        return selected_image_file_parts_from_rich_items(metadata)
    return extract_image_file_parts_from_message(message)
```

Extend `attach_image_parts_to_message()` with `image_parts` and
`inline_rich_response_v1`. Use an explicit list when supplied; otherwise call
`visible_image_file_parts()`. Retain existing text-part creation and URL
deduplication.

- [ ] **Step 4: Verify GREEN and regressions**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ai_sdk_context_window.py -k "image or rich" -v
```

Expected: PASS, including group order, deduplication, unknown-type skipping, and pre-v1 compatibility.

- [ ] **Step 5: Commit**

```powershell
git add -- app/services/event_streaming/ai_sdk_projection.py tests/test_ai_sdk_context_window.py
git commit -m "refactor: make rich image file projection capability-aware"
```

---

### Task 2: Apply one projection to AI SDK streaming and history

**Files:**
- Modify: `app/services/event_streaming/ai_sdk_v6.py:498-535`
- Modify: `app/api/ai_sdk.py:424-477`
- Test: `tests/test_rich_response_streaming.py:220-352`
- Test: `tests/test_ai_sdk_context_window.py:310-378`

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces: identical capability behavior for stream file events, terminal message parts, and history parts.
- Compute selected parts from the original message before rich metadata is stripped.

- [ ] **Step 1: Add a failing rich-capable stream test**

```python
@pytest.mark.asyncio
async def test_rich_capable_stream_keeps_image_rich_item_without_file_event():
    selected_url = "/web-images/11111111-1111-4111-8111-111111111111"

    async def source() -> Any:
        yield make_event("complete", sequence=1, data={"message": {
            "id": "m-image",
            "content": "Intro\n\n<!--rich:image:tool:c1:0-->",
            "message_metadata": {
                "rich_items_version": 1,
                "rich_items": [{
                    "id": "image:tool:c1:0",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Selected",
                    "payload": {"url": selected_url, "mime_type": "image/jpeg"},
                }],
            },
        }})

    state = StreamState(
        message_id="m-image", text_id="t-image", reasoning_id="r-image",
        inline_rich_response_v1=True,
    )
    response = _build_ui_message_stream_response(lambda: source(), state)
    content = "".join([chunk async for chunk in response.body_iterator])
    assert '"type":"file"' not in content
    assert '"rich_items"' in content
    assert selected_url in content
```

- [ ] **Step 2: Correct the non-rich stream contract test**

Rename the current non-capable selected-image test and assert:

```python
assert "<!--rich:" not in content
assert '"rich_items"' not in content
assert content.count('"type":"file"') == 1
assert "https://img.test/selected.png" in content
```

- [ ] **Step 3: Add failing history tests for both modes**

Create a persisted selected-image fixture and assert:

```python
def test_ai_sdk_rich_history_uses_image_rich_item_without_file_part():
    payload = _history_payload(_build_assistant_message_with_selected_image(), capable=True)
    assert payload["metadata"]["rich_items"][0]["type"] == "image"
    assert all(p["type"] != "file" for p in payload["parts"])


def test_ai_sdk_non_rich_history_projects_selected_image_once():
    payload = _history_payload(_build_assistant_message_with_selected_image(), capable=False)
    assert "rich_items" not in payload["metadata"]
    assert "<!--rich:" not in payload["content"]
    files = [p for p in payload["parts"] if p["type"] == "file"]
    assert len(files) == 1
    assert files[0]["url"].startswith("/web-images/")
```

Implement `_history_payload()` using the existing
`get_conversation_messages_ai_sdk`, `_build_message_service`, and
`MessagePaginationParams` test helpers; do not mock projection functions.

```python
def _history_payload(message: SimpleNamespace, *, capable: bool) -> dict:
    response = asyncio.run(
        get_conversation_messages_ai_sdk(
            uuid4(),
            _build_message_service([message]),
            uuid4(),
            MessagePaginationParams(),
            inline_rich_response_v1=capable,
        )
    )
    return response.data.messages[0].model_dump(by_alias=True)
```

- [ ] **Step 4: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rich_response_streaming.py tests\test_ai_sdk_context_window.py -k "selected_image or image_rich_item or non_capable_stream" -v
```

Expected: rich mode FAILS due to duplicates; compatibility mode FAILS because selection is currently read after metadata stripping.

- [ ] **Step 5: Fix stream ordering**

In `_complete()`, use this order:

```python
original_message = data.get("message") or {}
message = original_message
file_parts: list[dict[str, str]] = []
is_v1 = False
if isinstance(original_message, dict):
    is_v1 = is_v1_rich_items_message(find_message_metadata(original_message))
    file_parts = visible_image_file_parts(
        original_message,
        is_v1=is_v1,
        inline_rich_response_v1=state.inline_rich_response_v1,
    )
    message = project_ai_sdk_message_for_capability(
        original_message,
        inline_rich_response_v1=state.inline_rich_response_v1,
    )
    message = attach_image_parts_to_message(
        message, image_parts=file_parts, is_v1=is_v1
    )
```

Emit stream `file` events from `file_parts`, never by re-reading the projected message.

- [ ] **Step 6: Fix history ordering identically**

In `app/api/ai_sdk.py`, compute `file_parts` from `message_payload` before
`project_ai_sdk_message_for_capability()`, pass
`inline_rich_response_v1=rich_response_capable`, then attach the explicit list.
Import `visible_image_file_parts`; do not add adapter-local rich parsing.

- [ ] **Step 7: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rich_response_streaming.py tests\test_ai_sdk_context_window.py tests\test_ai_sdk_v6_stream_contract.py tests\test_article_image_flow.py -v
```

Expected: PASS without weakening hidden-candidate or legacy assertions.

- [ ] **Step 8: Commit**

```powershell
git add -- app/services/event_streaming/ai_sdk_v6.py app/api/ai_sdk.py tests/test_rich_response_streaming.py tests/test_ai_sdk_context_window.py
git commit -m "fix: unify rich image streaming and history projection"
```

---

### Task 3: Harden Streamlit loading and preserve failed group cells

**Files:**
- Modify: `demo.py:5340-5402, 7947-8014`
- Modify: `app/ui/rich_response.py:106-178`
- Test: `tests/test_demo_image_reference_rendering.py:13-155`
- Test: `tests/test_demo_rich_response.py:101-147`

**Interfaces:**
- Consumes: authenticated protected URLs and ordered persisted group cells.
- Produces: MIME-validated image data and renderer-local `_load_failed: bool` cells.
- `_load_failed` is never persisted or added to the public rich-item schema.

- [ ] **Step 1: Write a failing protected MIME test**

```python
def test_protected_fetch_rejects_non_image_content_type(monkeypatch):
    class Response:
        headers = {"Content-Type": "text/html; charset=utf-8"}
        content = b"<html>not an image</html>"
        def raise_for_status(self):
            return None

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(demo, "get_http_session", lambda: Session())
    result = demo._fetch_protected_image_data_uri.__wrapped__(
        "/web-images/11111111-1111-4111-8111-111111111111", "token"
    )
    assert result is None
```

- [ ] **Step 2: Write a failing pre-resolved group failure test**

```python
def test_inline_group_preserves_failed_protected_cell_in_place(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr(demo, "st", SimpleNamespace(
        session_state={"auth_token": "tok"},
        markdown=lambda html, **_kwargs: rendered.append(html),
    ))
    monkeypatch.setattr(
        demo, "_fetch_protected_image_data_uri",
        lambda url, _token: "data:image/jpeg;base64,QUJD" if url.endswith("/1") else None,
    )
    demo._render_inline_rich_item(
        {
            "type": "image_group",
            "alt_text": "Two selected images",
            "payload": {"items": [
                {"url": "/web-images/1", "mime_type": "image/jpeg"},
                {"url": "/web-images/2", "mime_type": "image/jpeg"},
            ]},
        },
        message_metadata={}, message_key="m1", auto_mount=False,
    )
    html = rendered[0]
    assert html.count('data-role="cell"') == 2
    assert html.count("<img") == 1
    assert 'data-state="failed"' in html
    assert "Visual unavailable" in html
```

Also add builder tests for a mixed group and a one-cell failed group:

```python
def test_pre_resolved_failed_cells_keep_group_shape():
    html = build_inline_image_group_html([
        {"_load_failed": True, "url": None},
        {"url": "data:image/jpeg;base64,QUJD", "mime_type": "image/jpeg"},
    ], alt_text="x")
    assert html.count('data-role="cell"') == 2
    assert html.count("<img") == 1
    assert 'data-state="failed"' in html


def test_one_cell_pre_resolved_failure_is_neutral_not_broken():
    html = build_inline_image_group_html(
        [{"_load_failed": True, "url": None}], alt_text="x"
    )
    assert "Visual unavailable" in html
    assert "<img" not in html
```

- [ ] **Step 3: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_demo_image_reference_rendering.py tests\test_demo_rich_response.py -k "protected_fetch_rejects or preserves_failed or pre_resolved" -v
```

Expected: MIME test FAILS because HTML is encoded; group tests FAIL because failed cells are filtered.

- [ ] **Step 4: Reject non-image protected responses**

In `_fetch_protected_image_data_uri()`:

```python
mime = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
if not mime.startswith("image/") or not response.content:
    return None
encoded = base64.b64encode(response.content).decode("ascii")
return f"data:{mime};base64,{encoded}"
```

Do not use a source or provenance URL after failure.

- [ ] **Step 5: Preserve one renderer cell per persisted cell**

In the `image_group` branch of `_render_inline_rich_item()`:

```python
resolved_cells: list[dict[str, Any]] = []
for cell in payload.get("items") or []:
    if not isinstance(cell, dict):
        continue
    cell_src = _resolve_displayable_image_src(
        cell.get("url"), None, cell.get("mime_type") or "image/png"
    )
    resolved_cells.append(
        {**cell, "url": cell_src}
        if cell_src
        else {**cell, "url": None, "_load_failed": True}
    )
group_html = build_inline_image_group_html(
    resolved_cells, alt_text=item.get("alt_text")
)
```

- [ ] **Step 6: Render stable failed cells**

Update `build_inline_image_group_html()` to accept cells with a URL or
`_load_failed is True`. A pre-failed cell gets a visible neutral fallback, no
`<img>`, and no source caption. A successful cell keeps an initially hidden
fallback and a `data-role="cell-caption"` attribution. Replace the current
all-cells-collapse JavaScript with:

```python
onerror = (
    "const c=this.closest('[data-role=cell]');"
    "c.dataset.state='failed';"
    "c.querySelector('[data-role=cell-fallback]').style.display='block';"
    "const cap=c.querySelector('[data-role=cell-caption]');"
    "if(cap)cap.remove();"
    "this.remove();"
)
```

Keep every cell in order even when all fail. A one-cell failed group returns one
neutral failed figure.

- [ ] **Step 7: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_demo_image_reference_rendering.py tests\test_demo_rich_response.py tests\test_demo_stream_rendering.py -v
```

Expected: PASS, including hostile metadata escaping, one-cell layout, protected images, and stream/history marker parity.

- [ ] **Step 8: Commit**

```powershell
git add -- demo.py app/ui/rich_response.py tests/test_demo_image_reference_rendering.py tests/test_demo_rich_response.py
git commit -m "fix: preserve rich image group failures in Streamlit"
```

---

### Task 4: Lock protected sidecar routes in OpenAPI

**Files:**
- Test: `tests/client_backend/test_image_stream_proxy.py:538-637`

**Interfaces:**
- Consumes: `client_backend.main.create_app()`.
- Produces: a release lock for all protected media routes.
- This characterizes behavior already added in commit `3f5beb2`; it should pass immediately and needs no redundant production change.

- [ ] **Step 1: Add the OpenAPI contract test**

```python
def test_sidecar_openapi_exposes_every_protected_image_route():
    paths = create_app().openapi()["paths"]
    assert {
        "/chat-images/{image_id}",
        "/api/chat-images/{image_id}",
        "/web-images/{image_id}",
        "/api/web-images/{image_id}",
    }.issubset(paths)
```

- [ ] **Step 2: Verify route and behavior locks**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\client_backend\test_image_stream_proxy.py -k "openapi_exposes or web_image_route or chat_image_read_route" -v
```

Expected: PASS immediately. If it fails, inspect sidecar router registration; do not introduce an external URL fallback.

- [ ] **Step 3: Commit**

```powershell
git add -- tests/client_backend/test_image_stream_proxy.py
git commit -m "test: lock protected image routes in sidecar OpenAPI"
```

---

### Task 5: Publish one normative FE rich-item contract

**Files:**
- Create: `plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md`
- Modify: `plans/AI_SDK_FE_CONTRACT.md:246-270, 774-835`
- Replace: `plans/AI_SDK_FE_CONTRACT_UPDATES.md`
- Replace: `docs/frontend/rich-image-rendering.md`

**Interfaces:**
- Consumes: the approved design and Tasks 1-4 behavior.
- Produces: one sendable normative contract; older documents become pointers.

- [ ] **Step 1: Create the sendable contract**

Create `plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md` with these sections:

```markdown
# AI SDK and Streamlit Rich-Item Contract
## Status and version
## Capability and projection matrix
## Canonical message and RichItem schemas
## Streaming algorithm
## History algorithm
## Authenticated protected media
## Rendering and accessibility
## Failure and security rules
## FE acceptance checklist
## Deployment diagnostic
```

The document must include:

- complete JSON shapes for the assistant message, `image`, `image_group`, widget,
  canvas, and tool-render items;
- rich-capable Streamlit/AI SDK marker rendering with no image files;
- non-rich AI SDK selected file compatibility exactly once;
- registry replacement at terminal completion and preview cleanup;
- a TypeScript authenticated fetch-to-Blob helper using `AbortController`,
  `image/*` validation, `URL.createObjectURL`, and `URL.revokeObjectURL`;
- stable group cell failures, one-cell groups, alt/caption/source ownership;
- 401/404/413/502/504 and decode-failure semantics;
- explicit prohibitions on direct protected `<img src>`, remote provenance
  fallback, token leakage, unselected galleries, and remote CSP expansion;
- a stale-sidecar diagnostic: missing `/web-images` in OpenAPI means rebuild and
  restart the sidecar.

- [ ] **Step 2: Replace old normative image documents with pointers**

Use this content, correcting the relative link for each location:

```markdown
# Superseded rich-item image contract

This document is retained only as a compatibility pointer. The sole normative
contract for rich items, assistant images, protected media loading, Streamlit,
and AI SDK rendering is [AI_SDK_FE_RICH_ITEM_CONTRACT.md](AI_SDK_FE_RICH_ITEM_CONTRACT.md).

Do not implement behavior from an older revision of this file.
```

For `docs/frontend/rich-image-rendering.md`, use
`../../plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md` as the target.

- [ ] **Step 3: Correct the general AI SDK contract**

In `plans/AI_SDK_FE_CONTRACT.md`:

- link prominently to the new normative rich-item contract;
- replace “represented twice” with the capability matrix;
- document finalized image rich metadata for capable clients and file
  compatibility only for non-rich clients;
- remove instructions to deduplicate a server-created file/rich duplicate;
- retain general protected endpoint and error semantics.

- [ ] **Step 4: Validate links and scan contradictions**

```powershell
rg -n "represented twice|deduplicate.*file|original_image_url.*fallback" plans/AI_SDK_FE_CONTRACT.md plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md plans/AI_SDK_FE_CONTRACT_UPDATES.md docs/frontend/rich-image-rendering.md
Test-Path plans\AI_SDK_FE_RICH_ITEM_CONTRACT.md
```

Expected: `Test-Path` prints `True`. `rg` may find explicit prohibitions, but no instruction to render/deduplicate duplicate server representations.

- [ ] **Step 5: Commit**

```powershell
git add -- plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md plans/AI_SDK_FE_CONTRACT.md plans/AI_SDK_FE_CONTRACT_UPDATES.md docs/frontend/rich-image-rendering.md
git commit -m "docs: publish canonical rich-item frontend contract"
```

---

### Task 6: Cross-path verification and review

**Files:**
- Modify only when a requirement-specific regression is found in files already touched by Tasks 1-5.

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: fresh release evidence and reviewed code.

- [ ] **Step 1: Run focused rich-item and image suites**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rich_response_contract.py tests\test_rich_response_metadata.py tests\test_rich_response_streaming.py tests\test_ai_sdk_context_window.py tests\test_ai_sdk_v6_stream_contract.py tests\test_article_image_flow.py tests\test_demo_rich_response.py tests\test_demo_stream_rendering.py tests\test_demo_image_reference_rendering.py tests\client_backend\test_image_stream_proxy.py -v
```

Expected: all selected tests PASS with zero failures.

- [ ] **Step 2: Run Ruff on modified Python files**

```powershell
.\.venv\Scripts\python.exe -m ruff check app/services/event_streaming/ai_sdk_projection.py app/services/event_streaming/ai_sdk_v6.py app/api/ai_sdk.py app/ui/rich_response.py demo.py tests/test_ai_sdk_context_window.py tests/test_rich_response_streaming.py tests/test_demo_image_reference_rendering.py tests/test_demo_rich_response.py tests/client_backend/test_image_stream_proxy.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Verify sidecar OpenAPI directly**

```powershell
.\.venv\Scripts\python.exe -c "from client_backend.main import create_app; p=create_app().openapi()['paths']; expected={'/chat-images/{image_id}','/api/chat-images/{image_id}','/web-images/{image_id}','/api/web-images/{image_id}'}; missing=expected-set(p); assert not missing, missing; print('protected image routes:', sorted(expected))"
```

Expected: all four paths print and the command exits `0`. Existing duplicate-operation-id warnings are unrelated.

- [ ] **Step 4: Inspect scope and contract invariants**

```powershell
git status --short
git diff 7dba071..HEAD --stat
rg -n "inlineRichResponseV1|AI SDK rich-capable|AI SDK compatibility|/web-images/\{id\}|/chat-images/\{id\}|URL.revokeObjectURL|Visual unavailable" plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md
```

Expected: only `superpowers-main.zip` is unrelated/untracked; every required contract term is present.

- [ ] **Step 5: Request code review**

Use `superpowers:requesting-code-review` for `7dba071..HEAD`. Address each
behavioral finding with a failing regression test before changing production
code.

- [ ] **Step 6: Re-run verification after review changes**

Repeat Steps 1-3 after the final review fix. Do not rely on earlier test output.

- [ ] **Step 7: Commit review fixes only when needed**

```powershell
git commit -m "fix: close rich-item image contract review findings"
```

Do not create an empty commit when review requires no changes.
