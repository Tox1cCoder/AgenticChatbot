# Remove Public Original Image URL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `original_image_url` from every public rich-item response while preserving the private upstream URL used by `/web-images/{id}`.

**Architecture:** Prevent new tool image candidates from adding the field, and add one shared rich-metadata sanitizer at public serialization boundaries for historical rows. Apply that sanitizer in `MessageRead`, rich-item validation/finalization, and AI SDK projection so Streamlit, standard history, AI SDK history, and terminal streams converge on the same payload.

**Tech Stack:** Python 3.11+, Pydantic v2, FastAPI, AI SDK v6 stream projection, pytest, Ruff, Markdown.

## Global Constraints

- Keep `WebImageReference.upstream_url`; `/web-images/{id}` still needs it to fetch the selected image.
- Keep `payload.url` as the protected renderable reference.
- Keep `payload.source_url` as attribution-only.
- Never expose `provenance.original_image_url` through new or historical public messages.
- Do not mutate stored message dictionaries while sanitizing a response.
- Do not change protected-route ownership, SSRF, MIME, size, decode, or caching behavior.

---

### Task 1: Sanitize rich-item provenance at the public boundary

**Files:**
- Modify: `tests/test_rich_response_sources.py:205-220`
- Modify: `tests/test_rich_response_metadata.py`
- Modify: `app/ai/tool_execution.py:290-314`
- Modify: `app/core/rich_response.py:225-315`

**Interfaces:**
- Consumes: raw image candidate dictionaries and persisted `metadata.rich_items` lists.
- Produces: `sanitize_public_rich_item(item: Any) -> Any`, which copies a rich-item dictionary and removes `original_image_url` from its provenance without mutating the input.
- Produces: `sanitize_public_rich_metadata(metadata: dict[str, Any]) -> dict[str, Any]`, which copies metadata and sanitizes every rich item.

- [ ] **Step 1: Change the Brave candidate regression test to require omission**

Replace the old assertion that preserved the original URL:

```python
def test_brave_image_candidate_keeps_safe_provider_provenance_only():
    [cand] = build_image_candidates_from_tool_result(
        _brave_payload(), tool_call_id="call_b", tool_name="brave_image_search"
    )
    prov = cand["provenance"]
    assert prov["thumbnail_url"] == "https://img.test/thumb-1.jpg"
    assert "original_image_url" not in prov
    assert prov["source_domain"] == "example.com"
    assert prov["provider"] == "brave_image_search"
```

The production mutation this catches is re-adding the raw asset URL while normalizing a tool result.

- [ ] **Step 2: Add finalization and non-mutation regression tests**

Add to `tests/test_rich_response_metadata.py`:

```python
def test_build_bot_metadata_strips_original_image_url_from_public_provenance():
    candidate = {
        "id": "image:tool:c1:0",
        "type": "image",
        "display_policy": "inline_only",
        "alt_text": "Selected image",
        "payload": {
            "url": "https://img.test/selected.png",
            "mime_type": "image/png",
        },
        "provenance": {
            "provider": "tavily",
            "original_image_url": "https://img.test/original.png",
        },
    }
    response = WorkflowResponse(
        message=WorkflowResponseMessage(
            content="See this.\n\n<!--rich:image:tool:c1:0-->"
        ),
        metadata={"_rich_item_candidates": [candidate]},
    )

    metadata = build_bot_metadata(response)

    assert "original_image_url" not in metadata["rich_items"][0]["provenance"]
    assert candidate["provenance"]["original_image_url"].endswith("original.png")
```

The second assertion proves sanitization does not mutate the private/input dictionary.

- [ ] **Step 3: Run the two tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rich_response_sources.py::test_brave_image_candidate_keeps_safe_provider_provenance_only tests\test_rich_response_metadata.py::test_build_bot_metadata_strips_original_image_url_from_public_provenance -v
```

Expected: both tests FAIL because the current source and finalization retain `original_image_url`.

- [ ] **Step 4: Add the shared sanitizer and apply it during validation**

In `app/core/rich_response.py`, add:

```python
_PUBLIC_RICH_PROVENANCE_OMIT_KEYS = frozenset({"original_image_url"})


def sanitize_public_rich_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    sanitized = dict(item)
    provenance = sanitized.get("provenance")
    if isinstance(provenance, dict):
        sanitized["provenance"] = {
            key: value
            for key, value in provenance.items()
            if key not in _PUBLIC_RICH_PROVENANCE_OMIT_KEYS
        }
    return sanitized


def sanitize_public_rich_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(metadata)
    rich_items = sanitized.get("rich_items")
    if isinstance(rich_items, list):
        sanitized["rich_items"] = [sanitize_public_rich_item(item) for item in rich_items]
    return sanitized
```

Change `validate_public_rich_item()` to validate `sanitize_public_rich_item(item)` rather than `item`. Export both helpers in `__all__`.

- [ ] **Step 5: Stop adding the field to new tool candidates**

Delete this block from `app/ai/tool_execution.py`:

```python
if original_url:
    provenance["original_image_url"] = original_url
```

Keep `original_url` for choosing the render/thumbnail URL earlier in candidate normalization; only its public provenance copy is removed.

- [ ] **Step 6: Run Task 1 tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rich_response_sources.py tests\test_rich_response_metadata.py -q
```

Expected: PASS with zero failures.

- [ ] **Step 7: Commit Task 1**

```powershell
git add -- app/ai/tool_execution.py app/core/rich_response.py tests/test_rich_response_sources.py tests/test_rich_response_metadata.py
git commit -m "fix: remove original image URL from public rich items"
```

---

### Task 2: Sanitize historical and cross-path projections

**Files:**
- Modify: `tests/test_ai_sdk_context_window.py`
- Modify: `tests/test_rich_response_streaming.py`
- Modify: `tests/test_image_stream_http_contract.py`
- Modify: `app/services/event_streaming/ai_sdk_projection.py:20-90`
- Modify: `app/schemas/message.py:1-175`

**Interfaces:**
- Consumes: `sanitize_public_rich_metadata(metadata)` from Task 1.
- Produces: standard message history, AI SDK history, and terminal AI SDK stream payloads with the forbidden provenance field removed.

- [ ] **Step 1: Add a standard-history schema regression test**

Add to `tests/test_image_stream_http_contract.py`:

```python
def test_message_read_strips_original_image_url_from_historical_rich_items():
    conversation_id = uuid4()
    row = _message_row(
        conversation_id=conversation_id,
        sender=MessageRole.assistant.value,
        content="Answer\n\n<!--rich:image:tool:c1:0-->",
        metadata={
            "rich_items_version": 1,
            "rich_items": [
                {
                    "id": "image:tool:c1:0",
                    "type": "image",
                    "display_policy": "inline_only",
                    "alt_text": "Selected",
                    "payload": {
                        "url": "/web-images/11111111-1111-4111-8111-111111111111",
                        "mime_type": "image/jpeg",
                    },
                    "provenance": {
                        "provider": "tavily",
                        "original_image_url": "https://img.test/original.jpg",
                    },
                }
            ],
        },
    )

    message = MessageRead.model_validate(row)

    provenance = message.message_metadata["rich_items"][0]["provenance"]
    assert provenance == {"provider": "tavily"}
    assert row.message_metadata["rich_items"][0]["provenance"]["original_image_url"]
```

This catches exposing an older persisted field through the standard conversation/Streamlit history path and verifies that response construction does not mutate the stored row.

- [ ] **Step 2: Add AI SDK history and stream regressions**

In `tests/test_ai_sdk_context_window.py`, add `original_image_url` to the selected-image fixture provenance and assert it is absent from rich-capable history output.

In `tests/test_rich_response_streaming.py::test_rich_capable_stream_keeps_image_rich_item_without_file_event`, add the same provenance field to the terminal message input and assert:

```python
assert "original_image_url" not in content
```

These tests catch bypassing the sanitizer in direct API projection or the terminal stream adapter.

- [ ] **Step 3: Run the three tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_image_stream_http_contract.py::test_message_read_strips_original_image_url_from_historical_rich_items tests\test_ai_sdk_context_window.py::test_ai_sdk_rich_history_uses_image_rich_item_without_file_part tests\test_rich_response_streaming.py::test_rich_capable_stream_keeps_image_rich_item_without_file_event -v
```

Expected: FAIL because historical metadata and direct AI SDK projections still expose the field.

- [ ] **Step 4: Sanitize `MessageRead` metadata**

Import `sanitize_public_rich_metadata` in `app/schemas/message.py`. At the start of `MessageRead._populate_from_metadata()`, after confirming metadata is a dict, assign:

```python
self.message_metadata = sanitize_public_rich_metadata(self.message_metadata)
```

Continue reading interrupts and suggested questions from the sanitized copy.

- [ ] **Step 5: Sanitize AI SDK metadata projection**

Import `sanitize_public_rich_metadata` in `app/services/event_streaming/ai_sdk_projection.py`. At the start of `scrub_legacy_metadata()`, create:

```python
public_metadata = sanitize_public_rich_metadata(metadata)
```

Filter `public_metadata.items()` instead of `metadata.items()`. This covers both terminal stream events and direct history tests whose service doubles bypass `MessageRead`.

- [ ] **Step 6: Run cross-path tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_image_stream_http_contract.py tests\test_ai_sdk_context_window.py tests\test_rich_response_streaming.py -q
```

Expected: PASS with zero failures.

- [ ] **Step 7: Commit Task 2**

```powershell
git add -- app/schemas/message.py app/services/event_streaming/ai_sdk_projection.py tests/test_image_stream_http_contract.py tests/test_ai_sdk_context_window.py tests/test_rich_response_streaming.py
git commit -m "fix: sanitize historical rich image provenance"
```

---

### Task 3: Update the frontend contract and verify the release

**Files:**
- Modify: `plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md:330-365`
- Modify: `plans/AI_SDK_FE_CONTRACT.md:810-840` only if it still mentions the field
- Test: existing Python suites; human-facing prose receives no source-text test.

**Interfaces:**
- Consumes: the public response behavior from Tasks 1-2.
- Produces: one sendable contract stating that `original_image_url` is absent from the wire.

- [ ] **Step 1: Correct the normative contract**

Replace the diagnostic-provenance guidance with:

```markdown
- `payload.url` is the protected media reference.
- `payload.source_url` is a clickable attribution link only.
- `provenance.original_image_url` is not part of the public wire contract.
- Never fetch publisher asset URLs as fallback media.
```

Update the deployment diagnostic and FE checklist so neither tells the FE to inspect or retain `original_image_url`.

- [ ] **Step 2: Scan public code and contracts**

Run:

```powershell
rg -n "original_image_url" app tests plans docs/frontend --glob "!docs/superpowers/**"
```

Expected: matches exist only in negative regression fixtures/assertions and the normative statement that the field is absent. No production assignment or positive frontend usage remains.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rich_response_sources.py tests\test_rich_response_metadata.py tests\test_ai_sdk_context_window.py tests\test_rich_response_streaming.py tests\test_image_stream_http_contract.py tests\test_message_service_web_image_externalization.py tests\client_backend\test_image_stream_proxy.py -q
```

Expected: PASS with zero failures. Existing FastAPI duplicate-operation-ID warnings may remain unrelated.

- [ ] **Step 4: Run Ruff**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check app/ai/tool_execution.py app/core/rich_response.py app/schemas/message.py app/services/event_streaming/ai_sdk_projection.py tests/test_rich_response_sources.py tests/test_rich_response_metadata.py tests/test_image_stream_http_contract.py tests/test_ai_sdk_context_window.py tests/test_rich_response_streaming.py
```

Expected: `All checks passed!`

- [ ] **Step 5: Run the full suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: zero failures.

- [ ] **Step 6: Inspect repository state**

Run:

```powershell
git diff --check
git status --short
```

Expected: only the intended contract change is uncommitted and the pre-existing `superpowers-main.zip` remains untouched/untracked.

- [ ] **Step 7: Commit the contract**

```powershell
git add -- plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md plans/AI_SDK_FE_CONTRACT.md
git commit -m "docs: remove original image URL from frontend contract"
```
