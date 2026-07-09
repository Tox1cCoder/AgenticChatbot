# Chat Image Attachment Paste Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make chat image paste work after New Chat and replace the separate chat image file uploader with a unified paste-style image intake.

**Architecture:** Keep `pending_image_attachments` as the only Streamlit-side image queue. Extend `app/ui/clipboard_image_capture/index.html` so each mounted component becomes the current active image receiver and supports paste, drag/drop, and click-to-pick. Update `demo.py` to route all chat image intake payloads through `_handle_pasted_image_payload()`.

**Tech Stack:** Streamlit, Streamlit custom component HTML/JavaScript, pytest source-inspection and helper tests.

---

### Task 1: Baseline

**Files:**
- Test: `tests/test_demo_image_paste.py`

- [ ] **Step 1: Run the existing focused test module**

Run:

```powershell
python -m pytest tests/test_demo_image_paste.py -q
```

Expected: existing tests pass before changing behavior.

### Task 2: Failing Tests

**Files:**
- Modify: `tests/test_demo_image_paste.py`

- [ ] **Step 1: Add source-inspection tests for the new component and UI contracts**

Append these tests to `tests/test_demo_image_paste.py`:

```python
def test_clipboard_component_reregisters_active_receiver_each_mount():
    html = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "ui"
        / "clipboard_image_capture"
        / "index.html"
    ).read_text(encoding="utf-8")

    assert "__chatImagePasteActiveReceiver" in html
    assert "__chatImagePasteActiveReceiver = setComponentValue" in html
    assert "__chatImagePasteSetComponentValue" not in html


def test_clipboard_component_supports_paste_drop_and_picker_input():
    html = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "ui"
        / "clipboard_image_capture"
        / "index.html"
    ).read_text(encoding="utf-8")

    assert 'type="file"' in html
    assert 'accept="image/*"' in html
    assert "multiple" in html
    assert '"drop"' in html
    assert '"dragover"' in html
    assert "selectImageFiles" in html


def test_demo_replaces_chat_image_file_uploader_with_paste_style_intake():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    assert 'st.file_uploader(\\n                "Attach images"' not in source
    assert "show_attachment_uploader" in source
    assert "capture_pasted_images(" in source
    assert "_handle_pasted_image_payload(pasted_payload)" in source
```

- [ ] **Step 2: Run the new tests and verify they fail for the expected missing behavior**

Run:

```powershell
python -m pytest tests/test_demo_image_paste.py -q
```

Expected: failures mention missing `__chatImagePasteActiveReceiver`, missing picker/drop support, and/or the still-present chat image `st.file_uploader`.

### Task 3: Component Implementation

**Files:**
- Modify: `app/ui/clipboard_image_capture/index.html`

- [ ] **Step 1: Replace the component body with active receiver, paste, drop, and picker support**

Implement:

- `topDoc.__chatImagePasteActiveReceiver = setComponentValue` on every mount.
- A one-time global paste listener that calls `topDoc.__chatImagePasteActiveReceiver`.
- A visible drop/pick target inside the component frame.
- Shared `filesToPayload()` and `sendImages()` helpers used by paste, drop, and picker events.

- [ ] **Step 2: Run focused tests**

Run:

```powershell
python -m pytest tests/test_demo_image_paste.py -q
```

Expected: component-related tests pass; the UI replacement test can still fail until Task 4.

### Task 4: Streamlit UI Integration

**Files:**
- Modify: `demo.py`

- [ ] **Step 1: Remove the chat image `st.file_uploader` block**

Delete only the chat input image uploader block that renders `st.file_uploader("Attach images", ...)`. Do not touch document upload blocks.

- [ ] **Step 2: Mount the image intake when the attachment panel is toggled**

Keep the always-mounted hidden paste receiver near the message input. When `show_attachment_uploader` is true, mount a second visible `capture_pasted_images(...)` component with a key derived from the conversation id, and pass its payload to `_handle_pasted_image_payload()`.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
python -m pytest tests/test_demo_image_paste.py -q
```

Expected: all focused image paste tests pass.

### Task 5: Verification

**Files:**
- Test: `tests/test_demo_image_paste.py`
- Test: `tests/test_demo_document_file_types.py`
- Test: `tests/test_demo_stream_rendering.py`

- [ ] **Step 1: Run focused and nearby tests**

Run:

```powershell
python -m pytest tests/test_demo_image_paste.py tests/test_demo_document_file_types.py tests/test_demo_stream_rendering.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Review diff**

Run:

```powershell
git diff -- demo.py app/ui/clipboard_image_capture/index.html tests/test_demo_image_paste.py docs/superpowers/plans/2026-07-09-chat-image-attachment-paste-fix.md
```

Expected: diff is limited to the approved image attachment scope plus the plan.

- [ ] **Step 3: Commit implementation**

Run:

```powershell
git add demo.py app/ui/clipboard_image_capture/index.html tests/test_demo_image_paste.py docs/superpowers/plans/2026-07-09-chat-image-attachment-paste-fix.md
git commit -m "fix: unify chat image attachment intake"
```

Expected: commit succeeds.
