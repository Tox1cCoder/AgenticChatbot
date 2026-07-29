# Remove Unavailable-Visual Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the rich-image UI that claims a visual is unavailable after one failed fetch attempt.

**Architecture:** Keep the successful-image component and protected delivery pipeline unchanged. Delete the unavailable fallback component and template; a server-side resolution failure renders nothing, while a later browser load failure silently removes only the current figure.

**Tech Stack:** Python 3.10+, Streamlit, HTML/JavaScript event attributes, pytest, Ruff

## Global Constraints

- A failed request proves only that one attempt failed; never infer permanent image unavailability.
- Preserve successful-image alt text, structured caption, source attribution, sizing, and lightbox behavior.
- Preserve image discovery, persistence, authenticated media routes, bounded retrieval, HTTP transport errors, and fetch metrics.
- Do not rewrite historical implementation plans.
- Use test-first red/green development for every behavior change.

---

### Task 1: Remove the Streamlit unavailable-visual fallback

**Files:**
- Modify: `tests/test_demo_rich_response.py:7-102`
- Modify: `tests/test_demo_image_reference_rendering.py:128-157`
- Modify: `app/ui/rich_response.py:31-135`
- Modify: `demo.py:7769-7811`

**Interfaces:**
- Consumes: `build_inline_image_html(src, *, alt_text, caption, source_url, width, height, max_width_px) -> str`
- Produces: successful-image HTML with no unavailable fallback markup; `_render_inline_rich_item(...)` emits no Markdown when protected resolution returns `None`
- Removes: `build_inline_image_unavailable_html(*, source_url) -> str`

- [ ] **Step 1: Replace fallback renderer tests with failing absence tests**

In `tests/test_demo_rich_response.py`, remove the import of
`build_inline_image_unavailable_html`, delete
`test_unavailable_image_is_compact_escaped_and_has_no_caption`, and replace
`test_image_loading_reserves_known_aspect_ratio_and_error_replaces_figure` with:

```python
def test_image_loading_has_no_unavailable_fallback_and_removes_failed_figure():
    out = build_inline_image_html(
        "https://img.test/a.png",
        alt_text="Example",
        caption="Trusted caption",
        source_url="https://publisher.example/story",
        width=640,
        height=360,
    )

    assert 'data-state="loading"' in out
    assert "aspect-ratio:640 / 360" in out
    assert "Visual unavailable" not in out
    assert "Open source" not in out
    assert "<template>" not in out
    assert "replaceChildren" not in out
    assert "this.closest('figure').remove()" in out
```

In `tests/test_demo_image_reference_rendering.py`, replace
`test_failed_protected_fetch_renders_complete_unavailable_state` with:

```python
def test_failed_protected_fetch_renders_no_fallback(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state={"auth_token": "tok"},
            markdown=lambda html, **_kwargs: rendered.append(html),
        ),
    )
    monkeypatch.setattr(demo, "_fetch_protected_image_data_uri", lambda *_: None)

    demo._render_inline_rich_item(
        {
            "type": "image",
            "alt_text": "Example",
            "payload": {
                "url": "/web-images/missing",
                "mime_type": "image/jpeg",
                "source_url": "https://publisher.example/story",
            },
        },
        message_metadata={},
        message_key="m1",
        auto_mount=False,
    )

    assert rendered == []
```

- [ ] **Step 2: Run the focused tests and verify the intended failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_demo_rich_response.py::test_image_loading_has_no_unavailable_fallback_and_removes_failed_figure tests/test_demo_image_reference_rendering.py::test_failed_protected_fetch_renders_no_fallback -q
```

Expected: FAIL because current HTML contains `Visual unavailable`, `<template>`, and `replaceChildren`, and the failed protected fetch currently invokes `st.markdown` with fallback HTML.

- [ ] **Step 3: Remove the fallback implementation**

In `app/ui/rich_response.py`:

1. Keep the existing loading skeleton, aspect-ratio reservation, footer, and successful `onload` behavior.
2. Replace the current `onerror` block with:

```python
    onerror = "this.closest('figure').remove()"
```

3. Delete `fallback = _unavailable_inner_html(source_url)` and return the figure without a template:

```python
    return (
        f'<figure id="rich-image-{component_id}" data-state="loading" '
        'style="margin:8px 0;text-align:center;">'
        f'<div class="rich-image-media" style="{wrapper_style}">{skeleton}{img}</div>'
        f"{footer}</figure>"
    )
```

4. Delete `build_inline_image_unavailable_html` and `_unavailable_inner_html`.
5. Simplify `_source_link_html` to `def _source_link_html(source_url: str | None) -> str` and always label a valid link with `Source: <hostname>`.
6. Remove `build_inline_image_unavailable_html` from `__all__`.

In `demo.py`, import only `build_inline_image_html` inside
`_render_inline_rich_item` and delete the entire `else` branch that renders
`build_inline_image_unavailable_html` when `src` is falsey.

- [ ] **Step 4: Run renderer tests and verify green**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_demo_rich_response.py tests/test_demo_image_reference_rendering.py -q
```

Expected: PASS. Successful images retain their single footer and failed protected resolution renders no fallback.

- [ ] **Step 5: Run scoped static checks**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check app/ui/rich_response.py demo.py tests/test_demo_rich_response.py tests/test_demo_image_reference_rendering.py
git diff --check
```

Expected: Ruff reports `All checks passed!`; `git diff --check` emits no errors.

- [ ] **Step 6: Commit the renderer rollback**

```powershell
git add app/ui/rich_response.py demo.py tests/test_demo_rich_response.py tests/test_demo_image_reference_rendering.py
git commit -m "fix: remove unavailable visual fallback"
```

---

### Task 2: Align the current frontend contract and verify the feature

**Files:**
- Modify: `docs/frontend/rich-image-rendering.md:77-94`
- Test: `tests/test_demo_rich_response.py`
- Test: `tests/test_demo_image_reference_rendering.py`
- Test: `tests/test_web_images_api.py`
- Test: `tests/client_backend/test_image_stream_proxy.py`

**Interfaces:**
- Consumes: fallback-free renderer behavior from Task 1
- Produces: current frontend guidance that distinguishes one failed attempt from permanent unavailability

- [ ] **Step 1: Update current frontend guidance**

In `docs/frontend/rich-image-rendering.md`, replace the `failed` state bullet and
the paragraph that says failures change the figure to an unavailable state with:

```markdown
- A failed fetch or decode attempt does not establish that the visual is
  permanently unavailable. Remove the current figure without rendering an
  unavailable label or fallback source action. A later message render may make
  another attempt according to the client's request-cache policy.

Network, authentication, publisher, MIME, size, decoding, and DNS failures are
transport outcomes for one request. They must not replace or fail the assistant
text and must not be presented as a permanent property of the image.
```

Remove the release-checklist requirement for a `Visual unavailable` state and
replace it with: `a failed attempt produces no unavailable label or fallback source action`.

- [ ] **Step 2: Run the complete related regression suite**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_demo_rich_response.py tests/test_demo_image_reference_rendering.py tests/test_web_images_api.py tests/client_backend/test_image_stream_proxy.py tests/test_article_image_flow.py tests/test_message_service_web_image_externalization.py -q
```

Expected: PASS. Backend transport behavior remains intact while the Streamlit fallback is absent.

- [ ] **Step 3: Run full verification**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check app/ui/rich_response.py demo.py
git status --short
```

Expected: full pytest passes with only environment-dependent skips; Ruff passes; status contains only the intended documentation edit before commit.

- [ ] **Step 4: Commit the contract update**

```powershell
git add docs/frontend/rich-image-rendering.md
git commit -m "docs: remove unavailable visual fallback contract"
```
