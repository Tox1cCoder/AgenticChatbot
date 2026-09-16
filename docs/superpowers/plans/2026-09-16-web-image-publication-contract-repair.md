# Web Image Publication Contract Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model-selected web images survive public metadata finalization by constructing them through the canonical rich-image contract at their producer boundary.

**Architecture:** `WebResearchSession` will build each prepared image with `ImageRichItem` and serialize the validated model into the existing dictionary transport shape. Grounding, streaming, persistence, and client projection remain unchanged; focused tests cover the producer contract and the complete grounding-to-metadata path.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, pytest-asyncio.

## Global Constraints

- Do not add automatic image selection, anchoring, provider-rank fallback, or a model repair call.
- Do not weaken late public rich-item validation.
- Preserve the existing protected `/web-images/{id}` delivery path.
- Preserve text answers when no image is selected.
- Modify only the web-image producer and its focused regression tests.
- Use `.venv\Scripts\python.exe` for verification in this checkout.

---

### Task 1: Enforce the canonical rich-image contract at preparation time

**Files:**
- Modify: `tests/test_web_research_images.py`
- Modify: `app/ai/web_research/service.py`

**Interfaces:**
- Consumes: `ImageRichItem`, `GENERIC_IMAGE_ALT_TEXT`, and `PreparedImage.rich_item: dict[str, Any]`.
- Produces: validated serialized rich-image dictionaries containing `alt_text`, `payload.source_url`, and the protected delivery URL.

- [ ] **Step 1: Write the failing producer test**

Add this test to `tests/test_web_research_images.py`:

```python
@pytest.mark.asyncio
async def test_prepared_web_image_satisfies_public_rich_contract() -> None:
    session, _bundle, _service = await _session((b"one",))
    item = session.prepared_images["I1"].rich_item

    validated = validate_public_rich_item(item)

    assert isinstance(validated, ImageRichItem)
    assert validated.alt_text
    assert str(validated.payload.source_url) == "https://source.test/article"
    assert validated.payload.url.startswith("/web-images/")
```

- [ ] **Step 2: Write the failing full-path test**

Add this test to the same file:

```python
@pytest.mark.asyncio
async def test_selected_web_image_survives_grounding_and_metadata_finalization(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    session, _bundle, _service = await _session((b"one",))
    resolution = GroundingParser(session).resolve(
        "Current view [[source:S1]].\n\n[[image:I1]]"
    )
    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=resolution.text),
        metadata={
            "_rich_item_candidates": list(resolution.rich_items),
            "_inline_rich_response_v1": True,
        },
    )

    metadata = build_bot_metadata(response)

    assert "<!--rich:image:web:" in response.message.content
    assert len(metadata["rich_items"]) == 1
    item = metadata["rich_items"][0]
    assert item["payload"]["url"].startswith("/web-images/")
    assert item["payload"]["source_url"] == "https://source.test/article"
    assert item["alt_text"]
    assert metadata["rich_reference_warnings"] == []
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_images.py -k "prepared_web_image_satisfies_public_rich_contract or selected_web_image_survives_grounding_and_metadata_finalization"
```

Expected: both tests fail because the prepared dictionary has no `alt_text`;
the full-path test also observes an empty `metadata.rich_items` registry.

- [ ] **Step 4: Construct the item through `ImageRichItem`**

Import the canonical contract in `app/ai/web_research/service.py`:

```python
from app.core.rich_response import GENERIC_IMAGE_ALT_TEXT, ImageRichItem, RichItemType
```

Replace the handwritten dictionary with this validated construction and pass
the serialized value to `PreparedImage`:

```python
rich_item = ImageRichItem(
    id=f"image:web:{persisted.id}",
    type=RichItemType.image,
    source="image_search",
    title=candidate.title,
    alt_text=str(candidate.description or candidate.title or GENERIC_IMAGE_ALT_TEXT),
    payload={
        "url": delivery_url,
        "mime_type": fetched.media_type,
        "source_url": str(source.url),
        "width": fetched.width,
        "height": fetched.height,
        "description": candidate.description,
    },
    provenance={"provider": candidate.provider, "source_id": source.source_id},
).model_dump(mode="json", exclude_none=True)
```

- [ ] **Step 5: Run focused and adjacent regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_web_research_images.py tests/test_web_grounding.py tests/test_rich_response_metadata.py tests/test_message_service_web_image_externalization.py tests/test_ai_sdk_v6_stream_contract.py
```

Expected: all tests pass. If the known native `pyarrow` collection crash recurs,
record it separately and run import-isolated checks for
`validate_public_rich_item()` and `build_bot_metadata()`.

- [ ] **Step 6: Run static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check app/ai/web_research/service.py tests/test_web_research_images.py
git diff --check
```

Expected: both commands succeed.

- [ ] **Step 7: Commit the implementation**

```powershell
git add -- app/ai/web_research/service.py tests/test_web_research_images.py docs/superpowers/plans/2026-09-16-web-image-publication-contract-repair.md
git commit -m "fix: preserve grounded web images"
```
