# Article-Style Rich Responses — Spec & Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Final answers read like an article — images and live widgets placed inline near the paragraphs they support — with the model never claiming it "cannot send images", and tool-result blobs stored in Postgres instead of `data/tool_result_blobs/` files.

**Architecture:** Three independent changes. (1) A deterministic, server-side placement engine (`app/core/rich_placement.py`) inserts `<!--rich:<id>-->` markers for relevant unreferenced images/widgets into the final markdown at persistence time — zero extra LLM calls, zero prompt-context cost. (2) A compact shared media-capability snippet is appended to all answer-producing system prompts, and the `inline_rich_response_enabled` rollout flag default flips to `True`. (3) `ToolResultBlobService` writes blob content to a new `content` column on the existing `tool_result_blobs` Postgres table; the filesystem path remains read-only fallback for legacy records.

**Tech Stack:** FastAPI, SQLAlchemy + Alembic (Postgres), Pydantic v2 settings, pytest, ruff.

---

## Part 1 — Specification

### Problem statement

1. **Images silently dropped.** Image rich items are `inline_only` ([app/core/rich_response.py:185](../app/core/rich_response.py)): if the model doesn't write a `<!--rich:<id>-->` marker, the image candidate is discarded at finalization ([app/core/response_constants.py:261](../app/core/response_constants.py)). Tavily search returns images with descriptions, but answers frequently appear text-only.
2. **Model claims it cannot send images.** No base system prompt ([app/ai/prompts.py](../app/ai/prompts.py)) states the image-display capability. Marker guidance arrives only conditionally (flag on + capability advertised + candidates already present), so before/without a search the model believes it has no image channel. Additionally `inline_rich_response_enabled` defaults to `False` ([app/core/config.py:971](../app/core/config.py)), disabling the entire pipeline unless overridden by env.
3. **Widgets bunch at the end.** Unreferenced widgets fall back to append-after-body, not near the supporting paragraph.
4. **Files created in repo.** Tool outputs >16k chars are written to `data/tool_result_blobs/<conversation_id>/<blob_id>.txt` ([app/services/tool_result_blob_service.py:43-47](../app/services/tool_result_blob_service.py)), polluting the working tree (untracked dirs in `git status`).

### Functional requirements

- **FR-1** Unreferenced image candidates whose description matches a paragraph of the answer are auto-placed inline (marker inserted after the best-matching paragraph) at persistence time. Matching is deterministic keyword overlap — no LLM call.
- **FR-2** Auto-placement is bounded: at most `rich_auto_place_max_images` images per answer (setting, default 3), only above a relevance threshold `rich_auto_place_min_score` (setting, default 0.25), at most one auto-placed item per paragraph. Items below threshold stay dropped (relevance gate).
- **FR-3** Unreferenced live widgets are auto-placed near their best-matching paragraph; when no paragraph matches, the existing append-after-body fallback still applies (no regression).
- **FR-4** Auto-placed markers are reflected consistently in the persisted message content AND in the finalized `rich_items` registry (`build_bot_metadata` parses the same content the message persists). The terminal `complete` stream event already carries the persisted message (`message_service.py:1146-1152`), so clients receive the placed layout without protocol changes.
- **FR-5** All answer-producing system prompts (chat, search ×2, RAG ×2) contain one shared, compact media-capability snippet stating the model CAN display images inline, must never claim otherwise, and should weave media into the narrative article-style. The snippet is defined once (no copy-paste per prompt).
- **FR-6** `inline_rich_response_enabled` defaults to `True` (the demo client already sends `inlineRichResponseV1: true`; the renderer exists in `app/ui/rich_response.py`). The flag is retained as a kill switch.
- **FR-7** New tool-result blobs store their full content in Postgres (`tool_result_blobs.content` TEXT column). No files are created under `data/`. `GET /tool-results/{blob_id}` keeps working unchanged.
- **FR-8** Legacy blobs (rows with `storage_path` set, `content` NULL) remain readable through the same `read_text()` path.
- **FR-9** No hardcoded values: placement cap, threshold, and toggle are Pydantic settings; blob behavior keeps its existing settings.
- **FR-10** Prompt context stays lean: the only prompt addition is the ~6-line shared snippet; the existing bounded inventory mechanism is unchanged.

### Non-goals

- No new LLM calls for placement or captioning.
- No changes to the AI-SDK stream protocol or the frontend contract (`plans/AI_SDK_FE_CONTRACT.md`).
- No image relevance ML/embedding scoring — keyword overlap only.
- No migration of existing on-disk blob files into Postgres (legacy read fallback covers them).
- No change to the widget runtime, Redis store, or WebSocket APIs.

### Edge cases

| Case | Behavior |
|---|---|
| Model already placed a marker for item X | X is skipped by auto-placement (already referenced) |
| Answer is one short paragraph | Items can place after it (one item max — one per paragraph rule) |
| Content is only code blocks | No placement (code blocks excluded from matching/insertion) |
| Image description in different language than answer | Likely below threshold → dropped (acceptable; relevance gate) |
| Duplicate candidates (same id) | First wins; `_finalize_rich_items` already dedups by id |
| `rich_auto_place_enabled=False` | Behavior identical to today |
| Empty/None content, no candidates | Returns content unchanged |
| Blob row with `content` NULL and `storage_path` NULL | `read_text` raises `ValueError` with blob id (corrupt record — fail fast) |
| Streaming clients | Deltas stream without markers; terminal `complete` event delivers the persisted, marker-bearing message (existing finalize-swap behavior) |

### Acceptance criteria

1. Given a Tavily result with an image described "Eiffel Tower illuminated at night in Paris" and an answer paragraph about the Eiffel Tower at night, the persisted message contains `<!--rich:image:tool:<call>:0-->` after that paragraph and the image is in `rich_items`. (Automated test.)
2. All five answer prompts contain the sentinel "You CAN display images inline". (Automated test.)
3. `ToolResultBlobService.offload_if_large` creates zero files; record carries `content`; `read_text` round-trips. (Automated test.)
4. Legacy record (storage_path set, content NULL) still reads from disk. (Automated test.)
5. `Settings(_env_file=None).inline_rich_response_enabled is True`. (Updated contract test.)
6. Full test suite and ruff pass.

---

## Part 2 — Technical context (verified, file:line)

- **Marker contract & parser:** `app/core/rich_response.py` — `parse_inline_rich_references()` (line 317), marker regex `^[ ]{0,3}<!--rich:([A-Za-z0-9_\-.:]+)-->[ \t]*$` (line 253), fence detection `_strip_fenced_code_blocks()` (line 258), `ImageRichItem.display_policy = inline_only` (line 185).
- **Finalization:** `app/core/response_constants.py` — `build_bot_metadata()` (line 331) pops `_rich_item_candidates` / `_inline_rich_response_v1`, calls `_finalize_rich_items()` (line 205) which drops unreferenced images (line 261). Content is read from `response.message.content` (lines 402-406).
- **Persistence call sites:** `app/services/message_service.py` — `resume_workflow()` builds content at line 2023, metadata at 2025; `_persist_completed_workflow_response()` builds content at 2364-2366 (with `fix_markdown_code_blocks`), metadata at 2367. Terminal `complete` event carries `bot_message.model_dump()` (lines 1146-1152, 1758-1764).
- **Candidates:** `app/ai/tool_execution.py` — `build_image_candidates_from_tool_result()` (line 62) produces ids `image:tool:<tool_call_id>:<index>` with `alt_text`/`title`/`payload.description`. Attached per-artifact as `_rich_item_candidates` (line 230), lifted into context in `app/ai/graph.py:1988-2009`, forwarded into `response.metadata` at `graph.py:429-444`.
- **Widgets:** `extract_live_widgets_from_artifacts()` (`response_constants.py:105`) → rich item id `widget:<widget_id>`, policy `inline_or_append` (line 173).
- **Prompt injection point:** inventory block reaches the system prompt via `_final_response_kwargs` → `rich_response_inventory` kwarg → `base_agent.py:1259-1261` (appended to system prompt). Base prompts live in `app/ai/prompts.py`: `CHAT_SYSTEM_PROMPT` (46), `RAG_SYSTEM_PROMPT` (89), `AGENTIC_RAG_SYSTEM_PROMPT` (119), `SEARCH_SYSTEM_PROMPT` (174), `SEARCH_WITH_RESULTS_SYSTEM_PROMPT` (430).
- **Flags:** `app/core/config.py` — `inline_rich_response_enabled` default False (971), inventory bounds (980-991), blob offload settings (646-661).
- **Blob storage:** model `app/models/tool_result_blob.py`, repo `app/repositories/tool_result_blob.py`, service `app/services/tool_result_blob_service.py`, API `app/api/tool_result_blobs.py`, DI wiring `app/core/container.py:252-258`. Alembic: `alembic.ini` at repo root, versions in `app/alembic/versions/`.
- **Existing tests:** `tests/test_rich_response_metadata.py`, `tests/test_rich_response_contract.py` (line 29 asserts flag default False — must flip), `tests/test_rich_response_prompt_inventory.py`, `tests/test_tool_result_blob_service.py`.

### File structure (new/modified)

| File | Action | Responsibility |
|---|---|---|
| `app/core/rich_placement.py` | Create | Deterministic placement engine + persistence-boundary integration fn |
| `tests/test_rich_placement.py` | Create | Unit tests for placement engine + integration fn |
| `app/core/config.py` | Modify | 3 new auto-place settings; flip rich flag default; blob dir description |
| `app/services/message_service.py` | Modify | Call `finalize_article_content` at 2 persistence sites |
| `app/ai/prompts.py` | Modify | `MEDIA_CAPABILITY_SNIPPET`, appended to 5 prompts |
| `tests/test_prompts_media_capability.py` | Create | Prompt sentinel assertions |
| `tests/test_article_image_flow.py` | Create | Tavily JSON → candidates → placement → `build_bot_metadata` E2E |
| `tests/test_rich_response_contract.py` | Modify | Flag-default assertion flips to True |
| `app/alembic/versions/<rev>_add_content_to_tool_result_blobs.py` | Create | `content` column; `storage_path` nullable |
| `app/models/tool_result_blob.py` | Modify | Add `content`; `storage_path` nullable |
| `app/services/tool_result_blob_service.py` | Modify | DB-backed write; legacy file read fallback |
| `tests/test_tool_result_blob_service.py` | Modify | New storage behavior + legacy fallback |
| `.gitignore` | Modify | Ignore `data/` |

---

## Part 3 — Implementation plan

Run all commands from the repo root: `c:\Users\ADMIN\Documents\Code Practice\Sample Chatbot`. Use the project venv (`.venv`) — on Git Bash: `.venv/Scripts/python`. `python -m pytest` below means `.venv/Scripts/python -m pytest`.

### Task 1: Auto-placement settings

**Files:**
- Modify: `app/core/config.py` (insert after `rich_item_selected_image_max_bytes`, ~line 998)

- [x] **Step 1: Add the three settings**

```python
    rich_auto_place_enabled: bool = Field(
        default=True,
        description=(
            "Deterministically insert inline markers for relevant unreferenced "
            "rich items (images, widgets) into the final answer at persistence time."
        ),
    )
    rich_auto_place_max_images: int = Field(
        default=3,
        description="Maximum number of image markers auto-placement may insert per answer.",
    )
    rich_auto_place_min_score: float = Field(
        default=0.25,
        description=(
            "Minimum keyword-overlap score (fraction of an item's descriptive tokens "
            "found in a paragraph) required to auto-place the item after that paragraph."
        ),
    )
```

- [x] **Step 2: Verify settings import cleanly**

Run: `python -c "from app.core.config import settings; print(settings.rich_auto_place_enabled, settings.rich_auto_place_max_images, settings.rich_auto_place_min_score)"`
Expected: `True 3 0.25`

- [x] **Step 3: Commit**

```bash
git add app/core/config.py
git commit -m "feat: add auto-placement settings for inline rich items"
```

### Task 2: Placement engine (`auto_place_rich_items`)

**Files:**
- Create: `app/core/rich_placement.py`
- Test: `tests/test_rich_placement.py`

- [x] **Step 1: Write the failing tests**

```python
"""Tests for deterministic article-style placement of rich items."""

from app.core.rich_placement import auto_place_rich_items

IMAGE = "image"
WIDGET = "live_widget"


def test_places_image_after_matching_paragraph():
    content = (
        "# Paris travel guide\n\n"
        "The Eiffel Tower is stunning at night, lit by thousands of lamps.\n\n"
        "The Louvre houses the Mona Lisa and countless other works."
    )
    items = [("image:tool:c1:0", IMAGE, "Eiffel Tower illuminated at night in Paris")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
    assert placed == ["image:tool:c1:0"]
    lines = new_content.split("\n")
    eiffel_line = next(i for i, ln in enumerate(lines) if "Eiffel Tower is stunning" in ln)
    marker_line = next(i for i, ln in enumerate(lines) if ln == "<!--rich:image:tool:c1:0-->")
    louvre_line = next(i for i, ln in enumerate(lines) if "Louvre" in ln)
    assert eiffel_line < marker_line < louvre_line


def test_skips_items_below_min_score():
    content = "A paragraph about quarterly revenue growth and profit margins."
    items = [("image:tool:c1:0", IMAGE, "A cat sleeping on a windowsill")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
    assert placed == []
    assert new_content == content


def test_respects_max_images_cap():
    content = (
        "Solar panels convert sunlight into electricity.\n\n"
        "Wind turbines harvest kinetic energy from moving air.\n\n"
        "Hydroelectric dams use falling water to spin turbines."
    )
    items = [
        ("img:0", IMAGE, "solar panels sunlight electricity"),
        ("img:1", IMAGE, "wind turbines kinetic energy air"),
        ("img:2", IMAGE, "hydroelectric dams falling water turbines"),
    ]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=2, min_score=0.25
    )
    assert len(placed) == 2
    assert new_content.count("<!--rich:") == 2


def test_one_item_per_paragraph():
    content = "Solar panels convert sunlight into electricity using semiconductors."
    items = [
        ("img:0", IMAGE, "solar panels sunlight electricity"),
        ("img:1", IMAGE, "solar panels converting sunlight semiconductors"),
    ]
    _, placed = auto_place_rich_items(content, items=items, max_images=3, min_score=0.25)
    assert placed == ["img:0"]


def test_skips_already_referenced_items():
    content = (
        "Solar panels convert sunlight into electricity.\n\n"
        "<!--rich:img:0-->\n"
    )
    items = [("img:0", IMAGE, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
    assert placed == []
    assert new_content == content


def test_never_places_inside_code_blocks():
    content = (
        "```python\n"
        "# solar panels sunlight electricity semiconductors\n"
        "print('solar panels sunlight electricity')\n"
        "```"
    )
    items = [("img:0", IMAGE, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
    assert placed == []
    assert new_content == content


def test_places_widget_near_matching_paragraph():
    content = (
        "Here is the revenue comparison between the two quarters.\n\n"
        "Overall the trend is positive."
    )
    items = [("widget:w1", WIDGET, "Quarterly revenue comparison chart")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=0, min_score=0.25
    )
    assert placed == ["widget:w1"]
    assert "<!--rich:widget:w1-->" in new_content


def test_widget_cap_independent_of_image_cap():
    content = "Quarterly revenue comparison for the two business units."
    items = [("widget:w1", WIDGET, "quarterly revenue comparison chart")]
    _, placed = auto_place_rich_items(content, items=items, max_images=0, min_score=0.25)
    assert placed == ["widget:w1"]


def test_empty_inputs_are_safe():
    assert auto_place_rich_items("", items=[("a", IMAGE, "x")], max_images=3, min_score=0.2) == ("", [])
    assert auto_place_rich_items("text", items=[], max_images=3, min_score=0.2) == ("text", [])


def test_inserted_marker_is_parseable():
    from app.core.rich_response import parse_inline_rich_references

    content = "Solar panels convert sunlight into electricity using semiconductors."
    items = [("img:0", IMAGE, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
    assert parse_inline_rich_references(new_content) == ["img:0"]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rich_placement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.rich_placement'`

- [x] **Step 3: Implement the placement engine**

Create `app/core/rich_placement.py`:

```python
"""Deterministic article-style placement of unreferenced rich items.

Inserts ``<!--rich:<id>-->`` markers into the final assistant markdown for
relevant rich items the model did not place itself, so answers read like an
article with inline media instead of silently dropping images. Placement is
keyword-overlap based and runs once at persistence time — no extra model
calls and no prompt-context cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import settings
from .response_constants import extract_live_widgets_from_artifacts
from .rich_response import (
    RichItemType,
    _strip_fenced_code_blocks,
    parse_inline_rich_references,
)

_WORD_RE = re.compile(r"[a-z0-9]{3,}")

#: Generic words that carry no placement signal for media descriptions.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "were", "has", "have", "had", "its", "but", "not", "you", "your",
        "they", "their", "about", "into", "over", "after", "before",
        "between", "image", "photo", "picture", "stock", "view",
    }
)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS)


@dataclass
class _Block:
    """A contiguous run of non-blank markdown lines outside fenced code."""

    end_line: int
    tokens: frozenset[str]
    is_code: bool


def _segment_blocks(lines: list[str]) -> list[_Block]:
    in_fence = _strip_fenced_code_blocks(lines)
    blocks: list[_Block] = []
    current: list[str] = []
    current_code = False
    current_end = -1
    for idx, (line, fenced) in enumerate(zip(lines, in_fence, strict=False)):
        if not line.strip():
            if current:
                blocks.append(_Block(current_end, _tokens(" ".join(current)), current_code))
                current, current_code = [], False
            continue
        current.append(line)
        current_end = idx
        current_code = current_code or fenced or line.startswith(("    ", "\t"))
    if current:
        blocks.append(_Block(current_end, _tokens(" ".join(current)), current_code))
    return blocks


def _score(item_tokens: frozenset[str], block_tokens: frozenset[str]) -> float:
    if not item_tokens or not block_tokens:
        return 0.0
    return len(item_tokens & block_tokens) / len(item_tokens)


def auto_place_rich_items(
    content: str,
    *,
    items: list[tuple[str, str, str]],
    max_images: int,
    min_score: float,
) -> tuple[str, list[str]]:
    """Insert markers for unreferenced items after their best-matching paragraph.

    ``items`` holds ``(item_id, item_type, descriptive_text)`` tuples in
    priority order. At most one item is placed per paragraph and at most
    ``max_images`` image items overall. Items already referenced in
    ``content`` or scoring below ``min_score`` are skipped. Returns
    ``(new_content, placed_ids)``; content is returned unchanged when nothing
    places.
    """
    if not content or not items:
        return content, []
    referenced = set(parse_inline_rich_references(content))
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    blocks = [b for b in _segment_blocks(lines) if not b.is_code]
    if not blocks:
        return content, []

    insertions: dict[int, str] = {}
    placed: list[str] = []
    images_placed = 0
    for item_id, item_type, text in items:
        if item_id in referenced:
            continue
        is_image = item_type == RichItemType.image.value
        if is_image and images_placed >= max_images:
            continue
        item_tokens = _tokens(text or "")
        best_line, best = -1, 0.0
        for block in blocks:
            if block.end_line in insertions:
                continue
            score = _score(item_tokens, block.tokens)
            if score > best:
                best_line, best = block.end_line, score
        if best_line < 0 or best < min_score:
            continue
        insertions[best_line] = f"<!--rich:{item_id}-->"
        placed.append(item_id)
        if is_image:
            images_placed += 1

    if not placed:
        return content, []

    out: list[str] = []
    for idx, line in enumerate(lines):
        out.append(line)
        marker = insertions.get(idx)
        if marker is not None:
            out.append("")
            out.append(marker)
    return "\n".join(out), placed
```

Note: `_strip_fenced_code_blocks` is module-private to `rich_response` but reused here deliberately (same package, same CommonMark fence semantics) — duplicating fence parsing would be the worse sin. If the linter objects, add `# noqa` with this justification.

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rich_placement.py -v`
Expected: 10 PASSED

- [x] **Step 5: Commit**

```bash
git add app/core/rich_placement.py tests/test_rich_placement.py
git commit -m "feat: deterministic article-style auto-placement engine for rich items"
```

### Task 3: Persistence-boundary integration (`finalize_article_content`)

**Files:**
- Modify: `app/core/rich_placement.py` (append)
- Test: `tests/test_rich_placement.py` (append)

- [x] **Step 1: Write the failing tests** (append to `tests/test_rich_placement.py`)

```python
from types import SimpleNamespace

from app.core.config import settings
from app.core.rich_placement import finalize_article_content


def _make_response(content, *, candidates=None, artifacts=None, capable=True):
    metadata = {"_inline_rich_response_v1": capable}
    if candidates is not None:
        metadata["_rich_item_candidates"] = candidates
    if artifacts is not None:
        metadata["tool_artifacts"] = artifacts
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        metadata=metadata,
        tool_artifacts=None,
    )


def _image_candidate(item_id="image:tool:c1:0", description="Eiffel Tower at night in Paris"):
    return {
        "id": item_id,
        "type": "image",
        "display_policy": "inline_only",
        "alt_text": description,
        "payload": {"url": "https://example.com/eiffel.jpg", "mime_type": "image/jpeg",
                    "description": description},
    }


def test_finalize_places_image_and_mutates_response_message(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    content = "The Eiffel Tower is stunning at night, lit by thousands of lamps."
    response = _make_response(content, candidates=[_image_candidate()])
    new_content = finalize_article_content(response, content)
    assert "<!--rich:image:tool:c1:0-->" in new_content
    assert response.message.content == new_content


def test_finalize_noop_when_feature_disabled(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", False)
    content = "The Eiffel Tower is stunning at night."
    response = _make_response(content, candidates=[_image_candidate()])
    assert finalize_article_content(response, content) == content


def test_finalize_noop_when_auto_place_disabled(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", False)
    content = "The Eiffel Tower is stunning at night."
    response = _make_response(content, candidates=[_image_candidate()])
    assert finalize_article_content(response, content) == content


def test_finalize_noop_without_capability(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    content = "The Eiffel Tower is stunning at night."
    response = _make_response(content, candidates=[_image_candidate()], capable=False)
    assert finalize_article_content(response, content) == content


def test_finalize_places_widget_from_artifacts(monkeypatch):
    import json

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    artifact = {
        "tool": "widget_create",
        "status": "success",
        "output": json.dumps(
            {"widget_id": "w1", "session_id": "s1", "widget_type": "chart",
             "title": "Quarterly revenue comparison", "status": "active", "version": 1}
        ),
    }
    content = "Here is the quarterly revenue comparison between both units."
    response = _make_response(content, artifacts=[artifact])
    new_content = finalize_article_content(response, content)
    assert "<!--rich:widget:w1-->" in new_content


def test_finalize_handles_none_response():
    assert finalize_article_content(None, "text") == "text"
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rich_placement.py -v -k finalize`
Expected: FAIL with `ImportError: cannot import name 'finalize_article_content'`

- [x] **Step 3: Implement** (append to `app/core/rich_placement.py`)

```python
def _widget_placement_entries(
    metadata: dict[str, Any], response_artifacts: list[dict[str, Any]] | None
) -> list[tuple[str, str, str]]:
    artifacts: list[dict[str, Any]] = []
    meta_artifacts = metadata.get("tool_artifacts")
    if isinstance(meta_artifacts, list):
        artifacts.extend(a for a in meta_artifacts if isinstance(a, dict))
    if response_artifacts:
        artifacts.extend(a for a in response_artifacts if isinstance(a, dict))
    entries: list[tuple[str, str, str]] = []
    for widget in extract_live_widgets_from_artifacts(artifacts):
        widget_id = widget.get("widget_id")
        if not widget_id:
            continue
        text = " ".join(
            str(part) for part in (widget.get("title"), widget.get("widget_type")) if part
        )
        entries.append((f"widget:{widget_id}", RichItemType.live_widget.value, text))
    return entries


def _image_placement_entries(metadata: dict[str, Any]) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for candidate in metadata.get("_rich_item_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("type") != RichItemType.image.value:
            continue
        item_id = candidate.get("id")
        if not isinstance(item_id, str) or not item_id:
            continue
        payload = candidate.get("payload")
        description = payload.get("description") if isinstance(payload, dict) else None
        text = " ".join(
            str(part)
            for part in (candidate.get("title"), candidate.get("alt_text"), description)
            if part
        )
        entries.append((item_id, RichItemType.image.value, text))
    return entries


def finalize_article_content(response: Any, content: str) -> str:
    """Apply article-style auto-placement to the final assistant markdown.

    Mutates ``response.message.content`` to the placed content so
    ``build_bot_metadata()`` resolves the exact marker set the persisted
    message carries. Returns the (possibly updated) content. No-op unless the
    inline rich-response feature and auto-placement are enabled and the
    response advertised the per-turn capability.
    """
    if not content or response is None:
        return content
    if not getattr(settings, "inline_rich_response_enabled", False):
        return content
    if not getattr(settings, "rich_auto_place_enabled", False):
        return content
    metadata = getattr(response, "metadata", None)
    if not isinstance(metadata, dict) or not metadata.get("_inline_rich_response_v1"):
        return content

    items = _widget_placement_entries(metadata, getattr(response, "tool_artifacts", None))
    items.extend(_image_placement_entries(metadata))
    if not items:
        return content

    new_content, placed = auto_place_rich_items(
        content,
        items=items,
        max_images=settings.rich_auto_place_max_images,
        min_score=settings.rich_auto_place_min_score,
    )
    if not placed:
        return content

    message = getattr(response, "message", None)
    if message is not None and isinstance(getattr(message, "content", None), str):
        message.content = new_content
    return new_content
```

- [x] **Step 4: Run the full placement test file**

Run: `python -m pytest tests/test_rich_placement.py -v`
Expected: 16 PASSED

- [x] **Step 5: Commit**

```bash
git add app/core/rich_placement.py tests/test_rich_placement.py
git commit -m "feat: finalize_article_content integration for persistence boundary"
```

### Task 4: Wire placement into message persistence

**Files:**
- Modify: `app/services/message_service.py` (two sites: `resume_workflow` ~line 2023, `_persist_completed_workflow_response` ~line 2364)

- [x] **Step 1: Add the import**

In `app/services/message_service.py`, next to the existing import of `build_bot_metadata` / `extract_response_content` (match the file's existing import style for `app.core.*`):

```python
from ..core.rich_placement import finalize_article_content
```

(If the file imports `from app.core....` absolute-style, use `from app.core.rich_placement import finalize_article_content` instead — match whichever form the neighboring imports use.)

- [x] **Step 2: Hook site 1 — `_persist_completed_workflow_response`**

Replace (currently ~lines 2364-2367):

```python
        bot_response_content = fix_markdown_code_blocks(
            extract_response_content(bot_response, fallback_content)
        )
        bot_metadata = build_bot_metadata(bot_response, sanitized_persona)
```

with:

```python
        bot_response_content = fix_markdown_code_blocks(
            extract_response_content(bot_response, fallback_content)
        )
        bot_response_content = finalize_article_content(bot_response, bot_response_content)
        bot_metadata = build_bot_metadata(bot_response, sanitized_persona)
```

- [x] **Step 3: Hook site 2 — `resume_workflow`**

Replace (currently ~lines 2023-2025):

```python
        bot_response_content = extract_response_content(bot_response, NO_RESPONSE_GENERATED)

        bot_metadata = build_bot_metadata(bot_response)
```

with:

```python
        bot_response_content = extract_response_content(bot_response, NO_RESPONSE_GENERATED)
        bot_response_content = finalize_article_content(bot_response, bot_response_content)

        bot_metadata = build_bot_metadata(bot_response)
```

- [x] **Step 4: Run the existing rich-response and message-service test files for regressions**

Run: `python -m pytest tests/test_rich_response_metadata.py tests/test_rich_response_streaming.py tests/test_message_history_pipeline.py tests/test_rich_placement.py -v`
Expected: ALL PASS (placement is additive: with no candidates or flag off it is a no-op)

- [x] **Step 5: Commit**

```bash
git add app/services/message_service.py
git commit -m "feat: auto-place inline rich markers at message persistence"
```

### Task 5: Shared media-capability prompt snippet

**Files:**
- Modify: `app/ai/prompts.py`
- Test: `tests/test_prompts_media_capability.py`

- [x] **Step 1: Write the failing test**

```python
"""Answer-producing prompts must state the inline media capability.

Guards against the model claiming it "cannot send images": every prompt that
produces user-facing answers carries one shared capability snippet.
"""

from app.ai import prompts


ANSWER_PROMPTS = (
    prompts.CHAT_SYSTEM_PROMPT,
    prompts.RAG_SYSTEM_PROMPT,
    prompts.AGENTIC_RAG_SYSTEM_PROMPT,
    prompts.SEARCH_SYSTEM_PROMPT,
    prompts.SEARCH_WITH_RESULTS_SYSTEM_PROMPT,
)


def test_snippet_defined_once_and_compact():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET
    assert "You CAN display images inline" in snippet
    assert len(snippet) < 1200, "media snippet must stay compact — do not bloat prompts"


def test_all_answer_prompts_carry_media_capability():
    for prompt in ANSWER_PROMPTS:
        assert "You CAN display images inline" in prompt
        assert "Never tell the user you cannot" in prompt


def test_non_answer_prompts_unchanged():
    for prompt in (
        prompts.ROUTER_SYSTEM_PROMPT,
        prompts.TITLE_GENERATION_PROMPT,
        prompts.PLANNING_EXECUTION_PROMPT,
        prompts.IMAGE_GENERATOR_SYSTEM_PROMPT,
    ):
        assert "You CAN display images inline" not in prompt
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prompts_media_capability.py -v`
Expected: FAIL with `AttributeError: module 'app.ai.prompts' has no attribute 'MEDIA_CAPABILITY_SNIPPET'`

- [x] **Step 3: Implement the snippet**

In `app/ai/prompts.py`, immediately after `INLINE_RICH_RESPONSE_SUFFIX` (after line 11), add:

```python
MEDIA_CAPABILITY_SNIPPET = """

Media and visuals:
- You CAN display images inline in your answers. When available rich items are listed for this turn, place a relevant image with its `<!--rich:<id>-->` marker on its own line near the paragraph it illustrates, followed by a short caption. Use only IDs from that list.
- Never tell the user you cannot send, show, or display images. If the user wants images and none are available yet, call a tool that returns images (such as web search) and then place the relevant results inline.
- Write answers like a well-edited article: weave images and widgets into the narrative where they support the text rather than bunching them at the end. Include only media that materially helps the reader."""
```

Then append the snippet to exactly five prompt constants by changing each closing `"""` to `""" + MEDIA_CAPABILITY_SNIPPET`:

1. `CHAT_SYSTEM_PROMPT` (ends `...suggest alternatives"""` → `...suggest alternatives""" + MEDIA_CAPABILITY_SNIPPET`)
2. `RAG_SYSTEM_PROMPT` (ends `...Match the user's language exactly"""`)
3. `AGENTIC_RAG_SYSTEM_PROMPT` (ends `...Match the user's language"""`)
4. `SEARCH_SYSTEM_PROMPT` (ends `...Match the user's language"""`)
5. `SEARCH_WITH_RESULTS_SYSTEM_PROMPT` (ends `...LANGUAGE: Match the user's language."""`)

Do NOT touch `IMAGE_GENERATOR_SYSTEM_PROMPT`, `ROUTER_SYSTEM_PROMPT`, `PLANNING_EXECUTION_PROMPT`, `TITLE_GENERATION_PROMPT`, `TOOL_EXPLORATION_SUFFIX`, `TOOL_CONTEXT_SUFFIX`, `DELEGATION_SUFFIX`, or `INLINE_RICH_RESPONSE_SUFFIX`.

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prompts_media_capability.py tests/test_rich_response_prompt_inventory.py -v`
Expected: ALL PASS

- [x] **Step 5: Commit**

```bash
git add app/ai/prompts.py tests/test_prompts_media_capability.py
git commit -m "feat: shared media-capability snippet in all answer prompts"
```

### Task 6: Enable inline rich responses by default

**Files:**
- Modify: `app/core/config.py:971-979`
- Modify: `tests/test_rich_response_contract.py:29`

- [x] **Step 1: Update the contract test first** (it pins the default)

In `tests/test_rich_response_contract.py` line 29, change:

```python
    assert Settings(_env_file=None).inline_rich_response_enabled is False
```

to:

```python
    assert Settings(_env_file=None).inline_rich_response_enabled is True
```

Also update the test's name/docstring if it says "disabled by default" — it now guards the enabled-by-default rollout.

- [x] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_rich_response_contract.py -v`
Expected: the flag-default test FAILS (default still False)

- [x] **Step 3: Flip the default**

In `app/core/config.py` change the field to:

```python
    inline_rich_response_enabled: bool = Field(
        default=True,
        description=(
            "Kill switch for the inline rich-response feature. When False, the "
            "backend never emits marker-bearing v1 content or rich-item stream "
            "events, regardless of the per-request capability."
        ),
    )
```

- [x] **Step 4: Run the full rich-response test set**

Run: `python -m pytest tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_rich_response_streaming.py tests/test_rich_response_prompt_inventory.py -v`
Expected: ALL PASS (other tests monkeypatch the flag explicitly, verified)

- [x] **Step 5: Commit**

```bash
git add app/core/config.py tests/test_rich_response_contract.py
git commit -m "feat: enable inline rich responses by default (flag becomes kill switch)"
```

### Task 7: End-to-end image flow test (tavily → candidates → placement → metadata)

**Files:**
- Create: `tests/test_article_image_flow.py`

- [x] **Step 1: Write the test**

```python
"""End-to-end guard for the image search → inline display path.

Covers the "cannot send images" complaint: a Tavily-shaped tool result must
produce image candidates that survive into the persisted message as inline
rich items even when the model writes no marker itself (auto-placement).
"""

import json
from types import SimpleNamespace

from app.ai.tool_execution import build_image_candidates_from_tool_result
from app.core.config import settings
from app.core.response_constants import build_bot_metadata
from app.core.rich_placement import finalize_article_content

TAVILY_RESULT = json.dumps(
    {
        "query": "eiffel tower at night",
        "answer": "The Eiffel Tower is lit nightly.",
        "images": [
            {
                "url": "https://example.com/eiffel.jpg",
                "description": "Eiffel Tower illuminated at night in Paris",
            }
        ],
        "results": [],
        "total_results": 0,
    }
)

ANSWER = (
    "The Eiffel Tower is stunning at night, illuminated by thousands of lamps "
    "across Paris.\n\nIt was completed in 1889 for the World's Fair."
)


def _workflow_response(content, candidates):
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        metadata={
            "_inline_rich_response_v1": True,
            "_rich_item_candidates": candidates,
        },
        tool_artifacts=None,
        error=None,
    )


def test_tavily_image_lands_inline_in_persisted_message(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)

    candidates = build_image_candidates_from_tool_result(
        TAVILY_RESULT, tool_call_id="call-1", tool_name="tavily_search"
    )
    assert candidates, "tavily-shaped results must produce image candidates"
    assert candidates[0]["id"] == "image:tool:call-1:0"

    response = _workflow_response(ANSWER, candidates)
    content = finalize_article_content(response, ANSWER)
    assert "<!--rich:image:tool:call-1:0-->" in content

    metadata = build_bot_metadata(response)
    rich_items = metadata.get("rich_items") or []
    image_items = [item for item in rich_items if item.get("type") == "image"]
    assert len(image_items) == 1
    assert image_items[0]["id"] == "image:tool:call-1:0"
    assert image_items[0]["payload"]["url"] == "https://example.com/eiffel.jpg"
    assert metadata.get("rich_reference_warnings") == []


def test_irrelevant_image_stays_dropped(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)

    irrelevant = json.dumps(
        {"images": [{"url": "https://example.com/cat.jpg",
                     "description": "A cat sleeping on a windowsill"}]}
    )
    candidates = build_image_candidates_from_tool_result(
        irrelevant, tool_call_id="call-2", tool_name="tavily_search"
    )
    response = _workflow_response(ANSWER, candidates)
    content = finalize_article_content(response, ANSWER)
    assert "<!--rich:" not in content

    metadata = build_bot_metadata(response)
    assert all(item.get("type") != "image" for item in metadata.get("rich_items") or [])
```

- [x] **Step 2: Run it**

Run: `python -m pytest tests/test_article_image_flow.py -v`
Expected: 2 PASSED (everything is already implemented by Tasks 2-4)

- [x] **Step 3: Commit**

```bash
git add tests/test_article_image_flow.py
git commit -m "test: end-to-end guard for image search to inline display path"
```

### Task 8: Postgres blob storage — migration + model

**Files:**
- Create: `app/alembic/versions/<rev>_add_content_to_tool_result_blobs.py`
- Modify: `app/models/tool_result_blob.py`

- [x] **Step 1: Find the current migration head**

Run: `python -m alembic heads` (alembic.ini is at the repo root)
Note the printed revision id — it is the `down_revision` for the new migration.

- [x] **Step 2: Generate the migration skeleton**

Run: `python -m alembic revision -m "add content to tool_result_blobs"`
This creates a correctly-chained file under `app/alembic/versions/`.

- [x] **Step 3: Fill in the migration**

Replace the generated `upgrade`/`downgrade` bodies with:

```python
def upgrade() -> None:
    op.add_column("tool_result_blobs", sa.Column("content", sa.Text(), nullable=True))
    op.alter_column(
        "tool_result_blobs",
        "storage_path",
        existing_type=sa.String(length=1024),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "tool_result_blobs",
        "storage_path",
        existing_type=sa.String(length=1024),
        nullable=False,
    )
    op.drop_column("tool_result_blobs", "content")
```

(`content` is nullable because legacy rows keep their payload on disk; `storage_path` becomes nullable because new rows have no file.)

- [x] **Step 4: Update the model**

In `app/models/tool_result_blob.py`:

1. Add `Text` to the SQLAlchemy import: `from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func`
2. Change `storage_path` to nullable and add `content` below it:

```python
    storage_path = Column(String(1024), nullable=True)
    content = Column(Text, nullable=True)
```

3. Update the class docstring to:

```python
    """
    Persistent record for large tool outputs that have been offloaded
    out of the model-visible ToolMessage.

    The full payload lives in the ``content`` column. ``storage_path`` is a
    legacy field: rows created before content moved to Postgres keep their
    payload on disk relative to the configured storage root. The model only
    sees a preview plus a blob_id pointer it can use later to read the full
    result.
    """
```

- [x] **Step 5: Apply and verify the migration**

Run: `python -m alembic upgrade head`
Expected: migration applies without error.
Run: `python -m alembic downgrade -1 && python -m alembic upgrade head`
Expected: clean round-trip (verifies downgrade works).

- [x] **Step 6: Commit**

```bash
git add app/alembic/versions app/models/tool_result_blob.py
git commit -m "feat: add content column to tool_result_blobs for DB-backed storage"
```

### Task 9: Postgres blob storage — service rewrite (TDD)

**Files:**
- Modify: `tests/test_tool_result_blob_service.py`
- Modify: `app/services/tool_result_blob_service.py`

- [x] **Step 1: Rewrite the tests to specify the new behavior**

Replace the body of `test_offload_if_large_writes_full_output_and_returns_preview` and add two tests, so the file becomes:

```python
from uuid import uuid4

import pytest

from app.services.tool_result_blob_service import ToolResultBlobService


class FakeRepository:
    def __init__(self):
        self.created = []

    def create(self, data):
        record = {"id": uuid4(), **data}
        self.created.append(record)
        return record


def test_offload_if_large_returns_inline_for_small_output(tmp_path):
    repo = FakeRepository()
    service = ToolResultBlobService(repo, storage_root=tmp_path, threshold_chars=20)

    result = service.offload_if_large(
        conversation_id=uuid4(),
        user_id=uuid4(),
        tool_call_id="call-1",
        tool_name="search_documents",
        output_text="short result",
    )

    assert result["output"] == "short result"
    assert result["blob_id"] is None
    assert repo.created == []


def test_offload_if_large_stores_content_in_db_and_creates_no_files(tmp_path):
    repo = FakeRepository()
    service = ToolResultBlobService(repo, storage_root=tmp_path, threshold_chars=10)
    conversation_id = uuid4()
    user_id = uuid4()

    result = service.offload_if_large(
        conversation_id=conversation_id,
        user_id=user_id,
        tool_call_id="call-1",
        tool_name="search_documents",
        output_text="abcdefghijklmnopqrstuvwxyz",
    )

    assert (
        result["output"] == "abcdefghij\n\n[Output offloaded: use blob_id to read the full result.]"
    )
    assert result["blob_id"]
    assert result["size_bytes"] == 26
    record = repo.created[0]
    assert record["conversation_id"] == conversation_id
    assert record["user_id"] == user_id
    assert record["tool_call_id"] == "call-1"
    assert record["content"] == "abcdefghijklmnopqrstuvwxyz"
    assert record["storage_path"] is None
    assert list(tmp_path.iterdir()) == [], "offload must not create files on disk"


def test_read_text_prefers_db_content(tmp_path):
    service = ToolResultBlobService(FakeRepository(), storage_root=tmp_path, threshold_chars=10)
    record = {"content": "full output", "storage_path": None}
    assert service.read_text(record) == "full output"


def test_read_text_falls_back_to_legacy_file(tmp_path):
    service = ToolResultBlobService(FakeRepository(), storage_root=tmp_path, threshold_chars=10)
    legacy_dir = tmp_path / "conv-1"
    legacy_dir.mkdir()
    (legacy_dir / "blob-1.txt").write_text("legacy payload", encoding="utf-8")
    record = {"id": "blob-1", "content": None, "storage_path": "conv-1/blob-1.txt"}
    assert service.read_text(record) == "legacy payload"


def test_read_text_raises_on_corrupt_record(tmp_path):
    service = ToolResultBlobService(FakeRepository(), storage_root=tmp_path, threshold_chars=10)
    record = {"id": "blob-x", "content": None, "storage_path": None}
    with pytest.raises(ValueError, match="blob-x"):
        service.read_text(record)
```

- [x] **Step 2: Run tests to verify the new ones fail**

Run: `python -m pytest tests/test_tool_result_blob_service.py -v`
Expected: `test_offload_if_large_stores_content_in_db_and_creates_no_files`, `test_read_text_prefers_db_content`, `test_read_text_raises_on_corrupt_record` FAIL; the small-output and legacy-file tests pass.

- [x] **Step 3: Rewrite the service**

Replace `app/services/tool_result_blob_service.py` with:

```python
"""Service for offloading large tool outputs to durable blob storage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


class ToolResultBlobService:
    """Persist full tool outputs out-of-band when they exceed a size threshold.

    New blobs store their content in the ``tool_result_blobs`` Postgres table.
    ``storage_root`` is retained only to read legacy records whose payload
    still lives on disk under ``storage_path``.
    """

    def __init__(
        self,
        repository,
        *,
        storage_root: str | Path,
        threshold_chars: int,
        preview_chars: int | None = None,
    ):
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.threshold_chars = max(1, int(threshold_chars))
        self.preview_chars = max(1, int(preview_chars or threshold_chars))

    def offload_if_large(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        tool_call_id: str | None,
        tool_name: str,
        output_text: str,
    ) -> dict[str, object]:
        if len(output_text) <= self.threshold_chars:
            return {
                "output": output_text,
                "blob_id": None,
                "size_bytes": len(output_text.encode("utf-8")),
            }

        encoded = output_text.encode("utf-8")
        record = self.repository.create(
            {
                "id": uuid4(),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": output_text,
                "storage_path": None,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size_bytes": len(encoded),
                "content_type": "text/plain",
            }
        )
        preview = output_text[: self.preview_chars].rstrip()
        record_id = record["id"] if isinstance(record, dict) else record.id
        return {
            "output": f"{preview}\n\n[Output offloaded: use blob_id to read the full result.]",
            "blob_id": str(record_id),
            "size_bytes": len(encoded),
        }

    def read_text(self, record: Any) -> str:
        content = record["content"] if isinstance(record, dict) else record.content
        if content is not None:
            return content
        storage_path = record["storage_path"] if isinstance(record, dict) else record.storage_path
        if not storage_path:
            record_id = record["id"] if isinstance(record, dict) else record.id
            raise ValueError(
                f"Tool result blob {record_id} has neither content nor storage_path; "
                "the record is corrupt and cannot be read."
            )
        return (self.storage_root / storage_path).read_text(encoding="utf-8")
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tool_result_blob_service.py -v`
Expected: 6 PASSED

- [x] **Step 5: Commit**

```bash
git add app/services/tool_result_blob_service.py tests/test_tool_result_blob_service.py
git commit -m "feat: store tool result blobs in Postgres, keep legacy file read fallback"
```

### Task 10: Config description, .gitignore, container sanity

**Files:**
- Modify: `app/core/config.py:658-661`
- Modify: `.gitignore`

- [x] **Step 1: Update the storage-dir setting description** (`app/core/config.py`)

```python
    tool_result_blob_storage_dir: str = Field(
        default="data/tool_result_blobs",
        description=(
            "Legacy read-only directory for blobs created before content moved "
            "to Postgres. New blobs are stored in the tool_result_blobs table."
        ),
    )
```

- [x] **Step 2: Ignore the legacy data directory**

Append to `.gitignore`:

```
# Legacy offloaded tool results (pre-Postgres blobs)
data/
```

Do NOT delete the existing `data/tool_result_blobs/` directory — legacy DB rows still point at those files via `storage_path`.

- [x] **Step 3: Confirm the DI wiring needs no change**

Read `app/core/container.py:252-258` — `tool_result_blob_service` passes `storage_root`, `threshold_chars`, `preview_chars`, which the rewritten service still accepts. No edit expected; verify only.

- [x] **Step 4: Verify git no longer reports the data dir**

Run: `git status --short`
Expected: `data/tool_result_blobs/` no longer listed as untracked.

- [x] **Step 5: Commit**

```bash
git add app/core/config.py .gitignore
git commit -m "chore: mark blob storage dir legacy and gitignore data dir"
```

### Task 11: Full verification

- [x] **Step 1: Full test suite**

Run: `python -m pytest -q`
Expected: ALL PASS, zero warnings introduced by this work. If any test asserts old blob/file behavior or prompt text outside the files already updated, fix it in the spirit of the spec (FR-1..FR-10) — do not weaken assertions.

- [x] **Step 2: Lint and types**

Run: `python -m ruff check app tests` (ruff is configured in `pyproject.toml`)
Expected: clean. Fix every finding; if `_strip_fenced_code_blocks` private-import triggers a rule, add an inline ignore with the justification comment from Task 2.

- [ ] **Step 3: Manual smoke (optional, requires running stack)**

Start the API + Streamlit demo (`demo.py`), ask the search agent a visual question (e.g. "What does the Eiffel Tower look like at night?"). Expected: the answer renders with an inline image near the relevant paragraph; no `data/tool_result_blobs/` files appear for large tool outputs; asking "can you send me images?" yields yes, not a refusal.

- [x] **Step 4: Final commit (if fixes were needed)**

```bash
git add -A
git commit -m "test: verification fixes for article-style rich responses"
```

---

## Risks & mitigations

- **Streamed text vs persisted text divergence:** deltas stream without auto-placed markers; the terminal `complete` event delivers the persisted message (`message_service.py:1146-1152`), which the demo already uses for finalized rendering (`app/ui/rich_response.py` `replace_with_finalized`). No protocol change; verify in Task 11 smoke.
- **Flag flip breaking clients without the capability:** safe — the per-request `inline_rich_response_v1` capability still gates everything (`graph.py:511`, `ai_sdk.py:920-921`); flag becomes AND-condition kill switch.
- **Auto-placement misplacing an image:** bounded by `min_score` relevance gate, per-paragraph uniqueness, and `max_images` cap; all tunable via settings without code change.
- **Legacy blob files:** preserved on disk and readable via fallback; only new blobs are DB-backed.

---

## Implementation log (progress & design decisions)

- **2026-06-12 — Task 1 done** (`feat: add auto-placement settings for inline rich items`). Three settings added verbatim from plan after `rich_item_selected_image_max_bytes`; verified `True 3 0.25` via settings import. No deviations.
- **2026-06-12 — Task 2 done** (`feat: deterministic article-style auto-placement engine for rich items`). TDD: 10 tests written first (failed with ModuleNotFoundError), then `app/core/rich_placement.py` created; 10/10 pass, ruff clean. **Decisions:** (1) `_strip_fenced_code_blocks` verified to have exactly the assumed list[str]→list[bool] semantics — reused as-is with `# noqa: PLC2701` (deliberate same-package reuse of CommonMark fence semantics). (2) Imports trimmed to what Task 2 actually uses (`settings`, `extract_live_widgets_from_artifacts`, `Any` deferred to Task 3 which needs them) to keep ruff clean per-commit. (3) One over-length assertion in `test_empty_inputs_are_safe` split across lines for the 100-char limit.
- **2026-06-12 — Task 3 done** (`feat: finalize_article_content integration for persistence boundary`). TDD: 6 finalize tests appended (failed with ImportError first), then `_widget_placement_entries` / `_image_placement_entries` / `finalize_article_content` appended verbatim from plan; 16/16 pass, ruff clean. **Decisions:** (1) `extract_live_widgets_from_artifacts` verified to accept `tool`/`tool_name` + JSON `output` artifact dicts exactly as the plan's test assumes — no adaptation. (2) No import cycle: `response_constants` does not import `rich_placement`. (3) Test-file imports hoisted to module top (E402-clean) instead of mid-file as the plan snippet showed.
- **2026-06-12 — Task 4 done** (`feat: auto-place inline rich markers at message persistence`). Import added absolute-style (`from app.core.rich_placement import ...`, matching neighbors, line 29); hooks inserted in `resume_workflow` (line 2025) and `_persist_completed_workflow_response` (line 2369), both between content extraction and `build_bot_metadata`. Regression run: 48 passed (only pre-existing langchain-community deprecation warning). No deviations.
- **2026-06-12 — Task 5 done** (`feat: shared media-capability snippet in all answer prompts`). Snippet added verbatim after `INLINE_RICH_RESPONSE_SUFFIX`; appended via `""" + MEDIA_CAPABILITY_SNIPPET` to exactly the 5 answer prompts; 12 tests pass (3 new + 9 inventory). **Decision:** `app/ai/prompts.py` had 73 pre-existing E501 (line-too-long) findings — all prompt prose. Added file-level `# ruff: noqa: E501` with justification comment (reflowing model-facing text harms readability) instead of reflowing; file is now ruff-clean.
- **2026-06-12 — Task 6 done** (`feat: enable inline rich responses by default (flag becomes kill switch)`). Contract test flipped first (confirmed red), then default flipped with kill-switch description; contract test renamed to `test_inline_rich_response_rollout_is_enabled_by_default_kill_switch`. Full rich-response set: 61 passed. No other test relied implicitly on the False default.
- **2026-06-12 — Task 7 done** (`test: end-to-end guard for image search to inline display path`). Both E2E tests passed on first run with zero adaptations — production interfaces (`build_image_candidates_from_tool_result`, `finalize_article_content`, `build_bot_metadata`) matched the plan's assumptions exactly. Acceptance criterion 1 now automated.
- **2026-06-12 — Task 8 done** (`feat: add content column to tool_result_blobs for DB-backed storage`). Migration `f03e63aa5a33` (down_revision `u7v8w9x0y1z2`) adds nullable `content` TEXT and makes `storage_path` nullable; model + docstring updated. Upgrade/downgrade round-trip verified against live Postgres; schema independently confirmed (`content TEXT NULL`, `storage_path VARCHAR(1024) NULL`). A version-table stamp hiccup during first apply was resolved with `stamp u7v8w9x0y1z2` + re-upgrade; final state clean at head.
- **2026-06-12 — Task 9 done** (`feat: store tool result blobs in Postgres, keep legacy file read fallback`). TDD: 3 new tests red first (KeyError 'content' / TypeError on None path), then service rewritten per plan; 5/5 pass, `-k blob` sweep green, ruff clean. **Verified against current code before rewrite:** offload-notice text, sync dict-based `repository.create`, and service-side `uuid4()` id generation all matched the plan snippet — no contract drift; constructor signature unchanged so DI container needs no edit.
- **2026-06-12 — Task 10 done** (`chore: mark blob storage dir legacy and gitignore data dir`). Description updated, `data/` gitignored (no longer untracked), DI wiring at `container.py:252-258` confirmed compatible (verify-only), container import sanity-checked. Legacy files left on disk for `storage_path` fallback.
- **2026-06-12 — Task 11 done (verification + review).**
  - **Full suite:** 1104 passed. One pre-existing failure excluded: `tests/client_backend/test_live_server_integration.py::test_live_document_upload_list_get_task_and_delete_flow` fails with `KeyError: 'document'` against the live server on :8000 — reproduced identically at base commit `37fb0ca` in a clean worktree, so environmental, not caused by this work. (Side finding: `shared/skills/` is needed by tests but untracked in git.)
  - **Ruff:** repo-wide baseline was NOT clean before this work (235 findings at base; 200 E501 prose). Now 161 — net −74; zero findings on lines added by this work (the one introduced 101-char docstring was fixed). Clearing the remaining pre-existing findings is out of scope for this plan.
  - **Final code review (whole branch):** image-marker injection via tool output confirmed not exploitable (ids are server-constructed); content/metadata consistency verified at both persist sites. Two Important findings fixed in `fix: split blocks at fence boundaries and validate ids before marker insert`: (I-1) `_segment_blocks` now splits blocks at fence on/off boundaries so prose adjacent to a fence (a shape `fix_markdown_code_blocks` itself produces) stays matchable; (I-2) `auto_place_rich_items` validates ids against `_ITEM_ID_PATTERN`/`RICH_ITEM_ID_MAX_LENGTH` before inserting, keeping inserter and parser symmetric. Minor M-3 addressed with a downgrade comment in the migration. Minors M-1/M-2 (whitespace-only divergence on no-op; CRLF→LF normalization when placing) accepted as harmless.
  - **Step 3 manual smoke:** skipped — requires restarting the running stack; the automated E2E test (`tests/test_article_image_flow.py`) covers the chain, and restarting the user's live server was out of bounds.
