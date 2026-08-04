# General Rich Image Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Select a small, deterministic set of relevant, usable images before prompt construction so dedicated image-search results outrank incidental page assets, while invalid or low-confidence results produce a text-only answer.

**Architecture:** Add a pure selector in app/core and a thin runtime adapter in app/ai. The selector ranks the complete candidate pool by intent, quality, relevance, and stable provider rank; normalizes groups; deduplicates locators; and applies the existing image cap. The adapter writes this canonical sequence back to context["rich_item_candidates"], records aggregate timing, and fails closed by retaining non-image rich items only. Prompt construction, placement, and metadata consume the same sequence without independently selecting images.

**Tech Stack:** Python 3.12, Pydantic response schemas, LangGraph workflow state, Prometheus metrics, pytest, Ruff.

## Global Constraints

- Do not add an LLM, embedding, image download, HEAD request, database, or cache call to the normal selection path.
- Do not hardcode subjects such as T1, Faker, Korea, people, flags, maps, logos, or diagrams.
- Keep legitimate high-quality flags, logos, maps, portraits, screenshots, and diagrams eligible.
- Treat dedicated image search as stronger intent than images incidentally returned by web-page search.
- Exclude query-level Tavily assets that are not tied to a source result.
- Apply settings.rich_auto_place_max_images only after ranking and deduplication.
- Preserve non-image rich candidates and current generated-image, document/RAG-image, and user-attachment behavior.
- Keep rejection and ranking facts in memory. Persist only selected rich items and aggregate content-free metrics.
- Do not add automatic image retry in this release. If no candidate passes, continue with text only.
- Do not add a migration, dependency, feature flag, or setting.
- Leave superpowers-main.zip and unrelated user changes untouched.

## File Map

| File | Responsibility |
|---|---|
| app/core/rich_image_selection.py | Pure policy, eligibility, ranking, group normalization, and deduplication. |
| app/ai/rich_image_selection.py | Settings-backed adapter, fail-closed behavior, and timing. |
| app/ai/tool_execution.py | Provider candidate construction and suppression of raw Tavily/Brave legacy images. |
| app/ai/workflow/tool_loop.py | Merge a completed tool batch, then run canonical selection. |
| app/ai/rag_tool_actions.py | Reapply selection after direct document-image registration. |
| app/ai/prompts.py | Consume canonical order; retain only a defensive cap. |
| app/core/rich_placement.py | Consume canonical IDs and remove invented/unselected image markers. |
| app/core/response_constants.py | Prevent typed candidates from duplicating into metadata.images. |
| app/observability/rich_images.py | Aggregate selector-duration telemetry. |
| tests/test_rich_image_selection.py | Pure unit and trace-regression tests. |
| tests/test_rich_image_selection_runtime.py | Workflow adapter and failure-path tests. |
| Existing rich response tests | Pipeline, metadata, placement, and compatibility regressions. |
| docs/frontend/rich-image-rendering.md | Canonical image contract and legacy-gallery separation. |

---

### Task 1: Build deterministic intent and relevance selection

**Files:**

- Create: app/core/rich_image_selection.py
- Create: tests/test_rich_image_selection.py
- Reference: app/core/config.py:1545-1630
- Reference: app/ai/tool_execution.py:153
- Reference: app/core/rich_response.py:727

**Interfaces:**

- Input: a sequence of rich-candidate mappings and an immutable ImageSelectionPolicy.
- Output: a new list with non-image candidates in stable order and at most max_items images/groups.
- Side effects: none. Do not mutate input, read settings, emit metrics, or perform I/O.

- [ ] **Step 1: Write failing trace-regression and intent tests**

Add these fixtures and assertions:

    from copy import deepcopy
    from dataclasses import replace

    from app.core.rich_image_selection import (
        ImageSelectionPolicy,
        select_rich_item_candidates,
    )

    POLICY = ImageSelectionPolicy(
        max_items=2,
        min_width_px=320,
        min_height_px=180,
        min_aspect_ratio=0.2,
        max_aspect_ratio=5.0,
    )

    def _image(
        item_id: str,
        *,
        source: str,
        url: str,
        query: str = "T1 roster",
        result_rank: int = 0,
        width: int | None = 1200,
        height: int | None = 800,
        source_url: str | None = None,
        description: str = "T1 roster players",
        query_level: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "url": url,
            "description": description,
            "width": width,
            "height": height,
        }
        if source_url is not None:
            payload["source_url"] = source_url
        return {
            "id": item_id,
            "type": "image",
            "source": source,
            "payload": payload,
            "provenance": {
                "provider": "test",
                "query": query,
                "result_rank": result_rank,
                "query_level": query_level,
            },
        }

    def test_dedicated_search_outranks_eight_source_bound_web_images() -> None:
        tavily = [
            _image(
                f"image:tavily:{index}",
                source="web_search",
                url=f"https://pages.example/assets/{index}.jpg",
                result_rank=index,
                source_url=f"https://pages.example/article/{index}",
                description="team article illustration",
            )
            for index in range(8)
        ]
        brave = _image(
            "image:brave:0",
            source="image_search",
            url="https://media.example/t1-roster.jpg",
            description="T1 roster players Faker Zeus Oner Gumayusi Keria",
        )
        original = deepcopy([*tavily, brave])

        selected = select_rich_item_candidates(original, policy=POLICY)

        assert [item["id"] for item in selected] == [
            "image:brave:0",
            "image:tavily:0",
        ]
        assert original == [*tavily, brave]

    def test_query_level_web_asset_is_excluded() -> None:
        candidate = _image(
            "image:tavily:query",
            source="web_search",
            url="https://search.example/generic.png",
            query_level=True,
        )
        assert select_rich_item_candidates([candidate], policy=POLICY) == []

    def test_direct_document_image_remains_intentional() -> None:
        candidates = [
            _image(
                "image:web:0",
                source="image_search",
                url="https://media.example/result.jpg",
            ),
            _image(
                "image:document:0",
                source="rag_document",
                url="https://files.example/figure-3.png",
                description="figure from the cited document",
            ),
        ]
        selected = select_rich_item_candidates(
            candidates,
            policy=replace(POLICY, max_items=1),
        )
        assert [item["id"] for item in selected] == ["image:document:0"]

    def test_non_image_items_are_preserved_in_original_order() -> None:
        widget = {"id": "widget:weather:0", "type": "weather", "payload": {}}
        selected = select_rich_item_candidates(
            [
                widget,
                _image(
                    "image:web:0",
                    source="image_search",
                    url="https://x.example/a.jpg",
                ),
            ],
            policy=replace(POLICY, max_items=0),
        )
        assert selected == [widget]

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run:

    .venv\Scripts\python.exe -m pytest tests/test_rich_image_selection.py -q

Expected: collection fails with ModuleNotFoundError for app.core.rich_image_selection.

- [ ] **Step 3: Implement the minimal pure selector**

Create these public contracts:

    @dataclass(frozen=True, slots=True)
    class ImageSelectionPolicy:
        max_items: int
        min_width_px: int
        min_height_px: int
        min_aspect_ratio: float
        max_aspect_ratio: float

    def is_image_candidate(candidate: Mapping[str, Any]) -> bool:
        return str(candidate.get("type") or "") in {"image", "image_group"}

    def select_rich_item_candidates(
        candidates: Sequence[Mapping[str, Any]],
        *,
        policy: ImageSelectionPolicy,
    ) -> list[dict[str, Any]]:
        copied = [deepcopy(dict(candidate)) for candidate in candidates]
        non_images = [
            candidate for candidate in copied if not is_image_candidate(candidate)
        ]
        indexed_images = [
            (index, candidate)
            for index, candidate in enumerate(copied)
            if is_image_candidate(candidate) and _intent_rank(candidate) is not None
        ]
        indexed_images.sort(key=_rank_key)
        return [
            *non_images,
            *(candidate for _, candidate in indexed_images[: policy.max_items]),
        ]

Implement _intent_rank with this generic order:

1. rag_document, tool_image, and generated_image;
2. image_search;
3. source-bound web_search with payload.source_url;
4. reject query-level or unbound web_search.

Implement normalized Unicode token overlap using provenance.query against title, alt_text, payload.description, and provenance.source_title. Use provider result_rank and original position as stable final tie-breakers. Do not inspect the subject or write ranking data back to candidates.

- [ ] **Step 4: Run tests**

    .venv\Scripts\python.exe -m pytest tests/test_rich_image_selection.py -q

Expected: Task 1 tests pass.

- [ ] **Step 5: Commit**

    git add app/core/rich_image_selection.py tests/test_rich_image_selection.py
    git commit -m "feat: add deterministic rich image selector"

---

### Task 2: Add quality gates, group normalization, and deduplication

**Files:**

- Modify: app/core/rich_image_selection.py
- Modify: app/ai/tool_execution.py:43-210
- Modify: tests/test_rich_image_selection.py
- Modify: tests/test_rich_response_sources.py

**Interfaces:**

- image_aspect_ratio_ok retains its current width, height, minimum, and maximum signature so existing callers remain compatible.
- is_junk_image_url rejects structural junk without rejecting an image category.
- A group with two or more survivors stays a group; one becomes an image; zero is removed.
- Exact display/original image locators are globally deduplicated in rank order.

- [ ] **Step 1: Add failing edge-case tests**

Add individual tests for:

    def test_known_tiny_image_is_rejected() -> None:
        candidate = _image(
            "image:tiny",
            source="image_search",
            url="https://media.example/tiny.png",
            width=45,
            height=30,
        )
        assert select_rich_item_candidates([candidate], policy=POLICY) == []

    def test_tiny_resize_url_is_rejected_when_dimensions_are_unknown() -> None:
        candidate = _image(
            "image:tiny-url",
            source="image_search",
            url="https://cdn.example/scale-to-width-down/45/logo.png",
            width=None,
            height=None,
        )
        assert select_rich_item_candidates([candidate], policy=POLICY) == []

    def test_unknown_dimensions_rank_after_known_usable_dimensions() -> None:
        unknown = _image(
            "image:unknown",
            source="image_search",
            url="https://media.example/unknown.jpg",
            width=None,
            height=None,
        )
        known = _image(
            "image:known",
            source="image_search",
            url="https://media.example/known.jpg",
            result_rank=1,
        )
        selected = select_rich_item_candidates(
            [unknown, known],
            policy=replace(POLICY, max_items=1),
        )
        assert [item["id"] for item in selected] == ["image:known"]

    @pytest.mark.parametrize(
        ("item_id", "description"),
        [
            ("image:flag", "high-resolution national flag"),
            ("image:logo", "high-resolution organization logo"),
            ("image:portrait", "official portrait"),
            ("image:map", "regional map"),
            ("image:diagram", "system architecture diagram"),
        ],
    )
    def test_content_category_alone_does_not_reject(
        item_id: str,
        description: str,
    ) -> None:
        candidate = _image(
            item_id,
            source="image_search",
            url=f"https://media.example/{item_id.removeprefix('image:')}.png",
            description=description,
        )
        selected = select_rich_item_candidates([candidate], policy=POLICY)
        assert [item["id"] for item in selected] == [item_id]

Also cover:

- duplicate standalone/group locators;
- blank, malformed, non-HTTP(S), tracking-pixel, sprite, and explicit placeholder URLs;
- inclusive aspect-ratio boundaries;
- a two-cell surviving group retaining cell order;
- a one-cell group collapsing to type image while retaining ID and provenance;
- a zero-cell group being removed;
- max_items=0 preserving only non-image candidates;
- repeated calls returning equivalent dictionaries.

- [ ] **Step 2: Run and observe the new failures**

    .venv\Scripts\python.exe -m pytest tests/test_rich_image_selection.py tests/test_rich_response_sources.py -q

Expected: new quality, group, and deduplication tests fail while Task 1 remains green.

- [ ] **Step 3: Move generic quality helpers into the core selector**

Move image_aspect_ratio_ok and is_junk_image_url out of tool_execution. Import them back into app/ai/tool_execution.py so existing imports remain compatible:

    from ..core.rich_image_selection import (
        image_aspect_ratio_ok,
        is_junk_image_url,
    )

Use only explicit structural junk patterns. Replace a broad match for any avatar path with explicit placeholder forms such as default-avatar, avatar-placeholder, or /avatars/default.

Normalize the existing single-result Brave source while making this change: emit source image_search when is_brave is true, web_search for Tavily, and tool_image for other typed-image tools. Update the existing Brave identity assertion accordingly. The selector applies remote URL and dimension gates to web_search and image_search candidates; rag_document, generated_image, and other direct tool_image candidates retain their already-validated data/local-reference behavior.

Extract bounded URL size hints for forms already observed in provider URLs:

    _URL_WIDTH_PATTERNS = (
        re.compile(
            r"/scale-to-width-down/(?P<width>\d{1,5})(?:[/?.]|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:[?&])(?:w|width)=(?P<width>\d{1,5})(?:&|$)",
            re.IGNORECASE,
        ),
    )
    _URL_HEIGHT_PATTERNS = (
        re.compile(
            r"(?:[?&])(?:h|height)=(?P<height>\d{1,5})(?:&|$)",
            re.IGNORECASE,
        ),
    )
    _URL_SIZE_PATTERN = re.compile(
        r"(?:^|[./_-])(?P<width>\d{1,5})x"
        r"(?P<height>\d{1,5})(?:[./_-]|$)",
        re.IGNORECASE,
    )

A known width or height below its corresponding minimum is a hard rejection. Otherwise unknown dimensions remain eligible but rank behind known usable dimensions.

- [ ] **Step 4: Normalize and deduplicate after ranking**

Rank prepared candidates by:

1. intent tier;
2. known usable dimensions before unknown dimensions;
3. greater normalized description/query overlap;
4. provider result_rank;
5. original input position.

Then accept locators in ranked order:

    def _select_ranked_images(
        indexed_candidates: list[tuple[int, dict[str, Any]]],
        policy: ImageSelectionPolicy,
    ) -> list[dict[str, Any]]:
        indexed_candidates.sort(
            key=lambda indexed: _rank_key(indexed, policy=policy)
        )
        selected: list[dict[str, Any]] = []
        seen_locators: set[str] = set()
        for _, candidate in indexed_candidates:
            normalized = _normalize_candidate(
                candidate,
                policy=policy,
                seen_locators=seen_locators,
            )
            if normalized is None:
                continue
            selected.append(normalized)
            if len(selected) == policy.max_items:
                break
        return selected

Apply the same cell validator to standalone images and group cells. For one surviving cell, retain the group ID and provenance, set type to image, and replace payload with that cell. Add locators to seen_locators only after acceptance. Do not add rejection reasons or classification fields to output.

- [ ] **Step 5: Run focused tests and lint**

    .venv\Scripts\python.exe -m pytest tests/test_rich_image_selection.py tests/test_rich_response_sources.py -q
    .venv\Scripts\python.exe -m ruff check app/core/rich_image_selection.py app/ai/tool_execution.py tests/test_rich_image_selection.py tests/test_rich_response_sources.py

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit**

    git add app/core/rich_image_selection.py app/ai/tool_execution.py tests/test_rich_image_selection.py tests/test_rich_response_sources.py
    git commit -m "feat: normalize and filter rich image candidates"

---

### Task 3: Make the selected sequence canonical across tool and RAG ingestion

**Files:**

- Create: app/ai/rich_image_selection.py
- Create: tests/test_rich_image_selection_runtime.py
- Modify: app/ai/workflow/tool_loop.py:44-66
- Modify: app/ai/rag_tool_actions.py:184-220
- Modify: tests/test_graph_tool_budget.py
- Modify: tests/test_rich_response_prompt_inventory.py
- Modify: tests/test_rich_placement.py

**Interfaces:**

- apply_rich_image_selection(context) is the sole settings-aware mutation point.
- It replaces context["rich_item_candidates"] with the canonical sequence.
- On an unexpected exception, it logs and retains non-image candidates only.
- Reapplying after a later batch permits a stronger new candidate to replace a prior selection.

- [ ] **Step 1: Add failing runtime tests**

Cover complete parallel batches, sequential late candidates, direct document registration, and fail-closed behavior:

    def test_selector_failure_keeps_widget_and_drops_images(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        context = {
            "rich_item_candidates": [
                {
                    "id": "widget:weather:0",
                    "type": "weather",
                    "payload": {},
                },
                _web_image(0),
            ]
        }

        def _raise(*args: object, **kwargs: object) -> list[dict[str, object]]:
            raise RuntimeError("synthetic selector failure")

        monkeypatch.setattr(
            "app.ai.rich_image_selection.select_rich_item_candidates",
            _raise,
        )
        apply_rich_image_selection(context)

        assert [item["id"] for item in context["rich_item_candidates"]] == [
            "widget:weather:0"
        ]

For a parallel batch containing eight Tavily candidates and one Brave group, assert the Brave group is first and the image count does not exceed rich_auto_place_max_images. Apply two sequential lifts and assert a later image-search group replaces an earlier source-bound web result. Register a document image and assert it remains the strongest direct source.

Also assert prompt inventory and placement anchor inputs expose the same image IDs and order as the canonical context.

- [ ] **Step 2: Run and confirm concatenation still occurs**

    .venv\Scripts\python.exe -m pytest tests/test_rich_image_selection_runtime.py tests/test_graph_tool_budget.py tests/test_rich_response_prompt_inventory.py tests/test_rich_placement.py -q

Expected: new tests fail because candidate lifting concatenates and document registration bypasses selection.

- [ ] **Step 3: Add the settings-backed fail-closed adapter**

Create app/ai/rich_image_selection.py:

    def _configured_policy() -> ImageSelectionPolicy:
        return ImageSelectionPolicy(
            max_items=max(0, settings.rich_auto_place_max_images),
            min_width_px=settings.rich_image_min_width_px,
            min_height_px=settings.rich_image_min_height_px,
            min_aspect_ratio=settings.rich_image_min_aspect_ratio,
            max_aspect_ratio=settings.rich_image_max_aspect_ratio,
        )

    def apply_rich_image_selection(
        context: MutableMapping[str, Any],
    ) -> None:
        raw = context.get("rich_item_candidates")
        candidates = (
            [item for item in raw if isinstance(item, dict)]
            if isinstance(raw, list)
            else []
        )
        try:
            context["rich_item_candidates"] = select_rich_item_candidates(
                candidates,
                policy=_configured_policy(),
            )
        except Exception:
            logger.exception(
                "Rich image selection failed; continuing without discovered images"
            )
            context["rich_item_candidates"] = [
                item for item in candidates if not is_image_candidate(item)
            ]

The broad exception belongs only at this workflow boundary so a selector defect degrades to text instead of failing the answer.

- [ ] **Step 4: Invoke selection after every ingestion point**

In ToolLoopMixin._lift_rich_candidates, retain current ID deduplication and artifact cleanup, assign the combined batch, then call:

    context["rich_item_candidates"] = combined_candidates
    apply_rich_image_selection(context)

All parallel artifacts are already available at this point, so Tavily and Brave compete in one pool.

In register_document_image_candidates, append direct document candidates and call the same adapter before returning. Do not add another graph-state list.

- [ ] **Step 5: Keep downstream caps defensive only**

Update tests/comments around build_rich_response_guidance and _image_anchor_entries. These functions may retain a bounds check but must preserve input order and must not compute scores or a different shortlist.

- [ ] **Step 6: Run workflow tests and lint**

    .venv\Scripts\python.exe -m pytest tests/test_rich_image_selection_runtime.py tests/test_graph_tool_budget.py tests/test_rich_response_prompt_inventory.py tests/test_rich_placement.py tests/test_article_image_flow.py -q
    .venv\Scripts\python.exe -m ruff check app/ai/rich_image_selection.py app/ai/workflow/tool_loop.py app/ai/rag_tool_actions.py tests/test_rich_image_selection_runtime.py

Expected: all pass and the trace-shaped fixture selects the dedicated image-search group first.

- [ ] **Step 7: Commit**

    git add app/ai/rich_image_selection.py app/ai/workflow/tool_loop.py app/ai/rag_tool_actions.py tests/test_rich_image_selection_runtime.py tests/test_graph_tool_budget.py tests/test_rich_response_prompt_inventory.py tests/test_rich_placement.py
    git commit -m "feat: apply canonical image selection in workflows"

---

### Task 4: Remove the legacy web-image leak and invalid image markers

**Files:**

- Modify: app/ai/tool_execution.py:1792-1820
- Modify: app/core/response_constants.py:250-530
- Modify: app/core/rich_placement.py:447-560
- Modify: tests/test_rich_response_sources.py
- Modify: tests/test_rich_response_metadata.py
- Modify: tests/test_rich_placement.py
- Modify: tests/test_article_image_flow.py

**Interfaces:**

- Tavily and Brave discovered images travel only through typed rich candidates/rich_items in v1.
- Other tools' generated or content-block images retain the legacy result path.
- A legacy entry duplicating any typed candidate ID or locator is removed.
- An inline image marker absent from the selected inventory is removed before persistence.

- [ ] **Step 1: Add failing provider-leak tests**

Make a Tavily-style fake result with 114 images. Unpack execute_tool_calls as outputs, artifacts, images and assert:

    assert len(
        artifacts[0]["_rich_item_candidates"]
    ) <= settings.rich_image_candidate_max_count
    assert images == []

Add the equivalent Brave assertion. Execute a non-search fake tool returning a generated/content image and assert it remains in images.

In the article flow test, combine the 114-image Tavily result with a relevant Brave result and assert:

    assert response.metadata.get("images") in (None, [])
    assert response.metadata["rich_items"][0]["id"].startswith(
        "imagegroup:brave:"
    )

In metadata tests:

- change the selected-group legacy-gallery test to expect the duplicate entry to be stripped;
- retain a generated/local image with no matching candidate;
- retain a user-owned attachment reference;
- ensure an unselected typed candidate does not survive in metadata.images.

- [ ] **Step 2: Add failing image-marker integrity tests**

Exercise finalize_article_content with inline capability enabled and automatic placement both enabled and disabled:

    def test_unknown_image_marker_is_removed_when_auto_place_is_disabled(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "rich_auto_place_enabled", False)
        content = 'Answer\n\n:::rich{ref="image:not-selected"}'
        metadata = {
            "_rich_item_candidates": [
                _selected_image("image:selected")
            ]
        }

        finalized = finalize_article_content(
            content,
            metadata=metadata,
            capabilities=INLINE_CAPABILITIES,
        )

        assert "image:not-selected" not in finalized
        assert ":::rich" not in finalized

Also prove that selected image and image_group IDs remain and an unknown non-image widget marker retains current warning behavior.

- [ ] **Step 3: Stop raw provider arrays at tool execution**

Define:

    _TYPED_WEB_IMAGE_TOOLS = frozenset(
        {"tavily_search", "brave_image_search"}
    )

Continue constructing _rich_item_candidates, but guard legacy extraction:

    if capture_images and tool_name not in _TYPED_WEB_IMAGE_TOOLS:
        images.extend(extract_images_from_tool_result(result_text))
        images.extend(extract_images_from_tool_content(result))

Do not suppress arbitrary tools based on payload shape.

- [ ] **Step 4: Strip typed-candidate duplicates from legacy metadata**

Replace the selected-only legacy filter with a helper that builds the set of every candidate ID and every standalone/group-cell locator, then removes matching metadata.images entries. Call it before _rich_item_candidates is popped.

    if image_id in candidate_ids or locator in candidate_locators:
        continue
    retained.append(image)

Do not remove unrelated generated/local or user-owned entries.

- [ ] **Step 5: Remove invented/discarded image markers**

In rich_placement.py, derive allowed IDs from every canonical candidate of type image or image_group, not only anchorable entries:

    def _remove_unknown_image_markers(
        content: str,
        allowed_ids: frozenset[str],
    ) -> str:
        cleaned = content
        for reference in parse_inline_rich_references(content):
            if (
                reference.startswith(("image:", "imagegroup:"))
                and reference not in allowed_ids
            ):
                cleaned = remove_inline_rich_reference(
                    cleaned,
                    reference,
                )
        return cleaned

Run this integrity pass whenever inline rich response is enabled, before the early return for disabled automatic placement. Reuse existing parsing/removal helpers for whitespace normalization.

- [ ] **Step 6: Run leak and marker tests**

    .venv\Scripts\python.exe -m pytest tests/test_rich_response_sources.py tests/test_rich_response_metadata.py tests/test_rich_placement.py tests/test_article_image_flow.py tests/test_ai_sdk_context_window.py tests/test_demo_rich_response.py -q
    .venv\Scripts\python.exe -m ruff check app/ai/tool_execution.py app/core/response_constants.py app/core/rich_placement.py tests/test_rich_response_sources.py tests/test_rich_response_metadata.py tests/test_rich_placement.py tests/test_article_image_flow.py

Expected: the raw 114-image list never reaches metadata.images, the Brave group is selected, and unrelated generated/user-owned images remain.

- [ ] **Step 7: Commit**

    git add app/ai/tool_execution.py app/core/response_constants.py app/core/rich_placement.py tests/test_rich_response_sources.py tests/test_rich_response_metadata.py tests/test_rich_placement.py tests/test_article_image_flow.py
    git commit -m "fix: prevent unselected web images from leaking"

---

### Task 5: Add bounded telemetry, document the contract, and verify end to end

**Files:**

- Modify: app/observability/rich_images.py
- Modify: app/ai/rich_image_selection.py
- Modify: tests/test_rich_image_metrics.py
- Modify: tests/test_rich_image_selection_runtime.py
- Modify: docs/frontend/rich-image-rendering.md

**Interfaces:**

- Metric: rich_image_selector_duration_seconds with no labels containing query, URL, provider result, subject, candidate ID, or rejection reason.
- Record one observation per adapter invocation, including fail-closed execution.
- Document that discovered web images appear only as selected rich_items in v1.

- [ ] **Step 1: Add failing aggregate-metric tests**

Follow the current isolated CollectorRegistry pattern:

    def test_selector_duration_has_no_content_labels() -> None:
        metrics = RichImageMetrics(registry=CollectorRegistry())
        metrics.record_selection_duration(0.004)

        samples = list(metrics.selector_duration.collect())[0].samples
        count = next(
            sample
            for sample in samples
            if sample.name.endswith("_count")
        )
        assert count.value == 1
        assert count.labels == {}

Monkeypatch record_selection_duration in runtime tests. Assert one non-negative float is recorded on success and on a synthetic selector exception.

- [ ] **Step 2: Run and confirm the metric API is missing**

    .venv\Scripts\python.exe -m pytest tests/test_rich_image_metrics.py tests/test_rich_image_selection_runtime.py -q

Expected: new metric assertions fail.

- [ ] **Step 3: Implement timing without content labels**

Add to RichImageMetrics:

    self.selector_duration = Histogram(
        "rich_image_selector_duration_seconds",
        "Time spent selecting canonical rich image candidates",
        registry=registry,
    )

    def record_selection_duration(
        self,
        duration_seconds: float,
    ) -> None:
        self.selector_duration.observe(max(0.0, duration_seconds))

In apply_rich_image_selection, wrap selection/fail-closed handling with time.perf_counter and record in finally. Do not log queries, URLs, descriptions, IDs, or per-candidate decisions.

- [ ] **Step 4: Update frontend/persistence documentation**

Document in docs/frontend/rich-image-rendering.md:

- backend rich_items are already ranked and capped;
- frontend code renders referenced selected IDs and does not choose alternatives;
- Tavily/Brave raw arrays are not a v1 gallery fallback;
- no selected image is a valid text-only result;
- two or more cells remain a group and one becomes an image;
- generated images and user attachments keep existing paths;
- selector classifications are transient and absent from the public schema.

- [ ] **Step 5: Run focused and regression suites**

    .venv\Scripts\python.exe -m pytest tests/test_rich_image_metrics.py tests/test_rich_image_selection.py tests/test_rich_image_selection_runtime.py -q
    .venv\Scripts\python.exe -m pytest tests/test_rich_response_sources.py tests/test_rich_response_prompt_inventory.py tests/test_rich_placement.py tests/test_rich_response_metadata.py tests/test_article_image_flow.py tests/test_graph_tool_budget.py tests/test_tool_execution_rendering.py tests/test_ai_sdk_context_window.py tests/test_demo_rich_response.py -q
    .venv\Scripts\python.exe -m ruff check app/core/rich_image_selection.py app/ai/rich_image_selection.py app/ai/tool_execution.py app/ai/workflow/tool_loop.py app/ai/rag_tool_actions.py app/ai/prompts.py app/core/rich_placement.py app/core/response_constants.py app/observability/rich_images.py tests/test_rich_image_selection.py tests/test_rich_image_selection_runtime.py tests/test_rich_image_metrics.py
    git diff --check

Expected:

- all tests pass;
- Ruff reports no errors;
- git diff --check has no output;
- tests perform no network access;
- selector work remains in-process and should take sub-millisecond to low-single-digit milliseconds at configured bounds.

- [ ] **Step 6: Audit acceptance criteria**

Verify from tests and diff:

- dedicated search wins over eight earlier source-bound web candidates;
- known tiny images and unambiguous tiny resize URLs are rejected;
- unknown dimensions remain eligible but rank behind known usable dimensions;
- high-quality flags, logos, maps, portraits, and diagrams remain eligible;
- query-level Tavily assets are excluded;
- groups normalize correctly for two-plus, one, or zero cells;
- exact locator duplicates appear once;
- prompt, placement, and persisted metadata see the same selected IDs;
- a 114-entry Tavily raw array does not reach the legacy gallery;
- generated images, RAG images, and user attachments retain behavior;
- empty or failed selection yields a text-only answer;
- no model call, download, probe, retry, stored classification, or content-bearing metric was added.

- [ ] **Step 7: Commit**

    git add app/observability/rich_images.py app/ai/rich_image_selection.py tests/test_rich_image_metrics.py tests/test_rich_image_selection_runtime.py docs/frontend/rich-image-rendering.md
    git commit -m "docs: finalize rich image selection contract"

---

## Final Verification and Handoff

- [ ] Confirm the worktree contains only the five implementation commits plus pre-existing superpowers-main.zip.
- [ ] Record exact test counts and durations from the final run.
- [ ] Compare trace-shaped fixture selected IDs with its candidate inventory and include that evidence in the handoff.
- [ ] Report any skipped required test; do not claim full verification if a required local test failed.
