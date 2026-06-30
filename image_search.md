# Brave Image Search for Article-Style Answers

## Summary

Add Brave Image Search as a production-ready visual search MCP tool that feeds the existing inline rich-response pipeline. The goal is not to create a second renderer or a fixed rules engine. The model remains responsible for deciding when visual evidence helps, choosing which returned images support the answer, and writing the article-style response. The backend only provides a bounded, normalized image candidate channel and keeps deterministic placement as a fallback when the model omits markers.

User-approved direction:

- Add Brave Image Search as a global default MCP server, similar to Tavily.
- Pin the Brave image search tool for `chat_agent` and `search_agent`.
- Leave other agents able to discover it through `tool_search` when actually needed.
- Avoid hardcoded visual-topic taxonomies beyond compact prompt guidance and configurable safety/budget limits.
- Keep prompts short and capability-oriented.

## Current Code Context

Relevant existing paths:

- `app/ai/mcp_config.json` defines server-managed global MCP tools. Enabled servers are currently `widgets`, `tavily`, and `time`.
- `tests/test_mcp_global_allowlist.py` enforces the exact enabled global server set.
- `app/ai/mcp_integration.py::MCPManager.DEFAULT_SERVERS` reserves managed core MCP servers from add/remove operations; a new global default should be added there too.
- `app/ai/mcp_servers/tavily_server.py` already returns `images` in the shape consumed by rich responses.
- `app/ai/deferred_tool_binding.py` pins `time::get_current_time` and `tavily::tavily_search` for `search`, and widget tools for `chat`, `rag`, and `search`.
- `app/ai/deferred_tool_binding.py::get_pinned_tools()` currently slices pinned specs by `settings.mcp_tool_search_max_pinned_tools`. Search already has five required pins (`time`, `tavily`, and three widget tools), so adding Brave must not be silently dropped by the cap.
- `app/ai/tool_execution.py::build_image_candidates_from_tool_result()` converts tool JSON with an `images` array into `image:tool:<call_id>:<index>` rich-item candidates.
- `app/ai/prompts.py` contains the shared media snippet and rich marker guidance. This should be tightened, not expanded heavily.
- `app/core/rich_response.py::ImagePayload` accepts only `url` or `data`, `mime_type`, `source_url`, and `description`; extra image metadata belongs in `provenance` unless the schema is deliberately extended.
- `app/core/rich_placement.py::finalize_article_content()` auto-places relevant unreferenced image/widget markers at persistence time.
- `app/core/response_constants.py::build_bot_metadata()` keeps only selected/placed image rich items for v1 responses and scrubs hidden candidates from legacy image fields.

## Problem

Article-style answers with inline images currently depend mostly on Tavily search being invoked. Tavily is good for web research, but visual/showcase requests often need a dedicated image search provider even when the user is not asking for current web facts.

Examples:

- "Spain architecture"
- "What does a turbine blade look like?"
- "Show me examples of brutalist interiors"
- "Explain Gothic vs Romanesque architecture with images"

The missing piece is a faithful visual acquisition tool that returns compact, well-described image candidates without bloating the model context or forcing broad deterministic routing.

## Goals

1. Provide a first-class image-search tool backed by Brave Image Search.
2. Keep the LLM flexible: it decides when visual search helps and how to use image candidates in the narrative.
3. Keep context bounded: the model sees compact image summaries and stable IDs, not raw provider payloads or large image data.
4. Keep latency bounded: image search must not become an unconditional preflight call.
5. Reuse the existing rich-item pipeline, marker contract, and deterministic article placement fallback.
6. Keep prompt edits compact and generic.
7. Preserve safe behavior for non-rich clients and unselected images.

## Non-Goals

- Do not add a new frontend rendering contract.
- Do not make Brave Image Search run before every answer.
- Do not build a large hardcoded classifier for visual topics.
- Do not ask a second LLM call to rank or place images.
- Do not append unselected image galleries to v1 rich responses.
- Do not replace Tavily. Tavily remains the web/news/general research tool.

## Recommended Architecture

### Provider Tool

Create a new MCP server:

- File: `app/ai/mcp_servers/brave_image_search_server.py`
- Server name: `brave_image_search`
- Tool name: `brave_image_search`
- Transport: stdio
- HTTP client: `httpx`, already present in project dependencies
- Brave endpoint: `https://api.search.brave.com/res/v1/images/search`

Tool signature:

```python
def brave_image_search(
    query: str,
    count: int | None = None,
    country: str | None = None,
    search_lang: str | None = None,
    safesearch: str | None = None,
) -> str:
    ...
```

The tool should return normalized JSON, not raw Brave output. Brave Image Search distinguishes page URLs from direct image URLs, so normalize this way:

- Brave `result.properties.url` -> `images[].url` (direct renderable image URL; required)
- Brave `result.url` -> `images[].source_url` (page where the image was found)
- Brave `result.thumbnail.src` -> `images[].thumbnail_url`
- Brave `result.properties.width` / `height` -> `images[].width` / `height`
- Brave `result.title`, `result.source`, and `result.meta_url.hostname` -> compact descriptive/source text

Normalized result shape:

```json
{
  "query": "Spain architecture",
  "provider": "brave_image_search",
  "images": [
    {
      "url": "https://...",
      "thumbnail_url": "https://...",
      "source_url": "https://...",
      "title": "Sagrada Familia exterior",
      "description": "Sagrada Familia basilica in Barcelona with tall organic spires",
      "mime_type": "image/jpeg",
      "width": 1200,
      "height": 800,
      "provider": "brave_image_search"
    }
  ],
  "total_results": 6
}
```

Only `url` and descriptive fields are required by the existing rich pipeline. Extra fields are useful for renderer/source attribution later, but they should stay compact.

### Configuration

Add settings in `app/core/config.py`:

- `brave_search_api_key: str`
- `brave_image_search_default_count: int = 6`
- `brave_image_search_max_count: int = 10`
- `brave_image_search_timeout_seconds: float = 2.5`
- `brave_image_search_default_safesearch: str = "strict"`

The count and timeout are operational limits, not behavior hardcoding. They keep the tool production-safe while still letting the model choose query wording and whether to call the tool. Brave Image Search currently supports `safesearch` values `off` and `strict`; default to `strict`.

Update `.env.example`:

```env
BRAVE_SEARCH_API_KEY=
BRAVE_IMAGE_SEARCH_DEFAULT_COUNT=6
BRAVE_IMAGE_SEARCH_MAX_COUNT=10
BRAVE_IMAGE_SEARCH_TIMEOUT_SECONDS=2.5
BRAVE_IMAGE_SEARCH_DEFAULT_SAFESEARCH=strict
```

### MCP Registration

Update `app/ai/mcp_config.json`:

```json
"brave_image_search": {
  "enabled": true,
  "transport": "stdio",
  "command": "python",
  "args": ["app/ai/mcp_servers/brave_image_search_server.py"],
  "description": "Image search using Brave Search API for visual references and article-style inline image candidates"
}
```

Update `tests/test_mcp_global_allowlist.py` expected global defaults:

```python
GLOBAL_DEFAULT_SERVERS = {"widgets", "tavily", "time", "brave_image_search"}
```

Update `app/ai/mcp_integration.py`:

```python
DEFAULT_SERVERS = {"calculator", "tavily", "time", "widgets", "brave_image_search"}
```

This keeps Brave protected as a managed built-in server, matching Tavily/widgets behavior.

### Tool Binding

Pin Brave Image Search only where it has high value:

- `chat_agent`: pinned by default for visual/showcase answers that are not necessarily current.
- `search_agent`: pinned by default for current/web/news answers where a visual article layout helps.
- Other agents: not pinned, but discoverable through `tool_search`.

Implementation target:

- Modify `app/ai/deferred_tool_binding.py`.
- Add a pinned spec such as `brave_image_search::brave_image_search`.
- Include it when `agent_key in {"chat", "search"}`.
- Refactor pin selection so required agent pins cannot be dropped by `mcp_tool_search_max_pinned_tools`. The cap should limit extra/user-configured pins, while system-required pins for the agent (`time`, `tavily`, widgets, Brave) are always eligible for binding.
- Add an actual binding test for the search agent under default settings. A test that only inspects `_get_pinned_specs("search")` is insufficient because the current cap can still drop later specs.

This gives the LLM dynamic access without forcing RAG/planning/canvas/image-generation turns to carry another schema by default.

### Prompt Strategy

Keep prompt changes short. Do not list many visual categories. Avoid hardcoding "architecture, fashion, food..." as a long routing table.

Refine the shared media snippet in `app/ai/prompts.py` to say:

```text
Media and visuals:
- You can display provided rich items inline with `<!--rich:<id>-->`; use only available IDs and never invent image URLs.
- If a visual reference would materially improve the answer and no image candidates are available, call an appropriate image/search tool once, then place only relevant returned images near the supporting text.
- Do not add media for decoration. Use images/widgets only when they clarify, compare, document, or illustrate the answer.
```

This preserves LLM flexibility:

- The model decides whether visuals materially improve the answer.
- The model decides which tool is appropriate.
- The model decides where selected markers belong.
- The deterministic placer remains a fallback, not the primary authoring mechanism.

Update search-specific prompt text only where it currently hardcodes Tavily as the first web search restriction. The rule should remain about `get_current_time` before web/news search, but image search for timeless visual references should not require a time lookup unless the user asks for current/recent images.

Suggested adjustment:

- Keep: call `get_current_time` before web/news/current searches.
- Do not require `get_current_time` before pure visual reference searches like "what does X look like?"
- Keep the no-first-Tavily rule only for actual Tavily/web search; do not make it apply to image search.

### Rich Candidate Flow

No new rich item type is needed.

Existing flow should remain:

1. `brave_image_search` returns normalized JSON with an `images` array.
2. `execute_tool_calls()` stores the tool result as text/artifact.
3. `build_image_candidates_from_tool_result()` detects `images` and creates image rich candidates.
4. The graph lifts `_rich_item_candidates` into context.
5. `build_rich_response_guidance()` exposes a bounded inventory to the model.
6. The model writes article prose and optional `<!--rich:<id>-->` markers.
7. `finalize_article_content()` repairs/auto-places relevant unreferenced candidates only when needed.
8. `build_bot_metadata()` persists only selected/placed rich image items.

Extend `build_image_candidates_from_tool_result()` only if needed:

- Preserve `source_url` in `payload`; it is already accepted by `ImagePayload`.
- Keep `thumbnail_url`, `width`, `height`, source domain, and provider in `provenance` unless the renderer needs them and the schema is narrowly extended.
- Keep `source` as the existing generic value (`tool_image`) unless there is a separate renderer or API need for `source="brave_image_search"`. Tests should primarily assert `provenance["tool"] == "brave_image_search"` for provider identity.

Do not expose base64 data in prompt inventory.

## Latency And Context Budgets

### Latency

Brave Image Search must be model-invoked, not automatically called for every turn.

Tool-level controls:

- Default `count = 6`.
- Clamp `count` to `brave_image_search_max_count`.
- Validate `safesearch` against Brave-supported values (`off`, `strict`); invalid values should fall back to the configured default or return a structured argument error.
- Timeout after `brave_image_search_timeout_seconds`.
- Return a structured error JSON on timeout/missing key/provider failure so the model can continue text-only.

Agent behavior:

- At most one Brave image search call per turn unless the user explicitly asks for more images or refinement.
- Do not call both Tavily and Brave just because both are available.
- If the answer needs current factual grounding and visuals, Tavily can be enough when it returns good images.
- If Tavily results lack usable images and visuals materially help, the model may call Brave once.

### Context

Provider raw output should not be forwarded wholesale.

Controls:

- Normalize and trim title/description strings in the MCP tool.
- Limit returned images to `count`.
- Let existing `rich_item_inventory_max_items`, `rich_item_inventory_max_chars`, and `rich_item_summary_max_chars` bound what reaches the model.
- Keep full URL payloads in metadata/artifacts, not repeated in prose.
- Hidden/unselected images stay scrubbed from v1 legacy image fields.

## Edge Cases

| Case | Expected Behavior |
|---|---|
| Brave API key missing | Tool returns JSON error with actionable message; answer can continue without images. |
| Brave timeout | Tool returns retryable/timeout-shaped JSON error; model should continue text-only or mention images were unavailable. |
| Invalid `safesearch` argument | Tool normalizes to configured default `strict` or returns structured argument error; it must not send unsupported values to Brave. |
| Brave returns no images | Tool returns `images: []`; no rich candidates are created. |
| Image result lacks `properties.url` direct image URL | Skip that item; do not use the page URL as the renderable image URL. |
| Image has title but no description | Use title/source text as alt text and placement signal. |
| Image has no meaningful title/description | Candidate may exist, but auto-placement should not place it. |
| User asks for latest/current image evidence | Search agent should still anchor time before current web/news search; pure image search alone is acceptable only if facts are not needed. |
| RAG/document answer needs external image search | Not pinned; model can discover via `tool_search` only if external visuals are actually requested and allowed by prompt/tool policy. |
| Non-rich client | Existing projection strips markers/rich metadata; no new behavior needed. |

## Implementation Plan

### Phase 1: Tests For Desired Contract

Add/modify tests first:

1. `tests/test_mcp_global_allowlist.py`
   - Expect `brave_image_search` in enabled global defaults.
   - Expect `brave_image_search` in `MCPManager.DEFAULT_SERVERS` or add a small MCP integration test for that reservation.

2. `tests/test_search_agent_time_context.py`
   - Search agent pinned specs include `brave_image_search::brave_image_search`.
   - Search agent binding includes `get_current_time`, `tavily_search`, the three widget tools, and `brave_image_search` under default settings. This is the regression guard for the existing five-tool pin cap.
   - Prompt no longer implies every visual/image search must be preceded by time lookup.

3. New test file: `tests/test_chat_agent_image_search_binding.py`
   - Chat agent pinned specs include Brave Image Search.
   - Chat binding exposes `brave_image_search` when the server tool is available.
   - RAG/planning/canvas/image-generator do not pin Brave by default.
   - User-configured extra pinned tools can still be capped without dropping agent-required Brave pins.

4. New test file: `tests/test_brave_image_search_server.py`
   - Missing API key returns JSON error.
   - Count is clamped to config maximum.
   - Default `safesearch` is `strict`, and only `off`/`strict` are sent to Brave.
   - Request includes `Accept: application/json`, `Accept-Encoding: gzip`, and `X-Subscription-Token`.
   - Timeout/provider exception returns JSON error.
   - Representative Brave response normalizes `properties.url` to `images[].url`, `result.url` to `source_url`, `thumbnail.src` to `thumbnail_url`, and `properties.width`/`height` to dimensions.
   - Items missing `properties.url` direct image URLs are skipped.

5. Extend `tests/test_rich_response_sources.py`
   - Brave-shaped payload creates image candidates.
   - Candidate provenance identifies `brave_image_search`; candidate `source` may remain the existing generic `tool_image`.
   - `source_url` is preserved in `payload`.
   - `thumbnail_url`, dimensions, source domain, and provider are preserved in `provenance` or another schema-approved location, never as forbidden `payload` extras.

6. Extend `tests/test_rich_response_prompt_inventory.py` or `tests/test_prompts_media_capability.py`
   - Media snippet remains compact.
   - It says to use available rich IDs only.
   - It does not contain a large hardcoded visual-topic list.
   - Update existing prompt-capability assertions that currently look for old phrases such as "You CAN display images inline" and "Never tell the user you cannot".

### Phase 2: Configuration

Modify `app/core/config.py`:

- Add Brave settings near the Tavily setting or MCP/search config.
- Validate numeric settings with existing positive validators where appropriate.
- Add range validation for timeout/count.
- Validate `brave_image_search_default_safesearch` as one of `off` or `strict`.

Modify `.env.example` and README with the new API key and budget settings. README updates should include:

- Optional API key sentence near `TAVILY_API_KEY`.
- API key/config table.
- Bundled MCP server table.
- Global default tools paragraph listing `time`, `tavily`, `widgets`, and `brave_image_search`.

### Phase 3: Brave MCP Server

Create `app/ai/mcp_servers/brave_image_search_server.py`.

Implementation requirements:

- Use `FastMCP("Brave Image Search")`.
- Read `BRAVE_SEARCH_API_KEY` from environment first, then settings.
- Use `httpx.Client(timeout=...)`.
- Call Brave Image Search with `q`, `count`, `country`, `search_lang`, `safesearch`.
- Send headers `Accept: application/json`, `Accept-Encoding: gzip`, and `X-Subscription-Token: <api key>`.
- Clamp count server-side.
- Validate or normalize `safesearch` before the request.
- Normalize response defensively.
- Use `result.properties.url` as the direct image URL. Do not use `result.url` as `images[].url`; it is the source page URL.
- Return JSON only.
- Do not include raw provider response unless behind an explicit debug flag; no debug flag is needed for the first implementation.

### Phase 4: MCP Registration And Pinning

Modify `app/ai/mcp_config.json`:

- Add enabled `brave_image_search` server.

Modify `app/ai/mcp_integration.py`:

- Add `brave_image_search` to `MCPManager.DEFAULT_SERVERS`.
- If settings-based API key forwarding remains needed for stdio child processes, mirror Tavily startup behavior by exporting `BRAVE_SEARCH_API_KEY` from settings during initialization.

Modify `app/ai/deferred_tool_binding.py`:

- Add a Brave image pinned spec.
- Pin it for `chat` and `search`.
- Keep it out of other agents by default.
- Ensure required agent pins are not truncated by `mcp_tool_search_max_pinned_tools`. Prefer including required system pins first and applying the cap only to optional settings-provided pins, or use an equivalent implementation that passes the default search binding test with all six required search tools present.

Update tests around global defaults and pinned specs.

### Phase 5: Prompt Tightening

Modify `app/ai/prompts.py`:

- Replace the current broader `MEDIA_CAPABILITY_SNIPPET` with the compact version above.
- Avoid long visual-topic lists.
- Make tool invocation conditional on material usefulness, not decorative media.
- Adjust search-agent time guidance so pure image reference search is not forced through the time tool.

Keep `INLINE_RICH_RESPONSE_SUFFIX` unchanged unless tests reveal marker guidance duplication.

### Phase 6: Candidate Metadata Compatibility

Inspect `app/core/rich_response.py` image payload schema.

Current schema accepts `url`/`data`, `mime_type`, `source_url`, and `description`, with `extra="forbid"`.

- Keep `source_url` and `description` in `payload`.
- Put `thumbnail_url`, dimensions, provider, and source domain in `provenance`.
- Add schema fields only if a renderer/API consumer actually needs them in public payload.

Do not broaden the schema to arbitrary extra fields.

### Phase 7: Verification

Run focused tests:

```bash
python -m pytest \
  tests/test_mcp_global_allowlist.py \
  tests/test_search_agent_time_context.py \
  tests/test_chat_agent_image_search_binding.py \
  tests/test_brave_image_search_server.py \
  tests/test_rich_response_sources.py \
  tests/test_rich_response_prompt_inventory.py \
  tests/test_prompts_media_capability.py \
  -q
```

Run broader regression:

```bash
python -m pytest tests/test_rich_response_* tests/test_article_image_flow.py -q
python -m pytest tests/test_tool_search_* tests/test_unified_tool_search.py -q
python -m ruff check app tests
```

Optional manual smoke:

1. Start the backend/demo.
2. Ask: "Spain architecture, make it visual like an article."
3. Expected:
   - Chat/search agent can call `brave_image_search`.
   - Answer contains prose plus inline rich image markers.
   - Final persisted message has `rich_items_version = 1`.
   - Only selected/placed images are visible.
   - No raw Brave JSON appears in the final answer.

## Acceptance Criteria

- Brave Image Search is an enabled global MCP server.
- Brave Image Search is reserved in `MCPManager.DEFAULT_SERVERS` as a managed built-in server.
- `chat_agent` and `search_agent` bind `brave_image_search` by default under deferred loading.
- Search-agent binding includes all required pins under default settings: `get_current_time`, `tavily_search`, three widget tools, and `brave_image_search`.
- Other agents can discover Brave through `tool_search` but do not pin it by default.
- A Brave-shaped response maps `properties.url` to the renderable image URL and creates rich image candidates without frontend contract changes.
- `source_url` stays in image `payload`; thumbnail, dimensions, provider, and source-domain metadata stay in schema-approved metadata/provenance.
- Prompt guidance stays compact and avoids large hardcoded visual-topic routing.
- The model can choose whether to call image search; no unconditional image-search preflight exists.
- Tool output is bounded by count and timeout settings.
- Brave requests use supported `safesearch` values only (`off` or `strict`), defaulting to `strict`.
- Missing key, timeout, and empty-result cases are graceful.
- Existing deterministic article placement continues to handle marker omissions.
- Focused tests and rich-response regressions pass.

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Extra pinned tool schema increases prompt/tool overhead | Pin only for chat/search; keep other agents discovery-only. |
| Search-agent required pins exceed the current five-tool cap | Make required agent pins non-droppable and cap only optional/user-configured pins; add an actual binding test. |
| Model overuses image search | Compact prompt says "materially improve"; add one-call-per-turn guidance and rely on tool-loop/test behavior. |
| Brave latency slows answers | Short timeout, small default count, no unconditional calls. |
| Context grows with raw image payloads | Normalize compact JSON; inventory caps already exist. |
| Brave page URL is accidentally rendered as an image | Map only `properties.url` to `images[].url`; keep result page URL as `source_url`; test both fields. |
| Unsupported Brave `safesearch` value causes provider errors | Validate to `off`/`strict` and default to `strict`. |
| Broken/hotlinked image URLs | Renderer should fail gracefully; source URL retained where possible for attribution/debugging. |
| Prompt becomes bloated | Replace current media snippet instead of appending another block. |
| Over-hardcoded visual routing | Use general usefulness guidance; no long taxonomy or deterministic classifier. |

## Implementation Progress

Executed test-first (RED → GREEN per step). Verification commands run with `.venv/Scripts/python.exe` (the app runtime venv; system `python` lacks `mcp`).

### Phase 2 — Configuration (DONE)

- Added to `app/core/config.py`: `brave_search_api_key`, `brave_image_search_default_count=6`,
  `brave_image_search_max_count=10`, `brave_image_search_timeout_seconds=2.5`,
  `brave_image_search_default_safesearch="strict"`.
- Validators: count fields added to existing `_positive_int`; new `_positive_float` for the
  timeout; new `_validate_brave_safesearch` restricting to `{off, strict}` (lower-cased, stripped).
- Test: `tests/test_brave_image_search_config.py` (4 tests) — RED (missing attrs), then GREEN.

Design decisions:
- **D1 (safesearch handling):** config-level default is validated to `off`/`strict`. At the tool
  layer (Phase 3), `None` → configured default; explicit invalid value → structured argument error
  (informative to the model) rather than silent normalization.
- **`.env.example` is NOT editable**: the global-guard security hook blocks read/write of `.env.*`.
  The required keys are listed at the end of this document for manual addition.

### Phase 3 — Brave MCP server (DONE)

- Created `app/ai/mcp_servers/brave_image_search_server.py`. Core logic is the plain function
  `brave_image_search(...)` registered via `mcp.tool()(brave_image_search)` (FastMCP returns the
  original fn, so the function is directly unit-testable without spawning the stdio process).
- Reads key from `BRAVE_SEARCH_API_KEY` env first, then `settings` (mirrors Tavily). Uses
  `httpx.Client(timeout=...)`. Sends `Accept: application/json`, `Accept-Encoding: gzip`,
  `X-Subscription-Token`. Clamps count to `brave_image_search_max_count`. Normalizes
  `properties.url`→`url`, `result.url`→`source_url`, `thumbnail.src`→`thumbnail_url`,
  `properties.width/height`→dimensions; drops items lacking a direct `properties.url`.
- Test: `tests/test_brave_image_search_server.py` (8 tests) — RED (ImportError), then GREEN.

Design decisions:
- **D3 (testability):** logic in a module-level function decorated with `mcp.tool()`; tests patch
  `httpx.Client`. No DI/client-factory parameter added to keep the tool signature clean for the
  model.
- **D5 (mime guess):** server best-effort guesses `mime_type` from the direct image URL extension;
  omitted when unknown (the candidate builder guesses as a fallback). Both yield an allowed type.
- **description fallback:** `description = title or source or meta_url.hostname`. This gives a
  meaningful alt-text/placement signal; when none exists, the candidate stays generic-alt so
  deterministic auto-placement correctly skips it.
- Extra fields emitted for downstream provenance: `thumbnail_url`, `width`, `height`,
  `source_domain`, `provider` (consumed in Phase 6).

### Phase 4 — MCP registration & pinning (DONE)

- `app/ai/mcp_config.json`: added enabled `brave_image_search` stdio server.
- `app/ai/mcp_integration.py`: added `brave_image_search` to `MCPManager.DEFAULT_SERVERS`
  (reserved/managed) and forwarded `settings.brave_search_api_key` → `BRAVE_SEARCH_API_KEY` env in
  `initialize()` so the stdio child inherits the key (mirrors Tavily).
- `app/ai/deferred_tool_binding.py`: added `_IMAGE_SEARCH_PINNED_SPEC` pinned for `{chat, search}`.
  Refactored pin selection (D2): new `_get_required_pinned_specs()` returns system-required pins;
  `get_pinned_tools()` now binds **all** required specs and applies `mcp_tool_search_max_pinned_tools`
  only to optional/user-configured pins. `_get_pinned_specs()` kept as the combined view for tests.
- Tests: `test_mcp_global_allowlist.py` (+reservation test), `test_search_agent_time_context.py`
  (+brave pin spec, +6-required-pin binding regression guard under default cap=5),
  new `test_chat_agent_image_search_binding.py` (4 tests). RED → GREEN (14 passed).

Design decision:
- **D2 (non-droppable required pins):** required pins are bound first and unconditionally; the cap
  only trims user extras. Previously `pinned_specs[:max_pinned]` could drop the 6th search pin.

### Phase 5 — Prompt tightening (DONE)

- Replaced `MEDIA_CAPABILITY_SNIPPET` with the compact 3-bullet version (use available IDs only,
  never invent URLs; call an image/search tool once when a visual materially helps; no decorative
  media). New snippet is ~500 chars (was ~700) and carries no visual-topic taxonomy.
- Adjusted `SEARCH_SYSTEM_PROMPT`: the `get_current_time`-before-search rule now reads "web/news
  search"; added an explicit carve-out that pure image reference searches ("what does X look like")
  do not require a time lookup and that the no-first-Tavily rule applies to web/news, not image
  search. Kept the existing web/news time-ordering lines intact.
- `INLINE_RICH_RESPONSE_SUFFIX` left unchanged (no marker-guidance duplication surfaced).
- Tests: rewrote `tests/test_prompts_media_capability.py` to the new contract (compact, "available
  IDs", "never invent", no taxonomy words) and added `test_search_prompt_exempts_image_reference_search_from_time_lookup`. RED → GREEN. Old phrases ("You CAN display images inline", "Never tell the
  user you cannot") now exist only in plan markdown, not in code/tests.

### Phase 6 — Candidate metadata compatibility (DONE)

- Extended `build_image_candidates_from_tool_result()` in `app/ai/tool_execution.py`: provenance now
  carries `thumbnail_url`, `width`, `height`, `source_domain`, and `provider` when present in the
  image dict (generic, applies to any tool; absent for Tavily so existing behavior is unchanged).
- `payload` is unchanged — only `url`/`data`, `mime_type`, `source_url`, `description` (the
  `ImagePayload` schema with `extra="forbid"`). No schema broadening needed.
- Provider identity is `provenance["tool"] == "brave_image_search"`; candidate `source` stays the
  generic `tool_image` (D4).
- Tests: 3 new tests in `tests/test_rich_response_sources.py` (identity/source_url, provenance-only
  metadata, and full `validate_public_rich_item` round-trip). RED on the provenance test → GREEN
  (18 passed in file).

### Phase 7 — Verification & docs (DONE)

- Focused suite (8 files from the plan): **59 passed**.
- Broader regression (`test_rich_response_*`, `test_article_image_flow`, `test_tool_search_*`,
  `test_unified_tool_search`, `test_client_tool_isolation`, `test_tool_execution_recovery`,
  `test_hitl_policy`): **157 passed**.
- `ruff check` clean on every new/edited file. `config.py` shows only pre-existing baseline E501s
  (line 179 `model_encryption_key`, etc.) — none from this change.
- Import smoke test confirms: settings load, `brave_image_search` reserved in `DEFAULT_SERVERS`,
  chat pins `{brave, widgets}`, search pins `{time, tavily, brave, 3 widgets}` (6), rag pins widgets
  only, missing-key path returns a structured error, enabled servers =
  `{brave_image_search, tavily, time, widgets}`.
- README updated: optional-key sentence, env-key/config table (key + 4 budget settings), bundled MCP
  server table, global-default-tools paragraph, source-tree comment, and agent capability table.

**`.env.example` could not be edited** (global-guard blocks `.env.*` read/write). Add these lines
manually:

```env
BRAVE_SEARCH_API_KEY=
BRAVE_IMAGE_SEARCH_DEFAULT_COUNT=6
BRAVE_IMAGE_SEARCH_MAX_COUNT=10
BRAVE_IMAGE_SEARCH_TIMEOUT_SECONDS=2.5
BRAVE_IMAGE_SEARCH_DEFAULT_SAFESEARCH=strict
```

### Files changed

- `app/core/config.py` — Brave settings + validators
- `app/ai/mcp_servers/brave_image_search_server.py` — new server
- `app/ai/mcp_config.json` — enabled `brave_image_search`
- `app/ai/mcp_integration.py` — `DEFAULT_SERVERS` + env forwarding
- `app/ai/deferred_tool_binding.py` — Brave pin + required/optional pin split
- `app/ai/prompts.py` — compact media snippet + search-agent time carve-out
- `app/ai/tool_execution.py` — candidate provenance metadata
- `README.md` — docs
- Tests: `test_brave_image_search_config.py`, `test_brave_image_search_server.py`,
  `test_chat_agent_image_search_binding.py` (new); `test_mcp_global_allowlist.py`,
  `test_search_agent_time_context.py`, `test_prompts_media_capability.py`,
  `test_rich_response_sources.py` (extended)

### Not done (out of scope / blocked)

- `.env.example` edit (blocked — see manual block above).
- Optional manual smoke against a live Brave API key + running backend (requires a real key).
