# HTML-Only Live Widgets Design

## Goal

Live widgets should produce bespoke, interactive iframe experiences instead of falling back to rigid structured tables or chart schemas. For conceptual prompts such as "giai thich dao dong dieu hoa", the expected widget is a self-contained HTML micro-app: animated visual state, parameter controls, live readouts, and a diagram or graph that helps the user understand the concept.

## Product Direction

- Live widgets have one supported type: `html`.
- Remove structured widget types: `table`, `chart`, `dashboard`, `form`, and `list`.
- Remove widget quality grading and `quality_guidance`.
- Keep minimal contract validation only: widget state must be a JSON object, HTML content must be non-empty, and iframe height must be present, numeric, and bounded.
- Keep the widget runtime, connection, WebSocket, inline placement, and action plumbing where they still apply to HTML widgets.
- Preserve old persisted structured-widget metadata only as a legacy/fallback concern. New widget create/update paths must not create or document structured widgets.

## HTML Widget State

The widget state is:

```json
{
  "html": "<!doctype html>...",
  "height": 620,
  "caption": "Optional short caption"
}
```

No state-field aliases are part of the new contract. New widgets must use `html`.
Do not accept or document aliases such as `document`, `content`, `srcdoc`, `iframe`, `micro_app`, `min_height`, or `minHeight`.

## Generation Guidance

Prompt and tool guidance should steer the model toward micro-apps, not fixed schemas.

For conceptual explanations involving motion, changing variables, systems, physics, math, processes, or "show how it works", the assistant should create an HTML widget with:

- animation or manipulable visual state
- sliders or controls for key parameters
- live numeric readouts
- canvas, SVG, or DOM-based diagrams/graphs when useful
- labels and captions in the user's language
- responsive inline CSS and vanilla JavaScript
- no external dependencies, auth assumptions, or cross-window requirements

Example guidance for harmonic oscillation:

- animate the oscillator position `x(t)`
- draw a time graph of displacement
- expose sliders for amplitude, angular frequency, and phase
- include pause/reset controls
- show live values for time and displacement
- label the experience in Vietnamese when the user asks in Vietnamese

## Code Changes

- `app/ai/mcp_servers/widgets_server.py`
  - Restrict `widget_create` to exactly `widget_type="html"`.
  - Restrict `widget_update` through the existing record type.
  - Replace structured-widget docstring examples with HTML micro-app guidance.
  - Remove `quality_guidance` behavior.
  - Return clear errors for unsupported widget types, including the removed structured types and former aliases.
  - Keep `_parse_widget_state` only as JSON/string parsing support; do not let parser fallback broaden the widget contract beyond a JSON object.

- `app/services/widget_quality.py`
  - Delete this module.
  - Move minimal HTML contract validation into the widget server or a small contract helper.
  - Move action template helpers (`render_action_template`, `resolve_widget_action_message`) to `app/api/widgets.py` or a small action helper module if actions remain supported. `app/api/widgets.py` imports this today, so deleting the module without moving these helpers will break the action endpoint.

- `app/api/widgets.py`
  - Update the import path for action-template helpers after `widget_quality.py` is removed.
  - Validate the merged state after `state_patch` in `POST /widgets/{widget_id}/actions/{action_key}` if patches are allowed to modify top-level `html`, `height`, or `caption`.
  - Validate the merged state after WebSocket `user_state_patch` for the same reason, or explicitly restrict patches to non-contract UI/action fields.
  - Decide and document legacy restore behavior for old `table`, `chart`, `dashboard`, `form`, and `list` widgets recovered from persisted message metadata. Prefer a non-crashing unsupported/legacy placeholder over silently trying to render a removed structured renderer.

- `app/ai/prompts.py`
  - Replace table/chart/dashboard/form/list widget guidance with HTML-only micro-app guidance.
  - Include the harmonic-oscillation example pattern.
  - Keep inline rich marker placement rules.
  - Update router boundary guidance that still names widgets, charts, tables, forms, and dashboards as in-chat widget reasons.

- `demo.py`
  - Remove structured live-widget render branches and support code for chart/table/dashboard/form/list widgets.
  - Keep only sandboxed iframe rendering for HTML widgets.
  - Remove state aliases currently accepted by the HTML renderer (`document`, `content`, `srcdoc`, `min_height`, `minHeight`) so the demo matches the new contract.
  - Keep action controls only if they work with HTML widget state and do not require the removed structured renderers.

- Documentation and tests
  - Update README and live-widget frontend docs to describe HTML-only widgets.
  - Update `plans/AI_SDK_FE_CONTRACT.md`. This is the file sent to frontend developers, and it currently documents structured widget examples such as `widget_type="table"` and renderer types such as `table` and `chart`.
    - Rich live widget examples must use `widget_type: "html"`.
    - Legacy `live_widgets[]`, connection response, and WebSocket event examples must show `widget_type: "html"`.
    - The Live Widgets section must say FE renders the widget state only as a sandboxed iframe from `state.html`.
    - The contract should document expected HTML state: `{ "html": "<!doctype html>...", "height": 620, "caption": "Optional short caption" }`.
    - Remove or mark obsolete any instruction to choose React renderers by `table`, `chart`, `dashboard`, `form`, or `list`.
    - Document that `state.html` is untrusted executable content and must not be injected into the main chat DOM.
  - Delete or rewrite tests that assert structured widget support.
  - Add tests that reject structured widget creation and accept a self-contained HTML simulation widget.
  - Add tests that reject unsupported aliases (`iframe`, `micro_app`) and state-field aliases (`document`, `content`, `srcdoc`, `min_height`, `minHeight`).
  - Add tests that no successful `widget_create` or `widget_update` response contains `quality_guidance`.

## Error Handling

Invalid live widgets should fail with clear, model-readable errors:

- unsupported widget type
- invalid JSON object
- missing HTML content
- missing height
- non-numeric height
- height outside the accepted iframe range

These errors are contract validation, not editorial scoring.

## Legacy Compatibility

Existing conversations may contain persisted `live_widgets` metadata or tool artifacts for structured widget types. The implementation must choose one behavior and test it:

- Render a lightweight unsupported/legacy placeholder that does not crash the Streamlit or AI SDK frontend path, or
- Continue to show old structured widgets behind a clearly isolated legacy renderer while preventing all new structured widget creation.

The migration should not silently advertise structured widgets in prompts, tool docs, or frontend contracts after the HTML-only change.

## Testing

Targeted tests:

- prompt text mentions HTML-only widgets and micro-app expectations
- `widget_create` rejects `table`, `chart`, `dashboard`, `form`, and `list`
- `widget_create` rejects unknown widget types and aliases such as `iframe` and `micro_app`
- `widget_create` accepts valid `html`
- `widget_create` rejects non-object state, empty HTML, missing height, non-numeric height, and out-of-range height
- `widget_update` preserves the existing record type and rejects invalid HTML state for existing HTML widgets
- widget action or WebSocket state patches either preserve the HTML contract after merge or are limited to non-contract UI/action fields
- successful widget create/update payloads do not include `quality_guidance`
- Streamlit live-widget component contains only the HTML iframe renderer path
- AI SDK frontend contract docs show HTML-only live widget examples and no structured widget renderer requirements
- widget action endpoint continues to render assistant messages if actions remain supported

Broader verification:

```powershell
python -m pytest tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_widget_actions_api.py tests/test_demo_meaningful_widgets.py tests/test_demo_plan_widget.py tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_rich_placement.py -q
```

The exact test set may change as structured-widget tests are removed or rewritten.

## Implementation Progress Log

Baseline before changes: `test_widget_quality.py test_widgets_api.py test_widget_actions_api.py test_widget_runtime.py` → 100 passed.

### Step 1 — `app/services/widget_contract.py` (done)

New module hosts the minimal HTML contract plus the action-template helpers moved
out of `widget_quality.py`.

- `assert_supported_widget_type(widget_type)` — raises a clear `ValueError` unless the
  type is exactly `html`. Rejects structured types, the `iframe`/`micro_app` aliases,
  and unknown/empty types.
- `validate_html_widget_state(state)` — raises on: non-object state, missing/empty
  `state.html`, missing height, non-numeric height, out-of-range height.
- `render_action_template` / `resolve_widget_action_message` — moved verbatim from
  `widget_quality.py` so the action endpoint keeps working after that module is deleted.

Design decisions:
- **Height bounds kept at 260..960** (unchanged from the old html-widget check) — the
  spec says "bounded" without new numbers, so the existing range is preserved.
- **Missing height is now an error.** The old html check defaulted to 560; the new
  contract requires height to be present. Matches the "missing height" error in the spec.
- **No aliases read.** Only `state.html` and `state.height` are inspected, so
  `document`/`content`/`srcdoc`/`min_height`/`minHeight` simply fail the html/height
  checks rather than being silently accepted.
- **Single module for both concerns** (contract + action helpers) satisfies the spec's
  "small contract helper" and "small action helper module" with one import path.

Verification: `tests/test_widget_contract.py` → 28 passed (watched it fail first with
`ModuleNotFoundError`, then pass).

### Step 2 — `app/ai/mcp_servers/widgets_server.py` (done)

- `widget_create` now: `assert_supported_widget_type(widget_type)` → parse →
  `validate_html_widget_state(state)` → store. Type is forced to
  `SUPPORTED_WIDGET_TYPE` on create. No `quality_guidance` in the response.
- `widget_update` now: parse → fetch existing → `assert_supported_widget_type(existing.widget_type)`
  → `validate_html_widget_state(new_state)` → store. The store keeps the original
  type, so the existing record type is preserved.
- Docstrings rewritten to steer toward HTML micro-apps, including the harmonic
  oscillation example. Structured/`quality_guidance` examples removed.

Design decisions:
- **Type check runs before state parsing/validation** so structured types get the
  clearest error regardless of payload shape.
- **`widget_update` rejects legacy structured widgets** (`assert_supported_widget_type`
  on the existing type). This prevents new writes from perpetuating structured
  widgets while leaving legacy persisted metadata untouched for read/restore paths.
- `_parse_widget_state` is unchanged — it stays as JSON/Python-literal parsing support;
  the contract validator (not the parser) enforces the object/`html`/`height` rules.

Rewrote `TestWidgetToolQualityEnforcement` → `TestWidgetToolHtmlContract` in
`tests/test_widget_runtime.py` (watched the structured-type cases fail with the old
"chart requires…" messages, then pass). Verification: `tests/test_widget_runtime.py`
→ 82 passed. (`TestPromptUpdates::test_prompt_mentions_article_style_widget_quality`
still passes here; it will be rewritten in Step 5 when the prompt changes.)

### Step 3 — `app/api/widgets.py` (done)

- Import path moved to `app.services.widget_contract` (`resolve_widget_action_message`,
  `validate_html_widget_state`, `SUPPORTED_WIDGET_TYPE`).
- Added `_html_patch_contract_error(record, patch)`: shallow-merges the patch into the
  current state and validates the HTML contract; returns an error string or `None`.
  Legacy non-HTML widgets are skipped (read/restore-only compatibility).
- Action endpoint (`POST /actions/{action_key}`): validates the prospective merge
  **before** calling `store.patch`, returning `400` if it would break the contract — so
  invalid state is never persisted.
- WebSocket `user_state_patch`: re-reads current state, validates the prospective merge,
  and sends an `error` event (skipping the patch) when it would break the contract.

Design decision: validate the *prospective* merge before writing, rather than patch-then-revert.
The WS handler re-`get`s current state (the connect-time `record` can be stale) for the merge base.

Rewrote `tests/test_widget_actions_api.py` to seed HTML widgets (with top-level `actions`),
added `test_widget_action_rejects_contract_breaking_state_patch`. Verification:
`test_widget_actions_api.py` + `test_widgets_api.py` → 15 passed.

### Step 4 — delete `app/services/widget_quality.py` (done)

Deleted `app/services/widget_quality.py` and `tests/test_widget_quality.py`. Confirmed no
remaining code importers (only a test-method name and docs strings matched). Imports verified;
`test_widget_contract/test_widget_runtime/test_widgets_api/test_widget_actions_api` → 125 passed.

Scope note: `test_widgets_api.py` connection/restore tests and the WS sync test still use
`table`/`chart`/`list` types via the low-level `store.create`. These exercise the **legacy
restore path** and the generic store/WS plumbing (type-agnostic), which the spec intends to
preserve — they do not assert structured *rendering*, so they are kept as legacy/plumbing
coverage. `plans/meaningful_widgets.md` is a stale internal plan (not a prompt/tool-doc/FE
contract) and is left out of scope.

### Step 5 — `app/ai/prompts.py` (done)

- `CHAT_SYSTEM_PROMPT`: replaced the chart/table/dashboard/form/list + presentation/
  controls/views guidance with HTML-only micro-app guidance — animation, sliders, live
  readouts, canvas/SVG diagrams, vanilla JS, the `{"html","height","caption"}` shape, and
  the harmonic-oscillation example. Kept the inline `<!--rich:widget:<id>-->` marker rule.
- `ROUTER_SYSTEM_PROMPT`: reason 5 reframed to "interactive HTML micro-apps"; dropped the
  ", chart, table, form, or dashboard" enumeration from the canvas/LiveUI boundary line
  (kept the "Do not route to canvas_agent merely because a widget" phrasing the router test
  pins on, and "LiveUI widgets are for compact in-chat aids").

Rewrote `TestPromptUpdates::test_prompt_mentions_article_style_widget_quality` →
`test_prompt_mentions_html_micro_app_widgets` and added
`test_prompt_drops_structured_widget_guidance` (watched both fail, then pass).
Verification: `TestPromptUpdates` → 4 passed.

### Step 6 — `demo.py` live-widget component (done)

`_build_live_widget_component_html` shrank from ~1410 lines to ~190: it now renders only
the sandboxed HTML iframe. Removed all structured renderers and their support code (table,
chart, dashboard, form, list, controls, views/variants, chart-hover tooltip, sort/series
toggles, table search, form-field collection, action buttons + `runWidgetAction` /
`streamAssistantResponse`). Kept the card chrome, status/connection/version display, error
slot, the full WebSocket connect/reconnect/ping-pong/state-sync logic, and `renderHtmlWidget`.

- `htmlState(data)` reads **only** `data.html`, `data.height`, `data.caption` — the
  `document`/`content`/`srcdoc`/`min_height`/`minHeight` aliases are gone, matching the
  contract. Height is clamped to 260..960 (default 620 if absent, for renderer robustness).
- `_live_widget_frame_height` collapses to a single constant (820) since all widgets are
  HTML now; the inner iframe sizes itself from `state.height` over the socket.

Design decisions:
- **Demo HTML widget is iframe-only — no external action buttons.** The action *endpoint*
  stays supported server-side (Step 3), but the Streamlit component is purely the iframe so
  it "contains only the HTML iframe renderer path" (a self-contained micro-app owns its own
  interactivity). The `runWidgetAction`/`streamAssistantResponse` JS and the action-hooks demo
  test were removed accordingly.
- The no-raw-newline-in-JS-strings regression test was kept (multi-line content stays inside
  backtick template literals).

Rewrote `tests/test_demo_meaningful_widgets.py` to assert the HTML-only path: sandboxed
iframe + `renderHtmlWidget`, WebSocket path retained, no structured renderers, no alias reads,
plus the JS-newline regression. Verification: `test_demo_meaningful_widgets.py` → 5 passed;
`test_demo_plan_widget.py` + `test_demo_rich_response.py` → 27 passed (placement/`render_live_widgets`
plumbing is widget-type-agnostic and unaffected).

### Step 8 — documentation (done)

- `README.md`: "Meaningful Widgets contract" → "HTML widget contract" — the `{html,height,caption}`
  envelope, sandboxed-iframe-from-`state.html` rendering, untrusted-content warning, contract
  validation in `widget_contract.py` (no `quality_guidance`), and the action endpoint with
  post-patch re-validation. Test command updated `test_widget_quality.py` → `test_widget_contract.py`.
- `plans/AI_SDK_FE_CONTRACT.md`: all live-widget / connection / WS examples now use
  `"widget_type": "html"`; the Live Widgets section documents the HTML state shape, the
  sandboxed-iframe rendering rule, the untrusted-content warning, and "do not choose a React
  renderer by widget_type"; the `widget_type` field row notes `html` is the only supported type
  with a legacy placeholder fallback; mount flow renders `state.html` via `srcdoc`.
- `plans/live-widgets-frontend-integration.md`: rewritten HTML-only — single iframe renderer,
  removed structured-renderer strategy / `table_ui` / `chart_ui` / `controls` / `views` /
  `variants` / presentation / hover sections; kept connection/WebSocket/reconnect/inline-rich
  plumbing; `WidgetType = "html"`; new §10 documents the HTML state contract + optional action
  endpoint with contract re-validation. README's stale "§ 11" reference fixed to "§§ 5 and 10".

Added `tests/test_widget_docs_html_only.py` (watched all 5 fail, then pass). Scope note:
`plans/meaningful_widgets.md` got a one-line "SUPERSEDED" banner pointing at this spec and the
HTML-only integration guide; its body is left as history.

### Step 9 — full verification (done)

Plan's canonical command:

```
pytest tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_widget_actions_api.py \
  tests/test_demo_meaningful_widgets.py tests/test_demo_plan_widget.py \
  tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_rich_placement.py -q
```
→ **184 passed.**

Wider affected surface (adds `test_widget_contract.py`, `test_widget_docs_html_only.py`,
`test_demo_rich_response.py`, `test_rich_response_streaming.py`, `test_rich_response_prompt_inventory.py`,
`test_rich_response_sources.py`, `test_message_history_pipeline.py`, `test_ai_sdk_context_window.py`,
`test_tool_result_rendering.py`, `client_backend/test_widget_action_proxy.py`) → all passing
(217 + 81 across two runs, no failures).

Integrity checks:
- No remaining code importers of `widget_quality` / `assess_widget_state` anywhere (only the
  superseded plan + test provenance comments + the doc guard test reference the name).
- `app.ai.mcp_servers.widgets_server`, `app.api.widgets`, `app.services.widget_contract`,
  `demo.py` all import / `py_compile` cleanly.
- ruff on changed files: only the 3 pre-existing `E402` warnings in `widgets_server.py` (the
  `sys.path` bootstrap must precede `app.*` imports — unchanged from before); no new lint.

### Decisions / scope summary

- Single new module `app/services/widget_contract.py` carries both the minimal HTML contract
  and the action helpers (one import path).
- `widget_create` forces `widget_type="html"`; `widget_update` keeps the existing record type
  and rejects legacy structured widgets. No `quality_guidance` anywhere.
- Patches (action `state_patch` + WS `user_state_patch`) re-validate the merged HTML contract
  before storing for HTML widgets; legacy non-HTML widgets are skipped (read/restore only).
- The Streamlit demo widget is iframe-only (no external action buttons); the action *endpoint*
  remains server-side for any client that wants it.
- Legacy compatibility chosen: **non-crashing legacy placeholder** — new create/update paths and
  all prompts/tool-docs/FE contracts only advertise `html`; connection/restore plumbing still
  reads legacy persisted metadata, and FE docs instruct rendering a legacy placeholder for any
  non-`html` type.
- Out of scope: `plans/meaningful_widgets.md` body (history, banner added); the
  `test_widgets_api.py` connection/restore + WS-sync plumbing tests keep structured types since
  they exercise the legacy-restore path and type-agnostic store/WS mechanics.
