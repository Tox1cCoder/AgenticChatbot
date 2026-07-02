# AI SDK Response Format Cleanup (2026-07-02)

**Goal:** Remove redundant legacy fields from all AI SDK-visible response surfaces and refactor the
projection code into a production-ready layering. Persisted metadata (DB) and the internal
`/messages/stream` path used by the Streamlit UI are unchanged.

**Approved scope (user decisions, 2026-07-02):**

- Removal scope: *everything AI SDK-visible* — legacy renderer fields are scrubbed from AI SDK
  responses; `rich_items` + AI SDK `parts` become the only render path.
- FE status: *client in development* — clean breaking changes now, prominently listed in the
  contract doc changelog. No deprecation mirrors.

## Breaking wire changes (AI SDK surfaces only)

1. `messageMetadata` mirror removed everywhere. `metadata` is the single metadata field on
   history messages, `data-assistant-message`, and the interrupt paused-message projection.
2. `data-interrupt.data.pendingToolCalls` removed. `data.interrupt.action_requests[]` is the only
   tool-approval list.
3. `AISDKMessagesData.total` removed. Use `data.meta.total`.
4. Legacy renderer fields scrubbed from every AI SDK metadata projection:
   `images`, `has_images`, `images_count`, `agentic_images_count`, `live_widgets`,
   `canvas_artifact`, `pending_tool_calls`, plus internal `_`-prefixed keys.
   - Images remain visible as `file` parts (extracted before the scrub; sourced from `rich_items`
     for v1 messages, from legacy `images` metadata for pre-v1 history).
   - Widgets, canvas artifacts, and tool renders are visible only via `rich_items`, so AI SDK
     clients must send `inlineRichResponseV1: true`. The server `inline_rich_response_enabled`
     kill switch now disables rich UI entirely for AI SDK clients (text + image parts only).
5. `data-user-message` / `data-error-message` payloads are projected (id, role, content,
   createdAt, metadata, parts) instead of raw DB dumps — `conversation_id`, `sender`,
   `updated_at` no longer leak to the wire.
6. History messages always carry `parts` (leading `text` part guaranteed) per AI SDK v5+
   `UIMessage` expectations.

## Refactor (non-breaking)

7. New module `app/services/event_streaming/ai_sdk_projection.py` owns all response projection
   helpers. `ai_sdk_v6.py` imports it top-level (removes the deferred service→API imports);
   `app/api/ai_sdk.py` keeps request parsing + routes only.
8. Heartbeat interval: one constant in `ai_sdk_v6.py`. Removes the phantom
   `settings.ai_sdk_heartbeat_interval_seconds` lookup (never defined in config) and the
   duplicate API-layer constant.
9. Dead code removed: `AISDKMessagePart` (never referenced), no-op branches in
   `_extract_data_from_candidate`.

## Follow-up fixes (same day)

- Fixed a pre-existing leak found during live verification: for v1 messages served to
  non-capable clients, the capability projection stripped `rich_items_version` before image
  parts were sourced, so leftover legacy `images` candidates leaked as `file` parts. v1-ness
  is now decided on the original metadata (`visible_image_file_parts(..., is_v1=...)`).
- Extended the wire scrub with database-redundant debug keys: `conversation_id`,
  `has_tool_calls`, `context_messages`. Removed the write-only `has_tool_calls` /
  `context_messages` writes at their agent sources (zero readers); `conversation_id` writes
  remain persisted (many call sites) but never reach the AI SDK wire.
- Added a fail-fast startup guard (`app/main.py::_ensure_selector_event_loop`): on Windows
  with checkpoints enabled, a ProactorEventLoop server now refuses to start instead of
  silently failing every checkpointer pool connection (uvicorn 0.46 hard-codes Proactor for
  non-subprocess launches).

## Out of scope

- `build_bot_metadata()` persistence (Streamlit still reads legacy fields from the DB).
- The internal `/messages/stream` SSE contract.
- RAG fields (`documents_cited`, `citations`, counters) — not legacy; kept.

## Verification

- Updated pinning tests: `test_ai_sdk_v6_stream_contract.py`, `test_ai_sdk_context_window.py`,
  `test_rich_response_streaming.py`, `client_backend/test_sse_keepalive.py`.
- New assertions: legacy scrub on history + stream side-channel, no `pendingToolCalls`,
  projected user-message event, guaranteed text part.
- `plans/AI_SDK_FE_CONTRACT.md` updated with a breaking-changes changelog.
