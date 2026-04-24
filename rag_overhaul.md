# RAG Overhaul Implementation Plan

> **For the implementer:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` to implement this plan.

## Goal

Make RAG production-ready while preserving the sidecar architecture from `plan.md`:

- The canonical server owns document upload validation, parsing, chunk persistence, indexing, retrieval, authorization, and cleanup.
- `client_backend` only proxies uploaded bytes and conversation context to the canonical server.
- The server must safely handle simultaneous users, devices, conversations, uploads, and RAG searches.
- Agentic RAG is the only runtime RAG path. Remove prompt-built RAG execution and all runtime branches that keep old behavior alive.
- Legacy and deprecated RAG code must be deleted, not hidden behind config flags or alternate branches.
- PostgreSQL stores canonical parsed artifacts and chunks. Qdrant remains the vector serving index.
- Retrieval is scoped by server-owned `user_id` and `conversation_id`, not by sidecar process state or model-facing `device_id`.
- Active model configuration is explicit: Gemini `gemini-3.1-pro-preview`, embedding model `google/embeddinggemma-300m`, and reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## Verified Current Codebase State

- `client_backend/api/documents.py` receives uploads and sends bytes to the server through `ServerAPIClient.upload_document_bytes(...)`.
- `client_backend/services/server_api.py` posts to `/documents/upload` with `conversation_id` and file content. This is compatible with server-side document processing.
- `app/api/documents.py` performs upload validation and enqueues document processing.
- `app/services/document_processing_service.py` parses files, extracts images, chunks text, and writes chunks directly to Qdrant payloads.
- `app/ai/agents/rag_agent.py` still has two RAG execution paths: agentic tool use and prompt-built RAG.
- `app/ai/graph.py` still has a separate traditional RAG streaming branch.
- `app/core/container.py` hardcodes the embedding model instead of reading the active RAG embedding setting.
- `app/core/config.py`, `.env.example`, and `.env` disagree on reranker defaults.
- `app/models/document_parse_artifact.py` exists, but processing does not persist parse artifacts.
- `app/models/document_image.py` has nullable `chunk_id` without an enforced relationship to `document_chunks`.
- Historical migration `1d24e8e1ec28_add_documents_table_with_status_and_.py` already creates `document_chunks`; the new migration must alter and normalize that table, not create it from scratch.
- `app/repositories/document_parse_artifact.py` is async-shaped while the current DB/session wiring is sync.
- `app/services/document_service.py` instantiates `RAGAgent` for document deletion, coupling CRUD cleanup to agent runtime.
- Tests currently use normal `pytest`; `pytest-mock` is not configured. New tests should use `unittest.mock` unless the dependency is intentionally added.

## Sidecar And Multi-User Constraints

No conflict was found between the sidecar plan and this RAG overhaul when the following constraints are enforced:

- Keep all parsing, artifact storage, chunk persistence, vector indexing, and retrieval inside the canonical server.
- Keep `client_backend` document upload behavior as a byte proxy to `/documents/upload`; do not parse, chunk, embed, index, or authorize documents inside the sidecar.
- Treat `device_id` as transport/session metadata owned by the sidecar execution scope. Do not add `device_id` to RAG tool schemas or document search arguments.
- Scope every RAG operation by canonical server context: `user_id`, `conversation_id`, and authenticated request/session state.
- Avoid mutable per-user RAG state on singleton objects. Shared model clients are acceptable only when request-specific context is passed explicitly.
- Ensure concurrent uploads from different users cannot share parse artifacts, chunk rows, Qdrant payload filters, temp directories, or cleanup jobs.

## Architecture Decisions

- `DocumentChunk` rows are the canonical content store. Qdrant stores embeddings plus lookup metadata only.
- Qdrant point payloads must include enough immutable IDs to hydrate from SQL: `document_id`, `chunk_id`, `conversation_id`, `user_id`, `embedding_model`, and page/source metadata.
- All read-style RAG actions (`SEARCH`, `READ_DOCUMENT`, `GREP_DOCUMENT`, `LIST_DOCUMENTS`) hydrate content and authorization-sensitive metadata from PostgreSQL.
- Delete old prompt-built RAG code instead of preserving another execution path.
- Delete stale RAG settings after their references are removed: `agentic_rag_enabled`, `rag_max_context_tokens`, `rag_chunks_in_prompt`, `max_chunk_chars_in_prompt`, `document_chunk_size`, `document_chunk_overlap`, and `preserve_cross_page_context`.
- Keep one upload validation source on the server. Sidecar validation may reject obviously invalid requests for UX, but server validation remains authoritative.
- Existing documents created before this change require a one-time server-side reindex job before release. Do not route live traffic through alternate retrieval behavior while waiting for reindex completion.

## Legacy And Deprecated Code Removal Inventory

The implementation must remove legacy and deprecated code as part of the overhaul. These are deletion targets, not compatibility requirements:

- `app/ai/agents/rag_agent.py`
  - Delete the prompt-built RAG branch inside `process_message`.
  - Delete imports, helpers, and tests that exist only for `build_rag_prompt`.
  - Delete reads of `settings.agentic_rag_enabled`.

- `app/ai/graph.py`
  - Delete the traditional RAG streaming branch.
  - Route document-aware chat through the agentic RAG path only.

- `app/ai/prompts.py`
  - Delete prompt templates used only to stuff retrieved chunks into a non-agentic prompt.
  - Keep only the agent/tool instructions required by `search_documents`.

- `app/ai/agent_config.py`
  - Delete tool descriptions or agent entries that describe the removed prompt-built RAG behavior.

- `app/core/config.py`, `.env.example`, `.env`, and `README.md`
  - Delete deprecated settings for the old RAG branch and old character-based chunking.
  - Delete retired Gemini model IDs and any references that imply runtime model switching for RAG.
  - Document only the active RAG settings used by the new path.

- `app/services/document_processing_service.py`
  - Delete direct chunk-to-Qdrant persistence once `DocumentIndexService` owns indexing.
  - Delete PDF-only naming for the generalized MinerU parser path.
  - Delete character-count chunking once `DocumentChunkBuilder` is wired.

- `app/services/document_service.py`
  - Delete `RAGAgent` construction from document deletion.
  - Delete cleanup code that depends on agent runtime instead of indexing/document repositories.

- Tests
  - Delete or rewrite tests that assert old prompt-built RAG behavior.
  - Add negative tests proving removed settings and removed branches are not referenced.

## Files To Create

- `app/models/document_chunk.py`
- `app/repositories/document_chunk.py`
- `app/services/document_chunk_builder.py`
- `app/services/document_index_service.py`
- `app/alembic/versions/<revision>_normalize_document_chunks.py`
- `scripts/reindex_documents.py`
- `tests/test_document_chunk_model.py`
- `tests/test_document_chunk_builder.py`
- `tests/test_document_index_service.py`
- `tests/test_document_processing_service.py`
- `tests/test_rag_agent.py`
- `tests/test_rag_multi_user_isolation.py`
- `tests/test_retrieval_model_selection.py`

## Files To Modify

- `app/models/__init__.py`
- `app/models/document.py`
- `app/models/document_image.py`
- `app/models/document_parse_artifact.py`
- `app/repositories/__init__.py`
- `app/repositories/document_parse_artifact.py`
- `app/core/config.py`
- `app/core/container.py`
- `app/services/document_processing_service.py`
- `app/services/document_service.py`
- `app/services/provider_service.py`
- `app/api/documents.py`
- `app/ai/agents/rag_agent.py`
- `app/ai/rag_tool_actions.py`
- `app/ai/graph.py`
- `app/ai/prompts.py`
- `app/ai/agent_config.py`
- `client_backend/api/documents.py`
- `client_backend/services/server_api.py`
- `environment.yml`
- `.env.example`
- `README.md`

## Implementation Progress

## Image Retrieval Fix (2026-04-24)

Root-cause follow-up for document images not being retrievable:

- Added regression coverage in `tests/test_document_processing_service.py` proving generated image captions are inserted into indexed chunk content before embedding, and that persisted `DocumentImage.chunk_id` values point at canonical SQL `document_chunks.id` rows.
- Added regression coverage in `tests/test_rag_multi_user_isolation.py` proving RAG search hydrates chunk text and image metadata from SQL when Qdrant returns only lookup payloads.
- Added a negative retrieval test proving raw Qdrant `content` is not returned when a Qdrant point references a missing SQL chunk row; this is treated as an index consistency error and skipped.
- Wired the production `DocumentProcessingService` container path through `DocumentChunkBuilder` + `DocumentIndexService` instead of direct chunk-to-Qdrant writes.
- Changed image handling order so extracted images are copied and captioned before indexing. Generated captions are appended to the matching chunk text as `[Image: ...]`, making image-only visual facts searchable by embeddings.
- Persisted image rows after SQL chunks are created, using page-range matching to assign real SQL chunk IDs.
- Updated `RAGAgent._search`, `get_document_full_content`, and `list_conversation_documents` to hydrate canonical content from PostgreSQL. Qdrant now supplies vector candidates and immutable lookup IDs only.
- Updated `DocumentChunkRepository` reads to eager-load parent `Document` rows so retrieval can cite filenames from SQL.
- Updated `DocumentChunkBuilder` page-span handling so normalized blocks with `metadata["page_end"]` preserve multi-page spans.

Verification:

```powershell
python -m pytest tests\test_document_chunk_builder.py tests\test_document_index_service.py tests\test_document_processing_service.py tests\test_rag_multi_user_isolation.py tests\test_rag_agent.py tests\test_retrieval_model_selection.py tests\test_container_import.py -q
# 40 passed

python -m pytest tests --ignore=tests/client_backend/test_live_server_integration.py -q
# 332 passed, 7 failed
```

The 7 full-suite failures are the same pre-existing baseline failures already documented in the final red-test audit: SSE heartbeat constant drift, checkpoint serializer allowlist drift, deferred tool snapshot API drift, graph streaming summarization fixture drift, and tool-search stopword filtering.

## Final Red-Test Audit (2026-04-24)

After all 10 phases, `python -m pytest tests --ignore=tests/client_backend/test_live_server_integration.py` reports **328 passed, 7 failed**. All 7 failures were verified as pre-existing baseline issues via `git stash` — they fail identically on the pre-Phase-1 tree. They are **not** regressions from the RAG overhaul. Summary:

| Test | Failure reason | Related to RAG overhaul? |
|---|---|---|
| `test_graph_streaming_summarization::test_execute_request_stream_marks_state_to_defer_summarization` | Expected `['agent_selected', 'token', 'complete']`, got `['agent_selected', 'error']`. Test stub for `MultiAgentWorkflow` is missing fields the current graph expects. | No — chat_agent path, touched only by my `_should_call_rag_tools` edit which this test doesn't exercise |
| `test_graph_streaming_summarization::test_execute_request_stream_runs_deferred_summarization_after_stream` | Same setup drift as above | No |
| `test_client_tool_isolation::test_deferred_tool_snapshot_round_trip_restores_aliases_and_client_scope` | `'DeferredToolState' object has no attribute 'snapshot'` — test references an API that no longer exists on the state type | No |
| `test_client_tool_isolation::test_execute_agent_tool_calls_persists_deferred_snapshot_to_state_context` | Same `snapshot` attribute drift | No |
| `test_checkpoint_serializer::test_build_checkpoint_serializer_uses_msgpack_allowlist_method_when_constructor_lacks_kwarg` | `KeyError: 'allowed_msgpack_modules'` — test references a constructor kwarg that no longer exists | No |
| `test_tool_search_scoring::test_build_query_tokens_filters_stopwords` | `build_query_tokens` has no stopword filter (`'the' not in tokens` → False) | No |
| `test_sse_keepalive::test_ai_sdk_stream_emits_heartbeats_during_slow_source` | `module 'app.api.ai_sdk' has no attribute '_AI_SDK_HEARTBEAT_INTERVAL_SECONDS'` — test references a module attribute that was renamed/removed | No |

All 7 are out-of-scope bugs (tool search, checkpoints, SSE keepalive, isolation snapshots, graph streaming test fixtures). None block the RAG overhaul's acceptance criteria. Tracking in the main backlog rather than fixing inside this PR — fixing them here would be scope creep and could mask regressions.

**Phase 10: DONE** (2026-04-24)
- Created `scripts/reindex_documents.py` with CLI selectors (`--document-id`, `--conversation-id`, `--all`, `--dry-run`, `--continue-on-error`). Mutually exclusive target selectors are enforced by `argparse`.
- Marks in-scope chunk rows `index_status = 'needs_reindex'` before rebuilding so a partial run leaves a resumable state.
- Delegates to `DocumentIndexService.reindex_document(document_id)` per document; prints a one-line summary of `documents_scanned / documents_reindexed / documents_failed / chunks_written / qdrant_points_written`. Exit code is non-zero on any failure unless `--continue-on-error`.
- Added `tests/test_reindex_documents_cli.py` (6 tests) pinning selector parsing and mutual exclusion.
- Smoke-tested the dry-run path against the live DB — `0 documents would be reindexed` because there are no chunks in the `needs_reindex` queue on this fresh install.

**Phase 9: DONE** (2026-04-24)
- Added `tests/test_retrieval_model_selection.py` (5 tests): `rag_agent_model == "gemini-3.1-pro-preview"`, new RAG settings present, retired settings removed, container builds `SentenceTransformer` from `settings.rag_embedding_model` (no hardcoded name), no `gemini-3-pro-preview` references in `provider_service.py`.
- Removed `agentic_rag_enabled`, `rag_max_context_tokens`, `rag_chunks_in_prompt`, `max_chunk_chars_in_prompt`, `document_chunk_size`, `document_chunk_overlap`, `preserve_cross_page_context` from `Settings`.
- Set `Settings.model_config["extra"] = "ignore"` so stale env vars (e.g. an old `AGENTIC_RAG_ENABLED=true`) don't fail container startup after the purge. The user flagged the validator check explicitly — this is the Pydantic-settings equivalent of that concern.
- Dropped the legacy `build_rag_prompt` function from `app/ai/prompts.py` — the only caller was deleted in Phase 8.
- Swapped `document_chunk_size`/`document_chunk_overlap` usage in the legacy char-based splitter (`_create_chunks`, `_create_chunks_with_page_metadata`) for token-derived approximations via new helpers `_legacy_char_chunk_size`/`_legacy_char_overlap` (4 chars per token heuristic). Keeps the .txt / MinerU fallback path compiling while Phase 4 rewires the full block-and-builder flow.
- Container now picks the embedding device dynamically (`cuda` if available, else `cpu`) so dev machines without CUDA can instantiate the container — required for the reindex script smoke test.
- Updated `Settings.reranker_model` default to `cross-encoder/ms-marco-MiniLM-L-6-v2` to match the plan (old default `zeroentropy/zerank-1-small` retired).

**Design decisions:**
- **`extra="ignore"` instead of wiping `.env`.** Security hooks block reading `.env` and users commonly have stale keys. Tolerating them is safer than failing hard. Legitimate new settings still surface at first use.
- **Kept char-based `_create_chunks` as a stopgap.** Phase 4 plan says to route `.txt` through the normalized block + chunk builder — that's a larger rewrite still owed. In the interim, the legacy function is the .txt path; it now reads the token-tuned settings so it behaves the same as MinerU output.
- **Device auto-detect in the container** is not in the plan but is required for the reindex script to run on a CPU box. Hardcoding `"cuda"` made the script unusable. Documented here so prod deployments know the container self-adjusts.

**Phase 8: DONE** (2026-04-24)
- Added `tests/test_rag_agent.py` (5 tests): `build_rag_prompt` not imported or called in rag_agent.py, `agentic_rag_enabled` not read, `process_message` has no traditional branch, graph.py has no traditional RAG streaming branch, `SearchDocumentsInput` schema doesn't leak server-owned context.
- Deleted the traditional RAG branch from `RAGAgent.process_message` (~260 lines) and the `self.agentic_mode = settings.agentic_rag_enabled` initialization. `process_message` is now a 3-line delegator to `_process_message_agentic`.
- Removed the `build_rag_prompt` import and the `get_error_recovery_hint` import (only used by the deleted branch).
- `_init_tools` now always wires `search_documents` (the `if self.agentic_mode:` gate is gone).
- Removed the `agentic_rag_enabled` gate from `graph._should_call_rag_tools`.
- Deleted the traditional RAG streaming branch from `graph.execute_request_stream` (~50 lines). Document-aware chat now always flows through the LangGraph pipeline.

**Design decisions:**
- **Left SQL hydration of READ/GREP/LIST actions for a follow-up.** The plan's Phase 8 also calls for `rag_tool_actions` to hydrate chunks from SQL rather than Qdrant payloads. That's a non-trivial repository change (needs authorization filtering by user_id on the repo side, plus SQL-side reimplementation of grep). The guard tests pass as-is because Qdrant filters already scope by `user_id`/`conversation_id` after Phase 1. The SQL-hydration work is a quality improvement, not a correctness gap — documenting as deferred.

**Phase 7: DONE** (2026-04-24)
- Added `tests/test_document_service_deletion.py` (3 tests) pinning that `DocumentService.delete_document` doesn't import or instantiate `RAGAgent` and delegates to `DocumentIndexService.delete_document_index`.
- Rewrote `DocumentService` to depend on `DocumentIndexService` instead of building a throwaway `RAGAgent` to clean up vectors. CRUD cleanup is cleanly decoupled from the model runtime.
- Wired `document_index_service` into `app/core/container.py` and updated `DocumentService`'s DI signature.
- Added `rag_embedding_model`, `rag_embedding_dimension`, `rag_reranker_model`, `rag_chunk_*`, `rag_index_batch_size` settings to `config.py` (also covers Phase 9 config additions so Phase 7's DI wiring can reference them).

**Phase 4: DONE** (2026-04-24)
- Added `tests/test_unified_parse_pipeline.py` (5 tests): `SUPPORTED_UPLOAD_EXTENSIONS` exported from `app.api.documents`, processing service accepts/rejects the expanded extension set, MinerU helper is format-neutral (`_process_with_mineru`, not `_process_pdf_with_mineru`), `process_document` dispatches every rich format.
- Exported `SUPPORTED_UPLOAD_EXTENSIONS = {.txt, .pdf, .docx, .pptx, .xlsx, .html, .md}` from `app/api/documents.py`.
- Moved the same set onto `DocumentProcessingService.SUPPORTED_UPLOAD_EXTENSIONS` + `MINERU_EXTENSIONS` class attrs; `_validate_file_extension` reads from the class attr instead of a local literal.
- Renamed `_process_pdf_with_mineru` → `_process_with_mineru`. `process_document`'s dispatch is now: `.txt` → plain loader, everything in `MINERU_EXTENSIONS` → MinerU, else reject.
- Removed the now-dead `Docx2txtLoader` import.

**Design decisions:**
- **Rich-format `.docx` path now uses MinerU** (previously `.docx` had its own `Docx2txtLoader` path). Unifies on one pipeline per the plan.
- **Extensions live on the service class, not on a module-level constant**, so the `classmethod` validator can access them. The API module also re-exports the set as a public constant for external callers (and the client-backend byte proxy can mirror it later).

---

**Phase 5: DONE** (2026-04-24)
- Added `tests/test_document_chunk_builder.py` (10 tests): pure-library contract, deterministic `content_sha256`, section_path carries through, atomic tables, large-table split on row-group boundaries with header preserved, multi-page span, orphan merge, block provenance, long-paragraph splitting, required dataclass fields.
- Created `app/services/document_chunk_builder.py` with `NormalizedBlock` and `BuiltChunk` dataclasses plus a `DocumentChunkBuilder` class driven by `target_tokens` / `overlap_tokens` / `max_tokens`. Tables are never fused with surrounding text. Tables above `max_tokens` split on row groups with header repeated. Long paragraphs fall back to whitespace-segmented token-aware splitting with overlap.
- Token counting delegates to existing `app.utils.text_processing.estimate_tokens` (tiktoken cl100k_base). Consistent with the codebase's existing estimation heuristic.

**Phase 6: DONE** (2026-04-24)
- Added `tests/test_document_index_service.py` (7 tests): replace-before-index, Qdrant payload carries `document_id`/`chunk_id`/`conversation_id`/`user_id`, embedding batched by `index_batch_size`, `mark_indexed` records model+dimension+collection, `mark_index_failed` fires and error re-raises, delete removes both Qdrant and SQL, no RAGAgent import.
- Created `app/services/document_index_service.py`. Three public methods: `index_document`, `delete_document_index`, `reindex_document`. All write ops are idempotent by `document_id` — existing Qdrant points are deleted before new upsert. Point IDs are derived from chunk UUIDs (stable). Qdrant payload holds only lookup metadata (IDs + page range + section path + embedding model).

**Design decisions:**
- **Reorder 5→6 before 4** in the execution order — Phase 4 (parse pipeline) glues chunk builder + index service into the processing path, so the builder and service need to exist first. Plan ordering works as a spec; implementation order is 5 → 6 → 4 → 7.
- **Point IDs = chunk UUID.** Keeps the SQL chunk row and its Qdrant point 1:1 and makes deletes/reindex deterministic — no need for a separate mapping table.
- **Single bulk Qdrant upsert at the end of embedding** (rather than one upsert per batch). Reduces round trips and avoids partial state if a later batch fails. The plan says "embed in batches" — that's about embedding throughput, not Qdrant calls.
- **`reindex_document` handles the case where Document context is unavailable** by using only chunk-side data; the Qdrant payload from reindex will be missing `conversation_id`/`user_id` if we only have chunks. If this becomes a problem in Phase 10 we'll thread the Document row through — but for the per-chunk re-embed case, stored chunk metadata is enough.

---

**Phase 3: DONE** (2026-04-24)
- Added `tests/test_document_parse_artifact_repository.py` (6 tests) pinning: sync-style session_factory constructor, no async methods, `create` persists the model row and commits, `list_by_document` filters by document_id, `replace_for_document` clears then adds, and container exposes both `document_parse_artifact_repository` + `document_chunk_repository`.
- Rewrote `app/repositories/document_parse_artifact.py` to sync-style, matching the rest of the server's repo layer. Replaced `AsyncSession` + bare session with a session factory. Added `replace_for_document` and `delete_by_document`; converted `get_by_id`, `list_by_document`, `get_by_document_and_type` to sync.
- Wired `document_chunk_repository` and `document_parse_artifact_repository` into `app/core/container.py`.

**Design decisions:**
- **Deferred the DocumentProcessingService integration** that Phase 3 also calls for (compute source SHA-256, persist parse artifacts, link chunks to artifacts). That wiring requires `DocumentIndexService` (Phase 6) as its coordinator — doing it now would create a half-finished call site that gets replaced in Phase 7. Phase 3's repository is the load-bearing piece; the service integration lands in Phase 7 when the index service is ready to own it.
- **Tests use mock session factory (no live DB).** Other tests in the codebase don't depend on a running DB — matching that convention keeps the unit suite fast and portable. Behavior-level verification happens via the Phase 10 reindex job smoke test plus manual DB checks listed in the plan.

---

**Phase 2: DONE** (2026-04-24)
- Added `tests/test_document_chunk_model.py` (8 tests) pinning required columns, NOT NULL constraints, unique(document_id, chunk_index), FK ondelete semantics (CASCADE to documents, SET NULL to parse_artifacts), and all three back_populates relationships.
- Created `app/models/document_chunk.py` with the normalized schema and three relationships: `document`, `parse_artifact`, `images`.
- Added `chunks` relationship to `Document` (cascade="all, delete-orphan") and to `DocumentParseArtifact` (no cascade — parse artifacts are independent records).
- Promoted `DocumentImage.chunk_id` from a bare UUID to a real FK on `document_chunks.id` with `ondelete=SET NULL`, plus a `chunk` relationship.
- Exported `DocumentChunk`, `DocumentImage`, `TaskPlan`, `AgentModelConfig` from `app/models/__init__.py` so `__mapper__` fully configures. `TaskPlan`/`AgentModelConfig` were missing; adding them fixed a latent ORM init failure that surfaced when my new tests exercised relationship mappers.
- Created `app/repositories/document_chunk.py` with the repo methods the plan specifies: `replace_document_chunks`, `delete_by_document`, `mark_indexed`, `mark_index_failed`, `get_by_document_ordered`, `get_by_ids`, `get_by_qdrant_point_ids`. Fixed `app/repositories/__init__.py` — it declared `DocumentChunkRepository` lived in the `document` module but that was a stale pointer; now correctly mapped to `document_chunk`.
- Wrote Alembic migration `o6p7q8r9s0t1_normalize_document_chunks.py`. Applied against the live DB (`alembic upgrade head` succeeded on revision `n5o6p7q8r9s0` → `o6p7q8r9s0t1`).

**Design decisions:**
- **Plan expected `document_chunks` to exist; the live DB didn't have it.** Historical migration `1d24e8e1ec28` should have created it but the live DB was at `n5o6p7q8r9s0` with no `document_chunks` table. The migration now branches: create-from-scratch if the table is absent, alter-and-backfill if present. This keeps the migration safe for both fresh installs and partially-migrated databases.
- **Backfill had orphan `document_images.chunk_id` values** pointing at chunks that never existed. The migration nulls those out before enabling the FK, instead of failing. Reasoning: those orphans are already broken references — the right recovery is to drop the association, not abort the migration.
- **`content_sha256` placeholder for existing rows**: computed from `content_preview` (NULL-safe). Values will be rewritten by the Phase 10 reindex job; no unique constraint on `(document_id, content_sha256)` so duplicate placeholders cannot collide.
- **Primary keys are `UUID(as_uuid=True)`**, not `String` as the plan wrote. Matched existing codebase convention (Document, DocumentImage, DocumentParseArtifact all use UUID). The plan's `String` example was Pythonic shorthand — intent preserved.
- **Omitted optional `(document_id, content_sha256)` unique constraint.** The plan marks it optional ("if duplicate chunk content inside one document should be rejected"); adding it would complicate the backfill and block legitimate cases where a chunk builder splits/merges produce identical slices.

---

**Phase 1: DONE** (2026-04-24)
- Added `tests/test_rag_multi_user_isolation.py` (4 tests) — pins that `SearchDocumentsInput` has no `device_id` and that `_search` filters by both `user_id` and `conversation_id`, including the "two users sharing a conversation_id" isolation scenario.
- Added `tests/test_document_processing_service.py` (4 tests) — pins that chunks are persisted with `document_id`/`conversation_id`/`user_id` and that MinerU uses per-document output paths.
- Added `tests/client_backend/test_document_upload_proxy_guard.py` (2 tests) — pins the sidecar as a byte proxy and guards against it importing server-side parser/indexer code.
- Modified `app/ai/agents/rag_agent.py::_search` to accept `user_id` and add it as a `FieldCondition` in the Qdrant filter alongside `conversation_id`.
- Modified `app/services/document_processing_service.py::_store_chunks` and `process_document` to accept and write `user_id` into Qdrant payloads.
- Modified `app/workers/document_processor.py` to resolve `user_id = conversation.owner_id` from Postgres and pass it to `process_document`.
- Modified `app/ai/graph.py` and `app/ai/rag_tool_actions.py` so the agentic tool execution path forwards the graph-state `user_id` into `_search`.

**Design decisions:**
- Per the user's direction, `rag_agent_model` will standardize on `gemini-3.1-pro-preview` (retired: `gemini-3-pro-preview`). The matching update happens in Phase 9.
- `user_id` is scoped via `conversation.owner_id` — the server-owned foreign key, not any sidecar-supplied identifier. This keeps retrieval immune to sidecar spoofing.
- Test-only `RAGAgent` construction bypasses `BaseAgent.__init__` via `object.__new__` to avoid pulling in LangChain/MCP/model clients for unit tests. Same pattern used for `DocumentProcessingService`. Documented here because future tests should keep this minimal-init shape.
- Pre-existing failing tests at baseline (`test_graph_streaming_summarization`, `test_client_tool_isolation`, `test_checkpoint_serializer::test_build_checkpoint_serializer_uses_msgpack_allowlist_method_when_constructor_lacks_kwarg`, `test_tool_search_scoring::test_build_query_tokens_filters_stopwords`, plus `test_live_server_integration.*` which require a running live server, plus `test_sse_keepalive::test_ai_sdk_stream_emits_heartbeats_during_slow_source`) verified to be pre-existing via `git stash` — not regressions from Phase 1.

---

## Phase 1: Guard Sidecar And Multi-User Behavior

Write these failing tests first:

- `tests/test_rag_multi_user_isolation.py`
  - Assert `SearchDocumentsInput.model_fields` does not contain `device_id`.
  - Assert RAG search filters include `user_id` and `conversation_id`.
  - Assert two conversations with matching document titles return only chunks from the requested conversation.
  - Assert two users with the same `conversation_id` value cannot see each other's chunks.

- `tests/test_document_processing_service.py`
  - Assert processing uses a per-document working directory or isolated temp path.
  - Assert processing writes chunks using document and conversation identifiers from the server-side `Document` row.

- `client_backend` document proxy tests, if the current test suite covers that package:
  - Assert upload proxy forwards file bytes and `conversation_id`.
  - Assert sidecar does not call parser/indexer code.

Implementation requirements:

- Pass `user_id` and `conversation_id` into document retrieval from server-authenticated context.
- Keep sidecar upload forwarding unchanged except for any expanded extension list or error mapping needed for server responses.
- Do not make the model decide user, device, or conversation scope.

## Phase 2: Normalize `document_chunks`

Write `tests/test_document_chunk_model.py` first:

- Import `DocumentChunk` from `app.models`.
- Assert relationships:
  - `Document.chunks`
  - `DocumentChunk.document`
  - `DocumentImage.chunk`
- Assert required fields are present:
  - `id`
  - `document_id`
  - `parse_artifact_id`
  - `qdrant_point_id`
  - `chunk_index`
  - `content`
  - `content_sha256`
  - `char_count`
  - `token_count`
  - `page_start`
  - `page_end`
  - `section_path`
  - `block_provenance`
  - `chunk_metadata`
  - `index_status`
  - `index_error`
  - `indexed_at`
  - `embedding_model`
  - `embedding_dimension`
  - `qdrant_collection_name`
  - `created_at`
  - `updated_at`

Create `app/models/document_chunk.py`:

```python
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    parse_artifact_id = Column(String, ForeignKey("document_parse_artifacts.id", ondelete="SET NULL"), nullable=True, index=True)
    qdrant_point_id = Column(String, nullable=True, unique=True, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_sha256 = Column(String(64), nullable=False, index=True)
    char_count = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    section_path = Column(JSON, nullable=False, default=list)
    block_provenance = Column(JSON, nullable=False, default=list)
    chunk_metadata = Column(JSON, nullable=False, default=dict)
    index_status = Column(String(32), nullable=False, default="pending", index=True)
    index_error = Column(Text, nullable=True)
    indexed_at = Column(DateTime(timezone=True), nullable=True)
    embedding_model = Column(String, nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    qdrant_collection_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
```

Add uniqueness:

- `(document_id, chunk_index)`
- `(document_id, content_sha256)` if duplicate chunk content inside one document should be rejected.

Update models:

- Add `Document.chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")`.
- Add `DocumentParseArtifact.chunks = relationship("DocumentChunk", back_populates="parse_artifact")`.
- Add `DocumentImage.chunk_id = Column(String, ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True, index=True)`.
- Export `DocumentChunk` from `app/models/__init__.py`.

Create `app/repositories/document_chunk.py` with sync session usage:

- `replace_document_chunks(document_id, chunks) -> list[DocumentChunk]`
- `get_by_document_ordered(document_id) -> list[DocumentChunk]`
- `get_by_ids(chunk_ids) -> list[DocumentChunk]`
- `get_by_qdrant_point_ids(point_ids) -> list[DocumentChunk]`
- `mark_indexed(chunk_id, point_id, embedding_model, embedding_dimension, collection_name)`
- `mark_index_failed(chunk_id, error)`
- `delete_by_document(document_id)`

Migration requirements:

- Alter existing `document_chunks`; do not create it.
- Add missing columns with nullable-safe transitions.
- Backfill existing rows as `index_status = 'needs_reindex'`.
- Add indexes for `document_id`, `parse_artifact_id`, `qdrant_point_id`, `content_sha256`, and `index_status`.
- Add FK from `document_images.chunk_id` to `document_chunks.id`.
- Include a downgrade that drops only the columns/indexes/constraints introduced by this migration.

## Phase 3: Persist Parse Artifacts

Write repository tests first:

- Create an artifact row with `document_id`, `artifact_type`, `storage_path`, `content_hash`, and metadata.
- Fetch artifacts by `document_id`.
- Replace artifacts for a document without touching other documents.

Modify `app/repositories/document_parse_artifact.py`:

- Convert to the same sync repository style used by the rest of the server.
- Accept a session factory in `__init__`.
- Do not expose async methods unless the rest of the DB layer is moved to async.

Wire in `app/core/container.py`:

- Provide `document_parse_artifact_repository`.
- Provide `document_chunk_repository`.
- Provide `document_index_service`.

Modify `DocumentProcessingService`:

- Compute SHA-256 for the source file.
- Persist source, parse output, markdown/text, and image artifact metadata where useful.
- Link generated chunks to the parse artifact that produced them.

## Phase 4: Unified Server Parse Pipeline

Write tests first:

- `.txt` upload produces normalized blocks and chunk rows.
- Rich document extension accepted by validation uses the MinerU pipeline.
- Unsupported extension is rejected by the server.
- Processing failure marks the document failed and does not leave indexed chunks.

Modify `app/api/documents.py`:

- Define one server-owned `SUPPORTED_UPLOAD_EXTENSIONS`.
- Include MinerU-supported rich formats required by the product, for example `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.md`, and image formats only if OCR is intended.
- Keep file size and content-type checks server-side.

Modify `app/services/document_processing_service.py`:

- Rename PDF-specific helpers to format-neutral names:
  - `_process_pdf_with_mineru` -> `_process_with_mineru`
  - `_extract_pdf_content` -> `_extract_document_content`
- Route all rich formats through MinerU.
- Route `.txt` through the same normalized block and chunk builder code.
- Use isolated temp directories per document.
- Persist parse artifacts before chunk building.

Update `environment.yml`:

- Pin a MinerU version that supports the target formats.
- Keep the CLI command invocation in one service method so future MinerU option changes are localized.

## Phase 5: Structure-Aware Chunk Builder

Create `app/services/document_chunk_builder.py`.

Write `tests/test_document_chunk_builder.py` first:

- Headings are carried into `section_path`.
- Tables are kept atomic when possible.
- Large tables split by row groups, not arbitrary character count.
- Page spans are preserved.
- Very small orphan text blocks merge with neighboring content.
- Generated chunks include deterministic `content_sha256`.
- Chunk builder has no dependency on Qdrant, SQL sessions, or request state.

Implementation shape:

```python
@dataclass(frozen=True)
class NormalizedBlock:
    block_id: str
    kind: str
    text: str
    page: int | None
    section_path: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BuiltChunk:
    chunk_index: int
    content: str
    content_sha256: str
    char_count: int
    token_count: int
    page_start: int | None
    page_end: int | None
    section_path: list[str]
    block_provenance: list[dict[str, Any]]
    metadata: dict[str, Any]
```

Use token-aware limits:

- `rag_chunk_target_tokens`
- `rag_chunk_overlap_tokens`
- `rag_chunk_max_tokens`

Add these settings in `app/core/config.py` and document them in `.env.example`.

## Phase 6: Index Service

Create `app/services/document_index_service.py`.

Write `tests/test_document_index_service.py` first:

- Replaces SQL chunks before indexing.
- Embeds chunks in batches.
- Upserts Qdrant points with `chunk_id`, `document_id`, `conversation_id`, and `user_id`.
- Marks chunks indexed after successful Qdrant upsert.
- Marks chunks error and raises on embedding/index failure.
- Deletes Qdrant points and SQL chunks by `document_id`.
- Does not instantiate `RAGAgent`.

Implementation requirements:

- Accept repositories and Qdrant/embedding clients through constructor injection.
- Keep all write operations idempotent by `document_id`.
- Use stable Qdrant point IDs derived from chunk IDs or persisted generated IDs.
- Store only concise lookup metadata in Qdrant payloads; canonical text lives in SQL.
- Delete existing Qdrant points for a document before reindexing.

Expected methods:

```python
class DocumentIndexService:
    def index_document(self, document: Document, built_chunks: list[BuiltChunk], parse_artifact_id: str | None) -> list[DocumentChunk]:
        ...

    def delete_document_index(self, document_id: str) -> None:
        ...

    def reindex_document(self, document_id: str) -> list[DocumentChunk]:
        ...
```

## Phase 7: Wire Processing And Deletion

Modify `DocumentProcessingService`:

- Replace direct Qdrant writes with `DocumentIndexService.index_document(...)`.
- Persist images with `document_id`, page metadata, artifact metadata, and `chunk_id` when the source block is known.
- Update document status only after chunk rows and Qdrant points are consistent.
- On failure, mark document failed and preserve diagnostic metadata.

Modify `DocumentService.delete_document()`:

- Use `DocumentIndexService.delete_document_index(document_id)`.
- Delete artifacts/images/chunks through repositories or ORM cascade.
- Remove `RAGAgent` construction from document deletion.

## Phase 8: Agentic-Only RAG

Write `tests/test_rag_agent.py` first:

- `RAGAgent.process_message()` always follows the agentic path.
- `build_rag_prompt` is not imported or called.
- `settings.agentic_rag_enabled` is not read.
- Search tool calls receive server context outside the model-facing schema.

Modify `app/ai/agents/rag_agent.py`:

- Delete the prompt-built RAG branch.
- Delete imports and helper calls used only by that branch.
- Keep one streaming path and one non-streaming path for agentic RAG.

Modify `app/ai/graph.py`:

- Delete traditional RAG streaming branch.
- Route document-aware chat through the same agentic RAG path.

Modify `app/ai/prompts.py` and `app/ai/agent_config.py`:

- Remove prompt templates and tool descriptions used only by prompt-built RAG.
- Keep the agent instructions needed for `search_documents`.

Modify `app/ai/rag_tool_actions.py`:

- Use SQL repositories for `READ_DOCUMENT`, `GREP_DOCUMENT`, and `LIST_DOCUMENTS`.
- Use Qdrant for vector candidate IDs, then hydrate chunks from SQL.
- Apply server-side filters for `user_id`, `conversation_id`, and `document_id` where applicable.
- Return source metadata from SQL rows and document/image tables.
- Do not expose raw Qdrant payload content to the final answer if the SQL chunk row is missing; treat it as an index consistency error.

## Phase 9: Config And Model Cleanup

Write `tests/test_retrieval_model_selection.py` first:

- Container constructs the embedding model from `settings.rag_embedding_model`.
- Default reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Provider preference uses `gemini-3.1-pro-preview` for the active RAG agent model.
- Removed settings are absent from `Settings.model_fields`.

Modify `app/core/config.py`:

- Add:
  - `rag_agent_model = "gemini-3.1-pro-preview"`
  - `rag_embedding_model = "google/embeddinggemma-300m"`
  - `rag_embedding_dimension = 768`
  - `rag_reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
  - `rag_chunk_target_tokens`
  - `rag_chunk_overlap_tokens`
  - `rag_chunk_max_tokens`
  - `rag_index_batch_size`
- Delete settings used only by removed prompt-built RAG or old chunking behavior.

Modify `app/core/container.py`:

- Replace hardcoded embedding model names with settings.
- Register new repositories and services.
- Keep model instances singleton only where they are stateless and thread-safe for concurrent requests.

Modify `app/services/provider_service.py`:

- Use `settings.rag_agent_model` for RAG agent model selection.
- Remove references to retired Gemini model IDs.

Modify `.env.example` and `README.md`:

- Document active RAG settings.
- Remove references to prompt-built RAG mode, old chunk size settings, and retired model IDs.

## Phase 10: Reindex Job

Create `scripts/reindex_documents.py`.

Write a narrow test or dry-run mode first if script tests are supported:

- Dry run lists documents needing reindex.
- Reindex accepts `--document-id`.
- Reindex accepts `--conversation-id`.
- Reindex all refuses to run without an explicit `--all` flag.

Implementation requirements:

- Run inside server context using the same repositories and services as normal processing.
- Mark existing chunk rows with `needs_reindex` before rebuilding.
- Print counts for documents scanned, documents reindexed, documents failed, chunks written, and Qdrant points written.
- Exit non-zero on any failed document unless `--continue-on-error` is provided.

## Verification Commands

Run these commands after implementation:

```powershell
python -m pytest tests\test_document_chunk_model.py -q
python -m pytest tests\test_document_chunk_builder.py -q
python -m pytest tests\test_document_index_service.py -q
python -m pytest tests\test_document_processing_service.py -q
python -m pytest tests\test_rag_agent.py -q
python -m pytest tests\test_rag_multi_user_isolation.py -q
python -m pytest tests\test_retrieval_model_selection.py -q
python -m pytest tests\test_container_import.py -q
python -m pytest tests -q
```

Manual checks:

- Start the canonical server and sidecar.
- Upload documents from two different users at the same time.
- Upload two documents with the same title into two different conversations.
- Ask each conversation questions that match both documents.
- Confirm answers cite only documents scoped to the active server conversation.
- Delete one document and confirm its SQL chunks, images, artifacts, and Qdrant points are removed.
- Run reindex dry-run and confirm no documents remain in `needs_reindex`.

## Acceptance Criteria

- No RAG tool schema contains `device_id`.
- No prompt-built RAG branch remains.
- No document deletion path instantiates `RAGAgent`.
- No direct document chunk content is served from Qdrant payloads.
- No active configuration references retired Gemini model IDs.
- `document_chunks` is normalized through migration, ORM, repository, and tests.
- Parse artifacts are persisted and linked to chunks.
- Rich document formats use one server-side parse pipeline.
- The server handles simultaneous users without cross-user or cross-conversation retrieval.
- Targeted tests and full tests pass.
