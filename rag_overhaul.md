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
