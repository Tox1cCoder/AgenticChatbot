# RAG Upload And Worker Parallelism Implementation Plan

Branch: current working tree | Date: 2026-05-19 | Input: add multi-document upload, same-conversation duplicate filename rejection, Streamlit batch upload, and real Celery document-processing parallelism.

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` before implementation changes. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make document ingestion production-ready for batch uploads by accepting multiple files in one request, rejecting duplicate filenames in the same conversation without blocking valid siblings, updating the Streamlit demo to upload multiple files at once, and making Celery document processing actually run concurrently under configurable worker settings.

**Architecture:** Add one canonical batch upload service path and let single-file behavior become a thin compatibility wrapper rather than a separate implementation. Enforce duplicate filename protection at both service and database levels using a normalized per-conversation filename key. Make Celery worker startup config-driven and switch Windows local runs away from the `solo` pool, while preserving task-local document processing state.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, SQLAlchemy/Alembic, Celery 5, Redis broker/result backend, Streamlit, pytest, unittest.mock.

---

## Current Implementation Audit

- `app/api/documents.py` exposes `POST /documents/upload` with one `UploadFile` named `file`.
- `client_backend/api/documents.py` exposes the same single-file proxy and reads the entire uploaded file into memory before forwarding bytes to the canonical server.
- `client_backend/services/server_api.py` has `upload_document_bytes(...)`, which posts to `/documents/upload` with one file.
- `upload_support.py` has one `upload_document(uploaded_file)` helper and one-file sidebar upload UI.
- `demo.py::render_documents_tab()` uses `st.file_uploader(..., accept_multiple_files=False)` implicitly and calls `upload_document(uploaded_file)`.
- `DocumentService.validate_and_create_document(...)` validates size/type and creates a `Document` row, but it does not check duplicate filenames.
- `DocumentRepository.create(...)` does not have a race-safe uniqueness constraint for `(conversation_id, filename)`.
- `Document` stores `filename`, but no normalized filename key exists for case-insensitive same-name detection.
- `app/workers/start_worker.py` hard-codes `--concurrency=2`, then appends `--pool=solo` on Windows. Celery `solo` runs one task at a time, so Windows local processing appears sequential even when `--concurrency=2` is present.
- `app/workers/celery_app.py` hard-codes `worker_prefetch_multiplier = 1`, `task_acks_late = True`, and `task_reject_on_worker_lost = True`. These are reasonable defaults, but they are not configurable.
- `app/core/config.py` has broker/result backend settings, but no worker pool, concurrency, time-limit, or max-tasks-per-child settings.
- `app/core/container.py` uses `providers.Factory` for `document_processing_service`, so each API request or Celery task receives a fresh `DocumentProcessingService` instance. Keep this property because the service has per-run mutable fields such as `_extracted_images` and `_mineru_output_path`.

## Clarified Requirements

- The canonical API should support uploading multiple document files in one request.
- Duplicate filename checks are scoped to the same conversation.
- Duplicate files should be rejected, not replaced.
- In a batch upload, duplicate or invalid files should be rejected per file while valid, non-duplicate files continue to be staged, persisted, and enqueued.
- Celery document-processing workers are the parallelism target, not Planning-mode subagents.
- Old duplicate upload implementations should not be copied forward. If compatibility endpoints remain, they must be thin wrappers around the canonical path.

## Non-Goals

- Do not change RAG retrieval behavior, chunking, embeddings, or Qdrant query logic except where upload responses or document metadata require it.
- Do not move document parsing/indexing into `client_backend`; it remains a byte proxy.
- Do not add background batch-job polling. Each uploaded file already has an individual Celery task id.
- Do not make generic LangGraph tool execution parallel.
- Do not introduce a separate upload queue table unless the existing `documents.processing_task_id` path proves insufficient.

## Design Decisions

### Decision 1: Add `POST /documents/uploads` as the canonical batch endpoint

Use a new canonical route:

```http
POST /documents/uploads
Content-Type: multipart/form-data

conversation_id=<uuid>
files=<file1>
files=<file2>
```

Rationale:

- A plural resource route communicates batch semantics clearly.
- The existing single-file `/documents/upload` route can call the same service helper and return the first item in the canonical result for compatibility.
- Streamlit and the sidecar should move to `/documents/uploads` so new code has one path.
- This avoids a messy endpoint that accepts both `file` and `files` fields as first-class code paths.

Compatibility stance:

- Keep `/documents/upload` only as a thin wrapper during this change because API clients, tests, and docs currently depend on it.
- Do not keep duplicate staging, duplicate validation, or task enqueue code in the wrapper.
- Mark `/documents/upload` as legacy in README after the UI and sidecar use `/documents/uploads`.

### Decision 2: Enforce duplicate filenames with a normalized key

Add `documents.filename_key` and a unique constraint on `(conversation_id, filename_key)`.

Normalization function:

```python
def normalize_document_filename(filename: str) -> str:
    name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = unicodedata.normalize("NFC", name)
    return name.casefold()
```

Rationale:

- Same-conversation duplicate checks must be race-safe when two uploads arrive at once.
- Case-folding matches user expectations on Windows and avoids `Report.pdf` / `report.pdf` duplicates.
- Keeping the original `filename` preserves display behavior.

Migration policy:

- Backfill `filename_key` from existing `filename`.
- If historical duplicates exist in the same conversation, keep the newest row on the plain key and suffix older duplicate keys with `::legacy::<document_id>` so the unique constraint can be created without deleting data.
- New uploads must reject when `filename_key` already exists for the conversation.

### Decision 3: Return per-file results

Batch response data should be ordered like the input files:

```json
{
  "conversation_id": "conv-id",
  "total_count": 3,
  "accepted_count": 2,
  "rejected_count": 1,
  "files": [
    {
      "filename": "alpha.pdf",
      "status": "accepted",
      "document": { "id": "doc-id", "filename": "alpha.pdf" },
      "processing": { "task_id": "celery-task-id" }
    },
    {
      "filename": "alpha.pdf",
      "status": "rejected",
      "error_code": "DUPLICATE_FILENAME",
      "message": "A document named 'alpha.pdf' already exists in this conversation."
    }
  ]
}
```

HTTP status policy:

- `201 Created` when at least one file is accepted and none are rejected.
- `207 Multi-Status` when at least one file is accepted and at least one file is rejected.
- `409 Conflict` when every file is rejected only because of duplicate filenames.
- `400 Bad Request` when every file is rejected because of upload validation errors such as unsupported extensions or oversized files.
- `400 Bad Request` when the request has no files or no valid multipart shape.
- `422 Unprocessable Entity` can remain FastAPI's default for malformed form fields.

### Decision 4: Make worker startup config-driven

Add settings:

```env
CELERY_WORKER_POOL=auto
CELERY_WORKER_CONCURRENCY=2
CELERY_WORKER_PREFETCH_MULTIPLIER=1
CELERY_WORKER_MAX_TASKS_PER_CHILD=10
CELERY_WORKER_TIME_LIMIT=300
CELERY_WORKER_SOFT_TIME_LIMIT=240
```

Pool resolution:

- `auto` on Windows resolves to `threads`.
- `auto` on non-Windows resolves to `prefork`.
- Explicit `solo` is allowed only for debugging and should force/effectively document concurrency `1`.
- Explicit `threads`, `prefork`, `gevent`, or `eventlet` should pass through.

Rationale:

- The current Windows `solo` default is the direct reason local processing appears one-at-a-time.
- `threads` is a practical Windows local default for subprocess-heavy parsing and network embedding/caption calls.
- `prefork` remains the production Linux default for stronger isolation.

## File Map

Create:

- `app/alembic/versions/t6u7v8w9x0y1_add_document_filename_key.py` - add and backfill `documents.filename_key`, then enforce `(conversation_id, filename_key)` uniqueness.
- `tests/test_document_upload_batch.py` - canonical server batch upload and duplicate behavior.
- `tests/client_backend/test_document_batch_routes.py` - sidecar batch proxy behavior.
- `tests/test_document_filename_duplicates.py` - repository/service duplicate checks and normalization.
- `tests/test_celery_worker_config.py` - worker command/config behavior.

Modify:

- `app/models/document.py` - add `filename_key` and unique/index metadata.
- `app/schemas/document.py` - add upload result schemas.
- `app/repositories/document.py` - add filename-key lookup and duplicate-safe create.
- `app/services/document_service.py` - enforce duplicate filename creation and expose batch-friendly helpers.
- `app/services/document_processing_service.py` - keep staging validation reusable for batch uploads; avoid adding shared state.
- `app/api/documents.py` - add `/documents/uploads`, make `/documents/upload` a wrapper.
- `client_backend/api/documents.py` - add `/documents/uploads` proxy and route UI traffic to it.
- `client_backend/services/server_api.py` - add `upload_documents_bytes(...)`.
- `upload_support.py` - replace single-file upload helper/UI with batch helper/UI.
- `demo.py` - enable `accept_multiple_files=True` in document upload surfaces and render per-file results.
- `app/core/config.py` - add Celery worker settings.
- `app/workers/celery_app.py` - read worker-related Celery defaults from settings.
- `app/workers/start_worker.py` - build worker command from settings and resolve pool by platform.
- `.env.example` - document new worker settings.
- `README.md` - document batch upload API, duplicate policy, and worker parallelism settings.
- Existing tests that assert single-file-only behavior, especially `tests/client_backend/test_document_routes.py`, `tests/client_backend/test_document_upload_proxy_guard.py`, `tests/test_start_worker.py`, and `tests/test_demo_document_file_types.py`.

---

## Phase 1: Baseline And Contracts

### Task 1.1: Record current focused baseline

- [x] Run focused tests before changes:

```powershell
python -m pytest tests\client_backend\test_document_routes.py tests\client_backend\test_document_upload_proxy_guard.py tests\test_document_processing_service.py tests\test_start_worker.py tests\test_demo_document_file_types.py -q
```

Expected:

- Current tests pass before changes, except for any unrelated known baseline failures already present in the local tree.
- Record any unrelated failures in the implementation notes before editing behavior.

**Result (2026-05-19):** Baseline 14 tests pass.

### Task 1.2: Add server batch upload contract tests

- [x] Create `tests/test_document_upload_batch.py`.
- [x] Test that `/documents/uploads` accepts multiple files and returns ordered per-file accepted results.
- [x] Test that an existing same-conversation filename returns `DUPLICATE_FILENAME` for that file and still enqueues non-duplicate siblings.
- [x] Test that duplicate names inside the same incoming batch accept the first candidate and reject later duplicates.
- [x] Test that all-duplicate batches return `409`.
- [x] Test that mixed accepted/rejected batches return `207`.

Representative test shape:

```python
def test_batch_upload_rejects_duplicate_and_accepts_valid_sibling(client, monkeypatch):
    response = client.post(
        "/documents/uploads",
        data={"conversation_id": str(conversation_id)},
        files=[
            ("files", ("existing.pdf", b"old", "application/pdf")),
            ("files", ("new.pdf", b"new", "application/pdf")),
        ],
    )

    assert response.status_code == 207
    payload = response.json()["data"]
    assert payload["accepted_count"] == 1
    assert payload["rejected_count"] == 1
    assert payload["files"][0]["status"] == "rejected"
    assert payload["files"][0]["error_code"] == "DUPLICATE_FILENAME"
    assert payload["files"][1]["status"] == "accepted"
```

### Task 1.3: Add duplicate filename service/repository tests

- [x] Create `tests/test_document_filename_duplicates.py`.
- [x] Test normalization casefolds and strips path segments.
- [x] Test `DocumentRepository.filename_exists_in_conversation(...)`.
- [x] Test `DocumentService.validate_and_create_document(...)` raises a duplicate exception before enqueue code creates another document.
- [x] Test repository create catches unique constraint violations and maps them to the duplicate exception.

### Task 1.4: Add worker command config tests

- [x] Create `tests/test_celery_worker_config.py` or extend `tests/test_start_worker.py`.
- [x] Replace the current Windows expectation that appends `--pool=solo`.
- [x] Test Windows `CELERY_WORKER_POOL=auto` resolves to `--pool=threads`.
- [x] Test non-Windows `auto` resolves to `--pool=prefork`.
- [x] Test configured concurrency appears in the command.
- [x] Test `solo` remains explicit debugging behavior.

Representative assertion:

```python
assert "--pool=threads" in " ".join(captured["cmd"])
assert "--concurrency=4" in " ".join(captured["cmd"])
```

---

## Phase 2: Duplicate Filename Data Model

### Task 2.1: Add `filename_key` model field

- [x] Modify `app/models/document.py`.
- [x] Add:

```python
filename_key = Column(String(255), nullable=False)
```

- [x] Add a unique constraint or unique index on:

```python
("conversation_id", "filename_key")
```

- [x] Keep `filename` as the display name.

### Task 2.2: Add Alembic migration

- [x] Create `app/alembic/versions/t6u7v8w9x0y1_add_document_filename_key.py`.
- [x] Add nullable `filename_key`.
- [x] Backfill normalized keys from existing filenames.
- [x] Resolve historical duplicates by suffixing older duplicate keys with `::legacy::<document_id>`.
- [x] Alter `filename_key` to non-null.
- [x] Create unique index/constraint for `(conversation_id, filename_key)`.

Migration acceptance:

- Fresh databases can migrate.
- Existing databases with no duplicates can migrate.
- Existing databases with duplicate filenames in the same conversation can migrate without deleting rows.

### Task 2.3: Add normalization helper

- [x] Add `normalize_document_filename(...)`.
- [x] Preferred location: `app/utils/validation/document_validation.py` if it fits current validation helpers; otherwise `app/utils/text_processing.py`.
- [x] Use `Path(filename).name`, `strip()`, `unicodedata.normalize("NFC", value)`, and `casefold()`.
- [x] Raise `FileValidationError` or `ValueError` for empty normalized names.

### Task 2.4: Update repository create path

- [x] Modify `DocumentRepository.create(...)` to persist `filename_key`.
- [x] Add:

```python
def get_by_conversation_and_filename_key(
    self, conversation_id: UUID, filename_key: str
) -> Document | None:
    with self.session_factory() as db:
        return (
            db.query(Document)
            .filter(
                Document.conversation_id == conversation_id,
                Document.filename_key == filename_key,
            )
            .first()
        )

def filename_exists_in_conversation(self, conversation_id: UUID, filename_key: str) -> bool:
    with self.session_factory() as db:
        query = db.query(Document).filter(
            Document.conversation_id == conversation_id,
            Document.filename_key == filename_key,
        )
        return db.query(query.exists()).scalar()
```

- [x] Catch `IntegrityError` from the unique constraint and raise a domain exception such as `DuplicateDocumentFilenameError`.

### Task 2.5: Update document schemas

- [x] Modify `app/schemas/document.py`.
- [x] Add `filename_key` to `DocumentCreate`.
- [x] Do not expose `filename_key` in `DocumentResponse` unless the implementation needs it for tests. It is an internal lookup key.
- [x] Add batch upload response schemas:

```python
class DocumentUploadFileStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"

class DocumentUploadFileResult(BaseModel):
    filename: str
    status: DocumentUploadFileStatus
    document: DocumentResponse | None = None
    processing: dict[str, Any] | None = None
    error_code: str | None = None
    message: str | None = None

class DocumentBatchUploadResponse(BaseModel):
    conversation_id: UUID
    total_count: int
    accepted_count: int
    rejected_count: int
    files: list[DocumentUploadFileResult]
```

---

## Phase 3: Canonical Server Batch Upload

### Task 3.1: Add domain exception and API mapping

- [x] Add `DuplicateDocumentFilenameError` under `app/core/exceptions/validation.py` or the existing validation exception module pattern.
- [x] Map it to HTTP `409 Conflict` for single-file uploads.
- [x] For batch uploads, convert it into a per-file rejected result.

### Task 3.2: Update `DocumentService.validate_and_create_document(...)`

- [x] Normalize filename to `filename_key`.
- [x] Check `filename_exists_in_conversation(...)` before create.
- [x] Create `DocumentCreate(conversation_id=conversation_id, filename=filename, filename_key=filename_key, file_type=content_type or "unknown", status=DocumentStatus.PROCESSING.value)`.
- [x] Catch unique constraint duplicate errors and surface the same duplicate exception.

### Task 3.3: Add reusable route helper

- [x] In `app/api/documents.py`, extract the current single-file route body into an internal helper:

```python
async def _stage_create_and_enqueue_document(
    *,
    document_service: IDocumentService,
    document_processing_service: DocumentProcessingService,
    current_user_id: UUID,
    file: UploadFile,
    conversation_id: UUID,
) -> DocumentUploadFileResult:
    staged_upload = await document_processing_service.stage_upload_file(
        file, file.filename or "unknown"
    )
    staged_file_path = Path(staged_upload["temp_file_path"])
    try:
        document = await document_service.validate_and_create_document(
            filename=file.filename or "",
            file_size=staged_upload["file_size"],
            content_type=file.content_type or "unknown",
            conversation_id=conversation_id,
        )
        task_info = await document_processing_service.start_processing_task(
            str(document.id),
            str(staged_file_path),
            file.filename or "unknown",
            staged_upload["file_size"],
        )
        if task_info.get("task_id"):
            document = await document_service.set_processing_task_id(
                document.id, task_info["task_id"]
            ) or document
        with contextlib.suppress(Exception):
            await get_event_bus().emit(
                DocumentEvent.UPLOAD_STARTED,
                DocumentEventData(
                    document_id=document.id,
                    conversation_id=conversation_id,
                    user_id=current_user_id,
                    filename=file.filename,
                    status="PROCESSING",
                    metadata={"task_id": task_info.get("task_id")},
                ),
            )
        return DocumentUploadFileResult(
            filename=file.filename or "unknown",
            status=DocumentUploadFileStatus.ACCEPTED,
            document=document,
            processing=task_info,
        )
    except Exception:
        if staged_file_path.is_file():
            staged_file_path.unlink()
        raise
```

- [x] The helper should:
  - Validate and stage the uploaded file.
  - Create the `Document` row through `DocumentService`.
  - Enqueue `process_document_task`.
  - Persist `processing_task_id`.
  - Emit `UPLOAD_STARTED`.
  - Delete staged temp file if any step fails before enqueue.

### Task 3.4: Add `/documents/uploads`

- [x] Add:

```python
@router.post("/uploads", response_model=ApiResponse[dict[str, Any]])
@AppAutoInjector.auto_inject()
async def upload_documents(
    response: Response,
    document_service: IDocumentService,
    document_processing_service: DocumentProcessingService,
    current_user_id: UUID,
    files: list[UploadFile] = File(...),
    conversation_id: UUID = Form(...),
) -> ApiResponse[dict[str, Any]]:
    ConversationValidationUtils(
        document_service.repository.session_factory
    ).validate_conversation_access(current_user_id, conversation_id)
    upload_result = await _upload_documents_batch(
        document_service=document_service,
        document_processing_service=document_processing_service,
        current_user_id=current_user_id,
        files=files,
        conversation_id=conversation_id,
    )
    response.status_code = _status_code_for_batch_result(upload_result)
    return ApiResponse(
        success=upload_result.accepted_count > 0,
        message=_message_for_batch_result(upload_result),
        data=upload_result.model_dump(),
    )
```

- [x] Validate conversation ownership once before looping files.
- [x] Reject empty file lists with `400`.
- [x] Track `seen_filename_keys` inside the incoming batch.
- [x] Reject already-seen batch duplicates without staging the later duplicate.
- [x] Reject database duplicates without staging if possible.
- [x] Process valid siblings independently.
- [x] Return `201`, `207`, `409`, or `400` per the status policy.

### Task 3.5: Make `/documents/upload` a thin wrapper

- [x] Keep route only as compatibility wrapper.
- [x] It should call the same helper used by `/documents/uploads`.
- [x] It must not contain duplicate staging/enqueue logic.
- [x] It should still return the current envelope shape as much as possible:

```json
{
  "document": { "id": "document-id", "filename": "alpha.pdf" },
  "processing": { "task_id": "celery-task-id", "success": true }
}
```

- [x] If duplicate, return `409`.

---

## Phase 4: Sidecar Batch Proxy

### Task 4.1: Add server client batch method

- [x] Modify `client_backend/services/server_api.py`.
- [x] Add a small status-aware return model:

```python
class UploadProxyResponse(BaseModel):
    status_code: int
    payload: dict[str, Any]
```

- [x] Add:

```python
async def upload_documents_bytes_with_status(
    self,
    *,
    conversation_id: str,
    files: list[dict[str, Any]],
) -> UploadProxyResponse:
    multipart = [
        ("files", (item["filename"], item["content"], item["content_type"]))
        for item in files
    ]
    response = await self.request_response(
        "POST",
        "/documents/uploads",
        data={"conversation_id": conversation_id},
        files=multipart,
    )
    payload = await self._handle_response(response)
    return UploadProxyResponse(status_code=response.status_code, payload=payload)
```

- [x] Add a payload-only convenience wrapper for callers that do not need status preservation:

```python
async def upload_documents_bytes(
    self,
    *,
    conversation_id: str,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    result = await self.upload_documents_bytes_with_status(
        conversation_id=conversation_id,
        files=files,
    )
    return result.payload
```

- [x] Keep `upload_document_bytes(...)` only as a wrapper around `upload_documents_bytes(...)` if existing tests or external callers still need it.

### Task 4.2: Add sidecar route

- [x] Modify `client_backend/api/documents.py`.
- [x] Add:

```python
@router.post("/uploads")
async def upload_documents(
    response: Response,
    files: list[UploadFile] = File(...),
    conversation_id: str = Form(...),
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    upload_items = []
    for file in files:
        upload_items.append(
            {
                "filename": file.filename or "upload.bin",
                "content": await file.read(),
                "content_type": file.content_type or "application/octet-stream",
            }
        )
    upstream_response = await get_server_client().upload_documents_bytes_with_status(
        conversation_id=conversation_id,
        files=upload_items,
    )
    response.status_code = upstream_response.status_code
    return upstream_response.payload
```

- [x] Read each `UploadFile` once and forward ordered bytes to `get_server_client().upload_documents_bytes_with_status(...)`.
- [x] Preserve status code from canonical server when practical. If the current `ServerAPIClient.post(...)` hides status code, use `request_response(...)` and `_handle_response(...)` or extend the client in a focused way.

### Task 4.3: Keep sidecar single wrapper minimal

- [x] Make sidecar `/documents/upload` call `upload_document_bytes(...)`, which itself wraps the batch method.
- [x] Do not import parser/indexer/RAG modules into `client_backend`.
- [x] Update `tests/client_backend/test_document_upload_proxy_guard.py` to assert the sidecar is still a byte proxy.

---

## Phase 5: Streamlit Multi-File UI

### Task 5.1: Add upload helper for many files

- [x] Modify `upload_support.py`.
- [x] Replace or supplement `upload_document(uploaded_file)` with:

```python
def upload_documents(uploaded_files: list[Any]) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    if st.session_state.get("auth_token"):
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"
    files = [
        ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
        for uploaded_file in uploaded_files
    ]
    response = get_http_session().post(
        f"{API_BASE_URL}/documents/uploads",
        files=files,
        data={"conversation_id": st.session_state.current_conversation_id},
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code in {201, 207, 409}:
        result = response.json()
        accepted_count = int((result.get("data") or {}).get("accepted_count") or 0)
        if accepted_count:
            _bump_api_cache_version()
        return result
    st.error(f"Upload failed: {response.status_code} - {response.text}")
    return None
```

- [x] Treat `201`, `207`, and `409` as parseable upload responses.
- [x] Show API-level failures only when the server returns an unexpected status.
- [x] Bump cache version after any accepted file.

### Task 5.2: Update `render_upload_section()`

- [x] Set `accept_multiple_files=True`.
- [x] Display selected file count and names.
- [x] Upload all selected files with one button.
- [x] Render accepted and rejected results separately.
- [x] On duplicates, show the server message without rerunning away immediately.

### Task 5.3: Update `demo.py::render_documents_tab()`

- [x] Set:

```python
uploaded_files = st.file_uploader(
    "Select files",
    type=["txt", "pdf", "docx", "pptx", "xlsx", "html", "md"],
    accept_multiple_files=True,
    key=uploader_key,
    help="Supported formats: TXT, PDF, DOCX, PPTX, XLSX, HTML, MD",
)
```

- [x] Replace single-file display with a compact selected-file list.
- [x] Call `upload_documents(uploaded_files)`.
- [x] Render per-file outcomes:
  - Accepted: filename, "processing started", task id when present.
  - Rejected duplicate: filename, duplicate message.
  - Rejected validation: filename, validation message.
- [x] Refresh the document list after accepted uploads.

### Task 5.4: Update demo/upload tests

- [x] Update `tests/test_demo_document_file_types.py` to also assert `accept_multiple_files=True` for document uploaders.
- [x] Add tests or source guards that `demo.py` and `upload_support.py` call `/documents/uploads`.

---

## Phase 6: Celery Worker Parallelism

### Task 6.1: Add worker settings

- [x] Modify `app/core/config.py`.
- [x] Add fields:

```python
celery_worker_pool: str = Field(default="auto")
celery_worker_concurrency: int = Field(default=2)
celery_worker_prefetch_multiplier: int = Field(default=1)
celery_worker_max_tasks_per_child: int = Field(default=10)
celery_worker_time_limit: int = Field(default=300)
celery_worker_soft_time_limit: int = Field(default=240)
```

- [x] Add validation to keep concurrency and time limits positive.
- [x] Document that Windows local parallelism uses `threads` by default, while Linux production uses `prefork`.

### Task 6.2: Apply settings in `celery_app.py`

- [x] Replace hard-coded `worker_prefetch_multiplier` with `settings.celery_worker_prefetch_multiplier`.
- [x] Add task time limits to Celery config:

```python
celery_app.conf.task_time_limit = settings.celery_worker_time_limit
celery_app.conf.task_soft_time_limit = settings.celery_worker_soft_time_limit
```

- [x] Keep:

```python
task_acks_late = True
task_reject_on_worker_lost = True
```

### Task 6.3: Refactor `start_worker.py`

- [x] Build the worker command from settings instead of hard-coded values.
- [x] Resolve pool:

```python
def _resolve_worker_pool(configured_pool: str, system: str) -> str:
    if configured_pool != "auto":
        return configured_pool
    return "threads" if system == "Windows" else "prefork"
```

- [x] Include:

```powershell
--pool=<resolved_pool>
--concurrency=<settings.celery_worker_concurrency>
--max-tasks-per-child=<settings.celery_worker_max_tasks_per_child>
--time-limit=<settings.celery_worker_time_limit>
--soft-time-limit=<settings.celery_worker_soft_time_limit>
```

- [x] Print a clear startup banner:

```text
Starting Celery worker: pool=threads concurrency=2 prefetch=1
```

### Task 6.4: Guard document processing task-local state

- [x] Add a test that `Container().document_processing_service()` returns a new service instance per call, or inspect the provider type if direct instantiation is too heavy.
- [x] Do not change `document_processing_service` to `providers.Singleton`.
- [x] Avoid adding class-level or module-level mutable document processing state.
- [x] If implementation introduces any shared caches, they must be read-only or protected by locks.

### Task 6.5: Optional manual parallelism verification

- [x] Start Redis and the canonical server.
- [x] Start the worker:

```powershell
$env:CELERY_WORKER_CONCURRENCY="2"
$env:CELERY_WORKER_POOL="threads"
python -m app.workers.start_worker
```

- [x] Upload two medium PDFs at once through the UI.
- [x] Expected console behavior:
  - Two `process_document_task` jobs are received before the first one completes.
  - Both document statuses move to `PROCESSING`.
  - Completion order may differ from upload order.

Production Linux verification:

```bash
CELERY_WORKER_POOL=prefork CELERY_WORKER_CONCURRENCY=4 python -m app.workers.start_worker
```

---

## Phase 7: Documentation And Cleanup

### Task 7.1: Update `.env.example`

- [x] Add the new Celery worker settings near existing Redis/Celery settings.
- [x] Include comments:

```env
# Windows local development: auto resolves to threads for real concurrency.
# Linux production: auto resolves to prefork.
CELERY_WORKER_POOL=auto
CELERY_WORKER_CONCURRENCY=2
CELERY_WORKER_PREFETCH_MULTIPLIER=1
CELERY_WORKER_MAX_TASKS_PER_CHILD=10
CELERY_WORKER_TIME_LIMIT=300
CELERY_WORKER_SOFT_TIME_LIMIT=240
```

### Task 7.2: Update README

- [x] Document `POST /documents/uploads`.
- [x] Document per-file duplicate rejection.
- [x] Mark `/documents/upload` as legacy single-file compatibility wrapper if kept.
- [x] Update troubleshooting:

```text
If uploads still process one at a time, check the worker banner. On Windows, pool must be `threads` or another parallel pool; `solo` is single-task debug mode.
```

### Task 7.3: Remove redundant upload code paths

- [x] After all callers use `/documents/uploads`, remove duplicate single-file helper logic.
- [x] Keep only thin wrappers where compatibility is explicitly required.
- [x] Delete tests that assert deprecated internals, and replace them with tests that assert wrappers delegate to the canonical batch path.

---

## Verification Plan

Run focused tests:

```powershell
python -m pytest tests\test_document_upload_batch.py tests\test_document_filename_duplicates.py tests\client_backend\test_document_batch_routes.py tests\client_backend\test_document_routes.py tests\client_backend\test_document_upload_proxy_guard.py tests\test_celery_worker_config.py tests\test_start_worker.py tests\test_demo_document_file_types.py -q
```

Run document/RAG regression tests:

```powershell
python -m pytest tests\test_document_processing_service.py tests\test_document_index_service.py tests\test_document_service_deletion.py tests\test_rag_agent.py tests\test_rag_multi_user_isolation.py -q
```

Run broader backend smoke tests if time allows:

```powershell
python -m pytest tests --ignore=tests/client_backend/test_live_server_integration.py -q
```

Manual UI verification:

- Open `demo.py`.
- Select an existing conversation.
- Select three documents at once.
- Include one file whose filename already exists in that conversation.
- Confirm non-duplicates are accepted and shown as processing.
- Confirm duplicate is shown as rejected without blocking accepted files.
- Refresh library and confirm only accepted files appear.

Manual worker verification:

- Start `python -m app.workers.start_worker`.
- Confirm startup banner shows `pool=threads concurrency=2` on Windows unless explicitly overridden.
- Upload multiple medium documents.
- Confirm Celery logs show overlapping task execution.

## Acceptance Criteria

- [x] `POST /documents/uploads` accepts multiple document files in one request.
- [x] Batch upload responses preserve input order.
- [x] Duplicate filename rejection is scoped to the same conversation.
- [x] Duplicate filename checks are case-insensitive and race-safe through a database uniqueness constraint.
- [x] Batch uploads reject duplicate siblings while still processing valid files.
- [x] Existing single-file upload behavior is either a thin compatibility wrapper or removed after all internal callers are migrated.
- [x] `client_backend` remains a byte proxy and does not import server parser/indexer/RAG modules.
- [x] `demo.py` document upload supports selecting multiple files simultaneously.
- [x] `upload_support.py` supports batch upload and renders accepted/rejected results clearly.
- [x] Celery worker startup is controlled by config, not hard-coded command flags.
- [x] On Windows, default worker startup uses a parallel-capable pool instead of `solo`.
- [x] Document processing service remains task-local and safe for concurrent Celery task execution.
- [x] README and `.env.example` document the new API and worker settings.

## Risks And Mitigations

Risk: historical duplicate document names block the migration.

Mitigation: backfill `filename_key` and suffix older duplicate keys before adding uniqueness. Do not delete historical rows in a schema migration.

Risk: Windows threaded workers expose shared mutable processing state.

Mitigation: keep `document_processing_service` as `providers.Factory`, add a guard test, and avoid singleton/shared mutable task state.

Risk: provider rate limits become more visible when multiple documents process concurrently.

Mitigation: keep conservative default concurrency `2`, keep prefetch `1`, and document that users can lower concurrency or run a dedicated MinerU API service if local OCR/model resources saturate.

Risk: partial batch success is confusing in the UI.

Mitigation: render accepted and rejected files separately and avoid a blanket "upload failed" message for `207` responses.

Risk: `207 Multi-Status` handling is missed by clients.

Mitigation: update Streamlit and sidecar helpers to treat `201`, `207`, and `409` as structured upload responses.

---

## Implementation Notes (2026-05-19)

### Test Strategy

- **Batch upload contract tests** were implemented against the helper
  function `_upload_documents_batch` rather than a full FastAPI
  TestClient app. Reason: the canonical server's
  `AppAutoInjector.auto_inject()` decorator depends on the container's
  wiring map being populated at import time, and standing that up for a
  TestClient pulls in qdrant / sentence-transformers / langchain. Unit
  testing the batch helper with mocked services captures the order,
  duplicate, and status-code policy directly. The HTTP plumbing is
  thin (`upload_documents` and `upload_document` both call
  `_upload_documents_batch`).
- **Sidecar batch route tests** still use FastAPI's `TestClient` because
  the sidecar router has no auto-injection machinery.

### Design Decisions

- `normalize_document_filename` lives in `app/services/document_service.py`
  rather than `app/utils/validation/document_validation.py` because the
  service module is the place all callers (service, route helpers, batch
  loop, alembic migration body) needed to import from. Keeping the helper
  next to the service that consumes it avoids a circular dependency from
  the validation utility module.
- `DuplicateDocumentFilenameError` lives in
  `app/core/exceptions/validation.py` next to `FileValidationError` so
  HTTP middleware that converts custom exceptions can pick it up
  uniformly.
- `DocumentRepository.create` translates `IntegrityError` into the
  domain exception. The service layer also checks
  `filename_exists_in_conversation` before insert as a fast path that
  surfaces a clean 409 in the common case; the repository fallback is
  the race-safe guarantee under concurrent uploads.
- `start_worker.py` deliberately calls `get_settings()` inside the
  function rather than at module import, so tests can monkeypatch
  env vars and call `get_settings.cache_clear()` to take effect.
- When the configured pool is `solo`, the worker command clamps
  `--concurrency` to `1` so the printed banner does not mislead the
  user about parallelism.
- The single-file `upload_support.upload_document` is kept as a thin
  wrapper over `upload_documents([file])`. Removing it would have
  required searching out callers across the legacy demo; the wrapper
  preserves the call surface while routing through the batch path.
- The `/documents/upload` server route is kept as a thin wrapper that
  also goes through `_upload_documents_batch` so duplicate handling,
  task enqueue, and event emission are not duplicated.
- The Alembic migration backfills historical duplicate `filename_key`
  values with a `::legacy::<document_id>` suffix so existing data is
  preserved when the unique constraint is added.

### Verification Results

- Baseline focused tests (2026-05-19): 14 passed.
- Final focused verification suite: 31 passed (batch upload contract,
  duplicate filename, sidecar batch proxy, sidecar single, sidecar
  guard, worker config, worker startup, demo file types).
- Document/RAG regression suite: 57 passed (processing, index, deletion,
  RAG agent, multi-user isolation).
- Full broader suite (excluding live integration tests): 583 passed.
