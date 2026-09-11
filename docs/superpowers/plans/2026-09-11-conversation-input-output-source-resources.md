# Conversation Inputs, Outputs, and Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a durable, verified conversation-resource system that gives Streamlit and AI SDK clients the same working Inputs, Outputs, and Sources, including safe tracking of files created by device-local MCP tools.

**Architecture:** A strict `ConversationResourceView` is the only public contract. Stateful inputs and outputs persist in new resource tables, while existing web and RAG source records project into the same view without duplicate source storage. Device-local candidates are extracted and verified by the sidecar, registered synchronously with tool results, and optionally snapshotted through an asynchronous resumable upload path; server-owned grounding and client adapters then render the same stable identities.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, SQLAlchemy/PostgreSQL, Alembic, Redis-backed client-runtime queues, LangChain/LangGraph, content-addressed filesystem storage, Streamlit, Vercel AI SDK v6 SSE, pytest/pytest-asyncio, Ruff.

## Global Constraints

- Execute after `docs/superpowers/plans/2026-09-10-production-web-research-and-image-grounding.md`; Task 9 consumes its canonical `SourceRecord` and `sources_upsert` contract.
- Use `python` for repository commands. Never embed a developer-home interpreter path in tracked files.
- Public resource payloads never contain local paths, storage paths, bearer tokens, private device identifiers, provider payloads, or inline arbitrary file bytes.
- Never infer files from path-like prose or whole-filesystem scans. Accept only typed MCP resources, registered qualified-tool extractors, or a destination deterministically present in approved arguments and confirmed by that extractor.
- Only a verified pre-call absence followed by a post-call regular file is called `created`; otherwise use `modified`, `unchanged`, or `unknown` according to evidence.
- Reject symlinks, Windows junctions/reparse points, directories, sockets, devices, named pipes, and identity changes across the verify/open/stat boundary.
- The managed snapshot default is 25 MiB. Files above it remain device references and are not fully hashed solely for display.
- Hashing and upload are asynchronous. They must not extend tool execution or delay the answer model's next step.
- Bound work by candidate count, byte count, chunk size, outstanding uploads, retries, storage quota, filename length, and version count. Connection and idle-read limits are liveness controls; do not add a guessed end-to-end upload deadline that interrupts progress.
- Managed blobs are immutable. A later file version creates a new resource linked by `supersedes_resource_id`.
- Tool-result blobs remain model-support text. New code uses `tool_result_blob_id`; no active path writes `artifact_ref` as though it were a user output.
- `R1`, `R2`, and related aliases are turn-scoped prompt tokens, never durable API identifiers.
- Web sources remain in canonical message metadata from the preceding project; RAG sources remain backed by existing document identities. Do not create a duplicate source-of-truth table.
- AI SDK emits native `file` and `source-url` parts where applicable plus `data-conversation-resources`; Streamlit consumes the same canonical records and ordering.
- New input clients use staged resource IDs rather than base64 message payloads. Legacy base64 images remain read-compatible during migration but are immediately externalized.
- Deterministic tests are the release gate. A live Desktop Commander canary is optional and absence of that external tool never fails the suite.
- Preserve unrelated working-tree changes, especially the in-progress `app/ui/stream_markdown.py` work. Resolve integration only after that work is committed or moved to an isolated worktree.
- Every task follows red-green-refactor and commits only its listed files.

## Planned File Map

New focused units:

- `app/schemas/conversation_resource.py`: strict public/private resource contracts and lifecycle validation.
- `app/models/conversation_resource.py`: resource, blob, message-link, and upload-session persistence.
- `app/repositories/conversation_resource.py`: ownership-scoped CRUD, idempotent registration, transitions, links, and cleanup queries.
- `app/services/conversation_resource_storage.py`: staged streaming writes, checksums, atomic blob promotion, and reads.
- `app/services/conversation_resource_service.py`: registration, authorization, public projection, actions, and source adapters.
- `app/services/conversation_resource_events.py`: bounded per-conversation resource event publication/subscription.
- `app/api/conversation_resources.py`: list, upload, download, refresh, save, and open-action endpoints.
- `app/ai/resource_grounding.py`: prompt aliases and validated `[[resource:R#]]` resolution.
- `client_backend/services/resource_extractors.py`: typed MCP and qualified-tool extraction.
- `client_backend/services/resource_verifier.py`: platform-safe local verification and change classification.
- `client_backend/services/resource_catalog.py`: profile-scoped opaque local reference persistence.
- `client_backend/services/resource_snapshots.py`: asynchronous resumable snapshot uploader.
- `client_backend/api/resources.py`: local open/refresh action endpoints.
- `app/ui/conversation_resources.py`: Streamlit registry state, inline actions, and panel.
- `app/observability/conversation_resources.py`: bounded-cardinality metrics and health.
- `app/evaluation/conversation_resources.py`: deterministic cross-client contract evaluation.

Existing integration files remain coordinators rather than absorbing the new logic: `app/ai/tool_execution.py`, `app/ai/client_runtime_tools.py`, `app/services/message_service.py`, event adapters, `app/api/ai_sdk.py`, `app/api/device_runtime.py`, `client_backend/services/runtime_bridge.py`, and `demo.py`.

---

### Task 1: Define the canonical resource and lifecycle contracts

**Files:**
- Create: `app/schemas/conversation_resource.py`
- Modify: `app/schemas/__init__.py`
- Test: `tests/test_conversation_resource_contracts.py`

**Interfaces:**
- Consumes: no new project types.
- Produces: `ResourceCategory`, `ResourceKind`, `ResourceStatus`, `ResourceAccess`, `ResourceChangeKind`, `ResourceOrigin`, `ResourceAction`, `ConversationResourceView`, `DeviceResourceCandidate`, `ResourceUpsertPayload`, `ResourceAlias`, `validate_resource_transition(old, new, access)`, `make_source_resource_id(generation_id, source_id)`, and `assign_resource_aliases(resources)`.

- [ ] **Step 1: Write strict failing contract tests**

```python
def test_managed_resource_requires_only_a_content_url():
    row = ConversationResourceView(
        resource_id="2a0ef560-e35d-4f68-a994-59ce09f2764d",
        version=1,
        categories=[ResourceCategory.output],
        kind=ResourceKind.file,
        status=ResourceStatus.available,
        access=ResourceAccess.managed,
        name="report.pdf",
        content_url="/conversation-resources/2a0ef560-e35d-4f68-a994-59ce09f2764d/content",
    )
    assert row.external_url is None
    assert row.device_action is None


def test_public_resource_rejects_a_local_path_and_duplicate_categories():
    with pytest.raises(ValidationError):
        ConversationResourceView.model_validate(
            {
                "resource_id": "R-secret",
                "version": 1,
                "categories": ["output", "output"],
                "kind": "file",
                "status": "available",
                "access": "device",
                "name": "secret.txt",
                "local_path": "private-path",
            }
        )


def test_snapshot_pending_cannot_transition_to_a_non_status_value():
    with pytest.raises(ValueError):
        validate_resource_transition(
            ResourceStatus.snapshot_pending,
            "device",
            ResourceAccess.device,
        )


def test_aliases_are_turn_scoped_and_deterministic():
    aliases = assign_resource_aliases([_resource("b"), _resource("a")])
    assert [(item.alias, item.resource_id) for item in aliases] == [("R1", "a"), ("R2", "b")]
```

- [ ] **Step 2: Run the new tests and verify the missing-contract failure**

Run: `python -m pytest tests/test_conversation_resource_contracts.py -q`

Expected: FAIL because `app.schemas.conversation_resource` does not exist.

- [ ] **Step 3: Implement the strict contracts**

Use `ConfigDict(extra="forbid")` on every wire model. `ConversationResourceView` uses `resource_id: str`, `version: int = Field(ge=1)`, unique ordered `categories`, one of the three access descriptors, safe optional provenance IDs, integrity metadata, and timestamps. Enforce these exact access rules:

```python
def validate_resource_transition(
    old: ResourceStatus,
    new: ResourceStatus,
    access: ResourceAccess,
) -> None:
    allowed = {
        ResourceStatus.reported: {
            ResourceStatus.verified,
            ResourceStatus.rejected,
            ResourceStatus.unavailable,
            ResourceStatus.deleted,
        },
        ResourceStatus.verified: {
            ResourceStatus.snapshot_pending,
            ResourceStatus.available,
            ResourceStatus.unavailable,
            ResourceStatus.rejected,
            ResourceStatus.deleted,
        },
        ResourceStatus.snapshot_pending: {
            ResourceStatus.available,
            ResourceStatus.unavailable,
            ResourceStatus.rejected,
            ResourceStatus.deleted,
        },
        ResourceStatus.available: (
            {ResourceStatus.unavailable, ResourceStatus.deleted}
            if access is ResourceAccess.device
            else {ResourceStatus.deleted}
        ),
        ResourceStatus.unavailable: {
            ResourceStatus.verified,
            ResourceStatus.available,
            ResourceStatus.deleted,
        },
        ResourceStatus.rejected: {ResourceStatus.deleted},
        ResourceStatus.deleted: set(),
    }
    if new not in allowed[old]:
        raise ValueError(f"invalid resource transition: {old.value}->{new.value}")
```

`DeviceResourceCandidate` contains `client_resource_id`, origin, kind, safe name, media type, optional byte size, verification status, change kind, snapshot eligibility, and safe rejection code. It contains no path field. `make_source_resource_id` returns `source:{generation_id}:{source_id}` after validating both components. Alias assignment sorts by display ordinal then stable resource ID.

- [ ] **Step 4: Run the contract tests**

Run: `python -m pytest tests/test_conversation_resource_contracts.py tests/test_production_readiness_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/schemas/conversation_resource.py app/schemas/__init__.py tests/test_conversation_resource_contracts.py
git commit -m "feat: define conversation resource contracts"
```

### Task 2: Add durable resource persistence and migration

**Files:**
- Create: `app/models/conversation_resource.py`
- Create: `app/repositories/conversation_resource.py`
- Create: `app/alembic/versions/d7f8a9b0c1e2_add_conversation_resources.py`
- Modify: `app/models/__init__.py`
- Modify: `app/models/conversation.py`
- Modify: `app/models/message.py`
- Test: `tests/test_conversation_resource_models.py`
- Test: `tests/test_conversation_resource_migration.py`
- Test: `tests/integration/test_conversation_resource_repository_postgres.py`

**Interfaces:**
- Consumes: Task 1 enums and transition validator.
- Produces: `ResourceIdempotencyScope`; `ConversationResource`, `ConversationResourceBlob`, `MessageResourceLink`, `ResourceUploadSession`; `ConversationResourceRepository.create_or_get(scope: ResourceIdempotencyScope, values: Mapping[str, Any]) -> ConversationResource`; `transition(resource_id: UUID, expected_version: int, new_status: ResourceStatus, values: Mapping[str, Any]) -> ConversationResource`; `link_message(resource_id: UUID, message_id: UUID, category: ResourceCategory, ordinal: int) -> None`; `list_for_conversation(user_id: UUID, conversation_id: UUID) -> list[ConversationResource]`; `get_for_user(resource_id: UUID, user_id: UUID) -> ConversationResource | None`; and `soft_delete_conversation_resources(conversation_id: UUID, user_id: UUID) -> int`.

- [ ] **Step 1: Write model, migration, idempotency, and optimistic-transition tests**

```python
def test_resource_model_has_private_device_fields_but_no_public_path_column():
    columns = {column.name for column in ConversationResource.__table__.columns}
    assert {"device_id", "device_session_id", "client_resource_id"} <= columns
    assert "local_path" not in columns


def test_registration_scope_is_idempotent(resource_repository, owned_conversation):
    scope = ResourceIdempotencyScope(
        conversation_id=owned_conversation.id,
        generation_id="generation-1",
        tool_call_id="call-1",
        candidate_ordinal=0,
        device_id="device-1",
        client_resource_id="local-1",
    )
    first = resource_repository.create_or_get(scope, _device_values())
    second = resource_repository.create_or_get(scope, _device_values())
    assert first.id == second.id


def test_migration_has_one_head_and_cascades_message_links(alembic_script):
    assert alembic_script.get_current_head() == "d7f8a9b0c1e2"
    assert _foreign_key_ondelete("message_resource_links", "message_id") == "CASCADE"
```

- [ ] **Step 2: Run the tests and verify failures**

Run: `python -m pytest tests/test_conversation_resource_models.py tests/test_conversation_resource_migration.py tests/integration/test_conversation_resource_repository_postgres.py -q`

Expected: FAIL because the models, repository, and migration do not exist.

- [ ] **Step 3: Implement tables and ownership-scoped repository methods**

Create:

- `conversation_resource_blobs`: user ID, SHA-256, size, media type, storage key, created/deleted timestamps, and a partial unique index on active `(user_id, sha256)`;
- `conversation_resources`: UUID, user/conversation ownership, origin kind, access, status, version, safe display metadata, blob/external/device locators, generation/tool provenance, idempotency key, change kind, superseded resource, and timestamps;
- `message_resource_links`: resource/message/category/ordinal with a unique `(message_id, resource_id, category)` constraint; and
- `resource_upload_sessions`: upload UUID, resource/user/device ownership, expected and received sizes, next offset, staging key, state, checksum, expiry, and timestamps.

Define `ResourceIdempotencyScope` as a frozen dataclass containing `conversation_id`, `generation_id`, `tool_call_id`, `candidate_ordinal`, `device_id`, and `client_resource_id`; its canonical digest is the unique idempotency key. Create revision `d7f8a9b0c1e2` with `down_revision = "f2a3b4c5d6e7"`. Use PostgreSQL enums with `create_type=False` after explicit `ENUM.create(checkfirst=True)`. Downgrade drops tables before enum types. `transition` executes `UPDATE conversation_resources SET status=:new_status, version=version+1 WHERE id=:id AND version=:expected_version`, applies the validated value fields in the same statement, and raises a typed stale-version error when no row changes.

- [ ] **Step 4: Run migration and repository tests**

Run: `python -m alembic heads`

Expected: exactly `d7f8a9b0c1e2 (head)`.

Run: `python -m pytest tests/test_conversation_resource_models.py tests/test_conversation_resource_migration.py tests/integration/test_conversation_resource_repository_postgres.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models/conversation_resource.py app/repositories/conversation_resource.py app/alembic/versions/d7f8a9b0c1e2_add_conversation_resources.py app/models/__init__.py app/models/conversation.py app/models/message.py tests/test_conversation_resource_models.py tests/test_conversation_resource_migration.py tests/integration/test_conversation_resource_repository_postgres.py
git commit -m "feat: persist conversation resources"
```

### Task 3: Build immutable managed storage and resumable upload sessions

**Files:**
- Create: `app/services/conversation_resource_storage.py`
- Create: `app/api/conversation_resources.py`
- Modify: `app/api/__init__.py`
- Modify: `app/main.py`
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Test: `tests/test_conversation_resource_storage.py`
- Test: `tests/test_conversation_resources_api.py`
- Test: `tests/test_conversation_resource_config.py`

**Interfaces:**
- Consumes: Task 2 blob/upload models and repository.
- Produces: `StagedUpload`; `ManagedResourceStorage.begin_upload(*, resource_id: UUID, user_id: UUID, expected_size: int, device_id: UUID | None) -> ResourceUploadSession`; `append_chunk(*, upload_id: UUID, user_id: UUID, offset: int, chunks: Iterable[bytes]) -> int`; `complete_upload(*, upload_id: UUID, user_id: UUID, expected_sha256: str) -> ConversationResourceBlob`; `abort_upload(*, upload_id: UUID, user_id: UUID) -> None`; `read_blob(blob: ConversationResourceBlob) -> BinaryIO`; and authenticated upload/content routes.

- [ ] **Step 1: Write failing storage and HTTP security tests**

```python
def test_chunk_offset_must_match_acknowledged_position(storage, upload):
    storage.append_chunk(upload_id=upload.id, user_id=upload.user_id, offset=0, chunks=[b"abc"])
    with pytest.raises(ResourceUploadOffsetConflict):
        storage.append_chunk(
            upload_id=upload.id,
            user_id=upload.user_id,
            offset=0,
            chunks=[b"duplicate"],
        )


def test_checksum_mismatch_removes_staging_bytes(storage, upload):
    storage.append_chunk(
        upload_id=upload.id,
        user_id=upload.user_id,
        offset=0,
        chunks=[b"content"],
    )
    with pytest.raises(ResourceChecksumMismatch):
        storage.complete_upload(
            upload_id=upload.id,
            user_id=upload.user_id,
            expected_sha256="0" * 64,
        )
    assert not storage.staging_path_for_test(upload.id).exists()


def test_active_content_downloads_as_an_attachment(auth_client, html_resource):
    response = auth_client.get(f"/conversation-resources/{html_resource.id}/content")
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"].startswith("attachment;")


def test_cross_owner_download_is_not_found(other_user_client, resource):
    assert other_user_client.get(
        f"/conversation-resources/{resource.id}/content"
    ).status_code == 404
```

- [ ] **Step 2: Run storage tests and verify failures**

Run: `python -m pytest tests/test_conversation_resource_storage.py tests/test_conversation_resources_api.py tests/test_conversation_resource_config.py -q`

Expected: FAIL because the storage service, routes, and settings are absent.

- [ ] **Step 3: Implement streaming content-addressed storage**

Add settings with these defaults: `conversation_resource_snapshot_max_bytes=25*1024*1024`, `conversation_resource_upload_chunk_bytes=1024*1024`, `conversation_resource_max_outstanding_uploads=4`, `conversation_resource_upload_ttl_seconds=1800`, `conversation_resource_storage_quota_bytes=1024*1024*1024`, and `conversation_resource_storage_path="app/storage/conversation_resources"`.

Write chunks to a random staging filename below the configured root, validate the exact next offset and cumulative byte count, update SHA-256 while streaming, fsync, and atomically promote to `<sha[:2]>/<sha>` only after size/checksum verification. Deduplicate active blobs per owner. Never derive a storage path from the display filename.

Expose:

- `POST /conversation-resources/uploads` to begin an authenticated staged upload;
- `PUT /conversation-resources/uploads/{upload_id}/chunks/{offset}` to append one bounded chunk;
- `POST /conversation-resources/uploads/{upload_id}/complete` to atomically finalize;
- `DELETE /conversation-resources/uploads/{upload_id}` to abort; and
- `GET /conversation-resources/{resource_id}/content` with ownership, attachment headers, and bounded range support.

Use `UploadFile`/request streaming; never call `await request.body()` for artifact bytes.

- [ ] **Step 4: Run focused storage/API tests**

Run: `python -m pytest tests/test_conversation_resource_storage.py tests/test_conversation_resources_api.py tests/test_conversation_resource_config.py tests/test_chat_images_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/conversation_resource_storage.py app/api/conversation_resources.py app/api/__init__.py app/main.py app/core/config.py .env.example tests/test_conversation_resource_storage.py tests/test_conversation_resources_api.py tests/test_conversation_resource_config.py
git commit -m "feat: store managed conversation resources"
```

### Task 4: Implement the central registry service and public projection

**Files:**
- Create: `app/services/conversation_resource_service.py`
- Modify: `app/core/container.py`
- Modify: `app/api/conversation_resources.py`
- Test: `tests/test_conversation_resource_service.py`
- Test: `tests/test_conversation_resource_projection.py`

**Interfaces:**
- Consumes: Tasks 1-3 contracts, repository, and storage.
- Produces: `ResourceRegistrationScope`; `ConversationResourceService.register_device_candidates(scope: ResourceRegistrationScope, candidates: Sequence[DeviceResourceCandidate]) -> list[ConversationResourceView]`; `register_managed_output(scope: ResourceRegistrationScope, blob_id: UUID, kind: ResourceKind, name: str) -> ConversationResourceView`; `register_external_input(user_id: UUID, conversation_id: UUID, url: str) -> ConversationResourceView`; `attach_external_source(user_id: UUID, conversation_id: UUID, generation_id: str, source_id: str, url: str) -> ConversationResourceView`; `link_to_message(resource_id: UUID, message_id: UUID, category: ResourceCategory, ordinal: int) -> None`; `transition(resource_id: UUID, expected_version: int, new_status: ResourceStatus, values: dict[str, Any]) -> ConversationResourceView`; `list_views(user_id: UUID, conversation_id: UUID) -> list[ConversationResourceView]`; `get_authorized(resource_id: UUID, user_id: UUID) -> ConversationResource`; and `delete_for_conversation(conversation_id: UUID, user_id: UUID) -> int`.

- [ ] **Step 1: Write failing service tests for registration, deduplication, and privacy**

```python
def test_register_device_candidates_returns_public_views_without_device_identity(service):
    views = service.register_device_candidates(_scope(), [_verified_candidate()])
    payload = views[0].model_dump(mode="json", exclude_none=True)
    assert payload["access"] == "device"
    assert "device_id" not in payload
    assert "client_resource_id" not in payload
    assert "local_path" not in json.dumps(payload)


def test_external_input_and_cited_source_share_canonical_url_identity(service):
    input_view = service.register_external_input(
        USER_ID, CONVERSATION_ID, "https://EXAMPLE.test/report#section"
    )
    source_view = service.attach_external_source(
        USER_ID,
        CONVERSATION_ID,
        "generation-1",
        "S1",
        "https://example.test/report",
    )
    assert input_view.resource_id == source_view.resource_id
    assert source_view.categories == [ResourceCategory.input, ResourceCategory.source]
```

- [ ] **Step 2: Run the tests and verify service failures**

Run: `python -m pytest tests/test_conversation_resource_service.py tests/test_conversation_resource_projection.py -q`

Expected: FAIL because the service and container providers do not exist.

- [ ] **Step 3: Implement ownership-scoped registration and projections**

Define:

```python
@dataclass(frozen=True)
class ResourceRegistrationScope:
    user_id: UUID
    conversation_id: UUID
    generation_id: str
    logical_turn_id: str
    tool_call_id: str
    device_id: UUID | None
    device_session_id: str | None
```

Registration caps candidates at eight per tool call, preserves ordinal, computes the idempotency key in the repository, links status/access correctly, and returns only `ConversationResourceView`. URL canonicalization lowercases/IDNA-normalizes the host, removes fragments/default ports, preserves meaningful paths and queries, and accepts HTTPS only for external resources.

Add `GET /conversations/{conversation_id}/resources` with stable category/ordinal/creation ordering. Project a managed action only when a committed blob exists, a device action only when its originating session is currently usable, and an external URL only after validation.

- [ ] **Step 4: Run focused service and authorization tests**

Run: `python -m pytest tests/test_conversation_resource_service.py tests/test_conversation_resource_projection.py tests/test_conversation_resources_api.py tests/test_client_invocation_isolation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/conversation_resource_service.py app/core/container.py app/api/conversation_resources.py tests/test_conversation_resource_service.py tests/test_conversation_resource_projection.py
git commit -m "feat: add conversation resource registry service"
```

### Task 5: Discover and verify device-local outputs without prose scraping

**Files:**
- Create: `client_backend/services/resource_extractors.py`
- Create: `client_backend/services/resource_verifier.py`
- Create: `client_backend/services/resource_catalog.py`
- Modify: `client_backend/core/config.py`
- Modify: `.env.client.example`
- Test: `tests/client_backend/test_resource_extractors.py`
- Test: `tests/client_backend/test_resource_verifier.py`
- Test: `tests/client_backend/test_resource_catalog.py`

**Interfaces:**
- Consumes: Task 1 `DeviceResourceCandidate` and current MCP `CallToolResult` shapes.
- Produces: private `LocalResourceCandidate`; `ResourceExtractorRegistry.declared_destinations(qualified_tool_id: str, arguments: dict[str, Any]) -> list[Path]`; `extract(qualified_tool_id: str, arguments: dict[str, Any], result: Any, before: Mapping[str, FileFingerprint | None]) -> list[LocalResourceCandidate]`; `LocalResourceVerifier.capture_before(paths: Sequence[Path]) -> dict[str, FileFingerprint | None]`; `verify(candidate: LocalResourceCandidate, before: FileFingerprint | None) -> VerifiedLocalResource`; `DeviceResourceCatalog.put(resource: VerifiedLocalResource) -> str`, `get(client_resource_id: str) -> CatalogEntry | None`, and `mark_unavailable(client_resource_id: str) -> None`; and `sanitize_client_tool_result(result: Any, resources: Sequence[VerifiedLocalResource]) -> Any`.

- [ ] **Step 1: Write failing generic and Desktop Commander fixture tests**

```python
def test_typed_resource_link_is_discovered_without_reading_prose(registry):
    result = {"content": [{"type": "resource_link", "uri": "file:///workspace/a.pdf"}]}
    candidates = registry.extract("any-server::export", {}, result, before={})
    assert [item.source for item in candidates] == ["typed_mcp_resource"]


def test_desktop_commander_write_file_uses_approved_path_argument(registry):
    candidates = registry.extract(
        "desktop-commander::write_file",
        {"path": "workspace/report.md", "content": "body"},
        {"content": [{"type": "text", "text": "Successfully wrote the file"}]},
        before={"workspace/report.md": None},
    )
    assert candidates[0].change_evidence == "destination_argument"


def test_path_like_prose_does_not_create_a_candidate(registry):
    assert registry.extract(
        "unknown::chat",
        {},
        "I mentioned workspace/report.md but created nothing",
        before={},
    ) == []


def test_replaced_or_linked_file_is_rejected(verifier, replacement_fixture):
    result = verifier.verify(replacement_fixture.candidate, replacement_fixture.before)
    assert result.verification_status == "rejected"
    assert result.rejection_code in {"link_or_reparse_point", "identity_changed"}


def test_sanitizer_preserves_blocks_but_removes_local_paths_and_embedded_bytes():
    result = {
        "content": [
            {"type": "resource_link", "uri": "file:///private/report.pdf"},
            {"type": "resource", "resource": {"blob": "c2VjcmV0"}},
        ]
    }
    sanitized = sanitize_client_tool_result(result, [_verified("local-1")])
    encoded = json.dumps(sanitized)
    assert "file:///private" not in encoded
    assert "c2VjcmV0" not in encoded
    assert sanitized["content"][0]["type"] == "resource_link"
    assert sanitized["content"][0]["client_resource_id"] == "local-1"


def test_bounded_embedded_resource_is_cataloged_without_websocket_bytes(extractor):
    result = _embedded_resource(name="notes.txt", blob=b"bounded content")
    resources, sanitized = extractor.extract_and_sanitize(result)
    assert resources[0].catalog_entry.is_sidecar_owned is True
    assert resources[0].byte_size == len(b"bounded content")
    assert "bounded content" not in json.dumps(sanitized)
```

- [ ] **Step 2: Run sidecar tests and verify failures**

Run: `python -m pytest tests/client_backend/test_resource_extractors.py tests/client_backend/test_resource_verifier.py tests/client_backend/test_resource_catalog.py -q`

Expected: FAIL because the sidecar resource modules do not exist.

- [ ] **Step 3: Implement extractor registry, verifier, and opaque catalog**

Implement typed `resource`, `resource_link`, and `resourceLink` extraction for any MCP tool. Typed HTTPS links pass through as external resources; local `file` resources enter verification; unsupported URI schemes remain non-actionable render content. Register exact Desktop Commander suffix contracts for `write_file` and `move_file`; accept server names `desktop-commander` and `desktop_commander`. `write_file` uses argument `path`; `move_file` uses destination argument `destination`. Do not claim outputs from shell/process tools because their filesystem effects are not structurally declared.

Capture pre-call fingerprints only for declared destination arguments. Verify with `lstat`, reject link/reparse/special entries, open without following links, compare file identity before/after open, and classify change evidence. For files over 25 MiB record size/stat only; do not compute a full hash.

Persist `{client_resource_id -> private canonical path, stat fingerprint, owner, installation, session}` below the existing profile root through an atomic replace. The public `DeviceResourceCandidate` returned to the server omits the path and private installation identity. Configure the sidecar defaults as `conversation_resource_snapshot_max_bytes=25*1024*1024`, `conversation_resource_max_candidates_per_tool_call=8`, and `conversation_resource_catalog_filename="conversation-resources.json"` under the existing profile root.

Sanitize the structured provider result before it crosses the runtime WebSocket. Preserve text and block shape, preserve validated HTTPS links, replace local-file URI/path fields with the opaque `client_resource_id`, and remove embedded blob/base64 bytes. Bounded device-executed embedded resources are decoded into a random file in a sidecar-owned output root, verified, cataloged, and later uploaded through the snapshot protocol; malformed or oversized blobs remain non-actionable previews or are rejected. This is a privacy boundary, not a renderer cleanup: private paths and bytes remain sidecar-only even when the MCP provider returned them.

- [ ] **Step 4: Run platform and security tests**

Run: `python -m pytest tests/client_backend/test_resource_extractors.py tests/client_backend/test_resource_verifier.py tests/client_backend/test_resource_catalog.py tests/client_backend/test_mcp_tool_execution_api.py tests/client_backend/test_local_mcp_manager.py -q`

Expected: PASS, with Windows reparse tests skipped only on non-Windows platforms and POSIX symlink tests skipped only on Windows.

- [ ] **Step 5: Commit**

```bash
git add client_backend/services/resource_extractors.py client_backend/services/resource_verifier.py client_backend/services/resource_catalog.py client_backend/core/config.py .env.client.example tests/client_backend/test_resource_extractors.py tests/client_backend/test_resource_verifier.py tests/client_backend/test_resource_catalog.py
git commit -m "feat: verify device tool output resources"
```

### Task 6: Preserve structured client-tool results and extend the runtime protocol

**Files:**
- Modify: `app/schemas/runtime_protocol.py`
- Modify: `app/services/client_runtime_store.py`
- Modify: `app/services/client_device_service.py`
- Modify: `app/api/device_runtime.py`
- Modify: `client_backend/services/runtime_bridge.py`
- Modify: `client_backend/services/local_mcp_manager.py`
- Modify: `client_backend/schemas/runtime.py`
- Test: `tests/test_runtime_protocol_contract.py`
- Test: `tests/test_client_runtime_store.py`
- Test: `tests/client_backend/test_runtime_bridge.py`

**Interfaces:**
- Consumes: Task 1 candidates and Task 5 sidecar services.
- Produces: `ResourceSnapshotRequest`, `ResourceSnapshotResult`, extended `RuntimeMessage`, `ToolDispatchResult.resource_candidates`, and a runtime request queue that carries tool or snapshot requests without conflating their results.

- [ ] **Step 1: Write failing protocol and bridge tests**

```python
def test_tool_result_preserves_mcp_blocks_and_verified_candidates():
    message = ToolDispatchResult(
        request_id="tool-1",
        success=True,
        result={
            "content": [
                {
                    "type": "resource_link",
                    "client_resource_id": "local-1",
                    "name": "report.pdf",
                }
            ]
        },
        resource_candidates=[_public_candidate()],
        execution_time_ms=4,
    )
    decoded = parse_runtime_message(dump_runtime_message(message))
    assert decoded.result["content"][0]["type"] == "resource_link"
    assert decoded.resource_candidates[0].client_resource_id == "local-1"
    encoded = dump_runtime_message(message)
    assert "file://" not in encoded
    assert "blob" not in encoded


@pytest.mark.asyncio
async def test_snapshot_request_runs_in_background_without_blocking_receive_loop(bridge):
    await bridge._handle_server_message(_snapshot_request("resource-1"))
    assert bridge.snapshot_tasks
    assert bridge.received_next_message.is_set()
```

- [ ] **Step 2: Run tests and verify protocol failures**

Run: `python -m pytest tests/test_runtime_protocol_contract.py tests/test_client_runtime_store.py tests/client_backend/test_runtime_bridge.py -q`

Expected: FAIL because the resource fields and snapshot message variants are absent.

- [ ] **Step 3: Implement the protocol without flattening results**

Add:

```python
class ResourceSnapshotRequest(BaseModel):
    type: Literal["resource_snapshot_request"] = "resource_snapshot_request"
    request_id: str
    resource_id: str
    client_resource_id: str
    expected_session_id: str
    max_bytes: int = Field(gt=0)
    upload_token: str


class ResourceSnapshotResult(BaseModel):
    type: Literal["resource_snapshot_result"] = "resource_snapshot_result"
    request_id: str
    resource_id: str
    success: bool
    status: Literal["available", "unavailable", "rejected"]
    access: Literal["managed", "device"] | None = None
    error_code: str | None = None
```

`ToolDispatchResult.resource_candidates` defaults to an empty list. The runtime store queue accepts `ToolDispatchRequest | ResourceSnapshotRequest`; only tool requests allocate result futures. The gateway publishes tool results as before and forwards snapshot results to `ConversationResourceService` after validating device/session ownership.

Issue `upload_token` as a short-lived, single-use credential scoped to user, resource, upload session, originating device, and maximum bytes. Redact it from runtime logs, traces, exception strings, and persisted messages. Consuming, expiring, or aborting the upload invalidates the token.

Before each eligible MCP invocation, the sidecar extractor captures declared destinations. After invocation, it extracts/verifies candidates, writes private catalog entries, sanitizes the original structured result through Task 5's privacy boundary, and includes only that safe result plus public candidates in `ToolDispatchResult`. Snapshot requests create tracked background tasks; disconnect cancels and consumes them cleanly.

- [ ] **Step 4: Run runtime regression tests**

Run: `python -m pytest tests/test_runtime_protocol_contract.py tests/test_client_runtime_store.py tests/test_client_invocation_isolation.py tests/test_client_tool_isolation.py tests/test_multi_sidecar_hardening.py tests/client_backend/test_runtime_bridge.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/schemas/runtime_protocol.py app/services/client_runtime_store.py app/services/client_device_service.py app/api/device_runtime.py client_backend/services/runtime_bridge.py client_backend/services/local_mcp_manager.py client_backend/schemas/runtime.py tests/test_runtime_protocol_contract.py tests/test_client_runtime_store.py tests/client_backend/test_runtime_bridge.py
git commit -m "feat: carry resources across the client runtime"
```

### Task 7: Register candidates during tool execution and launch snapshots

**Files:**
- Modify: `app/ai/client_runtime_tools.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/tool_result_rendering.py`
- Modify: `app/ai/workflow/tool_loop.py`
- Modify: `app/ai/tool_context.py`
- Modify: `app/ai/graph.py`
- Modify: `app/services/conversation_resource_service.py`
- Test: `tests/test_client_tool_structured_results.py`
- Test: `tests/test_tool_execution_resources.py`
- Test: `tests/test_tool_result_rendering.py`

**Interfaces:**
- Consumes: Tasks 4 and 6 service/protocol.
- Produces: `ClientToolInvocationResult`, `unwrap_client_tool_result(result)`, `ResourceCaptureScope`, registered resource views in tool artifacts under canonical `resources`, and snapshot requests queued after registration.

- [ ] **Step 1: Write failing end-to-end tool-result tests**

```python
@pytest.mark.asyncio
async def test_client_tool_keeps_structured_result_for_the_normalizer(client_tool):
    result = await client_tool.ainvoke({"path": "workspace/report.md", "content": "x"})
    assert isinstance(result, ClientToolInvocationResult)
    assert result.provider_result["content"][0]["type"] == "resource_link"
    assert "file://" not in json.dumps(result.provider_result)


@pytest.mark.asyncio
async def test_tool_execution_registers_resource_before_returning_model_content(harness):
    outputs, artifacts, _ = await harness.run_client_tool_result(
        provider_result={"content": [{"type": "text", "text": "created"}]},
        candidates=[_public_candidate()],
    )
    assert artifacts[0]["resources"][0]["resource_id"]
    assert "[[resource:R1]]" in outputs[0]["content"]
    assert harness.snapshot_request.resource_id == artifacts[0]["resources"][0]["resource_id"]
```

- [ ] **Step 2: Run tests and verify the stringification regression**

Run: `python -m pytest tests/test_client_tool_structured_results.py tests/test_tool_execution_resources.py tests/test_tool_result_rendering.py -q`

Expected: FAIL because `_format_tool_result` returns JSON text and tool execution has no resource service.

- [ ] **Step 3: Preserve structured results and register resources synchronously**

Replace `_format_tool_result` with:

```python
@dataclass(frozen=True)
class ClientToolInvocationResult:
    provider_result: Any
    resource_candidates: tuple[DeviceResourceCandidate, ...]
    bound_device_id: str
    bound_session_id: str


def unwrap_client_tool_result(result: Any) -> tuple[Any, tuple[DeviceResourceCandidate, ...]]:
    if isinstance(result, ClientToolInvocationResult):
        return result.provider_result, result.resource_candidates
    return result, ()
```

Extend `ToolContext` with `generation_id`. Thread `ResourceCaptureScope` and `ConversationResourceService` through the existing tool executor rather than registering inside the LangChain tool closure, because only the executor owns `tool_call_id`. Register candidates immediately after a successful invocation, add public views to the artifact's `resources`, append a compact alias inventory to model content, then enqueue snapshot requests without awaiting file hashing/upload.

`normalize_tool_result_for_rendering` receives the structured, sidecar-sanitized provider result and preserves typed resources in `render`. It never receives private paths, embedded bytes, or the private candidate catalog entry.

- [ ] **Step 4: Run tool execution, graph, and isolation tests**

Run: `python -m pytest tests/test_client_tool_structured_results.py tests/test_tool_execution_resources.py tests/test_tool_result_rendering.py tests/test_client_invocation_isolation.py tests/test_graph_tool_budget.py tests/test_specialist_middleware.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/ai/client_runtime_tools.py app/ai/tool_execution.py app/ai/tool_result_rendering.py app/ai/workflow/tool_loop.py app/ai/tool_context.py app/ai/graph.py app/services/conversation_resource_service.py tests/test_client_tool_structured_results.py tests/test_tool_execution_resources.py tests/test_tool_result_rendering.py
git commit -m "feat: register resources from client tool results"
```

### Task 8: Upload snapshots asynchronously and support device actions

**Files:**
- Create: `client_backend/services/resource_snapshots.py`
- Create: `client_backend/api/resources.py`
- Modify: `client_backend/main.py`
- Modify: `client_backend/services/runtime_bridge.py`
- Modify: `client_backend/services/server_api.py`
- Modify: `app/api/conversation_resources.py`
- Modify: `app/services/conversation_resource_service.py`
- Test: `tests/client_backend/test_resource_snapshots.py`
- Test: `tests/client_backend/test_resource_actions.py`
- Test: `tests/test_conversation_resource_snapshot_flow.py`

**Interfaces:**
- Consumes: Tasks 3, 5, and 6 upload protocol, catalog, and snapshot command.
- Produces: `ResourceSnapshotUploader.upload(request)`, `POST /conversation-resources/{id}/refresh`, `POST /conversation-resources/{id}/save`, `POST /conversation-resources/{id}/open`, and sidecar-local open/refresh handlers.

- [ ] **Step 1: Write failing async, resume, disconnect, and stale-file tests**

```python
@pytest.mark.asyncio
async def test_snapshot_resumes_from_server_acknowledged_offset(uploader, fake_server):
    fake_server.fail_after_offset_once = 1_048_576
    result = await uploader.upload(_request(), _catalog_entry(size=2_000_000))
    assert result.success is True
    assert fake_server.received_offsets == [0, 1_048_576, 1_048_576]


@pytest.mark.asyncio
async def test_changed_file_falls_back_to_device_without_upload(uploader, changed_entry):
    result = await uploader.upload(_request(), changed_entry)
    assert result.status == "available"
    assert result.access == "device"
    assert result.error_code == "file_changed"


def test_open_requires_originating_connected_device(auth_client, foreign_device_resource):
    response = auth_client.post(
        f"/conversation-resources/{foreign_device_resource.id}/open"
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "origin_device_unavailable"
```

- [ ] **Step 2: Run tests and verify failures**

Run: `python -m pytest tests/client_backend/test_resource_snapshots.py tests/client_backend/test_resource_actions.py tests/test_conversation_resource_snapshot_flow.py -q`

Expected: FAIL because snapshot uploader and actions do not exist.

- [ ] **Step 3: Implement non-blocking resumable upload and reverified actions**

The uploader resolves the opaque catalog entry, re-verifies file identity, begins or resumes an upload, sends 1 MiB chunks from the acknowledged offset, and computes SHA-256 during transfer. It uses at most three retry attempts for retryable connection failures and never applies a total elapsed deadline while acknowledged progress continues. Server idle/read liveness failures preserve the acknowledged offset for retry.

On completion the server atomically attaches the immutable blob and transitions the resource to `available/managed`. A changed, oversized, policy-denied, or explicitly non-snapshotted file transitions to `available/device` when it still exists; missing/disconnected becomes `unavailable`; unsafe becomes `rejected`.

For **Open on originating device**, the server authorizes the resource and queues an action bound to the current session. The sidecar resolves the opaque ID, re-verifies identity, and opens only the verified file. **Refresh** updates stat/status without exposing the path. **Save to conversation** runs the same snapshot policy after explicit user action.

- [ ] **Step 4: Run snapshot and device regression suites**

Run: `python -m pytest tests/client_backend/test_resource_snapshots.py tests/client_backend/test_resource_actions.py tests/test_conversation_resource_snapshot_flow.py tests/test_client_invocation_isolation.py tests/test_multi_sidecar_hardening.py tests/test_client_runtime_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add client_backend/services/resource_snapshots.py client_backend/api/resources.py client_backend/main.py client_backend/services/runtime_bridge.py client_backend/services/server_api.py app/api/conversation_resources.py app/services/conversation_resource_service.py tests/client_backend/test_resource_snapshots.py tests/client_backend/test_resource_actions.py tests/test_conversation_resource_snapshot_flow.py
git commit -m "feat: snapshot and open device resources"
```

### Task 9: Register application outputs and adapt canonical sources

**Files:**
- Modify: `app/services/conversation_resource_service.py`
- Modify: `app/services/message_service.py`
- Modify: `app/services/chat_image_service.py`
- Modify: `app/ai/graph.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/tool_result_rendering.py`
- Modify: `app/ui/rag_artifacts.py`
- Test: `tests/test_application_output_resources.py`
- Test: `tests/test_server_mcp_output_resources.py`
- Test: `tests/test_conversation_source_projection.py`
- Test: `tests/test_message_service_resource_persistence.py`

**Interfaces:**
- Consumes: Task 4 registry; canonical `SourceRecord` and source registry from the web research plan; existing RAG document identities, chat images, and canvas artifacts.
- Produces: `ConversationResourceService.register_generated_image(scope: ResourceRegistrationScope, image_id: UUID) -> ConversationResourceView`; `register_canvas_artifact(scope: ResourceRegistrationScope, item: RichItem) -> ConversationResourceView`; `register_server_mcp_resources(scope: ResourceRegistrationScope, qualified_tool_id: str, result: Any) -> list[ConversationResourceView]`; `project_web_sources(generation_id: str, sources: Sequence[SourceRecord]) -> list[ConversationResourceView]`; `project_rag_sources(records: Sequence[RagCitationRecord]) -> list[ConversationResourceView]`; and message-resource links for server-owned outputs/sources.

- [ ] **Step 1: Write failing application-output and source reuse tests**

```python
def test_generated_image_is_an_output_resource_with_existing_protected_url(service):
    view = service.register_generated_image(_scope(), _stored_chat_image())
    assert view.categories == [ResourceCategory.output]
    assert view.kind == ResourceKind.image
    assert view.content_url.startswith("/chat-images/")


def test_canvas_is_output_but_tool_render_is_not(service):
    canvas = service.register_canvas_artifact(_scope(), _canvas_item())
    assert canvas.kind == ResourceKind.artifact
    assert service.register_tool_render(_scope(), _transient_render()) is None


def test_web_source_uses_message_metadata_without_creating_resource_row(service, source):
    view = service.project_web_sources("generation-1", [source])[0]
    assert view.resource_id == "source:generation-1:S1"
    assert service.repository.created_count == 0


def test_bounded_server_mcp_embedded_resource_becomes_managed(service):
    views = service.register_server_mcp_resources(
        _scope(), "documents::export", _embedded_resource("report.pdf", b"pdf")
    )
    assert views[0].access == ResourceAccess.managed
    assert views[0].content_url.endswith("/content")


def test_server_mcp_prose_and_unapproved_local_uri_are_not_outputs(service):
    result = {"content": [{"type": "text", "text": "wrote /private/report.pdf"}]}
    assert service.register_server_mcp_resources(_scope(), "unknown::run", result) == []
```

- [ ] **Step 2: Run tests and verify missing adapters**

Run: `python -m pytest tests/test_application_output_resources.py tests/test_server_mcp_output_resources.py tests/test_conversation_source_projection.py tests/test_message_service_resource_persistence.py -q`

Expected: FAIL because application outputs and source adapters are not registered.

- [ ] **Step 3: Add explicit owning-service adapters**

Register generated images only after `ChatImageStorageService.store` returns a protected reference. Register a canvas only after finalization accepts its durable rich item. For server-executed MCP results, accept typed HTTPS resource links as external output resources and stream bounded embedded bytes directly into Task 3 managed storage; reject malformed/oversized embedded data and unapproved server-local file URIs. Preserve typed render blocks, but never infer an output from text. Keep widgets, tables, charts, and tool renders outside Outputs unless an owner explicitly exports a durable artifact.

Project canonical web sources directly from `SourceRecord`; source-only identity is `source:{generation_id}:{source_id}`. Project RAG sources from existing document/citation records and authenticated document routes. When canonical HTTPS URL equality finds an external input record, add a source message link to that resource instead of emitting a second identity.

Persist message links for completed, interrupted, partial, cancelled, and continued assistant messages. Use the same `bot_message_id` allocated before streaming so links can exist before terminal message persistence.

- [ ] **Step 4: Run application/source and message lifecycle tests**

Run: `python -m pytest tests/test_application_output_resources.py tests/test_server_mcp_output_resources.py tests/test_conversation_source_projection.py tests/test_message_service_resource_persistence.py tests/test_message_generation_lifecycle.py tests/test_message_service_event_streaming.py tests/test_rich_response_metadata.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/conversation_resource_service.py app/services/message_service.py app/services/chat_image_service.py app/ai/graph.py app/ai/tool_execution.py app/ai/tool_result_rendering.py app/ui/rag_artifacts.py tests/test_application_output_resources.py tests/test_server_mcp_output_resources.py tests/test_conversation_source_projection.py tests/test_message_service_resource_persistence.py
git commit -m "feat: register application outputs and sources"
```

### Task 10: Move new conversation inputs to staged resource IDs

**Files:**
- Modify: `app/schemas/message.py`
- Modify: `app/api/ai_sdk.py`
- Modify: `app/services/message_service.py`
- Modify: `app/ai/history.py`
- Modify: `app/api/conversation_resources.py`
- Modify: `demo.py`
- Test: `tests/test_message_input_resources.py`
- Test: `tests/test_ai_sdk_input_resource_contract.py`
- Test: `tests/test_legacy_attachment_compatibility.py`

**Interfaces:**
- Consumes: Tasks 3-4 upload and registry services.
- Produces: `MessageCreate.resource_ids: list[UUID]`, `MessageCreate.conversation_resources_v1: bool = False`, staged-upload finalization, typed AI SDK URL/file input extraction, and compatibility externalization for legacy base64 images.

- [ ] **Step 1: Write failing staged-input and compatibility tests**

```python
def test_message_links_owned_staged_resource_in_same_transaction(message_service):
    message = message_service.create_message(
        MessageCreate(
            conversation_id=CONVERSATION_ID,
            content="Analyze the attachment",
            resource_ids=[RESOURCE_ID],
        ),
        USER_ID,
    )
    assert message.resources[0].resource_id == str(RESOURCE_ID)
    assert message.resources[0].categories == [ResourceCategory.input]


def test_new_ai_sdk_file_part_uses_resource_id_not_base64(ai_sdk_request):
    parsed = extract_ai_sdk_input_resources(ai_sdk_request)
    assert parsed.resource_ids == [RESOURCE_ID]
    assert parsed.legacy_attachments == []


def test_legacy_image_is_externalized_and_linked_once(message_service):
    message = message_service.create_message(_legacy_base64_message(), USER_ID)
    assert message.message_metadata["attachments"][0]["image_id"]
    assert len(message.resources) == 1


def test_legacy_client_does_not_receive_resource_only_markers(message_service):
    message = message_service.create_message(
        MessageCreate(content="hello", conversation_resources_v1=False), USER_ID
    )
    assert "[[resource:" not in message.content
```

- [ ] **Step 2: Run tests and verify missing resource-ID input path**

Run: `python -m pytest tests/test_message_input_resources.py tests/test_ai_sdk_input_resource_contract.py tests/test_legacy_attachment_compatibility.py -q`

Expected: FAIL because messages accept only legacy attachments.

- [ ] **Step 3: Implement canonical inputs and retain read-only compatibility**

Validate that every supplied resource is staged/available, owned by the requester, and belongs to the same conversation before creating the message; link resources in the message transaction. Resolve vision-capable image resources through the existing protected image loader for model input. Ordinary files remain panel resources unless an existing document ingestion path explicitly consumes them.

AI SDK accepts a resource ID on file parts and typed HTTPS URL input parts. Add an explicit `conversation_resources_v1` request capability, thread it into `MessageCreate`, and have Streamlit opt in when it sends staged resource IDs. Only capable clients receive custom resource parts or resource-only marker content. Keep legacy `attachments` accepted at the API boundary, immediately pass them through current chat-image validation/storage, create canonical resource links, and forbid any new persistence of raw base64 in message metadata.

- [ ] **Step 4: Run input, history, and existing image tests**

Run: `python -m pytest tests/test_message_input_resources.py tests/test_ai_sdk_input_resource_contract.py tests/test_legacy_attachment_compatibility.py tests/test_message_history_pipeline.py tests/test_base_agent_image_history.py tests/test_ai_sdk_v6_stream_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/schemas/message.py app/api/ai_sdk.py app/services/message_service.py app/ai/history.py app/api/conversation_resources.py demo.py tests/test_message_input_resources.py tests/test_ai_sdk_input_resource_contract.py tests/test_legacy_attachment_compatibility.py
git commit -m "feat: attach canonical resources to user messages"
```

### Task 11: Ground answer resource tokens and device actions

**Files:**
- Create: `app/ai/resource_grounding.py`
- Modify: `app/core/rich_response.py`
- Modify: `app/ai/workflow/specialists.py`
- Modify: `app/ai/workflow/finalization.py`
- Modify: `app/ai/graph.py`
- Test: `tests/test_resource_grounding.py`
- Test: `tests/test_resource_link_rich_item.py`
- Test: `tests/test_output_validation.py`

**Interfaces:**
- Consumes: Task 1 aliases and Task 4 public views.
- Produces: `RESOURCE_TOKEN_PATTERN`, `ResourceGroundingResolver.resolve(text, aliases, rich_capable)`, `ResourceGroundingStreamFilter.feed(delta)`, `finish()`, and device-capable `ResourceLinkPayload` actions.

- [ ] **Step 1: Write failing token validation and split-stream tests**

```python
def test_managed_token_becomes_server_owned_markdown_link():
    result = resolver.resolve("Download [[resource:R1]].", [_managed_alias()], True)
    assert result.text == "Download [report.pdf](/conversation-resources/res-1/content)."


def test_device_token_becomes_rich_marker_without_fake_url():
    result = resolver.resolve("Open [[resource:R1]].", [_device_alias()], True)
    assert result.text == "Open\n<!--rich:resource:res-1-->\n."
    assert result.rich_items[0].payload.resource_id == "res-1"
    assert result.rich_items[0].payload.url is None


def test_unknown_and_cross_conversation_tokens_are_removed():
    result = resolver.resolve("[[resource:R999]] [[resource:R2]]", [_managed_alias()], True)
    assert "resource:" not in result.text
    assert result.violation_codes == ["unknown_resource_alias", "unknown_resource_alias"]


def test_stream_filter_holds_a_token_split_across_deltas():
    stream = ResourceGroundingStreamFilter([_managed_alias()], rich_capable=True)
    assert stream.feed("See [[res") == "See "
    assert stream.feed("ource:R1]] now") == "[report.pdf](/conversation-resources/res-1/content) now"
```

- [ ] **Step 2: Run tests and verify missing resolver failures**

Run: `python -m pytest tests/test_resource_grounding.py tests/test_resource_link_rich_item.py tests/test_output_validation.py -q`

Expected: FAIL because resource token grounding and URL-less device actions do not exist.

- [ ] **Step 3: Implement server-owned resolution and prompt inventory**

Extend `ResourceLinkPayload` with optional `resource_id`, optional URL, and a strict action list containing `open_device`, `refresh`, or `save_to_conversation`; require exactly one of an ordinary URL or resource ID/action descriptor. Preserve existing URL-only resource links.

Append bounded resource aliases to the model after each registered tool result. Prompt instructions permit only supplied `[[resource:R#]]` tokens. The resolver chooses labels/targets, records violations, and degrades device resources to plain text such as `report.pdf (available on the originating device)` when rich capability is absent. It never accepts durable IDs or URLs written by the model.

Run resolution for normal, partial, interrupted, cancelled, continued, and subagent-composed answers. No extra model call is introduced.

- [ ] **Step 4: Run grounding and rich-response regressions**

Run: `python -m pytest tests/test_resource_grounding.py tests/test_resource_link_rich_item.py tests/test_output_validation.py tests/test_rich_response_contract.py tests/test_rich_response_metadata.py tests/test_rich_response_streaming.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/ai/resource_grounding.py app/core/rich_response.py app/ai/workflow/specialists.py app/ai/workflow/finalization.py app/ai/graph.py tests/test_resource_grounding.py tests/test_resource_link_rich_item.py tests/test_output_validation.py
git commit -m "feat: ground conversation resource links"
```

### Task 12: Add canonical resource events and history parity

**Files:**
- Create: `app/services/conversation_resource_events.py`
- Modify: `app/services/event_streaming/events.py`
- Modify: `app/services/event_streaming/graph_public_projection.py`
- Modify: `app/services/event_streaming/internal_sse.py`
- Modify: `app/services/message_service.py`
- Modify: `app/schemas/message.py`
- Test: `tests/test_conversation_resource_events.py`
- Test: `tests/test_internal_sse_resource_contract.py`
- Test: `tests/test_message_resource_history.py`

**Interfaces:**
- Consumes: Tasks 4 and 11 public views and resolved answer items.
- Produces: canonical `resources_upsert` stream events, Redis-backed `ConversationResourceEventBus.publish/subscribe` with an in-memory test/development implementation, batched history projection, and `MessageRead.resources`.

- [ ] **Step 1: Write failing live/reload/order tests**

```python
@pytest.mark.asyncio
async def test_resource_upsert_precedes_text_that_references_it(stream_harness):
    events = await stream_harness.run_with_resource_answer()
    upsert_index = next(i for i, event in enumerate(events) if event.type == "resources_upsert")
    text_index = next(i for i, event in enumerate(events) if event.type == "message_delta")
    assert upsert_index < text_index


def test_history_uses_same_resource_id_version_and_order(live_events, projected_history):
    live = _latest_views(live_events)
    stored = projected_history[0].resources
    assert [(x.resource_id, x.version) for x in stored] == [
        (x.resource_id, x.version) for x in live
    ]
```

- [ ] **Step 2: Run tests and verify missing event failures**

Run: `python -m pytest tests/test_conversation_resource_events.py tests/test_internal_sse_resource_contract.py tests/test_message_resource_history.py -q`

Expected: FAIL because `resources_upsert` is not a canonical event and history has no resource projection.

- [ ] **Step 3: Implement bounded publication and batched history loading**

Add `resources_upsert` to `StreamEventType` with `ResourceUpsertPayload`. Publish initial registrations before answer deltas. Use Redis pub/sub in multi-worker production and an in-memory bus only for tests or explicit local development. Merge resource-bus updates into the active generation stream without waiting for snapshot completion; stop the subscription when the generation terminates. If upload completes later, the list/history endpoints remain authoritative and clients refresh on panel open or explicit action.

Batch-load message links for each history page to avoid N+1 queries. `MessageRead.resources` defaults to an empty list. Preserve the exact `(category ordinal, created_at, resource_id)` order in live events and reload. Internal SSE sends one `resources_upsert` envelope unchanged after public sanitization.

- [ ] **Step 4: Run stream, lifecycle, and history tests**

Run: `python -m pytest tests/test_conversation_resource_events.py tests/test_internal_sse_resource_contract.py tests/test_message_resource_history.py tests/test_message_service_event_streaming.py tests/test_message_generation_lifecycle.py tests/test_routing_v2_continuation_streaming.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/conversation_resource_events.py app/services/event_streaming/events.py app/services/event_streaming/graph_public_projection.py app/services/event_streaming/internal_sse.py app/services/message_service.py app/schemas/message.py tests/test_conversation_resource_events.py tests/test_internal_sse_resource_contract.py tests/test_message_resource_history.py
git commit -m "feat: stream and reload conversation resources"
```

### Task 13: Project native and custom resource parts to AI SDK v6

**Files:**
- Modify: `app/services/event_streaming/ai_sdk_v6.py`
- Modify: `app/api/ai_sdk.py`
- Test: `tests/test_ai_sdk_resource_stream_contract.py`
- Test: `tests/test_ai_sdk_context_window.py`
- Test: `tests/test_ai_sdk_v6_stream_contract.py`

**Interfaces:**
- Consumes: Task 10 `conversation_resources_v1`, Task 12 `resources_upsert` and `MessageRead.resources`; web plan `source-url` projection.
- Produces: `project_resource_view_to_ai_sdk_parts(view: ConversationResourceView, *, conversation_resources_v1: bool) -> list[dict[str, Any]]`, `data-conversation-resources` upserts, native managed `file` parts, native source `source-url` parts, and terminal/history parity.

- [ ] **Step 1: Write failing AI SDK live/history contract tests**

```python
@pytest.mark.asyncio
async def test_managed_output_emits_file_and_registry_parts(adapter):
    payloads = await _collect(adapter, _resources_upsert(_managed_output()))
    assert next(p for p in payloads if p["type"] == "file")["url"].endswith("/content")
    data = next(p for p in payloads if p["type"] == "data-conversation-resources")
    assert data["data"]["resources"][0]["resource_id"] == "res-1"


def test_device_output_has_custom_action_but_no_file_part():
    parts = project_resource_view_to_ai_sdk_parts(
        _device_output(), conversation_resources_v1=True
    )
    assert not any(part["type"] == "file" for part in parts)
    assert parts[0]["type"] == "data-conversation-resources"


def test_client_without_resource_capability_gets_no_custom_part(adapter):
    parts = adapter.project_resources(
        [_device_output()], conversation_resources_v1=False
    )
    assert not any(part["type"] == "data-conversation-resources" for part in parts)


def test_history_matches_live_source_and_resource_ids(ai_sdk_history, live_payloads):
    assert _resource_ids(ai_sdk_history) == _resource_ids(live_payloads)
    assert _source_ids(ai_sdk_history) == _source_ids(live_payloads)
```

- [ ] **Step 2: Run tests and verify projection failures**

Run: `python -m pytest tests/test_ai_sdk_resource_stream_contract.py tests/test_ai_sdk_context_window.py tests/test_ai_sdk_v6_stream_contract.py -q`

Expected: FAIL because the adapter ignores `resources_upsert` and history emits only legacy image files.

- [ ] **Step 3: Implement capability-safe AI SDK projection**

For managed file/image resources emit `{type: "file", url, mediaType, filename}` exactly once per stable `(resource_id, version)`. For web/external sources emit `{type: "source-url", sourceId, url, title}`. Emit `data-conversation-resources` only when `conversation_resources_v1` is true so the panel has categories, status, access, actions, and versions.

Do not emit a file part for device-only, unavailable, rejected, or deleted records. Terminal projection and history use the same helper as live streaming. Existing non-resource clients retain text and legacy image compatibility; private device or storage fields never reach parts.

- [ ] **Step 4: Run complete AI SDK contract suite**

Run: `python -m pytest tests/test_ai_sdk_resource_stream_contract.py tests/test_ai_sdk_context_window.py tests/test_ai_sdk_v6_stream_contract.py tests/test_ai_sdk_assistant_ui_compat.py tests/test_rich_response_streaming.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/event_streaming/ai_sdk_v6.py app/api/ai_sdk.py tests/test_ai_sdk_resource_stream_contract.py tests/test_ai_sdk_context_window.py tests/test_ai_sdk_v6_stream_contract.py
git commit -m "feat: project conversation resources to ai sdk"
```

### Task 14: Add the Streamlit Inputs, Outputs, and Sources experience

**Files:**
- Create: `app/ui/conversation_resources.py`
- Modify: `demo.py`
- Modify: `app/ui/stream_markdown.py`
- Test: `tests/test_demo_conversation_resources.py`
- Test: `tests/test_demo_stream_rendering.py`
- Test: `tests/test_demo_rich_response.py`

**Interfaces:**
- Consumes: Task 12 internal SSE/history objects and Task 11 rich device actions.
- Produces: `ConversationResourceState.upsert`, `group_resource_views`, `render_conversation_resources_panel`, `render_inline_resource`, and action callbacks for open/refresh/save.

- [ ] **Step 1: Reconcile the pre-existing renderer work before editing**

Run: `git status --short app/ui/stream_markdown.py tests/test_stream_markdown_bracket_math.py`

Expected: clean in the implementation worktree. If not clean, stop this task and integrate the owning session's commit before proceeding; do not overwrite it.

- [ ] **Step 2: Write failing Streamlit state/render tests**

```python
def test_upsert_replaces_only_with_a_newer_version():
    state = ConversationResourceState()
    state.upsert(_resource("res-1", version=2, status="available"))
    state.upsert(_resource("res-1", version=1, status="snapshot_pending"))
    assert state.items["res-1"].status == "available"


def test_panel_groups_one_multi_role_resource_without_duplication():
    grouped = group_resource_views([_input_and_source_url()])
    assert [x.resource_id for x in grouped.inputs] == ["res-url"]
    assert [x.resource_id for x in grouped.sources] == ["res-url"]


def test_unavailable_device_resource_renders_status_not_a_blank_link(render_harness):
    rendered = render_harness.resource(_unavailable_device_output())
    assert "Unavailable" in rendered.text
    assert rendered.links == []
```

- [ ] **Step 3: Run tests and verify missing UI module failures**

Run: `python -m pytest tests/test_demo_conversation_resources.py tests/test_demo_stream_rendering.py tests/test_demo_rich_response.py -q`

Expected: FAIL because canonical resource state and panel rendering do not exist.

- [ ] **Step 4: Implement focused UI state, panel, and actions**

Move resource-specific rendering out of `demo.py` into `app/ui/conversation_resources.py`. Upsert by `(resource_id, version)`, preserve server ordering, and render three tabs/sections: Inputs, Outputs, Sources. Managed/external entries use real links; device entries use buttons for open/refresh/save; pending and unavailable entries show explicit status.

Consume live `resources_upsert`, terminal message resources, and `GET /conversations/{id}/resources` through the same parser. Refresh the list when the panel opens, after actions, and after history pagination. Reuse `app/ui/stream_markdown.py` only for validated marker/token segmentation after incorporating the other session's bracket-math changes.

- [ ] **Step 5: Run Streamlit and shared stream tests**

Run: `python -m pytest tests/test_demo_conversation_resources.py tests/test_demo_stream_rendering.py tests/test_demo_rich_response.py tests/test_internal_sse_resource_contract.py tests/test_message_resource_history.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/ui/conversation_resources.py app/ui/stream_markdown.py demo.py tests/test_demo_conversation_resources.py tests/test_demo_stream_rendering.py tests/test_demo_rich_response.py
git commit -m "feat: render conversation resources in streamlit"
```

### Task 15: Add production evaluation, observability, cleanup, and release gates

**Files:**
- Create: `app/observability/conversation_resources.py`
- Create: `app/evaluation/conversation_resources.py`
- Create: `eval/conversation_resources/cases.json`
- Create: `scripts/evaluate_conversation_resources.py`
- Create: `docs/conversation-resource-runbook.md`
- Create: `docs/conversation-resource-cleanup-inventory.md`
- Modify: `app/api/health.py`
- Modify: `app/core/config.py`
- Modify: `app/ai/client_runtime_tools.py`
- Modify: `app/ai/tool_result_rendering.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `demo.py`
- Test: `tests/test_conversation_resource_metrics.py`
- Test: `tests/test_conversation_resource_evaluation.py`
- Test: `tests/test_conversation_resource_cleanup.py`
- Test: `tests/test_production_readiness_contract.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: `ConversationResourceMetrics`, `ConversationResourceHealth`, deterministic `evaluate_cases`, release report JSON, operator runbook, and checked cleanup inventory.

- [ ] **Step 1: Write failing metrics, evaluation, and active-path inventory tests**

```python
def test_metrics_reject_names_paths_urls_and_hashes_as_labels():
    labels = ConversationResourceMetrics.label_names()
    assert not {"filename", "path", "url", "sha256", "user_id"}.intersection(labels)


def test_general_cases_cover_required_resource_behaviors():
    cases = load_cases(CASES_PATH)
    assert REQUIRED_CAPABILITIES <= {cap for case in cases for cap in case.capabilities}
    assert all("T1" not in json.dumps(case.model_dump()) for case in cases)


def test_active_paths_do_not_flatten_or_guess_resources():
    violations = scan_active_resource_paths(REPOSITORY_ROOT)
    assert violations == []
```

`REQUIRED_CAPABILITIES` contains typed MCP preservation, Desktop Commander write, bounded server-MCP embedded output, deceptive prose rejection, create/modify/unknown classification, link/reparse rejection, managed snapshot, oversized device fallback, disconnected device, checksum mismatch, retry/resume, input upload, generated output, canvas output, web source, RAG source, multi-role URL deduplication, token grounding, AI SDK parity, Streamlit parity, history reload, ownership isolation, and deletion cleanup.

- [ ] **Step 2: Run tests and capture current failures**

Run: `python -m pytest tests/test_conversation_resource_metrics.py tests/test_conversation_resource_evaluation.py tests/test_conversation_resource_cleanup.py tests/test_production_readiness_contract.py -q`

Expected: FAIL because metrics/evaluation/inventory are absent and legacy active paths remain.

- [ ] **Step 3: Implement bounded metrics, health, and deterministic evaluation**

Record candidate origin/kind, bounded category combination, terminal status, access mode, safe failure code, byte bucket, upload retry/checksum outcome, report-to-verified duration, report-to-available duration, and client action outcome. Never label by resource ID, filename, path, URL, hash, query, user, conversation, or device.

Health reports only configured storage, staging writability, event projector readiness, and aggregate recent failure state. The deterministic evaluator consumes recorded generic fixtures and produces counts for every required capability plus `release_invariants_pass`; it makes no provider, model, MCP server, or human-review call.

- [ ] **Step 4: Remove or isolate legacy and redundant active paths**

Run caller inventories before each removal:

```powershell
rg -n "_format_tool_result|artifact_ref|resource_link|attachments|tool_artifacts|render_resource_tool_result" app client_backend demo.py tests
```

Then:

- remove `_format_tool_result` and all client-runtime non-string JSON flattening;
- rename new public tool-result pointers to `tool_result_blob_id`, retaining legacy `artifact_ref` reads only;
- make the registry service the only active durable resource writer;
- remove duplicate path/resource extraction from renderers;
- remove new-message base64 persistence while retaining boundary conversion for old clients;
- keep tool traces separate from Outputs;
- remove Streamlit resource logic superseded by `app/ui/conversation_resources.py`; and
- delete dead helpers proven caller-free.

Document every retained compatibility path, its exact caller, why it remains, removal condition, and protecting test in `docs/conversation-resource-cleanup-inventory.md`.

- [ ] **Step 5: Run focused evaluation and cleanup gates**

Run: `python -m pytest tests/test_conversation_resource_metrics.py tests/test_conversation_resource_evaluation.py tests/test_conversation_resource_cleanup.py tests/test_production_readiness_contract.py -q`

Expected: PASS.

Run: `python scripts/evaluate_conversation_resources.py --cases eval/conversation_resources/cases.json --output output/audits/conversation-resources-eval.json`

Expected: exit `0`; every required capability has at least one passing case and `release_invariants_pass` is `true`.

- [ ] **Step 6: Run the feature suite**

```powershell
python -m pytest tests/test_conversation_resource_contracts.py tests/test_conversation_resource_models.py tests/test_conversation_resource_migration.py tests/test_conversation_resource_storage.py tests/test_conversation_resources_api.py tests/test_conversation_resource_config.py tests/test_conversation_resource_service.py tests/test_conversation_resource_projection.py tests/test_runtime_protocol_contract.py tests/test_client_runtime_store.py tests/test_client_tool_structured_results.py tests/test_tool_execution_resources.py tests/test_tool_result_rendering.py tests/test_conversation_resource_snapshot_flow.py tests/test_application_output_resources.py tests/test_server_mcp_output_resources.py tests/test_conversation_source_projection.py tests/test_message_service_resource_persistence.py tests/test_message_input_resources.py tests/test_ai_sdk_input_resource_contract.py tests/test_legacy_attachment_compatibility.py tests/test_resource_grounding.py tests/test_resource_link_rich_item.py tests/test_conversation_resource_events.py tests/test_internal_sse_resource_contract.py tests/test_message_resource_history.py tests/test_ai_sdk_resource_stream_contract.py tests/test_ai_sdk_context_window.py tests/test_ai_sdk_v6_stream_contract.py tests/test_demo_conversation_resources.py tests/test_demo_stream_rendering.py tests/test_demo_rich_response.py tests/test_conversation_resource_metrics.py tests/test_conversation_resource_evaluation.py tests/test_conversation_resource_cleanup.py tests/client_backend/test_resource_extractors.py tests/client_backend/test_resource_verifier.py tests/client_backend/test_resource_catalog.py tests/client_backend/test_runtime_bridge.py tests/client_backend/test_resource_snapshots.py tests/client_backend/test_resource_actions.py -q
```

Expected: PASS without a live provider or Desktop Commander installation.

- [ ] **Step 7: Run repository quality and migration gates**

Run: `python -m ruff check app client_backend tests scripts`

Expected: PASS.

Run: `python -m alembic heads`

Expected: exactly one head.

Run: `python -m pytest -q -m "not live_provider"`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add app/observability/conversation_resources.py app/evaluation/conversation_resources.py eval/conversation_resources/cases.json scripts/evaluate_conversation_resources.py docs/conversation-resource-runbook.md docs/conversation-resource-cleanup-inventory.md app/api/health.py app/core/config.py app/ai/client_runtime_tools.py app/ai/tool_result_rendering.py app/ai/tool_execution.py demo.py tests/test_conversation_resource_metrics.py tests/test_conversation_resource_evaluation.py tests/test_conversation_resource_cleanup.py tests/test_production_readiness_contract.py
git commit -m "chore: qualify conversation resources for production"
```

## Completion Evidence

Before marking the project complete, record these outputs in the runbook:

1. Alembic reports one head and upgrades a clean PostgreSQL test database.
2. The focused feature suite passes without live external services.
3. Ruff passes on all application, sidecar, test, and script files.
4. The non-live repository suite passes with zero failures.
5. The deterministic evaluation report has `release_invariants_pass=true`.
6. Repository scans find no public local paths, prose path scraping, structured-result flattening, or new `artifact_ref` writes.
7. AI SDK live/history payloads and Streamlit live/history state expose identical resource IDs, versions, categories, ordering, and availability.
8. Managed output links download authenticated content, while missing device outputs render an explicit unavailable state with no dead URL.
