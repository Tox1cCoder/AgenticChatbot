# Fork Conversation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** Phase 1 implemented on 2026-10-09 (commits `8761d3a6`..`b277bf4f` on `Thai-Postgre-FastAPI`). Sections 1-8 are the reviewed design as written before implementation; section 9 records what was built, the chosen fallback and the measured benchmarks. Document-backed forks (section 6) are not implemented.

**Goal:** Let a user start an independent conversation containing a snapshot of an existing conversation through a selected completed assistant reply.

**Architecture:** Copy the selected transcript prefix into ordinary message rows owned by a new conversation. Record lineage on the child conversation, normalize copied metadata, and rebuild compacted memory from the child transcript. Subsequent messages use the existing generation and streaming pipeline with fresh turn identities.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2, PostgreSQL/Alembic, LangGraph, Celery, local FastAPI sidecar, Streamlit demo.

## Global constraints

- The cutoff is inclusive and based on `Message.sequence`, never timestamp or UI pagination.
- Every copied message gets a new UUID; its source sequence and timestamps are preserved. Sequence gaps are permitted.
- Source conversation and messages remain unchanged; edits and soft deletion of either conversation do not mutate the other's transcript.
- The fork HTTP operation never invokes a model or executes source tools, approvals, widgets, task plans, or workspace operations. Its only background dispatch is a fresh child compaction notification after commit; the child compaction worker may subsequently invoke its configured summary model.
- All database writes for a fork commit in one transaction using one repository session.
- Async routes use async repository transport; synchronous database work cannot run directly on the event loop.
- Existing camelCase aliases, response envelopes, authentication, and both sidecar route prefixes remain supported.
- Phase 1 supports text, owner-authorized chat/web image references, and canvas source stored in message metadata. Document-backed forks require the separate document phase below.
- This is a conversation snapshot. Account/project memory, project instructions, custom-agent definitions, provider settings, and local files remain current shared resources.
- Draft assumption: fork from a selected completed assistant reply. Editing an earlier user prompt and resuming a paused execution in a fork are outside Phase 1.

## 1. Codebase findings

| Area | Existing implementation | Implication |
| --- | --- | --- |
| Conversation identity/settings | `app/models/conversation.py`; `app/schemas/conversation.py` | Already stores owner, persona, planning mode, project, and sequence allocator. Add nullable lineage/idempotency fields. |
| Transcript | `app/models/message.py`; `app/repositories/message.py` | Messages belong to one conversation; `(conversation_id, sequence)` is unique. Snapshot copies fit every existing reader. |
| API/service | `app/api/conversations.py`; `app/services/conversation_service.py`; `app/interfaces/conversation_service_interface.py` | Conversation CRUD exists; no fork endpoint or clone operation was found in the inspected application paths. |
| Persistence boundaries | `app/repositories/command_strategy.py`; `app/repositories/session_transport.py` | Generic writes commit per operation. Calling ordinary create methods in a loop cannot produce an atomic fork. |
| Prompt memory | `app/ai/history.py`; `app/repositories/conversation_compaction.py` | Agents read owned compacted memory plus unsummarized sequence-scoped messages. Source summaries can extend beyond the cutoff. |
| Compaction durability | `app/models/conversation_memory_summary.py`; `app/models/conversation_summary_job.py` | Summary and job cursors reference messages by conversation and sequence. The child needs its own summary/job lifecycle. |
| Execution identity | `app/ai/workflow/state.py`; `app/models/generation.py`; `app/services/message_service.py` | Checkpoints are per conversation and logical turn. Durable generations carry pause/continuation state. Do not copy them. |
| Concurrent turns | `app/services/conversation_turn_coordinator.py` | PostgreSQL advisory locks already serialize generation per conversation. Forks should use the same coordinator. |
| Canvas | `app/ai/canvas_state.py`; `app/ai/history.py` | Latest canvas is reconstructed from persisted message metadata. Copying the prefix naturally gives the child the cutoff revision. |
| Images | `app/models/chat_image.py`; `app/api/chat_images.py`; `app/api/web_images.py` | Chat images are deduplicated per owner/hash; media reads are owner-authorized. Selected web-image references can also be reused for the same owner. |
| Custom agents/project | `app/models/custom_agent.py`; `app/services/project_context_service.py`; `app/repositories/project.py` | Copy current conversation-agent attachments, keep the project association, and resolve capabilities on the requesting device next turn. |
| Documents/RAG | `app/models/document.py`; `app/repositories/document_chunk.py`; `app/services/rag_retrieval.py` | Authorization and vector filters are directly conversation-scoped. Copying document IDs into metadata does not give a child document access. |
| Client/UI | `client_backend/api/conversations.py`; `demo.py` | Add a sidecar proxy and a message action. The external production frontend is not present here. |
| Route injection | `app/core/dependency_injection.py` | `AppAutoInjector` uses an explicit service-type map. A container provider alone does not make a new annotated route parameter injectable. |
| UI reset | `demo.py::reset_conversation_state`; `demo.py::_clear_inflight_state` | The existing reset retains `generation_snapshot` and `live_widget_mounts`; fork navigation needs explicit cleanup in addition to that reset. |

Baseline verified on 2026-10-02:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_conversation_memory_hydration.py tests/test_canvas_agent.py tests/client_backend/test_conversation_routes.py -q
```

Result: **14 passed**. This validates existing foundations, not the proposed feature. Database migration and concurrency checks have not been run for this draft.

Product reference: Claude Code describes branching as creating a separate session with the conversation history, and distinguishes conversation branching from rewinding files. See [Manage sessions](https://code.claude.com/docs/en/sessions). The behavior below is tailored to this repository rather than a claim of exact product parity.

## 2. Approaches and recommendation

| Approach | Benefit | Cost/risk | Recommendation |
| --- | --- | --- | --- |
| Copy transcript prefix into a new conversation | Ordinary reads, search, pagination, canvas, and streaming continue to work; source edits cannot affect child history | O(prefix length) writes and duplicated JSON/text | **Use for Phase 1** |
| Share immutable messages through branch membership or ancestry | Avoids repeated transcript storage; enables a branch tree | Requires changing history, search, pagination, mutations, summary cursors, and deletion semantics | Revisit only if measured copy costs justify it |
| Clone a LangGraph checkpoint | Could preserve fine-grained internal execution state | Checkpoints can contain pending effects, device-scoped state, and resume handles; they are per-turn and subject to retention | Keep for a separately designed execution-fork feature |

Example: source `[U1, A1, U2, A2, U3, A3]`, fork at `A2` -> child `[U1', A1', U2', A2']`. Sending `U-new` appends to the child only. A fork can itself be forked using the same operation.

## 3. Proposed API and schema

```http
POST /conversations/{source_conversation_id}/fork
Content-Type: application/json

{
  "messageId": "selected-assistant-message-uuid",
  "requestId": "client-generated-uuid-for-this-fork-attempt",
  "title": "Explore an alternative"
}
```

`messageId` and `requestId` are required UUIDs. `title` is optional, length 1..255. When omitted, use the source title trimmed to 248 characters plus ` (fork)`; do not generate a title with an LLM.

Return `201` with `ApiResponse[ConversationRead]`, including the child's new ID and lineage. An identical retry returns the same child with `200`. A new `requestId` intentionally creates another fork. Persist a SHA-256 fingerprint of canonical JSON containing source ID, message ID, and the requested title (including null); do not hash the generated default title.

Proposed Pydantic request:

```python
class ConversationForkRequest(BaseModel):
    message_id: UUID
    request_id: UUID
    title: str | None = Field(default=None, min_length=1, max_length=255)
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
```

Add to `Conversation`:

| Column | Definition/purpose |
| --- | --- |
| `forked_from_conversation_id` | Nullable UUID FK to conversations, `ON DELETE SET NULL`; indexed for future child listing |
| `forked_from_message_id` | Nullable UUID FK to messages, `ON DELETE SET NULL`; index for FK maintenance |
| `forked_through_sequence` | Nullable positive BigInteger; records the inclusive cutoff even if lineage references later disappear |
| `fork_request_id` | Nullable UUID; unique together with `owner_id`, including soft-deleted children |
| `fork_request_fingerprint` | Nullable String(64); internal request-replay comparison |

Expose only the three lineage fields in `ConversationRead` and `ConversationInDB`. Update the service's explicit read-schema conversion; adding ORM columns alone will not expose them. Avoid new ORM parent/child relationships initially. The new conversation-to-message lineage FK introduces a second join path, so explicitly bind existing `Conversation.messages` and `Message.conversation` relationships to `Message.conversation_id`. Name the lineage FK constraints and account for the conversation/message FK cycle in test schema creation and teardown.

Error cases:

| Condition | HTTP/code |
| --- | --- |
| Source missing/deleted or unauthorized | Existing conversation 404/403 conventions |
| Cutoff missing/deleted/not in source | 404 `FORK_MESSAGE_NOT_FOUND` |
| User message, empty placeholder, or unfinished cutoff | 409 `FORK_POINT_NOT_COMPLETED` |
| Running source turn or bounded turn/row lock acquisition fails | 409 `conversation_turn_conflict` (preserve the existing lowercase workflow code) |
| Same owner/request ID with different fingerprint | 409 `FORK_REQUEST_CONFLICT` |
| Retry targets a child subsequently soft-deleted | 410 `FORK_RESULT_DELETED`; never resurrect it |
| Source has conversation-scoped documents | 409 `FORK_DOCUMENT_CONTEXT_UNSUPPORTED` in Phase 1 |
| Prefix exceeds Phase 1 bounds | 413 `FORK_HISTORY_TOO_LARGE` |

Phase 1 proposed bounds: 5,000 copied message rows and 32 MiB of combined UTF-8 content/serialized metadata. Make these settings (`conversation_fork_max_messages`, `conversation_fork_max_bytes`) rather than hard-coded policy; benchmark before release. Count/size the prefix before materializing all metadata.

## 4. Snapshot and lifecycle rules

1. Check for an existing owner/request ID first. Compare its fingerprint and return the owned live child on replay, even if the source has since disappeared. If the child is deleted, return 410.
2. Acquire the source's existing turn-coordinator lock with the existing bounded timeout. Recheck ownership and cutoff after acquisition. No global lock.
3. After taking the turn lock, open a fresh repository session and configure its connection for PostgreSQL `REPEATABLE READ` before its first query. The early replay lookup must not leave a pre-lock transaction/snapshot open. Use that one transaction for source settings, prefix, agents, asset validation, and writes. Take a source conversation row lock to coordinate its concurrent update/delete, with a transaction-local `lock_timeout` bounded by the existing turn-lock timeout. This provides one snapshot even though message CRUD does not use the turn lock. Handle serialization failures (`40001`) with a bounded whole-transaction retry, a fresh session/snapshot, and the same request ID; map exhausted retries and row-lock timeout (`55P03`) to the retriable 409 `conversation_turn_conflict`. Do not retry unrelated database failures.
4. Reject an active durable source generation, including an abandoned active row; let the existing reaper resolve it. An earlier completed reply may be forked when a later turn is paused for approval or continuation. The paused reply itself is ineligible.
5. For cutoff eligibility, locate a matching durable Generation through its assistant message ID, recognized generation metadata, or original user-message/logical-turn identity. Require `completed` or `completed_partial` with `continuation_available=False` and the selected message equal to that generation's terminal assistant message. This prevents accepting an earlier partial epoch just because its generation eventually finished. Inspect pause/interrupt/stopped metadata for legacy replies without a matching generation. Do not reject solely on `partial=True`: completed continuation chains retain it. A continuation-chain message with live handles must not be accepted as a legacy final reply. Copied rows carry a server-computed `fork_cutoff_eligible` boolean so nested forks can distinguish copied terminal replies from copied historical partials after runtime IDs have been removed.
6. Read every non-deleted source message at `sequence <= cutoff`, ordered by sequence. Exclude empty hidden pause/interrupt artifacts using the existing helper. Retain substantive historical partial text as inert history, even if it preceded a later completed reply.
7. Generate the entire old/new message UUID map before transforming metadata. Preserve source sequences and timestamps; set the child's `next_message_sequence = cutoff + 1`. New conversation timestamps are now.
8. Copy title/persona/planning-mode preference/project ID and current live owned custom-agent attachments. Do not seed project defaults again. Reset `plan_lifecycle` to null and create no TaskPlan rows: mutable current tasks are not a historical snapshot at the cutoff.
9. Deep-copy and normalize metadata as described below. Validate referenced assets against the source owner; no upstream fetches or byte duplication during fork.
10. Insert the child, message rows, agent attachments, and a fresh pending summary job through the cutoff in one transaction. Leave the child's memory summary absent; source summary payloads, jobs, leases, and checkpoints are never copied.
11. Commit once. Publish a compaction notification after commit; if publishing fails, the durable pending job remains eligible for the existing reconciliation sweep. Do not fail a successfully committed fork because notification delivery failed.
12. Return the child. Its next user message uses the existing request/generation pipeline and a fresh checkpoint thread built from the child conversation and new logical turn IDs.

Enforce idempotency with the database unique constraint as well as the early lookup. On uniqueness contention, rollback and read the winning child/fingerprint; do not catch an unrelated integrity error as a successful replay. Exercise this across workers, not only with a mock repository.

The turn coordinator raises `WorkflowRoutingException`, not an HTTP exception. Translate its typed `.error.code` at the new service/route boundary to `CustomHTTPException(status_code=409, ..., error_code="conversation_turn_conflict")`. The generic exception handler currently turns an untranslated coordinator exception into 500 `internal_server_error`. Test this through the registered FastAPI exception handlers and preserve the code through the sidecar.

### Metadata transformation

Create a focused policy in `app/services/conversation_fork_metadata.py`. It accepts `(content, metadata, message_id_map)` and returns transformed content plus fresh metadata; it must never mutate its input.

- Preserve display data such as model/provider attribution, citations, validated image references, reasoning summaries, canvas source/revision, and noninteractive rich items through an explicit allowlist based on the actual metadata builders.
- Remap `reply_to_user_message_id` using the UUID map; remove a link whose target was deleted/excluded. Remap nested message references only in recognized structures, never by blind string substitution.
- Remove runtime capabilities: interrupt/resume payloads, checkpoint/thread IDs, generation/continuation IDs, logical-turn IDs, execution epochs, approval identifiers, device session handles, pending tool calls, widget connection credentials, and task execution controls.
- Preserve historical partial/stopped attribution as inert display data. Strip the fields used to render Continue, Approve, Execute, or next-task actions. Record `fork_history=True` so renderers can explicitly suppress interaction in copied rows. The repository computes `fork_cutoff_eligible` from the source's durable lifecycle (or the preserved eligibility of an already copied row); the metadata helper must not infer completion from a display flag.
- Preserve agent identity for routing continuity; drop cached device capability/session provenance. Resolve agent capabilities on the next request's device.
- Keep `canvas_artifact` source and `canvas_update`; canvas lookup remains child-scoped. The `canvas:main` identifier is already conversation-local.
- For live widgets and other executable rich items, keep an inert historical label or static snapshot when available. Remove their rich item descriptors and their exact placement markers together; add display text at the removed marker so the transcript has no dangling markers. Apply the same rule to legacy `live_widgets` metadata.
- Retain owner-authorized chat-image IDs and live selected web-image IDs. Missing references render as unavailable historical media; they do not delete the surrounding text. Do not copy pending web-image references or their expiry jobs.
- Copy no feedback rows or usage records. Historical attribution may be displayed, but the child's usage dashboard counts only new usage events.

The allowlist must be inventoried against the metadata builder, rich-response contracts, and renderer before implementation; generic JSON copying is insufficient.

### Memory behavior

Rebuild memory from copied rows rather than copying the source's latest summary. This handles a source summary beyond the cutoff and avoids copying stale derived state after source edits. Child prompts continue to use existing history budgets while compaction runs; preserving the transcript does not mean placing all history into every model request.

Test the immediate first child turn before the worker finishes as well as after compaction. `ConversationHistoryProvider.build_context` applies message/token limits before model-request preflight; `RequestBudgetService.preflight` only forces compaction when its already-built request crosses the hard threshold. Enqueuing a summary job or relying on that threshold therefore does not ensure earlier copied context reaches the first turn. For a child with no valid summary whose copied prefix exceeds the applicable history budget, add a bounded preparation step on the next-generation path before routing/history assembly. Reuse the durable compaction lifecycle and avoid concurrent worker leases; define and test the behavior when preparation is disabled, fails, or times out (ordinary bounded history with an explicit limitation, or a retriable preparation response). The implementation must choose one fallback consistently. Keep model calls outside the fork transaction and the fork HTTP request. Do not advertise lossless model context: the application's normal bounded memory behavior still applies.

## 5. Implementation tasks

### Task 1: Define persisted lineage and the request contract

**Files:** modify `app/models/conversation.py`, `app/models/message.py`, `app/schemas/conversation.py`, `app/services/conversation_service.py`, `app/core/config.py`; create an Alembic revision under `app/alembic/versions/`; create `tests/test_conversation_fork_schema.py` and `tests/test_conversation_fork_migration.py`.

- [ ] Write schema tests for UUID validation, camelCase/snake_case population, optional title boundaries, and absent lineage on existing conversations.
- [ ] Add the request model and nullable model fields/constraints/indexes described above. Update explicit read-schema mapping and set `foreign_keys` on existing conversation/message ORM relationships. Test mapper configuration, ordinary message reads, and schema teardown after adding the reverse lineage FK.
- [ ] Determine the actual current migration head at implementation time; create one revision following it. Add positive-sequence and request-ID/fingerprint consistency checks that still allow lineage FKs to become null after hard deletion.
- [ ] Verify upgrade/downgrade and existing-row compatibility against a disposable PostgreSQL database. Check autogeneration drift; do not backfill transcript copies.
- [ ] Add the proposed limits to settings and document them in `.env.example`.

**Deliverable:** additive schema changes with tested API serialization and no changes to ordinary conversation creation.

### Task 2: Implement the metadata snapshot policy

**Files:** create `app/services/conversation_fork_metadata.py`, `tests/test_conversation_fork_metadata.py`; inspect `app/core/response_constants.py`, `app/core/rich_response.py`, `app/services/message_service.py`, `app/ui/rich_response.py`, `app/ui/hitl_recovery.py`, and `demo.py` for producers/consumers.

**Interface:** `snapshot_fork_message(*, content: str, metadata: dict[str, Any] | None, message_id_map: dict[str, str]) -> tuple[str, dict[str, Any]]`.

- [ ] Build fixture metadata containing a reply link, partial continuation, approval, live widget marker, static rich item, image reference, and canvas artifact.
- [ ] Assert mapped reply links, unchanged canvas/image/static content, no live capabilities, matching removal of interactive descriptors/markers, and unchanged source inputs.
- [ ] Implement the explicit preservation/remapping/removal policy. Add a regression case for completed continued replies whose `partial` field remains true.
- [ ] Add renderer guards for `fork_history` wherever historical rows could still offer controls; do not rely solely on stripping one known field.

**Deliverable:** inert copied history with working images/canvas and deterministic UUID remapping.

### Task 3: Build the atomic fork repository and service

**Files:** create `app/repositories/conversation_fork.py`, `app/services/conversation_fork_service.py`, `tests/test_conversation_fork_service.py`, `tests/integration/test_conversation_fork_postgres.py`; modify `app/core/container.py` and `app/core/dependency_injection.py` for service wiring; modify `app/repositories/conversation_compaction.py` only to expose an in-session job-enqueue helper if needed.

**Interfaces:** service `async fork_conversation(*, source_conversation_id: UUID, owner_id: UUID, request: ConversationForkRequest) -> tuple[ConversationRead, bool]`; the boolean is true only for a newly created child. Repository exposes `async afork_snapshot(...)` and returns a detached child plus the created/replayed flag.

- [ ] Write integration cases with a prefix at sequences 1..4 and later rows 5..6; assert only the prefix copies, UUIDs differ, reply links map, and the next sequence is 5.
- [ ] Test deleted-row gaps, substantive partial history, nested forks, owner isolation, cutoff validation, live generations, and later paused approvals. Batch-load lifecycle eligibility for the prefix and persist `fork_cutoff_eligible` on copied rows; test that nested forks still reject historical partial epochs and accept copied final continued replies.
- [ ] Implement the owner/request replay lookup, existing coordinator lock, repeatable-read repository transaction, eligibility checks, document guard, size guard, bulk copy, agent links, and fresh compaction job.
- [ ] Use `Session.flush()` inside the transaction rather than existing independently committing CRUD methods. Create the child before agent/message FK rows; create copied messages before the summary-job cursor FK.
- [ ] Add request-ID contention handling and tests for different fingerprints, replay after source deletion, and replay after child deletion.
- [ ] Inject a failure after message insertion; verify zero child/message/agent/job rows survive. Test concurrent source edits/deletion and fork/continue races using real PostgreSQL sessions.
- [ ] Assert `SHOW transaction_isolation` is `repeatable read` in the fork transaction, verify that replay lookup cannot retain a pre-lock snapshot, and test fresh-snapshot retry on `40001`. Hold a source row lock in another session and verify a bounded 409 rather than an indefinitely waiting fork request.
- [ ] Publish compaction after commit, with failure/reconciliation coverage. Register the new service/repository in DI with the existing async transport and turn coordinator.
- [ ] Register `ConversationForkService` in `AppAutoInjector.setup_wiring_map` as well as in the container, or use an explicit `Depends` provider. Test real route registration/OpenAPI generation and injected provider resolution rather than only calling the route function directly.

**Deliverable:** one authorized atomic snapshot with retry-safe identity and fresh lifecycle state.

### Task 4: Expose canonical and sidecar HTTP routes

**Files:** modify `app/api/conversations.py`, `app/core/dependency_injection.py`, `client_backend/api/conversations.py`; create `tests/test_conversation_fork_api.py`; extend `tests/client_backend/test_conversation_routes.py` and `tests/client_backend/test_proxy_path_params.py` as appropriate.

- [ ] Add an async canonical POST route using `ConversationForkService`; return the existing envelope with 201/200 based on the service flag.
- [ ] Map the explicit error cases to existing application exception conventions; explicitly translate the coordinator's `WorkflowRoutingException` into a typed HTTP 409 preserving `conversation_turn_conflict`. Assert the response envelope through `register_exception_handlers`, including row-lock timeout and exhausted serialization retries.
- [ ] Add the authenticated sidecar proxy route. Confirm both `/conversations/.../fork` and `/api/conversations/.../fork` aliases, encoded path forwarding, body forwarding, upstream status/error preservation, and local-session rejection.
- [ ] Test authorization, malformed bodies, replay responses, and unchanged ordinary conversation routes without real model/provider calls.

**Deliverable:** direct-server and local-sidecar clients can fork through the same contract.

### Task 5: Add the demo action and frontend handoff contract

**Files:** modify `demo.py`; create `app/ui/conversation_fork.py`, `tests/test_demo_conversation_fork.py`, `docs/frontend/fork-conversation.md`; update `Chatbot API.postman_collection.json` and relevant README API documentation.

- [ ] Put pure request/action-state helpers in the small UI module; avoid growing `demo.py` with repository or cloning logic.
- [ ] Add “Fork conversation” beside a persisted assistant reply. Hide it for user rows and clear pending/interrupt placeholders; server validation remains authoritative for completed continued replies and stale state.
- [ ] Generate one request ID per action; retain it across retries, disable duplicate submission while pending, and show the server's failure without changing the selected conversation.
- [ ] On success, call `upsert_conversation_in_state`, switch `current_conversation_id`, call `reset_conversation_state`, reload server messages, and refresh the conversation list. Do not slice the currently paginated UI transcript to manufacture child history.
- [ ] Use the existing reset for composer attachments, pending approvals, and message thumbnails, then explicitly clear/replace the source `generation_snapshot` and source widget mount state/subscriptions during fork navigation. `reset_conversation_state` does not currently clear them, and `_clear_inflight_state` intentionally retains the generation snapshot for same-conversation Continue. Keep that same-conversation behavior; use a fork/navigation-specific helper for the additional cleanup. Capture any source stream/generation identity before clearing state so cancellation still targets the source ID.
- [ ] Test successful navigation from a source with a later paused turn and mounted widgets: the child must offer no source Continue/Stop/Approve controls or widget connections. On a failed fork, the source's selection, snapshot, and widget state must remain usable.
- [ ] Show a lineage label linking to the source when available; when the source is deleted, leave the child usable and display a nonlinked historical label.
- [ ] Document the same navigation/cache behavior for the external frontend and AI SDK clients: fork once, load the returned ID's persisted messages, then send subsequent requests to that ID. No new SSE protocol is required.

**Deliverable:** a usable demo flow and a concrete integration contract for the production frontend.

### Task 6: Validate child context, isolation, and release behavior

**Files:** create `tests/test_conversation_fork_history.py`; inspect/modify `app/ai/history.py`, `app/ai/request_compaction.py`, `app/ai/request_budget.py`, and the next-generation entry path in `app/services/message_service.py` as required for memory preparation; extend targeted hydration/canvas/image/generation tests only where the fork introduces a new behavior; update the feature document with benchmark results.

- [ ] Verify a child prompt, its eventual summary, and latest canvas never contain source rows beyond the cutoff.
- [ ] Exercise a long copied prefix before and after compaction. Verify normal budget behavior and reliable preparation of derived memory, including a failed publisher recovered by the sweep.
- [ ] Put a distinctive fact before the copied prefix's history-limit boundary and keep the recent suffix below the preflight soft/hard thresholds. Verify first-turn preparation happens before routing/history trimming, and test disabled compaction, a busy lease, timeout, and summary-provider failure against the chosen fallback.
- [ ] Send a child turn through both standard streaming and AI SDK entry points with fake providers. Assert a fresh generation/thread and zero reused source approval/continuation handles.
- [ ] Update/delete source and child messages independently; verify history cache and summary invalidation stay conversation-scoped. Soft-delete the source and verify child text/media/canvas still load for the owner.
- [ ] Verify copied custom-agent attachments resolve against the requesting device and missing/deleted definitions degrade through existing rules.
- [ ] Benchmark prefixes of 100, 1,000, and 5,000 messages, including large metadata, without model calls. Record latency, peak memory, rows/bytes copied, and query count; validate the proposed limits against actual deployment targets.
- [ ] Run the feature suite plus relevant existing suites. Use the repository's disposable `TEST_DATABASE_URL` isolation rules for database/migration tests.

Suggested initial regression command after implementation:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_conversation_fork_schema.py tests/test_conversation_fork_migration.py tests/test_conversation_fork_metadata.py tests/test_conversation_fork_service.py tests/test_conversation_fork_api.py tests/test_conversation_fork_history.py tests/test_demo_conversation_fork.py tests/client_backend/test_conversation_routes.py tests/integration/test_conversation_fork_postgres.py tests/test_conversation_memory_hydration.py tests/test_canvas_agent.py tests/test_conversation_service_delete.py tests/test_conversation_search.py -q
```

**Release acceptance:** source transcript/settings remain untouched; child cutoff is exact; retries create one child; source actions cannot execute from copied history; child starts/responds independently; no future transcript/summary leakage; partial writes cannot survive failure.

## 6. Separate phase: document-backed forks

Phase 1 must reject sources with scoped documents rather than silently lose RAG context. Full document preservation is a second design/implementation cycle because it changes authorization and retention beyond conversation cloning.

Recommended direction: immutable document resources and explicit conversation-document attachments. Capture resource membership at fork creation, and authorize reads/retrieval through the destination's attachment set. Do not grant implicit access to every document in an ancestor conversation or all owner documents.

Affected areas include `app/models/document.py`, `app/repositories/document.py`, `app/repositories/document_chunk.py`, `app/repositories/document_image.py`, `app/services/document_service.py`, `app/services/rag_retrieval.py`, `app/ai/rag_tool_actions.py`, document HTTP routes, Qdrant filters, and cleanup/retention paths.

That phase must define document version capture, pending-upload behavior, detach versus resource deletion, filename collision rules, inherited-reference lifetime after source deletion, lexical/vector parity, and explicit document-ID authorization. The current schema has upload timestamps but no message-sequence attachment log, so it cannot truthfully reconstruct document membership at an arbitrary historical reply. Choose and expose either membership at fork creation or add durable attachment events for exact historical membership.

A physical copy of document rows/chunks/indexes is another option, but entails remapping citations/images and maintaining Qdrant entries; compare its cost against attachment-based resources before implementation. Remove the Phase 1 document guard only after the full authorization and retention contract is tested.

## 7. Review decisions and self-review

The draft assumes selected completed assistant replies; alternative user-message editing changes eligibility and submission flow, but can reuse the snapshot repository. Phase 1 deliberately resets executable plans and preserves current project/account context. Those semantics and the document guard should be reviewed before implementation.

Self-review completed: copy ordering uses sequence; IDs and reply links are distinct; child summary jobs have valid child cursors; no summary/checkpoint crosses the cutoff; per-operation commits are avoided; continued completed replies are not rejected by the `partial` flag; both server and sidecar/UI contracts are included; document and shared-resource limits are explicit. Limits and benchmark targets are proposals, not measured guarantees.

## 8. Codebase verification on 2026-10-09

**Scope:** plan review and foundation verification, not feature implementation. Searches of `app`, `client_backend`, and `demo.py` found no conversation-fork route, lineage fields, or snapshot implementation. The fork-specific files and tests listed above are still deliverables, not existing coverage. All implementation checkboxes remain unchecked.

The transcript-copy architecture matches the existing sequence allocator, conversation-scoped history/canvas reads, owner-authorized image reads, and summary-job cursor constraints. Source summaries/checkpoints cannot be copied safely, and the document guard is necessary with the current conversation-scoped document authorization. Reusing selected web-image references supports source **soft deletion** under current owner-only delivery checks; it does not promise independent upstream-image availability or survival of future media-retention changes.

Review findings incorporated above:

| Priority | Finding in current code | Required implementation adjustment |
| --- | --- | --- |
| High | `AppAutoInjector.setup_wiring_map` explicitly enumerates service types | Add the fork service mapping or an explicit dependency; include route-registration coverage. |
| High | `ConversationTurnCoordinator.hold` raises `WorkflowRoutingException`; the generic API handler returns 500 | Translate to typed 409 with the existing lowercase `conversation_turn_conflict` code. |
| High | `reset_conversation_state` retains the generation snapshot and widget mounts | Add fork-navigation cleanup while preserving ordinary same-conversation Continue. |
| High | History is capped before request-budget compaction thresholds are evaluated | Prepare missing child memory before routing/history assembly when the prefix exceeds history budgets; test the fallback. |
| Medium | Async repository transport does not set isolation or bound source row-lock waits | Set isolation before queries on a fresh post-lock session; add local lock timeout and whole-transaction retry coverage. |
| Medium | The original global background-task prohibition contradicted child-compaction dispatch | Distinguish prohibited source execution from permitted post-commit child memory rebuilding. |

Two isolated runtime probes reproduced the HTTP and UI findings: extracting and calling the actual reset helpers cleared `pending_interrupt` but retained seeded `generation_snapshot`/`live_widget_mounts`; a minimal FastAPI app using the actual exception handlers returned 500 `internal_server_error` for an untranslated `conversation_turn_conflict`. These are integration gaps in the proposed fork path, not claims that an existing fork feature has failed.

`alembic heads` reports one current head, **`9b132832d676`**. Recheck at implementation time. No migration upgrade/downgrade or database race/rollback check was performed during this review because no dedicated `TEST_DATABASE_URL` is configured.

Transaction implementation references: configure isolation before the first operation, as documented in [SQLAlchemy transaction isolation](https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#setting-isolation-for-individual-transactions); serialization failures require restarting the entire transaction with a new snapshot, as documented in [PostgreSQL Repeatable Read](https://www.postgresql.org/docs/current/transaction-iso.html#XACT-REPEATABLE-READ).

Verification commands and results are recorded below. Passing foundation tests do not establish that the proposed fork behavior works; release acceptance still requires the fork-specific suite, disposable PostgreSQL integration/migration tests, and measured prefix benchmarks.

Final combined foundation run (no live model/provider calls):

```powershell
$forkReviewTests = @(
    'tests/test_conversation_memory_hydration.py'
    'tests/test_canvas_agent.py'
    'tests/client_backend/test_conversation_routes.py'
    'tests/test_conversation_service_delete.py'
    'tests/test_conversation_search.py'
    'tests/test_workflow_concurrency.py'
    'tests/test_continuation_epoch_chain.py'
    'tests/test_conversation_compaction_repository.py'
    'tests/test_web_images_api.py'
    'tests/test_chat_images_api.py'
    'tests/test_demo_generation_controls.py'
    'tests/test_demo_stop_generation.py'
    'tests/test_demo_conversation_manager.py'
    'tests/test_demo_live_ui_state.py'
    'tests/test_demo_rich_response.py'
    'tests/test_rich_response_metadata.py'
    'tests/test_rich_response_contract.py'
    'tests/test_rich_placement.py'
    'tests/test_bot_metadata_image_externalization.py'
    'tests/test_message_history_pipeline.py'
    'tests/test_request_compaction_coordinator.py'
    'tests/test_context_window_message_metadata.py'
    'tests/test_custom_agent_definition_resolution.py'
    'tests/test_custom_agent_capability_resolver.py'
    'tests/test_widget_conversation_isolation.py'
    'tests/test_exception_handler.py'
)
.venv/Scripts/python.exe -m pytest @forkReviewTests -q -rs
.venv/Scripts/python.exe -m alembic heads
```

Result: **305 passed, 25 skipped in 20.00s**, exit code 0. All 25 skips were database-backed conversation-search cases requiring `TEST_DATABASE_URL`. Migration head: **`9b132832d676`**. Only this plan was edited during the review; the pre-existing modification to `plans/SANDBOX_FE_CONTRACT.md` was left untouched.

## 9. Implementation results (2026-10-09)

### First-turn memory: chosen fallback

**Fallback: ordinary bounded history with a structured warning.** A forked child's next turn never fails, and never returns a retriable error, because its memory could not be prepared.

`ForkMemoryPreparer` (`app/services/conversation_fork_memory.py`) runs from `MessageService._build_user_message_workflow_request`. Both the standard streaming route and the AI SDK route reach it, and so does the non-streamed `create_message`. It runs before the workflow request is built, so before routing and history assembly. It acts only when all of these hold:

- the conversation is a fork child (`forked_through_sequence` is set);
- this is the child's first turn: no live row follows the copied prefix except this turn's own user message. Later turns never wait or claim here; background compaction owns them, including after an edit invalidates the child's memory. Without this gate a failing provider would cost every turn, because each persisted reply re-queues the job with its attempts reset;
- the owner has no valid memory summary;
- the copied prefix exceeds the largest history budget among the answering agents (chat, RAG, search, planning; the router is excluded because it does not answer). With the default settings that is 24 messages / 9,000 tokens. Forks below it are trimmed by smaller-budget agents exactly like ordinary conversations.

When those hold, it runs the child's committed summary job through the ordinary worker (`run_conversation_compaction_now`, `force=True`) in a worker thread. That path uses the same lease, compare-and-set memory write and retry accounting as the Celery task, and it adds no second summary path. The run is bounded by the new setting `conversation_fork_memory_prepare_timeout_seconds` (default 15 s).

| Situation | Behavior |
| --- | --- |
| Compaction disabled | Fallback `compaction_disabled`; the job is not touched |
| Job lease held by a worker (or taken between check and claim) | Waits within the same timeout, polling every 0.25 s: `prepared` if the worker's memory appears; if the lease is released without memory, re-enters the ordinary path and claims it; otherwise fallback `timeout`. Never a second concurrent lease |
| Forced run finds no complete turn (short fork with long replies) | Quiet skip `no_complete_turn`; no warning |
| Job dead | Fallback `summary_job_dead`; the dead job is not revived |
| Job in retry with `available_at` in the future | Fallback `summary_job_backoff`; the backoff is honoured (claiming by id would ignore it) |
| Job idle without memory (a background run found the prefix below its own trigger) | Reopened through `request_backfill`, then run |
| Timeout | Fallback `timeout`. The thread keeps its lease and finishes inside the worker's own provider timeout, so a later turn has the summary |
| Provider or summary failure | Fallback `summary_failed` with the worker's code; the job follows its normal retry path |

Every fallback logs one warning naming the conversation id and reason. A prepared child gets the application's normal bounded memory: a summary plus recent rows. Its full transcript is not placed in the prompt.

While building this, the integration test found a pre-existing defect. `ConversationCompactor.select_compactable_prefix` grouped turns through a helper that reads `role`, but the worker passes persisted `Message` rows, which carry `sender`. Every durable compaction therefore ended `no_complete_turn` and wrote no memory. The compactor now groups through its own role reader, which reads `sender`, and a regression test covers it. As a result, background compaction now produces summaries, and calls its summary model, for every conversation that crosses the trigger, not only for forks.

### Benchmark

`scripts/benchmark_conversation_fork.py` forks through the real `ConversationForkService` and repository against `chatbot_test`. It makes no model calls and publishes no notification.

Test conditions:

- Each source is made of completed turns. Each assistant row carries about 9 KiB of metadata: a reasoning summary, ten web sources, suggestions and a tool artifact.
- Each size had one warm-up fork, then 3 timed forks.
- Peak memory comes from one extra fork run under `tracemalloc`.
- Host: Windows 11, Intel 32 logical CPUs, Python 3.13.7, local PostgreSQL 18.3.

| Messages | Source bytes | Latency median (min-max) | Peak Python alloc | Rows copied | Bytes copied | SQL statements per fork (executemany) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.5 MiB | 78 ms (77-79) | 1.3 MiB | 100 | 0.5 MiB | 16 (1) |
| 1,000 | 5.2 MiB | 215 ms (211-318) | 11.3 MiB | 1,000 | 5.2 MiB | 16 (1) |
| 5,000 | 25.9 MiB | 895 ms (879-942) | 56.7 MiB | 5,000 | 26.0 MiB | 16 (1) |

The statement count does not grow with prefix size: the messages are inserted with one `executemany`. Latency and memory grow linearly: about 0.17 ms and 11 KiB of Python allocation per copied row, roughly 2.2 times the bytes copied.

**Assessment of the defaults (5,000 rows / 32 MiB):** the defaults are acceptable for a single local API worker. A request at the limit holds one pooled connection and the source row lock for under 1 s, and allocates under 60 MiB transiently.

These are single-host numbers. They do not cover a remote database, concurrent forks or a container memory limit. Re-measure on the deployment target before raising either limit. Ten concurrent forks near the limit would allocate about 0.6 GB.

A 5,000-row prefix is also more than one summary pass can take. With the default `gemini-2.5-flash` input bound, the compactor's input selection keeps about 1,300 rows of such a prefix (measured locally: 1.5 s of token counting, no model call). The first summary of a near-limit fork therefore covers its earliest part. Later background runs fold in the rest incrementally.

### Verification

- `tests/test_conversation_fork_history.py` (27 unit tests, fakes only) covers:
  - when preparation runs and when it is skipped, including the first-turn gate;
  - each fallback reason with its warning;
  - the timeout returning promptly;
  - container wiring;
  - preparation running before the workflow through both the streaming entry point and the AI SDK entry point.
- `tests/integration/test_conversation_fork_history_postgres.py` (17 tests, real PostgreSQL, fake summary model) covers:
  - a fact at sequence 1 of a 40-row prefix reaching the first child turn's router context;
  - an idled job reopened;
  - normal memory plus budget after compaction;
  - a lost notification recovered by `dispatch_due_jobs`;
  - the four fallbacks through the real streaming path;
  - a hanging and a fast-failing provider each costing only the first turn: turn 2 makes no claim, no provider call and no wait;
  - a later turn with invalidated memory not running synchronous preparation;
  - no post-cutoff rows in the prompt, the summary input or the latest canvas;
  - a fresh generation and checkpoint thread for streaming and for AI SDK turns, with no source approval or continuation handle in the request or in copied rows;
  - edits and deletes, cache invalidation and summary invalidation all scoped to one conversation;
  - child text, media and canvas loading after the source is soft-deleted;
  - copied custom-agent attachments resolving on the requesting device, with deleted definitions dropped.
- Mutation checks:
  - Removing the preparation call fails the first-turn fact test.
  - Running preparation without `force` fails the fact test and the idled-job test.
  - Reverting the compactor fix fails the fact test.
  - Removing the first-turn gate fails both two-turn regressions, the invalidated-memory test and the unit gate test.
