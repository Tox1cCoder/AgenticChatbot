# Planning Stack Remediation Plan

## Summary

This plan reflects the current codebase after removing the legacy plan-persistence path and dead execution flags from the planning stack.

The system now uses canonical todo payloads as the primary representation for plan state. Manual task creation appends correctly, active-task lookup prefers `in_progress`, conversational todo sync no longer writes directly through the repository, and the lossy `Plan(Task(description))` persistence path has been removed.

The remaining work is focused on production hardening: explicit lifecycle state, database invariants, race safety, tests, and a few non-planning security/performance gaps that were uncovered during review.

## Scope

In scope:

- planning agent prompts, routing, todo state, and execution semantics
- task plan service, repository, schemas, and persistence invariants
- message-service planning integration and HITL resume consistency
- planning-related API semantics
- Streamlit planning UI alignment in `demo.py`
- targeted security/performance fixes discovered during review when they directly affect planning workflows

Out of scope:

- multi-plan support per conversation
- rich dependency graphs between tasks
- drag-and-drop task ordering UI
- replacing the single-string task description with a structured schema

## Current State

### Completed in the current branch

1. Manual task creation now appends instead of restarting at `task_order = 0`.
2. Active-task lookup now prefers `in_progress`, then `pending`.
3. The dead `auto_execute_plan` flag has been removed.
4. Streamed assistant responses no longer auto-complete tasks without an explicit planning action.
5. Conversational plan sync and API-driven plan sync now converge on canonical todos instead of the old lossy `plan` payload.
6. Direct repository writes from `MessageService` to task plans have been removed.
7. The legacy `TaskPlanFactory` and `Plan` / `Task` planning payload layer have been removed.
8. `SET_TODOS` no longer blindly resets the active index to zero; it now respects `in_progress`.
9. HITL resume now resyncs todo state when the resumed planning turn completes with updated todos.
10. The document task-status endpoint no longer returns raw Celery result payloads or tracebacks.
11. CORS configuration now respects configured origins and avoids wildcard-plus-credentials misconfiguration.

### Remaining Findings

#### R1. Plan lifecycle is still implicit instead of persisted

Evidence:

- conversation state still only stores `planning_mode_enabled`
- there is still no persisted distinction between draft review, ready-to-execute, executing, paused, and completed

Impact:

- review-before-execute remains convention-driven rather than state-driven
- routing and UI behavior still depend on prompt heuristics instead of explicit lifecycle state

#### R2. Database invariants are still incomplete

Evidence:

- `task_plans` still has only a non-unique index on `(conversation_id, task_order)`
- no migration exists to resequence duplicate `task_order` rows

Impact:

- concurrent writes can still create duplicate visible ordering
- correctness depends on service behavior rather than an enforced database invariant

#### R3. `create_task_plan` / `modify_task_plan` race safety is still missing

Evidence:

- `create_task_plan()` still checks for existing tasks before deciding whether to call `modify_task_plan()`
- the existence check and subsequent write are not protected by a database constraint or lock

Impact:

- two concurrent create requests can still both conclude that no plan exists and proceed independently

#### R4. Detailed task descriptions are still prompt-guided, not enforced

Evidence:

- the planning prompt asks for actionable tasks, but there is no validator or regeneration gate for terse output

Impact:

- task quality still depends too heavily on model behavior
- low-detail plans remain possible in production

#### R5. Test coverage is still effectively absent

Evidence:

- there is still no `tests/` directory in the repository
- `python -m py_compile` succeeds for touched files, but there is no regression suite

Impact:

- planning changes are still not release-safe
- concurrency, routing, HITL, and ordering regressions are likely to recur

#### R6. Document upload remains memory- and broker-heavy

Evidence:

- upload requests still read the full file into memory before validation
- the API still sends raw file bytes through Celery instead of handing off a staged file path or object-store reference

Impact:

- large uploads multiply memory pressure across the API process, Redis broker, and worker
- this is avoidable operational risk

#### R7. Task-status ownership is still only partially hardened

Evidence:

- `/documents/task/{task_id}` now requires authentication and returns sanitized output
- however, there is still no persisted mapping from Celery task ID to document ownership

Impact:

- authenticated users can no longer see tracebacks or raw internal payloads
- but the endpoint still cannot prove that a given task ID belongs to the caller

## Product Requirements

### Functional requirements

1. Creating a plan yields ordered tasks with detailed descriptions.
2. Modifying a plan preserves existing task identity and status unless the user explicitly changes them.
3. Manual task creation on an existing plan appends to the end.
4. Users can review and edit a plan before execution begins.
5. Explicit execution requests route deterministically to the planning agent when a plan exists.
6. Execution continues task-by-task until completion or a valid pause condition.
7. The active task is always the first `in_progress` task, otherwise the first `pending` task.
8. Task completion is driven only by explicit planning actions.
9. UI and API behavior are consistent.

### Non-functional requirements

1. Ordering and plan updates are transactionally safe.
2. Task IDs remain stable across ordinary plan edits.
3. Database constraints enforce core ordering invariants.
4. Observability exists for plan creation, mutation, execution, pause reasons, and sync errors.
5. New behavior is covered by unit, service, API, and HITL tests.
6. Large document uploads do not require duplicating full payloads through broker memory.

## Proposed Design

### 1. Persist explicit plan lifecycle state

Add a conversation-level plan lifecycle field with values such as:

- `draft`
- `ready`
- `executing`
- `paused`
- `completed`

Why:

- `planning_mode_enabled` is too coarse
- lifecycle should drive routing, UI, and execution entry
- review-before-execute should be encoded in state, not only in prompt wording

### 2. Enforce database ordering invariants

Add:

- a unique constraint on `(conversation_id, task_order)`
- a migration that resequences existing duplicates deterministically using `task_order`, `created_at`, then `id`

Why:

- the service layer should not be the only protection against duplicate ordering
- the current append/sync logic should be backed by hard database guarantees

### 3. Make create/modify race-safe

Implement one of:

- a database lock scoped to conversation plan writes
- an upsert-style plan-state row that serializes plan creation
- a uniqueness-backed retry strategy

Why:

- the current existence-check pattern is still vulnerable to concurrent creates

### 4. Add a task-description quality gate

Keep the single `description` field, but require each generated task to include:

- the action to perform
- the scope or target area
- the expected deliverable or verification signal

Implementation:

- strengthen planning prompts with concrete examples
- validate generated tasks for minimum detail
- regenerate or reject overly terse tasks

### 5. Finish ownership hardening for background document tasks

Add a persisted `processing_task_id` (or equivalent) to `Document`, then:

- store the Celery task ID at enqueue time
- resolve `/documents/task/{task_id}` back to a document
- enforce document ownership before returning status

Why:

- authentication alone is not enough for object-level authorization

### 6. Rework upload handoff for large files

Replace the current raw-bytes queue payload with:

- staged temp-file handoff on shared storage, or
- object-store upload plus worker-side retrieval

Why:

- this removes redundant copies of large files
- it reduces pressure on the API process and Redis

## Implementation Workstreams

### Workstream A: Lifecycle state and routing

Files:

- `app/models/conversation.py`
- `app/schemas/conversation.py`
- `app/ai/prompts.py`
- `app/ai/graph.py`
- `app/services/message_service.py`
- `demo.py`

Tasks:

1. Add persisted plan lifecycle state.
2. Route execution based on lifecycle state rather than prompt-only inference.
3. Reflect lifecycle state in planning status APIs and UI.

Acceptance criteria:

- execution entry is deterministic when a plan exists
- UI can distinguish draft, ready, executing, paused, and completed

### Workstream B: Database hardening

Files:

- `app/models/task_plan.py`
- `app/alembic/versions/...`
- `app/services/task_plan_service.py`

Tasks:

1. Add the unique constraint on `(conversation_id, task_order)`.
2. Write the deterministic backfill migration.
3. Add race-safe create semantics for plans.

Acceptance criteria:

- duplicate task orders cannot be inserted after migration
- concurrent create requests for the same conversation do not produce duplicate plans

### Workstream C: Task quality and execution polish

Files:

- `app/ai/agents/planning_agent.py`
- `app/ai/prompts.py`
- `app/ai/graph.py`

Tasks:

1. Add a task-description quality validator.
2. Regenerate or reject under-specified tasks.
3. Surface explicit pause reasons for clarification, approval, failure, and budget stops.

Acceptance criteria:

- generated tasks are consistently detailed
- execution stop reasons are visible and machine-readable

### Workstream D: Document task ownership and upload performance

Files:

- `app/models/document.py`
- `app/schemas/document.py`
- `app/repositories/document.py`
- `app/api/documents.py`
- `app/services/document_processing_service.py`
- `app/workers/document_processor.py`
- `app/alembic/versions/...`

Tasks:

1. Persist background task IDs on documents.
2. Enforce ownership on task-status lookup.
3. Replace raw-byte Celery payloads with staged file handoff.

Acceptance criteria:

- task-status lookups are ownership-safe
- uploads do not serialize full document payloads through Redis

### Workstream E: Test coverage

Files:

- `tests/ai/test_todo_actions.py`
- `tests/services/test_task_plan_service.py`
- `tests/services/test_message_service.py`
- `tests/api/test_task_plans.py`
- `tests/api/test_documents.py`

Tasks:

1. Add unit tests for todo state transitions.
2. Add service tests for append, diff sync, active-task semantics, and concurrent create protection.
3. Add API tests for planning status, manual append, and document task-status authorization.
4. Add HITL resume tests for planning-state consistency.

Acceptance criteria:

- planning regressions are covered at unit, service, and API levels
- document task-status authorization is covered by tests

## Release Gates

- migration succeeds on production-like data
- no duplicate `task_order` rows remain
- concurrent create is race-safe
- lifecycle state is reflected consistently in API and UI
- no task completes implicitly
- background task-status lookup is ownership-safe
- large uploads no longer queue raw file bytes through Redis
- regression tests pass

## Notes

- Keep the single-string task `description`; improve quality via validation rather than schema expansion.
- Preserve incremental todo-sync semantics; avoid reintroducing lossy plan wrappers.
- Do not reintroduce dead execution flags or duplicate persistence paths.
