# Planning Mode and Planning Agent Implementation Plan

## Summary

This plan covers the backend and Streamlit UI for planning mode, plan authoring, plan review, and plan execution. The target behavior is:

- users can create a plan, review it, and modify it before execution
- task descriptions stay in a single `description` field, but are substantially more detailed
- manual task creation on an existing plan appends to the end instead of restarting at `Task 1`
- execution continues through the remaining plan until completion unless blocked by an interrupt, missing information, or a hard safety limit
- the planning stack is production-ready: consistent state transitions, stable task identity, transactional persistence, test coverage, and observability

## Scope

In scope:

- planning agent prompts, routing, tool-driven todo state, and execution loop
- task plan service, repository, schemas, persistence, and migrations
- planning-related message service behavior
- planning-related API semantics
- Streamlit planning UI in `demo.py`
- automated tests, migration/backfill strategy, and rollout safeguards

Out of scope:

- multi-plan support per conversation
- rich dependency graphs between tasks
- drag-and-drop task reordering UI
- changing task descriptions into a fully structured multi-field schema

## Current Findings

### F1. Manual task creation on an existing plan restarts ordering at zero

Evidence:

- `TaskPlanService.create_task_plan_from_list()` always calls `TaskPlanFactory.create_from_descriptions()` without checking existing tasks (`app/services/task_plan_service.py`)
- `TaskPlanFactory.create_from_descriptions()` always assigns `task_order = idx`, starting at `0` (`app/factories/task_plan_factory.py`)

Impact:

- adding tasks through the manual API or Streamlit planning tab can create duplicate visible numbering such as a new `Task 1`
- ordering becomes ambiguous because the database only has a non-unique index on `(conversation_id, task_order)` (`app/models/task_plan.py`)

### F2. Current-task resolution ignores `in_progress` tasks

Evidence:

- `TaskPlanRepository.get_pending_tasks()` only returns `status == pending`
- `TaskPlanRepository.get_next_task()` simply returns the first pending task (`app/repositories/task_plan.py`)

Impact:

- once a task is started, status and UI can point at the wrong "current" task
- execution may skip over a started task and move to a later pending task

### F3. Tasks can be auto-completed without explicit execution

Evidence:

- `MessageService._handle_task_completion()` marks the current pending task completed after a bot response, as long as `plan_saved` is false (`app/services/message_service.py`)

Impact:

- ordinary conversation turns in planning mode can advance the plan even if the agent never called `complete_todo`
- this is incompatible with reliable execution and auditability

### F4. Execution routing is misaligned with the intended execution flow

Evidence:

- router guidance explicitly sends `"start the plan"`, `"work on task 1"`, and `"implement step 2"` to `chat_agent` (`app/ai/prompts.py`)
- `auto_execute_plan` is calculated in `MessageService` and passed to `_run_plan_execution_loop`, but `_run_plan_execution_loop` ignores it entirely — a single unconditional AI call is made regardless of its value (`app/services/message_service.py`)

Impact:

- execution requests can bypass the planning agent entirely
- `auto_execute_plan` is dead code: unused both at the router level and inside the execution loop
- "continue until the end" behavior is unreliable and depends on incidental routing

### F5. There are two inconsistent plan persistence paths

Path A:

- `TaskPlanService.create_task_plan()` and `modify_task_plan()` use `PlanningAgent.generate_plan()` / `modify_plan()`
- this path converts plan payloads into `Plan(tasks=[Task(description=...)])`
- `_persist_plan(..., replace_existing=True)` deletes old tasks and recreates all tasks (`app/services/task_plan_service.py`, `app/ai/schemas.py`)

Path B:

- conversational planning uses graph-managed `todos` and `_sync_todos_to_database()` for incremental sync (`app/services/message_service.py`, `app/ai/graph.py`)

Impact:

- plan modifications through the API can reset task IDs, statuses, and completion timestamps
- chat-based plan modifications use different semantics than API-based plan modifications
- execution references become unstable across modifications

### F6. Todo-to-database sync is heuristic, lossy, and not order-safe

Evidence:

- `_sync_todos_to_database()` directly writes repository records instead of using a domain operation
- new tasks are inserted with `task_order = i` (the loop index through the in-memory todo list, not `max(existing) + 1`)
- deletes depend on a hard-coded numeric-ID heuristic for `"1"` through `"10"`
- skipped and reopened states are not fully synchronized (`app/services/message_service.py`)
- new-task creation inside `_sync_todos_to_database` calls `self.task_plan_service.task_plan_repository.create()` directly, bypassing the service layer entirely and skipping access control, validation, and future service hooks

Impact:

- add/remove/reorder behavior can drift between in-memory and persisted state
- task identity and order are not guaranteed
- status transitions are incomplete
- direct repository access from `MessageService` breaks the single-service-layer principle and cannot be instrumented or validated centrally

### F7. Plan state and execution state are not modeled explicitly

Evidence:

- conversation state only has `planning_mode_enabled`
- there is no persisted distinction between draft review, ready-to-execute, executing, paused, and completed

Impact:

- the "review before execute" contract is enforced only by prompts, not by state
- routing and UI behavior cannot reliably reflect lifecycle state

### F8. Detailed task descriptions are requested but not enforced

Evidence:

- prompts ask for actionable tasks, but there is no validation or quality gate for terse one-line outputs (`app/ai/prompts.py`, `app/ai/agents/planning_agent.py`)
- `Task` stores only a plain `description` string (`app/ai/schemas.py`)

Impact:

- task detail quality varies by model output
- tasks can remain too short for production planning use

### F9. Persistence and validation hardening is missing

Evidence:

- no unique constraint on `(conversation_id, task_order)` (`app/models/task_plan.py`)
- replace/create operations commit row-by-row instead of transactionally
- manual create requests do not normalize or reject blank/whitespace-only items

Impact:

- partial writes are possible during failure
- duplicate order values are possible
- invalid task descriptions can enter storage through the API

### F11. `SET_TODOS` blindly resets the active task index to zero in memory

Evidence:

- `apply_write_todos_action()` sets `current_task_index = 0` unconditionally when `action == SET_TODOS`, regardless of whether any item in the new list already has `IN_PROGRESS` status (`app/ai/todo_actions.py`)

Impact:

- the in-memory execution pointer diverges from the persisted `in_progress` task whenever `SET_TODOS` is called mid-plan
- this is the in-memory counterpart of F2: even after F2 is fixed in the repository, the graph state will still point at index 0

### F12. HITL resume path does not sync plan state

Evidence:

- `resume_message_creation()` does not call `_sync_todos_to_database()` or `_handle_task_completion()` after the graph resumes (`app/services/message_service.py`)

Impact:

- if a task was `in_progress` when an interrupt fired, the task remains `in_progress` in the database after the user approves and the graph completes
- plan state visible via the API and UI becomes stale after any HITL round-trip

### F13. Critical planning paths silently swallow exceptions

Evidence:

- `_prepare_planning_context()` catches all exceptions with a bare `except Exception: pass` and returns an empty context (`app/services/message_service.py`)
- `_sync_todos_to_database()` has no error handling; any repository failure leaves state partially written with no log entry
- `_handle_task_completion()` catches all exceptions with a bare `except Exception: pass`

Impact:

- plan creation, task sync, and completion failures are invisible in logs
- partial writes cannot be diagnosed or alerted on
- the observability non-functional requirement cannot be satisfied without fixing these paths

### F14. `create_task_plan` / `modify_task_plan` entry-point race condition

Evidence:

- `create_task_plan()` checks for existing tasks, then calls `modify_task_plan()` if they exist, and vice versa (`app/services/task_plan_service.py`)
- the existence check and the subsequent create are not atomic

Impact:

- two concurrent requests arriving when no plan exists can both pass the existence check simultaneously and each enter `create_task_plan`, producing two independent plan rows for the same conversation
- this creates duplicate `task_order = 0` rows and violates plan identity

### F10. Test coverage is effectively absent

Evidence:

- there is no `tests/` suite in the repository
- `python -m pytest -q` finds no tests in the current workspace

Impact:

- regressions in ordering, routing, state sync, and execution are likely
- the planning stack is not release-safe

## Product Requirements

### Functional requirements

1. Creating a plan yields ordered tasks with detailed descriptions.
2. Modifying a plan preserves existing task identity and status unless the user explicitly changes them.
3. Manual task creation on an existing plan appends to the end of the plan.
4. Users can review and edit the plan before execution begins.
5. Explicit execution requests route to the planning agent.
6. Execution continues task-by-task until all tasks complete or a valid stop condition occurs.
7. The active task is always the first `in_progress` task, otherwise the first `pending` task.
8. Task completion is driven only by explicit planning actions, not by generic bot responses.
9. UI and API behavior are consistent.

### Non-functional requirements

1. Ordering and plan updates are transactionally safe.
2. Task IDs remain stable across ordinary plan edits.
3. Observability exists for plan creation, mutation, execution, pause reasons, and sync errors.
4. New behavior is covered by unit, service, API, and execution-loop tests.
5. Existing duplicate ordering data can be migrated safely.

## Proposed Design

## 1. Make `TaskPlanService` the single source of truth

Introduce explicit domain operations in `TaskPlanService`:

- `append_tasks(conversation_id, descriptions, user_id)`
- `apply_plan_diff(conversation_id, todos, user_id)`
- `replace_plan(conversation_id, todos, user_id)` for intentional full replacement only
- `get_active_or_next_task(conversation_id, user_id)` returning `in_progress` first, then `pending`

Rules:

- all persistence flows, including chat-agent sync, use service methods rather than direct repository writes
- plan edits preserve existing task rows where possible
- only explicitly removed tasks are deleted
- only explicitly changed tasks are updated

## 2. Add explicit plan lifecycle state

Add a persisted conversation-level plan lifecycle field, for example:

- `draft`
- `ready`
- `executing`
- `paused`
- `completed`

Why:

- `planning_mode_enabled` alone is too coarse
- the review-before-execute requirement should be modeled, not inferred from prompt text
- routing and UI can then behave deterministically

Suggested storage:

- add a new field on `Conversation` rather than creating a separate plan aggregate, because the current product model is one active plan per conversation

Trade-off note:

- adding the field directly to `Conversation` is the simpler option and acceptable given the one-plan-per-conversation constraint
- an alternative is a separate `ConversationPlanState` table (one row per conversation, created lazily), which keeps the migration blast radius smaller and avoids widening the `Conversation` model further; evaluate this if `Conversation` has already grown significantly in scope

## 3. Unify agent plan payloads around todos, not lossy `Plan(Task(description))`

Replace the current lossy plan-conversion path with a canonical payload that includes:

- `id`
- `description`
- `status`
- `order`

The task description remains a single string, but the persistence payload keeps identity and ordering intact.

Implications:

- `PlanningAgent.generate_plan()` and `modify_plan()` should return canonical todo payloads
- API-driven plan creation and chat-driven plan edits should call the same diff-apply code
- `TaskPlanService._persist_plan()` should stop delete-and-recreate behavior for ordinary modifications

## 4. Enforce detailed task descriptions with a quality gate

Keep a single `description` field, but require each task description to contain:

- the action to perform
- the scope or target area
- the expected deliverable or verification signal

Example target style:

- "Implement the planning-mode append flow in `TaskPlanService` and expose it through the manual task endpoint; verify by adding tasks to an existing plan and confirming numbering continues from the current last task."

Implementation approach:

- update planning prompts with stronger examples and explicit minimum detail guidance
- add a post-generation validator that rejects or regenerates tasks that are too short or generic
- ensure the Streamlit UI can display long descriptions cleanly

## 5. Fix execution flow and remove accidental state transitions

Execution should be driven exclusively by planning-agent tool actions:

- `start_todo`
- `complete_todo`
- `update_todo` for explicit status changes when needed

Required changes:

- remove or disable `MessageService._handle_task_completion()` auto-completion behavior
- route explicit execution intents to `planning_agent` when a plan exists
- remove or replace the `auto_execute_plan` flag: it is computed, passed to `_run_plan_execution_loop`, and then completely ignored there; replace it with the persisted plan lifecycle state from Design point 2 to drive execution routing deterministically
- make active-task selection prefer `in_progress` in both the repository layer (F2) and in-memory `apply_write_todos_action` (F11)
- enforce at most one active `in_progress` task unless concurrency is explicitly introduced later
- sync plan state after HITL resume: call `_sync_todos_to_database` (or an equivalent service method) at the end of `resume_message_creation` so the persisted task state reflects the completed graph turn

## 6. Harden ordering and persistence invariants

Database and repository changes:

- add a unique constraint on `(conversation_id, task_order)`
- add a migration/backfill that re-sequences existing duplicate orders deterministically
- make multi-row plan updates transactional
- centralize order normalization in one service method

Behavior rules:

- internal order stays zero-based
- UI displays `task_order + 1`
- append always uses `max(existing task_order) + 1`
- full replacement renormalizes order sequentially from zero

## 7. Align UI with the backend lifecycle

Streamlit changes in `demo.py`:

- when tasks already exist, manual entry should be labeled as append behavior rather than generic create behavior
- show plan lifecycle status: draft, ready, executing, paused, completed
- support long descriptions with either wrapped text or an expander
- refresh task list and planning summary using active-or-next task semantics

## Implementation Workstreams

## Workstream A: Data model and migration hardening

Files:

- `app/models/task_plan.py`
- `app/models/conversation.py`
- `app/schemas/conversation.py`
- `app/alembic/versions/...`

Tasks:

1. Add a unique database constraint for `(conversation_id, task_order)`.
2. Add a conversation-level plan lifecycle field.
3. Write a backfill migration that:
   - groups tasks by conversation
   - sorts by current `task_order`, then `created_at`, then `id`
   - rewrites sequential order values from zero
4. Update schemas and ORM models accordingly.

Acceptance criteria:

- duplicate task orders cannot be inserted after migration
- existing task data survives migration with deterministic ordering

## Workstream B: Service-layer canonical plan operations

Files:

- `app/services/task_plan_service.py`
- `app/repositories/task_plan.py`
- `app/factories/task_plan_factory.py`
- `app/interfaces/task_plan_service_interface.py`

Tasks:

1. Introduce append and diff-apply service methods.
2. Replace delete-and-recreate modification behavior for normal edits.
3. Make `create_task_plan_from_list()` append when tasks already exist.
4. Replace `get_next_task()` semantics with `get_active_or_next_task()`.
5. Ensure all multi-row writes use a transaction boundary.
6. Normalize and validate manual descriptions before persistence.
7. Remove the direct `task_plan_repository.create()` call inside `MessageService._sync_todos_to_database()`; replace it with a call to the new `append_tasks` service method so all new-task creation goes through a single, validated code path (fixes F6 repository bypass).
8. Make `create_task_plan` and `modify_task_plan` race-condition-safe: gate the existence-check-plus-create on a database-level lock or an upsert strategy so two concurrent requests cannot both enter the create path and produce duplicate plan rows (fixes F14).

Acceptance criteria:

- manual append on an existing plan yields new `task_order` values after the current max
- API-based modification preserves untouched task IDs and statuses
- `in_progress` task is returned as the current task
- no `MessageService` code path calls the task plan repository directly
- two simultaneous requests on a conversation with no plan produce exactly one set of plan rows

## Workstream C: Planning-agent payload and prompt unification

Files:

- `app/ai/agents/planning_agent.py`
- `app/ai/schemas.py`
- `app/ai/prompts.py`
- `app/ai/todo_actions.py`

Tasks:

1. Make direct planning-agent create/modify operations emit canonical todo payloads.
2. Remove the lossy `Plan(Task(description))` conversion from the primary update path.
3. Strengthen prompts so task descriptions are detailed by default.
4. Add a quality gate for too-short task descriptions.
5. Fix `apply_write_todos_action` so `SET_TODOS` scans the incoming list for an existing `IN_PROGRESS` item and sets `current_task_index` to that item's index; only fall back to index `0` if no `IN_PROGRESS` item is found (fixes F11).
6. Enforce one active task at a time for sequential execution.

Acceptance criteria:

- generated tasks are detailed and consistent
- direct API plan generation and chat-based planning produce the same persisted shape
- task IDs and ordering survive ordinary edits
- after `SET_TODOS` with an `IN_PROGRESS` item, `current_task_index` points at that item, not index `0`

## Workstream D: Execution flow correctness

Files:

- `app/ai/prompts.py`
- `app/ai/graph.py`
- `app/services/message_service.py`
- `app/ai/agents/router.py`

Tasks:

1. Update routing guidance so execution intents go to `planning_agent` when a plan exists.
2. Remove auto-completion from `MessageService`.
3. Remove the `auto_execute_plan` flag entirely from `MessageService`; replace all routing and loop-control logic that depends on it with reads of the persisted plan lifecycle state introduced in Workstream A.
4. Ensure the execution loop can continue until all tasks complete or a valid pause occurs.
5. Make pause reasons explicit and user-visible when execution stops for budget, approval, clarification, or failure.
6. Add plan-state sync to `resume_message_creation`: after the graph successfully resumes and completes, call the same todo-sync and task-status logic used in the normal send path so that post-HITL state is consistent (fixes F12).
7. Replace bare `except Exception: pass` blocks in `_prepare_planning_context`, `_sync_todos_to_database`, and `_handle_task_completion` with `logging.warning(...)` calls that record the exception type and conversation ID (fixes F13).

Acceptance criteria:

- execution requests consistently hit the planning agent
- no task completes without an explicit planning action
- the plan can continue through all remaining tasks in a single run, subject to configured safety limits
- plan and task state are consistent after a HITL approve/reject/edit cycle
- no planning exception is silently discarded; all failures appear in logs with conversation context

## Workstream E: UI alignment

Files:

- `demo.py`

Tasks:

1. Rename manual task creation affordances to communicate append behavior when a plan already exists.
2. Display richer task descriptions cleanly.
3. Reflect lifecycle state and active task correctly in the planning tab and chat header.
4. Ensure task numbering always derives from persisted `taskOrder`, not creation order in the view.

Acceptance criteria:

- appending tasks through the UI never creates duplicate visible numbering
- detailed descriptions remain readable
- planning status reflects the active task correctly

## Test Plan

Add a new `tests/` suite covering:

### Unit tests

- `tests/ai/test_todo_actions.py`
  - `set_todos` with an `IN_PROGRESS` item sets `current_task_index` to that item, not `0`
  - `set_todos` with no `IN_PROGRESS` item sets `current_task_index` to `0`
  - `start_todo` enforces a single active task
  - `add_todo` preserves append ordering

- `tests/services/test_task_plan_service.py`
  - manual append on existing plan yields `task_order` values continuing from the current max
  - diff-apply preserves task IDs/statuses
  - active-or-next task prefers `in_progress`
  - blank descriptions are rejected
  - concurrent `create_task_plan` calls produce exactly one plan (race condition gate)

- `tests/services/test_message_service.py`
  - ordinary bot responses do not auto-complete tasks
  - todo sync uses service methods instead of direct repository writes (no direct repository call from `MessageService`)
  - planning exceptions are logged, not silently swallowed

### API tests

- `tests/api/test_task_plans.py`
  - `POST /conversations/{id}/task-plans/manual` appends to existing plans
  - `GET /conversations/{id}/planning-status` returns the active `in_progress` task
  - plan modification preserves stable task identity where appropriate

### Execution-loop and HITL tests

- explicit `"start the plan"` routes to `planning_agent`
- execution continues until all tasks are completed
- pause/interrupt conditions are surfaced without corrupting task state
- removal and addition of tasks during conversational planning produce correct persisted order
- after a HITL approve cycle, task status in the database matches the graph's final state
- after a HITL reject cycle, the interrupted task is not erroneously marked complete

## Rollout Plan

1. Implement migration and backfill first. Prepare a down-migration (rollback script) before deploying so the schema change can be reverted if needed. Test the migration against a copy of production-like data before any deploy.
2. Ship service-layer append and active-task fixes behind the updated tests.
3. Unify planning-agent persistence paths; remove the direct repository call from `_sync_todos_to_database`.
4. Remove auto-completion, remove the `auto_execute_plan` dead flag, and fix execution routing.
5. Add HITL resume plan-sync and fix exception logging.
6. Update Streamlit copy and rendering.
7. Add logging for:
   - plan creation/modification source
   - append operations
   - execution start/stop
   - pause reasons
   - sync failures
   - any caught exception in planning paths (with conversation ID)
8. Perform a manual regression pass on:
   - new plan creation
   - plan review and edit
   - manual append on existing plan
   - chat-based append/remove/update
   - execute-until-complete flow
   - interrupted execution and resume
   - HITL approve/reject/edit cycle followed by plan status check

## Release Gates

- all new tests pass
- migration succeeds on a copy of production-like data and the down-migration cleanly reverts it
- no duplicate `task_order` rows remain
- manual append and chat-based append produce identical persisted order semantics
- execution no longer completes tasks implicitly
- no `MessageService` code path calls the task plan repository directly
- plan and task state are consistent after a HITL round-trip
- no planning exception is silently discarded in production logs

## Recommended Implementation Sequence

1. Fix ordering invariants, add the unique constraint migration, and write the down-migration.
2. Introduce canonical service methods for append, diff-apply, and race-safe create.
3. Update manual/API flows to use the new service methods; remove the direct repository call from `_sync_todos_to_database`.
4. Remove auto-completion, remove the dead `auto_execute_plan` flag, and fix active-task lookup in both the repository and `apply_write_todos_action`.
5. Add HITL resume plan-sync and replace all bare `except Exception: pass` blocks with logged warnings.
6. Align router and execution behavior with plan lifecycle state.
7. Improve prompt quality and UI rendering for detailed task descriptions.
8. Add regression coverage for all planning paths, including HITL sync and concurrent-create scenarios.

## Notes for Implementation

- preserve the single-string `description` field, but allow longer sentence-style descriptions
- prefer incremental updates over full plan replacement to keep IDs and status stable
- only use full replacement for explicit user intent such as "replace the whole plan"
- if a direct full replacement still exists, make it transactional and clearly separate it from append/diff operations
