# Generation Guardrails and Cross-Client Continue/Stop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert model/tool-loop exhaustion into a validated partial answer with a durable Continue/Stop decision, and make Stop and Continue behave identically through internal SSE, AI SDK v6, the client-backend proxy, and `demo.py`.

**Architecture:** PostgreSQL owns a versioned generation lifecycle and opaque continuation identity. A per-epoch soft budget forces one tool-free synthesis; the validated result pauses at a LangGraph control interrupt. Redis broadcasts Stop to the owning worker while a process-local registry only accelerates cancellation. Both public stream adapters project the same canonical lifecycle events. Continue resumes the exact checkpoint without routing or adding a user message; Stop either cancels active work or accepts a continuable partial and enters the universal finalizer.

**Tech Stack:** Python 3.10+, FastAPI, SQLAlchemy/PostgreSQL, Alembic, Redis, LangGraph interrupts/checkpoints, LangChain middleware, AI SDK v6 UI Message Stream SSE, Streamlit, pytest/pytest-asyncio.

## Global Constraints

- The production routing refactor remains authoritative: one route per new turn, no route on resume, and only `finalize -> END` terminates the graph.
- `generation_id` controls lifecycle; message IDs never substitute for it.
- `generation_id` and `logical_turn_id` survive Continue; `execution_epoch` increments exactly once for each accepted Continue.
- Persist and validate partial assistant content before emitting it or offering Continue.
- A soft limit reserves exactly one model call and exposes no tools on that call.
- A hard limit is normalized into the same safe partial/continuable path, not a public generic execution error.
- Do not automatically continue. HTTP reconnect/resume is transport recovery, not semantic Continue.
- Mutations with `outcome_unknown` block Continue until reconciled.
- Stop is idempotent. A wait timeout returns `stop_requested`; it never falsely reports `stopped` or discards the owner task.
- Internal SSE and AI SDK consume the same service methods and lifecycle record.
- Keep tool-approval interrupts distinct from `execution_budget_exhausted` control interrupts.

## Design Revisions Required Before Implementation (2026-09-07)

`output/audits/2026-09-07-three-plan-review.md` and its recheck found five
contracts this plan asserts but does not specify. Each is confirmed against the
current code at `73b695f`; the file:line evidence is quoted so an implementer
can re-check rather than trust this list.

### R1 (P1). Middleware alone does not reach RAG, Planning, or delegated workers

Task 3 puts the soft budget in an `AgentMiddleware` and Task 4 reads
`execution_budget` off the specialist outcome. That covers exactly the agents
built through `create_agent`.

- `SpecialistFactory.invoke_specialist_subgraph` short-circuits `rag_agent` to
  `_invoke_rag_specialist`, which runs the shared compiled RAG graph rather
  than a `create_agent` subgraph (`app/ai/graph.py:1779`). No `AgentMiddleware`
  in the specialist stack ever executes for it.
- Planning runs its own parent-graph nodes
  (`app/ai/workflow/planning_execution.py`), and delegated workers are
  dispatched from there.

**Required contract.** Make the budget a *state* contract rather than a
middleware artefact:

- `ExecutionBudgetState` lives on `WorkflowState`, and one shared, framework-free
  accountant (`app/ai/workflow/execution_budget.py`) owns increment, threshold
  and forced-synthesis decisions. `SoftExecutionBudgetMiddleware` becomes a thin
  adapter that calls it for `create_agent` specialists.
- The RAG graph's own model/tool loop calls the same accountant at its own
  model and tool boundaries and produces the same `exhausted_by` value.
- Planning increments the same accountant per worker invocation, and a
  delegated worker that exhausts its epoch returns a validated partial to its
  parent rather than pausing the parent turn. Only the top-level turn pauses.

**Required tests.** The forced-synthesis and hard-limit assertions in Task 3
Step 1 must be parameterized over all three execution paths — specialist, RAG,
and a delegated Planning worker — not written once against the middleware.

### R2 (P1). Continue must rehydrate the evidence, not just reset the counters

Task 4 Step 3's resume `Command` sets `execution_epoch + 1`,
`execution_budget=None`, `execution_phase="executing"` and goes to the
specialist node. Nothing there restores what epoch 1 collected.

- The next invocation's messages are assembled as
  `[*request.history, *request.messages]`
  (`app/ai/workflow/specialists.py:_invocation_messages`), where `messages` is
  the current-turn slice from graph state.
- Everything the previous epoch produced — the assistant turns and the
  `ToolMessage`s carrying the evidence — is sliced off into the outcome
  (`_produced_messages`) and stored in `outcome.provenance.private_messages`.
  It never re-enters the request.

Left as written, epoch 2 sees the original question and no evidence. It will
re-run the work epoch 1 already paid for, which is the opposite of what
Continue is for, and it will do so with a budget that has just been reset.

**Required contract.**

- Define where the carried transcript lives and who writes it. Add
  `carried_messages: list[BaseMessage]` (or a serialized equivalent) to
  `WorkflowState`, written by the continuation pause node from
  `outcome.provenance.private_messages` before it interrupts.
- `_invocation_messages` becomes
  `[*history, *carried_messages, *current_turn_messages]`, with carried
  messages placed so tool-call/tool-result pairs stay adjacent and valid for
  every provider. An orphaned `ToolMessage` is a provider error, not a
  degraded prompt — assert pairing explicitly.
- Specify the bound. The carried transcript is capped by the same offload rules
  as any other context, and a carried tool result that was offloaded stays a
  `blob_id` reference rather than being rehydrated inline.
- Specify cleanup. The validated partial assistant message persisted at the
  pause is not re-appended by epoch 2, and the reserved assistant message ID
  for the continued answer is allocated at `prepare_continue`, not derived from
  the paused one.

**Required test.** Epoch 2 receives epoch 1's evidence and does not repeat a
completed tool call. Assert on the messages actually handed to the model, not
on the checkpoint — the earlier runtime probe passed the checkpoint assertion
while sending the model only the original human question.

### R3 (P2). Suppress tools at the request boundary, not in the middleware

Task 3 Step 4 has the middleware make "the next model request with `tools=[]`".
That does not survive the stack: `ToolExecutionMiddleware.awrap_model_call` is
the innermost model-call wrapper in `build_specialist_middleware`
(`app/ai/workflow/middleware.py:709-737` — it is appended after
`UsageRecordingMiddleware`, and first-in-list is outermost), and it
unconditionally re-offers the live factory's tools:

```python
tools = list(await self._tool_factory() or [])
self._scope.offer(tools)
return await handler(request.override(tools=tools))
```

Any earlier `tools=[]` is overwritten. Appending the budget middleware after
`tool_execution` instead does not work either: `ToolExecutionMiddleware.awrap_tool_call`
never calls `handler` for an ordinary tool, so an inner `awrap_tool_call` would
never run.

**Required contract.** Do not invent a new flag. The repository already has this
exact seam and it works one level up, at the request:

- `invocation_kwargs` carries `disable_tools`, which `_specialist_request_for`
  pops into `request.extras` (`app/ai/graph.py:1714-1726`), and every agent's
  `tool_factory` returns `[]` when it is set (`app/ai/agents/chat_agent.py:360`,
  and the same three lines in `canvas_agent`, `custom_agent`,
  `image_generator_agent`, `rag_agent`). Because suppression happens *inside*
  the factory, the innermost re-offer yields nothing and every provider retry
  and fallback re-reads it.
- The existing tool-budget path already does this
  (`app/ai/graph.py:1193` sets `disable_tools` plus a `tool_budget_notice`), so
  forced synthesis should extend that mechanism rather than compete with it.
  Reconcile the two notices into one instruction; two systems appending
  different "stop calling tools" sentences to the same call is a prompt
  conflict.
- The budget middleware keeps only `awrap_tool_call` — pairing the synthetic
  `ToolMessage` for the call it refuses — and must therefore stay *before*
  `tool_execution` in the list.

**Required test.** Assert the model's final call is made with no tools **after
the full assembled stack has run**, not by inspecting the request the budget
middleware returned. Include one provider-fallback retry in the scenario.

### R4 (P2). Research accounting is process-local and turn-scoped

Task 3 Step 1 requires cumulative turn quota metadata to survive an epoch
change. It cannot today:

- `app/ai/research_budget.py` keeps `_budgets: OrderedDict[str, ResearchBudget]`
  at module scope under a `threading.Lock`, keyed by conversation id alone
  (`_key`). Nothing is persisted.
- A new turn resets it (`app/ai/graph.py:646`).

So a Continue served by another worker sees no accounting at all, and a Continue
served by this one shares a single conversation-keyed budget with any other
turn in flight. The plan also releases the conversation lock while continuable,
which permits exactly that overlap.

**Required contract.**

- Key research accounting by `logical_turn_id`, not conversation id, and persist
  it on the generation row so any worker can rehydrate it.
- State which quotas Continue replenishes and which it does not. The default
  should be: per-epoch call caps reset, cross-epoch deduplication (the
  "same query already made this turn" guard) does **not** — otherwise Continue
  becomes a way to re-run the identical search the budget just refused.
- Define the behavior when rehydration fails: fail the Continue rather than
  proceeding with an empty budget, since an empty budget is indistinguishable
  from a fresh turn and silently grants a full new quota.

### R5 (P2). One last-command slot cannot fence a delayed retry

Task 1 Step 3 stores "last command idempotency key/action/result JSON" — a
single slot — and `StopGenerationRequest` (Task 6) carries no version or epoch.

Sequence: Stop `S` completes against epoch 0; Continue `C` is accepted and
overwrites the slot; a delayed retry of `S` arrives. Its key no longer matches
anything, so it is treated as new and can cancel epoch 1.

**Required contract.**

- Add `execution_epoch` (or the lifecycle `version`) to `StopGenerationRequest`
  and `ContinueGenerationRequest`, and reject a command whose fence is older
  than the row's current value with a distinguishable
  `stale_command` result rather than executing it.
- Keep a durable command ledger — one row per `(generation_id,
  idempotency_key)` recording action, fence and result — written in the same
  transaction as the transition it authorizes. A replay returns the recorded
  result; the single-slot design cannot.
- Add "delayed old-command replay after a later Continue" to the Task 8 race
  suite, alongside the existing Continue/Continue and Stop/Stop races.

### Execution order after these revisions

Tasks 1 and 2 are unaffected by R1-R4 and can proceed as written once R5's
ledger is folded into Task 1's schema. R1-R3 change Task 3 and Task 4
materially; do not start them from the current text.

---

## File Structure

- Create `app/models/generation.py`: generation lifecycle row and stable enums.
- Create `app/repositories/generation.py`: owner-scoped compare-and-set transitions.
- Create `app/services/generation_control_service.py`: lifecycle invariants and idempotent commands.
- Create `app/services/generation_control_bus.py`: Redis Stop publication/subscription with in-memory test implementation.
- Create `app/schemas/generation.py`: API commands and canonical snapshots.
- Create `app/alembic/versions/d0e1f2a3b4c5_add_generation_controls.py`.
- Modify `app/models/__init__.py`, `app/core/container.py`, and `app/core/config.py`.
- Create `app/ai/workflow/execution_budget.py`: per-epoch soft/hard enforcement.
- Create `app/ai/workflow/continuation.py`: typed pause payload and resume node.
- Modify workflow state/contracts, specialist construction, finalization, graph topology, and graph streaming.
- Modify `app/services/generation_registry.py`: key active tasks by generation ID.
- Modify `app/services/message_service.py` and `app/interfaces/message_service_interface.py`: orchestrate start, pause, Continue, Stop, and persistence.
- Modify canonical stream events and both adapters.
- Modify server APIs, client-backend proxies/client, and `demo.py`.
- Add unit, PostgreSQL integration, stream-contract, API, and UI state-machine tests.

### Task 1: Persist the Authoritative Generation Lifecycle

> **Landed 2026-09-07.** Verified against a real PostgreSQL (`chatbot_test`):
> 20 integration tests, the 50-test alembic chain applying and round-tripping
> the migration from empty, and `alembic check` reporting no new drift — only
> the two pre-existing `document_chunks` items. R5's command ledger replaced
> the single last-command column. Head is now `d0e1f2a3b4c5`; `_HEAD` in
> `tests/test_alembic_full_chain_postgres.py` and the README head reference
> were updated with it.
>
> ~~The migration is **not applied to the live `chatbot` database**~~ —
> **superseded 2026-09-08.** `alembic current` reports the live `chatbot`
> database at `d0e1f2a3b4c5`, so it had already been applied, presumably by an
> app startup. It is now at `e1f2a3b4c5d6`, applied with Thai's approval; see
> the Task 5 note for why that follow-up was needed.

**Files:**
- Create: `app/models/generation.py` (lifecycle row + command ledger)
- Create: `app/repositories/generation.py`
- Create: `app/schemas/generation.py`
- Create: `app/alembic/versions/d0e1f2a3b4c5_add_generation_controls.py`
- Modify: `app/models/__init__.py`
- Create: `tests/test_generation_repository.py`
- Create: `tests/integration/test_generation_repository_postgres.py`
- Modify: `tests/test_alembic_full_chain_postgres.py`

**Interfaces:**

```python
class GenerationStatus(str, enum.Enum):
    STARTING = "starting"
    RUNNING = "running"
    FINALIZING_AFTER_LIMIT = "finalizing_after_limit"
    CONTINUABLE = "continuable"
    CONTINUING = "continuing"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"
    COMPLETED = "completed"
    COMPLETED_PARTIAL = "completed_partial"
    FAILED = "failed"


class GenerationSnapshot(BaseModel):
    generation_id: UUID
    logical_turn_id: str
    conversation_id: UUID
    status: GenerationStatus
    version: int
    execution_epoch: int
    continuation_id: UUID | None
    continuation_available: bool
    continuation_block_reason: str | None
    assistant_message_id: UUID | None
    terminal_reason: str | None
```

- [x] **Step 1: Write failing model/repository tests**

Test owner-scoped lookup, unique `(logical_turn_id)`, version increments, and legal compare-and-set transitions. Two concurrent Continue attempts with the same expected version/idempotency key must produce one epoch increment. A different idempotency key against the consumed continuation must return the current snapshot, not increment again.

- [x] **Step 2: Run and verify import failures**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_generation_repository.py tests/integration/test_generation_repository_postgres.py
```

- [x] **Step 3: Define the row and indexes**

Use `generations` with UUID PK, owner/conversation FKs, unique logical turn, indexed checkpoint thread, status enum, version, epoch, active agent, budget JSON, research accounting JSON (R4), assistant message FK, continuation UUID, continuation availability/block reason, terminal reason, and timezone-aware lifecycle timestamps.

**Revised per R5:** the single "last command" slot is replaced by a
`generation_commands` table — `(generation_id, idempotency_key)` unique, with
action, the epoch/version fence the command was issued against, and the result
JSON. One row per command, written in the same transaction as the transition it
authorizes, so a delayed replay after a later Continue returns its own recorded
result instead of executing against the wrong epoch. Add a unique partial index allowing one active lifecycle (`starting`, `running`, `finalizing_after_limit`, `continuing`, `stop_requested`) per conversation.

The migration revision is `d0e1f2a3b4c5` with `down_revision = "c9d0e1f2a3b4"`. Before executing this task, run `alembic heads`; if another migration has landed, create a merge revision first rather than silently editing `down_revision` into a fork.

- [x] **Step 4: Implement repository compare-and-set methods**

```python
class GenerationRepository(RepositorySessionMixin):
    async def acreate(self, command: CreateGeneration) -> GenerationSnapshot: ...
    async def aget_owned(
        self, generation_id: UUID, user_id: UUID, conversation_id: UUID
    ) -> GenerationSnapshot | None: ...
    async def atransition(
        self,
        *,
        generation_id: UUID,
        user_id: UUID,
        conversation_id: UUID,
        expected_statuses: tuple[GenerationStatus, ...],
        expected_version: int,
        values: dict[str, Any],
    ) -> GenerationSnapshot | None: ...
    async def aclaim_command(
        self,
        *,
        generation_id: UUID,
        idempotency_key: str,
        action: str,
        fence: int,
    ) -> CommandClaim: ...
    async def arecord_command_result(
        self, *, generation_id: UUID, idempotency_key: str, result: dict[str, Any]
    ) -> None: ...
```

Every mutation is one `UPDATE ... WHERE id/user/conversation/status/version ... RETURNING`, never read-then-write.

- [x] **Step 5: Run migration and persistence verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_generation_repository.py tests/integration/test_generation_repository_postgres.py tests/test_alembic_full_chain_postgres.py tests/test_database_schema_contract.py
.\.venv\Scripts\python.exe -m ruff check app/models/generation.py app/repositories/generation.py app/schemas/generation.py tests/test_generation_repository.py
git add app/models/generation.py app/repositories/generation.py app/schemas/generation.py app/models/__init__.py app/alembic/versions/d0e1f2a3b4c5_add_generation_controls.py tests/test_generation_repository.py tests/integration/test_generation_repository_postgres.py tests/test_alembic_full_chain_postgres.py
git commit -m "feat: persist authoritative generation lifecycle"
```

### Task 2: Implement Idempotent Lifecycle Commands and Distributed Stop

> **Landed 2026-09-07.** 39 service/bus tests plus 9 registry tests; full suite
> 5069 passed. Deviations, all deliberate:
>
> - The registry re-key touched `app/services/message_service.py`, which Task 2's
>   file list does not include. The parameter rename would otherwise have broken
>   the one keyword call site and taken the suite red between commits; Task 5
>   changes *which* id is passed.
> - `await_stop_settled` was added beyond the listed interfaces. `request_stop`
>   returns as soon as the transition is durable, so something has to own the
>   bounded wait, and folding it into `request_stop` would have made the
>   idempotent command block on a worker in another process.
> - The four new settings are **absent from the environment template** — that
>   file is guard-blocked for this tooling and needs Thai. No test enforces
>   parity for the `generation_*` prefix, so nothing fails; they are simply
>   undocumented until added. Task 8's rollout step should cover them.

**Files:**
- Create: `app/services/generation_control_service.py`
- Create: `app/services/generation_control_bus.py`
- Modify: `app/services/generation_registry.py`
- Modify: `app/core/container.py`
- Modify: `app/core/config.py`
- Create: `tests/test_generation_control_service.py`
- Create: `tests/test_generation_control_bus.py`

**Interfaces:**

```python
class GenerationControlService:
    async def start_generation(self, command: StartGeneration) -> GenerationSnapshot: ...
    async def mark_running(self, generation_id: UUID, *, expected_version: int) -> GenerationSnapshot: ...
    async def mark_continuable(self, command: MarkContinuable) -> GenerationSnapshot: ...
    async def request_stop(self, command: StopGenerationCommand) -> GenerationSnapshot: ...
    async def prepare_continue(self, command: ContinueGenerationCommand) -> ContinuationLease: ...
    async def mark_stopped(self, command: MarkStopped) -> GenerationSnapshot: ...
    async def mark_completed(self, command: MarkCompleted) -> GenerationSnapshot: ...
```

- [x] **Step 1: Write the lifecycle table tests**

Parameterize every legal/illegal transition from the approved lifecycle. Explicitly assert:

- Stop on `running` returns `stop_requested` and publishes `(generation_id, version)`;
- Stop on `continuable` returns `completed_partial` without publishing cancellation;
- Stop repeated with the same or a new key is idempotent;
- Continue on `continuable` returns an execution lease with epoch + 1;
- Continue on `stopped` works only when `continuation_available` is true;
- Continue is rejected for terminal rows and `outcome_unknown` block reason;
- a Stop timeout leaves the durable state at `stop_requested`.

- [x] **Step 2: Run and verify failures**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_generation_control_service.py tests/test_generation_control_bus.py
```

- [x] **Step 3: Add the control bus**

Define a protocol plus two implementations:

```python
class GenerationControlBus(Protocol):
    async def publish_stop(self, generation_id: UUID, version: int) -> None: ...
    async def subscribe(self, handler: Callable[[StopSignal], Awaitable[None]]) -> None: ...
    async def close(self) -> None: ...
```

The Redis implementation publishes JSON containing only schema version, generation ID, and lifecycle version. Configure channel name and reconnect backoff in settings. `InMemoryGenerationControlBus` is deterministic for tests/dev.

- [x] **Step 4: Re-key the local registry**

Change `GenerationRegistry` to `generation_id` keys and add `task: asyncio.Task | None`. Preserve user/conversation/active-agent queries used by custom-agent mutation locks. Add `request_cancel(generation_id)` that sets the cooperative event and calls `task.cancel()` if active. Do not remove entries on HTTP wait timeout; remove only after the worker records a no-active-work status.

- [x] **Step 5: Wire service/repository/bus once in the container**

Register reusable providers in `app/core/container.py`. Do not instantiate an engine, Redis client, or subscriber inside request methods.

- [x] **Step 6: Run and commit Task 2**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_generation_control_service.py tests/test_generation_control_bus.py tests/test_custom_agents_message_service.py tests/test_custom_agents_service.py
.\.venv\Scripts\python.exe -m ruff check app/services/generation_control_service.py app/services/generation_control_bus.py app/services/generation_registry.py
git add app/services/generation_control_service.py app/services/generation_control_bus.py app/services/generation_registry.py app/core/container.py app/core/config.py tests/test_generation_control_service.py tests/test_generation_control_bus.py tests/test_custom_agents_message_service.py tests/test_custom_agents_service.py
git commit -m "feat: coordinate generation lifecycle and distributed stop"
```

### Task 3: Add Per-Epoch Soft/Hard Execution Budgets

> **Landed 2026-09-07.** Done and verified:
>
> - `app/ai/workflow/execution_budget.py` — the framework-free accountant (R1),
>   with 23 tests. Per-epoch counters reset on Continue, turn totals do not.
> - `app/ai/workflow/execution_budget_middleware.py` — the thin adapter, 12
>   tests. It pairs a `ToolMessage` with every refused call and turns the
>   framework's limit exception into an `exhausted_by="hard_limit"` outcome.
> - Wired into `SpecialistFactory._build`, covering top-level specialists **and
>   Planning's delegated workers** (`invoke_worker` uses the same builder).
> - Suppression via the tool factory, per R3 — the execution middleware
>   re-consults it on every model call, so it survives provider retries. Proven
>   through a real `create_agent` subgraph: the binding sequence is
>   `bind_tools, bind_tools, bind`, and `bind` is langchain's no-tools path.
> - Settings, with a cross-field validator so a hard rung can never sit at or
>   below its soft rung.
> - The framework ceilings now read the same settings, so there is one ladder.
>   Previously `specialist_max_model_calls` defaulted to 8 against a planned
>   hard limit of 9, which would have let the framework raise on exactly the
>   call the budget reserved.
>
> Completed in a second pass:
>
> - **RAG shares the accountant** — the last third of R1. The graph is compiled
>   once and shared, so the counters live in graph state and each node returns
>   what it advanced. Suppression rides the agent's existing
>   `rag_force_final_response` flag, which already maps to `disable_tools`.
> - **`WorkflowState.execution_budget` is populated**, from
>   `AgentResponse.metadata["execution_budget"]`. The RAG outcome writes the
>   same key, so the graph reads one place whichever path answered.
> - **The reserved call is told why its tools are gone**, once — a provider
>   fallback re-enters the chain, and a refused tool call already said it in
>   band. There was nothing to reconcile with the older `tool_budget_notice`:
>   `_mark_force_final_response` has no callers, so that path is dead.
> - **A hard limit is a server-owned partial**, not `agent_execution_limit`. It
>   carries the artifacts and images the pipeline recorded; the model's text
>   does not survive the exception, so the message says that rather than
>   inventing an answer. The metric still fires.
>
> **Resolved 2026-09-08 (the last R1 item).** `invoke_worker` no longer maps a
> hard limit to `_failed_worker(task, "agent_execution_limit")`. `WorkerStatus`
> gained `partial`, and the limit handler now returns
> `_partial_worker(task, tool_execution)` — the same server-owned text a
> top-level hard limit produces, carrying the artifacts and images the worker's
> tool pipeline had already recorded. `error_code` stays unset, because a
> partial is not an error and populating it renders as one wherever a worker
> end is shown; the `execution.limit.*` metric still fires, and
> `worker_completed` was already status-agnostic so `worker.partial.<agent>`
> needs no change. `render_worker_results` shows the synthesizing parent
> `status=partial`, and `build_planning_outcome` already aggregated evidence
> across results regardless of status — which is exactly what the old mapping
> was throwing away.

**Files:**
- Create: `app/ai/workflow/execution_budget.py`
- Modify: `app/ai/workflow/contracts.py`
- Modify: `app/ai/workflow/state.py`
- Modify: `app/ai/workflow/specialists.py`
- Modify: `app/core/config.py`
- Create: `tests/test_execution_budget_middleware.py`
- Modify: `tests/test_specialist_middleware.py`
- Modify: `tests/test_workflow_contracts.py`

**Interfaces:**

```python
class ExecutionBudgetState(BaseModel):
    execution_epoch: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    forced_synthesis: bool = False
    exhausted_by: Literal["model_calls", "tool_calls", "hard_limit"] | None = None


class SoftExecutionBudgetMiddleware(AgentMiddleware):
    async def awrap_model_call(self, request, handler): ...
    async def awrap_tool_call(self, request, handler): ...
```

- [x] **Step 1: Write failing middleware tests**

Use a scripted model/tool. At one below the soft threshold, a normal tool call runs. At the threshold, the middleware returns a paired synthetic `ToolMessage` saying evidence collection ended, then makes exactly one final model request with `tools=[]` and a server-owned synthesis instruction. Assert no handoff tool remains either. Test model-call and tool-call thresholds separately.

Assert counters reset when `execution_epoch` changes but cumulative turn quota metadata does not. Assert built-in hard-limit exceptions become `ExecutionBudgetState(exhausted_by="hard_limit")`, not `WorkflowError(code="agent_execution_limit")` exposed to the client.

- [x] **Step 2: Run and verify failures**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_execution_budget_middleware.py tests/test_specialist_middleware.py tests/test_workflow_contracts.py
```

- [x] **Step 3: Declare soft and hard settings**

```python
generation_soft_model_calls_per_epoch: int = Field(default=7, ge=1, le=50)
generation_hard_model_calls_per_epoch: int = Field(default=9, ge=2, le=60)
generation_soft_tool_calls_per_epoch: int = Field(default=12, ge=1, le=100)
generation_hard_tool_calls_per_epoch: int = Field(default=16, ge=2, le=120)
generation_total_epochs_per_turn: int = Field(default=5, ge=1, le=20)
generation_stop_wait_seconds: float = Field(default=5.0, ge=0.1, le=30.0)
```

Add cross-field validation: each hard threshold must exceed its soft threshold.

- [x] **Step 4: Implement forced synthesis**

The middleware owns counters for one specialist invocation. Before a call would exceed soft tool budget, return a correctly paired `ToolMessage` for the requested call and set `forced_synthesis`. The next model request receives no tools and this appended system instruction:

```text
Evidence gathering has ended for this execution epoch. Answer now using only
the evidence already present. State uncertainty and missing facts explicitly.
Do not request, promise, or imply another automatic tool call.
```

Mark the resulting `ResponseOutcome` metadata with the bounded budget snapshot. The specialist wrapper copies it into `WorkflowState.execution_budget` and routes normally to validation.

Retain LangChain's built-in run limit as a higher last-resort threshold. Catch its typed exception at the specialist boundary and create a deterministic server-owned fallback outcome from already collected evidence, marked `hard_limit`, so it still passes output validation.

- [x] **Step 5: Run and commit Task 3**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_execution_budget_middleware.py tests/test_specialist_middleware.py tests/test_workflow_contracts.py tests/test_specialist_subgraph_execution.py tests/test_graph_tool_budget.py tests/test_rag_tool_loop_finalization.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/execution_budget.py app/ai/workflow/contracts.py app/ai/workflow/state.py app/ai/workflow/specialists.py
git add app/ai/workflow/execution_budget.py app/ai/workflow/contracts.py app/ai/workflow/state.py app/ai/workflow/specialists.py app/core/config.py tests/test_execution_budget_middleware.py tests/test_specialist_middleware.py tests/test_workflow_contracts.py
git commit -m "feat: reserve tool-free synthesis at execution limits"
```

### Task 4: Pause after Validation and Resume without Re-routing

> **Landed 2026-09-07.** 29 tests. Built from R2, which the original text did
> not satisfy:
>
> - `carry_messages` keeps complete tool rounds and drops the epoch's own
>   answer, any human message, and any half-round. A provider rejects an
>   unanswered tool call, so a broken pair is dropped rather than repaired, and
>   an offloaded result stays a `blob_id` reference.
> - `SpecialistRequest.carried_messages` and `WorkflowState.carried_messages`
>   deliver it; `_invocation_messages` places it between history and the
>   question so a tool call is never separated from its result by a human
>   message.
> - `validate_output` routes a *validated* exhausted answer to
>   `continuation_pause`, and refuses to offer one when the turn has no epochs
>   left.
> - The pause node resumes into the agent the turn already chose, and every
>   refusal inside it goes to `finalize` rather than raising, because it sits on
>   the only path a paused turn can leave by.
>
> **Also landed:** `pending_continuation_payload` reads the pause back off a
> checkpoint, and is deliberately disjoint from
> `hitl_config.pending_interrupt_payload` — that one matches `action_requests`,
> this one the `type` literal, and both directions are asserted. It never
> raises: a checkpoint is read on every resume.
>
> **Still not done:** nothing *emits* the pause as a public stream event. The
> graph raises it and the reader can find it, but no adapter projects it, so a
> client cannot see a paused turn. That is Task 6's canonical
> `continuation_available` event, and until it lands the pause cannot be
> exercised end-to-end.

**Files:**
- Create: `app/ai/workflow/continuation.py`
- Modify: `app/ai/workflow/finalization.py:324-360`
- Modify: `app/ai/workflow/graph_builder.py:175-230`
- Modify: `app/ai/workflow/state.py`
- Modify: `app/ai/graph.py:2580-2905`
- Create: `tests/test_workflow_continuation.py`
- Modify: `tests/test_production_workflow_graph.py`
- Create: `tests/test_routing_v2_continuation_streaming.py`

**Interfaces:**

```python
class ContinuationPausePayload(BaseModel):
    type: Literal["execution_budget_exhausted"] = "execution_budget_exhausted"
    generation_id: UUID
    logical_turn_id: str
    execution_epoch: int
    active_agent_id: str
    validated_content: str
    budget: ExecutionBudgetState

class ContinuationResume(BaseModel):
    action: Literal["continue", "stop"]
    continuation_id: UUID
    expected_epoch: int
```

- [x] **Step 1: Write failing graph tests**

Assert normal outcomes still go `validate_output -> finalize -> END`. Exhausted outcomes go `validate_output -> continuation_pause`, and the checkpoint contains the same active agent/routing decision. `action="continue"` increments epoch and commands the existing active specialist directly; the route node is not called. `action="stop"` commands `finalize`. Invalid continuation ID/epoch does not execute the specialist.

- [x] **Step 2: Run and verify failures**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_continuation.py tests/test_production_workflow_graph.py tests/test_routing_v2_continuation_streaming.py
```

- [x] **Step 3: Add the continuation node**

`make_continuation_pause_node()` calls `interrupt(payload.model_dump(mode="json"))`. On Continue it returns:

```python
Command(
    update={
        "execution_epoch": state.get("execution_epoch", 0) + 1,
        "execution_budget": None,
        "execution_phase": "executing",
    },
    goto=resolve_node_for_agent_id(state["active_agent_id"]),
)
```

Resolve runtime agent IDs through the existing inventory/node mapping rather than assuming every agent ID equals a graph node. On Stop, set `execution_phase="finalizing"` and go to `finalize` without appending the validated message again.

- [x] **Step 4: Route validated exhausted outcomes to pause**

Extend `validate_output` destinations to `("continuation_pause", "finalize")`. It must first construct the same validated `AgentResponse`; only then branch on `execution_budget.exhausted_by`. Add `continuation_pause` to the graph, while preserving `finalize -> END` as the only terminal edge.

Extend graph stream normalization to emit a typed internal continuation event instead of treating this payload as a tool-approval interrupt.

- [x] **Step 5: Run and commit Task 4**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_continuation.py tests/test_production_workflow_graph.py tests/test_routing_v2_continuation_streaming.py tests/test_output_validation.py tests/test_routing_service.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/continuation.py app/ai/workflow/finalization.py app/ai/workflow/graph_builder.py app/ai/graph.py
git add app/ai/workflow/continuation.py app/ai/workflow/finalization.py app/ai/workflow/graph_builder.py app/ai/workflow/state.py app/ai/graph.py tests/test_workflow_continuation.py tests/test_production_workflow_graph.py tests/test_routing_v2_continuation_streaming.py
git commit -m "feat: pause validated limited responses for continuation"
```

### Task 5: Make MessageService Own Start, Pause, Continue, and Stop

> **Landed 2026-09-08** (`b907eec`). 48 tests across
> `test_message_generation_lifecycle.py` and
> `test_message_stream_cancellation.py`.
>
> The row is allocated before the first streamed event, so a Stop arriving on
> the very first token has something durable to transition. `run_start` — a
> declared-but-unemitted event type until now — carries the identity and the
> version that fences a later command. The registry keeps its entry keyed by
> generation id and now holds the producer task, because the cooperative event
> cannot reach a worker blocked inside a provider call.
>
> **Stop has one implementation, not two.** `stop_generation` transitions
> durably first and signals this process second; `stop_message_generation`
> becomes the turn-scoped entry point, resolving a user message id through the
> *logical turn* rather than through "whatever is active in this conversation"
> — that shortcut would let a stale turn id cancel the turn running now, which
> is the R5 defect one level up. A new repository read, `aget_by_logical_turn`,
> is what makes that fenced.
>
> Two things found by writing the tests rather than by reading the code:
>
> - A Stop that lost the race to the turn's own ending raised
>   `IllegalTransition` — a late Stop click was a 500. It now reports where the
>   turn landed, and the catch is narrow: an illegal transition on a row that
>   is still active stays an exception, because swallowing it would make Stop
>   appear to work while doing nothing.
> - `generations.assistant_message_id` was a plain foreign key, so deleting an
>   assistant message any generation referenced failed with
>   `ForeignKeyViolation`. Bookkeeping was outranking the thing it books.
>   Migration `e1f2a3b4c5d6` makes it `ON DELETE SET NULL`, matching
>   `model_usage_events.request_message_id`; `CASCADE` would have destroyed the
>   lifecycle history instead. **Applied to the live `chatbot` database**,
>   which was already at `d0e1f2a3b4c5` — the Task 1 note claiming otherwise
>   was stale.
>
> `ContinuationLease` gained `paused_epoch`. `prepare_continue` advances the
> *row*, but the checkpoint has not moved and the pause node fences against its
> own state, so the value sent back into the graph is the paused epoch. Sending
> the leased one is refused as stale, and that refusal is indistinguishable
> from a legitimately expired continuation — an off-by-one that would have made
> every Continue look broken for a reason nothing pointed at. Two named fields,
> not one and a subtraction.

**Files:**
- Modify: `app/services/message_service.py:1043-2225`
- Modify: `app/interfaces/message_service_interface.py`
- Modify: `app/services/ai_service.py:319-510`
- Modify: `app/services/generation_registry.py`
- Create: `tests/test_message_generation_lifecycle.py`
- Create: `tests/test_message_stream_cancellation.py`
- Modify: `tests/test_workflow_concurrency.py`

**Interfaces:**

```python
async def continue_message_generation_stream(
    self,
    *,
    generation_id: UUID,
    continuation_id: UUID,
    conversation_id: UUID,
    user_id: UUID,
    idempotency_key: str,
    bot_message_id: UUID | None = None,
    inline_rich_response_v1: bool = False,
) -> AsyncIterator[V3StreamEvent]: ...

async def stop_message_generation(
    self,
    *,
    generation_id: UUID,
    conversation_id: UUID,
    user_id: UUID,
    idempotency_key: str,
) -> GenerationSnapshot: ...
```

- [x] **Step 1: Write service lifecycle tests**

Prove generation row allocation occurs before `run_start`, whose data includes `generation_id`, `logical_turn_id`, and epoch. At a continuation pause, assert the validated assistant message is committed before `message_delta` and `continuation_available`. Then assert the turn advisory lock is released.

Continue must reacquire the same conversation lock, use the exact checkpoint/active agent, create a new assistant message, and not create a user message. Stop during active work cancels the task and persists partial text once. Stop while continuable resumes the pause with `action="stop"`, finalizes without duplicating the message, and returns `completed_partial`.

Test disconnect separately: closing the HTTP stream requests Stop but reports only authoritative lifecycle states; no registry-only `cancelled` result remains.

- [x] **Step 2: Run and verify failures**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_message_generation_lifecycle.py tests/test_message_stream_cancellation.py tests/test_workflow_concurrency.py
```

- [x] **Step 3: Allocate identity and register the owner task**

In `create_message_stream`, persist the user message, create generation identity, emit `run_start`, and register the current producer task by generation ID. The existing `user_message_created` event may carry the ID for compatibility, but Stop logic must ignore it.

At every graph event and before/after tool/model boundaries, check the cooperative cancel event and durable status. On `CancelledError`, shield the short persistence/classification transaction, then re-raise only after state is authoritative.

- [x] **Step 4: Persist before publishing continuation**

Buffer the validated pause response, write the assistant partial with metadata `{generation_id, logical_turn_id, execution_epoch, partial: true}`, transition to `continuable` with a new opaque continuation ID, then emit public deltas and `continuation_available`. If persistence fails, mark `failed` and emit no answer/control event.

- [x] **Step 5: Add exact-checkpoint Continue**

Call `prepare_continue`, then `AIService.resume_generation_control_stream` which wraps `workflow.resume_with_continuation_stream(Command(resume={...}))`. Do not call `create_message_stream`, the router, `sendMessage`, regenerate, or append a synthetic/hidden user message.

- [x] **Step 6: Replace registry-only Stop**

Authorize through `GenerationControlService`, transition durably, publish Redis, signal the local owner if present, and await its completion only up to the configured timeout. Return the current `GenerationSnapshot`; `stop_requested` is a successful pending state, not a timeout exception.

Classify mutation receipts before marking resumable. Any `outcome_unknown` sets `continuation_available=false` and `continuation_block_reason="mutation_outcome_unknown"`.

- [x] **Step 7: Run and commit Task 5**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_message_generation_lifecycle.py tests/test_message_stream_cancellation.py tests/test_workflow_concurrency.py tests/test_generation_control_service.py tests/test_tool_execution_receipt_service.py
.\.venv\Scripts\python.exe -m ruff check app/services/message_service.py app/services/ai_service.py app/services/generation_registry.py
git add app/services/message_service.py app/interfaces/message_service_interface.py app/services/ai_service.py app/services/generation_registry.py tests/test_message_generation_lifecycle.py tests/test_message_stream_cancellation.py tests/test_workflow_concurrency.py
git commit -m "feat: orchestrate durable stop and exact continuation"
```

### Task 6: Expose One Control API Through Both Transports

> **Landed 2026-09-08** (`ed8c384`). 26 API contract tests plus 7 proxy tests.
>
> `run_start` is reused as the start event rather than adding the
> `generation_start` this plan named: a second canonical event carrying an
> identical payload is two systems doing one job. The internal SSE *publishes*
> it as `generation_start`, and the AI SDK folds all three lifecycle events
> into one `data-generation` part discriminated by `phase`. A test compares the
> two adapters' published field sets rather than trusting them to agree.
>
> The part is not `transient`: a reconnecting client still needs the identity
> and the version that make a command addressable at all.
>
> Three fields never cross the boundary - the validated partial's text (it
> arrived as deltas and lives in the message row), the execution budget, and
> the checkpoint thread. Asserted by searching the rendered response, not by
> reviewing the projection.
>
> Two boundary defects fixed while wiring it:
>
> - The client-backend stop proxy returned the parsed body, flattening a `202`
>   into a `200` and telling the client a turn had ended that no worker had
>   confirmed. It now returns the upstream response.
> - `/messages/generations/{id}` has to be declared *before*
>   `/messages/{message_id}` in both the server and the sidecar, because
>   FastAPI matches in order and the parameterized route otherwise captures
>   "generations" as a message id. A test asserts the ordering, since the
>   symptom is a 404 that reads like an ownership refusal.
>
> Stop accepts either `generationId` (canonical) or `userMessageId` (the
> turn-scoped form). Both `StopGenerationRequest` and
> `ContinueGenerationRequest` carry `expectedVersion` per R5; it is optional
> only for a client predating `run_start`, and the endpoint then reads the row
> rather than inventing a fence the client never held.

**Files:**
- Modify: `app/schemas/generation.py`
- Modify: `app/api/messages.py`
- Modify: `app/api/ai_sdk.py`
- Modify: `app/services/event_streaming/events.py`
- Modify: `app/services/event_streaming/internal_sse.py`
- Modify: `app/services/event_streaming/ai_sdk_v6.py`
- Modify: `client_backend/api/messages.py`
- Modify: `client_backend/services/server_api.py`
- Create: `tests/test_generation_control_api.py`
- Modify: `tests/test_internal_sse_stream_contract.py`
- Modify: `tests/test_ai_sdk_v6_stream_contract.py`
- Modify: `tests/client_backend/test_messages.py`

**Canonical commands:**

```python
class StopGenerationRequest(BaseModel):
    generation_id: UUID
    conversation_id: UUID
    idempotency_key: str = Field(min_length=8, max_length=160)

class ContinueGenerationRequest(BaseModel):
    generation_id: UUID
    continuation_id: UUID
    conversation_id: UUID
    idempotency_key: str = Field(min_length=8, max_length=160)
    inline_rich_response_v1: bool = False
```

- [x] **Step 1: Write API and adapter contract tests**

Add owner-scope 404 tests, idempotency tests, and these endpoints:

- `POST /messages/stop` → snapshot JSON;
- `GET /messages/generations/{generation_id}?conversation_id=...` → snapshot JSON;
- `POST /messages/continue` → internal SSE;
- `POST /ai/continue` → AI SDK v6 stream;
- client-backend proxies for all four.

Both stream adapters must expose `generation-start`, `generation-status`, and `continuation-available` from canonical events. AI SDK uses custom `data-generation` parts; internal SSE uses the canonical names unchanged. Neither adapter derives state from socket close.

- [x] **Step 2: Run and verify failures**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_generation_control_api.py tests/test_internal_sse_stream_contract.py tests/test_ai_sdk_v6_stream_contract.py tests/client_backend/test_messages.py
```

- [x] **Step 3: Add canonical events and routes**

Extend `StreamEventType` with `generation_start`, `generation_status`, and `continuation_available`. The continuation event data contains IDs, status, epoch, assistant message ID, and safe block reason only—no checkpoint internals or raw content.

Both Continue routes call the same MessageService method and differ only in adapter. Both Stop callers use the same JSON route. Return `202` when status remains `stop_requested`, otherwise `200`.

- [x] **Step 4: Document AI SDK client semantics**

In the route description and stream-contract tests, pin this client flow:

```ts
async function stopGeneration(snapshot: GenerationSnapshot) {
  stop(); // abort useChat's fetch immediately
  return fetch("/messages/stop", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      generationId: snapshot.generationId,
      conversationId: snapshot.conversationId,
      idempotencyKey: crypto.randomUUID(),
    }),
  });
}
```

Continue posts to `/ai/continue` and consumes its UI Message Stream. It must not call `resumeStream`, `regenerate`, or `sendMessage`; `resume: false` stays configured because semantic continuation is explicit.

- [x] **Step 5: Run and commit Task 6**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_generation_control_api.py tests/test_internal_sse_stream_contract.py tests/test_ai_sdk_v6_stream_contract.py tests/client_backend/test_messages.py tests/test_message_service_event_streaming.py
.\.venv\Scripts\python.exe -m ruff check app/api/messages.py app/api/ai_sdk.py app/services/event_streaming client_backend/api/messages.py client_backend/services/server_api.py
git add app/schemas/generation.py app/api/messages.py app/api/ai_sdk.py app/services/event_streaming/events.py app/services/event_streaming/internal_sse.py app/services/event_streaming/ai_sdk_v6.py client_backend/api/messages.py client_backend/services/server_api.py tests/test_generation_control_api.py tests/test_internal_sse_stream_contract.py tests/test_ai_sdk_v6_stream_contract.py tests/client_backend/test_messages.py
git commit -m "feat: expose generation controls across stream transports"
```

### Task 7: Implement the Streamlit Continue/Stop State Machine

> **Landed 2026-09-08** (`7984a7f`). 36 tests.
>
> `generation_controls` is a pure function from canonical status to which
> buttons appear, so a closed socket changes nothing. A guard test asserts every
> declared `GenerationStatus` is covered by one of the three sets it branches
> on - an uncovered status falls through to "no controls" silently, and both
> buttons would vanish from a turn that is still running.
>
> `_handle_stop_rerun` no longer reports "Generation stopped" for a
> `stop_requested`. That was the one claim this control must never make.
>
> The idempotency key is derived from (action, generation, version) rather than
> generated per render: Streamlit reruns the whole script on every click, so a
> key held in a local is fresh each time - which is how a control that is "safe
> because idempotent" stops being idempotent.
>
> `generation_snapshot` deliberately survives `_clear_inflight_state`: a paused
> turn is not in flight but is still continuable.
>
> **One deliberate shortfall.** `_handle_continue` drives its own renderer
> rather than the ~200-line inline block a new turn uses, so "feeds its events
> through the same renderer" is met only at the level of the event vocabulary.
> Extracting that shared renderer is a refactor of untested UI code and was not
> attempted.

**Files:**
- Modify: `demo.py:2680-2805`
- Modify: `demo.py:8900-9100`
- Modify: `demo.py:10570-10680`
- Modify: `tests/test_demo_stop_generation.py`
- Create: `tests/test_demo_generation_controls.py`

- [x] **Step 1: Write failing UI state tests**

Test button visibility and state transitions:

| Canonical status | Stop | Continue | UI behavior |
|---|---:|---:|---|
| `starting`, `running`, `continuing`, `finalizing_after_limit` | enabled | hidden | Stop aborts local iterator and posts command |
| `stop_requested` | disabled | hidden | poll/refetch until authoritative transition |
| `continuable` | enabled | enabled | Stop accepts partial; Continue opens new stream |
| `stopped` + resumable | hidden | enabled | Continue opens new stream |
| `completed`, `completed_partial`, `failed` | hidden | hidden | ordinary message view |

Assert Continue uses generation/continuation IDs, never creates a pending user bubble, and appends deltas to a new assistant message. Assert repeated clicks reuse an idempotency key until a response arrives.

- [x] **Step 2: Run and verify failures**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_demo_stop_generation.py tests/test_demo_generation_controls.py
```

- [x] **Step 3: Store canonical generation snapshots in session state**

Replace the two-phase `user_message_id` cancellation token with a generation snapshot. Consume `generation_start`, `generation_status`, and `continuation_available` events. A network close alone changes no canonical status.

- [x] **Step 4: Implement buttons against shared endpoints**

Stop first closes the local iterator for responsiveness, posts `/messages/stop`, then reconciles from its snapshot/status endpoint. Continue posts `/messages/continue` and feeds its events through the same renderer used for a new stream. Stop from `continuable` keeps the partial message and resolves to `completed_partial`.

- [x] **Step 5: Run and commit Task 7**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_demo_stop_generation.py tests/test_demo_generation_controls.py tests/test_hitl_demo_panel.py tests/test_rich_response_streaming.py
.\.venv\Scripts\python.exe -m ruff check demo.py tests/test_demo_generation_controls.py
git add demo.py tests/test_demo_stop_generation.py tests/test_demo_generation_controls.py
git commit -m "feat: add consistent Streamlit generation controls"
```

### Task 8: Verify Parity, Races, and Rollout Safety

> **Landed 2026-09-08.** 16 parity tests, 13 PostgreSQL race tests, and the
> rollout procedure in `docs/operations/routing-v2-rollout.md`.
>
> The race suite taught something worth recording. Written first, all of it
> passed with `WHERE version = :expected` deleted from the repository - so it
> was verifying "one winner" while resting entirely on a predicate that only
> happens to cover those cases. Two other things settle those races before the
> fence is reached: the service's own `_require_fresh` refuses a command whose
> version has moved (the sequential retry), and `atransition`'s
> `status IN (...)` predicate refuses a concurrent loser whose winner moved the
> row out of the declared set.
>
> `test_the_repository_version_fence_refuses_the_second_writer` isolates the
> fence itself - two writers reading one version and targeting `stop_requested`,
> a status that is *itself* stoppable, so nothing but the version tells them
> apart. It is asserted against `atransition` directly, and it is the one test
> in the file that fails when the fence is removed. Verified by removing it.
>
> The parity tests normalize both adapter projections to a transport-independent
> shape and compare them as values, across nine scenarios. Verified by making
> one adapter stop stripping the private pause fields: seven tests failed.
>
> The rollout section documents the settings (all carried defaults), migration
> order - `e1f2a3b4c5d6` before any retention job, or message deletion fails -
> subscriber health, the SQL for stuck `stop_requested` rows, and the metrics
> that exist versus the three that do not.

**Files:**
- Create: `tests/test_generation_transport_parity.py`
- Create: `tests/integration/test_generation_control_races_postgres.py`
- Modify: `docs/operations/routing-v2-rollout.md`

- [x] **Step 1: Add black-box parity scenarios**

Run the same scripted workflow through internal SSE and AI SDK and compare normalized event projections for: normal completion, soft-limit continuation, Stop during provider wait, Stop timeout, Stop while continuable, Continue after pause, repeated Continue, disconnect, and blocked mutation outcome. Assert identical final lifecycle snapshots and persisted messages.

- [x] **Step 2: Add PostgreSQL race tests**

Race Continue/Continue, Stop/Stop, and Continue/Stop from separate sessions. Assert one legal winning transition, one epoch increment, no duplicate assistant message, and monotonic versions.

- [x] **Step 3: Add live rollout procedure**

Document feature flags, migration order, Redis subscriber health, stuck `stop_requested` reconciliation, lifecycle metrics, execution-budget metrics, continuation conversion, duplicate-command count, and rollback. Roll out server persistence/events first, then client-backend proxy, then Streamlit/AI SDK buttons, then enable soft-limit pausing.

- [x] **Step 4: Run the complete verification set**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_generation_repository.py tests/integration/test_generation_repository_postgres.py tests/test_generation_control_service.py tests/test_generation_control_bus.py tests/test_execution_budget_middleware.py tests/test_workflow_continuation.py tests/test_message_generation_lifecycle.py tests/test_generation_control_api.py tests/test_generation_transport_parity.py tests/integration/test_generation_control_races_postgres.py tests/test_internal_sse_stream_contract.py tests/test_ai_sdk_v6_stream_contract.py tests/test_demo_stop_generation.py tests/test_demo_generation_controls.py tests/test_production_workflow_graph.py tests/test_routing_service.py tests/test_routing_v2_continuation_streaming.py tests/test_tool_execution_receipt_service.py
.\.venv\Scripts\python.exe -m ruff check app/models/generation.py app/repositories/generation.py app/schemas/generation.py app/services/generation_control_service.py app/services/generation_control_bus.py app/services/generation_registry.py app/ai/workflow/execution_budget.py app/ai/workflow/continuation.py app/ai/workflow/contracts.py app/ai/workflow/state.py app/ai/workflow/specialists.py app/ai/workflow/finalization.py app/ai/workflow/graph_builder.py app/ai/graph.py app/services/message_service.py app/services/ai_service.py app/api/messages.py app/api/ai_sdk.py app/services/event_streaming client_backend/api/messages.py client_backend/services/server_api.py demo.py
```

- [x] **Step 5: Commit Task 8**

```powershell
git add tests/test_generation_transport_parity.py tests/integration/test_generation_control_races_postgres.py docs/operations/routing-v2-rollout.md
git commit -m "test: verify generation control parity and races"
```

## Acceptance Checklist

Verified 2026-09-08. Suite: 5385 passed, 130 skipped. The PostgreSQL items
ran against a real `chatbot_test` (141 tests) and the control-bus items
against a real Redis, not fakes.

**Every item below is now checked.** Two were checked only after the notes
excusing them turned out to be wrong: see the Stop entry, whose
"verified by construction" claim described a mechanism with no receiving
half.

- [x] Soft model/tool limits always reserve one tool-free answer call.
- [x] Hard-limit defects produce a validated partial/fallback, not a generic public error.
- [x] Continuation pauses only after assistant content is validated and persisted.
- [x] Continue resumes the exact checkpoint/agent without routing or a user message.
- [x] Stop is durable, distributed, idempotent, and honest about pending cancellation.
  **The "distributed" claim was false when first written here, and the note
  that excused it was wrong twice over.** It said a worker missing the signal
  "finds `stop_requested` on its next check" — there was no such check — and it
  called the delivery path verified by construction when the construction had
  no receiving half at all: `bus.subscribe` was never called anywhere in
  production, so `publish_stop` broadcast into a void. A Stop landing on a
  worker other than the streaming one transitioned the row and interrupted
  nothing.
  Both halves now exist. `app/services/generation_stop_subscriber.py` attaches
  this process's registry to the bus at startup (the fast path, best effort),
  and `_DurableStopWatch` has the streaming worker read its own
  `generations.status` at tool and model boundaries (the authoritative path,
  since the row cannot un-stop). Verified over a real Redis in
  `tests/integration/test_generation_stop_bus_redis.py` — two buses, separate
  connections, publisher and subscriber sharing no Python state — and end to
  end through the real stream loop in `tests/test_generation_stop_delivery.py`.
  Both were confirmed by removing the mechanism and watching them fail.
- [x] Unknown mutation outcomes block continuation.
  Wired 2026-09-08, having been plumbed-but-unreachable: `ToolExecutionMiddleware`
  records the undecidable mutation, `_to_outcome` carries it, the pause payload
  reports it, and `_apublish_continuation_pause` turns it into
  `continuation_block_reason="mutation_outcome_unknown"` with no continuation id
  minted. The partial answer is still persisted and shown.
- [x] Internal SSE, AI SDK, client-backend proxy, and Streamlit produce the same lifecycle result.
  `tests/test_generation_transport_parity.py` normalizes both adapter
  projections and compares them as values across nine scenarios, and asserts
  their published field sets are identical.
- [x] AI SDK Stop combines local `useChat.stop()` with the explicit server Stop command.
  Pinned in the `/ai/continue` route description. Not executed — it is client
  TypeScript this repository does not own.
- [x] Streamlit exposes both Continue and Stop according to the canonical status table.
- [x] Race tests show one epoch increment and no duplicate assistant message.
- [x] `finalize -> END` remains the graph's sole terminal edge.
- [x] Forced synthesis and hard-limit handling are asserted on the specialist, RAG, and delegated-worker paths (R1).
- [x] Epoch 2 receives epoch 1's evidence, asserted on the messages handed to the model (R2).
- [x] The final tool-free call is asserted after the full middleware stack, including one provider fallback (R3).
- [x] Research accounting is keyed by logical turn, persisted, and rehydrated; a failed rehydration fails the Continue (R4).
  Done 2026-09-08. The store is keyed by logical turn, threaded through
  `ToolContext.logical_turn_id` to all three `tool_execution_context` call
  sites and both `RagExecutionRequest` constructions. `to_state`/
  `research_budget_from_state` persist the dedup memory onto the row; the pause
  writes it and both Continue paths rehydrate through the same
  `install_research_budget`, so a same-worker Continue cannot get a different
  allowance than a cross-worker one.
  What a Continue replenishes is now explicit and tested: the per-epoch call
  caps reset, the "already searched this" memory does not. Only token sets
  travel, never result text — the next epoch already has the previous one's
  `ToolMessage`s through `carried_messages`, and a row is the wrong place for
  provider output.
  An unreadable payload raises `ResearchAccountingUnreadable` and fails the
  Continue with a typed `research_accounting_unreadable` error. Degrading to an
  empty budget is the one thing it must not do, because an empty budget is
  indistinguishable from a fresh turn's full quota. Eleven parameterized cases
  cover the malformed shapes; both guarantees were confirmed by mutation.
  The accessors are keyword-only. Before R4 the single positional parameter was
  the conversation id, so making the turn id positional would have let every
  existing call keep working against a *different* key — a silently wrong cache
  key, which the image-research tests caught immediately.
- [x] A delayed Stop replay after a later Continue is refused as stale, not executed (R5).
  Tested at the service boundary and against the real fence in
  `tests/integration/test_generation_control_races_postgres.py`.

### Not covered by any test

Named here rather than left for a reader to discover:

- **No turn has been continued by a real model.** The pause, the carried
  evidence and the resume are covered by tests including a real compiled
  LangGraph with a real checkpointer, but every model in them is scripted.
- **No Stop has been raced across two OS processes.** The database half is
  verified against a real PostgreSQL from separate sessions, and the Redis hop
  is verified between two buses on separate connections — which is what Redis
  actually distinguishes. What remains unexercised is a process boundary, to
  which Redis is indifferent.
- **The seven execution rungs are carried defaults.** They decide when a user's
  turn is cut short in favour of a partial answer, and none was selected from
  measurement.
- **The lifecycle has no metrics.** Transitions, continuation conversion and
  duplicate command claims are observable only by SQL. The queries are in
  `docs/operations/routing-v2-rollout.md`.

## Execution Handoff

**Read "Design Revisions Required Before Implementation" first.** Tasks 3 and 4
must not be started from their original text: the budget cannot live only in an
`AgentMiddleware` (R1), Continue must rehydrate the evidence (R2), and tool
suppression belongs at the request boundary where the repository already
implements it (R3).

Execute this plan after the focused-web and trace-parenting plans. Rebase it on
the completed production routing refactor before Task 3, because Tasks 3-5
intentionally extend the routing-v2 state, topology, validation, and resume
contracts rather than replacing them.
