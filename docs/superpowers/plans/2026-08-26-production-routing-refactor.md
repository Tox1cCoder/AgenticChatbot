# Production Routing Refactor Root-Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete routing-v2 by making Planning approval resumes replay-safe, replacing duplicate Planning/RAG execution paths, and making mutation retries durable and truthful.

**Architecture:** The outer checkpointed `StateGraph` owns Planning model, dispatch, worker, collect, action, and package nodes. One reusable RAG graph and one specialist worker path serve public and delegated work; pending LangGraph interrupts are the recovery authority, and durable execution receipts close the mutation/checkpoint crash gap.

**Tech Stack:** Python 3.10+, LangGraph 1.2.9, LangChain 1.3.14, Pydantic v2, SQLAlchemy, PostgreSQL, Alembic, pytest, Ruff.

## Global Constraints

- This is an atomic breaking cutover. Add no feature flag, compatibility alias, fallback route, dual execution path, or old-checkpoint reader.
- Planning fan-out must be real outer-graph topology. A tool, helper, or compiled Planning child graph must never invoke it.
- `dispatch_subagents` is a model-facing control schema and must never reach `execute_tool_calls`.
- Validate a dispatch completely before scheduling any task. Reject invalid or oversized input; never truncate it.
- Turn-wide defaults are exactly 8 tasks, 4 concurrent workers, 2 dispatch waves, 4,000 objective characters, and 12,000 parent-context characters.
- Worker identity is `(dispatch_id, task_id)`; collection order uses server-owned `position`.
- `GraphBubbleUp` and `GraphInterrupt` cross every generic exception boundary unchanged.
- Compile RAG once per workflow construction. Authenticated tools, credentials, messages, evidence allocators, and state remain invocation-scoped.
- Grounding is mandatory and reports rather than rewrites. Validation records findings; it never regenerates an answer or substitutes an abstention, because a draft that can be replaced wholesale cannot be streamed. The reader is protected at the citation marker instead. (Amended 2026-09-03; supersedes the regenerate-once-then-abstain constraint, which predates the 2026-08-27 streaming decision in `553eebb`.)
- Pending checkpoint interrupts—not parent messages or node-name allowlists—are authoritative.
- Planning uses injected `StreamWriter` events. Do not checkpoint callbacks, queues, weak references, or sink tokens.
- Models cannot provide or read mutation execution keys. Claim external exactly-once behavior only for providers supporting idempotency.
- Only `finalize` reaches `END`; public answer deltas remain withheld until validation and transactional response persistence succeed.
- Deterministic test commands set `LANGSMITH_TRACING=false`.

## Landed Baseline

The branch already contains typed routing-v2 state, structured route-once routing, standard specialist subgraphs, parent-command handoffs, provenance validation/finalization, per-turn checkpoint IDs, turn coordination, retention, and preliminary Planning/RAG helper types. Do not recreate those components. The known-bad live paths are `app/ai/workflow/rag_loop.py`, `app/ai/workflow/planning_loop.py`, `app/ai/planning_subagents.py`, and `MultiAgentWorkflow._run_agent_in_isolated_context`.

## File and Responsibility Map

- `app/ai/workflow/contracts.py` — canonical worker and dispatch identities.
- `app/ai/workflow/state.py` — Planning checkpoint fields and collision-safe reducers.
- `app/ai/workflow/planning_execution.py` — dispatch validation and Planning node factory; no child fan-out graph.
- `app/ai/workflow/rag_execution.py` — once-compiled RAG topology used by both entry points.
- `app/ai/workflow/specialists.py` — one public/worker specialist construction path with worker scope enforcement.
- `app/ai/workflow/graph_builder.py` — sole owner of production topology.
- `app/ai/workflow/runtime_context.py` — non-checkpointed node collaborators.
- `app/models/tool_execution_receipt.py`, `app/repositories/tool_execution_receipt.py`, `app/services/tool_execution_receipt_service.py` — durable mutation state machine.
- `app/ai/workflow/middleware.py` and `app/ai/tool_execution.py` — authorization, approval, receipts, then execution.
- `app/ai/graph.py` — dependency wiring and authoritative interrupt recovery.
- `app/services/event_streaming/graph_public_projection.py` — custom worker event projection without answer leakage.
- `app/core/config.py` and `app/core/container.py` — validated limits and production dependency construction.

---

### Task 1: Make Planning contracts canonical and atomic

**Files:**
- Modify: `app/ai/workflow/contracts.py:170`
- Modify: `app/ai/workflow/state.py:66`
- Modify: `app/ai/workflow/planning_execution.py:48`
- Modify: `app/core/config.py:1183`
- Test: `tests/test_workflow_contracts.py`
- Test: `tests/test_workflow_state.py`
- Test: `tests/test_planning_execution_graph.py`

**Interfaces:**
- Consumes: existing `RoutingInventory` and routing-v2 checkpoint serializer.
- Produces: `WorkerTaskProposal`, `DispatchSubagentsInput`, `WorkerTask`, `WorkerResult`, `PlanningDispatch`, `PlanningLimits`, `append_worker_results()`, and `validate_dispatch_call()`.

- [ ] **Step 1: Write failing contract and reducer tests**

```python
def test_worker_result_identity_is_dispatch_and_task() -> None:
    first = WorkerResult(dispatch_id="d1", task_id="t1", position=0,
                         agent_id="chat_agent", status="completed", content="one")
    second = first.model_copy(update={"dispatch_id": "d2", "content": "two"})
    assert append_worker_results([], [first, second]) == [first, second]
    with pytest.raises(InvalidWorkflowStateUpdate, match="d1.*t1"):
        append_worker_results([first], [first])


def test_model_proposal_cannot_widen_tool_scope() -> None:
    with pytest.raises(ValidationError):
        WorkerTaskProposal.model_validate({
            "task_id": "t1", "objective": "inspect", "agent_id": "chat_agent",
            "allowed_tool_ids": ["admin::delete"],
        })
```

Also test that the ninth task, duplicate IDs, recursive Planning, detached custom agents, oversized objectives/context, mixed dispatch plus handoff, and a third wave start zero workers and produce one paired control error.

- [ ] **Step 2: Run tests and confirm old shape/limit failures**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_contracts.py tests/test_workflow_state.py tests/test_planning_execution_graph.py
```

Expected: failures show missing `dispatch_id`/`position`, collision by `task_id` alone, truncation behavior, and no dispatch-wave setting.

- [ ] **Step 3: Implement exact control and checkpoint types**

```python
class WorkerTaskProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=4000)
    agent_id: str = Field(min_length=1, max_length=160)
    related_todo_ids: tuple[str, ...] = ()


class DispatchSubagentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tasks: tuple[WorkerTaskProposal, ...] = Field(min_length=1)
    rationale: str | None = Field(default=None, max_length=1000)


class WorkerTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dispatch_id: str
    task_id: str
    position: int = Field(ge=0)
    objective: str = Field(min_length=1, max_length=4000)
    agent_id: str
    parent_context: dict[str, JsonValue] = Field(default_factory=dict)
    allowed_tool_ids: tuple[str, ...] = ()
    model_request: dict[str, JsonValue] | None = None
    related_todo_ids: tuple[str, ...] = ()


class WorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dispatch_id: str
    task_id: str
    position: int = Field(ge=0)
    agent_id: str
    status: Literal["completed", "failed"]
    content: str = ""
    artifacts: tuple[dict[str, JsonValue], ...] = ()
    evidence: tuple[dict[str, JsonValue], ...] = ()
    images: tuple[dict[str, JsonValue], ...] = ()
    error_code: str | None = None


class PlanningDispatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dispatch_id: str
    tool_call_id: str
    wave: int = Field(ge=1)
    tasks: tuple[WorkerTask, ...]
```

Add `planning_dispatch`, `planning_dispatch_waves`, `planning_dispatched_task_count`, and `planning_control_call_id` to `WorkflowState`. Key `append_worker_results()` by `(dispatch_id, task_id)`. Add `PlanningLimits.max_dispatch_waves` and `planning_worker_max_dispatch_waves: int = Field(default=2, gt=0, le=8)`.

- [ ] **Step 4: Implement all-or-nothing dispatch validation**

```python
class InvalidPlanningDispatch(ValueError):
    def __init__(self, code: str, tool_call_id: str) -> None:
        super().__init__(code)
        self.code = code
        self.tool_call_id = tool_call_id


def validate_dispatch_call(
    *, tool_call: dict[str, Any], state: WorkflowState,
    inventory: RoutingInventory, limits: PlanningLimits,
    resolve_allowed_tools: Callable[[str, WorkflowState], tuple[str, ...]],
) -> PlanningDispatch:
    """Validate the entire proposal before returning server-owned tasks."""
```

Validate all tasks before constructing any `Send`. Generate `dispatch_id` as `sha256(f"{tool_call_id}:{wave}".encode()).hexdigest()[:32]`, preserve input positions, and derive tool scope from live definitions/device/authorization. Raise `InvalidPlanningDispatch(code, tool_call_id)`; the Planning model node later turns it into `ToolMessage(id=f"planning-control:{tool_call_id}")`.

- [ ] **Step 5: Run focused tests and commit**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_contracts.py tests/test_workflow_state.py tests/test_checkpoint_serializer.py tests/test_planning_execution_graph.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/contracts.py app/ai/workflow/state.py app/ai/workflow/planning_execution.py app/core/config.py
git add app/ai/workflow/contracts.py app/ai/workflow/state.py app/ai/workflow/planning_execution.py app/core/config.py tests/test_workflow_contracts.py tests/test_workflow_state.py tests/test_planning_execution_graph.py
git commit -m "refactor: define canonical planning dispatch state"
```

---

### Task 2: Build one production RAG graph and one scoped worker path

**Files:**
- Modify: `app/ai/workflow/rag_execution.py`
- Modify: `app/ai/workflow/specialists.py:436`
- Modify: `app/ai/workflow/planning_execution.py:139`
- Modify: `app/ai/workflow/runtime_context.py`
- Modify: `app/ai/graph.py:230`
- Test: `tests/test_rag_execution_graph.py`
- Test: `tests/test_worker_rag_grounding.py`
- Test: `tests/test_planning_execution_graph.py`
- Test: `tests/test_worker_approval_gated_tools.py`
- Test: `tests/test_custom_agents_planning.py`

**Interfaces:**
- Consumes: Task 1 types, current RAG tools/grounding gate, specialist definitions, request HITL policy, custom-agent inventory, attachments, and model request.
- Produces: one `RagExecutionGraph.ainvoke()` object and `PlanningWorkerRuntime.run(task, state, writer) -> WorkerResult`.

- [ ] **Step 1: Write failing RAG and worker-scope tests**

```python
async def test_invalid_grounding_is_recorded_without_a_second_generation(rag_graph) -> None:
    result = await rag_graph.ainvoke(_request(), config=_config(), context=_context())
    assert result.grounding.regeneration_count == 0
    assert result.grounding.outcome == "accepted_with_findings"
    assert result.abstained is False


async def test_public_and_worker_rag_share_compiled_graph(workflow) -> None:
    assert workflow.rag_execution_graph is workflow.planning_worker_runtime.rag_execution_graph
    assert workflow.rag_execution_graph.compile_count == 1


async def test_worker_receives_objective_and_restricted_scope(runtime) -> None:
    await runtime.run(_task(objective="Calculate tax", allowed_tool_ids=("calc::tax",)),
                      _parent_state(), _writer)
    request = runtime.specialist_factory.last_request
    assert request.messages[-1] == HumanMessage(content="Calculate tax")
    assert request.extras["allowed_tool_ids"] == ("calc::tax",)
    assert request.extras["hitl_policy"] == _parent_state()["context"]["hitl_policy"]
```

Cover actual evidence/artifact/image packaging, zero-evidence validation, request-scoped custom workers, attachment descriptors, and an attempted tool call outside `allowed_tool_ids`.

- [ ] **Step 2: Run tests and observe the incomplete linear/empty-message failures**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_execution_graph.py tests/test_worker_rag_grounding.py tests/test_planning_execution_graph.py tests/test_worker_approval_gated_tools.py tests/test_custom_agents_planning.py
```

Expected: failures show no RAG tool loop/regeneration/provenance, per-run build behavior, empty worker messages, unenforced tool scope, or swallowed control flow.

- [ ] **Step 3: Compile the reusable RAG topology once**

Give private RAG state an `Annotated[list[BaseMessage], add_messages]` channel and evidence/artifact/image reducers. Compile once with `checkpointer=True`:

```python
graph.add_node("rag_model", self._model_node)
graph.add_node("rag_tools", ToolNode(self._tools, handle_tool_errors=False))
graph.add_node("collect_rag_outputs", self._collect_outputs)
graph.add_node("validate_grounding", self._validate)
graph.add_node("package_rag_result", self._package)
graph.add_edge(START, "rag_model")
graph.add_conditional_edges("rag_model", self._after_model,
                            {"tools": "rag_tools", "validate": "validate_grounding"})
graph.add_edge("rag_tools", "collect_rag_outputs")
graph.add_edge("collect_rag_outputs", "rag_model")
graph.add_edge("validate_grounding", "package_rag_result")
graph.add_edge("package_rag_result", END)
self._compiled = graph.compile(checkpointer=True)
```

Every result passes through `validate_grounding`, and there is no path around it. Validation records what it found and packages the same answer: an unresolvable citation is neutralized where it renders, and a coverage shortfall is a recorded finding. Populate result evidence, artifacts, and images from server-produced tool records.

- [ ] **Step 4: Build the canonical specialist worker request**

```python
def build_worker_request(task: WorkerTask, state: WorkflowState) -> SpecialistRequest:
    return SpecialistRequest(
        agent_id=task.agent_id,
        conversation_id=state.get("conversation_id"), user_id=state.get("user_id"),
        device_id=state.get("device_id"), persona=state.get("persona"),
        model_request=task.model_request,
        messages=[HumanMessage(content=task.objective)], history=[],
        state={"attachments": state.get("attachments") or []},
        extras={"worker": True, "dispatch_id": task.dispatch_id,
                "task_id": task.task_id, "parent_context": task.parent_context,
                "allowed_tool_ids": task.allowed_tool_ids,
                "hitl_policy": (state.get("context") or {}).get("hitl_policy", {}),
                "custom_agents": state.get("custom_agents") or {}},
    )
```

Resolve request-scoped custom definitions before compilation. Filter discovered tools to `allowed_tool_ids` and reject any runtime identity outside that set before approval. Keep objective/context out of system instructions.

At `PlanningWorkerRuntime.run()` and `SpecialistFactory.invoke_worker()`, use exception order `GraphBubbleUp` re-raise, model/tool limit mapping, timeout mapping, unavailable-agent mapping, then generic failure. Every failed result retains dispatch/task/position identity.

- [ ] **Step 5: Wire both RAG entry points to the same object, verify, and commit**

Construct `self.rag_execution_graph` once in `MultiAgentWorkflow.__init__`. Top-level RAG maps its result to `ResponseOutcome`; the RAG worker maps the same result to `WorkerResult`. Neither calls `.build()`.

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_execution_graph.py tests/test_rag_tool_loop_finalization.py tests/test_worker_rag_grounding.py tests/test_rag_grounding.py tests/test_planning_execution_graph.py tests/test_worker_approval_gated_tools.py tests/test_custom_agents_planning.py tests/test_specialist_tool_pipeline.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/rag_execution.py app/ai/workflow/specialists.py app/ai/workflow/planning_execution.py app/ai/workflow/runtime_context.py app/ai/graph.py
git add app/ai/workflow/rag_execution.py app/ai/workflow/specialists.py app/ai/workflow/planning_execution.py app/ai/workflow/runtime_context.py app/ai/graph.py tests/test_rag_execution_graph.py tests/test_rag_tool_loop_finalization.py tests/test_worker_rag_grounding.py tests/test_planning_execution_graph.py tests/test_worker_approval_gated_tools.py tests/test_custom_agents_planning.py
git commit -m "refactor: unify rag and planning worker execution"
```

---

### Task 3: Add durable mutation execution receipts

**Files:**
- Create: `app/models/tool_execution_receipt.py`
- Create: `app/repositories/tool_execution_receipt.py`
- Create: `app/services/tool_execution_receipt_service.py`
- Create: `app/alembic/versions/b8c9d0e1f2a3_add_tool_execution_receipts.py`
- Modify: `app/models/__init__.py`
- Modify: `app/ai/workflow/middleware.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/core/container.py`
- Create: `tests/test_tool_execution_receipt_service.py`
- Test: `tests/test_tool_execution_recovery.py`
- Create: `tests/integration/test_tool_execution_receipt_repository_postgres.py`

**Interfaces:**
- Consumes: checkpoint thread, dispatch/task/tool-call identities, qualified tool identity, execution policy, and provider idempotency capability.
- Produces: `ToolExecutionReceiptService.execute_mutation(scope, invoke) -> NormalizedToolResult` in the common tool pipeline.

- [ ] **Step 1: Write failing key/lifecycle/ownership tests**

```python
def test_execution_key_is_stable() -> None:
    scope = MutationExecutionScope(thread_id="routing-v2:c:t", dispatch_id="d",
                                   task_id="w", tool_call_id="call", tool_id="mcp::write",
                                   user_id=USER_ID, conversation_id=CONVERSATION_ID, turn_id="t")
    assert execution_key(scope) == execution_key(scope)
    assert len(execution_key(scope)) == 64


async def test_completed_receipt_returns_recorded_result(service) -> None:
    service.repository.completed_result = {"content": "created", "artifact_ref": "blob:1"}
    invoke = AsyncMock()
    assert (await service.execute_mutation(_scope(), invoke)).content == "created"
    invoke.assert_not_awaited()


async def test_reserved_non_idempotent_call_becomes_unknown(service) -> None:
    service.repository.status = ReceiptStatus.RESERVED
    with pytest.raises(MutationOutcomeUnknown):
        await service.execute_mutation(_scope(provider_idempotency=False), AsyncMock())
```

PostgreSQL tests cover unique keys, owner filtering, compare-and-set transitions, concurrent reservation, and bounded results.

- [ ] **Step 2: Run tests and confirm the service/table are absent**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_execution_receipt_service.py tests/test_tool_execution_recovery.py
```

Expected: import or fixture failures for the new receipt types.

- [ ] **Step 3: Create model, repository, and migration**

Use status enum `reserved | completed | failed | outcome_unknown`. Store a 64-character SHA-256 key, user/conversation/turn/thread owners, dispatch/task/tool-call IDs, qualified tool ID, provider-idempotency flag, bounded `JSONB` result or artifact reference, provider receipt ID, sanitized error code, and timezone-aware timestamps. Migration `b8c9d0e1f2a3` has `down_revision = "a7b8c9d0e1f2"`, one unique key, and owner/status indexes.

```python
class ReceiptStatus(str, Enum):
    RESERVED = "reserved"
    COMPLETED = "completed"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"


class MutationOutcomeUnknown(RuntimeError):
    def __init__(self, execution_key: str) -> None:
        super().__init__("mutation_outcome_unknown")
        self.execution_key = execution_key
```

- [ ] **Step 4: Implement the state machine and common middleware hook**

```python
class MutationExecutionScope(BaseModel):
    thread_id: str
    dispatch_id: str
    task_id: str
    tool_call_id: str
    tool_id: str
    user_id: UUID
    conversation_id: UUID
    turn_id: str
    provider_idempotency: bool = False


def execution_key(scope: MutationExecutionScope) -> str:
    raw = "\x1f".join((scope.thread_id, scope.dispatch_id, scope.task_id,
                         scope.tool_call_id))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
```

Atomically reserve; return a completed stored result; retry `reserved` only through a provider-idempotent adapter using the same key; otherwise transition to `outcome_unknown`. Application-owned mutations commit completion in the same SQL transaction. Insert this after authorization and approval, before mutation invocation. Runtime/config metadata—not model arguments—supplies identities. Top-level calls use dispatch `top-level` and task equal to active specialist. Strip keys and provider receipts from model messages, artifacts, logs, and streams.

- [ ] **Step 5: Run unit/PostgreSQL/migration gates and commit**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_tool_execution_receipt_service.py tests/test_tool_execution_recovery.py tests/test_worker_approval_gated_tools.py
.\.venv\Scripts\python.exe -m pytest -q tests/integration/test_tool_execution_receipt_repository_postgres.py
.\.venv\Scripts\python.exe -m alembic check
.\.venv\Scripts\python.exe -m ruff check app/models/tool_execution_receipt.py app/repositories/tool_execution_receipt.py app/services/tool_execution_receipt_service.py app/ai/workflow/middleware.py app/ai/tool_execution.py app/core/container.py
git add app/models/tool_execution_receipt.py app/repositories/tool_execution_receipt.py app/services/tool_execution_receipt_service.py app/alembic/versions/b8c9d0e1f2a3_add_tool_execution_receipts.py app/models/__init__.py app/ai/workflow/middleware.py app/ai/tool_execution.py app/core/container.py tests/test_tool_execution_receipt_service.py tests/test_tool_execution_recovery.py tests/integration/test_tool_execution_receipt_repository_postgres.py
git commit -m "feat: make mutation retries receipt backed"
```

---

### Task 4: Lift Planning fan-out into the outer production graph

**Files:**
- Modify: `app/ai/workflow/planning_execution.py`
- Modify: `app/ai/workflow/graph_builder.py:160`
- Modify: `app/ai/workflow/runtime_context.py`
- Modify: `app/ai/workflow/inventory.py`
- Modify: `app/ai/graph.py:700`
- Test: `tests/test_planning_worker_fanout.py`
- Test: `tests/test_planning_execution_graph.py`
- Test: `tests/test_production_workflow_graph.py`
- Test: `tests/test_worker_approval_gated_tools.py`
- Create: `tests/integration/test_planning_worker_resume_postgres.py`

**Interfaces:**
- Consumes: Tasks 1-3, current Planning model/rubric/todo actions, transition resolver, and outer checkpointer.
- Produces: real outer nodes `planning_model`, `planning_dispatch`, `planning_worker`, `planning_collect`, `planning_actions`, and `planning_package`.

- [ ] **Step 1: Write the actual replay regression and topology assertions**

```python
first = await graph.ainvoke(_planning_turn(), config=_config())
assert side_effects == ["w1"]
snapshot = await graph.aget_state(_config())
assert len(snapshot.interrupts) == 1

await graph.ainvoke(Command(resume=_approve(snapshot.interrupts[0])), config=_config())
assert side_effects == ["w1", "w2"]
assert _result_keys(await graph.aget_state(_config())) == {("d1", "w1"), ("d1", "w2")}
```

Assert every required Planning node exists and `planning_tools` does not. Repeat restart/resume with a new workflow instance and production PostgreSQL saver.

- [ ] **Step 2: Run regression and observe the duplicate worker**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_planning_worker_fanout.py tests/test_production_workflow_graph.py tests/test_worker_approval_gated_tools.py
```

Expected: old topology produces `w1, w1, w2` or lacks the required nodes.

- [ ] **Step 3: Implement `PlanningNodeFactory`**

Register the exact node methods through one inspectable descriptor table:

```python
def descriptors(self) -> tuple[tuple[str, Callable[..., Any], tuple[str, ...]], ...]:
    return (
        ("planning_model", self.planning_model,
         ("planning_dispatch", "planning_actions", "planning_package",
          "resolve_transition", "finalize")),
        ("planning_dispatch", self.planning_dispatch, ("planning_worker",)),
        ("planning_worker", self.planning_worker, ("planning_collect",)),
        ("planning_collect", self.planning_collect, ("planning_model", "finalize")),
        ("planning_actions", self.planning_actions, ("planning_model", "finalize")),
        ("planning_package", self.planning_package, ("validate_output", "finalize")),
    )
```

The methods have these call contracts: `planning_model(state, runtime) ->
Command`, `planning_dispatch(state) -> list[Send]`,
`planning_worker(payload, writer) -> dict[str, Any]`,
`planning_collect(state) -> Command`, `planning_actions(state) -> Command`, and
`planning_package(state) -> Command`.

`planning_model` binds `dispatch_subagents` only as a non-executable schema with `write_todos` and `hand_off`. Route final text to package, todos to actions, exclusive handoff to `resolve_transition`, and validated dispatch to `planning_dispatch`. Invalid calls append deterministic paired feedback and return to `planning_model`.

`planning_dispatch` returns `Send("planning_worker", {"worker_task": task, "worker_parent_state": bounded_scope})` for validated tasks. It never invokes/compiles a graph. The outer invocation supplies `max_concurrency=4` from settings. `planning_collect` selects the current dispatch, sorts by position, appends one `ToolMessage` paired to the original call, clears pending dispatch, and returns to the model. Preserve plan create/modify/review behavior, rubric feedback, todos, lifecycle metadata, model overrides, and the two-wave counters in `planning_actions` and `planning_model`. `planning_package` aggregates server-owned worker evidence/artifacts/images into `OutcomeProvenance`, selects `rag_grounding` whenever evidence is present, creates `ResponseOutcome`, and goes to `validate_output` so synthesis is revalidated before publication.

- [ ] **Step 4: Register nodes directly in `build_workflow_graph()`**

Map public agent ID `planning_agent` to graph node `planning_model` while retaining `active_agent_id="planning_agent"`. Add all six nodes with declared dynamic destinations and no static outgoing edge. Remove parent `planning_tools`. Preserve `finalize -> END` as the sole terminal edge.

- [ ] **Step 5: Run memory/PostgreSQL replay gates and commit**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_planning_worker_fanout.py tests/test_planning_execution_graph.py tests/test_production_workflow_graph.py tests/test_worker_approval_gated_tools.py
.\.venv\Scripts\python.exe -m pytest -q tests/integration/test_planning_worker_resume_postgres.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/planning_execution.py app/ai/workflow/graph_builder.py app/ai/workflow/runtime_context.py app/ai/workflow/inventory.py app/ai/graph.py
git add app/ai/workflow/planning_execution.py app/ai/workflow/graph_builder.py app/ai/workflow/runtime_context.py app/ai/workflow/inventory.py app/ai/graph.py tests/test_planning_worker_fanout.py tests/test_planning_execution_graph.py tests/test_production_workflow_graph.py tests/test_worker_approval_gated_tools.py tests/integration/test_planning_worker_resume_postgres.py
git commit -m "fix: move planning fanout into parent topology"
```

---

### Task 5: Make pending interrupts and graph writers authoritative

**Files:**
- Modify: `app/ai/graph.py:126`
- Modify: `app/ai/utils.py:513`
- Modify: `app/ai/hitl_config.py:377`
- Modify: `app/services/message_service.py:1700`
- Modify: `app/ai/workflow/planning_execution.py`
- Modify: `app/ai/workflow/rag_execution.py`
- Modify: `app/services/event_streaming/graph_public_projection.py`
- Modify: `app/services/event_streaming/subagents.py`
- Modify: `app/services/event_streaming/langchain_v3.py`
- Test: `tests/test_hitl_interrupt_payload_recovery.py`
- Test: `tests/test_interrupt_resume_addressing.py`
- Test: `tests/test_graph_stream_projection.py`
- Test: `tests/test_event_streaming_subagents.py`

**Interfaces:**
- Consumes: checkpoint `snapshot.tasks[*].interrupts`, exact interrupt IDs, injected `StreamWriter`, and dispatch/task/tool lifecycle IDs.
- Produces: `pending_interrupt_payload(snapshot) -> PendingInterruptPayload | None` and typed custom worker events.

- [ ] **Step 1: Write failing worker-private and parallel-interrupt tests**

```python
def test_worker_interrupt_needs_no_parent_ai_tool_call(workflow) -> None:
    snapshot = _snapshot(parent_messages=[HumanMessage("run")],
                         interrupts=[_interrupt("i1", "c1")])
    response = workflow._build_interrupt_agent_response(snapshot, "routing-v2:c:t")
    assert response.metadata["interrupt"]["action_requests"][0]["tool_call_id"] == "c1"


def test_parallel_metadata_is_keyed_per_tool_call(workflow) -> None:
    payload = pending_interrupt_payload(
        _snapshot(interrupts=[_interrupt("i1", "c1"), _interrupt("i2", "c2")]))
    assert set(payload.metadata_by_tool_call_id) == {"c1", "c2"}
```

Cover approve/edit/reject/respond by exact interrupt ID, unanswered interrupts remaining pending, interleaved worker tool events, and no internal message deltas.

- [ ] **Step 2: Run HITL/stream tests and observe current gating/sink failures**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_hitl_interrupt_payload_recovery.py tests/test_interrupt_resume_addressing.py tests/test_graph_stream_projection.py tests/test_event_streaming_subagents.py tests/test_graph_planning_subagents.py
```

Expected: worker-private pauses are not recovered, metadata overwrites, or events depend on legacy `planning_tools`/sink plumbing.

- [ ] **Step 3: Replace pause heuristics with one normalizer**

Delete `_APPROVAL_INTERRUPT_NODES` and `_has_approval_interrupt`.

```python
class PendingInterruptPayload(BaseModel):
    action_requests: tuple[dict[str, JsonValue], ...]
    interrupt_ids: tuple[str, ...]
    metadata_by_tool_call_id: dict[str, dict[str, JsonValue]]


def pending_interrupt_payload(snapshot: Any) -> PendingInterruptPayload | None:
    """Normalize every live pending interrupt without consulting parent messages."""
```

Use it in non-stream response, stream interrupt event, `get_state()`, and resume validation. Merge provenance beneath each tool-call ID instead of `metadata.update`. Build `Command(resume={item.interrupt_id: item.decision for item in accepted_decisions})`; reject unknown, duplicate, resolved, or cross-user IDs and leave omitted IDs pending.

- [ ] **Step 4: Emit and project typed custom events**

Planning nodes receive an injected writer and emit:

```python
writer({"type": "planning_worker", "phase": "start",
        "dispatch_id": task.dispatch_id, "task_id": task.task_id,
        "agent_id": task.agent_id})
```

Emit dispatch validated, worker start/end, worker tool start/end, interrupt, collect, and wave complete. RAG worker events carry the same task IDs. Exclude objectives, execution keys, provider receipts, document content, and credentials. Project these into existing subagent events, filter internal nested namespaces from answer deltas, and derive Planning completion from `planning_package`.

- [ ] **Step 5: Run HITL/stream suites and commit**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_hitl_interrupt_payload_recovery.py tests/test_interrupt_resume_addressing.py tests/test_graph_resume_image_stream.py tests/test_hitl_backend_regressions.py tests/test_graph_stream_projection.py tests/test_event_streaming_subagents.py tests/test_graph_planning_subagents.py tests/test_message_service_subagent_streaming.py tests/test_message_service_event_streaming.py
.\.venv\Scripts\python.exe -m ruff check app/ai/graph.py app/ai/utils.py app/ai/hitl_config.py app/services/message_service.py app/ai/workflow/planning_execution.py app/services/event_streaming/graph_public_projection.py
git add app/ai/graph.py app/ai/utils.py app/ai/hitl_config.py app/services/message_service.py app/ai/workflow/planning_execution.py app/ai/workflow/rag_execution.py app/services/event_streaming/graph_public_projection.py app/services/event_streaming/subagents.py app/services/event_streaming/langchain_v3.py tests/test_hitl_interrupt_payload_recovery.py tests/test_interrupt_resume_addressing.py tests/test_graph_stream_projection.py tests/test_event_streaming_subagents.py tests/test_graph_planning_subagents.py
git commit -m "fix: recover and stream planning interrupts natively"
```

---

### Task 6: Delete every superseded Planning and RAG path

**Files:**
- Delete: `app/ai/workflow/planning_loop.py`
- Delete: `app/ai/workflow/rag_loop.py`
- Delete: `app/ai/planning_subagents.py`
- Modify: `app/ai/graph.py`
- Modify: `app/ai/workflow/graph_builder.py`
- Modify: `app/ai/workflow/__init__.py`
- Modify: `app/ai/agents/planning_agent.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/tool_execution_policy.py`
- Modify: `app/core/container.py`
- Test: `tests/test_routing_legacy_removal.py`
- Test: `tests/test_rag_dead_code_cleanup.py`
- Test: `tests/test_production_workflow_graph.py`

**Interfaces:**
- Consumes: production replacements from Tasks 2-5.
- Produces: one legacy-free graph and dependency graph.

- [ ] **Step 1: Tighten deletion tests**

Assert no application matches for:

```python
FORBIDDEN = (
    "planning_tools", "rag_tools", "PlanningSubagentDispatcher",
    "create_dispatch_subagents_tool", "_run_agent_in_isolated_context",
    "_refuse_worker_approval_gated_calls", "_fan_out_graph",
    "_has_approval_interrupt", "_APPROVAL_INTERRUPT_NODES",
    "event_sink_token", "WeakValueDictionary", "_recover_terminal_response",
)
```

Allow `dispatch_subagents` only in the new control schema, Planning prompt, typed event renderer, and tests. Fail if the common tool pipeline or execution-policy allowlist can resolve it.

- [ ] **Step 2: Run deletion tests and capture expected old matches**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_legacy_removal.py tests/test_rag_dead_code_cleanup.py tests/test_production_workflow_graph.py
```

- [ ] **Step 3: Delete modules and remove callers**

Delete the three files. Remove `PlanningLoopMixin`, `RagLoopMixin`, the 530-line isolated worker, both nested tool loops, worker approval refusal, dispatcher factories, duplicate task/result JSON models, the `dispatch_subagents` timeout/policy exemption, weak event-sink state, and stale terminal-response salvage. Remove constructor fields existing only for those branches.

Keep `PlanningAgent` only as Planning model/prompt/rubric support for `PlanningNodeFactory`. Keep `RAGAgent` only as retrieval/model/tool capabilities used by `RagExecutionGraph`; neither owns a second graph route.

- [ ] **Step 4: Finish production dependency wiring**

Construct receipt service, shared RAG graph, and Planning node factory before `build_workflow_graph()`. Assert `rag_agent` maps to the shared RAG wrapper, `planning_agent` maps to `planning_model`, no parent RAG/Planning tool stage exists, and `finalize -> END` is the only terminal edge.

- [ ] **Step 5: Run searches/tests and commit atomic deletion**

```powershell
rg -n "planning_tools|rag_tools|PlanningSubagentDispatcher|create_dispatch_subagents_tool|_run_agent_in_isolated_context|_refuse_worker_approval_gated_calls|_fan_out_graph|_has_approval_interrupt|_APPROVAL_INTERRUPT_NODES|event_sink_token|WeakValueDictionary|_recover_terminal_response" app
rg -n "dispatch_subagents" app/ai/tool_execution.py app/ai/tool_execution_policy.py
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_legacy_removal.py tests/test_rag_dead_code_cleanup.py tests/test_production_workflow_graph.py tests/test_planning_worker_fanout.py tests/test_worker_approval_gated_tools.py tests/test_rag_execution_graph.py
git add -- app/ai/graph.py app/ai/workflow/graph_builder.py app/ai/workflow/__init__.py app/ai/agents/planning_agent.py app/ai/tool_execution.py app/ai/tool_execution_policy.py app/core/container.py tests/test_routing_legacy_removal.py tests/test_rag_dead_code_cleanup.py tests/test_production_workflow_graph.py
git add -u -- app/ai/workflow/planning_loop.py app/ai/workflow/rag_loop.py app/ai/planning_subagents.py
git commit -m "refactor: remove legacy planning and rag execution"
```

Expected: both searches return no matches and all focused tests pass.

---

### Task 7: Lock down restart, limits, crash gaps, and operations

**Files:**
- Modify: `tests/integration/test_planning_worker_resume_postgres.py`
- Modify: `tests/test_workflow_end_to_end.py`
- Modify: `tests/test_workflow_concurrency.py`
- Modify: `tests/test_workflow_checkpoint_v2.py`
- Modify: `tests/test_worker_approval_gated_tools.py`
- Modify: `tests/test_message_service_event_streaming.py`
- Modify: `docs/operations/tool-execution-policy.md`
- Modify: `docs/operations/routing-v2-rollout.md`

**Interfaces:**
- Consumes: complete production graph, PostgreSQL checkpointer, durable receipts, HITL repository, and message persistence.
- Produces: release-blocking end-to-end coverage and cutover runbook.

- [ ] **Step 1: Complete the restart/resume production scenario**

Workflow A dispatches `w1` and approval-gated `w2`; verify `w1` completion and one durable interrupt. Discard A, construct workflow B, approve the exact ID on the same thread, and verify side effects exactly `w1, w2`, two unique result/receipt identities, and one routing call.

- [ ] **Step 2: Add atomic limit and two-wave scenarios**

```python
@pytest.mark.parametrize("invalid", ["ninth_task", "duplicate_id", "oversized_objective",
                                      "recursive_planning", "detached_custom_agent",
                                      "dispatch_plus_handoff"])
async def test_invalid_dispatch_starts_no_workers(invalid, workflow) -> None:
    result = await workflow.ainvoke(_turn_with_invalid_dispatch(invalid))
    assert result["worker_results"] == []
    assert _paired_control_error(result)


async def test_third_dispatch_wave_is_rejected(workflow) -> None:
    result = await workflow.ainvoke(_three_wave_turn())
    assert result["planning_dispatch_waves"] == 2
    assert _control_error_code(result) == "dispatch_wave_limit"
```

Assert no more than four live workers, eight total tasks across waves, and position-ordered collection.

- [ ] **Step 3: Add multi-interrupt and receipt crash-gap scenarios**

Pause two workers; resolve one exact ID; restart; resolve the other. Verify separate provenance. Simulate external non-idempotent success followed by process loss before checkpoint persistence: status becomes `outcome_unknown` and provider invocation is not repeated. For a provider-idempotent adapter, assert retry uses the same execution key.

- [ ] **Step 4: Update operational documentation**

Document exact topology/limits, migration order, `outcome_unknown` reconciliation, canary metrics, new-checkpoint verification, and rollback by deploying the prior artifact. State that rollback is not a runtime switch and v1 interrupts do not resume in v2.

- [ ] **Step 5: Run production scenarios and commit**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_end_to_end.py tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_worker_approval_gated_tools.py tests/test_message_service_event_streaming.py tests/integration/test_planning_worker_resume_postgres.py tests/integration/test_tool_execution_receipt_repository_postgres.py
.\.venv\Scripts\python.exe -m ruff check tests/test_workflow_end_to_end.py tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/integration/test_planning_worker_resume_postgres.py
git add tests/test_workflow_end_to_end.py tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_worker_approval_gated_tools.py tests/test_message_service_event_streaming.py tests/integration/test_planning_worker_resume_postgres.py docs/operations/tool-execution-policy.md docs/operations/routing-v2-rollout.md
git commit -m "test: lock down planning resume cutover"
```

---

### Task 8: Build the versioned multilingual routing release gate

**Files:**
- Create: `app/evaluation/routing/__init__.py`
- Create: `app/evaluation/routing/contracts.py`
- Create: `app/evaluation/routing/metrics.py`
- Create: `app/evaluation/routing/release_gate.py`
- Create: `scripts/evaluate_routing.py`
- Create: `scripts/check_routing_release.py`
- Create: `eval/routing/golden_v1.jsonl`
- Create: `eval/routing/golden_v1.review.json`
- Create: `tests/test_routing_evaluation_contracts.py`
- Create: `tests/test_routing_release_gate.py`

**Interfaces:**
- Consumes: production `RoutingService`, exact provider/model/inventory tuple, and the human-reviewed dataset hash.
- Produces: a versioned JSON report and a fail-closed release decision used by Task 9.

- [ ] **Step 1: Write failing dataset/report/release tests**

```python
def test_dataset_requires_primary_inside_acceptable_set() -> None:
    with pytest.raises(ValidationError):
        RoutingEvalCase(case_id="thai-001", language="th", category="planning",
                        message="ช่วยวางแผนงาน", primary_agent_id="planning_agent",
                        acceptable_agent_ids=("chat_agent",))


def test_release_rejects_stale_or_mismatched_tuple(report, review) -> None:
    report.provider = "different-provider"
    decision = check_routing_release(report=report, review=review,
                                     expected_provider="configured-provider",
                                     expected_model="configured-model",
                                     expected_inventory_version="inventory-v1")
    assert decision.passed is False
    assert "provider_mismatch" in decision.reason_codes
```

Cover missing review, unapproved review, dataset hash mismatch, fewer than 210 cases, fewer than 30 cases in any required language, fewer than 10 cases in a category, stale report, metric threshold failures, and any silent chat substitution/finalizer bypass/unknown evidence ID.

- [ ] **Step 2: Run tests and confirm evaluation modules are absent**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_evaluation_contracts.py tests/test_routing_release_gate.py
```

Expected: import failures for `app.evaluation.routing`.

- [ ] **Step 3: Implement exact evaluation contracts and metrics**

```python
class RoutingPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    predicted_agent_id: str | None
    attempts: int = Field(ge=1, le=2)
    structured_success: bool
    error_code: str | None = None


class RoutingDatasetReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_sha256: str
    approved: bool
    reviewing_team: str
    reviewed_at: datetime
    label_guideline_version: str


class RoutingReleaseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    passed: bool
    reason_codes: tuple[str, ...] = ()


class RoutingEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    language: Literal["en", "th", "vi", "zh", "ja", "ar", "mixed"]
    category: str
    message: str
    context: dict[str, JsonValue] = Field(default_factory=dict)
    primary_agent_id: str
    acceptable_agent_ids: tuple[str, ...]


class RoutingEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_sha256: str
    provider: str
    model: str
    inventory_version: str
    generated_at: datetime
    case_count: int
    macro_f1: float
    accuracy_by_language: dict[str, float]
    first_attempt_structured_success: float
    after_retry_structured_success: float
    silent_chat_substitutions: int
    finalizer_bypasses: int
    unknown_published_evidence_ids: int
    predictions: tuple[RoutingPrediction, ...]
```

Compute acceptable-set accuracy separately from canonical-label precision/recall/F1 and confusion matrix. Count every case once under `primary_agent_id`; do not double-count acceptable alternatives.

- [ ] **Step 4: Author and independently review the v1 dataset**

Create at least 210 JSONL cases: at least 30 each for English, Thai, Vietnamese, Chinese, Japanese, Arabic, and mixed-language input; every intent category appears at least 10 times. Include ambiguous follow-ups, canvas, document/RAG, planning, custom-agent, image, current-information/search, and general chat. The review manifest contains exactly `dataset_sha256`, `approved`, `reviewing_team`, `reviewed_at`, and `label_guideline_version`. A reviewer other than the dataset generator verifies labels and sets `approved=true`; the checker fails closed until that happens.

- [ ] **Step 5: Implement evaluator and release checker CLIs**

`evaluate_routing.py` loads the validated JSONL, invokes the production `RoutingService` without provider fallback, records first/second structured attempts, and writes `RoutingEvaluationReport`. `check_routing_release.py` verifies review approval/hash, exact provider/model/inventory tuple, report freshness, dataset composition, and these thresholds: macro-F1 `>= 0.90`; every language accuracy `>= 0.85`; each language no more than `0.05` below English; first-attempt structured success `>= 0.99`; after-retry success `>= 0.999`; and all three violation counters equal zero.

- [ ] **Step 6: Run deterministic evaluation tests and commit**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_evaluation_contracts.py tests/test_routing_release_gate.py
.\.venv\Scripts\python.exe -m ruff check app/evaluation/routing scripts/evaluate_routing.py scripts/check_routing_release.py tests/test_routing_evaluation_contracts.py tests/test_routing_release_gate.py
git add app/evaluation/routing scripts/evaluate_routing.py scripts/check_routing_release.py eval/routing/golden_v1.jsonl eval/routing/golden_v1.review.json tests/test_routing_evaluation_contracts.py tests/test_routing_release_gate.py
git commit -m "test: add multilingual routing release gate"
```

---

### Task 9: Run the final production acceptance gate

**Files:**
- Modify only when verification identifies an in-scope defect.

**Interfaces:**
- Consumes: Tasks 1-8.
- Produces: evidence required to call the root problem fixed.

- [ ] **Step 1: Verify schema and graph shape**

```powershell
.\.venv\Scripts\python.exe -m alembic heads
.\.venv\Scripts\python.exe -m alembic check
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_production_workflow_graph.py tests/test_routing_legacy_removal.py
```

Expected: one Alembic head, no drift, required Planning nodes present, forbidden nodes absent, and only `finalize -> END` terminates.

- [ ] **Step 2: Run formatting and lint**

```powershell
.\.venv\Scripts\python.exe -m ruff format --check app tests
.\.venv\Scripts\python.exe -m ruff check app tests
```

- [ ] **Step 3: Run the focused routing/Planning/RAG/HITL/streaming gate**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_contracts.py tests/test_workflow_state.py tests/test_checkpoint_serializer.py tests/test_production_workflow_graph.py tests/test_planning_execution_graph.py tests/test_planning_worker_fanout.py tests/test_worker_approval_gated_tools.py tests/test_rag_execution_graph.py tests/test_worker_rag_grounding.py tests/test_rag_grounding.py tests/test_hitl_interrupt_payload_recovery.py tests/test_interrupt_resume_addressing.py tests/test_graph_stream_projection.py tests/test_message_service_event_streaming.py tests/test_workflow_end_to_end.py tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_routing_legacy_removal.py
```

- [ ] **Step 4: Run PostgreSQL integration gates**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/integration/test_planning_worker_resume_postgres.py tests/integration/test_tool_execution_receipt_repository_postgres.py tests/test_checkpoint_retention_v2_threads.py
```

- [ ] **Step 5: Run the full non-live suite**

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q -m "not live_provider and not routing_live"
```

Expected: all pass. A Windows native `pyarrow` diagnostic is non-failing only when pytest exits zero; otherwise rerun in Linux CI/container and do not accept the gate until it passes there.

- [ ] **Step 6: Search for legacy and secret leakage**

```powershell
rg -n "planning_tools|rag_tools|PlanningSubagentDispatcher|create_dispatch_subagents_tool|_run_agent_in_isolated_context|_refuse_worker_approval_gated_calls|_fan_out_graph|_has_approval_interrupt|_APPROVAL_INTERRUPT_NODES|event_sink_token|WeakValueDictionary|_recover_terminal_response" app
rg -n "execution_key|provider_receipt" app/services/event_streaming app/ai/tool_result_rendering.py
git diff --check
git status --short
```

Expected: no matches and no whitespace errors.

- [ ] **Step 7: Run the tuple-matched live routing release gate**

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_routing.py --dataset eval/routing/golden_v1.jsonl --output .artifacts/routing-eval-v1.json
.\.venv\Scripts\python.exe scripts/check_routing_release.py --dataset eval/routing/golden_v1.jsonl --review eval/routing/golden_v1.review.json --report .artifacts/routing-eval-v1.json
```

Expected: macro-F1 at least 0.90; each language at least 0.85; no language over 0.05 below English; first-attempt structured success at least 0.99; after-retry success at least 0.999; zero silent chat substitutions, finalizer bypasses, and unknown published evidence IDs. Do not commit `.artifacts` containing request text.

- [ ] **Step 8: Review final diff and record evidence**

```powershell
git status --short
git log --oneline --decorate -12
git diff 35e4b7b..HEAD --stat
git diff 35e4b7b..HEAD --check
```

Record commands, exit codes, and test counts in the execution log. Commit only defects uncovered by verification; do not create an empty verification commit.

## Final Acceptance Checklist

- [ ] Parent-level Planning nodes own fan-out; no executable/nested dispatch remains.
- [ ] Memory and PostgreSQL replay tests produce `w1, w2`, never `w1, w1, w2`.
- [ ] Invalid dispatches start zero workers; turn limits are 8 tasks, 4 concurrent, and 2 waves.
- [ ] Worker objective, HITL policy, custom-agent snapshot, attachments, model request, and restricted tools reach one worker path.
- [ ] LangGraph control-flow exceptions are never normalized as failures.
- [ ] Top-level/worker RAG share one graph, preserve provenance, and validate every result without regenerating or abstaining.
- [ ] Pending interrupts drive non-stream, stream, state, and resume without message/node heuristics.
- [ ] Parallel interrupts retain distinct IDs and provenance.
- [ ] Receipts prevent completed mutation replay and expose unsupported crash gaps.
- [ ] Worker events are task-correlated and cannot leak internal answer text.
- [ ] Legacy loops, dispatcher, isolated worker, refusal, sink registry, stale recovery, and policy exemption are deleted.
- [ ] Only `finalize` reaches `END`; validation/persistence precede public deltas.
- [ ] Focused, PostgreSQL, non-live, lint, migration, removal, and live gates pass.

## Execution Log

| Date | Task | Result | Evidence |
|---|---|---|---|
| 2026-08-28 | Plan rewrite | Complete | Approved root-fix design translated into atomic Tasks 1-8; implementation has not started. |
| 2026-09-03 | Task 1 | Complete | `a45cf03`. Worker identity is `(dispatch_id, task_id)`; `WorkerTaskProposal`/`DispatchSubagentsInput`/`WorkerTask`/`PlanningDispatch` added; `validate_dispatch_call` validates whole proposals and rejects rather than truncates; `planning_worker_max_dispatch_waves=2`. 67 focused tests pass, ruff clean. |
| 2026-09-03 | Plan amendment | Complete | Grounding constraint, Task 2 Step 1/3, and checklist item 6 amended to record-only validation. The regenerate-once-then-abstain rule predated the 2026-08-27 streaming decision (`553eebb`) and is guarded against by `test_stranded_enforcement_machinery_has_no_production_caller`. Confirmed with Thai before amending. |
| 2026-09-03 | Task 2 | Complete | One `RagExecutionGraph` compiled once per workflow (`compile_count == 1`), model-driven tool loop (`rag_model -> rag_tools -> collect_rag_outputs -> rag_model`), invocation-scoped evidence allocator on the run config. `PlanningWorkerRuntime.run(task, state, writer)` replaces `PlanningOrchestrator`; `build_worker_request` is the single place objective/HITL policy/custom agents/attachments/model request/tool scope are assembled. `WorkerToolScopeMiddleware` refuses out-of-scope calls *before* approval (composed last so its `after_model` runs first). `GraphBubbleUp` re-raised ahead of every normalization. |
| 2026-09-03 | Task 3 | Complete | `tool_execution_receipts` model/repository/service + migration `b8c9d0e1f2a3` (down_revision `a7b8c9d0e1f2`, single head), **applied to the live database** with a verified downgrade/upgrade round trip. `execution_key` = SHA-256 of thread/dispatch/task/tool-call joined by a reserved 0x1F separator; `provider_idempotency` deliberately excluded from identity. Receipts hook into `ToolExecutionMiddleware.awrap_tool_call` — after authorization and approval, before invocation. Every repository transition is a compare-and-set; reservation relies on the unique index, not a read-then-write. 16 service tests pass, and the 8 PostgreSQL integration tests pass against a reachable database. |
| 2026-09-03 | Task 3 defects found by verification | Fixed | (1) `op.create_table` auto-creates any enum type a column references with no `checkfirst`, so the explicit `ENUM.create(checkfirst=True)` plus a `create_type=True` column made CREATE TYPE run twice and the upgrade abort on `DuplicateObject` — fixed with `create_type=False` on the column's ENUM. (2) The model declared `unique=True` (a UNIQUE *constraint*) while the migration created a unique *index*; autogenerate reports that as permanent drift, so both now declare the index. (3) The async PostgreSQL tests needed `pytest.mark.selector_event_loop` — psycopg raises `InterfaceError` on Windows' default ProactorEventLoop. |
| 2026-09-03 | Schema drift reconciled | Complete | `alembic check` now reports "No new upgrade operations detected." Four items fixed: `agent_model_configs.reasoning_effort` declared `String(32)` to match migration `e8f9a0b1c2d3` (the model had drifted to `Text`); `idx_document_chunks_index_generation_id` and the `idx_document_chunks_content_simple_fts` GIN expression index declared on `DocumentChunk` to match migrations `c3d4e5f6a7b8` / `d4e5f6a7b8c9`; and `ix_chat_images_sha256` dropped by new migration `c9d0e1f2a3b4`. That index existed live but no migration ever created it — it came from a `create_all` against the models — and `pg_stat_user_indexes` showed 0 scans against 3217 on `ix_chat_images_user_id`, so it was carrying write cost for nothing. Head is now `c9d0e1f2a3b4`; downgrade/upgrade round trip verified. Task 9 Step 1's clean-check expectation is now satisfiable. |
| 2026-09-03 | Integration-test database guard | Complete | `tests/integration/conftest.py` now fails collection when `TEST_DATABASE_URL` resolves to the same host/port/database as the app's `DATABASE_URL`. Every module there runs `Base.metadata.create_all`, which builds schema from the models and bypasses Alembic — the direct cause of the stray `ix_chat_images_sha256`. Verified: it raises against the app DB and stays silent when unset. The correct target is the pre-existing `chatbot_test` database. |
| 2026-09-03 | Pre-existing failures in test_model_usage_repository_postgres.py | Recorded, out of scope | Against a *clean dedicated* database this module has 3 failures + 1 error at `1f3debf` with all local changes stashed, so they are not caused by this refactor and not explained by live data. They are test-isolation defects in that module: `test_reconcile_minute_rebuilds_exactly_from_raw_events` **passes in isolation** and fails in-module because sibling tests seed rows into the same minute bucket, and `test_latest_conversation_context_uses_visible_assistant_metadata_not_helper_event` errors in teardown with `ForeignKeyViolation` — it deletes a conversation while its messages still reference it. Nothing in the model-usage ledger is touched by this plan; fixing them is a separate change. |

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-26-production-routing-refactor.md`.

Two execution options:

1. **Subagent-Driven (recommended):** Use `superpowers:subagent-driven-development`, one implementation task at a time with review gates.
2. **Inline execution:** Use `superpowers:executing-plans` in this session, sequentially with the listed verification checkpoints.
