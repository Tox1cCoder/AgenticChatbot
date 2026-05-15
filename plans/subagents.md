# Implementation Plan: Planning-Mode Subagents

Branch: current working tree | Date: 2026-05-08 | Spec: this document | Input: user request to add Planning-mode-only subagents that inherit existing graph-agent capabilities and run synchronously in the current chat.

## Summary

Add a Planning-mode-only subagent feature that lets the `planning_agent` delegate independent work to isolated worker agents, run those workers in parallel inside the same chat turn, wait for their results, and then continue as the supervisor that reconciles the plan. The implementation should follow LangChain's subagent pattern: a supervisor invokes subagents through a tool, subagents run with isolated context, and only model-safe outputs return to the supervisor.

The feature should not replace the current top-level router or `hand_off` flow. The current orchestrator remains responsible for selecting the top-level agent. Subagents become an internal Planning execution primitive, available only when Planning mode is active and the plan lifecycle is in execution.

## Technical Context

Language/Version: Python 3.10+.

Primary Dependencies: FastAPI service layer, LangGraph `StateGraph`, LangChain tool schemas, Pydantic models, existing Gemini/OpenAI/Anthropic runtime model resolution, existing MCP/client tool execution.

Storage: PostgreSQL-backed conversations and task plans remain unchanged for the first implementation. Subagent runs are same-turn execution artifacts, not persisted background jobs.

Testing: `pytest` with focused unit tests for schema validation, dispatcher concurrency, graph binding, planning loop behavior, and regression coverage for ordered normal tool execution.

Target Platform: Existing backend service under `app/`, used by the current Streamlit/client backend paths.

Project Type: Python web service with LangGraph-based agent orchestration.

Performance Goals:

- Dispatch independent subagent tasks concurrently in the current chat turn.
- Preserve deterministic result ordering.
- Avoid background work and avoid returning before subagent work finishes.
- Avoid broad prompt or full-tool-schema bloat in the Planning Agent.

Constraints:

- Subagents work only in the active chat turn.
- Subagents are available only to `planning_agent`.
- Subagents are available only when `planning_mode_enabled` is true. Prompt policy controls whether a planning-phase turn should use dispatch or only update the plan.
- The Planning Agent remains the only actor allowed to mutate `todos` through `write_todos`.
- The implementation must not make generic `execute_tool_calls()` parallel by default because ordered tool calls such as `tool_search` followed by a newly loaded tool depend on sequential execution and tool-map refresh.

Scale/Scope:

- Initial static worker registry: `chat_agent`, `rag_agent`, `search_agent`, `image_generator_agent`, and `canvas_agent`.
- Explicitly exclude `planning_agent` as a subagent target to prevent recursive planning supervisors.
- No subagent-specific task-count, worker-timeout, worker-iteration, or parallelism config caps. Existing provider/tool timeouts, request cancellation, HITL gating, and generic tool-result size controls remain the production safety boundary.

## Constitution Check

No `.specify/memory/constitution.md` exists in this repository. Apply the repository's current engineering constraints instead.

Simplicity Gate:

- Pass: one new focused runtime module, targeted graph/agent integration, and tests.
- Pass: no new worker queue, Celery job type, database table, or websocket protocol change.
- Pass: no speculative dynamic subagent registry in the first implementation.

Anti-Abstraction Gate:

- Pass: use existing `BaseAgent`, `PlanningAgent`, `MultiAgentWorkflow`, `execute_tool_calls`, model resolution, and tool context paths directly.
- Pass: introduce only the boundary required for isolated subagent execution.
- Pass: avoid reimplementing the entire graph. Extract reusable helper behavior where duplication would otherwise be high.

Integration-First Gate:

- Pass: contract-like tests define tool input/output shape before implementation.
- Pass: integration tests exercise Planning Agent binding and graph state behavior.
- Pass: concurrency tests prove real overlap without requiring external providers.

Security and Safety Gate:

- Pass: no background jobs and no cross-conversation execution.
- Pass: carry `conversation_id`, `user_id`, and `device_id` through existing scoped tool binding.
- Pass: request-scoped worker execution, underlying timeout/error normalization, generic tool-result sizing, and explicit failure reporting.
- Pass: no subagent direct user interaction; all output returns through the Planning Agent.

Re-check after design: still pass. The feature adds a request-scoped supervisor tool rather than another orchestrator, so it does not conflict with the existing multi-agent graph.

## Current Architecture Findings

- `app/ai/graph.py` defines the top-level LangGraph state machine. It routes one user turn to `chat_agent`, `rag_agent`, `search_agent`, `image_generator_agent`, `planning_agent`, or `canvas_agent`.
- `app/ai/agents/router.py` already receives `planning_mode_enabled` and `has_existing_plan`, and the router prompt prefers `planning_agent` for task-plan management.
- `app/ai/hand_off_tool.py` is graph-level delegation. It changes `selected_agent` after a tool result and continues with another top-level graph node. This is not subagent fan-out/fan-in.
- `app/ai/agents/planning_agent.py` always binds `write_todos` as an internal tool. It is the right supervisor for plan mutation.
- `app/ai/graph.py::_planning_node` invokes the Planning Agent with current todos and planning phase.
- `app/ai/graph.py::_planning_tools_node` separates `write_todos` calls from external calls. External calls run before `write_todos`, then `write_todos` mutates in-memory `todos`.
- `app/ai/tool_execution.py::execute_tool_calls` executes tool calls sequentially. This is correct for normal tools because `tool_search` may mutate the deferred tool map before later tool calls in the same batch.
- `app/services/message_service.py` prepares planning context from persisted tasks and later syncs response plan state back to `TaskPlanService`. Subagents should not write directly to `TaskPlanService`.
- `app/services/task_plan_service.py` owns persisted task lifecycle and has concurrency protections for task replacement. Subagent execution should keep using the existing response metadata sync path.

## Research Decisions

### Decision 1: Use a Single Planning Dispatch Tool

Use one internal tool, `dispatch_subagents`, rather than one tool per worker agent.

Rationale:

- The worker list is small and static, so an enum-constrained dispatch schema is explicit and type-safe.
- A single tool gives the Planning Agent one obvious delegation surface.
- It supports multi-worker fan-out in one tool call, which is a closer fit for parallel execution than many independent tool calls.
- It keeps the public graph unchanged.

Rejected alternative: reuse `hand_off`. It only routes to one top-level agent and does not isolate context or aggregate parallel worker results.

Rejected alternative: add static LangGraph subnodes. It improves state visibility, but it is heavier and unnecessary for the requested same-chat, Planning-only feature.

### Decision 2: Synchronous Same-Chat Execution

Subagent dispatch blocks until all selected workers complete, fail, time out, or require user action. It does not create background jobs.

Rationale:

- The user explicitly wants behavior like GPT Codex or Claude Code: parallel agent work within the current task, not offloaded work.
- The Planning Agent needs worker results before it can mark tasks complete or decide what to do next.
- This avoids job tables, job-status APIs, notifications, and stale task-plan state.

### Decision 3: Isolated Worker State, Shared Capabilities

Each worker runs in a fresh isolated execution context, but uses the existing graph agent's model configuration, tools, MCP/client runtime scoping, deferred tool search behavior, user/device identifiers, and conversation/document context.

Rationale:

- This matches the purpose of subagents: context isolation without losing capabilities.
- It prevents worker intermediate messages from bloating the main Planning Agent context.
- It lets the Planning Agent see only structured worker summaries.

Implementation implication:

- Do not call the compiled graph recursively for each worker.
- Add a workflow helper that can run a selected existing agent in an isolated mini-loop using existing agent/tool execution helpers.
- Keep subagent state in memory and return a structured JSON result through the `dispatch_subagents` tool.

### Decision 4: Keep Normal Tool Execution Sequential

Only the new dispatch tool introduces parallel execution internally. Do not parallelize arbitrary tool calls in `execute_tool_calls()`.

Rationale:

- Current deferred tool loading depends on ordered execution when `tool_search` appears before a later newly loaded tool.
- Some tools mutate scoped state or rely on shared session ordering.
- A special-purpose dispatcher can enforce safe worker isolation and deterministic aggregation without changing generic tool execution semantics.

### Decision 5: Planning Agent Owns Todo Mutations

Subagents never call `write_todos` and never update persisted tasks. They report outcomes; the Planning Agent calls `write_todos` after reading results.

Rationale:

- This keeps one write authority for plan state.
- It avoids races between parallel workers changing the same todo list.
- It preserves the existing `MessageService` plan-state sync path.

## Functional Requirements

FR-001: The subagent dispatch tool must be unavailable when Planning mode is inactive.

FR-002: During Planning Agent planning/editing phase, the dispatch tool may be bound for explicit user delegation/testing, but the prompt must prefer `write_todos` for normal plan creation/editing.

FR-003: The subagent dispatch tool must be available during Planning Agent execution phase.

FR-004: The dispatch tool must accept a non-empty list of independent worker tasks.

FR-005: Each worker task must choose one existing graph agent target from an enum: `chat_agent`, `rag_agent`, `search_agent`, `image_generator_agent`, or `canvas_agent`.

FR-006: The dispatch tool must reject `planning_agent` as a target.

FR-007: Worker tasks must run with isolated message state and must not append intermediate worker messages to the parent Planning Agent's graph messages.

FR-008: Worker tasks must inherit the parent `conversation_id`, `user_id`, `device_id`, `model_request`, `persona`, and relevant planning context. Workers must not inherit the parent's rolling `history_summary`; their task prompt is the isolation boundary.

FR-009: Worker tasks must inherit each target agent's existing model config, tool allowlist, MCP tools, client runtime tools, deferred tool search behavior, and fallback provider behavior.

FR-010: Multiple worker tasks in one dispatch must run concurrently and preserve ordered aggregation.

FR-011: Worker results must be returned in the same order as the input task list.

FR-012: A failed worker must produce a structured failed result without cancelling successful sibling workers, unless the entire dispatch is cancelled by timeout or process shutdown.

FR-013: If an underlying worker operation raises a timeout, the dispatcher must produce a structured timeout result. The dispatcher must not impose a subagent-specific timeout wrapper.

FR-014: If a worker attempts an operation that requires human approval, the first production version should stop that worker and return `requires_approval` in the dispatch result. The Planning Agent should explain that the user must approve or run that operation directly in the main Planning turn. Native nested interrupts can be added later.

FR-015: The Planning Agent must decide todo updates after reading subagent results. The dispatcher must not mark tasks complete directly.

FR-016: Dispatch output must include enough metadata for debugging and todo reconciliation: worker id, agent target, status, elapsed milliseconds, output summary, related todo ids, and error message when present. Worker tool artifacts/images must not be embedded in dispatch results.

FR-017: Dispatch output sent back to the model must use the same generic tool-result size controls as the rest of the graph and remain deterministic.

FR-018: The top-level `dispatch_subagents` tool artifact/render should be attached to response metadata/tool artifacts for UI/debug visibility when practical.

## Non-Goals

- Do not create background jobs, polling APIs, job notifications, or Celery tasks.
- Do not expose subagent dispatch to `chat_agent`, `search_agent`, `rag_agent`, `image_generator_agent`, or `canvas_agent`.
- Do not create new persisted subagent-run tables in the first implementation.
- Do not let subagents mutate `TaskPlan` rows directly.
- Do not replace the existing router.
- Do not remove or repurpose `hand_off`.
- Do not parallelize generic tool execution.
- Do not add dynamic subagent discovery in the first implementation.

## Public Behavior

When a user is in Planning mode and asks to execute a plan, the Planning Agent can identify independent tasks and call `dispatch_subagents`. Example internal tool input:

```json
{
  "tasks": [
    {
      "id": "worker-1",
      "agent": "search_agent",
      "task": "Find current documentation constraints for the API migration.",
      "related_todo_ids": ["todo-7"],
      "expected_output": "Return concise findings with source links."
    },
    {
      "id": "worker-2",
      "agent": "chat_agent",
      "task": "Inspect the implementation plan and identify likely backend files to change.",
      "related_todo_ids": ["todo-8"],
      "expected_output": "Return a short file map and risks."
    }
  ],
  "rationale": "These tasks are independent and can be researched in parallel."
}
```

Example tool result returned to the Planning Agent:

```json
{
  "status": "completed",
  "results": [
    {
      "id": "worker-1",
      "agent": "search_agent",
      "status": "completed",
      "elapsed_ms": 1834,
      "summary": "Current docs confirm ...",
      "related_todo_ids": ["todo-7"]
    },
    {
      "id": "worker-2",
      "agent": "chat_agent",
      "status": "completed",
      "elapsed_ms": 912,
      "summary": "Likely files: app/ai/graph.py, app/ai/agents/planning_agent.py ...",
      "related_todo_ids": ["todo-8"]
    }
  ]
}
```

The Planning Agent then calls `write_todos` to complete or update relevant tasks.

## Data Model

Create these Pydantic models in `app/ai/planning_subagents.py` unless implementation size justifies splitting schemas into `app/ai/subagent_schemas.py`.

### `PlanningSubagentName`

Enum values:

- `chat_agent`
- `rag_agent`
- `search_agent`
- `image_generator_agent`
- `canvas_agent`

Do not include `planning_agent`.

### `PlanningSubagentTask`

Fields:

- `id: str`
- `agent: PlanningSubagentName`
- `task: str`
- `related_todo_ids: list[str] = []`
- `expected_output: str | None = None`
- `context: dict[str, Any] = {}`

Validation:

- `id` must be non-empty.
- `task` must be non-empty.
- `related_todo_ids` are advisory metadata only.
- `context` must be JSON-serializable.

### `DispatchSubagentsInput`

Fields:

- `tasks: list[PlanningSubagentTask]`
- `rationale: str | None = None`

Validation:

- `tasks` length must be at least 1.
- Duplicate task ids are rejected.

### `PlanningSubagentResult`

Fields:

- `id: str`
- `agent: PlanningSubagentName`
- `status: Literal["completed", "failed", "timeout", "requires_approval"]`
- `elapsed_ms: int`
- `summary: str`
- `related_todo_ids: list[str] = []`
- `error: str | None = None`

Serialization rule:

- Model-facing JSON and `GraphContext["subagent_results"]` must omit worker `artifacts` and `images`. Large worker tool payloads are not useful for supervisor reconciliation and can bloat LLM input and persisted message metadata. Summary text uses the existing generic tool-result size control rather than subagent-specific config.

### `DispatchSubagentsResult`

Fields:

- `status: Literal["completed", "partial", "failed"]`
- `results: list[PlanningSubagentResult]`
- `rationale: str | None = None`

Aggregate status rules:

- `completed`: every worker completed.
- `partial`: at least one worker completed and at least one worker failed, timed out, or requires approval.
- `failed`: no worker completed.

## Internal Contracts

### `PlanningSubagentDispatcher`

Create in `app/ai/planning_subagents.py`.

Responsibilities:

- Validate dispatch input.
- Build isolated worker state.
- Run workers concurrently with `asyncio.gather`.
- Do not apply a subagent-specific per-worker timeout.
- Preserve input order in output.
- Normalize worker output into `PlanningSubagentResult`.
- Keep model-facing summaries within the generic tool-result size budget.

Constructor dependencies:

- `workflow: MultiAgentWorkflow` or a narrow protocol exposing the reusable worker-runner helper.
- `settings`.

Primary methods:

```python
async def dispatch(
    self,
    request: DispatchSubagentsInput,
    *,
    parent_state: GraphState,
) -> DispatchSubagentsResult:
    ...
```

```python
async def run_one(
    self,
    task: PlanningSubagentTask,
    *,
    parent_state: GraphState,
) -> PlanningSubagentResult:
    ...
```

### `create_dispatch_subagents_tool`

Create in `app/ai/planning_subagents.py`.

Responsibilities:

- Return an async LangChain tool with `args_schema=DispatchSubagentsInput`.
- Close over dispatcher and a parent-state provider.
- Return a JSON string generated from `DispatchSubagentsResult.model_dump()`.

Tool name:

- `dispatch_subagents`

Tool description:

- State that it is for Planning execution only.
- State that it runs independent workers in parallel and waits for all results.
- State that workers cannot update the plan directly.
- State that the Planning Agent must call `write_todos` after interpreting results.

### `MultiAgentWorkflow._run_agent_in_isolated_context`

Create in `app/ai/graph.py`.

Responsibilities:

- Run one existing target agent against an isolated `GraphState`.
- Reuse existing agent invocation and tool execution paths where possible.
- Prevent worker state from appending to parent `messages`.
- Carry scoped identifiers and runtime settings from parent state.
- Stop on final response, worker error, request cancellation, or approval-required operation.

Suggested signature:

```python
async def _run_agent_in_isolated_context(
    self,
    *,
    agent_name: str,
    task_prompt: str,
    parent_state: GraphState,
    related_todo_ids: list[str] | None = None,
) -> AgentResponse:
    ...
```

Implementation note:

- Prefer extracting shared helper logic from `_chat_node`, `_search_node`, `_canvas_node`, `_image_generator_node`, `_rag_node`, `_tool_node`, and `_rag_tools_node` instead of copy-pasting large blocks.
- Do not invoke the compiled graph recursively with the same thread/checkpointer.
- Do not call the router. The dispatch input already specifies the worker agent.

### `BaseAgent.invoke_model_with_history`

Modify in `app/ai/agents/base_agent.py`.

Change:

- Add optional `internal_tools: list[BaseTool] | None = None`.
- Pass `internal_tools` into `_get_llm_with_tools(...)`.
- Use the same `internal_tools` when computing `bound_tools`.
- Preserve the argument through fallback-provider retry branches.

Rationale:

- Existing `_get_llm_with_tools` and `_get_tools_for_binding` already support `internal_tools`.
- `PlanningAgent` already prepends `write_todos` and deduplicates internal tools.
- This enables `_planning_node` to bind `dispatch_subagents` only for the exact phase where it is allowed.

### `PlanningAgent` Prompt Contract

Modify `app/ai/agents/planning_agent.py`.

Execution phase prompt additions:

- Use `dispatch_subagents` only for independent tasks.
- Do not dispatch dependent tasks together.
- Do not dispatch trivial tasks that the Planning Agent can do directly.
- Do not dispatch more than the configured batch limit.
- After dispatch results return, decide which todos to complete or update with `write_todos`.
- If a worker result is failed, timed out, or requires approval, keep the corresponding todo pending or update it with a clear blocker.

Planning phase prompt:

- Do not mention or encourage `dispatch_subagents`.

## Source Code Structure

### Files to Create

- `app/ai/planning_subagents.py`
  - Pydantic input/output schemas.
  - `PlanningSubagentDispatcher`.
  - `create_dispatch_subagents_tool`.
  - Output truncation helpers.

- `tests/test_planning_subagents.py`
  - Schema validation tests.
  - Dispatcher concurrency tests.
  - Result ordering/failure/timeout tests.
  - Tool output JSON tests.

- `tests/test_graph_planning_subagents.py`
  - Planning-node binding tests.
  - Isolated worker runner tests.
  - Planning-loop integration tests.

### Files to Modify

- `app/core/config.py`
  - Add `planning_subagents_enabled: bool = True`.
  - Do not add subagent-specific task-count, parallelism, timeout, worker-iteration, or result-size config caps.

- `app/ai/schemas.py`
  - Extend `GraphContext` with optional compact `subagent_results` and `subagent_dispatches` for activity UI.

- `app/ai/agents/base_agent.py`
  - Add `internal_tools` support to `invoke_model_with_history`.

- `app/ai/agents/planning_agent.py`
  - Update execution-phase prompt.
  - Keep `write_todos` as the first internal tool.
  - Ensure no duplicate internal tools.

- `app/ai/graph.py`
  - Bind the dispatch tool schema for Planning execution turns.
  - Instantiate the executable dispatcher/tool only for actual `dispatch_subagents` tool calls.
  - Pass the dispatch tool to `planning_agent.invoke_model_with_history(...)` only when allowed.
  - Add isolated worker runner helper.
  - Attach compact dispatch results to graph context.
  - Prevent generic tool execution from becoming parallel.

- `app/ai/tool_execution.py`
  - No generic parallelization.
  - Add tests or comments only if needed to preserve ordered semantics.

- `README.md`
  - Document Planning-mode subagents under the Planning mode or multi-agent workflow section after implementation.

### Files Not to Modify

- `app/ai/hand_off_tool.py`
  - Keep graph-level handoff unchanged.

- `app/services/task_plan_service.py`
  - Keep persistence contract unchanged for the first implementation.

- Alembic migrations
  - No database schema change in the first implementation.

## Implementation Phases

### Phase 0: Safety Tests and Contracts — DONE 2026-05-08

- [x] Add schema tests in `tests/test_planning_subagents.py` proving valid dispatch input accepts known worker agents and rejects `planning_agent`.
- [x] Add schema tests proving empty tasks, too many tasks, duplicate task ids, and blank task descriptions are rejected.
- [x] Add a tool contract test proving `dispatch_subagents` returns valid JSON with `status` and ordered `results`.
- [x] Add a regression test proving normal `execute_tool_calls()` still executes tools sequentially. Use two fake tools where the second depends on mutation done by the first.
- [x] Confirmed pre-implementation collection failure with `ModuleNotFoundError: No module named 'app.ai.planning_subagents'` — expected.
- [ ] Run targeted tests and confirm new tests fail before implementation:

```powershell
rtk pytest tests/test_planning_subagents.py tests/test_graph_planning_subagents.py tests/test_tool_execution_recovery.py -q
```

Expected: new tests fail because the module/tool/helper do not exist yet.

### Phase 1: Config and Schemas — UPDATED 2026-05-15

- [x] Added Planning subagent feature flag to `app/core/config.py` (`planning_subagents_enabled`).
- [x] Removed subagent-specific config caps (`planning_subagents_max_tasks`, `planning_subagents_max_parallel`, `planning_subagents_worker_timeout_seconds`, `planning_subagents_max_iterations`, `planning_subagents_result_max_chars`) to avoid artificial worker failures such as `subagent_iteration_limit`.
- [x] Created `app/ai/planning_subagents.py` with `PlanningSubagentName`, `PlanningSubagentTask`, `DispatchSubagentsInput`, `PlanningSubagentResult`, `DispatchSubagentsResult`.
- [x] Implemented validation for non-empty task lists, duplicate ids, blank ids, blank tasks, JSON-serializable `context` field.
- [x] Added `DispatchSubagentsResult.from_results(...)` aggregating to `completed` / `partial` / `failed`.
- [x] Extended `GraphContext` with optional `subagent_dispatches` and `subagent_results` for UI/debug visibility.
- [x] Verified `tests/test_planning_subagents.py` passes 18/18 schema and dispatcher tests.
- [ ] Run:

```powershell
rtk pytest tests/test_planning_subagents.py -q
```

Expected: schema tests pass; dispatcher tests still fail until later phases.

### Phase 2: Internal Tool Binding — UPDATED 2026-05-15

- [x] Added optional `internal_tools` parameter to `BaseAgent.invoke_model_with_history`. Threaded through the primary model binding plus all three fallback retry branches (OpenAI reasoning fallback, OpenAI provider fallback, generic provider fallback).
- [x] Added `_build_planning_internal_tools(state)` helper on `MultiAgentWorkflow` that returns the dispatch tool list iff all gates pass: `settings.planning_subagents_enabled`, `state["planning_mode_enabled"] is True`.
- [x] `_planning_node` calls the helper and forwards `internal_tools=...` to `planning_agent.invoke_model_with_history`.
- [x] Added tests proving the tool is bound while Planning mode is active, not bound when Planning mode is inactive, and not bound when the feature is disabled.

Design decision: the dispatcher is built per-turn using `lambda: state` as the parent-state provider. This keeps `MultiAgentWorkflow.__init__` unchanged and avoids storing per-conversation dispatcher instances.

### Phase 3: Dispatcher Runtime — UPDATED 2026-05-15

- [x] `PlanningSubagentDispatcher.dispatch(...)` runs workers with `asyncio.gather(...)`.
- [x] Removed subagent-specific `asyncio.wait_for(...)`; underlying provider/tool timeout errors are still normalized if they occur.
- [x] Result ordering preserved automatically because `gather` returns in input order and we feed it tasks in iteration order.
- [x] Worker exceptions converted to `PlanningSubagentResult(status="failed", error=str(exc))`.
- [x] Underlying `asyncio.TimeoutError` converted to `PlanningSubagentResult(status="timeout", error="timeout")`.
- [x] Model-facing `summary` uses the existing generic `tool_result_max_chars` budget with a `…[truncated]` notice.
- [x] Concurrency, ordering, failure-isolation, and timeout tests added in `tests/test_planning_subagents.py` and pass.

Design decision: subagents should not fail from subagent-only task, timeout, parallelism, or iteration caps. Operational protection comes from request cancellation, provider/tool timeouts, HITL gating, generic tool-result size controls, and the Planning Agent prompt, not from a second set of subagent-specific budget knobs.

### Phase 4: Isolated Existing-Agent Runner — UPDATED 2026-05-15

- [x] Added `MultiAgentWorkflow._run_agent_in_isolated_context(...)` to `app/ai/graph.py`.
- [x] Worker prompt is wrapped in a single `HumanMessage`. Worker keeps an isolated `worker_messages` list — the parent `GraphState["messages"]` is never appended to.
- [x] Worker calls inherit `conversation_id`, `user_id`, `device_id`, `persona`, and `model_request`.
- [x] For non-RAG worker agents, the runner runs an isolated tool loop:
  1. `agent.invoke_model_with_history(...)` against the local message list
  2. If there are tool calls, execute them via `execute_tool_calls(...)` under `tool_execution_context(...)` (sequential, NOT parallel — per Decision 4)
  3. Append `ToolMessage`s into the worker's local list and loop
  4. Stop on no-more-tool-calls (final answer), worker error, request cancellation, or HITL approval requirement
- [x] For `rag_agent`, the runner builds an `AgentMessage` and calls `process_message(...)` so the existing agentic-RAG `search_documents` loop is reused unchanged.
- [x] `requires_approval` is signalled by setting `metadata["requires_approval"] = True` on the response so the dispatcher can map it to the `requires_approval` worker status.
- [x] `planning_agent` rejected as a worker target with `ValueError`.
- [x] Tests `test_run_agent_in_isolated_context_*` cover parent-message isolation, identifier inheritance, worker loop continuation beyond the legacy subagent iteration cap, and the planning-agent rejection.

Design decisions:
- The runner does **not** rebuild a full child `GraphState` for non-RAG agents. The agent invocation path only needs `messages` plus the scoped identifiers, so we pass them as keyword args directly. This keeps the runner simple and avoids touching the `GraphState` typed-dict surface.
- Conversation history and rolling `history_summary` are intentionally omitted from workers. Workers receive their context through the task prompt and structured task `context`; re-reading chat memory defeats isolation and adds repeated tokens.
- RAG workers run a local agentic loop that calls `process_message`, executes `search_documents` results, feeds tool output back through `tool_context`, and stops on final answer, HITL requirement, worker error, or request cancellation.

### Phase 5: Planning Prompt and Todo Reconciliation — DONE 2026-05-08

- [x] Updated the executing-phase block in `PlanningAgent._build_system_prompt` to introduce `dispatch_subagents`, restrict it to independent work, and explicitly state the reconciliation contract (workers cannot mutate todos; the supervisor must call `write_todos` after reading results; failed/timeout/requires_approval results leave the related todo pending).
- [x] Planning phase prompt left unchanged so it does not encourage dispatch during plan creation/editing.
- [x] Prompt-invariant tests added in `tests/test_planning_subagents.py`: executing phase mentions `dispatch_subagents`, planning phase does not, the prompt says workers cannot mutate todos, and the prompt says the supervisor must reconcile with `write_todos`.

### Phase 6: Graph Integration and Artifacts — DONE 2026-05-08

- [x] `_planning_node` now binds a schema-only `dispatch_subagents` tool via `_build_planning_internal_tools(state)` and threads it into `planning_agent.invoke_model_with_history(...)` as `internal_tools`. It does not construct an executable dispatcher during schema binding.
- [x] `_planning_tools_node` only constructs and injects the executable dispatch tool when the AI message actually contains a `dispatch_subagents` call. The planning agent's resulting `dispatch_subagents` tool call is treated like any other external tool call: it goes through `_execute_agent_tool_calls`, which produces a normal `ToolMessage` for the next planning model call to read.
- [x] Dispatch tool stashes a compact summary on `state["context"]["subagent_dispatches"]` (per-call: rationale, task ids, agents, aggregate status) and compact per-result entries on `state["context"]["subagent_results"]`. Both the model-facing JSON and final metadata omit nested worker artifacts/images.
- [x] The top-level `dispatch_subagents` tool artifact remains available through the existing `tool_artifacts` pipeline for UI activity rendering; nested worker artifacts are not copied into subagent result payloads.
- [x] Graph integration test `test_planning_tools_node_executes_dispatch_subagents` verifies the end-to-end path: dispatch tool call → executed → ToolMessage in state → context summaries populated → both worker prompts dispatched.

Design decision: rather than introducing a separate `subagent_node`, we let the dispatcher tool flow through the existing `_planning_tools_node` external-tool branch. This keeps the LangGraph topology unchanged and reuses HITL gating, artifact collection, and ToolMessage emission for free.

### Phase 7: Documentation and Manual Validation — DONE 2026-05-08

- [x] Added `### Planning-mode subagents` section to `README.md` defining the feature, contrasting it with `hand_off`, calling out same-chat synchronous behavior, and listing the remaining env var (`PLANNING_SUBAGENTS_ENABLED`).
- [x] Manual validation scenario remains a runtime/UX exercise; covered by integration tests in `tests/test_graph_planning_subagents.py` for the supervised paths.
- [x] Focused test suite run:
  - `pytest tests/test_planning_subagents.py tests/test_graph_planning_subagents.py` → **30/30 passed**.
  - `pytest tests/test_tool_execution_recovery.py tests/test_graph_tool_budget.py tests/test_router.py tests/test_client_tool_scope.py` → **16/16 passed**.
  - `pytest tests/test_rag_agent.py tests/test_rag_tool_loop_finalization.py tests/test_multi_sidecar_hardening.py tests/test_tool_execution_rendering.py` → **62/62 passed**.
  - Pre-existing unrelated failures (live-server integration tests, `_AI_SDK_HEARTBEAT_INTERVAL_SECONDS`, `allowed_msgpack_modules`, `DeferredToolState.snapshot`, `tool_search_scoring stopwords`) confirmed to also fail on the unmodified branch — not caused by this change.

```powershell
rtk pytest tests/test_planning_subagents.py tests/test_graph_planning_subagents.py tests/test_graph_tool_budget.py tests/test_tool_execution_recovery.py tests/test_client_tool_scope.py -q
```

- [ ] Run broader agent/orchestration suite:

```powershell
rtk pytest tests/test_router.py tests/test_rag_agent.py tests/test_rag_tool_loop_finalization.py tests/test_multi_sidecar_hardening.py tests/test_tool_execution_rendering.py -q
```

### Phase 8: Review Hardening — DONE 2026-05-12

- [x] Removed nested worker artifacts/images from `PlanningSubagentResult` outputs. `dispatch_subagents` now returns compact result JSON to the Planning Agent and stores compact `subagent_results` in graph context/final metadata.
- [x] Changed Planning binding to use a schema-only dispatch tool so `_planning_node` does not allocate an executable dispatcher just to expose the tool schema.
- [x] Gated executable dispatch-tool construction in `_planning_tools_node` on the presence of a `dispatch_subagents` call. `write_todos`-only turns no longer build the dispatcher.
- [x] Reused each generic worker's execution `tool_map` across iterations, letting `execute_tool_calls(...)` keep `tool_search` refreshes in-place for follow-up worker tool calls.
- [x] Dropped parent `history_summary` from isolated worker model calls to preserve context isolation and avoid repeated memory-block tokens.
- [x] Implemented a local RAG worker tool loop so `rag_agent` subagents actually execute `search_documents` and feed results back through `tool_context` before returning a final worker answer.
- [x] Added regression tests for compact dispatch payloads, schema-only binding, dispatch construction gating, worker tool-map reuse, omitted worker history summary, and RAG worker tool-loop execution.
- [x] Focused verification:
  - `rtk pytest tests/test_planning_subagents.py tests/test_graph_planning_subagents.py tests/test_message_service_subagent_streaming.py tests/test_tool_result_rendering.py -q` -> **58/58 passed**.
  - `rtk pytest tests/test_tool_execution_recovery.py tests/test_graph_tool_budget.py tests/test_router.py tests/test_client_tool_scope.py tests/test_rag_agent.py tests/test_rag_tool_loop_finalization.py tests/test_tool_execution_rendering.py -q` -> **57/57 passed**.
  - `rtk pytest tests/test_demo_subagent_activity.py tests/test_message_service_subagent_streaming.py -q` -> **10/10 passed**.

### Phase 9: Remove Subagent-Specific Caps — UPDATED 2026-05-15

- [x] Root cause: workers that needed more than `planning_subagents_max_iterations` model/tool rounds were converted to `error="subagent_iteration_limit"` even when they were still making valid progress.
- [x] Removed subagent-specific limit settings from `app/core/config.py`; `PLANNING_SUBAGENTS_ENABLED` is the only subagent-specific env flag.
- [x] Removed `planning_subagents_max_tasks` validation from `DispatchSubagentsInput`.
- [x] Removed subagent-specific per-worker `asyncio.wait_for(...)` from the dispatcher. Underlying provider/tool `asyncio.TimeoutError` is still normalized to a `timeout` worker result.
- [x] Removed subagent-specific worker iteration caps from generic and RAG isolated worker loops.
- [x] Kept model-facing summary sizing on the existing generic `tool_result_max_chars` setting so dispatch output follows the same context-budget safety rule as other tool results.
- [x] Updated the Planning prompt to keep dispatch calls focused instead of referring to an oversized-batch rejection rule.
- [x] Updated README and this plan to reflect Planning-mode availability and the removal of subagent-specific config caps.
- [x] Verification:

```powershell
rtk pytest tests/test_planning_subagents.py tests/test_graph_planning_subagents.py tests/test_config_redis.py -q
```

Result: **62/62 passed**.

Additional adjacent verification:

```powershell
rtk pytest tests/test_message_service_subagent_streaming.py tests/test_demo_subagent_activity.py tests/test_tool_result_rendering.py tests/test_tool_execution_recovery.py tests/test_graph_tool_budget.py -q
```

Result: **29/29 passed**.

## Acceptance Criteria

- [ ] In a non-Planning conversation, `dispatch_subagents` is unavailable.
- [ ] In Planning mode, `dispatch_subagents` is available to `planning_agent` when the feature flag is enabled.
- [ ] The Planning prompt uses `dispatch_subagents` for explicit delegation/testing and independent work, while normal plan creation/editing stays on `write_todos`.
- [ ] `dispatch_subagents` rejects `planning_agent`.
- [ ] Two independent fake worker tasks execute concurrently under the dispatcher.
- [ ] Dispatch results are ordered the same as input tasks.
- [ ] One worker failure does not erase successful sibling results.
- [ ] Underlying worker timeout errors are reported as `timeout`, without a subagent-specific timeout wrapper.
- [ ] Worker loops do not fail with `subagent_iteration_limit`.
- [ ] Dispatch input has no subagent-specific maximum task-count validation.
- [ ] Parent graph messages do not include worker intermediate messages.
- [ ] Worker calls carry user/device/conversation context for scoped tool access.
- [ ] Worker calls do not receive the parent's rolling `history_summary`.
- [ ] `dispatch_subagents` model-facing JSON and `subagent_results` metadata omit nested worker artifacts/images.
- [ ] `_planning_tools_node` does not construct the executable dispatcher when no `dispatch_subagents` call exists.
- [ ] Worker tool maps are reused across iterations so `tool_search` refreshes remain available to follow-up tool calls.
- [ ] RAG workers execute their local `search_documents` loop before returning a final answer.
- [ ] The Planning Agent remains the only actor that calls `write_todos`.
- [ ] Existing `hand_off` tests still pass.
- [ ] Existing deferred-tool-search same-batch behavior still passes.

## Risks and Mitigations

Risk: nested human approval from a subagent can be hard to resume because tool-invoked subagents are not statically visible in LangGraph state.

Mitigation: first implementation returns `requires_approval` as a worker status and lets the Planning Agent pause/explain. Add native nested interrupt support later only if UX requires it.

Risk: RAG worker loops duplicate graph logic.

Mitigation: keep the RAG worker loop inside `MultiAgentWorkflow._run_agent_in_isolated_context` so it can reuse `execute_search_documents_action`, tool artifact construction, offload handling, and graph agent configuration without recursively invoking the compiled graph.

Risk: parallel workers share process-wide deferred tool state.

Mitigation: preserve existing scoping by `conversation_id`, `agent_key`, `device_id`, and active session. Add tests for inherited scope and avoid generic parallel tool execution.

Risk: subagent output bloats the Planning Agent context.

Mitigation: use the existing generic tool-result size control for model-facing summaries and omit nested worker artifacts/images from both the model-facing dispatch JSON and persisted `subagent_results` metadata. Keep only the top-level `dispatch_subagents` artifact/render payload for UI activity.

Risk: Planning Agent over-dispatches trivial work.

Mitigation: prompt guard, explicit independent-work criteria, and supervisor-only todo reconciliation.

## Future Extensions

- Dynamic `list_subagents` discovery if the agent registry grows beyond the static graph agents.
- Native nested HITL interrupt propagation for subagent tools.
- Persisted subagent run summaries for audit/debugging.
- UI display for subagent worker progress and results.
- Per-agent worker capability descriptions stored alongside agent registry configuration.

## Open Questions Resolved

- Scope: Planning mode only.
- Execution mode: synchronous same-chat execution, no background jobs.
- Worker capability inheritance: workers reuse existing graph agents and their runtime/tool capabilities.
- Worker memory: workers inherit scoped ids and model/persona settings, but not parent chat history or rolling `history_summary`.
- Dispatch visibility: subagent activity is compact metadata plus the top-level dispatch tool render, not nested worker artifacts.
- Parallelism: only inside `dispatch_subagents`, not generic tool execution.

## References

- LangChain subagents documentation: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents
- GitHub Spec Kit plan template: https://github.com/github/spec-kit/blob/main/templates/plan-template.md
- GitHub Spec Kit SDD overview: https://github.com/github/spec-kit/blob/main/spec-driven.md
