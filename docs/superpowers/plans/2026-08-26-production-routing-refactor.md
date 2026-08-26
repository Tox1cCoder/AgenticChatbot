# Production Routing Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current multilingual-fragile routing shortcuts, mutable `selected_agent` control flow, duplicated RAG execution, and direct-to-`END` agent paths with one provider-agnostic LLM router, typed LangGraph transitions, shared specialist subgraphs, mandatory grounding, and universal public-response finalization.

**Architecture:** Every new user turn enters one `route` node. The configured router model returns a schema-constrained `RoutingDecision`; Python validates only inventory and control-plane invariants and never interprets message language or intent. The parent graph records an immutable initial decision, tracks the active agent and append-only transitions separately, invokes per-request specialist subgraphs, routes all public outcomes through validation and finalization, and uses one shared RAG graph for top-level and Planning-worker retrieval. Router failure is a typed retriable workflow failure—never an agent substitution.

**Tech Stack:** Python 3.10+, Pydantic 2, LangChain 1.3 (`create_agent`, structured output, middleware), LangGraph 1.2 (`StateGraph`, `Command`, `Send`, `ToolNode`, `interrupt`, PostgreSQL checkpointer), FastAPI, SQLAlchemy/PostgreSQL, LangSmith-compatible tracing, pytest, pytest-asyncio, Ruff.

**Approved design:** `docs/superpowers/specs/2026-08-26-production-routing-refactor-design.md`

## Global Constraints

- Do not add regexes, token matching, translated keyword lists, explicit-name matchers, phrase rules, or default-agent branches that inspect user text.
- Canvas state, documents, planning state, previous agent, custom-agent names, tools, skills, locale, and time are router context only. None may select a route before the model call.
- Run the router exactly once for a new user turn. `Command(resume=...)` continues the interrupted execution and must not route again.
- Use the configured runtime provider/model abstraction. The router must not import `google.genai`, instantiate a provider SDK client, or switch provider/model after failure.
- The router may make at most two calls to the same configured model. Default total-attempt timeout is 8 seconds and is configurable.
- `RoutingDecision.confidence` is telemetry only. It must never determine a branch.
- `routing_decision` is immutable after acceptance. Handoffs update `active_agent_id` and append `agent_history`; they never rewrite the initial decision.
- A node that returns a dynamic `Command(goto=...)` has no static outgoing edge.
- Standard specialists use LangChain `create_agent`; RAG uses one bespoke `StateGraph` with `ToolNode`; Planning uses `Send` for independent workers.
- Every public answer passes through `validate_output` and `finalize`. Only `finalize` may append the terminal public `AIMessage` or reach `END`.
- RAG grounding is mandatory for top-level RAG, RAG workers, and any public Planning synthesis that carries RAG evidence. One invalid answer may regenerate once; a second invalid answer becomes an explicit abstention.
- Worker subgraphs never append public assistant messages and never perform parent-level handoffs.
- Tests use deterministic fake models and tools. Live provider evaluation is a separately marked pre-deployment job.
- This is a breaking cutover. Do not preserve `selected_agent`, custom/canvas stickiness, pre-routing, direct Gemini routing, chat fallback, auto-continuation, duplicated RAG loops, shadow grounding, or stale-response recovery.
- Preserve unrelated user changes in the worktree. Use small commits after each passing task.

## File and Responsibility Map

- `app/ai/workflow/contracts.py`: routing, transition, outcome, worker, error, and execution-phase Pydantic contracts.
- `app/ai/workflow/state.py`: `GraphState`, reducers, typed runtime context, and state accessors.
- `app/ai/workflow/inventory.py`: live base/custom specialist inventory and version hash; no message interpretation.
- `app/ai/workflow/routing.py`: bounded context builder, structured model invocation, retry/timeout handling, and target validation.
- `app/ai/workflow/specialists.py`: per-invocation `create_agent` factories and parent wrapper nodes.
- `app/ai/workflow/middleware.py`: scoped model/tool resolution, limits, authorization, HITL, artifact capture, offloading, and usage instrumentation.
- `app/ai/workflow/transitions.py`: dynamic handoff tool and the sole parent transition resolver.
- `app/ai/workflow/rag_execution.py`: shared compiled RAG model/tool/evidence/grounding subgraph.
- `app/ai/workflow/planning_execution.py`: Planning orchestrator, `Send` fan-out, reducers, and synthesis.
- `app/ai/workflow/finalization.py`: provenance-based validation, worker finalization, and public finalization.
- `app/ai/workflow/graph_builder.py`: parent graph topology only.
- `app/ai/graph.py`: thin `IWorkflowRuntime` adapter for request preparation, graph invoke/resume, streaming, and checkpoint compaction.
- `app/observability/routing.py`: content-free routing/transition/finalizer metrics.
- `app/evaluation/routing/`: deterministic metric and release-gate code.
- `eval/routing/`: versioned multilingual live-routing dataset and thresholds.

---

### Task 1: Add typed workflow contracts and replace the state vocabulary

**Files:**
- Create: `app/ai/workflow/contracts.py`
- Create: `app/ai/workflow/state.py`
- Modify: `app/ai/schemas.py`
- Create: `tests/test_workflow_contracts.py`
- Create: `tests/test_workflow_state.py`

**Interfaces:**
- Consumes: base/custom agent IDs, model routing output, specialist results, handoff requests, and worker results.
- Produces: `RoutingDecision`, `AgentTransition`, `ResponseOutcome`, `HandoffOutcome`, `WorkerResult`, `WorkflowError`, `WorkflowState`, and append-only reducers.

- [ ] **Step 1: Write failing schema and reducer tests**

```python
def test_routing_decision_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        RoutingDecision.model_validate(
            {"agent_id": "chat_agent", "confidence": 0.8,
             "reason": "best capability", "selected_agent": "legacy"}
        )


def test_transition_reducer_is_append_only():
    first = AgentTransition(from_agent_id="chat_agent", to_agent_id="search_agent",
                            source="handoff", tool_call_id="call-1")
    second = AgentTransition(from_agent_id="search_agent", to_agent_id="chat_agent",
                             source="handoff", tool_call_id="call-2")
    assert append_transitions([first], [second]) == [first, second]


def test_graph_state_has_no_selected_agent_field():
    assert "selected_agent" not in WorkflowState.__annotations__
    assert "last_agent" not in WorkflowState.__annotations__
    assert "delegation_count" not in WorkflowState.__annotations__
```

- [ ] **Step 2: Run the tests and confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_contracts.py tests/test_workflow_state.py
```

Expected: collection fails because the new modules do not exist.

- [ ] **Step 3: Implement strict contracts**

```python
class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    agent_id: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)


class AgentTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    from_agent_id: str | None
    to_agent_id: str = Field(min_length=1, max_length=160)
    source: Literal["router", "handoff", "resume"]
    tool_call_id: str | None = None


class WorkflowError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: Literal[
        "routing_timeout", "routing_provider_unavailable", "routing_invalid_output",
        "routing_target_unavailable", "agent_execution_limit", "tool_execution_failed",
        "response_validation_failed", "finalization_failed",
    ]
    retriable: bool
    request_id: str
    details: dict[str, JsonValue] = Field(default_factory=dict)
```

Define `ResponseOutcome` and `HandoffOutcome` as a discriminated union on `kind`; define `WorkerResult.status` as `completed | failed | awaiting_approval`; define `ExecutionPhase` as `routing | executing | awaiting_approval | validating | finalizing | completed | failed`. Add `WorkflowRoutingException(RuntimeError)` with one immutable `error: WorkflowError` attribute so service and streaming boundaries can translate failures without parsing exception text.

- [ ] **Step 4: Add reducers and the new graph state**

Use `Annotated[list[AgentTransition], append_transitions]` and `Annotated[list[WorkerResult], append_worker_results]`. State must contain `routing_decision`, `routing_inventory_version`, `active_agent_id`, `final_agent_id`, `agent_history`, `pending_transition`, `agent_outcome`, `worker_results`, `execution_phase`, and `workflow_error`, plus the existing request scopes, planning data, attachments, artifacts, and messages that remain valid.

Do not delete the old fields from `app/ai/schemas.py` yet; import/re-export the new contracts there only long enough for Tasks 2–11 to migrate call sites. The final cutover removes the old `GraphState` and re-exports.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_contracts.py tests/test_workflow_state.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/contracts.py app/ai/workflow/state.py app/ai/schemas.py tests/test_workflow_contracts.py tests/test_workflow_state.py
git add app/ai/workflow/contracts.py app/ai/workflow/state.py app/ai/schemas.py tests/test_workflow_contracts.py tests/test_workflow_state.py
git commit -m "refactor: add typed workflow state contracts"
```

Expected: tests pass; reducers preserve order; invalid contracts fail closed.

---

### Task 2: Build a bounded, language-neutral specialist inventory and routing context

**Files:**
- Create: `app/ai/workflow/inventory.py`
- Modify: `app/ai/workflow/routing.py` (create the context-building portion only)
- Modify: `app/ai/history.py`
- Create: `tests/test_routing_inventory.py`
- Create: `tests/test_routing_context.py`

**Interfaces:**
- Consumes: request, canonical history summary/recent messages, attached custom agents, current canvas snapshot, planning state, and capability summaries.
- Produces: one immutable `RoutingInventory` and one bounded `RoutingContext` serialized as trusted instructions plus clearly delimited untrusted data.

- [ ] **Step 1: Add failing inventory/context tests**

```python
async def test_context_includes_state_without_selecting_from_it(builder):
    context = await builder.build(request_with_all_context())
    assert context.active_canvas.title == "Current Site"
    assert context.planning.lifecycle == "executing"
    assert context.previous_final_agent_id == "custom_agent:alpha"
    assert context.documents[0].filename == "คู่มือ.pdf"
    assert not hasattr(builder, "select_agent")


def test_context_excludes_canvas_source_and_document_content(builder):
    payload = builder.serialize(context_with_secrets())
    assert "SECRET CANVAS SOURCE" not in payload
    assert "SECRET DOCUMENT BODY" not in payload


def test_inventory_version_is_stable_and_order_independent():
    assert inventory_version([descriptor_b, descriptor_a]) == inventory_version(
        [descriptor_a, descriptor_b]
    )
```

Also inspect `app/ai/workflow/routing.py` source in a contract test and reject `re`, `tokenize_text`, `_match_explicit_custom_agent`, message `.lower()`, and hard-coded return values ending in `_agent`.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_inventory.py tests/test_routing_context.py
```

- [ ] **Step 3: Implement typed descriptors and a stable inventory hash**

```python
class AgentDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    agent_id: str
    display_name: str
    capability_description: str
    enabled: bool
    attached: bool = True
    kind: Literal["base", "custom"]


class RoutingInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str
    agents: tuple[AgentDescriptor, ...]
```

The version is SHA-256 over canonical JSON sorted by `agent_id`. Base descriptors come from one registry; custom descriptors come only from the authenticated attached-agent map. Do not read user text in this module.

- [ ] **Step 4: Implement bounded context construction**

Reuse `ConversationHistoryProvider.build_context(..., agent_key="router")` so summary ownership, message ordering, and token limits remain canonical. Add explicit router history limits to settings rather than slicing ad hoc. Include original-language text unchanged; metadata-only document and canvas descriptors; bounded plan/todo summaries; previous `final_agent_id`; tool/skill summaries; and locale/time.

Serialize the router system instruction separately from a `RoutingContext.model_dump_json()`. Mark conversation content, custom personas/descriptions, filenames, and skill text as untrusted reference data. Do not interpolate those fields into the system instruction.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_inventory.py tests/test_routing_context.py tests/test_history_provider.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/inventory.py app/ai/workflow/routing.py app/ai/history.py tests/test_routing_inventory.py tests/test_routing_context.py
git add app/ai/workflow/inventory.py app/ai/workflow/routing.py app/ai/history.py tests/test_routing_inventory.py tests/test_routing_context.py
git commit -m "feat: build bounded semantic routing context"
```

---

### Task 3: Implement provider-agnostic structured routing with typed failures

**Files:**
- Modify: `app/ai/workflow/routing.py`
- Modify: `app/ai/model_factory.py`
- Modify: `app/interfaces/runtime_model_resolver_interface.py`
- Modify: `app/services/model_config_service.py`
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Create: `app/observability/routing.py`
- Replace: `tests/test_router.py`
- Create: `tests/test_routing_service.py`
- Create: `tests/test_routing_startup_validation.py`
- Modify: `tests/test_model_usage_workflow_instrumentation.py`

**Interfaces:**
- Consumes: `RoutingContext`, `RoutingInventory`, user ID, request model override, runtime model resolver, model factory, and usage recorder.
- Produces: a validated immutable `RoutingDecision` or a `WorkflowRoutingException` containing a stable `WorkflowError`.

- [ ] **Step 1: Replace free-text router tests with structured-output tests**

```python
async def test_routes_with_configured_provider_and_structured_output(service, fake_model):
    fake_model.structured_result = RoutingDecision(
        agent_id="search_agent", confidence=0.91, reason="needs current sources"
    )
    decision = await service.route(context, inventory, user_id=USER_ID, model_request=None)
    assert decision.agent_id == "search_agent"
    assert fake_model.structured_schema is RoutingDecision


async def test_unknown_target_is_not_reinterpreted_or_replaced(service, fake_model):
    fake_model.structured_result = RoutingDecision(
        agent_id="missing", confidence=1.0, reason="invalid target"
    )
    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(context, inventory, user_id=USER_ID, model_request=None)
    assert exc.value.error.code == "routing_target_unavailable"
    assert fake_model.calls == 2


async def test_two_failures_never_fall_back_to_chat(service, fake_model):
    fake_model.side_effect = TimeoutError()
    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(context, inventory, user_id=USER_ID, model_request=None)
    assert exc.value.error.code == "routing_timeout"
    assert fake_model.calls == 2
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_router.py tests/test_routing_service.py tests/test_routing_startup_validation.py
```

- [ ] **Step 3: Add a router-capability contract to runtime resolution**

Add `require_capabilities: frozenset[str] = frozenset()` to `resolve_runtime_config`. `ModelConfigService` must reject the router configuration unless its capability map includes structured/schema output. Add:

```python
routing_timeout_seconds: float = Field(default=8.0, gt=0.0, le=30.0)
routing_max_attempts: int = Field(default=2, ge=1, le=2)
workflow_graph_version: str = Field(default="routing-v2")
```

Startup initialization calls `RoutingService.validate_configuration()` and fails startup with a configuration error if the selected model lacks structured output or credentials. Do not add a fallback configuration.

- [ ] **Step 4: Implement the structured call and bounded retry**

```python
configured = self._resolver.resolve_runtime_config(
    user_id, "router", request_override,
    require_capabilities=frozenset({"structured_output"}),
)
model = self._model_factory.create_model_from_runtime(configured)
structured = model.with_structured_output(RoutingDecision, include_raw=True)
deadline = time.monotonic() + self._timeout_seconds

for attempt in range(1, self._max_attempts + 1):
    try:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("routing deadline exhausted")
        result = await asyncio.wait_for(
            structured.ainvoke(messages, config=run_config),
            timeout=remaining_seconds,
        )
        decision = RoutingDecision.model_validate(result["parsed"])
        self._validator.validate(decision, inventory)
        return decision
    except RETRIABLE_ROUTING_EXCEPTIONS as exc:
        last_error = exc
```

The second attempt uses the same provider, model, schema, inventory, context, and total deadline. Map timeout, provider/transport, parse/schema, and target-race failures to the four approved error codes. Record attempt count, provider/model, latency, inventory version, and schema outcome without raw prompts.

- [ ] **Step 5: Delete the Gemini/free-text implementation**

Delete `app/ai/agents/router.py` after moving `ROUTER_SYSTEM_PROMPT` into `routing.py` or a focused prompt constant. Remove direct `google.genai` imports, `_extract_agent_name`, `_match_explicit_custom_agent`, free-text parsing, and every `return "chat_agent"` fallback. Update imports and the model-usage callsite manifest.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_router.py tests/test_routing_service.py tests/test_routing_startup_validation.py tests/test_model_usage_workflow_instrumentation.py tests/test_model_config_reasoning.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/routing.py app/ai/model_factory.py app/interfaces/runtime_model_resolver_interface.py app/services/model_config_service.py app/core/config.py app/core/container.py app/observability/routing.py tests/test_router.py tests/test_routing_service.py tests/test_routing_startup_validation.py
git add app/ai/workflow/routing.py app/ai/model_factory.py app/interfaces/runtime_model_resolver_interface.py app/services/model_config_service.py app/core/config.py app/core/container.py app/observability/routing.py tests/test_router.py tests/test_routing_service.py tests/test_routing_startup_validation.py tests/test_model_usage_workflow_instrumentation.py tests/fixtures/model_usage_callsite_manifest.json
git rm app/ai/agents/router.py
git commit -m "refactor: replace router with structured runtime model routing"
```

Expected: all tests pass and repository search finds no Gemini client or semantic text parser in routing code.

---

### Task 4: Build the new parent graph shell and route once per new turn

**Files:**
- Rewrite: `app/ai/workflow/graph_builder.py`
- Modify: `app/ai/graph.py`
- Modify: `app/core/container.py`
- Create: `tests/test_production_workflow_graph.py`
- Replace: `tests/test_graph_refactor_contract.py`
- Modify: `tests/test_graph_route_node_async_documents.py`
- Modify: `tests/test_ai_service_initialization.py`

**Interfaces:**
- Consumes: prepared `WorkflowState`, `RoutingService`, specialist node registry, transition resolver, validators, finalizer, and checkpointer.
- Produces: a compiled `routing-v2` parent graph whose `route` node returns `Command(update=..., goto=...)`.

- [ ] **Step 1: Write failing topology and invocation-count tests**

```python
def test_only_finalizer_reaches_end(compiled_graph):
    graph = compiled_graph.get_graph()
    end_sources = {edge.source for edge in graph.edges if edge.target == "__end__"}
    assert end_sources == {"finalize"}


async def test_new_turn_routes_exactly_once(workflow, routing_service):
    await workflow.execute_request(request("สรุปเอกสารนี้"))
    routing_service.route.assert_awaited_once()


async def test_resume_does_not_route_again(paused_workflow, routing_service):
    await paused_workflow.resume_with_decisions_stream("thread-1", [approve("call-1")])
    routing_service.route.assert_not_awaited()
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_production_workflow_graph.py tests/test_graph_refactor_contract.py tests/test_graph_route_node_async_documents.py
```

- [ ] **Step 3: Replace conditional routing with `Command`**

```python
async def route_node(state: WorkflowState, runtime: Runtime[WorkflowRuntimeContext]) -> Command:
    decision = await runtime.context.routing_service.route_for_state(state)
    target_node = runtime.context.inventory.resolve_node(decision.agent_id)
    return Command(
        update={
            "routing_decision": decision,
            "routing_inventory_version": runtime.context.inventory.version,
            "active_agent_id": decision.agent_id,
            "agent_history": [AgentTransition(
                from_agent_id=None, to_agent_id=decision.agent_id, source="router"
            )],
            "execution_phase": "executing",
        },
        goto=target_node,
    )
```

On `WorkflowRoutingException`, return `Command(update={"workflow_error": ..., "execution_phase": "failed"}, goto="finalize")`. The finalizer recognizes failed state, records terminal failure metadata, publishes no assistant message, and lets the graph end through the same universal terminal node.

- [ ] **Step 4: Wire the parent topology**

Build `START -> route`. Register stable wrapper nodes for base agents plus one `custom_agent` wrapper, `resolve_transition`, `validate_output`, and `finalize`. Specialist wrappers and `resolve_transition` return dynamic `Command`s and therefore receive no static outgoing edges. Add only `finalize -> END`; both successful and failed executions terminate there.

Use checkpoint configuration `configurable.thread_id = f"routing-v2:{thread_id}"`. A resume uses the exact stored versioned thread ID; a new request never loads the old namespace.

- [ ] **Step 5: Remove streaming pre-routing from the runtime shell**

Delete the call around current `app/ai/graph.py:2938` that invokes `_route_node` before `astream`. Both sync and streaming paths construct state with `routing_decision=None` and let graph execution enter `route`. Keep old specialist method bodies temporarily so later tasks can migrate them behind the new wrappers.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_production_workflow_graph.py tests/test_graph_refactor_contract.py tests/test_graph_route_node_async_documents.py tests/test_ai_service_initialization.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/graph_builder.py app/ai/graph.py app/core/container.py tests/test_production_workflow_graph.py tests/test_graph_refactor_contract.py
git add app/ai/workflow/graph_builder.py app/ai/graph.py app/core/container.py tests/test_production_workflow_graph.py tests/test_graph_refactor_contract.py tests/test_graph_route_node_async_documents.py tests/test_ai_service_initialization.py
git commit -m "refactor: introduce single-entry production workflow graph"
```

---

### Task 5: Replace bespoke standard-agent loops with per-invocation `create_agent` subgraphs

**Files:**
- Create: `app/ai/workflow/middleware.py`
- Create: `app/ai/workflow/specialists.py`
- Modify: `app/ai/agents/chat_agent.py`
- Modify: `app/ai/agents/search_agent.py`
- Modify: `app/ai/agents/canvas_agent.py`
- Modify: `app/ai/agents/image_generator_agent.py`
- Modify: `app/ai/agents/custom_agent.py`
- Modify: `app/ai/custom_agent_runtime.py`
- Modify: `app/ai/deferred_tool_binding.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/hitl_config.py`
- Create: `tests/test_specialist_runtime.py`
- Create: `tests/test_specialist_middleware.py`
- Modify: `tests/test_custom_agents_graph.py`
- Modify: `tests/test_hitl_gate_policy.py`
- Modify: `tests/test_graph_tool_budget.py`

**Interfaces:**
- Consumes: `SpecialistDefinition`, `WorkflowRuntimeContext`, authenticated request scope, runtime model config, tools, history, and parent state.
- Produces: one per-invocation compiled `create_agent` subgraph and an `AgentOutcome`; no standard specialist mutates parent routing state directly.

- [ ] **Step 1: Write failing runtime and isolation tests**

```python
async def test_standard_specialist_is_created_per_invocation(factory):
    first = await factory.build("chat_agent", context_for(device_id="device-a"))
    second = await factory.build("chat_agent", context_for(device_id="device-b"))
    assert first is not second
    assert first.runtime_context.device_id == "device-a"
    assert second.runtime_context.device_id == "device-b"


async def test_specialist_response_returns_parent_validation_command(wrapper):
    command = await wrapper(state_with_active_agent("chat_agent"), runtime)
    assert command.goto == "validate_output"
    assert command.update["agent_outcome"].kind == "response"


async def test_worker_mode_never_appends_public_message(factory):
    result = await factory.invoke_worker("search_agent", worker_request())
    assert isinstance(result, WorkerResult)
    assert "public_messages" not in result.model_fields
```

Add middleware tests proving model/tool call limits, user/device/conversation scope, tool authorization before execution, usage recording, artifact offloading, and HITL interrupt/resume.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_specialist_runtime.py tests/test_specialist_middleware.py tests/test_custom_agents_graph.py tests/test_hitl_gate_policy.py tests/test_graph_tool_budget.py
```

- [ ] **Step 3: Define specialist configuration without execution loops**

Each current agent module exports a `SpecialistDefinition` instead of owning a model/tool loop:

```python
@dataclass(frozen=True)
class SpecialistDefinition:
    agent_id: str
    agent_type: AgentType
    model_config_key: str
    system_prompt_factory: Callable[[SpecialistRequest], Awaitable[str] | str]
    tool_factory: Callable[[SpecialistRequest], Awaitable[list[BaseTool]]]
    output_policy_ids: tuple[str, ...]
```

Move domain-specific prompt building, canvas snapshot handling, image delivery, and search tool selection into these factories. Dynamic custom agents produce the same definition from the attached authenticated descriptor. Do not cache a compiled graph across users or devices.

- [ ] **Step 4: Implement focused middleware and framework limits**

Use LangChain model/tool call limit middleware for call counts. Add small application middleware for:

- resolving a model using `IRuntimeModelResolver` and `ModelFactory`;
- resolving authorized dynamic tools from current user/device scope;
- checking tool policy before implementation invocation;
- translating current HITL policy into `interrupt()`/resume behavior;
- recording usage exactly once per provider attempt;
- collecting tool artifacts, images, canvas metadata, and offloaded results into private subgraph state.

Do not duplicate a `while tool_calls` loop. Do not implement auto-continuation. When the framework limit is reached, return `WorkflowError(code="agent_execution_limit", retriable=False, ...)`.

- [ ] **Step 5: Build and invoke `create_agent`**

```python
agent = create_agent(
    model=resolved_model,
    tools=authorized_tools,
    system_prompt=system_prompt,
    middleware=middleware_stack,
    context_schema=SpecialistRuntimeContext,
)
result = await agent.ainvoke(
    {"messages": invocation_messages},
    context=runtime_context,
    config=specialist_run_config,
)
```

Convert the private result to `ResponseOutcome` or propagate a `HandoffOutcome` produced by Task 6. The parent wrapper must validate that its requested ID equals `state.active_agent_id` before invocation.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_specialist_runtime.py tests/test_specialist_middleware.py tests/test_custom_agents_graph.py tests/test_hitl_gate_policy.py tests/test_graph_tool_budget.py tests/test_client_invocation_isolation.py tests/test_client_tool_isolation.py tests/test_model_usage_workflow_wiring.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/middleware.py app/ai/workflow/specialists.py app/ai/agents/chat_agent.py app/ai/agents/search_agent.py app/ai/agents/canvas_agent.py app/ai/agents/image_generator_agent.py app/ai/agents/custom_agent.py
git add app/ai/workflow/middleware.py app/ai/workflow/specialists.py app/ai/agents/chat_agent.py app/ai/agents/search_agent.py app/ai/agents/canvas_agent.py app/ai/agents/image_generator_agent.py app/ai/agents/custom_agent.py app/ai/custom_agent_runtime.py app/ai/deferred_tool_binding.py app/ai/tool_execution.py app/ai/hitl_config.py tests/test_specialist_runtime.py tests/test_specialist_middleware.py tests/test_custom_agents_graph.py tests/test_hitl_gate_policy.py tests/test_graph_tool_budget.py
git commit -m "refactor: run standard specialists with langchain agents"
```

---

### Task 6: Replace JSON handoff mutation with LangGraph parent commands

**Files:**
- Rewrite: `app/ai/hand_off_tool.py`
- Create: `app/ai/workflow/transitions.py`
- Modify: `app/ai/workflow/specialists.py`
- Modify: `app/ai/workflow/graph_builder.py`
- Create: `tests/test_agent_transitions.py`
- Replace: `tests/test_base_agent_dynamic_handoff.py`
- Modify: `tests/test_graph_handoff_streaming.py`
- Modify: `tests/test_rag_tool_loop_finalization.py`

**Interfaces:**
- Consumes: source agent, live reachable inventory, target selected through tool schema, originating tool call ID, active state, and transition-depth limit.
- Produces: `Command(graph=Command.PARENT, goto="resolve_transition")` plus a paired `ToolMessage`, followed by either an accepted transition command or model-visible rejection feedback.

- [ ] **Step 1: Write failing command and invariant tests**

```python
def test_handoff_tool_returns_parent_command_with_paired_tool_message():
    command = invoke_handoff(source="chat_agent", target="search_agent", call_id="call-1")
    assert command.graph == Command.PARENT
    assert command.goto == "resolve_transition"
    assert command.update["messages"][0].tool_call_id == "call-1"


async def test_accepted_handoff_preserves_initial_decision(resolver):
    state = routed_state(initial="chat_agent", active="chat_agent")
    command = await resolver(state_with_pending(state, target="search_agent"))
    assert command.update["active_agent_id"] == "search_agent"
    assert "routing_decision" not in command.update
    assert command.goto == "search_agent"


@pytest.mark.parametrize("case", ["self", "cycle", "detached", "unknown", "over_depth", "bad_call_id"])
async def test_rejected_handoff_returns_to_source_with_paired_feedback(resolver, case):
    command = await resolver(invalid_transition_state(case))
    assert command.goto == command.update["active_agent_id"]
    assert command.update["pending_transition"] is None
    assert isinstance(command.update["messages"][0], ToolMessage)
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_agent_transitions.py tests/test_base_agent_dynamic_handoff.py tests/test_graph_handoff_streaming.py
```

- [ ] **Step 3: Build the dynamic handoff tool from reachable inventory**

Use the current specialist's allowed target descriptors in the tool description and validate the chosen string at execution time. Inject `InjectedToolCallId` and return:

```python
return Command(
    graph=Command.PARENT,
    update={
        "pending_transition": PendingTransition(
            from_agent_id=source_agent_id,
            to_agent_id=target_agent_id,
            tool_call_id=tool_call_id,
            reason=reason,
        ),
        "messages": [ToolMessage(
            content="handoff_requested",
            name="hand_off",
            tool_call_id=tool_call_id,
        )],
    },
    goto="resolve_transition",
)
```

The tool must not JSON-encode a control message and must not mutate state in place.

- [ ] **Step 4: Implement the sole transition resolver**

Validate source/active identity, target existence/attachment/reachability, tool-call pairing, visited agents, and `max_handoff_delegation_depth`. An accepted transition appends one `AgentTransition` and routes to the target wrapper. A rejected transition replaces the request marker with a stable structured feedback payload and routes back to the source wrapper. Only control-plane IDs are compared; no message text is interpreted.

- [ ] **Step 5: Delete manual handoff parsing**

Remove `_apply_hand_off_if_present`, JSON parsing of `{"hand_off": ...}`, `context["handoff"]`, `delegation_count`, and every assignment to `selected_agent` from `app/ai/workflow/tool_loop.py`, `rag_loop.py`, `planning_loop.py`, and `graph.py`. Task 11 deletes these legacy loop modules once all other behavior is migrated.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_agent_transitions.py tests/test_base_agent_dynamic_handoff.py tests/test_graph_handoff_streaming.py tests/test_rag_tool_loop_finalization.py -k "handoff or transition"
.\.venv\Scripts\python.exe -m ruff check app/ai/hand_off_tool.py app/ai/workflow/transitions.py app/ai/workflow/specialists.py app/ai/workflow/graph_builder.py tests/test_agent_transitions.py tests/test_graph_handoff_streaming.py
git add app/ai/hand_off_tool.py app/ai/workflow/transitions.py app/ai/workflow/specialists.py app/ai/workflow/graph_builder.py app/ai/workflow/tool_loop.py app/ai/workflow/rag_loop.py app/ai/workflow/planning_loop.py app/ai/graph.py tests/test_agent_transitions.py tests/test_base_agent_dynamic_handoff.py tests/test_graph_handoff_streaming.py tests/test_rag_tool_loop_finalization.py
git commit -m "refactor: model handoffs as langgraph transitions"
```

---

### Task 7: Build one shared RAG execution graph with mandatory grounding

**Files:**
- Create: `app/ai/workflow/rag_execution.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `app/ai/rag_tools.py`
- Modify: `app/ai/rag_tool_actions.py`
- Modify: `app/services/rag_grounding.py`
- Modify: `app/observability/rag.py`
- Create: `tests/test_rag_execution_graph.py`
- Modify: `tests/test_rag_tool_loop_finalization.py`
- Modify: `tests/test_rag_grounding.py`
- Modify: `tests/test_rag_inline_worker_evidence.py`
- Modify: `tests/test_rag_artifact_visibility.py`

**Interfaces:**
- Consumes: `RagExecutionRequest` with objective, authenticated scope, allowed tools, model request, bounded history, and worker/public mode.
- Produces: `RagExecutionResult` with grounded content or abstention, server-owned evidence, artifacts, images, usage, and validation metadata.

- [ ] **Step 1: Write failing shared-path and grounding tests**

```python
async def test_top_level_and_worker_use_same_rag_factory(workflow, planning):
    await workflow.invoke_top_level_rag(rag_request())
    await planning.invoke_rag_worker(rag_worker_request())
    assert workflow.rag_factory.build.call_count == 2
    assert planning.rag_factory is workflow.rag_factory


async def test_invalid_citations_regenerate_once_then_abstain(rag_graph, fake_model):
    fake_model.answers = [answer_with("E99"), answer_with("E98")]
    result = await rag_graph.ainvoke(rag_request_with_evidence("E1"))
    assert result["rag_result"].abstained is True
    assert result["rag_result"].grounding.regeneration_count == 1
    assert fake_model.final_answer_calls == 2


async def test_unknown_evidence_id_never_reaches_public_result(rag_graph):
    result = await rag_graph.ainvoke(rag_request_with_model_citation("E404"))
    assert "E404" not in result["rag_result"].content
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_execution_graph.py tests/test_rag_tool_loop_finalization.py tests/test_rag_grounding.py tests/test_rag_inline_worker_evidence.py
```

- [ ] **Step 3: Define RAG state and topology**

Build a private `StateGraph(RagExecutionState)` with:

```text
START -> prepare_request -> rag_model
rag_model --tool calls--> rag_tools -> collect_evidence -> rag_model
rag_model --answer------> validate_grounding
validate_grounding --valid------> package_result -> END
validate_grounding --first fail-> regenerate -> validate_grounding
validate_grounding --second fail> abstain -> package_result -> END
```

Use `ToolNode` for execution and middleware/wrappers for authorization, approval, evidence collection, artifact offloading, and error normalization. The graph receives runtime context; per-invocation subgraphs inherit the parent checkpointer.

- [ ] **Step 4: Move RAG behavior into the shared graph**

Move model invocation and constrained regeneration out of the `RAGAgent` loop into focused functions consumed by `rag_execution.py`. Reuse `execute_search_documents_action`, evidence packs, token budgets, image provenance, and `GroundedAnswerGate`. Preserve server-owned evidence IDs across every model round.

Remove the `rag_grounded_answer_gate_enabled` branch: construction always installs the gate, validation always runs when evidence is present, and metrics record `accepted | regenerated | abstained`. There is no shadow-only outcome.

- [ ] **Step 5: Adapt top-level RAG to `AgentOutcome`**

The parent `rag_agent` wrapper invokes the shared graph and converts `RagExecutionResult` to `ResponseOutcome`; it never reaches `END`. Handoffs from RAG use Task 6's parent command, not the RAG tool-result router.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rag_execution_graph.py tests/test_rag_tool_loop_finalization.py tests/test_rag_grounding.py tests/test_rag_inline_worker_evidence.py tests/test_rag_artifact_visibility.py tests/test_rag_multi_user_isolation.py tests/test_rag_agent.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/rag_execution.py app/ai/agents/rag_agent.py app/ai/rag_tools.py app/ai/rag_tool_actions.py app/services/rag_grounding.py app/observability/rag.py tests/test_rag_execution_graph.py
git add app/ai/workflow/rag_execution.py app/ai/agents/rag_agent.py app/ai/rag_tools.py app/ai/rag_tool_actions.py app/services/rag_grounding.py app/observability/rag.py tests/test_rag_execution_graph.py tests/test_rag_tool_loop_finalization.py tests/test_rag_grounding.py tests/test_rag_inline_worker_evidence.py tests/test_rag_artifact_visibility.py
git commit -m "refactor: unify rag execution and enforce grounding"
```

---

### Task 8: Rebuild Planning as an orchestrator with typed `Send` workers

**Files:**
- Create: `app/ai/workflow/planning_execution.py`
- Modify: `app/ai/planning_subagents.py`
- Modify: `app/ai/planning_runtime_adapter.py`
- Modify: `app/ai/agents/planning_agent.py`
- Modify: `app/ai/workflow/specialists.py`
- Modify: `app/ai/workflow/graph_builder.py`
- Create: `tests/test_planning_execution_graph.py`
- Modify: `tests/test_graph_planning_subagents.py`
- Modify: `tests/test_planning_subagents.py`
- Modify: `tests/test_custom_agents_planning.py`

**Interfaces:**
- Consumes: a semantically routed Planning request and typed worker tasks.
- Produces: isolated `WorkerResult` values collected by reducers and one synthesized `ResponseOutcome` for the parent validator.

- [ ] **Step 1: Write failing dispatch/isolation tests**

```python
def test_dispatch_returns_send_for_each_independent_task():
    sends = dispatch_workers(planning_state_with_three_tasks())
    assert [send.node for send in sends] == ["worker", "worker", "worker"]
    assert {send.arg["task"].task_id for send in sends} == {"t1", "t2", "t3"}


async def test_worker_result_is_typed_and_private(planning_graph):
    result = await planning_graph.ainvoke(plan_with_search_worker())
    worker = result["worker_results"][0]
    assert isinstance(worker, WorkerResult)
    assert worker.agent_id == "search_agent"
    assert "public_messages" not in worker.model_fields


async def test_planning_rag_worker_uses_shared_grounding_graph(planning_graph, rag_factory):
    await planning_graph.ainvoke(plan_with_rag_worker())
    rag_factory.build.assert_called_once()
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_planning_execution_graph.py tests/test_graph_planning_subagents.py tests/test_planning_subagents.py tests/test_custom_agents_planning.py
```

- [ ] **Step 3: Define task and result contracts**

Use `WorkerTask(task_id, objective, agent_id, allowed_tool_ids, model_request, related_todo_ids)` and the approved `WorkerResult`. Worker state contains only the task, bounded parent context, authenticated scope, and private messages. The reducer rejects duplicate task IDs rather than silently overwriting them.

- [ ] **Step 4: Implement `Send` fan-out and worker execution**

```python
def dispatch_workers(state: PlanningState) -> list[Send]:
    return [Send("worker", {"task": task, "runtime_request": state["runtime_request"]})
            for task in state["worker_tasks"]]
```

The `worker` node resolves the requested specialist from the same live inventory. Standard workers call `SpecialistFactory.invoke_worker`; RAG workers call `RagExecutionGraphFactory`; recursive Planning is rejected as a typed failed `WorkerResult`. Workers may return `awaiting_approval` but cannot issue a parent-level handoff.

- [ ] **Step 5: Implement Planning synthesis**

The Planning model receives ordered typed results, artifacts, and evidence. It returns one `ResponseOutcome`. If any result contains evidence, propagate the complete server-owned evidence set and add the `rag_grounding` output policy so Task 9 revalidates the public synthesis.

- [ ] **Step 6: Remove inline isolated execution**

Delete the RAG-specific branch and the generic manual loop in `_run_agent_in_isolated_context` at current `app/ai/graph.py:1784`. Remove `dispatch_subagents` JSON payload parsing that duplicates `Send`; retain only domain helpers still called by the new Planning graph.

- [ ] **Step 7: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_planning_execution_graph.py tests/test_graph_planning_subagents.py tests/test_planning_subagents.py tests/test_custom_agents_planning.py tests/test_rag_inline_worker_evidence.py tests/test_event_streaming_subagents.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/planning_execution.py app/ai/planning_subagents.py app/ai/planning_runtime_adapter.py app/ai/agents/planning_agent.py app/ai/workflow/specialists.py tests/test_planning_execution_graph.py
git add app/ai/workflow/planning_execution.py app/ai/planning_subagents.py app/ai/planning_runtime_adapter.py app/ai/agents/planning_agent.py app/ai/workflow/specialists.py app/ai/workflow/graph_builder.py app/ai/graph.py tests/test_planning_execution_graph.py tests/test_graph_planning_subagents.py tests/test_planning_subagents.py tests/test_custom_agents_planning.py tests/test_rag_inline_worker_evidence.py
git commit -m "refactor: orchestrate planning workers with langgraph send"
```

---

### Task 9: Add provenance-based validation and universal finalization

**Files:**
- Create: `app/ai/workflow/finalization.py`
- Modify: `app/ai/agent_metadata.py`
- Modify: `app/ai/canvas_state.py`
- Modify: `app/ai/selected_image_sink.py`
- Modify: `app/core/rich_response.py`
- Modify: `app/ai/workflow/graph_builder.py`
- Create: `tests/test_public_response_finalizer.py`
- Create: `tests/test_output_validation.py`
- Modify: `tests/test_agent_metadata.py`
- Modify: `tests/test_rich_response_metadata.py`
- Modify: `tests/test_selected_image_sink.py`
- Modify: `tests/test_production_workflow_graph.py`

**Interfaces:**
- Consumes: `AgentOutcome`, immutable routing decision, active/final agent identity, transitions, artifacts, images, canvas/planning metadata, evidence, and reserved assistant message ID.
- Produces: either a validated outcome followed by exactly one terminal `AgentResponse`/`AIMessage`, or a typed terminal workflow error with no published answer.

- [ ] **Step 1: Write failing validation/finalization tests**

```python
@pytest.mark.parametrize("agent_id", [
    "chat_agent", "search_agent", "image_generator_agent", "canvas_agent",
    "planning_agent", "rag_agent", "custom_agent:123",
])
async def test_every_public_outcome_traverses_validation_and_finalization(workflow, agent_id):
    response = await workflow.invoke_with_forced_route(agent_id)
    assert response.metadata["validation"]["passed"] is True
    assert response.metadata["workflow"]["execution_phase"] == "completed"
    assert workflow.finalizer.calls == 1


def test_finalizer_appends_exactly_one_terminal_ai_message(finalizer):
    state = finalizer.finalize(validated_state())
    terminal = [m for m in state["messages"] if isinstance(m, AIMessage) and m.id == ASSISTANT_ID]
    assert len(terminal) == 1


async def test_planning_synthesis_with_rag_evidence_is_revalidated(validator):
    outcome = planning_outcome(content="Unsupported [E9]", evidence=[evidence("E1")])
    validated = await validator.validate(outcome)
    assert validated.grounding.outcome in {"regenerated", "abstained"}
    assert "E9" not in validated.content
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_public_response_finalizer.py tests/test_output_validation.py tests/test_agent_metadata.py tests/test_production_workflow_graph.py
```

- [ ] **Step 3: Implement a provenance policy registry**

```python
class OutputPolicy(Protocol):
    policy_id: str
    async def validate(self, outcome: ResponseOutcome, context: ValidationContext) \
        -> ResponseOutcome: ...


POLICIES = {
    "rag_grounding": RagGroundingPolicy(...),
    "artifact_provenance": ArtifactProvenancePolicy(...),
    "tool_message_pairing": ToolMessagePairingPolicy(...),
    "canvas_contract": CanvasOutputPolicy(...),
    "image_delivery": ImageDeliveryPolicy(...),
    "public_content": PublicContentPolicy(...),
}
```

Choose policies from `outcome.provenance` and declared `output_policy_ids`, not merely the final agent ID. Evidence presence always activates `rag_grounding`; artifacts/images always activate their provenance validators.

- [ ] **Step 4: Implement worker and public finalizers**

`WorkerOutputFinalizer` validates the same evidence/artifact contracts but returns `WorkerResult` without a public ID or conversation-history write.

`PublicResponseFinalizer` must:

1. require `assistant_message_id`, `routing_decision`, and a successful validated outcome;
2. set `final_agent_id = active_agent_id`;
3. build metadata containing initial, active, and final IDs plus ordered transition history;
4. merge only validated artifacts, rich items, images, canvas, planning, and grounding data;
5. normalize rich-response placement;
6. append exactly one `AIMessage(id=assistant_message_id, ...)`;
7. create the service-facing `AgentResponse`;
8. record final usage/metrics;
9. return a state update with `execution_phase="completed"`; the graph's sole static terminal edge is `finalize -> END`.

For failed state, the same finalizer records failure telemetry, verifies that no public assistant message was appended, preserves the `WorkflowError`, and returns `execution_phase="failed"` for the same static terminal edge.

Do not scan prior messages to recover a plausible response.

- [ ] **Step 5: Replace selected-agent metadata**

Change `attach_agent_metadata` to accept `routing_decision`, `active_agent_id`, `final_agent_id`, and `agent_history`. Public metadata must preserve the initial decision even after handoffs. Delete `selected_agent_id` and the metadata source value `selected_agent`.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_public_response_finalizer.py tests/test_output_validation.py tests/test_agent_metadata.py tests/test_rich_response_metadata.py tests/test_selected_image_sink.py tests/test_production_workflow_graph.py tests/test_rag_grounding.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/finalization.py app/ai/agent_metadata.py app/ai/canvas_state.py app/ai/selected_image_sink.py app/core/rich_response.py tests/test_public_response_finalizer.py tests/test_output_validation.py
git add app/ai/workflow/finalization.py app/ai/agent_metadata.py app/ai/canvas_state.py app/ai/selected_image_sink.py app/core/rich_response.py app/ai/workflow/graph_builder.py tests/test_public_response_finalizer.py tests/test_output_validation.py tests/test_agent_metadata.py tests/test_rich_response_metadata.py tests/test_selected_image_sink.py tests/test_production_workflow_graph.py
git commit -m "feat: validate and finalize every public response"
```

---

### Task 10: Project graph-native routing, transitions, errors, and resumes to the public stream

**Files:**
- Modify: `app/services/event_streaming/events.py`
- Modify: `app/services/event_streaming/langchain_v3.py`
- Modify: `app/services/event_streaming/graph_public_projection.py`
- Modify: `app/services/event_streaming/subagents.py`
- Modify: `app/services/event_streaming/ai_sdk_v6.py`
- Modify: `app/services/ai_service.py`
- Modify: `app/schemas/workflow.py`
- Modify: `app/utils/exception_handler.py`
- Modify: `app/api/messages.py`
- Create: `tests/test_workflow_error_contract.py`
- Modify: `tests/test_graph_stream_projection.py`
- Modify: `tests/test_event_streaming_langgraph_normalizer.py`
- Modify: `tests/test_event_streaming_subagents.py`
- Modify: `tests/test_message_stream_errors.py`
- Modify: `tests/test_graph_resume_image_stream.py`

**Interfaces:**
- Consumes: graph updates/messages/interrupts from initial execution or `Command(resume=...)`.
- Produces: ordered v3/public events with one selection per accepted route/transition, isolated worker events, one completion, or one typed error.

- [ ] **Step 1: Write failing ordering and error tests**

```python
async def test_stream_emits_route_handoff_and_complete_once(stream):
    events = [event async for event in stream]
    assert [(e.type, e.agent) for e in events if e.type == "agent_selected"] == [
        ("agent_selected", "chat_agent"),
        ("agent_selected", "search_agent"),
    ]
    assert sum(e.type == "complete" for e in events) == 1
    assert not [e for e in events if e.type == "error"]


async def test_router_failure_is_typed_and_has_no_complete_event(stream):
    events = [event async for event in stream]
    error = next(e for e in events if e.type == "error")
    assert error.data["code"] == "routing_provider_unavailable"
    assert error.data["retriable"] is True
    assert not [e for e in events if e.type == "complete"]


async def test_resume_continues_without_second_initial_selection(stream_after_resume):
    events = [event async for event in stream_after_resume]
    assert not [e for e in events if e.type == "agent_selected" and e.data["cause"] == "route"]
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_error_contract.py tests/test_graph_stream_projection.py tests/test_event_streaming_langgraph_normalizer.py tests/test_message_stream_errors.py tests/test_graph_resume_image_stream.py
```

- [ ] **Step 3: Derive events from new state updates**

Emit initial `agent_selected` when `routing_decision` first appears, with `data={"cause": "route", "confidence": ..., "inventory_version": ...}`. Emit subsequent selections only for newly appended accepted `AgentTransition` values, with `data={"cause": "handoff", "source_agent": ..., "tool_call_id": ...}`.

Stop comparing `selected_agent` snapshots. Track the routing-decision identity and transition count in `StreamProjectionContext`. Planning worker deltas remain `subagent_*` events and never become `message_delta` in the main answer stream.

- [ ] **Step 4: Replace string errors with the stable payload**

Change both AI-layer and service-layer response schemas from `error: str | None` to `error: WorkflowError | None`. The API adapter may add localized display text, but `code`, `retriable`, `request_id`, and structured details remain intact. Remove logic that prefixes strings with `"Error:"` or maps a missing response to an English generic answer.

- [ ] **Step 5: Make completion finalizer-driven**

Emit `complete` only after observing `execution_phase="completed"` plus the finalizer-owned response. Emit `error` only after `execution_phase="failed"`. Delete terminal content recovery from accumulated stream chunks and checkpoint message scanning.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_error_contract.py tests/test_graph_stream_projection.py tests/test_event_streaming_langgraph_normalizer.py tests/test_event_streaming_subagents.py tests/test_message_stream_errors.py tests/test_graph_resume_image_stream.py tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py
.\.venv\Scripts\python.exe -m ruff check app/services/event_streaming app/services/ai_service.py app/schemas/workflow.py app/utils/exception_handler.py app/api/messages.py tests/test_workflow_error_contract.py tests/test_graph_stream_projection.py
git add app/services/event_streaming app/services/ai_service.py app/schemas/workflow.py app/utils/exception_handler.py app/api/messages.py tests/test_workflow_error_contract.py tests/test_graph_stream_projection.py tests/test_event_streaming_langgraph_normalizer.py tests/test_event_streaming_subagents.py tests/test_message_stream_errors.py tests/test_graph_resume_image_stream.py
git commit -m "refactor: stream graph-native routing and typed failures"
```

---

### Task 11: Perform the breaking cutover and delete legacy routing/execution paths

**Files:**
- Rewrite: `app/ai/graph.py`
- Modify: `app/ai/schemas.py`
- Modify: `app/ai/workflow/__init__.py`
- Modify: `app/ai/agents/__init__.py`
- Delete: `app/ai/workflow/tool_loop.py`
- Delete: `app/ai/workflow/rag_loop.py`
- Delete: `app/ai/workflow/planning_loop.py`
- Delete: `app/ai/workflow/custom_agents.py`
- Delete: `app/ai/agents/base_agent.py`
- Modify: `app/ai/agent_config.py`
- Modify: `app/core/config.py`
- Modify: `app/ai/langgraph.json`
- Delete: `tests/test_custom_agent_stickiness.py`
- Replace: `tests/test_graph_no_fast_path_helpers.py`
- Create: `tests/test_routing_legacy_removal.py`
- Modify: `tests/test_production_readiness_contract.py`
- Modify: `tests/test_rag_dead_code_cleanup.py`

**Interfaces:**
- Consumes: the completed v2 workflow components from Tasks 1–10.
- Produces: one production runtime with no reachable or retained legacy compatibility implementation.

- [ ] **Step 1: Add a failing source and topology deletion contract**

```python
FORBIDDEN_RUNTIME_TOKENS = (
    "selected_agent", "last_agent", "delegation_count",
    "_sticky_custom_agent_for_followup", "_match_explicit_custom_agent",
    "_extract_agent_name", "_apply_hand_off_if_present",
    "_run_agent_in_isolated_context", "continuation_round",
    "rag_grounded_answer_gate_enabled", "_recover_terminal_response",
)


def test_legacy_routing_and_execution_tokens_are_absent():
    source = read_runtime_source(APP_AI_PATHS)
    for token in FORBIDDEN_RUNTIME_TOKENS:
        assert token not in source


def test_removed_loop_modules_do_not_exist():
    for path in REMOVED_LOOP_PATHS:
        assert not path.exists()
```

The scan excludes migrations, historical docs/specs/plans, and fixture prose; it includes `app/ai`, workflow streaming projection, service schemas, and live tests.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_legacy_removal.py tests/test_production_readiness_contract.py tests/test_rag_dead_code_cleanup.py
```

- [ ] **Step 3: Shrink `MultiAgentWorkflow` to the runtime adapter**

Keep only dependency initialization, request-to-state construction, `graph.ainvoke`, graph event streaming, `Command(resume=...)`, checkpoint compaction, and history-cache invalidation. Move node logic to the focused workflow modules. Remove mixin inheritance and manual agent registries that contain executing agent objects.

- [ ] **Step 4: Remove old state and semantic overrides**

Delete the old `GraphState`, `GraphStateView.selected_agent`, `last_agent`, `delegation_count`, custom stickiness, canvas preselection, planning force, explicit-name matching, no-document chat substitution, preselected-agent exit, and auto-continuation. Re-export `WorkflowState` as the sole graph state type.

- [ ] **Step 5: Delete duplicated loops and fallback settings**

Delete the four workflow mixin modules and obsolete `BaseAgent` loop after all imports have migrated. Delete the old router model config path if it is superseded by runtime model configuration. Remove the shadow grounding setting and continuation settings that no live component reads. Remove graph topology entries for `tools`, `approval`, `rag_tools`, and `planning_tools`; those now live inside specialist subgraphs.

- [ ] **Step 6: Remove superseded tests, update living contracts, and search**

Delete `tests/test_custom_agent_stickiness.py`; replace expectations in graph, RAG, planning, handoff, and stream tests with v2 contracts. Then run:

```powershell
rg -n "selected_agent|last_agent|delegation_count|_sticky_custom_agent_for_followup|_match_explicit_custom_agent|_extract_agent_name|_apply_hand_off_if_present|_run_agent_in_isolated_context|continuation_round|rag_grounded_answer_gate_enabled|_recover_terminal_response" app tests
```

Expected: no runtime matches. A test filename or historical test description is not an acceptable retained compatibility path; update it.

- [ ] **Step 7: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_legacy_removal.py tests/test_production_readiness_contract.py tests/test_rag_dead_code_cleanup.py tests/test_production_workflow_graph.py tests/test_custom_agents_graph.py tests/test_graph_planning_subagents.py tests/test_rag_execution_graph.py
.\.venv\Scripts\python.exe -m ruff check app/ai app/services/event_streaming app/services/ai_service.py tests/test_routing_legacy_removal.py tests/test_production_readiness_contract.py tests/test_rag_dead_code_cleanup.py
git rm app/ai/workflow/tool_loop.py app/ai/workflow/rag_loop.py app/ai/workflow/planning_loop.py app/ai/workflow/custom_agents.py app/ai/agents/base_agent.py tests/test_custom_agent_stickiness.py
git add app/ai/graph.py app/ai/schemas.py app/ai/workflow/__init__.py app/ai/agents/__init__.py app/ai/agent_config.py app/core/config.py app/ai/langgraph.json tests/test_graph_no_fast_path_helpers.py tests/test_routing_legacy_removal.py tests/test_production_readiness_contract.py tests/test_rag_dead_code_cleanup.py
git commit -m "refactor: remove legacy routing and duplicated agent loops"
```

Expected: the runtime contains only v2 routing/execution; old checkpoint data remains stored but cannot be loaded because the namespace changed.

---

### Task 12: Add routing observability and a versioned multilingual evaluation gate

**Files:**
- Modify: `app/observability/routing.py`
- Create: `app/evaluation/routing/__init__.py`
- Create: `app/evaluation/routing/contracts.py`
- Create: `app/evaluation/routing/metrics.py`
- Create: `app/evaluation/routing/release_gates.py`
- Create: `app/evaluation/routing/target.py`
- Create: `eval/routing/golden_v1.jsonl`
- Create: `eval/routing/release_gates.json`
- Create: `scripts/evaluate_routing.py`
- Create: `tests/test_routing_evaluation_contracts.py`
- Create: `tests/test_routing_evaluation_metrics.py`
- Create: `tests/test_routing_observability.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: live-model structured routing results, labels containing one or more acceptable targets, language/category metadata, and content-free runtime telemetry.
- Produces: macro-F1, per-language accuracy/gap, schema success before/after retry, route/handoff/failure metrics, and a pass/fail release report tied to provider/model version.

- [ ] **Step 1: Write failing dataset/metric/telemetry tests**

```python
def test_multilingual_dataset_has_required_categories_and_scripts(dataset):
    assert REQUIRED_CATEGORIES <= {row.metadata.category for row in dataset}
    assert {"en", "th", "vi", "zh", "ja", "ar", "mixed"} <= {
        row.metadata.language for row in dataset
    }
    assert all(row.reference.acceptable_agent_ids for row in dataset)


def test_release_gate_checks_language_gap():
    report = report_for(language_accuracy={"en": 0.94, "th": 0.88})
    assert evaluate_release_gates(report).passed is False
    assert "language_gap" in evaluate_release_gates(report).failed_gates


def test_routing_metrics_do_not_record_message_content(recorder):
    recorder.routing_completed(event_with_message("secret prompt"))
    assert "secret prompt" not in json.dumps(recorder.export())
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_evaluation_contracts.py tests/test_routing_evaluation_metrics.py tests/test_routing_observability.py
```

- [ ] **Step 3: Implement deterministic evaluation contracts and metrics**

Each JSONL row contains `id`, `inputs` (message plus bounded context fixture), `reference.acceptable_agent_ids`, and metadata (`language`, `category`, `difficulty`). Seed at least 210 human-reviewed cases: at least 30 each for English, Thai, Vietnamese, Chinese, Japanese, Arabic, and mixed-language input, with every intent category represented by at least 10 cases. Cover general chat, current information/search, document/RAG, canvas continuation and departure, planning with/without an existing plan, explicit and implicit custom-agent requests, image generation, ambiguous follow-ups, and mixed intents. Labels may list multiple acceptable routes.

Implement exact accuracy against acceptable sets, per-agent precision/recall/F1, macro-F1, confusion matrix, per-language accuracy, English gap, first-attempt schema rate, after-retry schema rate, latency percentiles, and failure-code counts.

- [ ] **Step 4: Encode the approved release gates**

`eval/routing/release_gates.json`:

```json
{
  "macro_f1_min": 0.90,
  "per_language_accuracy_min": 0.85,
  "max_gap_below_english": 0.05,
  "first_attempt_structured_success_min": 0.99,
  "after_retry_structured_success_min": 0.999,
  "silent_chat_substitutions_max": 0,
  "finalizer_bypasses_max": 0,
  "unknown_published_evidence_ids_max": 0
}
```

- [ ] **Step 5: Add the live evaluation CLI**

Register `routing_live: calls the configured live router model` in pytest markers. `scripts/evaluate_routing.py` accepts `--dataset`, `--output`, `--provider`, `--model`, and `--compare-baseline`; it records provider/model identifiers, inventory version, timestamp, and attempt counts. It calls `RoutingService` through `app/evaluation/routing/target.py`, never a second evaluation-only router.

- [ ] **Step 6: Add content-free runtime metrics**

Record route volume, latency, attempt/schema result, target races, accepted/rejected handoffs, correction rate, transition depth, execution limits, worker status, grounding outcomes, finalizer policies, and terminal error codes. Exclude prompt text, raw document data, credentials, and unbounded custom-agent descriptions.

- [ ] **Step 7: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_evaluation_contracts.py tests/test_routing_evaluation_metrics.py tests/test_routing_observability.py
.\.venv\Scripts\python.exe -m ruff check app/evaluation/routing app/observability/routing.py scripts/evaluate_routing.py tests/test_routing_evaluation_contracts.py tests/test_routing_evaluation_metrics.py tests/test_routing_observability.py
.\.venv\Scripts\python.exe scripts/evaluate_routing.py --dataset eval/routing/golden_v1.jsonl --output .artifacts/routing-eval-v1.json
git add app/evaluation/routing app/observability/routing.py eval/routing scripts/evaluate_routing.py tests/test_routing_evaluation_contracts.py tests/test_routing_evaluation_metrics.py tests/test_routing_observability.py pyproject.toml
git commit -m "test: add multilingual routing release gates"
```

Expected: deterministic tests pass. The live command must pass every threshold before production deployment; if credentials are intentionally unavailable, record the job as not run rather than treating it as a pass.

---

### Task 13: Add concurrency, checkpoint-version, and end-to-end production tests

**Files:**
- Create: `tests/test_workflow_concurrency.py`
- Create: `tests/test_workflow_checkpoint_v2.py`
- Create: `tests/test_workflow_end_to_end.py`
- Modify: `tests/test_client_invocation_isolation.py`
- Modify: `tests/test_hitl_backend_regressions.py`
- Modify: `tests/test_message_service_event_streaming.py`
- Modify: `tests/test_model_usage_callsite_inventory.py`
- Modify: `docs/operations/tool-execution-policy.md`
- Create: `docs/operations/routing-v2-rollout.md`

**Interfaces:**
- Consumes: the complete v2 runtime with fake provider/tool integrations and a real test checkpointer where supported.
- Produces: proof of turn isolation, durable resume, finalizer coverage, error behavior, and an operational canary/rollback procedure.

- [ ] **Step 1: Write failing concurrent-turn tests**

```python
async def test_concurrent_turns_do_not_leak_agent_or_device_context(workflow):
    left, right = await asyncio.gather(
        workflow.execute_request(request("A", user="u1", device="d1", target="search_agent")),
        workflow.execute_request(request("B", user="u2", device="d2", target="custom_agent:b")),
    )
    assert left.metadata["workflow"]["final_agent_id"] == "search_agent"
    assert right.metadata["workflow"]["final_agent_id"] == "custom_agent:b"
    assert left.metadata["device_id"] == "d1"
    assert right.metadata["device_id"] == "d2"


async def test_old_namespace_checkpoint_is_ignored(workflow, checkpointer):
    await seed_checkpoint(checkpointer, thread_id="conversation-1", selected_agent="canvas_agent")
    await workflow.execute_request(request("new turn", thread_id="conversation-1"))
    assert workflow.routing_service.route.await_count == 1
    assert await checkpointer.aget_tuple(config_for("routing-v2:conversation-1")) is not None
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_workflow_end_to_end.py
```

- [ ] **Step 3: Cover the production scenarios**

Add deterministic end-to-end cases for every base specialist and a dynamic custom specialist; one and two handoffs; invalid/cyclic handoffs; RAG regeneration/abstention; Planning parallel workers including RAG; HITL approve/edit/reject and durable resume; router timeout/provider/schema/target failures; specialist limit failure; finalization failure; streamed and non-streamed parity; exactly one durable assistant message; and no cross-user/device/tool/evidence leakage.

- [ ] **Step 4: Document deployment and rollback**

`routing-v2-rollout.md` must specify:

1. startup structured-output capability check;
2. deterministic suite and live evaluation artifacts required for canary;
3. canary dashboards/alerts for router error rate, p95 latency, invalid schema, target race, handoff correction, grounding abstention, and finalizer failures;
4. verification that new checkpoints use `routing-v2:`;
5. rollback by deploying the prior artifact—not by enabling old branches in the new code;
6. retention of old checkpoints for audit and their exclusion from v2 reads.

- [ ] **Step 5: Run the focused production suite and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_workflow_end_to_end.py tests/test_client_invocation_isolation.py tests/test_hitl_backend_regressions.py tests/test_message_service_event_streaming.py tests/test_model_usage_callsite_inventory.py
.\.venv\Scripts\python.exe -m ruff check app tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_workflow_end_to_end.py
git add tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_workflow_end_to_end.py tests/test_client_invocation_isolation.py tests/test_hitl_backend_regressions.py tests/test_message_service_event_streaming.py tests/test_model_usage_callsite_inventory.py tests/fixtures/model_usage_callsite_manifest.json docs/operations/tool-execution-policy.md docs/operations/routing-v2-rollout.md
git commit -m "test: verify routing v2 production behavior"
```

---

### Task 14: Run the final acceptance gate

**Files:**
- Modify only if a verification failure reveals a defect in an in-scope file.

- [ ] **Step 1: Run formatting and lint**

```powershell
.\.venv\Scripts\python.exe -m ruff format --check app tests scripts
.\.venv\Scripts\python.exe -m ruff check app tests scripts
```

Expected: zero violations. If formatting fails, run `ruff format` only on files changed by this implementation, then rerun both commands.

- [ ] **Step 2: Run routing/workflow/RAG/streaming/HITL suites**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_contracts.py tests/test_workflow_state.py tests/test_routing_inventory.py tests/test_routing_context.py tests/test_router.py tests/test_routing_service.py tests/test_routing_startup_validation.py tests/test_production_workflow_graph.py tests/test_specialist_runtime.py tests/test_specialist_middleware.py tests/test_agent_transitions.py tests/test_rag_execution_graph.py tests/test_planning_execution_graph.py tests/test_output_validation.py tests/test_public_response_finalizer.py tests/test_workflow_error_contract.py tests/test_graph_stream_projection.py tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_workflow_end_to_end.py tests/test_hitl_backend_regressions.py tests/test_rag_grounding.py
```

Expected: all pass; no process crash. If Windows still raises the previously observed native `pyarrow` access violation, record the exact command/exit code, run the same suite in the Linux CI/container image, and do not mark the gate passed until that environment succeeds.

- [ ] **Step 3: Run the full non-live suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not live_provider and not routing_live"
```

Expected: all non-live tests pass.

- [ ] **Step 4: Run legacy-removal searches**

```powershell
rg -n "selected_agent|last_agent|delegation_count|_sticky_custom_agent_for_followup|_match_explicit_custom_agent|_extract_agent_name|_apply_hand_off_if_present|_run_agent_in_isolated_context|continuation_round|rag_grounded_answer_gate_enabled|_recover_terminal_response" app tests
rg -n "google\.genai|from google import genai|return \"chat_agent\"" app/ai/workflow/routing.py app/ai/graph.py
```

Expected: no matches. The only allowed references are this implementation plan, the approved design, archived git history, and operational migration notes explaining removed names.

- [ ] **Step 5: Run and archive the live routing evaluation**

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_routing.py --dataset eval/routing/golden_v1.jsonl --output .artifacts/routing-eval-v1.json
```

Expected: macro-F1 ≥ 0.90; every represented language ≥ 0.85; no language more than 0.05 below English; first-attempt structured success ≥ 0.99; after-retry success ≥ 0.999; zero silent chat substitutions; zero finalizer bypasses; zero unknown published evidence IDs.

- [ ] **Step 6: Review the final diff and commit verification-only fixes**

```powershell
git status --short
git diff --stat 643404d..HEAD
git diff --check
```

Confirm the diff contains no credentials, captured prompts, generated evaluation secrets, or `.artifacts` output. Commit only actual code/test/document fixes; do not commit the live evaluation output if it contains request text.

## Final Acceptance Checklist

- [ ] Every new user turn calls the configured LLM router exactly once.
- [ ] No application routing branch interprets words, names, scripts, or language.
- [ ] Canvas, custom-agent continuity/names, documents, and existing plans are model context, not preselection.
- [ ] Router failure yields a typed retriable error and never changes agent/provider/model.
- [ ] `routing_decision`, `active_agent_id`, `final_agent_id`, and append-only `agent_history` are the only top-level routing/execution identity fields.
- [ ] Handoffs use paired `ToolMessage` plus `Command(graph=Command.PARENT, ...)` and preserve the initial decision.
- [ ] Standard specialists use per-invocation `create_agent` subgraphs.
- [ ] Top-level and Planning-worker RAG use the same graph factory and mandatory grounding policy.
- [ ] Planning workers use `Send`, return typed private results, and cannot publish or hand off at parent scope.
- [ ] Every public response traverses validation and universal finalization; only finalization reaches `END`.
- [ ] Streaming derives routing/handoff events from graph state and never pre-runs a node.
- [ ] Resume continues the checkpointed node without routing again.
- [ ] Legacy routing, fallback, continuation, duplicated RAG, shadow grounding, and stale recovery code are deleted.
- [ ] Deterministic, concurrency, streaming, HITL, grounding, full non-live, and live multilingual gates pass.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-26-production-routing-refactor.md`.

Two execution options:

1. **Subagent-Driven (recommended):** Use `superpowers:subagent-driven-development` in this session, assign one task at a time, and perform spec/code reviews between tasks.
2. **Inline execution:** Use `superpowers:executing-plans` in a fresh session and execute sequentially with the listed verification checkpoints.
