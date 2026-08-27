# Production Routing Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current multilingual-fragile routing shortcuts, mutable `selected_agent` control flow, duplicated RAG execution, and direct-to-`END` agent paths with one provider-agnostic LLM router, typed LangGraph transitions, shared specialist subgraphs, mandatory grounding, and universal public-response finalization.

**Architecture:** Every new user turn gets a unique turn ID and checkpoint thread, then enters one `route` node. The strictly resolved router model returns a schema-constrained `RoutingDecision`; Python validates only inventory and control-plane invariants and never interprets message language or intent. The parent graph records an immutable initial decision, tracks the active agent and append-only transitions separately, invokes per-request specialist subgraphs, buffers public answer text until validation succeeds, and uses one shared RAG graph for top-level and Planning-worker retrieval. Router failure is a typed retriable workflow failure—never an agent or provider substitution.

**Tech Stack:** Python 3.10+, Pydantic 2, LangChain 1.3 (`create_agent`, structured output, middleware), LangGraph 1.2 (`StateGraph`, `Command`, `Send`, `ToolNode`, `interrupt`, PostgreSQL checkpointer), FastAPI, SQLAlchemy/PostgreSQL, LangSmith-compatible tracing, pytest, pytest-asyncio, Ruff.

**Approved design:** `docs/superpowers/specs/2026-08-26-production-routing-refactor-design.md`

**Design reconciliation:** This implementation plan is authoritative where it tightens the approved design after codebase review: checkpoints are per turn (`routing-v2:{conversation_id}:{turn_id}`), HITL is graph interrupt state rather than `WorkerResult(status="awaiting_approval")`, the graph finalizer does not claim database durability, answer deltas are buffered until validation, and compatibility adapters remain only until the atomic Task 11 cutover. These reviewed constraints supersede conflicting examples in the design document.

## Global Constraints

- Do not add regexes, token matching, translated keyword lists, explicit-name matchers, phrase rules, or default-agent branches that inspect user text.
- Canvas state, documents, planning state, previous agent, custom-agent names, tools, skills, locale, and time are router context only. None may select a route before the model call.
- Run the router exactly once for a new user turn. `Command(resume=...)` continues the interrupted execution and must not route again.
- "Exactly once" means one `route` node and one `RoutingService.route(...)` invocation; that invocation may make at most two attempts against the same resolved model.
- Use the configured runtime provider/model abstraction. The router must not import `google.genai`, instantiate a provider SDK client, or switch provider/model after failure.
- Router runtime resolution is strict: `allow_provider_fallback=False`. Static startup validation checks adapter/settings compatibility; user credentials and request overrides are validated at request time because they are user scoped.
- The router may make at most two calls to the same configured model. Default total-attempt timeout is 8 seconds and is configurable.
- `RoutingDecision.confidence` is telemetry only. It must never determine a branch.
- `routing_decision` is immutable after acceptance. Handoffs update `active_agent_id` and append `agent_history`; they never rewrite the initial decision.
- Every new turn uses `configurable.thread_id = f"routing-v2:{conversation_id}:{turn_id}"`, where `turn_id` is the persisted `user_message_id` when available. Resume uses the exact stored versioned thread ID. Turn-scoped reducer fields are never reused across new turns.
- A node that returns a dynamic `Command(goto=...)` has no static outgoing edge.
- Standard specialists use LangChain `create_agent`; RAG uses one bespoke `StateGraph` with `ToolNode`; Planning uses `Send` for independent workers.
- Every public answer passes through `validate_output` and `finalize`. Only `finalize` may append the terminal public `AIMessage` or reach `END`.
- Public answer deltas that may be rewritten or rejected are buffered. The API may stream thinking, progress, tools, artifacts, and previews before validation, but it must not publish unvalidated answer text.
- RAG grounding is mandatory for top-level RAG, RAG workers, and any public Planning synthesis that carries RAG evidence. One invalid answer may regenerate once; a second invalid answer becomes an explicit abstention.
- Worker subgraphs never append public assistant messages and never perform parent-level handoffs.
- A worker that calls `interrupt()` pauses the Planning graph; it does not fabricate an `awaiting_approval` result. On resume, only unfinished worker branches continue and completed writes are reused from the checkpoint.
- Tests use deterministic fake models and tools. Live provider evaluation is a separately marked pre-deployment job.
- This is a breaking cutover. Do not preserve `selected_agent`, custom/canvas stickiness, pre-routing, direct Gemini routing, chat fallback, auto-continuation, duplicated RAG loops, shadow grounding, or stale-response recovery.
- Preserve unrelated user changes in the worktree. Use small commits after each passing task.
- Every intermediate commit must import successfully and pass its focused tests. Legacy adapters may remain temporarily, but the final cutover removes them atomically.
- Checkpoint serializers must round-trip every checkpointed Pydantic contract without degrading it to `dict`.
- Public metrics must not use dynamic custom-agent IDs as metric labels; record bounded agent kind/base ID labels and keep full custom IDs only in access-controlled traces.

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
- `app/ai/checkpoint.py`: allowlisted round-trip serialization for all v2 workflow contracts.
- `app/ai/token_instrumentation.py`: canonical router history-budget lookup.
- `app/repositories/document.py`: bounded async document-descriptor lookup for routing context.
- `app/services/generation_registry.py`: active-agent tracking without legacy routing-state vocabulary.
- `app/services/message_service.py`: turn ID/thread namespace propagation, durable interrupt ownership, and typed terminal errors.
- `app/services/checkpoint_retention_service.py`: deletion of exact v1 and v2 checkpoint thread IDs.
- `app/observability/routing.py`: content-free routing/transition/finalizer metrics.
- `app/evaluation/routing/`: deterministic metric and release-gate code.
- `eval/routing/`: versioned multilingual live-routing dataset and thresholds.

---

### Task 1: Add typed workflow contracts and replace the state vocabulary

**Files:**
- Create: `app/ai/workflow/contracts.py`
- Create: `app/ai/workflow/state.py`
- Modify: `app/ai/schemas.py`
- Modify: `app/schemas/workflow.py`
- Modify: `app/ai/checkpoint.py`
- Create: `tests/test_workflow_contracts.py`
- Create: `tests/test_workflow_state.py`
- Modify: `tests/test_checkpoint_serializer.py`

**Interfaces:**
- Consumes: base/custom agent IDs, model routing output, specialist results, handoff requests, and worker results.
- Produces: `TurnIdentity`, `RoutingDecision`, `PendingTransition`, `AgentTransition`, `OutcomeProvenance`, `ResponseOutcome`, `HandoffOutcome`, `WorkerResult`, `WorkflowError`, `WorkflowState`, set-once routing and append-only reducers, and checkpoint round-trip support.

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


def test_routing_decision_reducer_rejects_replacement_within_turn():
    accepted = RoutingDecision(agent_id="chat_agent", confidence=0.8, reason="general help")
    replacement = RoutingDecision(agent_id="search_agent", confidence=0.9, reason="changed")
    with pytest.raises(InvalidWorkflowStateUpdate):
        set_routing_decision_once(accepted, replacement)


def test_checkpoint_serializer_round_trips_v2_contract_types():
    serializer = _build_checkpoint_serializer()
    restored = serializer.loads_typed(serializer.dumps_typed(routed_checkpoint_state()))
    assert isinstance(restored["routing_decision"], RoutingDecision)
    assert isinstance(restored["agent_history"][0], AgentTransition)
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
        "response_validation_failed", "finalization_failed", "response_persistence_failed",
        "conversation_turn_conflict",
    ]
    retriable: bool
    request_id: str
    details: dict[str, JsonValue] = Field(default_factory=dict)
```

Define the following before any state or graph code consumes them:

```python
class TurnIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str = Field(min_length=1, max_length=160)
    turn_id: str = Field(min_length=1, max_length=160)
    checkpoint_thread_id: str = Field(min_length=1, max_length=320)


class PendingTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    from_agent_id: str
    to_agent_id: str
    tool_call_id: str
    tool_message_id: str
    reason: str = Field(min_length=1, max_length=500)


class OutcomeProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output_policy_ids: tuple[str, ...] = ()
    evidence: tuple[dict[str, JsonValue], ...] = ()
    artifacts: tuple[dict[str, JsonValue], ...] = ()
    images: tuple[dict[str, JsonValue], ...] = ()
    private_messages: tuple[BaseMessage, ...] = ()
```

Define `ResponseOutcome` and `HandoffOutcome` as a discriminated union on `kind`; `ResponseOutcome` owns server-created `OutcomeProvenance` rather than trusting model-authored response metadata. Define `WorkerResult.status` as `completed | failed`; timeouts and tool/approval failures use stable `error_code` values. An actual HITL pause is represented by graph interrupt state, not a completed worker result. Define `ExecutionPhase` as `routing | executing | awaiting_approval | validating | finalizing | completed | failed`. Add `WorkflowRoutingException(RuntimeError)` with one immutable `error: WorkflowError` attribute so service and streaming boundaries can translate failures without parsing exception text.

Move shared response types needed by `contracts.py` into a dependency-safe module or make `contracts.py` the owner and re-export them from `app/ai/schemas.py`. Do not create a circular import where `contracts.py` imports `AgentResponse` from `schemas.py` while `schemas.py` imports the workflow contracts.

- [ ] **Step 4: Add reducers and the new graph state**

Use `Annotated[RoutingDecision | None, set_routing_decision_once]`, `Annotated[list[AgentTransition], append_transitions]`, and `Annotated[list[WorkerResult], append_worker_results]`. `set_routing_decision_once` accepts `None -> decision` and idempotent replay of the same frozen value, but raises `InvalidWorkflowStateUpdate` for replacement. State must contain `turn_identity`, `routing_decision`, `routing_inventory_version`, `active_agent_id`, `final_agent_id`, `agent_history`, `pending_transition`, `agent_outcome`, `worker_results`, `execution_phase`, and `workflow_error`, plus request scopes, planning data, attachments, artifacts, and messages that remain valid.

Add `request_id` and `turn_id` to both workflow request schemas. At the service boundary, use the persisted `user_message_id` for `turn_id`; use the API correlation ID when available for `request_id`, otherwise the same stable user-message ID. Build `checkpoint_thread_id` once as `routing-v2:{conversation_id}:{turn_id}`. Direct runtime tests without a database must provide explicit IDs.

Do not delete the old fields from `app/ai/schemas.py` yet; import/re-export the new contracts there only long enough for Tasks 2–11 to migrate call sites. The final cutover removes the old `GraphState` and re-exports. Because each new turn has a unique checkpoint thread, append reducers are turn-local; add a two-turn test proving histories do not accumulate across distinct `turn_id` values.

- [ ] **Step 5: Allowlist and round-trip checkpointed contracts**

Add every Pydantic type stored directly in graph state to `_CHECKPOINT_ALLOWED_TYPES`. Test both direct and nested round trips and assert types, frozen behavior, tuple fields, messages, and JSON-safe error details survive. A warning followed by restoration as `dict` is a failure.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_contracts.py tests/test_workflow_state.py tests/test_checkpoint_serializer.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/contracts.py app/ai/workflow/state.py app/ai/schemas.py app/schemas/workflow.py app/ai/checkpoint.py tests/test_workflow_contracts.py tests/test_workflow_state.py tests/test_checkpoint_serializer.py
git add app/ai/workflow/contracts.py app/ai/workflow/state.py app/ai/schemas.py app/schemas/workflow.py app/ai/checkpoint.py tests/test_workflow_contracts.py tests/test_workflow_state.py tests/test_checkpoint_serializer.py
git commit -m "refactor: add typed workflow state contracts"
```

Expected: tests pass; reducers preserve order; invalid contracts fail closed.

---

### Task 2: Build a bounded, language-neutral specialist inventory and routing context

**Files:**
- Create: `app/ai/workflow/inventory.py`
- Create: `app/ai/workflow/routing.py` (context-building portion only)
- Modify: `app/ai/history.py`
- Modify: `app/ai/token_instrumentation.py`
- Modify: `app/core/config.py`
- Modify: `app/repositories/document.py`
- Create: `tests/test_routing_inventory.py`
- Create: `tests/test_routing_context.py`
- Modify: `tests/test_history_provider.py`
- Modify: `tests/test_repository_async_twins.py`

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


def test_context_enforces_every_collection_and_text_bound(builder):
    context = await builder.build(request_with_oversized_untrusted_context())
    assert len(context.documents) <= settings.router_context_max_documents
    assert len(context.tools) <= settings.router_context_max_tools
    assert len(context.skills) <= settings.router_context_max_skills
    assert len(context.custom_agents) <= settings.router_context_max_custom_agents
    assert len(context.serialized_json) <= settings.router_context_max_chars


async def test_previous_final_agent_comes_from_owned_durable_metadata(builder):
    context = await builder.build(request_after_handoff())
    assert context.previous_final_agent_id == "search_agent"
    assert builder.history_provider.last_lookup_user_id == USER_ID


def test_inventory_version_is_stable_and_order_independent():
    assert inventory_version([descriptor_b, descriptor_a]) == inventory_version(
        [descriptor_a, descriptor_b]
    )
```

Also inspect `app/ai/workflow/routing.py` with `ast.parse`. Reject imports of `re` and `tokenize_text`, calls to `_match_explicit_custom_agent`, `.lower()` calls on message content, and return statements containing hard-coded agent IDs. Do not use raw substring tests such as rejecting `"re"`, because they produce unrelated false positives.

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

Reuse `ConversationHistoryProvider.build_context(..., agent_key="router")` so summary ownership, message ordering, and token limits remain canonical. Add `router_history_max_messages` and `router_history_max_tokens` to settings and cover them through `HistoryBudgetConfig.for_agent` tests rather than slicing ad hoc.

Add an owned durable-history lookup that returns only the previous terminal assistant workflow identity metadata. It must validate `conversation_id` and `user_id`, ignore deleted/empty/paused messages, and never depend on the previous checkpoint. Do not add `final_agent_id` to generic prompt-history message metadata.

Add `DocumentRepository.aget_routing_descriptors(conversation_id, limit)` using the repository's async session transport. Return only ID, filename, file type, status, and upload timestamp. Do not call the synchronous repository from an async routing node.

Define and enforce these default bounds in settings:

```python
router_history_max_messages: int = Field(default=12, ge=0, le=50)
router_history_max_tokens: int = Field(default=3000, ge=0, le=12000)
router_context_max_documents: int = Field(default=20, ge=0, le=100)
router_context_max_tools: int = Field(default=40, ge=0, le=200)
router_context_max_skills: int = Field(default=20, ge=0, le=100)
router_context_max_custom_agents: int = Field(default=20, ge=0, le=100)
router_context_field_max_chars: int = Field(default=500, ge=64, le=4000)
router_context_max_chars: int = Field(default=24000, ge=2000, le=64000)
```

Include original-language text unchanged within those bounds; metadata-only document and canvas descriptors; bounded plan/todo summaries; previous `final_agent_id`; tool/skill summaries resolved from the authenticated server and active device catalogs; and locale/time when explicitly present in trusted request/device metadata. Missing locale is `None`, never guessed.

Serialize the router system instruction as a `SystemMessage` and the `RoutingContext.model_dump_json()` as a separate `HumanMessage`. Mark conversation content, custom personas/descriptions, filenames, tool descriptions, and skill text as untrusted reference data. Do not interpolate those fields into the system instruction. Apply per-field truncation before total-size truncation, preserve valid JSON, and record only counts/truncation flags in telemetry.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_inventory.py tests/test_routing_context.py tests/test_history_provider.py tests/test_repository_async_twins.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/inventory.py app/ai/workflow/routing.py app/ai/history.py app/ai/token_instrumentation.py app/core/config.py app/repositories/document.py tests/test_routing_inventory.py tests/test_routing_context.py
git add app/ai/workflow/inventory.py app/ai/workflow/routing.py app/ai/history.py app/ai/token_instrumentation.py app/core/config.py app/repositories/document.py tests/test_routing_inventory.py tests/test_routing_context.py tests/test_history_provider.py tests/test_repository_async_twins.py
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
- Modify: `app/ai/agents/router.py` (temporary compatibility adapter; delete in Task 11)
- Modify: `tests/test_router.py`
- Create: `tests/test_routing_service.py`
- Create: `tests/test_routing_startup_validation.py`
- Modify: `tests/test_runtime_model_overrides.py`
- Modify: `tests/test_model_config_reasoning.py`
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


async def test_router_resolution_never_uses_provider_fallback(service, resolver):
    resolver.resolved.provider_fallback = {
        "from": "openai", "to": "gemini", "reason": "provider_not_configured"
    }
    with pytest.raises(WorkflowRoutingException) as exc:
        await service.route(context, inventory, user_id=USER_ID, model_request=None)
    assert exc.value.error.code == "routing_provider_unavailable"
    assert resolver.last_allow_provider_fallback is False


def test_static_validation_does_not_require_user_scoped_credentials(service):
    service.validate_static_configuration()
    assert service.resolver.resolve_runtime_config.call_count == 0
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_router.py tests/test_routing_service.py tests/test_routing_startup_validation.py
```

- [ ] **Step 3: Add a router-capability contract to runtime resolution**

Add the runtime-only `router` key and add both `require_capabilities: frozenset[str] = frozenset()` and `allow_provider_fallback: bool = True` to `resolve_runtime_config`. Existing agent callers preserve current behavior through the default; `RoutingService` always passes `False`. If the requested/selected provider lacks credentials or capabilities, strict resolution raises without constructing `fallback_config` or changing provider/model.

`ModelConfigService._build_capabilities` must expose `supports_structured_output` for the installed Gemini and OpenAI LangChain adapters. Unknown providers/models fail closed for router use. `ModelFactory.create_model_from_runtime(...)` verifies a non-empty API key and returns the configured LangChain chat model without fallback. Add:

```python
routing_timeout_seconds: float = Field(default=8.0, gt=0.0, le=30.0)
routing_max_attempts: int = Field(default=2, ge=1, le=2)
workflow_graph_version: str = Field(default="routing-v2")
```

Startup initialization calls `RoutingService.validate_static_configuration()` and fails only when the configured default provider/model is syntactically invalid or the installed adapter lacks structured-output support. User-scoped credentials and request model overrides cannot be known at process startup; `route(...)` validates them strictly per request and returns `routing_provider_unavailable` before a provider call. Do not probe every user's credentials or call a live model during startup.

- [ ] **Step 4: Implement the structured call and bounded retry**

```python
configured = self._resolver.resolve_runtime_config(
    user_id, "router", request_override,
    require_capabilities=frozenset({"supports_structured_output"}),
    allow_provider_fallback=False,
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

Reject any resolved config whose `provider_fallback` is non-null or whose final provider/model differs from the requested strict selection. The second attempt uses the same model object, provider, model, schema, inventory, context, and total deadline. Map timeout, provider/transport, parse/schema, and target-validation failures to the four approved error codes. Record attempt count, provider/model, latency, inventory version, and schema outcome without raw prompts or model-generated `reason` text.

`RoutingDecisionValidator` validates against the immutable request inventory, then performs one live availability check immediately before returning for dynamic custom targets. The check uses the authenticated custom-agent attachment service/repository and distinguishes unknown initial output from a target removed after inventory construction. Both fail closed; the latter increments the target-race metric.

- [ ] **Step 5: Install a temporary compatibility adapter**

Move `ROUTER_SYSTEM_PROMPT` into `routing.py`. Replace the body of `app/ai/agents/router.py` with a thin adapter over `RoutingService` only if current imports require the module before Task 4; it must contain no `google.genai`, free-text parsing, keyword/name matching, or fallback. Do not delete the module in this task because the pre-v2 graph still imports it. Task 4 migrates the graph import; Task 11 deletes the compatibility file. Update the model-usage callsite manifest.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_router.py tests/test_routing_service.py tests/test_routing_startup_validation.py tests/test_model_usage_workflow_instrumentation.py tests/test_model_config_reasoning.py tests/test_runtime_model_overrides.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/routing.py app/ai/model_factory.py app/interfaces/runtime_model_resolver_interface.py app/services/model_config_service.py app/core/config.py app/core/container.py app/observability/routing.py tests/test_router.py tests/test_routing_service.py tests/test_routing_startup_validation.py
git add app/ai/workflow/routing.py app/ai/model_factory.py app/ai/agents/router.py app/interfaces/runtime_model_resolver_interface.py app/services/model_config_service.py app/core/config.py app/core/container.py app/observability/routing.py tests/test_router.py tests/test_routing_service.py tests/test_routing_startup_validation.py tests/test_model_usage_workflow_instrumentation.py tests/test_model_config_reasoning.py tests/test_runtime_model_overrides.py tests/fixtures/model_usage_callsite_manifest.json
git commit -m "refactor: replace router with structured runtime model routing"
```

Expected: all tests pass and repository search finds no Gemini client or semantic text parser in routing code.

---

### Task 4: Build the new parent graph shell and route once per new turn

**Files:**
- Rewrite: `app/ai/workflow/graph_builder.py`
- Create: `app/ai/workflow/specialists.py` (compatibility wrappers replaced in Task 5)
- Create: `app/ai/workflow/finalization.py` (minimal terminal boundary expanded in Task 9)
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


async def test_two_turns_in_one_conversation_use_distinct_checkpoint_threads(workflow):
    first = request("first", conversation_id="conversation-1", turn_id="message-1")
    second = request("second", conversation_id="conversation-1", turn_id="message-2")
    await workflow.execute_request(first)
    await workflow.execute_request(second)
    assert workflow.invoked_thread_ids == [
        "routing-v2:conversation-1:message-1",
        "routing-v2:conversation-1:message-2",
    ]


def test_compatibility_specialists_still_end_through_finalizer(compiled_graph):
    graph = compiled_graph.get_graph()
    assert not any(
        edge.target == "__end__" and edge.source != "finalize" for edge in graph.edges
    )
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

On `WorkflowRoutingException`, return `Command(update={"workflow_error": ..., "execution_phase": "failed"}, goto="finalize")`. Create the minimal exception-safe `finalize` node in this task: it records failure metadata, publishes no assistant message for failed state, and returns a terminal state update. For a successful compatibility outcome, it appends the reserved terminal message and creates the service response. Task 9 adds the full provenance policy registry without changing this graph contract.

- [ ] **Step 4: Wire the parent topology**

Build `StateGraph(WorkflowState, context_schema=WorkflowRuntimeContext)` and `START -> route`. Register stable compatibility wrapper nodes for base agents plus one `custom_agent` wrapper, `validate_output`, and `finalize`. Compatibility wrappers invoke the current specialist methods but copy their terminal content/artifacts into `ResponseOutcome`; they must prevent the old `_finalize_agent_response` path from appending a parent `AIMessage`. They return `Command(goto="validate_output")` and receive no static outgoing edge. Task 5 replaces their execution internals, and Task 6 registers `resolve_transition`. Add only `finalize -> END`; both successful and failed executions terminate there.

The Task 4 validator is deliberately minimal but real: it verifies non-empty public content/error shape and server-owned outcome construction. It is not a placeholder or bypass; Task 9 adds evidence, artifact, image, canvas, and tool-pairing policies behind the same interface.

Use the precomputed `state.turn_identity.checkpoint_thread_id`, exactly `routing-v2:{conversation_id}:{turn_id}`. The public interrupt payload and durable HITL row store this exact ID. Resume accepts only that stored ID and must never reconstruct it from conversation ID. A new turn has a different `turn_id`, so it never loads an earlier v2 turn or any v1 checkpoint.

- [ ] **Step 5: Remove streaming pre-routing from the runtime shell**

Delete the call around current `app/ai/graph.py:2938` that invokes `_route_node` before `astream`. Both sync and streaming paths construct state with `routing_decision=None` and let graph execution enter `route`. Replace the `Router` import/field with injected `RoutingService`. Keep old specialist method bodies temporarily behind compatibility wrappers, but remove their authority to append terminal parent messages or reach `END`.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_production_workflow_graph.py tests/test_graph_refactor_contract.py tests/test_graph_route_node_async_documents.py tests/test_ai_service_initialization.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/graph_builder.py app/ai/workflow/specialists.py app/ai/workflow/finalization.py app/ai/graph.py app/core/container.py tests/test_production_workflow_graph.py tests/test_graph_refactor_contract.py
git add app/ai/workflow/graph_builder.py app/ai/workflow/specialists.py app/ai/workflow/finalization.py app/ai/graph.py app/core/container.py tests/test_production_workflow_graph.py tests/test_graph_refactor_contract.py tests/test_graph_route_node_async_documents.py tests/test_ai_service_initialization.py
git commit -m "refactor: introduce single-entry production workflow graph"
```

---

### Task 5: Replace bespoke standard-agent loops with per-invocation `create_agent` subgraphs

**Files:**
- Create: `app/ai/workflow/middleware.py`
- Modify: `app/ai/workflow/specialists.py`
- Modify: `app/ai/request_budget.py`
- Modify: `app/ai/token_instrumentation.py`
- Modify: `app/ai/tool_scope.py`
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
- Modify: `tests/test_runtime_model_overrides.py`
- Modify: `tests/test_context_window_message_metadata.py`
- Modify: `tests/test_conversation_memory_hydration.py`
- Modify: `tests/test_request_budget.py`
- Modify: `tests/test_runtime_time_context.py`
- Modify: `tests/test_widget_runtime.py`
- Modify: `tests/test_web_research_binding.py`
- Modify: `tests/test_read_tool_result_binding.py`
- Modify: `tests/test_canvas_agent.py`
- Modify: `tests/test_image_generator_harvest.py`
- Modify: `tests/test_provider_selected_image_injection.py`
- Modify: `tests/test_model_usage_workflow_instrumentation.py`
- Modify: `tests/test_client_tool_scope.py`
- Modify: `tests/test_client_tool_isolation.py`
- Modify: `tests/test_custom_agents_graph.py`
- Modify: `tests/test_hitl_gate_policy.py`
- Modify: `tests/test_graph_tool_budget.py`

**Interfaces:**
- Consumes: `SpecialistDefinition`, `WorkflowRuntimeContext`, authenticated request scope, runtime model config, tools, history, and parent state.
- Produces: one per-invocation compiled `create_agent` subgraph and an `AgentOutcome`; no standard specialist mutates parent routing state directly.

- [ ] **Step 1: Write failing runtime and isolation tests**

```python
async def test_standard_specialist_is_created_per_invocation(factory):
    observed_contexts = []
    first_graph_id = await factory.invoke(
        "chat_agent", context_for(device_id="device-a"), observe=observed_contexts.append
    )
    second_graph_id = await factory.invoke(
        "chat_agent", context_for(device_id="device-b"), observe=observed_contexts.append
    )
    assert first_graph_id != second_graph_id
    assert [item.device_id for item in observed_contexts] == ["device-a", "device-b"]


async def test_specialist_response_returns_parent_validation_command(wrapper):
    command = await wrapper(state_with_active_agent("chat_agent"), runtime)
    assert command.goto == "validate_output"
    assert command.update["agent_outcome"].kind == "response"


async def test_worker_mode_never_appends_public_message(factory):
    result = await factory.invoke_worker("search_agent", worker_request())
    assert isinstance(result, WorkerResult)
    assert "public_messages" not in result.model_fields
```

Assert only documented invocation behavior; do not inspect private or undocumented attributes on the compiled agent. Add middleware tests proving model/tool call limits, user/device/conversation scope, tool authorization before execution, usage recording, artifact offloading, and HITL interrupt/resume. For HITL, cover approve, edit, reject, and respond using a checkpointer-backed initial invocation plus `Command(resume=...)`; assert the side effect runs at most once and that resume does not duplicate the tool message.

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

Preserve every runtime responsibility currently implemented by the legacy agents, with one explicit owner:

- runtime model overrides and custom-agent model selection: `middleware.py`;
- bounded hydrated history and message metadata: `specialists.py`;
- request token/tool budget accounting: `request_budget.py` and `token_instrumentation.py`;
- authenticated client/tool bindings, widget tools, read-tool-result, and web research: `tool_scope.py` plus `middleware.py`;
- model usage emission, including failed provider attempts: `middleware.py`;
- request-time clock/timezone context: the specialist prompt factory;
- canvas snapshots, provider-selected image injection, secondary image delivery, and artifact harvesting: their domain specialist factories.

Port or replace the listed regression tests in this task; do not postpone these behaviors to the legacy-deletion task.

- [ ] **Step 4: Implement focused middleware and framework limits**

Use `ModelCallLimitMiddleware(..., exit_behavior="error")` and `ToolCallLimitMiddleware(..., exit_behavior="error")` for call counts. Add small application middleware for:

- resolving a model using `IRuntimeModelResolver` and `ModelFactory`;
- resolving authorized dynamic tools from current user/device scope;
- checking tool policy before implementation invocation;
- translating current HITL policy into `interrupt()`/resume behavior;
- recording usage exactly once per provider attempt;
- collecting tool artifacts, images, canvas metadata, and offloaded results into private subgraph state.

Authorization occurs before HITL policy evaluation, and HITL approval occurs before the implementation is invoked. Do not duplicate a `while tool_calls` loop or implement auto-continuation. Catch only `ModelCallLimitExceededError` and `ToolCallLimitExceededError` at the specialist boundary and translate them to a parent `Command` targeting `finalize` with `WorkflowError(code="agent_execution_limit", retriable=False, ...)`; unrelated exceptions retain their original typed failure mapping.

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
.\.venv\Scripts\python.exe -m pytest -q tests/test_specialist_runtime.py tests/test_specialist_middleware.py tests/test_custom_agents_graph.py tests/test_hitl_gate_policy.py tests/test_graph_tool_budget.py tests/test_client_invocation_isolation.py tests/test_client_tool_isolation.py tests/test_client_tool_scope.py tests/test_runtime_model_overrides.py tests/test_context_window_message_metadata.py tests/test_conversation_memory_hydration.py tests/test_request_budget.py tests/test_runtime_time_context.py tests/test_widget_runtime.py tests/test_web_research_binding.py tests/test_read_tool_result_binding.py tests/test_canvas_agent.py tests/test_image_generator_harvest.py tests/test_provider_selected_image_injection.py tests/test_model_usage_workflow_instrumentation.py tests/test_model_usage_workflow_wiring.py
.\.venv\Scripts\python.exe -m ruff check app/ai/workflow/middleware.py app/ai/workflow/specialists.py app/ai/request_budget.py app/ai/token_instrumentation.py app/ai/tool_scope.py app/ai/agents/chat_agent.py app/ai/agents/search_agent.py app/ai/agents/canvas_agent.py app/ai/agents/image_generator_agent.py app/ai/agents/custom_agent.py
git add app/ai/workflow/middleware.py app/ai/workflow/specialists.py app/ai/request_budget.py app/ai/token_instrumentation.py app/ai/tool_scope.py app/ai/agents/chat_agent.py app/ai/agents/search_agent.py app/ai/agents/canvas_agent.py app/ai/agents/image_generator_agent.py app/ai/agents/custom_agent.py app/ai/custom_agent_runtime.py app/ai/deferred_tool_binding.py app/ai/tool_execution.py app/ai/hitl_config.py tests/test_specialist_runtime.py tests/test_specialist_middleware.py tests/test_custom_agents_graph.py tests/test_hitl_gate_policy.py tests/test_graph_tool_budget.py tests/test_runtime_model_overrides.py tests/test_context_window_message_metadata.py tests/test_conversation_memory_hydration.py tests/test_request_budget.py tests/test_runtime_time_context.py tests/test_widget_runtime.py tests/test_web_research_binding.py tests/test_read_tool_result_binding.py tests/test_canvas_agent.py tests/test_image_generator_harvest.py tests/test_provider_selected_image_injection.py tests/test_model_usage_workflow_instrumentation.py tests/test_client_tool_scope.py tests/test_client_tool_isolation.py
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
    assert command.update["messages"][0].id == "handoff:call-1"


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
    assert command.update["messages"][0].id == "handoff:call-1"
    assert len(tool_messages_for(command, "call-1")) == 1


async def test_custom_target_routes_to_resolved_node_name(resolver):
    command = await resolver(valid_custom_transition("custom_agent:123"))
    assert command.goto == resolver.inventory.resolve_node("custom_agent:123")
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
            tool_message_id=f"handoff:{tool_call_id}",
            reason=reason,
        ),
        "messages": [ToolMessage(
            id=f"handoff:{tool_call_id}",
            content="handoff_requested",
            name="hand_off",
            tool_call_id=tool_call_id,
        )],
    },
    goto="resolve_transition",
)
```

The tool must not JSON-encode a control message and must not mutate state in place. The deterministic message ID is part of the transition contract and survives checkpoint serialization.

- [ ] **Step 4: Implement the sole transition resolver**

Validate source/active identity, target existence/attachment/reachability, tool-call pairing, visited agents, and `max_handoff_delegation_depth`. An accepted transition retains exactly one paired tool message, appends one `AgentTransition(source="handoff")`, and routes to `inventory.resolve_node(target_agent_id)` rather than assuming the agent ID is a graph node name. A rejected transition returns a `ToolMessage` with the same deterministic ID, causing the message reducer to replace the request marker instead of appending a duplicate, and routes back to the resolved source wrapper. Only accepted handoffs consume transition depth; resume bookkeeping does not. Only control-plane IDs are compared; no message text is interpreted.

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


async def test_no_evidence_still_runs_grounding_policy_and_cannot_claim_sources(rag_graph):
    result = await rag_graph.ainvoke(rag_request_with_no_retrieval_hits())
    assert result["rag_result"].grounding.outcome in {"abstained", "clarification"}
    assert result["rag_result"].grounding.validated is True
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

Use `ToolNode` for execution and state-aware `Command` updates for evidence/artifact collection; middleware/wrappers own authorization, approval, artifact offloading, and error normalization. The graph receives runtime context; per-invocation subgraphs inherit the parent checkpointer. Allocate evidence IDs from one per-run server-owned allocator. Reject duplicate or ambiguous IDs instead of accepting the first match.

- [ ] **Step 4: Move RAG behavior into the shared graph**

Move model invocation and constrained regeneration out of the `RAGAgent` loop into focused functions consumed by `rag_execution.py`. Reuse `execute_search_documents_action`, evidence packs, token budgets, image provenance, and `GroundedAnswerGate`. Preserve server-owned evidence IDs across every model round and merge evidence only through the typed reducer.

Remove the `rag_grounded_answer_gate_enabled` branch: construction always installs the gate and validation runs for every RAG result, including zero-evidence retrieval. A zero-evidence answer may ask a bounded clarification or abstain, but it cannot make source-backed claims. Metrics record `accepted | regenerated | clarification | abstained`; there is no shadow-only outcome.

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
- Modify: `app/core/config.py`
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


async def test_planning_preserves_plan_lifecycle_and_todos(planning_graph):
    result = await planning_graph.ainvoke(existing_plan_request(action="modify"))
    assert result["planning_result"].plan_revision == 4
    assert result["planning_result"].todo_changes


async def test_worker_hitl_interrupt_resumes_exact_task_once(planning_graph, checkpointer):
    interrupted = await invoke_until_interrupt(planning_graph, approval_plan(), checkpointer)
    resumed = await planning_graph.ainvoke(Command(resume={"decision": "approve"}), interrupted.config)
    assert resumed["worker_results"][0].task_id == "t1"
    assert tool_side_effect_count("t1") == 1
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_planning_execution_graph.py tests/test_graph_planning_subagents.py tests/test_planning_subagents.py tests/test_custom_agents_planning.py
```

- [ ] **Step 3: Define task and result contracts**

Use `WorkerTask(task_id, objective, agent_id, allowed_tool_ids, model_request, related_todo_ids)` and the approved `WorkerResult`. Worker state contains only the task, bounded parent context, authenticated scope, and private messages. Delimit worker objectives and results as untrusted data rather than interpolating them into system instructions. The reducer rejects duplicate task IDs rather than silently overwriting them.

Preserve the current Planning contract: plan create/modify/review actions, plan revision and lifecycle metadata, rubric output, todo creation/update/completion, existing-plan context, custom-agent/model overrides, and planning/subagent stream events. Add settings with conservative defaults: `planning_worker_max_tasks=8`, `planning_worker_max_concurrency=4`, `planning_worker_objective_max_chars=4000`, and `planning_parent_context_max_chars=12000`; validate positive values at startup.

- [ ] **Step 4: Implement `Send` fan-out and worker execution**

```python
def dispatch_workers(state: PlanningState) -> list[Send]:
    bounded = state["worker_tasks"][: state["limits"].max_tasks]
    return [Send("worker", {"task": task, "runtime_request": state["runtime_request"]})
            for task in bounded]
```

Use topology `START -> planning_model -> dispatch_workers -> worker -> collect_results -> planning_model_or_synthesize -> package_result -> END`. The planning model may revise the typed plan once after worker results but cannot dispatch an unbounded second wave. Pass `max_concurrency=planning_worker_max_concurrency` in the child run configuration and preserve deterministic result ordering by original task position.

The `worker` node resolves the requested specialist from the same live inventory. Standard workers call `SpecialistFactory.invoke_worker`; RAG workers call `RagExecutionGraphFactory`; recursive Planning is rejected as `WorkerResult(status="failed", error_code="recursive_planning")`. A worker needing approval interrupts the graph and resumes the same checkpointed task; it does not manufacture an `awaiting_approval` result. Timeouts and execution limits become typed failed results with `worker_timeout` or `agent_execution_limit`. Workers cannot issue a parent-level handoff.

- [ ] **Step 5: Implement Planning synthesis**

The Planning model receives ordered typed results, artifacts, and evidence as delimited untrusted payloads. It returns one `ResponseOutcome` plus validated plan/rubric/todo metadata. If any result contains evidence, propagate the complete server-owned evidence set and add the `rag_grounding` output policy so Task 9 revalidates the public synthesis. Emit task-correlated `subagent_start`, `subagent_delta`, `subagent_complete`, and `subagent_error` custom events through the graph stream writer; do not store streaming callbacks in checkpointed state.

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
- Modify: `app/ai/workflow/finalization.py`
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


def test_outcome_provenance_is_server_owned_and_complete(finalizer):
    response = finalizer.finalize(validated_state_with_artifacts())
    assert response.metadata["validation"]["policy_versions"]
    assert response.metadata["provenance"]["artifact_ids"] == ["artifact-1"]


async def test_no_public_delta_is_released_before_validation(stream):
    events = [event async for event in stream(outcome_rewritten_by_policy())]
    assert join_public_deltas(events) == finalized_content(events)
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

Choose policies from server-owned `outcome.provenance` and declared `output_policy_ids`, not merely the final agent ID. Evidence presence always activates `rag_grounding`; a RAG outcome also declares `rag_grounding` when its evidence set is empty. Artifacts/images always activate their provenance validators. Store policy ID and implementation version in final metadata so results are auditable across deployments.

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
9. expose `validated_public_content` for the stream projector;
10. return a state update with `execution_phase="completed"`; the graph's sole static terminal edge is `finalize -> END`.

Build the response and state update in local immutable values before returning them; do not partially mutate state if metadata normalization fails. For failed state, the same finalizer records failure telemetry, verifies that no public assistant message was appended, preserves the `WorkflowError`, and returns `execution_phase="failed"` for the same static terminal edge.

The graph finalizer guarantees graph-level response construction and validation, not database durability. `MessageService` remains the sole owner of the transactional conversation-history write in Task 10; a write failure returns `WorkflowError(code="response_persistence_failed", ...)` and must not be reported as `finalization_failed`.

Do not scan prior messages to recover a plausible response. Specialist answer tokens remain private/buffered until this node succeeds; only `validated_public_content` is eligible for public projection.

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
- Modify: `app/services/message_service.py`
- Modify: `app/services/generation_registry.py`
- Modify: `app/schemas/workflow.py`
- Modify: `app/utils/exception_handler.py`
- Modify: `app/api/messages.py`
- Modify: `app/api/ai_sdk.py`
- Create: `tests/test_workflow_error_contract.py`
- Modify: `tests/test_graph_stream_projection.py`
- Modify: `tests/test_event_streaming_langgraph_normalizer.py`
- Modify: `tests/test_event_streaming_subagents.py`
- Modify: `tests/test_message_stream_errors.py`
- Modify: `tests/test_graph_resume_image_stream.py`
- Modify: `tests/test_message_service_event_streaming.py`
- Modify: `tests/test_ai_sdk_context_window.py`

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


async def test_specialist_tokens_are_buffered_until_finalizer_accepts(stream):
    events = [event async for event in stream]
    assert stream.observed_order.index("response_persisted") < stream.observed_order.index(
        "first_public_message_delta"
    )
    assert join_public_deltas(events) == stream.finalized_content
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_error_contract.py tests/test_graph_stream_projection.py tests/test_event_streaming_langgraph_normalizer.py tests/test_message_stream_errors.py tests/test_graph_resume_image_stream.py
```

- [ ] **Step 3: Derive events from new state updates**

Emit initial `agent_selected` when `routing_decision` first appears, with `data={"cause": "route", "confidence": ..., "inventory_version": ...}`. Emit subsequent selections only for newly appended accepted `AgentTransition` values, with `data={"cause": "handoff", "source_agent": ..., "tool_call_id": ...}`.

Stop comparing `selected_agent` snapshots. Track the routing-decision identity and transition count in `StreamProjectionContext`. A resume may append `AgentTransition(source="resume")` for audit history, but the projector ignores it for `agent_selected` and it does not consume handoff depth. Filter nested graph namespaces explicitly: Planning worker deltas remain task-correlated `subagent_*` events and never become `message_delta` in the main answer stream; specialist/internal model messages are private until finalization.

- [ ] **Step 4: Replace string errors with the stable payload**

Change both AI-layer and service-layer response schemas from `error: str | None` to `error: WorkflowError | None`. Source `request_id` from `TurnIdentity`, never from an optional provider response. The API adapter may add localized display text, but `code`, `retriable`, `request_id`, and allowlisted structured details remain intact; redact provider payloads, credentials, document content, and stack traces. Remove logic that prefixes strings with `"Error:"` or maps a missing response to an English generic answer.

- [ ] **Step 5: Make completion finalizer-driven**

Do not forward answer tokens directly from router, specialist, RAG, Planning, worker, or validation namespaces. After the graph reaches `execution_phase="completed"`, persist the finalizer-owned response transactionally through `MessageService`; only after that commit succeeds chunk `validated_public_content` into the existing public `message_delta` events and emit one `complete`. On graph failure or `response_persistence_failed`, emit one `error` and no public answer delta or `complete`. Delete terminal content recovery from accumulated stream chunks and checkpoint message scanning.

On resume, load the durable HITL record and pass its exact `checkpoint_thread_id` to `Command(resume=...)`. Never reconstruct the ID from conversation ID. Update `GenerationRegistry`, `MessageService`, and both API adapters to use `initial_agent_id`, `active_agent_id`, and `final_agent_id`; remove `selected_agent` as a registry or response field.

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
- Delete: `app/ai/agents/router.py`
- Modify: `app/ai/agent_config.py`
- Modify: `app/ai/prompts.py`
- Modify: `app/core/config.py`
- Modify: `app/ai/langgraph.json`
- Delete: `tests/test_custom_agent_stickiness.py`
- Replace: `tests/test_graph_no_fast_path_helpers.py`
- Create: `tests/test_routing_legacy_removal.py`
- Modify: `tests/test_production_readiness_contract.py`
- Modify: `tests/test_rag_dead_code_cleanup.py`
- Modify: `tests/test_ai_sdk_context_window.py`
- Modify: `tests/test_canvas_agent.py`
- Modify: `tests/test_client_tool_isolation.py`
- Modify: `tests/test_client_tool_scope.py`
- Modify: `tests/test_context_window_message_metadata.py`
- Modify: `tests/test_conversation_memory_hydration.py`
- Modify: `tests/test_graph_tool_budget.py`
- Modify: `tests/test_model_usage_workflow_instrumentation.py`
- Modify: `tests/test_model_usage_workflow_wiring.py`
- Modify: `tests/test_multi_sidecar_hardening.py`
- Modify: `tests/test_planning_subagents.py`
- Modify: `tests/test_rag_evidence.py`
- Modify: `tests/test_rag_reranker.py`
- Modify: `tests/test_read_tool_result_binding.py`
- Modify: `tests/test_request_budget.py`
- Modify: `tests/test_runtime_model_overrides.py`
- Modify: `tests/test_runtime_time_context.py`
- Modify: `tests/test_web_research_binding.py`
- Modify: `tests/test_widget_runtime.py`
- Modify: `tests/test_custom_agents_api.py`
- Modify: `tests/test_custom_agents_message_service.py`
- Modify: `tests/test_custom_agents_service.py`
- Modify: `tests/test_message_history_pipeline.py`

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

The scan excludes migrations and historical docs/specs/plans; it includes all runtime packages, service/API adapters, and live tests. Tests may construct v1 checkpoint fixtures through neutral helper keys, but must not import or execute a legacy runtime class.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_legacy_removal.py tests/test_production_readiness_contract.py tests/test_rag_dead_code_cleanup.py
```

- [ ] **Step 3: Shrink `MultiAgentWorkflow` to the runtime adapter**

Keep only dependency initialization, request-to-state construction, `graph.ainvoke`, graph event streaming, `Command(resume=...)`, checkpoint compaction, and history-cache invalidation. Move node logic to the focused workflow modules. Remove mixin inheritance and manual agent registries that contain executing agent objects.

- [ ] **Step 4: Remove old state and semantic overrides**

Delete the old `GraphState`, `GraphStateView.selected_agent`, `last_agent`, `delegation_count`, custom stickiness, canvas preselection, planning force, explicit-name matching, no-document chat substitution, preselected-agent exit, and auto-continuation. Re-export `WorkflowState` as the sole graph state type.

- [ ] **Step 5: Delete duplicated loops and fallback settings**

Delete the four workflow mixin modules, obsolete `BaseAgent` loop, and the temporary `Router` compatibility adapter only after all imports have migrated. Delete the old router model config path if it is superseded by runtime model configuration. Remove the shadow grounding setting and continuation settings that no live component reads. Remove graph topology entries for `tools`, `approval`, `rag_tools`, and `planning_tools`; those now live inside specialist subgraphs. Remove obsolete checkpoint serializer allowlist entries for deleted legacy Pydantic types.

- [ ] **Step 6: Remove superseded tests, update living contracts, and search**

Delete `tests/test_custom_agent_stickiness.py`; replace expectations in graph, RAG, planning, handoff, stream, runtime-model, history, tool-scope, request-budget, canvas/image, usage, and API/service tests with v2 contracts. Generate the edit inventory before deletion so no legacy import is missed:

```powershell
rg -l "selected_agent|last_agent|delegation_count|BaseAgent|agents\.router" app tests
rg -n "selected_agent|last_agent|delegation_count|_sticky_custom_agent_for_followup|_match_explicit_custom_agent|_extract_agent_name|_apply_hand_off_if_present|_run_agent_in_isolated_context|continuation_round|rag_grounded_answer_gate_enabled|_recover_terminal_response" app tests
```

Expected: no runtime matches. A test filename or historical test description is not an acceptable retained compatibility path; update it.

- [ ] **Step 7: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_legacy_removal.py tests/test_production_readiness_contract.py tests/test_rag_dead_code_cleanup.py tests/test_production_workflow_graph.py tests/test_custom_agents_graph.py tests/test_graph_planning_subagents.py tests/test_rag_execution_graph.py
.\.venv\Scripts\python.exe -m ruff check app/ai app/services/event_streaming app/services/ai_service.py tests/test_routing_legacy_removal.py tests/test_production_readiness_contract.py tests/test_rag_dead_code_cleanup.py
git rm app/ai/workflow/tool_loop.py app/ai/workflow/rag_loop.py app/ai/workflow/planning_loop.py app/ai/workflow/custom_agents.py app/ai/agents/base_agent.py app/ai/agents/router.py tests/test_custom_agent_stickiness.py
git add app/ai/graph.py app/ai/schemas.py app/ai/workflow/__init__.py app/ai/agents/__init__.py app/ai/agent_config.py app/ai/prompts.py app/core/config.py app/ai/langgraph.json tests/test_graph_no_fast_path_helpers.py tests/test_routing_legacy_removal.py tests/test_production_readiness_contract.py tests/test_rag_dead_code_cleanup.py tests/test_ai_sdk_context_window.py tests/test_canvas_agent.py tests/test_client_tool_isolation.py tests/test_client_tool_scope.py tests/test_context_window_message_metadata.py tests/test_conversation_memory_hydration.py tests/test_graph_tool_budget.py tests/test_model_usage_workflow_instrumentation.py tests/test_model_usage_workflow_wiring.py tests/test_multi_sidecar_hardening.py tests/test_planning_subagents.py tests/test_rag_evidence.py tests/test_rag_reranker.py tests/test_read_tool_result_binding.py tests/test_request_budget.py tests/test_runtime_model_overrides.py tests/test_runtime_time_context.py tests/test_web_research_binding.py tests/test_widget_runtime.py tests/test_custom_agents_api.py tests/test_custom_agents_message_service.py tests/test_custom_agents_service.py tests/test_message_history_pipeline.py
git commit -m "refactor: remove legacy routing and duplicated agent loops"
```

Expected: the runtime contains only v2 routing/execution; old checkpoint data remains stored but cannot be loaded because the namespace changed.

---

### Task 12: Add routing observability and a versioned multilingual evaluation gate

> **SKIPPED — marked redundant by Thai on 2026-08-26.** The live multilingual
> evaluation program (golden dataset, human-review manifest, release-gate CLI)
> is not being built. Content-free runtime routing metrics already landed in
> Task 3 as `app/observability/routing.py`, which covers the observability half
> of this task. Consequences carried forward: the router's real-world routing
> accuracy is **unmeasured**, and the Task 14 acceptance gate loses its
> `macro_f1`, per-language accuracy, and structured-output-success thresholds.
> Do not describe routing quality as validated.

**Files:**
- Modify: `app/observability/routing.py`
- Create: `app/evaluation/routing/__init__.py`
- Create: `app/evaluation/routing/contracts.py`
- Create: `app/evaluation/routing/metrics.py`
- Create: `app/evaluation/routing/release_gates.py`
- Create: `app/evaluation/routing/target.py`
- Create: `eval/routing/golden_v1.jsonl`
- Create: `eval/routing/golden_v1.review.json`
- Create: `eval/routing/release_gates.json`
- Create: `scripts/evaluate_routing.py`
- Create: `scripts/check_routing_release.py`
- Create: `docs/operations/routing-v2-evaluation-gate.md`
- Create: `tests/test_routing_evaluation_contracts.py`
- Create: `tests/test_routing_evaluation_metrics.py`
- Create: `tests/test_routing_observability.py`
- Create: `tests/test_routing_release_gate.py`
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
    assert all(row.reference.primary_agent_id in row.reference.acceptable_agent_ids for row in dataset)


def test_release_gate_checks_language_gap():
    report = report_for(language_accuracy={"en": 0.94, "th": 0.88})
    assert evaluate_release_gates(report).passed is False
    assert "language_gap" in evaluate_release_gates(report).failed_gates


def test_routing_metrics_do_not_record_message_content(recorder):
    recorder.routing_completed(event_with_message("secret prompt"))
    assert "secret prompt" not in json.dumps(recorder.export())


def test_unreviewed_or_missing_live_report_cannot_pass_release_gate(tmp_path):
    assert check_release(dataset_review="pending", live_report=None).passed is False
```

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_evaluation_contracts.py tests/test_routing_evaluation_metrics.py tests/test_routing_observability.py tests/test_routing_release_gate.py
```

- [ ] **Step 3: Implement deterministic evaluation contracts and metrics**

Each JSONL row contains `id`, `inputs` (message plus bounded context fixture), `reference.primary_agent_id`, `reference.acceptable_agent_ids`, and metadata (`language`, `category`, `difficulty`). Seed at least 210 human-reviewed cases: at least 30 each for English, Thai, Vietnamese, Chinese, Japanese, Arabic, and mixed-language input, with every intent category represented by at least 10 cases. Cover general chat, current information/search, document/RAG, canvas continuation and departure, planning with/without an existing plan, explicit and implicit custom-agent requests, image generation, ambiguous follow-ups, and mixed intents. Labels may list multiple acceptable routes, but the canonical primary label is required for per-class metrics and confusion matrices.

`golden_v1.review.json` records dataset SHA-256, review status, reviewing team alias, reviewed timestamp, and label-guideline version. The loader rejects a missing hash, non-approved review status, or a hash mismatch. Human review and approval are a release prerequisite, not something the seed-generation script can self-assert.

Implement acceptable-set accuracy as `predicted_agent_id in acceptable_agent_ids`. Compute per-agent precision/recall/F1, macro-F1, and the confusion matrix against `primary_agent_id` so multi-label rows are not double-counted. Also compute per-language acceptable-set accuracy, English gap, first-attempt schema rate, after-retry schema rate, latency percentiles, and failure-code counts. Unit tests use deterministic fixtures and set `LANGSMITH_TRACING=false`.

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

Register `routing_live: calls the configured live router model` in pytest markers. `scripts/evaluate_routing.py` accepts `--dataset`, `--output`, `--provider`, `--model`, and `--compare-baseline`; it records provider/model identifiers, inventory version, dataset hash, timestamp, and attempt counts. It calls `RoutingService` through `app/evaluation/routing/target.py`, never a second evaluation-only router.

`scripts/check_routing_release.py` verifies the reviewed dataset hash, a fresh successful live report for the exact provider/model/inventory tuple being deployed, and every threshold. Document it as a mandatory pre-deployment job with the model credential supplied by the deployment environment. A missing, stale, mismatched, or `not_run` live report fails closed.

- [ ] **Step 6: Add content-free runtime metrics**

Record route volume, latency, attempt/schema result, target races, accepted/rejected handoffs, correction rate, transition depth, execution limits, worker status, grounding outcomes, finalizer policies, and terminal error codes. Metric labels are allowlisted enums plus bounded provider/model/inventory identifiers; request, conversation, user, agent-instance, evidence, and message IDs belong only in sampled traces/logs, never metric labels. Exclude prompt text, raw document data, credentials, and unbounded custom-agent descriptions.

- [ ] **Step 7: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_routing_evaluation_contracts.py tests/test_routing_evaluation_metrics.py tests/test_routing_observability.py tests/test_routing_release_gate.py
.\.venv\Scripts\python.exe -m ruff check app/evaluation/routing app/observability/routing.py scripts/evaluate_routing.py scripts/check_routing_release.py tests/test_routing_evaluation_contracts.py tests/test_routing_evaluation_metrics.py tests/test_routing_observability.py tests/test_routing_release_gate.py
.\.venv\Scripts\python.exe scripts/evaluate_routing.py --dataset eval/routing/golden_v1.jsonl --output .artifacts/routing-eval-v1.json
.\.venv\Scripts\python.exe scripts/check_routing_release.py --dataset eval/routing/golden_v1.jsonl --review eval/routing/golden_v1.review.json --report .artifacts/routing-eval-v1.json
git add app/evaluation/routing app/observability/routing.py eval/routing scripts/evaluate_routing.py scripts/check_routing_release.py docs/operations/routing-v2-evaluation-gate.md tests/test_routing_evaluation_contracts.py tests/test_routing_evaluation_metrics.py tests/test_routing_observability.py tests/test_routing_release_gate.py pyproject.toml
git commit -m "test: add multilingual routing release gates"
```

Expected: deterministic tests pass. The live command must pass every threshold before production deployment; if credentials are intentionally unavailable, record the job as not run rather than treating it as a pass.

---

### Task 13: Add concurrency, checkpoint-version, and end-to-end production tests

**Files:**
- Create: `app/services/conversation_turn_coordinator.py`
- Modify: `app/services/checkpoint_retention_service.py`
- Modify: `app/workers/cleanup_tasks.py`
- Create: `tests/test_workflow_concurrency.py`
- Create: `tests/test_workflow_checkpoint_v2.py`
- Modify: `tests/test_checkpoint_retention_service.py`
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
    request = request_with_turn("new turn", conversation_id="conversation-1", turn_id="message-2")
    await workflow.execute_request(request)
    assert workflow.routing_service.route.await_count == 1
    assert await checkpointer.aget_tuple(
        config_for("routing-v2:conversation-1:message-2")
    ) is not None


async def test_same_conversation_turns_never_overlap_context_through_persistence(workflow):
    first, second = await asyncio.gather(
        workflow.execute_request(request_with_turn("first", "c1", "m1")),
        workflow.execute_request(request_with_turn("second", "c1", "m2")),
    )
    revisions = sorted([first.metadata["context_revision"], second.metadata["context_revision"]])
    assert revisions[1] == revisions[0] + 1
    assert workflow.turn_coordinator.max_active_for("c1") == 1
```

Different conversations execute concurrently. Turns in the same conversation acquire a cross-process database advisory lock (or an equivalently durable coordinator owned by `conversation_turn_coordinator.py`) before history/context snapshotting and hold it through response persistence. Bound lock acquisition and return a typed retriable conflict/timeout error; never fall back to an in-process-only lock in production.

- [ ] **Step 2: Confirm RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_checkpoint_retention_service.py tests/test_workflow_end_to_end.py
```

- [ ] **Step 3: Cover the production scenarios**

Add deterministic end-to-end cases for every base specialist and a dynamic custom specialist; one and two handoffs; invalid/cyclic handoffs; RAG regeneration/abstention; Planning parallel workers including RAG; HITL approve/edit/reject and durable resume after constructing a fresh workflow instance against the same checkpointer/database; router timeout/provider/schema/target failures; specialist limit failure; finalization and persistence failure; streamed and non-streamed parity; exactly one durable assistant message; unique per-turn checkpoint IDs; and no cross-user/device/tool/evidence leakage. Disable external tracing in deterministic tests.

- [ ] **Step 4: Implement versioned checkpoint retention and privacy cleanup**

Define explicit retention for completed v2 turns, interrupted/HITL turns, failed turns, and unreadable v1 checkpoints. Cleanup enumerates exact validated `routing-v2:{conversation_id}:{turn_id}` IDs from owned metadata; it never deletes by an unbounded prefix. Conversation/account deletion removes all owned checkpoint IDs and HITL rows. Expired v1 data is deleted by a separately reviewed namespace job after the rollback window; v2 readers continue to reject it. Add tests for active-interrupt preservation, completed-turn expiry, malformed ID rejection, conversation deletion, and idempotent retries.

- [ ] **Step 5: Document deployment and rollback**

`routing-v2-rollout.md` must specify:

1. startup structured-output capability check;
2. deterministic suite and live evaluation artifacts required for canary;
3. canary dashboards/alerts for router error rate, p95 latency, invalid schema, target race, handoff correction, grounding abstention, and finalizer failures;
4. verification that new checkpoints use `routing-v2:`;
5. rollback by deploying the prior artifact—not by enabling old branches in the new code;
6. retention/expiry of v1 and v2 checkpoints, active-HITL preservation, privacy deletion, and exclusion of v1 from v2 reads.

- [ ] **Step 6: Run the focused production suite and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_checkpoint_retention_service.py tests/test_workflow_end_to_end.py tests/test_client_invocation_isolation.py tests/test_hitl_backend_regressions.py tests/test_message_service_event_streaming.py tests/test_model_usage_callsite_inventory.py
.\.venv\Scripts\python.exe -m ruff check app/services/conversation_turn_coordinator.py app/services/checkpoint_retention_service.py app/workers/cleanup_tasks.py tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_checkpoint_retention_service.py tests/test_workflow_end_to_end.py
git add app/services/conversation_turn_coordinator.py app/services/checkpoint_retention_service.py app/workers/cleanup_tasks.py tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_checkpoint_retention_service.py tests/test_workflow_end_to_end.py tests/test_client_invocation_isolation.py tests/test_hitl_backend_regressions.py tests/test_message_service_event_streaming.py tests/test_model_usage_callsite_inventory.py tests/fixtures/model_usage_callsite_manifest.json docs/operations/tool-execution-policy.md docs/operations/routing-v2-rollout.md
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
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q tests/test_workflow_contracts.py tests/test_workflow_state.py tests/test_checkpoint_serializer.py tests/test_routing_inventory.py tests/test_routing_context.py tests/test_router.py tests/test_routing_service.py tests/test_routing_startup_validation.py tests/test_production_workflow_graph.py tests/test_specialist_runtime.py tests/test_specialist_middleware.py tests/test_agent_transitions.py tests/test_rag_execution_graph.py tests/test_planning_execution_graph.py tests/test_output_validation.py tests/test_public_response_finalizer.py tests/test_workflow_error_contract.py tests/test_graph_stream_projection.py tests/test_workflow_concurrency.py tests/test_workflow_checkpoint_v2.py tests/test_checkpoint_retention_service.py tests/test_workflow_end_to_end.py tests/test_hitl_backend_regressions.py tests/test_rag_grounding.py tests/test_routing_release_gate.py
```

Expected: all pass; no process crash. If Windows still raises the previously observed native `pyarrow` access violation, record the exact command/exit code, run the same suite in the Linux CI/container image, and do not mark the gate passed until that environment succeeds.

- [ ] **Step 3: Run the full non-live suite**

```powershell
$env:LANGSMITH_TRACING='false'
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
.\.venv\Scripts\python.exe scripts/check_routing_release.py --dataset eval/routing/golden_v1.jsonl --review eval/routing/golden_v1.review.json --report .artifacts/routing-eval-v1.json
```

Expected: macro-F1 ≥ 0.90; every represented language ≥ 0.85; no language more than 0.05 below English; first-attempt structured success ≥ 0.99; after-retry success ≥ 0.999; zero silent chat substitutions; zero finalizer bypasses; zero unknown published evidence IDs.

- [ ] **Step 6: Review the final diff and commit verification-only fixes**

```powershell
git status --short
git diff --stat 643404d..HEAD
git diff --check
```

Confirm the diff contains no credentials, captured prompts, generated evaluation secrets, or `.artifacts` output. Commit only actual code/test/document fixes; do not commit the live evaluation output if it contains request text.

## Implementation Status — 2026-08-26

Tasks 1-11 are implemented and committed on `Thai-Postgre-FastAPI`. Task 12 was
skipped as redundant (see its section). Task 13 is partially done: the
conversation turn coordinator, per-turn checkpoint retention, and the rollout
runbook have landed; the end-to-end scenario suite has not. Task 14 is not
started.

**Landed and verified** (full non-live suite: 4534 passed, 3 pre-existing
failures unrelated to this work; `ruff check app tests` clean):

- Typed workflow contracts, `WorkflowState`, set-once/append-only reducers, and
  checkpoint round-trip allowlisting.
- Bounded language-neutral routing context and live specialist inventory.
- `RoutingService`: schema-constrained output, strict runtime resolution with no
  provider fallback, four typed failure codes, at most two attempts, no
  `chat_agent` substitution anywhere.
- The routing-v2 parent graph: one `route` node per new turn, dynamic
  `Command` transitions, per-turn checkpoint threads, `finalize -> END` as the
  only terminal edge, and no streaming pre-routing.
- Standard specialists (chat, search, canvas, image, custom) running inside
  per-invocation `create_agent` subgraphs with focused middleware.
- Handoffs as parent commands with a single transition resolver.
- The shared RAG execution graph, with grounding made mandatory and the
  shadow-mode rollout flags deleted.
- The Planning orchestrator with `Send` fan-out and typed worker results.
- The provenance policy registry and universal public finalization.
- Typed `WorkflowError` at the service boundary with allowlisted details.

**Also landed since:** `ConversationTurnCoordinator` (per-conversation advisory
lock, bounded acquisition, released in a `finally`, in-process backend rejected
in production); per-turn checkpoint retention that enumerates exact owned thread
IDs rather than deleting by prefix — which also fixed a real leak where deleting
a conversation left every turn's checkpoint behind; and
`docs/operations/routing-v2-rollout.md`.

**Known gaps, tracked by `tests/test_routing_legacy_removal.py`:**

1. The `rag_agent` and `planning_agent` graph nodes still run the pre-v2 loops.
   `RagExecutionGraphFactory` and `PlanningOrchestrator` are built and tested
   but are not yet what production executes. Grounding enforcement *did* ship
   on the live RAG path.
2. ~~`_recover_terminal_response` can publish unvalidated text.~~ **Fixed.**
   It no longer scans stream chunks or checkpoint messages; it returns the
   finalizer's response or nothing, and a turn with no finalized response
   yields a typed error.
3. `_tool_node` / `_approval_node` are unreachable from the graph but retained:
   they still hold canvas-edit denial and HITL edit-rewrite behavior that the
   v2 middleware has not absorbed.
4. Routing accuracy is unmeasured — Task 12's evaluation program was skipped,
   and no component here has been exercised against a live model.

---

## Final Acceptance Checklist

- [ ] Every new user turn enters one route node and calls `RoutingService.route(...)` once; that call makes at most two attempts against the same resolved provider/model.
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
- [ ] No public answer delta is emitted before output validation; the released deltas exactly reproduce finalized content.
- [ ] Every turn uses `routing-v2:{conversation_id}:{turn_id}`; resume uses the exact durable thread ID and continues without routing again.
- [ ] Same-conversation turns are serialized through persistence; different conversations remain concurrent.
- [ ] Checkpoint retention preserves active interrupts, expires completed/failed turns by policy, and honors conversation/account deletion.
- [ ] Legacy routing, fallback, continuation, duplicated RAG, shadow grounding, and stale recovery code are deleted.
- [ ] Deterministic, concurrency, streaming, HITL, grounding, and full non-live gates pass. (The live multilingual routing gate was removed with Task 12; routing accuracy is unmeasured.)

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-26-production-routing-refactor.md`.

Two execution options:

1. **Subagent-Driven (recommended):** Use `superpowers:subagent-driven-development` in this session, assign one task at a time, and perform spec/code reviews between tasks.
2. **Inline execution:** Use `superpowers:executing-plans` in a fresh session and execute sequentially with the listed verification checkpoints.
