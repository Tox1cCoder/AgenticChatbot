# Production Routing and Agent Execution Refactor Design

**Status:** Approved design, amended after production interrupt/replay verification

**Date:** 2026-08-26; root-cause amendment approved 2026-08-28

## Context

The current workflow combines several different concerns in one mutable
`selected_agent` field:

- new-turn semantic routing;
- stream preselection;
- canvas continuity;
- custom-agent stickiness;
- planning supervision;
- handoff execution;
- continuation and resume state.

This creates implicit precedence between unrelated policies. A canvas artifact
or previous custom agent can bypass semantic routing, a truthy preselection can
skip routing without validation, and a handoff overwrites the original routing
decision. The router is coupled directly to Gemini and silently defaults to
`chat_agent` on provider or parsing failure. Agent nodes also terminate through
different response paths, and the planning worker contains a second RAG loop
that does not invoke the graph RAG grounding gate.

The refactor is intentionally breaking. It does not preserve the old graph
state, checkpoint compatibility, routing fallbacks, or duplicated execution
paths.

## Goals

1. Make an LLM the sole authority for semantic routing on every new user turn.
2. Support multilingual routing without keyword, regex, language, product-name,
   or agent-name matching in application code.
3. Separate the immutable initial routing decision from execution-time agent
   transitions.
4. Use LangGraph and LangChain primitives for orchestration, tool loops,
   structured output, handoffs, interrupts, persistence, and subgraphs.
5. Route every public response through one mandatory validation and finalization
   boundary.
6. Use one RAG execution and grounding implementation for top-level work,
   planning workers, and future delegations.
7. Fail explicitly and observably when routing or validation cannot be
   completed; never guess `chat_agent`.
8. Provide deterministic control-plane validation, durable resume, bounded
   execution, tracing, and production evaluation coverage.
9. Isolate checkpoint state per user turn and define explicit concurrency,
   retention, and privacy-deletion behavior.
10. Make Planning-worker approval resume-safe without replaying completed
    workers, and make mutation retry guarantees explicit and enforceable.

## Non-goals

- Preserving old LangGraph checkpoints or interrupted runs.
- Maintaining the existing `selected_agent` API or compatibility aliases.
- Keeping a second provider, model, or `chat_agent` fallback inside the router.
- Encoding semantic routing decisions in Python conditionals.
- Replacing product-specific tool authorization, device routing, rich-response
  conversion, evidence construction, or public API event formatting with
  generic framework behavior.
- Rebuilding all product agents as one general-purpose prompt.

## Design principles

### Model-owned semantics

The routing model decides what the user means. Application code supplies
bounded context and validates the returned control-plane values. State such as
an active canvas, uploaded documents, an existing plan, attached custom agents,
and the previous responding agent is descriptive context, not a forced route.

### Code-owned invariants

Language-independent invariants remain deterministic. Code verifies that an
agent exists, is enabled, is attached when required, has permission to execute
its tools, has not exceeded transition or execution budgets, and has produced a
valid public response. These checks do not infer user intent.

### One owner per concern

- `RoutingService` owns new-turn classification.
- LangGraph owns state transitions, persistence, interrupts, and resume.
- Agent subgraphs own model/tool loops.
- `RagExecutionGraph` owns document evidence acquisition and grounding.
- `PublicResponseFinalizer` owns graph-level validation and response construction.
- `MessageService` owns the transactional conversation-history write.
- The API stream adapter owns the public event protocol and releases answer text
  only after validation and persistence succeed.

### No silent recovery

No failure may change the requested behavior without telling the caller.
Provider errors, invalid structured output, unavailable targets, execution
limits, and finalization failures return typed errors. Recovery never publishes
stale assistant text from a previous graph state.

## Architecture

```text
Persisted user message and turn ID
        |
        v
LLM triage router
        |
        v
Structured RoutingDecision
        |
        v
Selected specialist subgraph
        |
        v
Resolve execution transition
   |                     |
   | accepted handoff    | final response
   v                     v
Specialist subgraph   Validate output
                          |
                          v
                  Universal finalizer
                          |
                          v
                      graph END
                          |
                          v
                  MessageService persistence
                          |
                          v
                  public answer + complete
```

Planning is an orchestrator-worker specialist. RAG is a reusable specialist
subgraph. Standard chat, search, image, canvas, and custom agents use LangChain
agent loops with focused middleware. No specialist node connects directly to
`END`.

Each new turn uses a unique checkpoint thread ID:
`routing-v2:{conversation_id}:{turn_id}`, where `turn_id` is the persisted user
message ID. Resume uses the exact thread ID stored with the durable interrupt;
it never reconstructs a conversation-scoped ID. This prevents append reducers
from loading state belonging to an earlier turn.

## State model

The old `selected_agent` field is removed.

```python
from typing import Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from typing_extensions import NotRequired, TypedDict


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    agent_id: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)


class TurnIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str = Field(min_length=1, max_length=160)
    turn_id: str = Field(min_length=1, max_length=160)
    checkpoint_thread_id: str = Field(min_length=1, max_length=320)


class AgentTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    from_agent_id: str | None
    to_agent_id: str = Field(min_length=1, max_length=160)
    source: Literal["router", "handoff", "resume"]
    tool_call_id: str | None = None


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


class ResponseOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["response"] = "response"
    agent_id: str
    response: "AgentResponse"
    provenance: OutcomeProvenance


class HandoffOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["handoff"] = "handoff"
    agent_id: str
    handoff: AgentTransition


AgentOutcome = ResponseOutcome | HandoffOutcome


class TurnExecutionState(TypedDict):
    turn_identity: TurnIdentity
    routing_decision: NotRequired[RoutingDecision | None]
    active_agent_id: NotRequired[str | None]
    final_agent_id: NotRequired[str | None]
    agent_history: list[AgentTransition]
    pending_transition: NotRequired[PendingTransition | None]
    execution_phase: Literal[
        "routing",
        "executing",
        "awaiting_approval",
        "validating",
        "finalizing",
        "completed",
        "failed",
    ]
    response: NotRequired["AgentResponse | None"]
```

The actual graph state continues to include messages, conversation and user
identifiers, attachments, model request, artifacts, evidence, plan data, and
interrupt metadata. Reducers append messages, transitions, artifacts, evidence,
and worker results without overwriting earlier state. `routing_decision` uses a
set-once reducer: `None -> decision` and replay of the same frozen value are
valid, while replacement is a state error. All checkpointed Pydantic types are
explicitly allowlisted and must round-trip without degrading to dictionaries.
`OutcomeProvenance` is assembled from server-owned tool/runtime records; model
text cannot declare its own evidence, artifact, image, or validation authority.

`routing_decision` is written exactly once per new turn and is immutable for
that turn. `active_agent_id` changes as execution moves between agents.
`agent_history` records the initial route, every accepted handoff, and a resumed
execution transition. Worker identities live in structured planning worker
records and do not replace the public `active_agent_id`.

`TurnIdentity.request_id` comes from the API correlation ID when available and
otherwise from the stable user-message ID. `turn_id` is the persisted
user-message ID. The service constructs `checkpoint_thread_id` once and stores
that exact value in every interrupt payload and durable HITL record.

The model-reported confidence value is telemetry. It is not treated as a
calibrated probability and does not trigger a hard-coded routing branch.

## Routing context

`RoutingContextBuilder` creates one bounded, serializable routing request. It
contains:

- the current user message in its original language;
- bounded recent conversation context and the durable conversation summary;
- available base-agent descriptors and capability descriptions;
- attached and enabled custom-agent IDs, names, and descriptions;
- uploaded-document descriptors without raw document contents;
- active-canvas identity, title, revision, and ownership metadata without
  artifact source code;
- planning mode, lifecycle, and a bounded plan summary;
- the previous final agent descriptor;
- available tool and skill summaries relevant to agent selection;
- runtime time and locale context when available.

Default limits are 12 recent messages, 3,000 history tokens, 20 documents, 40
tools, 20 skills, 20 custom agents, 500 characters per untrusted descriptor
field, and 24,000 characters for the serialized routing payload. Settings are
validated as positive and may be tuned downward. Truncation is deterministic,
preserves descriptor IDs, and records content-free telemetry.

History and the previous final agent come from user-owned durable message
metadata, not the current checkpoint. Document descriptors use the repository's
bounded asynchronous lookup. Tool and skill descriptors are derived from the
authenticated request scope. Locale is passed only when supplied by trusted
request/account metadata; the application does not guess language from text.

The context contains no instruction that unconditionally selects an agent.
Untrusted user-supplied names, descriptions, personas, skill descriptions, and
conversation text are clearly delimited as data. Router instructions use a
`SystemMessage`; the bounded context is canonical JSON in a separate
`HumanMessage`. They are never concatenated into one user prompt.

## Routing service

`RoutingService` resolves the configured router model through the same runtime
model/provider abstraction used by other agents. Direct imports and clients for
Gemini are removed from the router.

The runtime resolver has a dedicated `router` key, requires the
`supports_structured_output` capability, and is always called with
`allow_provider_fallback=False`. A missing credential, unsupported adapter, or
provider/model mismatch fails closed; the router never consumes a fallback
candidate. Startup validates only static settings and installed adapter
capabilities. User-scoped credentials and request overrides are validated at
request time because they cannot be known safely at process startup.

The router is one logical `RoutingService.route(...)` invocation using a
Pydantic `RoutingDecision` schema through LangChain
`with_structured_output(include_raw=True)`. The configured router provider and
model must support schema-constrained output; unsupported configurations fail
startup validation. The application does not parse free text, search for agent
tokens, match explicit custom-agent names, or implement a tool-call fallback
strategy.

The router receives the live agent inventory as data. After the model returns,
`RoutingDecisionValidator` verifies:

- Pydantic schema validity;
- confidence range and bounded reason length;
- target presence in the supplied inventory;
- target enablement and attachment;
- target runtime availability.

Validation does not reinterpret the message or substitute another agent.

The routing node and `RoutingService.route(...)` each run exactly once per new
turn. The service may make at most two attempts against the same resolved model
object, provider, model, schema, inventory, context, and total deadline. It
returns a LangGraph
`Command(update=..., goto=...)`. Streaming observes this state update and emits
the initial `agent_selected` event. The streaming API no longer invokes the
routing node before starting the graph.

## Framework reuse boundary

The implementation uses the installed LangGraph 1.2 and LangChain 1.3 APIs.

### Use framework primitives for

- `StateGraph`, state reducers, and typed runtime context;
- `Command` for state updates and dynamic navigation;
- `Send` for planning worker fan-out;
- per-invocation specialist subgraphs;
- PostgreSQL checkpoint persistence;
- allowlisted checkpoint serialization for v2 workflow contracts;
- `interrupt()` and `Command(resume=...)` for human approval;
- `create_agent` for standard ReAct model/tool loops;
- `ToolNode` for bespoke RAG tool execution;
- structured model output through `with_structured_output`;
- built-in model-call and tool-call limit middleware;
- built-in human-in-the-loop middleware where the current approval policy maps
  directly to tool names;
- focused custom middleware where authorization, device provenance, artifacts,
  offloading, or usage recording requires application behavior.

### Keep application code for

- routing context construction and target validation;
- configured provider and model resolution;
- dynamic tool discovery and authorization;
- device-scoped execution;
- tool-result offloading and rich-response artifacts;
- RAG evidence budgets, evidence packs, citations, and grounding;
- usage accounting and domain metrics;
- public SSE event projection and API response schemas.

The implementation does not wrap a framework primitive with an equivalent
home-grown loop. Custom middleware remains small, single-purpose, and tested
independently.

## Specialist execution

Chat, search, image, canvas, and custom specialists are compiled as
per-invocation subgraphs using `create_agent`. Dynamic model choice, prompts,
tools, permissions, usage recording, and artifact capture enter through runtime
context or middleware rather than bespoke graph loops.

Each specialist wrapper returns an `AgentOutcome`. A response outcome returns a
parent-level `Command` to `validate_output`. A handoff tool inside a specialist
subgraph returns `Command(graph=Command.PARENT, goto="resolve_transition")`
with the pending transition update. Specialist wrappers have no static outgoing
edges, so dynamic `Command` routing cannot accidentally execute a second path.
The parent graph's `resolve_transition` node is the only component that chooses
the next top-level specialist.

Model and tool limits are enforced by middleware. The custom auto-continuation
round mechanism is removed. A complex task that requires explicit decomposition
uses Planning rather than silently starting another top-level execution round.

Specifically, specialists use `ModelCallLimitMiddleware` and
`ToolCallLimitMiddleware` with `exit_behavior="error"`. Only
`ModelCallLimitExceededError` and `ToolCallLimitExceededError` map to
`agent_execution_limit`; unrelated failures retain their typed mappings.
Authorization runs before HITL policy evaluation, and approval runs before the
tool implementation. Approve, edit, reject, and respond resume the checkpointed
invocation without duplicating a tool side effect or paired message.

The migration preserves current runtime behavior with explicit ownership:
runtime model overrides and usage recording live in middleware; bounded history
and message metadata live in the specialist factory; request budgets live in
request-budget/token instrumentation; authenticated client, widget,
read-tool-result, and web-research bindings live in tool-scope middleware; and
canvas snapshots, image injection/delivery, artifact harvesting, and request
time context remain in their domain factories. Compiled specialist graphs are
not cached across users or devices.

## Handoffs

The live handoff tool is built from the current reachable-agent inventory. The
model selects the target through the tool's schema. The tool returns a LangGraph
`Command` that records a pending transition and appends the matching
`ToolMessage`, preserving valid tool-call history. The request marker has the
deterministic message ID `handoff:{tool_call_id}`, which is also stored as
`PendingTransition.tool_message_id` and survives checkpoint serialization.

`resolve_transition` validates only control-plane invariants:

- the source is the current active agent;
- the target exists, is enabled, and is reachable;
- attached custom agents remain attached;
- the target has not already handled the turn;
- the configured delegation-depth limit is not exceeded;
- the tool-call ID matches the originating handoff call.

An accepted handoff retains exactly one paired tool message, appends
`AgentTransition(source="handoff")`, updates `active_agent_id`, clears the
pending transition, and navigates through the inventory's resolved graph node
name. A rejected handoff returns structured feedback with the same deterministic
message ID, causing the message reducer to replace the request marker rather
than append a duplicate, then returns to the resolved originating specialist.
Only accepted handoffs consume transition depth; resume transitions do not. The
original `routing_decision` never changes.

## Planning orchestration

Planning is selected semantically by the router; an existing plan does not
force it before routing. Once selected, Planning acts as the orchestrator for
that turn.

The Planning fan-out is part of the outer production `StateGraph`. It is not a
compiled child graph invoked from `planning_agent`, `planning_tools`, a tool
function, or an orchestration helper. This placement is a correctness
requirement: LangGraph resumes an interrupted subgraph call by re-entering the
parent node that invoked it. If that parent node constructs or invokes the
fan-out, already-completed sibling workers can execute again before their
pending writes are reused. Registering `planning_dispatch` and
`planning_worker` as real nodes in the same graph that owns the turn
checkpointer lets LangGraph preserve completed sibling writes and resume only
unfinished worker tasks.

Planning exposes `dispatch_subagents` to the model as a control schema only. It
is never executable through `execute_tool_calls`. The graph validates a whole
dispatch before scheduling any worker. A dispatch containing duplicate task
IDs, recursive Planning, unavailable agents, too many tasks, oversized fields,
or a mix of dispatch and handoff control calls receives paired deterministic
`ToolMessage` feedback and starts no partial fan-out.

Model-proposed dispatch data is converted to server-owned, checkpoint-safe
worker tasks. The server supplies dispatch identity, original position,
authenticated scope, effective model request, and the intersection of the
specialist's tools, device capabilities, and authorization policy. Model input
cannot widen a worker's tool scope.

```python
class WorkerTask(BaseModel):
    dispatch_id: str
    task_id: str
    position: int
    objective: str
    agent_id: str
    parent_context: dict[str, JsonValue]
    allowed_tool_ids: tuple[str, ...]
    model_request: dict[str, JsonValue] | None = None
    related_todo_ids: tuple[str, ...] = ()


class WorkerResult(BaseModel):
    dispatch_id: str
    task_id: str
    position: int
    agent_id: str
    status: Literal["completed", "failed"]
    content: str
    artifacts: list[dict]
    evidence: list[dict]
    images: list[dict]
    error_code: str | None = None
```

The reducer treats `(dispatch_id, task_id)` as the unique worker identity and
rejects duplicate writes. Collection orders results by `position`, then creates
one deterministic `ToolMessage` paired with the original
`dispatch_subagents` tool-call ID. Planning sees worker content as delimited
untrusted data and may then update todos or synthesize the public answer.

Workers receive an explicit objective as a bounded `HumanMessage`, bounded
parent context, allowed tools, model request, user/device/conversation scope,
HITL policy, custom-agent inventory, attachment descriptors, and output
contract. Objectives and model-proposed context never enter system
instructions. Workers cannot publish public assistant messages or perform
top-level handoffs; Planning owns delegation and synthesis. Worker intermediate
messages remain inside their per-invocation specialist execution.

A worker that requires human approval interrupts the outer workflow graph. It
does not fabricate an `awaiting_approval` result and is not given model-visible
refusal feedback. `GraphBubbleUp` and `GraphInterrupt` are always re-raised
before ordinary worker exception normalization. Resume continues the exact
checkpointed task and reuses completed sibling writes. Timeouts and execution
limits become failed results with stable `worker_timeout` or
`agent_execution_limit` codes. Agent disappearance becomes
`agent_unavailable`; recursive Planning is rejected as `recursive_planning`.

The parent-level topology is:

```text
planning_model
  | final answer --------------------------> planning_package
  | write_todos ---------------------------> planning_actions -> planning_model
  | hand_off ------------------------------> resolve_transition
  | dispatch_subagents
  v
planning_dispatch -- Send --> planning_worker -- join --> planning_collect
                                                        |
                                                        v
                                                 planning_model
```

`planning_package` routes to universal output validation and never to `END`.
`planning_actions` owns deterministic todo changes and rubric state; Planning
does not execute arbitrary product tools directly. A handoff is exclusive with
a dispatch in one model message because it changes parent control flow.

The topology preserves plan create/modify/review actions, revision/lifecycle
metadata, rubrics, todo changes, existing-plan context, custom-agent/model
overrides, and task-correlated subagent events. Defaults bound a turn to eight
worker tasks total, four concurrent workers, two dispatch waves, 4,000 objective
characters per task, and 12,000 parent-context characters. Limits apply across
the turn rather than independently to each dispatch. Oversized dispatches fail
validation visibly; they are not silently truncated. The second wave is the
only dependency/revision wave, preventing unbounded recursive delegation.

The corresponding validated settings are `planning_worker_max_tasks=8`,
`planning_worker_max_concurrency=4`,
`planning_worker_max_dispatch_waves=2`,
`planning_worker_objective_max_chars=4000`, and
`planning_parent_context_max_chars=12000`.

Planning synthesizes the collected results and returns one `AgentOutcome` to
the parent graph. Evidence and artifacts retain server-owned provenance through
synthesis. Task-correlated worker progress uses graph custom events rather than
checkpointed callback objects.

## Unified RAG execution and grounding

The inline RAG branch in `_run_agent_in_isolated_context` and the separate graph
RAG loop are replaced by one compiled `RagExecutionGraph`:

```text
prepare request
      |
      v
RAG model
  | tool calls
  v
ToolNode
  |
  v
collect evidence and artifacts
  |
  +----------------------> RAG model
                              |
                              | final answer
                              v
                     grounding validation
                       | valid
                       +----------> result
                       | invalid, first pass
                       +----------> one grounded regeneration
                       | invalid, second pass
                       +----------> explicit abstention
```

The RAG topology is compiled once when the production workflow is constructed
and invoked with request-scoped state for both top-level RAG and Planning
workers. Compiled graph structure is reusable; authenticated state, tools,
credentials, evidence allocation, and messages are not shared between
invocations. The RAG graph inherits the outer turn checkpointer and its approval
middleware interrupts before any gated tool in the model batch executes. There
is no per-run graph construction and no inline special case.

Evidence IDs, token budgets, evidence packs, tool artifacts, image provenance,
regeneration, and abstention follow one code path. A per-run server-owned
allocator creates evidence IDs, and typed reducers merge evidence. Duplicate or
ambiguous IDs fail validation rather than selecting the first match.

Grounding enforcement is mandatory for every RAG result, including a retrieval
with no evidence. A zero-evidence result may return a bounded clarification or
abstention but cannot make source-backed claims. The shadow-only and enforcement
rollout branches are removed. A RAG worker result is validated before it reaches
Planning. In addition, any public synthesis carrying RAG evidence is validated
against the accumulated server-owned evidence before publication. This prevents
a planning synthesis from distorting an otherwise grounded worker result.

## Output validation and universal finalization

Every specialist response flows through `validate_output` and `finalize`; no
agent has a direct edge to `END`.

`validate_output` selects policies from response provenance rather than relying
only on the final agent name. It applies:

- grounded claim and citation validation whenever RAG evidence is present and
  for every outcome that declares the `rag_grounding` policy, including an
  empty-evidence RAG result;
- artifact and selected-image provenance validation;
- tool-call/message pairing validation;
- public content and error-shape validation;
- agent-specific output contracts required by canvas and image delivery.

`PublicResponseFinalizer` then:

- requires the reserved durable assistant message ID and immutable routing
  decision;
- appends exactly one terminal `AIMessage`;
- attaches the immutable routing decision;
- sets `final_agent_id = active_agent_id` and attaches the initial, active, and
  final identities;
- attaches ordered transition history;
- merges validated tool artifacts, images, canvas data, planning metadata, and
  grounding metadata;
- normalizes rich-response placement;
- records applied policy IDs and implementation versions;
- records final usage and observability data;
- exposes `validated_public_content` for later stream projection;
- sets `execution_phase="completed"` and routes to `END`.

The finalizer builds an immutable response/state update locally before returning
it, so metadata normalization cannot leave partially mutated state. It
guarantees graph-level validation, response construction, and serialization; it
does not claim that the database write has committed. After graph completion,
`MessageService` transactionally persists the finalizer-owned response. A
persistence failure returns `response_persistence_failed` and publishes no
answer or completion event.

Intermediate tool-calling messages remain internal and receive derived IDs only
when checkpoint cleanup requires them. Terminal recovery does not scan old
messages for plausible assistant text. Specialist answer tokens remain private
until validation succeeds, and `validated_public_content` remains buffered until
the `MessageService` transaction commits.

Worker outputs use the same validation policy registry through a
`WorkerOutputFinalizer`, but they do not receive public message IDs or enter the
public conversation history.

## Error handling

Errors use a stable API payload:

```python
class WorkflowError(BaseModel):
    code: Literal[
        "routing_timeout",
        "routing_provider_unavailable",
        "routing_invalid_output",
        "routing_target_unavailable",
        "agent_execution_limit",
        "tool_execution_failed",
        "response_validation_failed",
        "finalization_failed",
        "response_persistence_failed",
        "conversation_turn_conflict",
    ]
    retriable: bool
    request_id: str
    details: dict[str, JsonValue] = Field(default_factory=dict)
```

The API/localization layer owns user-facing copy. Core workflow errors do not
depend on English messages. `request_id` comes from `TurnIdentity`, never an
optional provider response. Structured details use an allowlist and exclude
credentials, provider payloads, document content, prompt text, and stack traces.

### Routing

- `routing_timeout`: router exceeded `routing_timeout_seconds`; retriable.
- `routing_provider_unavailable`: configured provider failed; retriable.
- `routing_invalid_output`: structured output remained invalid; retriable.
- `routing_target_unavailable`: the selected live target disappeared before
  execution; retriable.

The router allows at most two provider calls per turn. A transient transport,
rate-limit, server, or schema failure may trigger the second call. It never
changes provider, model, or agent. After the second failure, the graph enters
`failed` and the API returns the typed error. Default
`routing_timeout_seconds` is 8.0 and remains configurable.

### Execution

- Recoverable tool invocation errors become bounded `ToolMessage` feedback.
- Authorization failures never reach the tool implementation.
- Approval-required calls interrupt and resume through LangGraph persistence.
- Worker and specialist boundaries re-raise LangGraph control-flow exceptions;
  they never normalize an interrupt into `tool_execution_failed`.
- Invalid worker dispatches fail atomically with paired model-visible feedback;
  no subset of their tasks starts.
- Worker-local failures use stable result codes: `worker_timeout`,
  `agent_execution_limit`, `agent_unavailable`, `recursive_planning`, or
  `tool_execution_failed`.
- `agent_execution_limit` ends the run when model or tool limits are exhausted.
- Repeated identical tool failures are terminated by middleware and reported as
  `tool_execution_failed`.
- Unexpected infrastructure exceptions bubble to the API boundary and tracing.

Checkpoint topology prevents an ordinary approval resume from replaying a
completed worker. It does not by itself make an external mutation exactly once
across a process crash after the mutation succeeds but before its checkpoint
write commits. Every mutation therefore receives a server-generated execution
key derived from `(checkpoint_thread_id, dispatch_id, task_id, tool_call_id)`.
Application-owned mutations record that key in a durable execution-receipt
table with a unique constraint in the same transaction as the mutation.
External adapters pass the key to providers that support idempotency and record
the provider receipt. A provider without idempotency support is described as
retry-safe only; the runtime never makes an untrue exactly-once guarantee.

The receipt stores a SHA-256 execution key, owning user/conversation/turn IDs,
tool identity, status (`reserved | completed | failed | outcome_unknown`), a
bounded serialized result or server-owned artifact reference, provider receipt
ID when available, and timestamps. The model cannot supply or read the key.
Repeated completed executions return the recorded result. A reserved execution
is retried only through an adapter with provider idempotency; otherwise a crash
after dispatch becomes `outcome_unknown` and requires explicit reconciliation
instead of automatic mutation replay.

### Validation

- A first RAG grounding failure performs one constrained regeneration.
- A second grounding failure returns a validated abstention.
- `response_validation_failed` fails the turn when a non-grounding public
  contract cannot be satisfied.
- `finalization_failed` fails the graph before publication if the terminal
  response cannot be normalized or serialized.
- `response_persistence_failed` fails the service boundary if the finalized
  response cannot be committed to conversation history.
- `conversation_turn_conflict` is a bounded, retriable failure to acquire the
  same-conversation turn coordinator.

No error path silently routes to chat, publishes stale content, or skips the
finalizer.

## Streaming and events

The public stream adapter consumes LangGraph update, message, interrupt, task,
and custom events and projects them into the existing API event vocabulary. It
does not invoke graph nodes itself.

A normal turn emits, in order:

1. one initial `agent_selected` event after the routing decision;
2. progress, tool, artifact, preview, and thinking events from the active
   specialist, but no public answer text;
3. an additional `agent_selected` event for every accepted handoff;
4. zero or more interrupt/resume events;
5. after validation, graph completion, and successful response persistence,
   public `message_delta` chunks that exactly reproduce
   `validated_public_content`;
6. exactly one `complete` event, or one typed `error` with no finalized answer,
   answer delta, or completion event after terminal/persistence failure.

Planning nodes receive an injected LangGraph `StreamWriter` and emit typed
custom events carrying dispatch, task, worker, tool-call, and lifecycle IDs.
This works on the supported Python 3.10 floor without relying on async context
variable propagation. Worker events never enter the main answer token stream.
RAG worker and top-level RAG events share the same event schema. The projector
filters nested graph namespaces so internal specialist/RAG/model messages
cannot leak into the main answer stream. No weak-reference event-sink registry,
sink token, or queue object enters checkpoint state.

Initial selection is derived from the first `routing_decision` state update.
Later selections are derived only from newly appended accepted handoff
transitions. A `source="resume"` transition is retained for audit history but
does not emit another selection or consume handoff depth. Resume loads the exact
durable checkpoint thread ID and never routes again.

## Concurrency and checkpoint lifecycle

Planning schedules worker branches through parent-level `Send` tasks. The
outer graph invocation applies `planning_worker_max_concurrency` as its
server-owned concurrency bound. If one worker interrupts, successfully
completed siblings remain as checkpoint pending writes; resuming the same turn
does not execute those sibling nodes again. Multiple interrupted workers retain
separate interrupt IDs. Resume decisions are mapped to exact interrupt IDs and
unanswered workers remain pending.

Interrupt presentation and recovery use the checkpoint's pending interrupt
objects as the source of truth. They do not require a worker-private tool call
to appear in the parent's last `AIMessage`, and they do not classify approval
by a hard-coded node-name allowlist. Aggregation merges provenance per tool-call
ID without allowing one parallel worker's metadata to overwrite another's.

Different conversations may execute concurrently. Turns for the same
conversation acquire a cross-process PostgreSQL advisory lock, or an
equivalently durable coordinator, before history/context snapshotting and hold
it through response persistence. Lock acquisition is bounded; failure returns
the retriable `conversation_turn_conflict` error. Production does not fall back
to an in-process-only lock. The coordinator releases the lock in a `finally`
path after success or failure.

Retention distinguishes completed v2 turns, failed v2 turns, active/interrupted
HITL turns, and unreadable v1 checkpoints. Cleanup obtains exact owned thread
IDs from durable metadata and validates the
`routing-v2:{conversation_id}:{turn_id}` shape before deletion; it never deletes
by an unbounded prefix. Active interrupts survive normal cleanup. Conversation
or account deletion removes every owned checkpoint ID and HITL row. V1 data is
ignored by v2 readers and expires through a separately reviewed namespace job
after the rollback window. Cleanup retries are idempotent.

## Observability

Every turn records:

- router provider, model, latency, attempt count, usage, and schema outcome;
- the routing target/confidence and live inventory version; model-generated
  reason text is excluded from metrics and ordinary logs;
- initial, active, and final agent IDs;
- accepted and rejected transition records;
- model/tool calls and configured limit consumption;
- interrupt and resume lifecycle;
- worker dispatch, duration, result status, and evidence counts;
- RAG validation, regeneration, abstention, coverage, and ambiguous evidence;
- finalizer validation policies and completion/failure;
- terminal workflow error code and retriable flag.

Metrics include route volume by bounded base-agent kind, invalid routing output
rate, target-race rate, routing latency, handoff correction rate, rejected
handoffs, transition depth, agent execution limits, worker failures, grounding
outcomes, and finalization failures. Metric labels are allowlisted enums plus bounded
provider/model/inventory identifiers. Request, conversation, user,
custom-agent-instance, message, and evidence IDs are forbidden as metric labels
and may appear only in access-controlled sampled logs/traces. Traces retain
bounded metadata and avoid raw secrets, credentials, document contents, and
unrestricted prompts.

## Testing strategy

### Router unit tests

- valid structured decisions for base and dynamic custom agents;
- timeout, transient error, schema error, and two-attempt exhaustion;
- strict runtime resolution with no provider/model fallback and request-time
  user-credential validation;
- unknown, disabled, detached, and concurrently removed targets;
- canvas, documents, planning, previous agent, tools, and skills appear only in
  routing context;
- no semantic message inspection in Python;
- multilingual and mixed-language fixtures across multiple scripts;
- exactly one logical routing-node execution per new turn;
- exactly one `RoutingService.route(...)` invocation per new turn, with no more
  than two same-model provider attempts.

### Graph integration tests

- routing uses `Command` to reach every specialist type;
- resume continues the interrupted node without starting a new routing turn;
- a real parent-level Planning fan-out completes `w1`, interrupts `w2`, and
  resumes in a newly constructed workflow against the same checkpointer with
  side effects exactly `w1, w2`, never `w1, w1, w2`;
- the same replay test runs against both the in-memory checkpointer and the
  PostgreSQL checkpointer used in production;
- a compiled Planning child graph, tool-invoked fan-out, or
  `planning_tools -> dispatch_subagents` execution path is absent from the
  production topology;
- multiple worker interrupts preserve separate payloads and provenance;
  approve, edit, and reject decisions address exact interrupt IDs while
  unanswered workers stay pending;
- all response paths traverse `validate_output` and `finalize`;
- initial decisions remain unchanged across handoffs;
- transition history is ordered and append-only;
- invalid, cyclic, self, detached, and over-depth handoffs produce paired
  model-visible feedback with one stable tool-message ID;
- HITL approve, edit, reject, and respond paths resume durably without duplicate
  messages or side effects;
- streaming emits selections, transitions, interrupts, completion, and errors
  exactly once and in order;
- different-conversation turns remain concurrent, same-conversation turns do
  not overlap context snapshot through persistence, and neither case leaks
  routing, tools, workers, evidence, or device context;
- every new turn uses a distinct v2 checkpoint thread, a fresh workflow process
  can resume the exact durable ID, and v1 state is ignored;
- checkpoint serialization restores typed frozen contracts rather than plain
  dictionaries;
- retention preserves active interrupts, expires eligible exact thread IDs,
  rejects malformed IDs, and honors conversation/account deletion.
- mutation execution receipts reject repeated execution keys, return the prior
  result on safe retry, and never leak keys or provider receipts into model
  arguments.

### Agent and RAG contract tests

- standard specialists use framework agent loops and configured middleware;
- top-level and worker RAG use the same graph factory and policy registry;
- both RAG entry points enforce the same evidence budget and grounding result;
- invalid citations regenerate once and then abstain;
- zero-evidence RAG still runs validation and returns clarification/abstention
  rather than unsupported source-backed claims;
- a Planning synthesis containing RAG evidence is validated again before
  publication;
- workers return `WorkerResult` and never public `AIMessage` objects;
- every worker receives its bounded objective, authenticated HITL policy,
  custom-agent snapshot, task-local model request, and server-restricted tool
  scope;
- `GraphBubbleUp` passes through Planning, specialist, RAG, and tool-execution
  exception boundaries;
- dispatch limits are turn-wide, oversized or invalid dispatches start zero
  workers, and at most one dependency/revision wave follows the initial wave;
- tool/model limits produce typed terminal errors;
- final messages always contain IDs, artifacts, routing metadata, transition
  history, and final-agent identity;
- no public answer delta appears before validation and response persistence, and
  emitted deltas exactly reproduce finalized content.

### Live routing evaluation

A versioned evaluation dataset contains multilingual, mixed-language,
multi-intent, ambiguous-follow-up, canvas, document, planning, custom-agent,
image, current-information, and general-chat cases. Labels may specify one
canonical `primary_agent_id` and an explicit non-empty set of
`acceptable_agent_ids` containing that primary label. Acceptable-set accuracy
counts a prediction as correct when it belongs to that set. Per-agent
precision/recall/F1, macro-F1, and the confusion matrix use the canonical
primary label so multi-label rows are not double-counted.

The dataset contains at least 210 cases: at least 30 each for English, Thai,
Vietnamese, Chinese, Japanese, Arabic, and mixed-language input, with every
intent category represented at least ten times. A separate review manifest
records the dataset SHA-256, approved status, reviewing team alias, timestamp,
and label-guideline version. Missing approval or a hash mismatch fails closed;
dataset generation cannot self-approve its labels.

The pre-deployment gate requires:

- overall routing macro-F1 of at least 0.90;
- accuracy of at least 0.85 for each represented language;
- no represented language more than 0.05 below English accuracy;
- structured-output success of at least 0.99 before retry;
- structured-output success of at least 0.999 after the bounded retry;
- zero silent `chat_agent` substitutions;
- zero published answers that bypass mandatory finalization in deterministic
  graph tests;
- zero unknown evidence IDs in published deterministic RAG tests.

Live model evaluation runs as a pre-deployment job, not ordinary unit-test CI.
It records model/provider versions with its results so a model update can be
compared against the prior accepted baseline. The release checker requires a
fresh passing report for the exact provider/model/inventory tuple being
deployed. A missing, stale, mismatched, or `not_run` report does not pass the
deployment gate. Deterministic test commands set `LANGSMITH_TRACING=false`.

## Breaking cutover

The new graph uses the per-turn
`routing-v2:{conversation_id}:{turn_id}` checkpoint namespace and graph version.
Existing conversation messages and durable summaries remain available as input
history, but previous checkpoints are never used as new-turn state. Old graph
checkpoints and interrupted executions are retained only for the documented
rollback window, ignored by the new graph, and then removed by the reviewed v1
retention job. Privacy deletion overrides ordinary retention.

The deployment contains:

- no dual routing implementation;
- no `selected_agent` compatibility field;
- no canvas or custom stickiness branch;
- no explicit-name regex matcher;
- no Gemini-specific router client;
- no `chat_agent` routing fallback;
- no inline planning-worker RAG loop;
- no executable `dispatch_subagents` tool or nested Planning fan-out graph;
- no `_run_agent_in_isolated_context` worker loop;
- no worker approval-refusal branch;
- no parent `planning_tools` or `rag_tools` stage;
- no weak-reference subagent event sink or checkpointed sink token;
- no duplicate Planning task/result contract or legacy dispatch-result JSON
  plumbing;
- no direct agent-to-`END` edges;
- no auto-continuation outer loop;
- no stale terminal-response recovery;
- no shadow-only grounding branch.

The change is deployed to a canary environment after unit, integration, RAG,
streaming, concurrency, and live routing evaluations pass. Operational rollback
reverts the deployment artifact; old runtime branches are not kept in the new
code. The live release report must match the provider, model, inventory, and
human-reviewed dataset hash used for the canary.

## Acceptance criteria

The refactor is complete when:

1. Every new user turn enters one route node and calls
   `RoutingService.route(...)` once; the call uses at most two attempts against
   the same configured provider/model and returns a validated
   `RoutingDecision`.
2. Application code contains no language-dependent or phrase-dependent routing
   rules.
3. Router failure returns a typed retriable error and never selects chat.
4. `routing_decision`, `active_agent_id`, `final_agent_id`, and append-only
   `agent_history` replace `selected_agent` throughout the runtime and public
   metadata.
5. Streaming does not pre-run the routing node.
6. Handoffs use LangGraph state commands and preserve valid message pairing.
7. Standard specialists use framework agent loops; RAG uses one bespoke shared
   subgraph.
8. Planning dispatch and worker fan-out are real nodes in the outer checkpointed
   graph; completed sibling writes survive approval resume, workers use isolated
   per-invocation specialist execution, and results are typed and uniquely keyed
   by dispatch/task identity.
9. Top-level RAG, worker RAG, and evidence-bearing public synthesis enforce the
   same grounding policy.
10. Every public response traverses validation and universal graph finalization;
    answer text and completion are released only after transactional persistence.
11. Every new turn uses a unique per-turn v2 checkpoint; resume uses the exact
    durable ID without routing, and retention/privacy cleanup follows the
    documented lifecycle.
12. Different conversations remain concurrent while same-conversation turns are
    serialized through persistence.
13. Legacy routing, fallback, continuation, duplicated RAG, executable
    `dispatch_subagents`, nested Planning fan-out, worker refusal, sink-token
    streaming, and terminal recovery paths are removed.
14. The reviewed dataset hash and the documented unit, integration, concurrency,
    streaming, grounding, non-live, and tuple-matched live-evaluation gates pass.
15. Mutation retries use stable server-owned execution keys and durable receipts;
    external exactly-once claims are made only when the provider supports them.

## Public references

- [OpenAI: A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- [OpenAI: Function calling and Structured Outputs](https://help.openai.com/en/articles/8555517-function-calling-updates)
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [LangChain: Structured model output](https://docs.langchain.com/oss/python/langchain/models#structured-output)
- [LangGraph: Graph API and Command](https://docs.langchain.com/oss/python/langgraph/graph-api#command)
- [LangChain: Multi-agent handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)
- [LangGraph: Subgraphs and persistence](https://docs.langchain.com/oss/python/langgraph/use-subgraphs#subgraph-persistence)
- [LangGraph: Interrupts and node replay](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph: Custom streaming data](https://docs.langchain.com/oss/python/langgraph/streaming#custom-data)
- [LangChain: Human-in-the-loop middleware](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [LangChain: ToolNode](https://docs.langchain.com/oss/python/langchain/tools#toolnode)
