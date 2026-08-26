# Production Routing and Agent Execution Refactor Design

**Status:** Approved design

**Date:** 2026-08-26

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
- `PublicResponseFinalizer` owns the public output contract.
- The API stream adapter owns the public event protocol.

### No silent recovery

No failure may change the requested behavior without telling the caller.
Provider errors, invalid structured output, unavailable targets, execution
limits, and finalization failures return typed errors. Recovery never publishes
stale assistant text from a previous graph state.

## Architecture

```text
New user message
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
                         END
```

Planning is an orchestrator-worker specialist. RAG is a reusable specialist
subgraph. Standard chat, search, image, canvas, and custom agents use LangChain
agent loops with focused middleware. No specialist node connects directly to
`END`.

## State model

The old `selected_agent` field is removed.

```python
from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


class RoutingDecision(BaseModel):
    agent_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)


class AgentTransition(BaseModel):
    from_agent_id: str | None
    to_agent_id: str = Field(min_length=1)
    source: Literal["router", "handoff", "resume"]
    tool_call_id: str | None = None


class ResponseOutcome(BaseModel):
    kind: Literal["response"] = "response"
    agent_id: str
    response: "AgentResponse"


class HandoffOutcome(BaseModel):
    kind: Literal["handoff"] = "handoff"
    agent_id: str
    handoff: AgentTransition


AgentOutcome = ResponseOutcome | HandoffOutcome


class TurnExecutionState(TypedDict):
    routing_decision: NotRequired[RoutingDecision | None]
    active_agent_id: NotRequired[str | None]
    agent_history: list[AgentTransition]
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
and worker results without overwriting earlier state.

`routing_decision` is written exactly once per new turn and is immutable for
that turn. `active_agent_id` changes as execution moves between agents.
`agent_history` records the initial route, every accepted handoff, and a resumed
execution transition. Worker identities live in structured planning worker
records and do not replace the public `active_agent_id`.

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

The context contains no instruction that unconditionally selects an agent.
Untrusted user-supplied names, descriptions, personas, skill descriptions, and
conversation text are clearly delimited as data. Router system instructions are
passed through the provider's system-instruction interface rather than
concatenated into one user prompt.

## Routing service

`RoutingService` resolves the configured router model through the same runtime
model/provider abstraction used by other agents. Direct imports and clients for
Gemini are removed from the router.

The router is a one-shot model call using a Pydantic `RoutingDecision` schema
through LangChain `with_structured_output`. The configured router provider and
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

The routing node runs once per new turn and returns a LangGraph
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

## Handoffs

The live handoff tool is built from the current reachable-agent inventory. The
model selects the target through the tool's schema. The tool returns a LangGraph
`Command` that records a pending transition and appends the matching
`ToolMessage`, preserving valid tool-call history.

`resolve_transition` validates only control-plane invariants:

- the source is the current active agent;
- the target exists, is enabled, and is reachable;
- attached custom agents remain attached;
- the target has not already handled the turn;
- the configured delegation-depth limit is not exceeded;
- the tool-call ID matches the originating handoff call.

An accepted handoff appends `AgentTransition`, updates `active_agent_id`, clears
the pending transition, and navigates to the target specialist. A rejected
handoff appends model-visible structured feedback and returns to the originating
specialist. The original `routing_decision` never changes.

## Planning orchestration

Planning is selected semantically by the router; an existing plan does not
force it before routing. Once selected, Planning acts as the orchestrator for
that turn.

Planning creates typed worker tasks. LangGraph `Send` fans independent tasks out
to per-invocation specialist subgraphs. Reducers collect typed results:

```python
class WorkerResult(BaseModel):
    task_id: str
    agent_id: str
    status: Literal["completed", "failed", "awaiting_approval"]
    content: str
    artifacts: list[dict]
    evidence: list[dict]
    error_code: str | None = None
```

Workers receive an explicit objective, bounded context, allowed tools, model
request, user/device/conversation scope, and output contract. Workers cannot
publish public assistant messages or perform top-level handoffs; Planning owns
delegation and synthesis. Worker intermediate messages remain inside their
subgraphs.

Planning synthesizes the collected results and returns one `AgentOutcome` to
the parent graph. Evidence and artifacts retain server-owned provenance through
synthesis.

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

The same graph instance/factory is used for top-level RAG and planning workers.
There is no inline special case. Evidence IDs, token budgets, evidence packs,
tool artifacts, image provenance, regeneration, and abstention follow one code
path.

Grounding enforcement is mandatory. The shadow-only and enforcement rollout
branches are removed. A RAG worker result is validated before it reaches
Planning. In addition, any public synthesis carrying RAG evidence is validated
against the accumulated server-owned evidence before publication. This prevents
a planning synthesis from distorting an otherwise grounded worker result.

## Output validation and universal finalization

Every specialist response flows through `validate_output` and `finalize`; no
agent has a direct edge to `END`.

`validate_output` selects policies from response provenance rather than relying
only on the final agent name. It applies:

- grounded claim and citation validation whenever RAG evidence is present;
- artifact and selected-image provenance validation;
- tool-call/message pairing validation;
- public content and error-shape validation;
- agent-specific output contracts required by canvas and image delivery.

`PublicResponseFinalizer` then:

- assigns the durable assistant message ID;
- appends exactly one terminal `AIMessage`;
- attaches the immutable routing decision;
- attaches the initial, active, and final agent identities;
- attaches ordered transition history;
- merges validated tool artifacts, images, canvas data, planning metadata, and
  grounding metadata;
- normalizes rich-response placement;
- records final usage and observability data;
- sets `execution_phase="completed"` and routes to `END`.

Intermediate tool-calling messages remain internal and receive derived IDs only
when checkpoint cleanup requires them. Terminal recovery does not scan old
messages for plausible assistant text.

Worker outputs use the same validation policy registry through a
`WorkerOutputFinalizer`, but they do not receive public message IDs or enter the
public conversation history.

## Error handling

Errors use a stable API payload:

```python
class WorkflowError(BaseModel):
    code: str
    retriable: bool
    request_id: str
    details: dict[str, object] = Field(default_factory=dict)
```

The API/localization layer owns user-facing copy. Core workflow errors do not
depend on English messages.

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
- `agent_execution_limit` ends the run when model or tool limits are exhausted.
- Repeated identical tool failures are terminated by middleware and reported as
  `tool_execution_failed`.
- Unexpected infrastructure exceptions bubble to the API boundary and tracing.

### Validation

- A first RAG grounding failure performs one constrained regeneration.
- A second grounding failure returns a validated abstention.
- `response_validation_failed` fails the turn when a non-grounding public
  contract cannot be satisfied.
- `finalization_failed` fails the turn before publication if the terminal
  response cannot be serialized or persisted correctly.

No error path silently routes to chat, publishes stale content, or skips the
finalizer.

## Streaming and events

The public stream adapter consumes LangGraph update/message/interrupt events and
projects them into the existing API event vocabulary. It does not invoke graph
nodes itself.

A normal turn emits, in order:

1. one initial `agent_selected` event after the routing decision;
2. model, tool, artifact, and thinking events from the active specialist;
3. an additional `agent_selected` event for every accepted handoff;
4. zero or more interrupt/resume events;
5. exactly one `complete` event after universal finalization, or one `error`
   event after terminal failure.

Planning worker events carry task and worker IDs and never enter the main answer
token stream. RAG worker and top-level RAG events share the same event schema.

## Observability

Every turn records:

- router provider, model, latency, attempt count, usage, and schema outcome;
- the structured routing decision and live inventory version;
- initial, active, and final agent IDs;
- accepted and rejected transition records;
- model/tool calls and configured limit consumption;
- interrupt and resume lifecycle;
- worker dispatch, duration, result status, and evidence counts;
- RAG validation, regeneration, abstention, coverage, and ambiguous evidence;
- finalizer validation policies and completion/failure;
- terminal workflow error code and retriable flag.

Metrics include route volume by agent, invalid routing output rate, target-race
rate, routing latency, handoff correction rate, rejected handoffs, transition
depth, agent execution limits, worker failures, grounding outcomes, and
finalization failures. Traces retain bounded metadata and avoid raw secrets,
credentials, document contents, and unrestricted prompts.

## Testing strategy

### Router unit tests

- valid structured decisions for base and dynamic custom agents;
- timeout, transient error, schema error, and two-attempt exhaustion;
- unknown, disabled, detached, and concurrently removed targets;
- canvas, documents, planning, previous agent, tools, and skills appear only in
  routing context;
- no semantic message inspection in Python;
- multilingual and mixed-language fixtures across multiple scripts;
- exactly one logical routing-node execution per new turn.

### Graph integration tests

- routing uses `Command` to reach every specialist type;
- resume continues the interrupted node without starting a new routing turn;
- all response paths traverse `validate_output` and `finalize`;
- initial decisions remain unchanged across handoffs;
- transition history is ordered and append-only;
- invalid, cyclic, self, detached, and over-depth handoffs produce paired
  model-visible feedback;
- HITL approve, edit, and reject paths resume durably;
- streaming emits selections, transitions, interrupts, completion, and errors
  exactly once and in order;
- concurrent turns do not leak routing, tools, workers, evidence, or device
  context.

### Agent and RAG contract tests

- standard specialists use framework agent loops and configured middleware;
- top-level and worker RAG use the same graph factory and policy registry;
- both RAG entry points enforce the same evidence budget and grounding result;
- invalid citations regenerate once and then abstain;
- a Planning synthesis containing RAG evidence is validated again before
  publication;
- workers return `WorkerResult` and never public `AIMessage` objects;
- tool/model limits produce typed terminal errors;
- final messages always contain IDs, artifacts, routing metadata, transition
  history, and final-agent identity.

### Live routing evaluation

A versioned evaluation dataset contains multilingual, mixed-language,
multi-intent, ambiguous-follow-up, canvas, document, planning, custom-agent,
image, current-information, and general-chat cases. Labels may specify one
expected route or an explicit set of acceptable routes.

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
compared against the prior accepted baseline.

## Breaking cutover

The new graph uses a new checkpoint namespace and graph version. Existing
conversation messages and durable summaries remain available as input history.
Old graph checkpoints and interrupted executions are retained in storage for
audit/operational rollback but are ignored by the new graph.

The deployment contains:

- no dual routing implementation;
- no `selected_agent` compatibility field;
- no canvas or custom stickiness branch;
- no explicit-name regex matcher;
- no Gemini-specific router client;
- no `chat_agent` routing fallback;
- no inline planning-worker RAG loop;
- no direct agent-to-`END` edges;
- no auto-continuation outer loop;
- no stale terminal-response recovery;
- no shadow-only grounding branch.

The change is deployed to a canary environment after unit, integration, RAG,
streaming, concurrency, and live routing evaluations pass. Operational rollback
reverts the deployment artifact; old runtime branches are not kept in the new
code.

## Acceptance criteria

The refactor is complete when:

1. Every new user turn is semantically routed by the configured LLM using a
   validated `RoutingDecision`.
2. Application code contains no language-dependent or phrase-dependent routing
   rules.
3. Router failure returns a typed retriable error and never selects chat.
4. `routing_decision`, `active_agent_id`, and `agent_history` replace
   `selected_agent` throughout the runtime and public metadata.
5. Streaming does not pre-run the routing node.
6. Handoffs use LangGraph state commands and preserve valid message pairing.
7. Standard specialists use framework agent loops; RAG uses one bespoke shared
   subgraph.
8. Planning workers use isolated per-invocation subgraphs and typed results.
9. Top-level RAG, worker RAG, and evidence-bearing public synthesis enforce the
   same grounding policy.
10. Every public response traverses validation and universal finalization before
    `END`.
11. Legacy routing, fallback, continuation, duplicated RAG, and terminal
    recovery paths are removed.
12. The documented unit, integration, concurrency, streaming, grounding, and
    live-evaluation gates pass.

## Public references

- [OpenAI: A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- [OpenAI: Function calling and Structured Outputs](https://help.openai.com/en/articles/8555517-function-calling-updates)
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [LangChain: Structured model output](https://docs.langchain.com/oss/python/langchain/models#structured-output)
- [LangGraph: Graph API and Command](https://docs.langchain.com/oss/python/langgraph/graph-api#command)
- [LangChain: Multi-agent handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)
- [LangGraph: Subgraphs and persistence](https://docs.langchain.com/oss/python/langgraph/use-subgraphs#subgraph-persistence)
- [LangChain: Human-in-the-loop middleware](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [LangChain: ToolNode](https://docs.langchain.com/oss/python/langchain/tools#toolnode)
