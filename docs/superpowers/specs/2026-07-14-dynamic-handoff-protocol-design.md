# Dynamic Multi-Agent Handoff Design

## Goal

Make `hand_off` a consistent, LLM-directed graph transition for every agent
execution path. A RAG agent that decides it needs a web-capable agent must
transfer the same turn to that agent once, rather than re-entering its RAG
tool loop until the iteration cap is reached.

## Root cause

`hand_off` currently changes `selected_agent` only when the standard tool loop
or the planning tool loop invokes `_apply_hand_off_if_present`. The RAG tool
loop executes the tool and appends its `ToolMessage`, but it does not apply the
handoff state transition. Its graph edge also always returns to `rag_agent`.

Consequently, an LLM-selected RAG-to-search handoff leaves
`selected_agent == "rag_agent"` and routes to `rag_agent` again. The model
retries the same `hand_off` with different `reason` text until
`agentic_max_iterations` ends the loop.

## Design decisions

- Agent selection remains LLM-directed. There will be no keyword rule that
  routes document conversations or current-events questions to a particular
  agent.
- `hand_off` accepts exactly one argument: `target_agent`. Model-supplied
  `reason` is removed from the schema, tool output, persisted handoff metadata,
  prompts, and tests. The fixed stream classification `reason="handoff"`
  remains; it is not model-supplied rationale.
- The graph owns the transition. Agents select from a live, dynamically built
  roster; graph code validates and executes the selected target.
- Standard, planning, and RAG tool loops use the same handoff interpreter.
  Any present or future specialized tool loop must call that interpreter before
  deciding its local continuation edge.
- Execution budgets stay bounded. The iteration cap exposed this defect; it did
  not cause it. Tool, recursion, wall-clock, auto-continuation, and delegation
  loop guards remain necessary production controls.

## Handoff contract

`create_hand_off_tool` exposes a structured schema containing only
`target_agent: str` and returns canonical control-plane JSON:

```json
{"hand_off":"search_agent"}
```

The current agent receives an instance bound with the target ids currently
reachable from its live graph state. That roster contains all registered base
agents and all attached custom agents except the active agent. It is generated
for base agents even when no custom agent is attached; this removes the static
base-only tool's self-targeting and stale-roster behavior.

Custom agents continue to receive a dynamically scoped tool, but their runtime
authorization and the graph's execution-time authorization are derived from
the same roster helper. A free-form string in a model response is never enough
to access an unattached, disallowed, or self target.

## Transition flow

```text
active agent invokes hand_off(target_agent)
  -> current tool runner executes the pure tool
  -> shared handoff interpreter validates the live target and rewrites its output if rejected
  -> normal tool persistence writes the matching ToolMessage and interpreter updates state
  -> selected_agent becomes target_agent
  -> the tool-loop conditional edge enters the target's graph node
  -> target agent answers the original user request
```

The canonical graph state retains the handoff `AIMessage` and matching
`ToolMessage`, satisfying the tool-call pairing expected by LangChain and
LangGraph. The receiving agent's prompt excludes that control-plane pair, so
it sees the original user request rather than routing narration or raw handoff
JSON. This is deliberate context engineering, not a loss of audit state.

The handoff control-plane state records:

- `active: true` (so delegated-prompt filtering can identify an active transfer)
- `source_agent`
- `target_agent`
- `tool_call_id`

The existing stream uses a selected-agent change to emit one `agent_selected`
event with `reason="handoff"`; no free-text model rationale is emitted or
persisted.

## Shared interpreter and graph integration

Refactor `_apply_hand_off_if_present` into a graph-level helper that:

1. Detects `hand_off` outputs deterministically. Exactly one handoff call is
   permitted in one model message; multiple calls are rejected as ambiguous.
2. Parses only the canonical `{"hand_off": "target"}` shape and validates the
   target against the active agent's live roster.
3. Rewrites the matching output record to a normal tool-error response for a
   malformed, unknown/unattached, self, repeated, depth-exhausted, or ambiguous
   handoff. The runner then writes exactly one `ToolMessage` for each tool-call
   id; the interpreter never appends a duplicate message itself.
4. Updates `selected_agent`, delegation state, and the per-turn agent trail on
   acceptance.

The standard `tools`, `planning_tools`, and `rag_tools` nodes call the helper
before any `ToolMessage` is written. A valid handoff may accompany other tool
calls in the same model message; those calls retain their ordinary execution
and pairing semantics, while the graph transfers after the whole tool batch.
`rag_tools` detects a successful RAG-originated transfer before applying RAG
error or iteration-budget logic and returns the selected target's graph node.

The graph builder's routing map after `rag_tools` includes every base node plus
the static `custom_agent` multiplexing node, as the standard tool routing map
already does. This lets a dynamic runtime target be mapped through
`_route_target_for` without hard-coded agent names.

Every base-agent invocation, including RAG and Planning, receives the same
dynamic handoff tool and roster description. Planning merges it with its
supervisor tools. RAG accepts it through its agentic binding API. The tool
execution map receives the same invocation-scoped tool so its binding and
execution views agree. Standard-loop HITL approval and provenance helpers use
that same scoped execution map, rather than rebuilding an unscoped one.
Custom-agent runtime construction uses the same roster helper. There is no
static default handoff tool.

All delegated targets use the same control-plane message filtering. Planning
uses `_messages_for_selected_agent`; RAG continues to derive the original user
request and excludes `hand_off` ToolMessages from RAG evidence.

## Safety policy

`MAX_HANDOFF_DELEGATION_DEPTH` is a positive, validated configuration setting
(default `5`). It is a separate delegation guard, not a substitute for tool or
recursion limits. The handoff interpreter also prevents an agent from handing
off to itself or selecting any agent already present in the current turn's
agent-invocation trail. Those checks stop reciprocal RAG/search transfers
without constraining the LLM's first choice of the best available specialist.

When a handoff is invalid, the graph stays with the active agent and appends a
tool error explaining the constraint. The agent can answer directly or choose
another live target. A valid handoff changes agent ownership immediately and
does not loop back through the source agent's specialized continuation path.

## LangGraph alignment

LangGraph describes a handoff as a tool-driven state update followed by a
transition to the selected node, with a matching `ToolMessage` for the tool
call. This design preserves that state-and-control-flow contract while keeping
the project's centralized tool execution, HITL, deferred-tool binding, and SSE
projection layers. A direct `Command` return from the tool would require a
larger migration through those shared layers and is out of scope for this bug
fix.

References:

- https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
- https://docs.langchain.com/oss/python/langgraph/graph-api

## Verification

Add focused tests that prove:

1. A RAG-generated `hand_off(search_agent)` changes `selected_agent`, writes
   handoff metadata without a reason, and routes to `search_agent` in the same
   turn.
2. The RAG graph edge accepts all registered base targets and attached custom
   targets through the shared routing map.
3. Standard, planning, and RAG paths use the same validation behavior.
4. Bound and execution target rosters are live, exclude the active agent, and
   authorize only targets actually bound for the current agent, including RAG
   and Planning.
5. A malformed, self, unknown, unattached, repeated, or depth-exhausted
   handoff produces a paired tool error and does not mutate agent ownership.
6. Source handoff messages remain valid in canonical state but are excluded
   from every delegated target's prompt, including Planning.
7. Existing stream behavior still emits the delegated agent selection and
   returns that agent's final reply.
8. HITL approval and interrupt provenance use the same scoped handoff map as
   normal tool execution.

Run the focused handoff, RAG finalization, graph-contract, and router suites,
then the relevant complete AI workflow test suite. Existing baseline evidence
is 12 passing focused RAG/handoff tests in the project `.venv`.
