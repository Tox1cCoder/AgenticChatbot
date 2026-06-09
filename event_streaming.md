# Event Streaming V3 and AI SDK V6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor backend streaming to a LangChain v3-style canonical event layer, keep the assistant-ui AI SDK v6 frontend working, keep Streamlit endpoints, stream custom subagents like DeepAgents subagents, and remove legacy streaming code after compatibility tests pass.

**Architecture:** LangGraph and custom subagent streams feed one internal v3 canonical event model. Two public adapters consume that model: an AI SDK v6 UI Message Stream adapter for assistant-ui and a legacy-compatible internal SSE adapter for Streamlit. The canonical layer separates model token deltas, reasoning, model tool-call availability, real tool execution, subagent lifecycle, interrupts, rich items, completion, errors, and heartbeats.

**Tech Stack:** Python, FastAPI `StreamingResponse`, LangChain/LangGraph event streaming, Pydantic, pytest, Vercel AI SDK UI Message Stream protocol, assistant-ui `@assistant-ui/react-ai-sdk`, Streamlit.

---

## Sources

- LangChain event streaming guide: https://docs.langchain.com/oss/python/langchain/event-streaming
- DeepAgents event streaming guide: https://docs.langchain.com/oss/python/deepagents/event-streaming
- LangChain Python `stream_events` reference: https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/stream_events
- assistant-ui AI SDK v6 runtime guide: https://www.assistant-ui.com/docs/runtimes/ai-sdk/v6
- Vercel AI SDK UI Message Stream protocol: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol

## Captured Requirements

- The frontend being built by another developer uses latest AI SDK v6 through assistant-ui.
- Current AI SDK endpoint behavior works with the frontend and must be captured in golden tests before refactoring.
- Streamlit endpoints must stay: `/messages/stream` and `/messages/resume-interrupt`.
- The backend should use best-practice, production-ready streaming semantics.
- Legacy and deprecated streaming code should be removed after new adapters and tests are in place.
- Custom `dispatch_subagents` should function like library subagents at the event level, even if the dispatcher remains custom.

## Contract Decision

Use AI SDK v6 as the external frontend wire protocol. Use LangChain v3-style events as the internal backend streaming contract.

These are separate version axes:

- AI SDK UI Message Stream currently keeps the response header `x-vercel-ai-ui-message-stream: v1`. Keep that header because assistant-ui and the AI SDK transport expect it.
- LangChain `stream_events(..., version="v3")` is the backend event source contract. Where the installed LangGraph package cannot emit official v3 events yet, add a normalizer that converts the existing `astream(..., stream_mode=["messages", "updates"])` tuples into the same internal v3 model. Keep that fallback enabled until the official v3 probe, graph integration tests, and production dependency pins all pass together.

Production implementation rules:

- Treat the canonical v3 event model as the only internal streaming contract after Task 7. Public adapters may emit legacy shapes, but service and graph code should not invent new dict event names.
- Treat model tool-call availability and actual tool execution as separate phases. `tool_call_available` is safe before HITL approval; `tool_execution_start` and `tool_execution_end` are only emitted after execution is actually authorized and underway.
- Preserve ordering with a monotonic `sequence` assigned at the canonical layer. Adapters must not reorder terminal events, title updates, interrupts, or subagent lifecycle events.
- Keep fallback tuple normalization as a production safety valve until a CI job proves `astream_events(..., version="v3")` works with the exact pinned resolver set in `environment.yml`.

## Current Implementation Audit

- `app/ai/graph.py` uses `self.graph.astream(..., stream_mode=["messages", "updates"])` in both `resume_with_decisions_stream` and `execute_request_stream`.
- `app/ai/graph.py` manually parses text, reasoning, tool-call chunks, tool results, node updates, interrupts, and final response in two duplicated loops.
- `app/services/ai_service.py` maps graph-specific events into custom events, but the vocabulary is mixed: `token`, `thinking`, `tool`, `node_complete`, `agent_selected`, `rich_items`, `interrupt`, `complete`, `error`.
- `app/services/stream_events.py` calls the vocabulary "canonical", but the list omits events currently emitted by the service layer, including `user_message_created`, `heartbeat`, and `title_updated`.
- Current `tool_start` means "the model exposed a tool call", not "the tool process started". This is a semantic problem for HITL approvals.
- `app/services/message_service.py` emits `title_updated` after `complete`, but `app/api/messages.py` terminates the SSE stream when `complete` is sent, so Streamlit cannot receive that title update.
- `app/api/ai_sdk.py` manually builds AI SDK UI Message Stream chunks and already emits many correct v6-compatible chunk names: `start`, `start-step`, `text-start`, `text-delta`, `reasoning-start`, `reasoning-delta`, `tool-input-start`, `tool-input-available`, `tool-output-available`, `finish-step`, `finish`, and `[DONE]`.
- `app/ai/planning_subagents.py` implements custom subagents through a blocking `dispatch_subagents` tool. Worker activity is collapsed into the tool output instead of streaming as `stream.subagents`.
- Current dependency pins are inconsistent: `pyproject.toml` allows broad LangChain ranges, while `environment.yml` pins concrete older versions than the resolver currently selects for v3 streaming. The installed local graph class can still expose only `astream_events` v1/v2 unless the exact active environment is upgraded as a coherent LangChain/LangGraph set.
- Legacy `tool_start` / `tool_end` events are not isolated to `app/ai/graph.py`; direct agent stream helpers in `app/ai/agents/search_agent.py` and `app/ai/agents/rag_agent.py` also emit those names and must be migrated or proven unreachable before cleanup is complete.

## Target Event Model

Create one internal canonical model under `app/services/event_streaming/`.

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


StreamSchemaVersion = Literal["v3"]

StreamEventType = Literal[
    "run_start",
    "message_start",
    "message_delta",
    "message_end",
    "reasoning_start",
    "reasoning_delta",
    "reasoning_end",
    "tool_call_delta",
    "tool_call_available",
    "tool_execution_start",
    "tool_execution_delta",
    "tool_execution_end",
    "agent_selected",
    "subagent_start",
    "subagent_message_delta",
    "subagent_tool_call_available",
    "subagent_tool_execution_start",
    "subagent_tool_execution_end",
    "subagent_end",
    "state_snapshot",
    "rich_items",
    "interrupt",
    "complete",
    "error",
    "heartbeat",
    "user_message_created",
    "title_updated",
]


class SubagentRef(BaseModel):
    id: str
    name: str
    path: list[str] = Field(default_factory=list)
    status: Literal["running", "completed", "failed", "timeout", "requires_approval"]


class V3StreamEvent(BaseModel):
    schema_version: StreamSchemaVersion = "v3"
    type: StreamEventType
    sequence: int
    run_id: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    agent: str | None = None
    node: str | None = None
    namespace: list[str] = Field(default_factory=list)
    subagent: SubagentRef | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Projection mapping:

- LangChain `stream.messages`: `message_start`, `message_delta`, `message_end`, `reasoning_start`, `reasoning_delta`, `reasoning_end`, `tool_call_delta`, `tool_call_available`.
- LangChain `stream.tool_calls`: `tool_execution_start`, `tool_execution_delta`, `tool_execution_end`.
- LangChain `stream.values`: `state_snapshot`.
- LangChain `stream.output`: `complete`.
- DeepAgents `stream.subagents`: `subagent_start`, `subagent_message_delta`, `subagent_tool_call_available`, `subagent_tool_execution_start`, `subagent_tool_execution_end`, `subagent_end`.

## File Structure

- Create `app/services/event_streaming/__init__.py`: public exports.
- Create `app/services/event_streaming/events.py`: Pydantic event models and helper constructors.
- Create `app/services/event_streaming/compat.py`: temporary legacy dict-to-v3 coercion used only during migration.
- Create `app/services/event_streaming/langchain_v3.py`: official `astream_events(version="v3")` reader and fallback tuple normalizer.
- Create `app/services/event_streaming/subagents.py`: event sink protocol and queue implementation for custom subagents.
- Create `app/services/event_streaming/ai_sdk_v6.py`: canonical v3 to AI SDK UI Message Stream adapter.
- Create `app/services/event_streaming/internal_sse.py`: canonical v3 to Streamlit legacy internal SSE adapter.
- Modify `app/ai/graph.py`: delegate stream parsing to `langchain_v3.py`; pass subagent event sink into planning dispatch state.
- Modify `app/ai/planning_subagents.py`: emit subagent lifecycle events while preserving current JSON tool result.
- Modify `app/ai/agents/search_agent.py`: remove direct legacy `tool_start` output or route it through the canonical bridge if this helper is still used.
- Modify `app/ai/agents/rag_agent.py`: remove direct legacy `tool_start` / `tool_end` output or route it through the canonical bridge if this helper is still used.
- Modify `app/services/ai_service.py`: consume canonical v3 events and keep rich-item capability filtering.
- Modify `app/services/message_service.py`: consume canonical v3 events, preserve persistence and cancellation, emit title update before terminal completion for Streamlit.
- Modify `app/api/ai_sdk.py`: move AI SDK chunk construction into `ai_sdk_v6.py` and keep endpoint paths.
- Modify `app/api/messages.py`: use `internal_sse.py` and keep endpoint paths.
- Modify `app/services/stream_events.py`: shrink to compatibility shims, then delete once no imports remain.
- Modify `README.md`: replace stale streaming event names with AI SDK v6 and internal v3 terminology.
- Modify `pyproject.toml` and `environment.yml`: align LangChain/LangGraph constraints that support v3 streaming.

## Implementation Log

Progress and design decisions recorded during execution.

### Environment facts (discovered 2026-06-09)
- Active interpreter for tests: `C:\Python314\python.exe` (Python 3.14, user site-packages at `%APPDATA%\Python\Python314\site-packages`). The `.venv`/`.conda` dirs in the repo are not what `python` resolves to in the working shell.
- `environment.yml` is a **separate** Python 3.11 conda freeze (`prefix: C:\Users\Vina32\anaconda3\envs\agents`). Its langchain pins were older/inconsistent than the active interpreter — the audit's "inconsistent pins" finding is literally true.
- pytest config: `asyncio_mode = "auto"`, `pytest-asyncio 1.3.0`.

### Task 1 findings — official v3 streaming is experimental (KEY DECISION)
Installed-version reality (langchain-core 1.4.2 / langgraph 1.2.4):
- `astream_events(version="v3")` is **only** implemented on `BaseChatModel` and `CompiledGraph`. On a generic `Runnable` (e.g. `RunnableLambda`, which the plan's original probe used) it raises `NotImplementedError` via `_astream_events_v3_unsupported`.
- On a `CompiledGraph` the call returns a **coroutine** that must be `await`ed first; awaiting yields an `AsyncGraphRunStream`. So the contract is `stream = await graph.astream_events(..., version="v3"); async for e in stream: ...` — NOT direct `async for`.
- The emitted events are dicts shaped `{type, method, params, seq}` (JSON-RPC-like), **not** the v1/v2 `{event, data, run_id, metadata, parent_ids}` shape the plan's probe + `iter_v3_events_from_graph` assumed.
- Iterating triggers `LangChainBetaWarning: The v3 streaming protocol on Pregel is experimental.`
- The pre-existing `astream(stream_mode=["messages","updates"])` path still works after the upgrade (26 streaming regression tests pass).

**Decision (user-approved): Adopt experimental v3.** Implement the Task 1 probe against the real contract (await + `CompiledGraph`) and rewrite Task 3's normalizer to parse the real `{type, method, params, seq}` schema. The fallback tuple normalizer is retained for runnables/versions that don't implement v3. The internal canonical `V3StreamEvent` model (Tasks 2,4–10) is independent of the upstream API and proceeds as written.

### Empirical v3 protocol reference (langgraph 1.2.4, in-process compiled graph)
Captured from `await compiled.astream_events(input, version="v3")`. Top-level event: `{"type":"event", "method":<channel>, "params":{...}, "seq":<int>, "event_id"?}`.

Default channels emitted locally (from `StreamMux` factories `ValuesTransformer, MessagesTransformer, LifecycleTransformer, SubgraphTransformer`): **`messages`, `values`, `lifecycle`**. There is **no `tools` channel** locally (the protocol's `tools`/`tool-started`/`tool-finished` events are server/deploy-only). Tool execution results are observed as `ToolMessage`s in `values` state.

- **`messages`**: `params.data == [message_event, metadata]`.
  - `message_event.event`: `message-start` (`{role, id, metadata?}`) → `content-block-start` (`{index, content:{type,...}}`, informational) → `content-block-delta` (`{index, delta}`) → `content-block-finish` (`{index, content}`) → `message-finish` (`{usage?}`).
  - `delta.type`: `text-delta` (`{text}`), `reasoning-delta` (`{reasoning}`), `block-delta` (`{fields:{type:"tool_call_chunk", id, name, args}}`), `data-delta`.
  - finish `content.type`: `text` (`{text}`), `reasoning` (`{reasoning}`), `tool_call` (`{id, name, args:dict}`), `invalid_tool_call`, image/audio/video/file, etc.
  - `metadata.langgraph_node` = node name; `metadata.run_id` present.
- **`values`**: `params.data` = full state dict; `params.interrupts` = tuple/list (non-empty ⇒ interrupt pending). Emitted once per superstep with the complete message list. Detect tool execution by new `ToolMessage` ids appearing.
- **`lifecycle`** (subgraph-scoped): `params.data` = `{event: started|running|completed|failed|interrupted, namespace, graph_name, trigger_call_id?, cause?, error?}`. Maps to subagent lifecycle for subgraph subagents only.
- **`namespace`**: `[]` top-level; `["subflow:<uuid>"]` for a subgraph node.

Canonical mapping used by the v3 normalizer: text-delta→`message_delta`, reasoning-delta→`reasoning_delta`, block-delta(tool_call_chunk)→`tool_call_delta`, finish tool_call→`tool_call_available`, new ToolMessage in values→`tool_execution_end`, values→`state_snapshot` (+`interrupt` when `params.interrupts`), lifecycle started/running→`subagent_start`, completed/failed/interrupted→`subagent_end`. `from_message_chunk`/`normalize_update_chunk` (the v1/v2 tuple helpers from the plan) are kept verbatim as the fallback path for non-graph runnables and are still unit-tested.

## Implementation Tasks

> **Task status:** Task 1 ✅ complete (probe rewritten for the real v3 contract; deps upgraded + pinned; 26 existing streaming tests still green).

### Task 1: Add Dependency Probe and Align LangChain Pins — ✅ COMPLETE

**Deviation from plan:** The plan's probe asserted `RunnableLambda.astream_events(version="v3")` yields `{event, data}` dicts and expected it to PASS after upgrade. Reality (see Implementation Log): v3 is only on `BaseChatModel`/`CompiledGraph`, must be awaited, and emits `{type, method, params, seq}`. The probe was rewritten to gate the real contract — `test_compiled_graph_astream_events_emits_v3_protocol` (positive) + `test_plain_runnable_does_not_implement_v3` (negative). Both pass. pyproject lower-bounds and environment.yml exact pins updated to the resolver set (incl. resolver-coupled `langchain-classic`, `langgraph-prebuilt`, `langgraph-sdk`, `langchain-protocol`, `langchain-text-splitters`, `google-genai`, `google-auth`). Installed versions match the pins exactly.

**Files:**
- Create: `tests/test_langchain_v3_dependency_probe.py`
- Modify: `pyproject.toml`
- Modify: `environment.yml`

- [ ] **Step 1: Write the dependency probe**

Create `tests/test_langchain_v3_dependency_probe.py`:

```python
from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda


@pytest.mark.asyncio
async def test_langchain_astream_events_accepts_v3():
    chain = RunnableLambda(lambda value: value)

    events = [event async for event in chain.astream_events("hello", version="v3")]

    assert events
    assert all(isinstance(event, dict) for event in events)
    assert all("event" in event for event in events)
    assert all("data" in event for event in events)
```

- [ ] **Step 2: Run the probe and record the current failure**

Run: `python -m pytest tests/test_langchain_v3_dependency_probe.py -q`

Expected before dependency update: FAIL because the installed LangChain/LangGraph stack does not accept event streaming version `v3`.

- [ ] **Step 3: Verify the dependency set resolves as a coherent unit**

Run this dry-run before changing files:

```powershell
python -m pip install --dry-run --upgrade "langchain>=1.3.4,<2.0.0" "langchain-community>=0.4.2,<1.0.0" "langchain-core>=1.4.2,<2.0.0" "langgraph>=1.2.4,<2.0.0" "langgraph-checkpoint>=4.1.1,<5.0.0" "langgraph-checkpoint-postgres>=3.1.0,<4.0.0" "langsmith>=0.8.11,<1.0.0" "langchain-google-genai>=4.2.4,<5.0.0" "langchain-openai>=1.2.2,<2.0.0" "langchain-mcp-adapters>=0.2.2,<1.0.0"
```

Expected: resolver succeeds and prints an install set compatible with:

```text
langchain==1.3.4
langchain-community==0.4.2
langchain-core==1.4.2
langgraph==1.2.4
langgraph-checkpoint==4.1.1
langgraph-checkpoint-postgres==3.1.0
langsmith==0.8.11
langchain-google-genai==4.2.4
langchain-openai==1.2.2
langchain-mcp-adapters==0.2.2
```

If the resolver selects newer patch versions, use the resolver output for exact `environment.yml` pins and keep the lower bounds in `pyproject.toml` at the lowest version that made the v3 probe pass.

- [ ] **Step 4: Update project dependency constraints**

In `pyproject.toml`, change the LangChain block to:

```toml
    # AI & LangChain Ecosystem
    "langchain>=1.3.4,<2.0.0",
    "langchain-community>=0.4.2,<1.0.0",
    "langchain-core>=1.4.2,<2.0.0",
    "langgraph>=1.2.4,<2.0.0",
    "langgraph-checkpoint>=4.1.1,<5.0.0",
    "langgraph-checkpoint-postgres>=3.1.0,<4.0.0",
    "langsmith>=0.8.11,<1.0.0",
    "langchain-google-genai>=4.2.4,<5.0.0",
    "langchain-openai>=1.2.2,<2.0.0",
    "langchain-mcp-adapters>=0.2.2,<1.0.0",
```

In `environment.yml`, update the pip pins for the same packages and resolver-coupled dependencies to exact versions. Include the selected `langgraph-prebuilt`, `langgraph-sdk`, `langchain-classic`, `langchain-protocol`, `google-genai`, and `google-auth` pins from the dry-run output when they change. This prevents CI and deployment from mixing an old checkpoint package with a newer graph package.

- [ ] **Step 5: Install and verify the probe passes**

Run:

```powershell
python -m pip install --upgrade "langchain>=1.3.4,<2.0.0" "langchain-community>=0.4.2,<1.0.0" "langchain-core>=1.4.2,<2.0.0" "langgraph>=1.2.4,<2.0.0" "langgraph-checkpoint>=4.1.1,<5.0.0" "langgraph-checkpoint-postgres>=3.1.0,<4.0.0" "langsmith>=0.8.11,<1.0.0" "langchain-openai>=1.2.2,<2.0.0" "langchain-google-genai>=4.2.4,<5.0.0" "langchain-mcp-adapters>=0.2.2,<1.0.0"
python -m pytest tests/test_langchain_v3_dependency_probe.py -q
```

Expected: PASS.

- [ ] **Step 6: Record the exact active versions**

Run:

```powershell
python -c "import importlib.metadata as m; pkgs=['langchain','langchain-community','langchain-core','langgraph','langgraph-checkpoint','langgraph-checkpoint-postgres','langsmith','langchain-openai','langchain-google-genai','langchain-mcp-adapters']; print('\n'.join(f'{p}=={m.version(p)}' for p in pkgs))"
```

Expected: output matches the exact pins added to `environment.yml`. If it does not, fix the pins or the environment before continuing.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml environment.yml tests/test_langchain_v3_dependency_probe.py
git commit -m "test: require langchain v3 event streaming support"
```

### Task 2: Define Canonical V3 Event Types

**Files:**
- Create: `app/services/event_streaming/__init__.py`
- Create: `app/services/event_streaming/events.py`
- Create: `tests/test_event_streaming_events.py`

- [ ] **Step 1: Write model tests**

Create `tests/test_event_streaming_events.py`:

```python
from __future__ import annotations

from app.services.event_streaming.events import (
    SubagentRef,
    V3StreamEvent,
    make_event,
)


def test_make_event_assigns_schema_and_sequence():
    event = make_event(
        "message_delta",
        sequence=7,
        agent="chat_agent",
        node="chat_agent",
        data={"text": "hello"},
    )

    assert isinstance(event, V3StreamEvent)
    assert event.schema_version == "v3"
    assert event.type == "message_delta"
    assert event.sequence == 7
    assert event.agent == "chat_agent"
    assert event.data == {"text": "hello"}


def test_subagent_ref_serializes_deepagents_shape():
    event = make_event(
        "subagent_start",
        sequence=1,
        subagent=SubagentRef(
            id="worker-a",
            name="search_agent",
            path=["planning_agent", "worker-a"],
            status="running",
        ),
    )

    payload = event.model_dump(mode="json")

    assert payload["schema_version"] == "v3"
    assert payload["subagent"]["id"] == "worker-a"
    assert payload["subagent"]["name"] == "search_agent"
    assert payload["subagent"]["path"] == ["planning_agent", "worker-a"]
    assert payload["subagent"]["status"] == "running"
```

- [ ] **Step 2: Run the model tests and verify they fail**

Run: `python -m pytest tests/test_event_streaming_events.py -q`

Expected: FAIL with an import error for `app.services.event_streaming.events`.

- [ ] **Step 3: Create event models**

Create `app/services/event_streaming/events.py` with the target event model from the "Target Event Model" section and add this constructor:

```python
def make_event(
    event_type: StreamEventType,
    *,
    sequence: int,
    run_id: str | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    agent: str | None = None,
    node: str | None = None,
    namespace: list[str] | None = None,
    subagent: SubagentRef | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> V3StreamEvent:
    return V3StreamEvent(
        type=event_type,
        sequence=sequence,
        run_id=run_id,
        conversation_id=conversation_id,
        message_id=message_id,
        agent=agent,
        node=node,
        namespace=list(namespace or []),
        subagent=subagent,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        data=dict(data or {}),
        metadata=dict(metadata or {}),
    )
```

Create `app/services/event_streaming/__init__.py`:

```python
from .events import SubagentRef, V3StreamEvent, make_event

__all__ = ["SubagentRef", "V3StreamEvent", "make_event"]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_event_streaming_events.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/event_streaming tests/test_event_streaming_events.py
git commit -m "feat: define v3 stream event model"
```

### Task 3: Build LangGraph to V3 Normalizer

**Files:**
- Create: `app/services/event_streaming/langchain_v3.py`
- Create: `tests/test_event_streaming_langgraph_normalizer.py`
- Modify: `app/ai/graph.py`

- [ ] **Step 1: Write normalizer tests**

Create `tests/test_event_streaming_langgraph_normalizer.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.services.event_streaming.langchain_v3 import (
    LangGraphV3Normalizer,
    normalize_update_chunk,
)


def test_message_chunk_text_becomes_message_delta():
    normalizer = LangGraphV3Normalizer()
    chunk = AIMessageChunk(content="hello")
    event = normalizer.from_message_chunk(chunk, metadata={"langgraph_node": "chat_agent"})

    assert event is not None
    assert event.type == "message_delta"
    assert event.node == "chat_agent"
    assert event.data["text"] == "hello"


def test_reasoning_content_block_becomes_reasoning_delta():
    normalizer = LangGraphV3Normalizer()
    chunk = SimpleNamespace(
        content="",
        content_blocks=[{"type": "reasoning", "reasoning": "thinking"}],
    )

    event = normalizer.from_message_chunk(chunk, metadata={"langgraph_node": "chat_agent"})

    assert event is not None
    assert event.type == "reasoning_delta"
    assert event.data["text"] == "thinking"


def test_tool_call_chunk_is_not_tool_execution_start():
    normalizer = LangGraphV3Normalizer()
    chunk = SimpleNamespace(
        content="",
        content_blocks=[
            {
                "type": "tool_call_chunk",
                "id": "call-1",
                "name": "search_documents",
                "args": '{"query":"x"}',
            }
        ],
    )

    event = normalizer.from_message_chunk(chunk, metadata={"langgraph_node": "rag_agent"})

    assert event is not None
    assert event.type == "tool_call_delta"
    assert event.tool_call_id == "call-1"
    assert event.tool_name == "search_documents"
    assert event.data["args_delta"] == '{"query":"x"}'


def test_tool_message_becomes_tool_execution_end():
    events = list(
        normalize_update_chunk(
            {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content="result",
                            tool_call_id="call-1",
                            name="search_documents",
                        )
                    ]
                }
            },
            sequence_start=10,
        )
    )

    assert len(events) == 1
    assert events[0].type == "tool_execution_end"
    assert events[0].sequence == 10
    assert events[0].tool_call_id == "call-1"
    assert events[0].tool_name == "search_documents"
    assert events[0].data["output"] == "result"


def test_final_ai_message_tool_calls_become_tool_call_available():
    events = list(
        normalize_update_chunk(
            {
                "chat_agent": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {"id": "call-1", "name": "search_documents", "args": {"query": "x"}}
                            ],
                        )
                    ]
                }
            },
            sequence_start=1,
        )
    )

    assert events[0].type == "tool_call_available"
    assert events[0].data["args"] == {"query": "x"}
```

- [ ] **Step 2: Run the normalizer tests and verify they fail**

Run: `python -m pytest tests/test_event_streaming_langgraph_normalizer.py -q`

Expected: FAIL with an import error for `app.services.event_streaming.langchain_v3`.

- [ ] **Step 3: Implement the normalizer**

Create `app/services/event_streaming/langchain_v3.py` with these public functions:

```python
from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Iterable
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from .events import V3StreamEvent, make_event


def _node_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("langgraph_node") or metadata.get("node")
    return str(value) if value else None


def _text_from_block(block: dict[str, Any]) -> str:
    for key in ("text", "content", "reasoning"):
        value = block.get(key)
        if isinstance(value, str):
            return value
    return ""


class LangGraphV3Normalizer:
    def __init__(self) -> None:
        self._sequence = 0

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def from_message_chunk(
        self,
        chunk: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> V3StreamEvent | None:
        node = _node_from_metadata(metadata)
        blocks = getattr(chunk, "content_blocks", None)
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type in {"thinking", "reasoning"}:
                    text = _text_from_block(block)
                    if text:
                        return make_event(
                            "reasoning_delta",
                            sequence=self.next_sequence(),
                            node=node,
                            agent=node,
                            data={"text": text},
                        )
                if block_type == "tool_call_chunk":
                    call_id = block.get("id") or block.get("tool_call_id")
                    return make_event(
                        "tool_call_delta",
                        sequence=self.next_sequence(),
                        node=node,
                        agent=node,
                        tool_call_id=str(call_id) if call_id else None,
                        tool_name=block.get("name"),
                        data={"args_delta": block.get("args") or ""},
                    )

        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            return make_event(
                "message_delta",
                sequence=self.next_sequence(),
                node=node,
                agent=node,
                data={"text": content},
            )
        return None


def _message_list_from_node_state(node_state: Any) -> list[Any]:
    if not isinstance(node_state, dict):
        return []
    messages = node_state.get("messages")
    return list(messages) if isinstance(messages, list) else []


def normalize_update_chunk(
    chunk: dict[str, Any],
    *,
    sequence_start: int,
) -> Iterable[V3StreamEvent]:
    sequence = sequence_start
    for node, node_state in chunk.items():
        for message in _message_list_from_node_state(node_state):
            if isinstance(message, ToolMessage):
                yield make_event(
                    "tool_execution_end",
                    sequence=sequence,
                    node=str(node),
                    agent=str(node),
                    tool_call_id=str(message.tool_call_id) if message.tool_call_id else None,
                    tool_name=getattr(message, "name", None) or "tool",
                    data={"output": message.content},
                )
                sequence += 1
            elif isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    yield make_event(
                        "tool_call_available",
                        sequence=sequence,
                        node=str(node),
                        agent=str(node),
                        tool_call_id=str(tool_call.get("id")) if tool_call.get("id") else None,
                        tool_name=tool_call.get("name"),
                        data={"args": tool_call.get("args") or {}},
                    )
                    sequence += 1
```

Add a stream wrapper in the same file:

```python
async def iter_v3_events_from_graph(
    graph: Any,
    state: dict[str, Any],
    *,
    config: dict[str, Any],
) -> AsyncGenerator[V3StreamEvent, None]:
    normalizer = LangGraphV3Normalizer()
    try:
        async for raw_event in graph.astream_events(state, config=config, version="v3"):
            yield make_event(
                "state_snapshot" if raw_event.get("event") == "on_chain_end" else "message_delta",
                sequence=normalizer.next_sequence(),
                run_id=str(raw_event.get("run_id")) if raw_event.get("run_id") else None,
                node=(raw_event.get("metadata") or {}).get("langgraph_node"),
                namespace=list(raw_event.get("parent_ids") or []),
                data={"raw": raw_event},
            )
    except (NotImplementedError, TypeError, ValueError):
        async for raw_chunk in graph.astream(
            state,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if not isinstance(raw_chunk, tuple) or len(raw_chunk) != 2:
                continue
            mode, payload = raw_chunk
            if mode == "messages":
                message_chunk, metadata = payload
                event = normalizer.from_message_chunk(message_chunk, metadata=metadata)
                if event is not None:
                    yield event
            elif mode == "updates" and isinstance(payload, dict):
                for event in normalize_update_chunk(
                    payload,
                    sequence_start=normalizer.next_sequence(),
                ):
                    yield event
```

- [ ] **Step 4: Refactor graph stream methods to call the wrapper**

In `app/ai/graph.py`, import `iter_v3_events_from_graph` and replace duplicated `self.graph.astream(... stream_mode=["messages", "updates"])` loops with:

```python
async for event in iter_v3_events_from_graph(self.graph, current_state, config=config):
    async for public_event in self._map_v3_stream_event(event, current_state=current_state):
        yield public_event
```

Keep `_map_v3_stream_event` in `app/ai/graph.py` during this task so persistence and interrupt behavior do not move at the same time.

- [ ] **Step 5: Run focused graph streaming tests**

Run:

```powershell
python -m pytest tests/test_event_streaming_langgraph_normalizer.py tests/test_graph_streaming_tool_events.py tests/test_graph_handoff_streaming.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add app/services/event_streaming/langchain_v3.py app/ai/graph.py tests/test_event_streaming_langgraph_normalizer.py
git commit -m "feat: normalize langgraph stream events to v3"
```

### Task 4: Stream Custom Subagents Like DeepAgents Subagents

**Files:**
- Create: `app/services/event_streaming/subagents.py`
- Create: `tests/test_event_streaming_subagents.py`
- Modify: `app/ai/planning_subagents.py`
- Modify: `app/ai/graph.py`

- [ ] **Step 1: Write subagent sink tests**

Create `tests/test_event_streaming_subagents.py`:

```python
from __future__ import annotations

import asyncio
import pytest

from app.ai.planning_subagents import (
    DispatchSubagentsInput,
    PlanningSubagentDispatcher,
    PlanningSubagentTask,
)
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.services.event_streaming.subagents import SubagentEventSink


class FakeWorkflow:
    async def _run_agent_in_isolated_context(self, **kwargs):
        return AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id=kwargs["agent_name"],
            message=AgentMessage(role=MessageRole.ASSISTANT, content="worker answer"),
            metadata={"provider": "openai", "model": "gpt-5"},
            tool_artifacts=[
                {
                    "tool_call_id": "sub-call-1",
                    "tool": "search_documents",
                    "output": "source text",
                    "status": "success",
                }
            ],
        )


@pytest.mark.asyncio
async def test_dispatcher_emits_subagent_start_and_end_events():
    sink = SubagentEventSink()
    dispatcher = PlanningSubagentDispatcher(workflow=FakeWorkflow(), event_sink=sink)
    request = DispatchSubagentsInput(
        tasks=[
            PlanningSubagentTask(
                id="worker-a",
                agent="search_agent",
                task="Find source material.",
            )
        ]
    )

    result = await dispatcher.dispatch(request, parent_state={"context": {}})
    events = await sink.drain()

    assert result.status == "completed"
    assert [event.type for event in events] == [
        "subagent_start",
        "subagent_tool_execution_end",
        "subagent_end",
    ]
    assert events[0].subagent.id == "worker-a"
    assert events[0].subagent.name == "search_agent"
    assert events[0].subagent.path == ["planning_agent", "worker-a"]
    assert events[-1].subagent.status == "completed"
    assert events[-1].data["output"] == "worker answer"
```

- [ ] **Step 2: Run the subagent tests and verify they fail**

Run: `python -m pytest tests/test_event_streaming_subagents.py -q`

Expected: FAIL because `SubagentEventSink` and dispatcher `event_sink` do not exist.

- [ ] **Step 3: Implement the subagent event sink**

Create `app/services/event_streaming/subagents.py`:

```python
from __future__ import annotations

import asyncio
from typing import Any

from .events import SubagentRef, V3StreamEvent, make_event


class SubagentEventSink:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[V3StreamEvent] = asyncio.Queue()
        self._sequence = 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    async def emit(
        self,
        event_type: str,
        *,
        task_id: str,
        agent_name: str,
        status: str,
        data: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        await self._queue.put(
            make_event(
                event_type,
                sequence=self._next_sequence(),
                subagent=SubagentRef(
                    id=task_id,
                    name=agent_name,
                    path=["planning_agent", task_id],
                    status=status,
                ),
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                data=data or {},
            )
        )

    async def drain(self) -> list[V3StreamEvent]:
        events: list[V3StreamEvent] = []
        while not self._queue.empty():
            events.append(await self._queue.get())
        return events
```

- [ ] **Step 4: Emit worker lifecycle events from the dispatcher**

Modify `PlanningSubagentDispatcher.__init__` in `app/ai/planning_subagents.py`:

```python
def __init__(
    self,
    *,
    workflow: _IsolatedAgentRunner,
    settings: Any | None = None,
    event_sink: Any | None = None,
) -> None:
    self._workflow = workflow
    self._settings = settings or global_settings
    self._event_sink = event_sink
```

At the start of `run_one`, emit:

```python
if self._event_sink is not None:
    await self._event_sink.emit(
        "subagent_start",
        task_id=task.id,
        agent_name=task.agent,
        status="running",
        data={
            "task": task.task,
            "related_todo_ids": list(task.related_todo_ids),
            "expected_output": task.expected_output,
        },
    )
```

After `worker_artifacts = list(response.tool_artifacts or [])`, emit each artifact:

```python
if self._event_sink is not None:
    for artifact in worker_artifacts:
        await self._event_sink.emit(
            "subagent_tool_execution_end",
            task_id=task.id,
            agent_name=task.agent,
            status="running",
            tool_call_id=str(artifact.get("tool_call_id"))
            if artifact.get("tool_call_id")
            else None,
            tool_name=artifact.get("tool"),
            data={
                "output": artifact.get("output"),
                "error": artifact.get("error"),
                "render": artifact.get("render"),
                "status": artifact.get("status"),
            },
        )
```

Before each return from `run_one`, emit `subagent_end` with the result status and answer:

```python
async def _emit_end(result: PlanningSubagentResult) -> PlanningSubagentResult:
    if self._event_sink is not None:
        await self._event_sink.emit(
            "subagent_end",
            task_id=result.id,
            agent_name=result.agent,
            status=result.status,
            data={
                "output": result.answer,
                "summary": result.summary,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
                "requested_model": result.requested_model,
                "resolved_model": result.resolved_model,
            },
        )
    return result
```

Wrap each `return _result(...)` in `return await _emit_end(_result(...))`.

- [ ] **Step 5: Wire the sink into graph streaming**

In `app/ai/graph.py`, create a `SubagentEventSink` before streaming starts, put it into the graph state context, and pass it into `PlanningSubagentDispatcher` when executable tools are built:

```python
from app.services.event_streaming.subagents import SubagentEventSink

subagent_event_sink = SubagentEventSink()
current_state.setdefault("context", {})["subagent_event_sink"] = subagent_event_sink
```

In `_build_planning_internal_tools`, read it:

```python
context = state.get("context") if isinstance(state.get("context"), dict) else {}
event_sink = context.get("subagent_event_sink")
dispatcher = (
    PlanningSubagentDispatcher(
        workflow=self,
        settings=settings,
        event_sink=event_sink,
    )
    if executable
    else None
)
```

Drain sink events between graph chunks and yield them as canonical v3 events.

- [ ] **Step 6: Run subagent tests**

Run:

```powershell
python -m pytest tests/test_event_streaming_subagents.py tests/test_message_service_subagent_streaming.py tests/test_demo_subagent_activity.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add app/services/event_streaming/subagents.py app/ai/planning_subagents.py app/ai/graph.py tests/test_event_streaming_subagents.py
git commit -m "feat: stream custom subagent lifecycle events"
```

### Task 5: Add AI SDK V6 Adapter Golden Tests

**Files:**
- Create: `app/services/event_streaming/ai_sdk_v6.py`
- Create: `tests/test_ai_sdk_v6_stream_contract.py`
- Modify: `app/api/ai_sdk.py`

- [ ] **Step 1: Write AI SDK v6 contract tests**

Create `tests/test_ai_sdk_v6_stream_contract.py`:

```python
from __future__ import annotations

import json

import pytest

from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import V3StreamEvent, make_event


async def _collect_payloads(source):
    adapter = AISDKV6StreamAdapter(
        lambda: source(),
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
    )
    chunks = [chunk async for chunk in adapter.iter_sse()]
    payloads = []
    for line in "".join(chunks).splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        payloads.append(raw if raw == "[DONE]" else json.loads(raw))
    return payloads


@pytest.mark.asyncio
async def test_text_stream_maps_to_ui_message_chunks():
    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "hel"})
        yield make_event("message_delta", sequence=2, data={"text": "lo"})
        yield make_event("complete", sequence=3, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)

    assert [payload["type"] for payload in payloads[:-1]] == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
        "text-delta",
        "text-end",
        "finish-step",
        "finish",
    ]
    assert payloads[3]["delta"] == "hel"
    assert payloads[4]["delta"] == "lo"
    assert payloads[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_reasoning_stream_maps_to_ai_sdk_reasoning_chunks():
    async def source():
        yield make_event("reasoning_delta", sequence=1, data={"text": "plan"})
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)
    types = [payload["type"] for payload in payloads if payload != "[DONE]"]

    assert "reasoning-start" in types
    assert "reasoning-delta" in types
    assert "reasoning-end" in types


@pytest.mark.asyncio
async def test_tool_call_and_tool_output_map_to_ai_sdk_tool_chunks():
    async def source():
        yield make_event(
            "tool_call_available",
            sequence=1,
            tool_call_id="call-1",
            tool_name="search_documents",
            data={"args": {"query": "x"}},
        )
        yield make_event(
            "tool_execution_end",
            sequence=2,
            tool_call_id="call-1",
            tool_name="search_documents",
            data={"output": "result", "render": {"type": "text", "text": "result"}},
        )
        yield make_event("complete", sequence=3, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)
    tool_payloads = [
        payload for payload in payloads
        if payload != "[DONE]" and str(payload.get("type", "")).startswith("tool-")
    ]

    assert [payload["type"] for payload in tool_payloads] == [
        "tool-input-start",
        "tool-input-available",
        "tool-output-available",
    ]
    assert tool_payloads[1]["input"] == {"query": "x"}
    assert tool_payloads[2]["output"] == "result"
    assert tool_payloads[2]["render"]["type"] == "text"


@pytest.mark.asyncio
async def test_interrupt_terminates_ui_stream():
    async def source():
        yield make_event(
            "interrupt",
            sequence=1,
            data={
                "thread_id": "thread-1",
                "pending_tool_calls": [],
                "interrupt": {"interrupt_id": "int-1"},
                "message": "Approval needed",
            },
        )
        yield make_event("message_delta", sequence=2, data={"text": "unreachable"})

    payloads = await _collect_payloads(source)

    assert any(payload != "[DONE]" and payload.get("type") == "data-interrupt" for payload in payloads)
    assert payloads[-1] == "[DONE]"
    assert not any(payload != "[DONE]" and payload.get("delta") == "unreachable" for payload in payloads)
```

- [ ] **Step 2: Run the contract tests and verify they fail**

Run: `python -m pytest tests/test_ai_sdk_v6_stream_contract.py -q`

Expected: FAIL with an import error for `app.services.event_streaming.ai_sdk_v6`.

- [ ] **Step 3: Move AI SDK stream state and SSE construction into the adapter**

Create `app/services/event_streaming/ai_sdk_v6.py` by moving the current `StreamState`, `_sse`, heartbeat wrapper, and event handlers from `app/api/ai_sdk.py`. Rename `StreamState` to `AISDKV6StreamState` and expose:

```python
class AISDKV6StreamAdapter:
    def __init__(
        self,
        event_source_factory: Callable[[], AsyncGenerator[dict[str, Any] | V3StreamEvent, None]],
        state: AISDKV6StreamState,
    ) -> None:
        self._event_source_factory = event_source_factory
        self._state = state

    async def iter_sse(self) -> AsyncGenerator[str, None]:
        yield _sse({"type": "start", "messageId": self._state.message_id})
        yield _sse({"type": "start-step"})
        yield _sse({"type": "text-start", "id": self._state.text_id})
        self._state.text_started = True
        async for raw_event in self._events_with_heartbeats():
            event = self._coerce_event(raw_event)
            async for chunk in self._map_event(event):
                yield chunk
            if event.type in {"interrupt", "error", "complete"}:
                break
        if self._state.text_started:
            yield _sse({"type": "text-end", "id": self._state.text_id})
        if self._state.reasoning_started:
            yield _sse({"type": "reasoning-end", "id": self._state.reasoning_id})
        yield _sse({"type": "finish-step"})
        yield _sse({"type": "finish"})
        yield "data: [DONE]\n\n"
```

Mapping rules:

- `dict` input is accepted only as a temporary migration bridge. Coerce legacy `token`, `thinking`, `tool`, `tool_start`, `tool_end`, `interrupt`, `complete`, `error`, `heartbeat`, `rich_items`, `user_message_created`, and `title_updated` into `V3StreamEvent` before mapping. Delete this bridge in Task 9 after `MessageService` yields only canonical events.
- `message_delta` -> `text-delta`.
- `reasoning_delta` -> lazy `reasoning-start`, then `reasoning-delta`.
- `tool_call_available` -> `tool-input-start`, then `tool-input-available`.
- `tool_execution_end` -> `tool-output-available`.
- `rich_items` -> `data-rich-items` only when `inline_rich_response_v1` is true.
- `interrupt` -> `data-interrupt`, then terminal chunks.
- `complete` -> optional file parts and `data-assistant-message`, then terminal chunks.
- `heartbeat` -> `heartbeat`.
- `agent_selected`, `subagent_*`, `state_snapshot`, `title_updated`, and `user_message_created` -> transient `data-*` chunks only when a frontend capability flag requests them. Without that flag, do not send them to assistant-ui. Legacy `node_complete` and `continuation_start` must enter this adapter as `state_snapshot` events with `data["legacy_type"]`.

- [ ] **Step 4: Keep API compatibility in `app/api/ai_sdk.py`**

Replace the local stream builder with:

```python
from app.services.event_streaming.ai_sdk_v6 import (
    AISDKV6StreamAdapter,
    AISDKV6StreamState as StreamState,
)


def _build_ui_message_stream_response(
    event_source_factory,
    state: StreamState,
) -> StreamingResponse:
    adapter = AISDKV6StreamAdapter(event_source_factory, state)
    return StreamingResponse(
        adapter.iter_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )
```

Keep the public function name `_build_ui_message_stream_response` so existing tests and endpoints keep working during the transition.

- [ ] **Step 5: Run AI SDK tests**

Run:

```powershell
python -m pytest tests/test_ai_sdk_v6_stream_contract.py tests/test_rich_response_streaming.py tests/client_backend/test_sse_keepalive.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add app/services/event_streaming/ai_sdk_v6.py app/api/ai_sdk.py tests/test_ai_sdk_v6_stream_contract.py
git commit -m "feat: add ai sdk v6 stream adapter"
```

### Task 6: Add Streamlit Internal SSE Adapter

**Files:**
- Create: `app/services/event_streaming/internal_sse.py`
- Create: `tests/test_internal_sse_stream_contract.py`
- Modify: `app/api/messages.py`
- Modify: `app/services/message_service.py`

- [ ] **Step 1: Write internal SSE contract tests**

Create `tests/test_internal_sse_stream_contract.py`:

```python
from __future__ import annotations

import json

import pytest

from app.services.event_streaming.events import make_event
from app.services.event_streaming.internal_sse import legacy_event_from_v3


def test_message_delta_maps_to_streamlit_token_event():
    event = make_event("message_delta", sequence=1, data={"text": "hello"})

    legacy = legacy_event_from_v3(event)

    assert legacy == {"type": "token", "content": "hello"}


def test_reasoning_delta_maps_to_streamlit_thinking_event():
    event = make_event("reasoning_delta", sequence=1, data={"text": "plan"})

    legacy = legacy_event_from_v3(event)

    assert legacy == {"type": "thinking", "content": "plan"}


def test_tool_call_available_maps_to_single_legacy_tool_start():
    event = make_event(
        "tool_call_available",
        sequence=1,
        tool_call_id="call-1",
        tool_name="search_documents",
        data={"args": {"query": "x"}},
    )

    legacy = legacy_event_from_v3(event)

    assert legacy["type"] == "tool"
    assert legacy["phase"] == "start"
    assert legacy["state"] == "queued"
    assert legacy["name"] == "search_documents"
    assert legacy["args"] == {"query": "x"}


def test_tool_execution_end_maps_to_legacy_tool_end():
    event = make_event(
        "tool_execution_end",
        sequence=1,
        tool_call_id="call-1",
        tool_name="search_documents",
        data={"output": "result", "render": {"type": "text", "text": "result"}},
    )

    legacy = legacy_event_from_v3(event)

    assert legacy["type"] == "tool"
    assert legacy["phase"] == "end"
    assert legacy["state"] == "completed"
    assert legacy["result"] == "result"
    assert legacy["render"]["type"] == "text"


def test_title_update_is_non_terminal():
    event = make_event(
        "title_updated",
        sequence=1,
        data={"title": "New title", "conversation_id": "conv-1"},
    )

    legacy = legacy_event_from_v3(event)

    assert legacy == {
        "type": "title_updated",
        "title": "New title",
        "conversation_id": "conv-1",
    }


def test_state_snapshot_preserves_legacy_node_complete_for_streamlit():
    event = make_event(
        "state_snapshot",
        sequence=1,
        node="planning_agent",
        data={
            "legacy_type": "node_complete",
            "node": "planning_agent",
            "tool_calls": [{"name": "dispatch_subagents", "id": "call-1", "args": {}}],
        },
    )

    legacy = legacy_event_from_v3(event)

    assert legacy == {
        "type": "node_complete",
        "node": "planning_agent",
        "tool_calls": [{"name": "dispatch_subagents", "id": "call-1", "args": {}}],
    }


def test_state_snapshot_preserves_legacy_continuation_marker():
    event = make_event(
        "state_snapshot",
        sequence=1,
        data={
            "legacy_type": "continuation_start",
            "round": 2,
            "max_rounds": 4,
            "reason": "recursion_limit",
        },
    )

    legacy = legacy_event_from_v3(event)

    assert legacy == {
        "type": "continuation_start",
        "round": 2,
        "max_rounds": 4,
        "reason": "recursion_limit",
    }
```

- [ ] **Step 2: Run the internal SSE tests and verify they fail**

Run: `python -m pytest tests/test_internal_sse_stream_contract.py -q`

Expected: FAIL with an import error for `app.services.event_streaming.internal_sse`.

- [ ] **Step 3: Implement legacy adapter mapping**

Create `app/services/event_streaming/internal_sse.py`:

```python
from __future__ import annotations

from typing import Any

from .events import V3StreamEvent


def legacy_event_from_v3(event: V3StreamEvent) -> dict[str, Any] | None:
    if event.type == "message_delta":
        return {"type": "token", "content": event.data.get("text", "")}
    if event.type == "reasoning_delta":
        return {"type": "thinking", "content": event.data.get("text", "")}
    if event.type == "tool_call_available":
        return {
            "type": "tool",
            "phase": "start",
            "status": "start",
            "state": "queued",
            "name": event.tool_name or "unknown",
            "tool_call_id": event.tool_call_id,
            "args": event.data.get("args"),
        }
    if event.type == "tool_execution_end":
        error = event.data.get("error")
        return {
            "type": "tool",
            "phase": "end",
            "status": "end",
            "state": "error" if error else "completed",
            "name": event.tool_name or "unknown",
            "tool_call_id": event.tool_call_id,
            "result": event.data.get("output"),
            "render": event.data.get("render"),
        }
    if event.type == "rich_items":
        return {
            "type": "rich_items",
            "operation": event.data.get("operation", "upsert"),
            "items": list(event.data.get("items") or []),
        }
    if event.type == "agent_selected":
        return {
            "type": "agent_selected",
            "agent": event.agent or event.data.get("agent"),
            "reason": event.data.get("reason"),
        }
    if event.type == "interrupt":
        payload = dict(event.data)
        payload["type"] = "interrupt"
        return payload
    if event.type == "complete":
        payload = dict(event.data)
        payload["type"] = "complete"
        return payload
    if event.type == "state_snapshot":
        legacy_type = event.data.get("legacy_type")
        if legacy_type in {"node_complete", "continuation_start"}:
            payload = {
                key: value
                for key, value in event.data.items()
                if key != "legacy_type"
            }
            payload["type"] = legacy_type
            if legacy_type == "node_complete" and "node" not in payload and event.node:
                payload["node"] = event.node
            return payload
        return None
    if event.type == "error":
        return {"type": "error", "error": event.data.get("error") or event.data.get("message")}
    if event.type == "title_updated":
        return {
            "type": "title_updated",
            "title": event.data.get("title"),
            "conversation_id": event.data.get("conversation_id"),
        }
    if event.type == "user_message_created":
        return {"type": "user_message_created", "message": event.data.get("message")}
    if event.type == "heartbeat":
        return {"type": "heartbeat"}
    return None
```

- [ ] **Step 4: Preserve title updates before terminal completion**

In `app/services/message_service.py`, await `title_task` before yielding the terminal `complete` event:

```python
title_event = None
if title_task:
    generated_title = await title_task
    if generated_title:
        title_event = {
            "type": "title_updated",
            "title": generated_title,
            "conversation_id": str(message_create_data.conversation_id),
        }

if title_event:
    yield title_event

yield {
    "type": "complete",
    "message": bot_message.model_dump(mode="json"),
}
```

Remove the old post-complete `title_updated` yield.

- [ ] **Step 5: Keep `/messages/stream` and `/messages/resume-interrupt` unchanged**

In `app/api/messages.py`, keep the same route paths and use the same terminal set: `complete`, `error`, `interrupt`. Do not add `[DONE]` to internal SSE because the existing Streamlit client expects JSON-only `data:` events.

Project canonical events at the route boundary:

```python
from app.services.event_streaming.events import V3StreamEvent
from app.services.event_streaming.internal_sse import legacy_event_from_v3


def _to_internal_sse_event(event: dict | V3StreamEvent) -> dict | None:
    if isinstance(event, V3StreamEvent):
        return legacy_event_from_v3(event)
    return event
```

Use it immediately before writing the SSE line:

```python
public_event = _to_internal_sse_event(event)
if public_event is None:
    continue

event_type = public_event.get("type")
yield f"data: {json.dumps(public_event)}\n\n"

if event_type in ("complete", "error", "interrupt"):
    break
```

- [ ] **Step 6: Run Streamlit and sidecar tests**

Run:

```powershell
python -m pytest tests/test_internal_sse_stream_contract.py tests/test_demo_stream_rendering.py tests/test_demo_subagent_activity.py tests/client_backend/test_messages.py tests/client_backend/test_server_api.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add app/services/event_streaming/internal_sse.py app/api/messages.py app/services/message_service.py tests/test_internal_sse_stream_contract.py
git commit -m "feat: preserve streamlit stream contract through v3 adapter"
```

### Task 7: Move Service Layer to Canonical Events

**Files:**
- Create: `app/services/event_streaming/compat.py`
- Create: `tests/test_message_service_event_streaming.py`
- Modify: `app/services/ai_service.py`
- Modify: `app/services/message_service.py`
- Modify: `app/interfaces/message_service_interface.py`

- [ ] **Step 1: Write service forwarding tests**

Create `tests/test_message_service_event_streaming.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.models.enums import MessageRole
from app.schemas.message import MessageCreate, MessageRead
from app.schemas.workflow import (
    WorkflowExecutionRequest,
    WorkflowPlanningContext,
    WorkflowResponse,
    WorkflowResponseMessage,
)
from app.services.event_streaming.events import make_event
from app.services.message_service import MessageService


def _message_row(*, conversation_id, sender: int, content: str) -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=uuid4(),
        conversation_id=conversation_id,
        sender=sender,
        content=content,
        message_metadata={},
        feedback=None,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


@pytest.mark.asyncio
async def test_message_service_accumulates_v3_text_and_persists_once():
    conversation_id = uuid4()
    user_id = uuid4()
    persisted = []

    service = MessageService.__new__(MessageService)
    service.repository = SimpleNamespace(
        create=lambda entity: _message_row(
            conversation_id=conversation_id,
            sender=MessageRole.user.value,
            content=entity.content,
        )
    )
    service.conversation_validation_utils = SimpleNamespace(
        validate_conversation_access=lambda *_args: None,
        conversation_repository=SimpleNamespace(
            get_by_id=lambda _conversation_id: SimpleNamespace(title="Existing chat")
        ),
    )
    workflow_request = WorkflowExecutionRequest(
        message="hello",
        conversation_id=str(conversation_id),
        user_id=str(user_id),
        planning=WorkflowPlanningContext(),
    )
    service._build_user_message_workflow_request = AsyncMock(
        return_value=(user_id, None, workflow_request)
    )

    async def source(_request):
        yield make_event("message_delta", sequence=1, data={"text": "hello"})
        yield make_event(
            "complete",
            sequence=2,
            data={
                "response": WorkflowResponse(
                    message=WorkflowResponseMessage(content="hello"),
                    metadata={},
                )
            },
        )

    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_args: None,
        execute_request_stream=source,
    )
    service._persist_completed_workflow_response = AsyncMock(
        side_effect=lambda **kwargs: persisted.append(kwargs) or MessageRead.model_validate(
            _message_row(
                conversation_id=conversation_id,
                sender=MessageRole.assistant.value,
                content="hello",
            )
        )
    )
    service._compact_checkpoint_after_persist = AsyncMock()

    events = [
        event
        async for event in service.create_message_stream(
            MessageCreate(conversation_id=conversation_id, content="hello"),
            user_id,
        )
    ]

    assert all(isinstance(event, V3StreamEvent) for event in events)
    assert any(event.type == "message_delta" and event.data["text"] == "hello" for event in events)
    assert events[-1].type == "complete"
    assert events[-1].data["message"]["content"] == "hello"
    assert len(persisted) == 1


@pytest.mark.asyncio
async def test_resume_stream_accepts_v3_events_and_persists_once():
    conversation_id = uuid4()
    user_id = uuid4()
    bot_message_id = uuid4()
    persisted = []

    service = MessageService.__new__(MessageService)
    service.hitl_interrupt_repository = None
    service._validate_and_claim_interrupt_resume = lambda **_kwargs: None
    service._get_conversation_context = lambda *_args: (user_id, None)
    service._revalidate_resume_custom_agent = lambda *_args: None
    service._audit_interrupt_resume_decisions = lambda **_kwargs: None
    service._resolve_custom_agents_state = lambda *_args: {}
    service._clear_redis_interrupt = lambda *_args: None
    service._sync_response_plan_state = lambda **_kwargs: False

    async def resume_source(**_kwargs):
        yield make_event("message_delta", sequence=1, data={"text": "resumed"})
        yield make_event(
            "complete",
            sequence=2,
            data={
                "response": WorkflowResponse(
                    message=WorkflowResponseMessage(content="resumed"),
                    metadata={},
                )
            },
        )

    service.ai_service = SimpleNamespace(
        invalidate_history_cache=lambda *_args: None,
        resume_interrupted_execution_stream=resume_source,
    )
    service._persist_completed_workflow_response = AsyncMock(
        side_effect=lambda **kwargs: persisted.append(kwargs) or MessageRead.model_validate(
            _message_row(
                conversation_id=conversation_id,
                sender=MessageRole.assistant.value,
                content="resumed",
            )
        )
    )
    service._compact_checkpoint_after_persist = AsyncMock()

    events = [
        event
        async for event in service.resume_message_creation_stream(
            thread_id=str(conversation_id),
            conversation_id=conversation_id,
            user_id=user_id,
            decisions=[],
            bot_message_id=bot_message_id,
        )
    ]

    assert all(isinstance(event, V3StreamEvent) for event in events)
    assert any(event.type == "message_delta" and event.data["text"] == "resumed" for event in events)
    assert events[-1].type == "complete"
    assert events[-1].data["message"]["content"] == "resumed"
    assert len(persisted) == 1
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_message_service_event_streaming.py -q`

Expected: FAIL because service code still consumes custom dict events directly.

- [ ] **Step 3: Create the temporary legacy-to-v3 compatibility bridge**

Create `app/services/event_streaming/compat.py`. This file is temporary and must be deleted in Task 9 after all producers emit `V3StreamEvent` directly:

```python
from __future__ import annotations

from typing import Any, cast

from app.services.event_streaming.events import (
    StreamEventType,
    V3StreamEvent,
    make_event,
)


_DIRECT_EVENT_TYPES: set[str] = {
    "agent_selected",
    "rich_items",
    "interrupt",
    "complete",
    "error",
    "heartbeat",
    "user_message_created",
    "title_updated",
}


def coerce_legacy_event_to_v3(
    event: dict[str, Any] | V3StreamEvent,
    *,
    sequence: int,
) -> V3StreamEvent:
    if isinstance(event, V3StreamEvent):
        return event

    event_type = event.get("type")
    if event_type == "token":
        return make_event("message_delta", sequence=sequence, data={"text": event.get("content", "")})
    if event_type == "thinking":
        return make_event("reasoning_delta", sequence=sequence, data={"text": event.get("content", "")})
    if event_type in {"tool", "tool_start"}:
        phase = event.get("phase")
        if event_type == "tool_start" or phase == "start":
            return make_event(
                "tool_call_available",
                sequence=sequence,
                tool_call_id=event.get("tool_call_id"),
                tool_name=event.get("name"),
                data={"args": event.get("args")},
            )
        return make_event(
            "tool_execution_end",
            sequence=sequence,
            tool_call_id=event.get("tool_call_id"),
            tool_name=event.get("name"),
            data={
                "output": event.get("result"),
                "render": event.get("render"),
                "error": event.get("error"),
            },
        )
    if event_type == "tool_end":
        return make_event(
            "tool_execution_end",
            sequence=sequence,
            tool_call_id=event.get("tool_call_id"),
            tool_name=event.get("name"),
            data={
                "output": event.get("result"),
                "render": event.get("render"),
                "error": event.get("error"),
            },
        )
    if event_type in {"node_complete", "continuation_start"}:
        payload = dict(event)
        payload["legacy_type"] = event_type
        return make_event(
            "state_snapshot",
            sequence=sequence,
            node=event.get("node"),
            data=payload,
        )
    if event_type in _DIRECT_EVENT_TYPES:
        return make_event(cast(StreamEventType, event_type), sequence=sequence, data=dict(event))
    return make_event("state_snapshot", sequence=sequence, data={"raw": dict(event)})
```

- [ ] **Step 4: Normalize event input at AI service boundary**

In `app/services/ai_service.py`, convert graph output to `V3StreamEvent` as early as possible and leave capability filtering here:

```python
from app.services.event_streaming.events import V3StreamEvent, make_event
from app.services.event_streaming.compat import coerce_legacy_event_to_v3


sequence = 0
async for raw_event in workflow_stream:
    sequence += 1
    event = coerce_legacy_event_to_v3(raw_event, sequence=sequence)
    yield event
```

Keep this coercion as a temporary bridge until `app/ai/graph.py` emits only `V3StreamEvent`.

- [ ] **Step 5: Convert `MessageService` to emit canonical events**

In `app/services/message_service.py`, keep persistence, cancellation, partial-text accumulation, interrupt persistence, and title generation in the service, but yield `V3StreamEvent` objects instead of public legacy dicts. Public wire compatibility belongs in `app/api/messages.py` and `app/api/ai_sdk.py`, not in `MessageService`.

Add a service-local coercion helper during migration:

```python
from app.services.event_streaming.events import V3StreamEvent
from app.services.event_streaming.compat import coerce_legacy_event_to_v3


def _service_event_from_ai_event(
    event: dict | V3StreamEvent,
    *,
    sequence: int,
) -> V3StreamEvent:
    if isinstance(event, V3StreamEvent):
        return event
    return coerce_legacy_event_to_v3(event, sequence=sequence)
```

At the start of each `create_message_stream` and `resume_message_creation_stream` AI loop:

```python
sequence += 1
event = _service_event_from_ai_event(raw_event, sequence=sequence)
```

When the service creates its own events, emit canonical events directly:

```python
sequence += 1
yield make_event(
    "user_message_created",
    sequence=sequence,
    conversation_id=str(message_create_data.conversation_id),
    message_id=str(created_message.id),
    data={"message": MessageRead.model_validate(created_message).model_dump(mode="json")},
)

sequence += 1
yield make_event(
    "message_delta",
    sequence=sequence,
    conversation_id=str(message_create_data.conversation_id),
    message_id=str(bot_message_id),
    data={"text": token_content},
)

sequence += 1
yield make_event(
    "complete",
    sequence=sequence,
    conversation_id=str(message_create_data.conversation_id),
    message_id=str(bot_message.id),
    data={"message": bot_message.model_dump(mode="json")},
)
```

For Streamlit compatibility, `app/api/messages.py` must call `legacy_event_from_v3(event)` immediately before serializing `data: ...`. For assistant-ui compatibility, `app/api/ai_sdk.py` passes canonical events to `AISDKV6StreamAdapter`.

- [ ] **Step 6: Run service tests**

Run:

```powershell
python -m pytest tests/test_message_service_event_streaming.py tests/test_rich_response_streaming.py tests/test_message_service_subagent_streaming.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add app/services/event_streaming/compat.py app/services/ai_service.py app/services/message_service.py app/interfaces/message_service_interface.py tests/test_message_service_event_streaming.py
git commit -m "feat: move message streaming through v3 event layer"
```

### Task 8: Preserve Assistant-UI Endpoint Behavior

**Files:**
- Create: `tests/test_ai_sdk_assistant_ui_compat.py`
- Modify: `app/api/ai_sdk.py`
- Modify: `README.md`

- [ ] **Step 1: Write endpoint compatibility tests**

Create `tests/test_ai_sdk_assistant_ui_compat.py`:

```python
from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.ai_sdk import AISDKChatRequest, chat_ui_message_stream
from app.services.event_streaming.events import make_event


@pytest.mark.asyncio
async def test_chat_endpoint_returns_assistant_ui_stream(monkeypatch):
    conversation_id = uuid4()
    user_id = uuid4()

    class FakeMessageService:
        def create_message_stream(self, *_args, **_kwargs):
            async def source():
                yield make_event("message_delta", sequence=1, data={"text": "hello"})
                yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

            return source()

    response = await chat_ui_message_stream(
        conversation_id=conversation_id,
        payload=AISDKChatRequest(
            messages=[{"role": "user", "parts": [{"type": "text", "text": "hello"}]}]
        ),
        message_service=FakeMessageService(),
        current_user_id=user_id,
    )

    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
    body = "".join([chunk async for chunk in response.body_iterator])
    payloads = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]

    assert payloads[0]["type"] == "start"
    assert any(payload.get("type") == "text-delta" for payload in payloads)
    assert payloads[-1]["type"] == "finish"
    assert body.rstrip().endswith("data: [DONE]")
```

- [ ] **Step 2: Run the endpoint compatibility test**

Run: `python -m pytest tests/test_ai_sdk_assistant_ui_compat.py -q`

Expected: PASS after Task 5.

- [ ] **Step 3: Verify endpoint paths stay unchanged**

Run:

```powershell
rg -n "\"/api/chat/\\{conversation_id\\}\"|\"/ai/chat/\\{conversation_id\\}\"|\"/ai/resume-interrupt\"" app/api/ai_sdk.py
```

Expected output includes all three route paths.

- [ ] **Step 4: Update README streaming docs**

In `README.md`, replace the old AI SDK event vocabulary with:

```markdown
| API namespace | Endpoint | Stream protocol | Primary events |
| --- | --- | --- | --- |
| assistant-ui / AI SDK v6 | `POST /api/chat/{conversation_id}` and `POST /ai/chat/{conversation_id}` | Vercel AI SDK UI Message Stream over SSE | `start`, `start-step`, `text-start`, `text-delta`, `reasoning-start`, `reasoning-delta`, `tool-input-start`, `tool-input-available`, `tool-output-available`, `data-interrupt`, `data-rich-items`, `finish-step`, `finish`, `[DONE]` |
| Streamlit internal client | `POST /messages/stream` and `POST /messages/resume-interrupt` | Backend JSON SSE compatibility stream | `user_message_created`, `agent_selected`, `token`, `thinking`, `tool`, `rich_items`, `interrupt`, `title_updated`, `complete`, `error`, `heartbeat` |
| Backend internal | service layer | Canonical v3 event model | `message_delta`, `reasoning_delta`, `tool_call_available`, `tool_execution_end`, `subagent_start`, `subagent_end`, `interrupt`, `complete`, `error` |
```

- [ ] **Step 5: Run AI SDK and docs-adjacent tests**

Run:

```powershell
python -m pytest tests/test_ai_sdk_assistant_ui_compat.py tests/test_ai_sdk_v6_stream_contract.py tests/test_rich_response_streaming.py tests/client_backend/test_sse_keepalive.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add app/api/ai_sdk.py README.md tests/test_ai_sdk_assistant_ui_compat.py
git commit -m "test: preserve assistant ui ai sdk stream behavior"
```

### Task 9: Clean Up Legacy and Deprecated Streaming Code

**Files:**
- Modify: `app/ai/graph.py`
- Modify: `app/ai/agents/search_agent.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `app/api/ai_sdk.py`
- Modify: `app/services/stream_events.py`
- Modify: `app/services/ai_service.py`
- Modify: `app/services/message_service.py`
- Modify: `README.md`

- [ ] **Step 1: Find remaining legacy references**

Run:

```powershell
rg -n "CANONICAL_STREAM_EVENT_TYPES|tool_start|tool_end|node_complete|continuation_start|stream_mode=\\[\"messages\", \"updates\"\\]|title_updated|StreamState|EventHandlerFactory|normalize_tool_phase|build_canonical_tool_event" app tests README.md demo.py client_backend
```

Expected:

- No legacy emitters remain in `app/ai/graph.py`, `app/ai/agents/search_agent.py`, `app/ai/agents/rag_agent.py`, `app/services/ai_service.py`, or `app/services/message_service.py`.
- References may remain in Streamlit rendering code, client-backend compatibility tests, migration tests, and the new `event_streaming/internal_sse.py` compatibility adapter.
- `title_updated` remains only in `message_service.py` before terminal `complete`, `demo.py` rendering, and tests that assert the ordering.

- [ ] **Step 2: Remove duplicate graph stream parsing**

In `app/ai/graph.py`, delete duplicated code blocks that parse `content_blocks` and tool messages inside both `resume_with_decisions_stream` and `execute_request_stream`. Keep one call path through `iter_v3_events_from_graph`.

- [ ] **Step 3: Remove direct legacy agent stream emitters**

Inspect whether `app/ai/agents/search_agent.py::stream_message` and `app/ai/agents/rag_agent.py::stream_message` are still reachable. If they are used, route their outputs through `_coerce_v3_event` and then through the public adapter at the call site. If they are not reachable, delete the methods and update or remove tests that only covered those stale helpers.

After this step, this command must return no direct legacy emitters in agent code:

```powershell
rg -n "\"tool_start\"|\"tool_end\"" app/ai/agents app/ai/graph.py
```

Expected: no matches.

- [ ] **Step 4: Delete stale canonical constants**

If `rg -n "CANONICAL_STREAM_EVENT_TYPES" app tests` returns only `app/services/stream_events.py`, delete `CANONICAL_STREAM_EVENT_TYPES`. If `app/services/stream_events.py` has no imports left, delete the file and update imports to `app/services/event_streaming/internal_sse.py` or `app/services/event_streaming/events.py`.

- [ ] **Step 5: Remove AI SDK handler duplication**

After `app/services/event_streaming/ai_sdk_v6.py` owns the handlers, remove these classes from `app/api/ai_sdk.py`:

```text
EventHandler
TokenEventHandler
ThinkingEventHandler
AgentSelectedEventHandler
UserMessageCreatedEventHandler
ToolEventHandler
InterruptEventHandler
ErrorEventHandler
CompleteEventHandler
ContinuationEventHandler
NodeCompleteEventHandler
HeartbeatEventHandler
RichItemsEventHandler
EventHandlerFactory
```

Keep request/response models and route functions in `app/api/ai_sdk.py`.

- [ ] **Step 6: Remove temporary dict-to-v3 adapter bridges**

After `MessageService` emits only `V3StreamEvent`, remove temporary legacy dict coercion from `AISDKV6StreamAdapter` and any service-local bridge that exists only to handle pre-migration dict events. Keep `legacy_event_from_v3` because Streamlit still intentionally consumes legacy JSON SSE.

Run:

```powershell
rg -n "_coerce_event|_coerce_v3_event|dict\\[str, Any\\] \\| V3StreamEvent|event\\.get\\(\"type\"\\)" app/services/event_streaming app/services/message_service.py app/api/ai_sdk.py
```

Expected: no matches in `app/services/event_streaming/ai_sdk_v6.py` or `app/services/message_service.py` for migration-only dict coercion. Matches may remain in `internal_sse.py` tests or route code that intentionally handles public JSON dicts.

- [ ] **Step 7: Verify no unreachable title update remains**

Run:

```powershell
rg -n "title_updated" app/services/message_service.py app/api/messages.py demo.py
```

Expected:

- `app/services/message_service.py` emits `title_updated` before `complete`.
- `app/api/messages.py` still breaks after `complete`, `error`, and `interrupt`.
- `demo.py` keeps its existing `title_updated` handling.

- [ ] **Step 8: Run cleanup verification**

Run:

```powershell
python -m pytest tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py tests/test_event_streaming_langgraph_normalizer.py tests/test_event_streaming_subagents.py tests/test_rich_response_streaming.py tests/client_backend/test_sse_keepalive.py tests/test_graph_streaming_tool_events.py tests/test_message_service_subagent_streaming.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```powershell
git add app README.md tests
git commit -m "refactor: remove legacy streaming event code"
```

### Task 10: Full Regression Verification

**Files:**
- No code changes.

- [ ] **Step 1: Run focused streaming suite**

Run:

```powershell
python -m pytest tests/test_ai_sdk_v6_stream_contract.py tests/test_ai_sdk_assistant_ui_compat.py tests/test_internal_sse_stream_contract.py tests/test_event_streaming_events.py tests/test_event_streaming_langgraph_normalizer.py tests/test_event_streaming_subagents.py -q
```

Expected: PASS.

- [ ] **Step 2: Run current streaming regression suite**

Run:

```powershell
python -m pytest tests/test_graph_streaming_tool_events.py tests/test_message_service_subagent_streaming.py tests/test_rich_response_streaming.py tests/client_backend/test_sse_keepalive.py -q
```

Expected: PASS.

- [ ] **Step 3: Run Streamlit compatibility suite**

Run:

```powershell
python -m pytest tests/test_demo_stream_rendering.py tests/test_demo_subagent_activity.py tests/test_demo_rich_response.py tests/client_backend/test_messages.py tests/client_backend/test_server_api.py -q
```

Expected: PASS.

- [ ] **Step 4: Run all tests**

Run: `python -m pytest -q`

Expected: PASS.

- [ ] **Step 5: Manual SSE smoke test for AI SDK endpoint**

Start the backend using the repo's normal command, then send a real request to `/api/chat/{conversation_id}` from a development account. Confirm the response starts with:

```text
data: {"type":"start"
data: {"type":"start-step"}
data: {"type":"text-start"
```

Confirm the response ends with:

```text
data: {"type":"finish-step"}
data: {"type":"finish"}
data: [DONE]
```

- [ ] **Step 6: Manual SSE smoke test for AI SDK resume endpoint**

Create or reuse a conversation with a pending HITL interrupt, then send a real approval or rejection request to `/ai/resume-interrupt`. Confirm the response starts with AI SDK UI Message Stream chunks:

```text
data: {"type":"start"
data: {"type":"start-step"}
data: {"type":"text-start"
```

Confirm the response ends with:

```text
data: {"type":"finish-step"}
data: {"type":"finish"}
data: [DONE]
```

Confirm no internal legacy JSON event such as `{"type":"token"}` or `{"type":"tool"}` appears on this AI SDK endpoint.

- [ ] **Step 7: Manual SSE smoke test for Streamlit endpoint**

Send a real request to `/messages/stream`. Confirm JSON `data:` events include `user_message_created`, then `token` or `thinking`, and end with exactly one terminal event: `complete`, `interrupt`, or `error`.

- [ ] **Step 8: Manual SSE smoke test for Streamlit resume endpoint**

Create or reuse a pending HITL interrupt, then send a real approval or rejection request to `/messages/resume-interrupt`. Confirm JSON `data:` events include `token` or `thinking`, optional `tool` / `rich_items`, and end with exactly one terminal event: `complete`, `interrupt`, or `error`. Confirm this endpoint does not emit `data: [DONE]`.

- [ ] **Step 9: Commit verification notes**

```powershell
git status --short
git commit --allow-empty -m "test: verify event streaming refactor"
```

## Production Risks and Controls

- Tool-call semantics: never map model tool-call availability to real execution start internally. This prevents HITL approval UI from showing an unapproved tool as already running.
- Assistant-ui compatibility: keep AI SDK v6 chunks and the `x-vercel-ai-ui-message-stream: v1` header unchanged. Add frontend capability flags before exposing transient subagent state to assistant-ui.
- Streamlit compatibility: keep JSON-only `data:` events and no `[DONE]` sentinel on `/messages/stream`.
- Subagent stream ordering: emit per-worker lifecycle events as they happen, but keep the final `dispatch_subagents` JSON result ordered by input task order for the Planning Agent.
- Persistence safety: persist the assistant message only on `complete`, persist the interrupt message only on `interrupt`, and persist partial text only on cancellation or disconnect.
- Resume-path parity: every first-pass stream change must have matching coverage for `resume_message_creation_stream`, `/messages/resume-interrupt`, and `/ai/resume-interrupt`.
- Adapter isolation: AI SDK and Streamlit formatting code must live in adapters. Graph and service layers emit `V3StreamEvent` only after Task 7.
- Dependency safety: do not remove the fallback tuple normalizer until the v3 dependency probe, graph integration tests, and exact `environment.yml` resolver pins pass in CI.
- Observability safety: every terminal `complete`, `interrupt`, and `error` event should include enough metadata to correlate `conversation_id`, `message_id`, `thread_id`, and `run_id` when available. Do not log tool outputs, uploaded file contents, or raw image data in stream diagnostics.

## Acceptance Criteria

- AI SDK v6 endpoints keep working with assistant-ui: `/api/chat/{conversation_id}`, `/ai/chat/{conversation_id}`, and `/ai/resume-interrupt`.
- Streamlit endpoints remain available and compatible: `/messages/stream` and `/messages/resume-interrupt`.
- Internal stream events carry `schema_version="v3"`.
- Tool events distinguish `tool_call_available` from `tool_execution_start` and `tool_execution_end`.
- Custom `dispatch_subagents` emits `subagent_start`, subagent tool events, and `subagent_end` with `name`, `path`, and `status` fields.
- `title_updated` is reachable by Streamlit before terminal `complete`, or included in terminal completion metadata if product chooses that presentation later.
- Legacy duplicated parsing, direct `tool_start` / `tool_end` emitters, and stale event constants are removed after tests pass.
- First-pass and resume streams both consume canonical v3 service events before projecting to public protocols.
- Dependency probe passes in the exact active Python environment, and `environment.yml` exact pins match the verified resolver set.
- Focused streaming tests and current regression tests pass.

## Execution Order

1. Dependency probe and constraints.
2. Canonical v3 event model.
3. LangGraph normalizer.
4. Custom subagent event sink.
5. AI SDK v6 adapter.
6. Streamlit internal SSE adapter.
7. Service layer migration.
8. Endpoint compatibility and README update.
9. Legacy cleanup.
10. Full regression verification.
