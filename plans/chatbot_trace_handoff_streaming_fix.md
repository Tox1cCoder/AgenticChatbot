# Chatbot Trace, Handoff, And Streaming Render Fix Plan

Branch: current working tree | Date: 2026-05-21 | Spec: this document | Input: fix three reported chatbot issues around LangSmith checkpoint cleanup traces, Planning Agent handoff behavior, and Streamlit live markdown rendering.

## Summary

Fix three user-visible reliability issues without changing the overall multi-agent architecture:

1. Keep LangGraph checkpoint compaction, but suppress or replace the traced cleanup update that appears in LangSmith as `LangGraphUpdateState` with `remove - No data`.
2. Make `hand_off` a control-plane transfer that routes to the delegated agent and returns that agent's answer in the same turn.
3. Normalize live Streamlit token rendering so HTML entities like `&quot;` do not leak into markdown text.

## Technical Context

Language/Version: Python 3.10+.

Primary Dependencies: FastAPI, Streamlit, LangGraph 1.x, LangChain, LangSmith 0.6.x, pytest.

Storage: PostgreSQL conversation/message tables remain canonical. LangGraph checkpoints are transient execution state only.

Testing: `pytest` focused backend/unit tests plus optional Streamlit smoke verification.

Target Platform: Existing backend service under `app/`, sidecar/client under `client_backend/`, and Streamlit UI in `demo.py`.

## Constitution Check

No `.specify/memory/constitution.md` exists in this repository. Apply current repo constraints:

- Keep DB as canonical transcript source.
- Do not reintroduce checkpoint transcript as prompt history.
- Do not replace the router or LangGraph topology wholesale.
- Preserve HITL, tool artifacts, subagent activity, and cancellation behavior.
- Add focused regression tests before implementation.

## Current Findings

### Finding 1: LangSmith `remove - No data` Is Injected By Code

Confirmed source:

- `app/ai/graph.py::_compact_checkpoint_after_terminal_response`
- `app/services/message_service.py::_compact_checkpoint_after_persist`

Current flow:

1. `MessageService.create_message_stream()` persists the assistant response.
2. `_compact_checkpoint_after_persist()` calls `AIService.compact_checkpoint_after_terminal_response()`.
3. `MultiAgentWorkflow._compact_checkpoint_after_terminal_response()` reads the final LangGraph state.
4. It creates one `RemoveMessage(id=...)` per checkpoint message.
5. It calls `self.graph.aupdate_state(config, {"messages": removals})`.
6. LangSmith traces that manual LangGraph state update as `LangGraphUpdateState`, with hidden/empty remove payloads displayed as `remove - No data`.

This cleanup is intentional checkpoint compaction, not model output. The noisy trace is an observability side effect of doing cleanup through the traced LangGraph update API.

### Finding 1A: Aim And Redundancy Assessment For `LangGraphUpdateState`

`LangGraphUpdateState` is not an application node or agent step. It is LangGraph's/LangSmith's trace label for the manual `graph.aupdate_state(...)` call used by checkpoint compaction.

The specific `remove - No data` rows come from `langchain_core.messages.RemoveMessage`. `RemoveMessage` intentionally has message type `remove` and empty content; LangGraph's messages reducer interprets it as "delete the message with this id." LangSmith therefore has no useful payload to show.

Aim:

- Treat PostgreSQL message rows as the canonical transcript after the assistant response has been durably persisted.
- Clear the transient LangGraph checkpoint `messages` channel for terminal turns.
- Prevent old human, assistant, AI tool-call, and tool-result messages from accumulating in a reused graph `thread_id`.
- Prevent stale tool-call state from leaking into the next turn's prompt, stream, router state, or trace.

Redundancy assessment:

- The cleanup operation is not redundant in the current architecture because the same conversation-level LangGraph `thread_id` is reused across turns and the `messages` channel uses message-merge semantics.
- The checkpoint copy of persisted conversation history is redundant after terminal persistence; that is exactly why compaction removes it.
- The visible LangSmith trace entry is redundant/noisy because it records an internal cleanup detail after the user-facing turn is complete.
- The cleanup is already designed to be a no-op when there is no checkpointer, no thread id, no removable message ids, or the graph is interrupted with pending `snapshot.next` nodes.

Implementation implication: do not remove `_compact_checkpoint_after_terminal_response()` as the first fix. Keep the cleanup semantics and only suppress or replace its LangSmith visibility. Only consider removing compaction if a separate A/B test proves that two or more tool-using turns leave `aget_state(config).values["messages"]` empty without this cleanup.

### Finding 2: Handoff Routing Exists But Handoff Leaks As User-Facing Text

Relevant files:

- `app/ai/hand_off_tool.py`
- `app/ai/graph.py::_apply_hand_off_if_present`
- `app/ai/graph.py::_should_continue_planning`
- `app/ai/graph.py::execute_request_stream`
- `app/ai/agents/planning_agent.py`

Current architecture already has graph edges for handoff:

- `planning_agent -> planning_tools`
- `planning_tools -> delegated agent` when `selected_agent` changes
- `tools -> delegated agent` for non-planning handoff

Likely failure mode:

- The Planning Agent emits natural-language handoff text such as `Transfering your request to Search Agent...` while also calling `hand_off`.
- Streaming sends those Planning Agent tokens immediately.
- The delegated agent receives a current-turn message list polluted by the Planning Agent's handoff AIMessage and handoff ToolMessage.
- The UI keeps rendering accumulated intermediate text and does not clearly replace it with the delegated agent's final answer until reload.

So the handoff tool is not fundamentally conflicting with the router architecture. The bug is that handoff is treated partly as content and partly as routing. It should be treated as control-plane state.

### Finding 3: Streamlit Live Markdown Skips Existing Normalization

Relevant files:

- `demo.py::sanitize_message_content`
- `demo.py::render_message_bubble`
- `demo.py::render_chat_view`
- `demo.py::_submit_interrupt_decisions`

Persisted message rendering currently uses `st.markdown(content_text)`.

Live streaming rendering directly does:

```python
response_placeholder.markdown(accumulated_content)
```

This bypasses any entity normalization. If streamed content contains escaped quotes, Streamlit can display text like:

```text
model: &quot;gemini-3.1-pro&quot;, provider: &quot;gemini&quot;:
```

The existing `sanitize_message_content()` unescapes entities but returns sanitized HTML and is not suitable as-is for every live token update because live rendering currently relies on native markdown/LaTeX behavior.

## Functional Requirements

FR-001: LangSmith traces should not show a noisy post-turn `LangGraphUpdateState` cleanup run for checkpoint compaction by default.

FR-002: Checkpoint compaction must still remove transient checkpoint messages after terminal, persisted assistant responses.

FR-003: Checkpoint compaction must remain a no-op when the graph is interrupted or has pending `next` nodes.

FR-004: Checkpoint compaction must be treated as internal state maintenance, not as a user-facing agent, node, or model response.

FR-005: The implementation must not remove `_compact_checkpoint_after_terminal_response()` unless an explicit regression test proves checkpoint messages do not accumulate across multiple tool-using turns without it.

FR-006: `hand_off` must route to the requested valid target agent in the same LangGraph invocation.

FR-007: A Planning Agent handoff to `search_agent` must produce the Search Agent's user-facing answer in the same chat turn, not stop after a transfer notice.

FR-008: Handoff narration from the source agent must not be persisted or rendered as the assistant's final answer.

FR-009: The delegated agent should receive the user's original request cleanly, without source-agent handoff AIMessage/ToolMessage noise as conversational context.

FR-010: Streaming should emit an agent-change event when handoff changes `selected_agent`, so UI status reflects the delegated agent.

FR-011: Live Streamlit markdown should render common escaped quote entities as quotes in the response body.

FR-012: Entity normalization must not enable unsafe HTML execution in streamed content.

FR-013: Existing final message rendering, trace rendering, subagent activity, and RAG artifact rendering must remain compatible.

## Non-Goals

- Do not remove checkpoint compaction entirely.
- Do not treat `LangGraphUpdateState` as a model-generated assistant response.
- Do not replace the router with handoff.
- Do not make Planning Agent subagent dispatch use `hand_off`.
- Do not rewrite Streamlit rendering around unsafe HTML for live token content.
- Do not add new database tables.

## File Plan

Modify:

- `app/ai/graph.py`
  - Suppress LangSmith tracing around checkpoint compaction or use a non-traced compaction helper.
  - Add explicit handoff metadata to `state["context"]`.
  - Strip handoff control messages before invoking delegated agents.
  - Emit an `agent_selected` event when `selected_agent` changes during streaming.

- `app/services/message_service.py`
  - Keep compaction call after assistant persistence.
  - Adjust tests only if event ordering changes.

- `demo.py`
  - Add a live markdown normalization helper.
  - Use the helper in both normal message streaming and interrupt resume streaming.
  - Optionally replace live placeholder content with final completed message content before rerun.

Modify tests:

- `tests/test_message_history_pipeline.py`
  - Pin compaction behavior and tracing suppression.

- `tests/test_graph_planning_subagents.py`
  - Add a same-turn Planning Agent handoff integration regression.

- `tests/test_graph_streaming_tool_events.py` or new `tests/test_graph_handoff_streaming.py`
  - Pin streaming events around `agent_selected` after handoff and no final transfer-only answer.

- `tests/test_demo_subagent_activity.py` or new `tests/test_demo_stream_rendering.py`
  - Pin live markdown entity normalization.

## Design

### Design 1: Non-Traced Checkpoint Compaction

Keep `_compact_checkpoint_after_terminal_response()` as the post-persistence cleanup mechanism. Its state mutation is still needed to remove transient checkpoint messages; only the observability surface should change.

Use LangSmith's local tracing context to disable tracing only for checkpoint cleanup:

```python
from langsmith import tracing_context

with tracing_context(enabled=False):
    await self.graph.aupdate_state(config, {"messages": removals})
```

If a manual LangSmith smoke test still shows `LangGraphUpdateState`, replace the `aupdate_state()` compaction path with a lower-level checkpoint rewrite helper in a follow-up task. Keep the first implementation minimal because local `langsmith.tracing_context(enabled=False)` is available in the installed package.

Do not replace this with "do nothing" unless the follow-up investigation captures evidence that checkpoint state does not retain messages across repeated tool-using turns without compaction.

### Design 2: Handoff As Control Plane

When `_apply_hand_off_if_present()` sees a valid `hand_off` tool output:

- Set `state["selected_agent"] = target_agent`.
- Increment `delegation_count`.
- Store metadata in `state["context"]["handoff"]`:

```python
{
    "active": True,
    "source_agent": previous_agent,
    "target_agent": target_agent,
    "reason": reason,
    "tool_call_id": hand_off_output["tool_call_id"],
}
```

Add a helper:

```python
def _messages_for_selected_agent(self, state: GraphState, agent_name: str, messages: list) -> list:
    ...
```

For a delegated target, this helper returns only the last user `HumanMessage` plus any non-handoff tool context that the target legitimately needs. It excludes:

- Source-agent AIMessage whose tool calls include `hand_off`.
- The matching `ToolMessage(name="hand_off")`.

Use this helper in `_chat_node`, `_search_node`, `_image_generator_node`, `_canvas_node`, and Planning handoff paths. Leave RAG's document-specific original-query extraction intact, but make sure `hand_off` control ToolMessages do not become RAG evidence.

### Design 3: Handoff Streaming Event

In both stream loops (`execute_request_stream` and `resume_with_decisions_stream`), track the last emitted selected agent:

```python
last_emitted_agent = selected_agent
...
new_agent = node_state.get("selected_agent")
if isinstance(new_agent, str) and new_agent != last_emitted_agent:
    last_emitted_agent = new_agent
    yield {"type": "agent_selected", "agent": new_agent, "reason": "handoff"}
```

Do this in `updates` mode after `last_state_values.update(node_state)`.

### Design 4: Live Markdown Entity Normalization

Add a narrow helper in `demo.py`:

```python
_STREAM_MARKDOWN_ENTITY_MAP = {
    "&quot;": '"',
    "&#34;": '"',
    "&#x22;": '"',
    "&#39;": "'",
    "&#x27;": "'",
}

def normalize_stream_markdown_text(content: str) -> str:
    if not isinstance(content, str) or not content:
        return ""
    normalized = content
    for entity, replacement in _STREAM_MARKDOWN_ENTITY_MAP.items():
        normalized = normalized.replace(entity, replacement)
    return normalized
```

Use this helper only before native `st.markdown()` calls for live assistant body content:

```python
response_placeholder.markdown(normalize_stream_markdown_text(accumulated_content))
```

Do not call broad `html.unescape()` in the live path, because that can turn escaped angle brackets into actual HTML. Narrow quote unescaping fixes the reported model/provider text without expanding the rendering attack surface.

## Implementation Tasks

### Task 1: Add Failing Tests For Checkpoint Compaction Trace Suppression

Files:

- Modify: `tests/test_message_history_pipeline.py`
- Modify: `app/ai/graph.py`

Steps:

1. Add a test that monkeypatches `app.ai.graph.tracing_context` and asserts `_compact_checkpoint_after_terminal_response()` wraps `aupdate_state()` in `enabled=False`.

```python
@pytest.mark.asyncio
async def test_checkpoint_compaction_disables_langsmith_tracing(monkeypatch):
    from types import SimpleNamespace
    from langchain_core.messages import AIMessage
    from app.ai.graph import MultiAgentWorkflow

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = object()
    calls = []

    class FakeTracingContext:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
        def __enter__(self):
            calls.append(("enter", None))
        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", None))

    async def fake_get_state(_config):
        return SimpleNamespace(next=[], values={"messages": [AIMessage(content="x", id="m1")]})

    async def fake_update_state(_config, _payload):
        calls.append(("update", None))

    workflow.graph = SimpleNamespace(aget_state=fake_get_state, aupdate_state=fake_update_state)
    monkeypatch.setattr("app.ai.graph.tracing_context", FakeTracingContext)

    await workflow._compact_checkpoint_after_terminal_response(
        config={"configurable": {"thread_id": "conv-1"}},
        thread_id="conv-1",
    )

    assert ("init", {"enabled": False}) in calls
    assert calls.index(("enter", None)) < calls.index(("update", None)) < calls.index(("exit", None))
```

2. Run:

```powershell
python -m pytest tests/test_message_history_pipeline.py::test_checkpoint_compaction_disables_langsmith_tracing -q
```

Expected: fails because `tracing_context` is not imported or used yet.

3. Implement by importing `tracing_context` in `app/ai/graph.py` and wrapping only `self.graph.aupdate_state(...)`.

4. Run:

```powershell
python -m pytest tests/test_message_history_pipeline.py -q
```

Expected: pass.

5. Run the existing compaction tests to prove cleanup behavior was preserved:

```powershell
python -m pytest tests/test_message_history_pipeline.py::test_checkpoint_compaction_runs_after_complete_not_after_interrupt tests/test_message_history_pipeline.py::test_message_service_compacts_checkpoint_after_persist -q
```

Expected: pass. These tests are the guardrail that `LangGraphUpdateState` cleanup is still functionally present even if its LangSmith trace is suppressed.

6. Optional diagnostic if someone proposes deleting compaction:

```powershell
python -m pytest tests/test_message_history_pipeline.py -q
```

Then temporarily disable `_compact_checkpoint_after_terminal_response()` locally and run a two-turn tool-using workflow inspection that reads `await workflow.graph.aget_state(config)`. Only consider removal if `snapshot.values["messages"]` remains empty after both turns. Revert the temporary diagnostic change before committing.

### Task 2: Add Handoff Control Metadata And Clean Delegated Messages

Files:

- Modify: `app/ai/graph.py`
- Modify: `tests/test_graph_planning_subagents.py`

Steps:

1. Add a helper-level failing test that proves handoff control messages are removed for the delegated agent input.

```python
def test_delegated_agent_messages_strip_handoff_control_messages():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from app.ai.graph import MultiAgentWorkflow

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    messages = [
        HumanMessage(content="what changed in the latest release?"),
        AIMessage(
            content="Transfering your request to Search Agent...",
            tool_calls=[{"id": "handoff-1", "name": "hand_off", "args": {"target_agent": "search_agent"}}],
        ),
        ToolMessage(
            content='{"hand_off": "search_agent", "reason": "needs current info"}',
            tool_call_id="handoff-1",
            name="hand_off",
        ),
    ]
    state = {
        "selected_agent": "search_agent",
        "messages": messages,
        "context": {
            "handoff": {
                "active": True,
                "source_agent": "planning_agent",
                "target_agent": "search_agent",
                "tool_call_id": "handoff-1",
            }
        },
    }

    delegated = workflow._messages_for_selected_agent(state, "search_agent", messages)

    assert delegated == [messages[0]]
```

2. Run:

```powershell
python -m pytest tests/test_graph_planning_subagents.py::test_delegated_agent_messages_strip_handoff_control_messages -q
```

Expected: fails because the helper does not exist.

3. Implement `_messages_for_selected_agent()` in `app/ai/graph.py`.

4. Update `_chat_node`, `_search_node`, `_image_generator_node`, and `_canvas_node` to call:

```python
current_turn_messages = self._messages_for_selected_agent(
    state,
    state.get("selected_agent") or "<agent_name>",
    messages,
)
```

5. For `_rag_node`, keep `original_query` from the last `HumanMessage`, but filter `tool_context` so `ToolMessage(name="hand_off")` is ignored.

6. Run:

```powershell
python -m pytest tests/test_graph_planning_subagents.py -q
```

Expected: pass.

### Task 3: Add Same-Turn Planning Handoff Streaming Regression

Files:

- Create: `tests/test_graph_handoff_streaming.py`
- Modify: `app/ai/graph.py`

Steps:

1. Add a test with fake graph node behavior proving the stream does not complete with only source-agent transfer text.

Test shape:

```python
@pytest.mark.asyncio
async def test_planning_handoff_stream_returns_delegated_agent_answer(monkeypatch):
    from types import SimpleNamespace
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from app.ai.graph import MultiAgentWorkflow
    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole, WorkflowExecutionRequest

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = None
    workflow.agents = {"planning_agent": object(), "search_agent": object()}
    workflow._build_graph_config = lambda thread_id=None: {}
    workflow._resolve_thread_id = lambda thread_id, conversation_id: conversation_id
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow._route_node = AsyncMock(return_value={"selected_agent": "planning_agent"})

    final_response = AgentResponse(
        agent_type=AgentType.SEARCH,
        agent_id="search_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content="Search Agent final answer."),
        metadata={},
    )
    final_state = {
        "selected_agent": "search_agent",
        "messages": [
            HumanMessage(content="latest info please"),
            AIMessage(content="", tool_calls=[{"id": "h1", "name": "hand_off", "args": {}}]),
            ToolMessage(content='{"hand_off":"search_agent"}', tool_call_id="h1", name="hand_off"),
            AIMessage(content="Search Agent final answer."),
        ],
        "response": final_response,
        "context": {"handoff": {"active": True, "target_agent": "search_agent", "tool_call_id": "h1"}},
    }

    class FakeGraph:
        async def astream(self, *_args, **_kwargs):
            yield ("updates", {"planning_tools": {"selected_agent": "search_agent", "context": final_state["context"]}})
            yield ("updates", {"search_agent": final_state})

    workflow.graph = FakeGraph()

    events = [event async for event in workflow.execute_request_stream(WorkflowExecutionRequest(message="latest info please", conversation_id="conv-1"))]

    assert any(event == {"type": "agent_selected", "agent": "search_agent", "reason": "handoff"} for event in events)
    complete = next(event for event in events if event["type"] == "complete")
    assert complete["response"].message.content == "Search Agent final answer."
```

2. Run:

```powershell
python -m pytest tests/test_graph_handoff_streaming.py -q
```

Expected: fail until handoff agent-change events are implemented.

3. Implement selected-agent change event emission in both `execute_request_stream()` and `resume_with_decisions_stream()`.

4. Run:

```powershell
python -m pytest tests/test_graph_handoff_streaming.py tests/test_graph_planning_subagents.py -q
```

Expected: pass.

### Task 4: Prevent Handoff Narration From Becoming Final Fallback

Files:

- Modify: `app/ai/graph.py`
- Modify: `tests/test_graph_handoff_streaming.py`

Steps:

1. Add a test for `_recover_terminal_response()` where the only AI content is a handoff AIMessage with tool calls.

```python
def test_recover_terminal_response_ignores_handoff_narration():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from app.ai.graph import MultiAgentWorkflow

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {
        "selected_agent": "search_agent",
        "messages": [
            HumanMessage(content="latest info please"),
            AIMessage(
                content="Transfering your request to Search Agent...",
                tool_calls=[{"id": "h1", "name": "hand_off", "args": {}}],
            ),
            ToolMessage(content='{"hand_off":"search_agent"}', tool_call_id="h1", name="hand_off"),
        ],
        "context": {"handoff": {"active": True, "target_agent": "search_agent", "tool_call_id": "h1"}},
    }

    assert workflow._recover_terminal_response(state) is None
```

2. Run:

```powershell
python -m pytest tests/test_graph_handoff_streaming.py::test_recover_terminal_response_ignores_handoff_narration -q
```

Expected: pass if current logic already ignores AIMessage with tool calls; keep it as regression coverage.

3. If the test fails, update `_recover_terminal_response()` to ignore AIMessage entries whose tool calls include `hand_off`.

### Task 5: Add Live Markdown Entity Normalization

Files:

- Modify: `demo.py`
- Create: `tests/test_demo_stream_rendering.py`

Steps:

1. Add a helper test:

```python
def test_normalize_stream_markdown_text_unescapes_quotes_only():
    from demo import normalize_stream_markdown_text

    raw = 'model: &quot;gemini-3.1-pro&quot;, provider: &#34;gemini&#34;: &lt;b&gt;safe text&lt;/b&gt;'

    assert normalize_stream_markdown_text(raw) == (
        'model: "gemini-3.1-pro", provider: "gemini": &lt;b&gt;safe text&lt;/b&gt;'
    )
```

2. Run:

```powershell
python -m pytest tests/test_demo_stream_rendering.py -q
```

Expected: fail because helper does not exist.

3. Implement `normalize_stream_markdown_text()` in `demo.py`.

4. Replace both live markdown calls:

```python
response_placeholder.markdown(accumulated_content)
```

with:

```python
response_placeholder.markdown(normalize_stream_markdown_text(accumulated_content))
```

Locations:

- `demo.py` normal message stream around current line 8001.
- `demo.py` resume stream around current line 7425.

5. Run:

```powershell
python -m pytest tests/test_demo_stream_rendering.py tests/test_demo_subagent_activity.py -q
```

Expected: pass.

### Task 6: Focused Regression Suite

Run:

```powershell
python -m pytest tests/test_message_history_pipeline.py tests/test_graph_planning_subagents.py tests/test_graph_handoff_streaming.py tests/test_demo_stream_rendering.py tests/test_demo_subagent_activity.py tests/test_message_service_subagent_streaming.py -q
```

Expected: all focused tests pass.

### Task 7: Optional Manual Smoke Checks

Run the app:

```powershell
streamlit run demo.py
```

Manual checks:

- With LangSmith tracing enabled, send a normal chat turn. Confirm no visible noisy `LangGraphUpdateState` cleanup run appears after the turn. If it still appears, implement the lower-level checkpoint rewrite fallback instead of `graph.aupdate_state()`.
- In Planning mode with an existing plan, ask a current-info question that requires Search Agent. Confirm status switches from Planning Agent to Search Agent and the Search Agent answers in the same turn.
- Stream an answer containing `model: &quot;gemini-3.1-pro&quot;, provider: &quot;gemini&quot;:`. Confirm live UI shows quotes, not entities.

## Acceptance Criteria

- LangSmith no longer shows the post-turn checkpoint cleanup as a noisy `LangGraphUpdateState` by default.
- Checkpoint compaction still removes terminal checkpoint messages and skips interrupted checkpoints.
- `LangGraphUpdateState` is documented and understood as an internal cleanup trace label, not an app-defined agent or model response.
- Planning Agent handoff to Search Agent completes with Search Agent's final answer in the same turn.
- Handoff source-agent transfer prose is not the persisted final assistant answer.
- The stream emits a second `agent_selected` event when handoff changes the active agent.
- Live Streamlit rendering displays escaped quotes as quotes while preserving escaped angle brackets.
- Focused regression suite passes.

## Risks And Mitigations

- Risk: `tracing_context(enabled=False)` may not suppress LangGraph's internal update-state tracing.
  - Mitigation: keep the test and manual LangSmith smoke. If still noisy, move compaction to a direct checkpointer operation in a follow-up patch.

- Risk: stripping too many messages for delegated agents could remove useful tool evidence.
  - Mitigation: strip only handoff control AI/Tool messages by matching `tool_call_id`; keep non-handoff tool messages.

- Risk: broad HTML entity unescaping could render unsafe HTML.
  - Mitigation: normalize only quote entities in the live markdown path.

## Implementation Progress (2026-05-21)

Plan executed end-to-end on branch `Thai-Postgre-FastAPI`. All tasks completed and verified.

### Task 1 — DONE
- Added failing test `test_checkpoint_compaction_disables_langsmith_tracing` in [tests/test_message_history_pipeline.py](../tests/test_message_history_pipeline.py).
- Imported `tracing_context` from `langsmith` in [app/ai/graph.py](../app/ai/graph.py) and wrapped `self.graph.aupdate_state(...)` inside `with tracing_context(enabled=False):` in `_compact_checkpoint_after_terminal_response`.
- Verification: `python -m pytest tests/test_message_history_pipeline.py -q` → 7 passed.

### Task 2 — DONE
- Extended `_apply_hand_off_if_present` in [app/ai/graph.py](../app/ai/graph.py) to stamp `state["context"]["handoff"] = {active, source_agent, target_agent, reason, tool_call_id}` whenever it successfully reroutes.
- Added new helper `_messages_for_selected_agent(state, agent_name, messages)` that strips the source-agent handoff AIMessage and matching `ToolMessage(name="hand_off")` when an active handoff targets `agent_name`. Falls back to `_get_current_turn_messages` semantics otherwise.
- Wired the helper into `_chat_node`, `_search_node`, `_image_generator_node`, and `_canvas_node` (replacing the previous `_get_current_turn_messages` calls).
- `_rag_node` now skips `ToolMessage(name="hand_off")` when building `tool_context` so handoff JSON does not leak into RAG evidence.
- Added regression tests in [tests/test_graph_planning_subagents.py](../tests/test_graph_planning_subagents.py):
  - `test_delegated_agent_messages_strip_handoff_control_messages`
  - `test_apply_hand_off_records_control_metadata`
  - `test_delegated_agent_messages_passthrough_when_no_active_handoff`
- Verification: `python -m pytest tests/test_graph_planning_subagents.py -q` → 36 passed.

### Task 3 — DONE
- Added `last_emitted_agent` tracking after the initial `agent_selected` yield in both `execute_request_stream` and `resume_with_decisions_stream`.
- In each stream's `updates` handler, immediately after `last_state_values.update(node_state)`, the streamer now compares `node_state.get("selected_agent")` against `last_emitted_agent` and yields `{"type": "agent_selected", "agent": new_agent, "reason": "handoff"}` whenever the active agent changes.
- Created [tests/test_graph_handoff_streaming.py](../tests/test_graph_handoff_streaming.py) with `test_planning_handoff_stream_returns_delegated_agent_answer` pinning the second `agent_selected` event and the delegated agent's final answer.
- Verification: `python -m pytest tests/test_graph_handoff_streaming.py tests/test_graph_planning_subagents.py -q` → all green.

### Task 4 — DONE (no impl change needed)
- Added `test_recover_terminal_response_ignores_handoff_narration` to [tests/test_graph_handoff_streaming.py](../tests/test_graph_handoff_streaming.py).
- The existing `_recover_terminal_response` (graph.py:3122–3127, 3131–3140) already skips AIMessages with `tool_calls`, so the handoff narration cannot become the recovered final response. Kept as regression coverage so future changes cannot regress this.
- Verification: passes alongside Task 3.

### Task 5 — DONE
- Created new module [app/ui/stream_markdown.py](../app/ui/stream_markdown.py) hosting `_STREAM_MARKDOWN_ENTITY_MAP` and `normalize_stream_markdown_text(content)`.
- [demo.py](../demo.py) re-exports `normalize_stream_markdown_text` from the new module and uses it in both live markdown call sites:
  - Resume stream around former line 7425.
  - Normal message stream around former line 8001.
- Created [tests/test_demo_stream_rendering.py](../tests/test_demo_stream_rendering.py) with three cases covering the quote-only allowlist, apostrophe-entity coverage, and empty/non-string inputs.
- Verification: `python -m pytest tests/test_demo_stream_rendering.py -q` → 3 passed.

### Task 6 — DONE
- Full focused regression suite passes:
  `python -m pytest tests/test_message_history_pipeline.py tests/test_graph_planning_subagents.py tests/test_graph_handoff_streaming.py tests/test_demo_stream_rendering.py tests/test_demo_subagent_activity.py tests/test_message_service_subagent_streaming.py -q` → **61 passed**.
- Broader sanity sweep: `python -m pytest tests/ -k "graph or hand_off or handoff or stream or compact" --ignore=tests/client_backend -q` → **69 passed**.

### Design Decisions Made During Implementation

1. **`normalize_stream_markdown_text` lives in `app/ui/stream_markdown.py`, not `demo.py`.** The plan's test (`from demo import normalize_stream_markdown_text`) could not run because `demo.py` imports `streamlit`, `markdown`, and `dateutil` at module load — none of which are installed in the test environment. Trying to stub streamlit deeply enough to let `demo.py` import cleanly was fragile (it transitively touches `set_page_config`, `session_state`, decorators). Extracting the pure helper to `app/ui/stream_markdown.py` and re-exporting it from `demo.py` keeps the public surface identical for callers (`from demo import normalize_stream_markdown_text` still works at runtime), makes the helper headlessly testable, and removes the streamlit-stub coupling from the test. Net effect: the test in the plan would still pass against the re-export, but our test imports the module directly to skip the streamlit dependency entirely.

2. **`context["handoff"]` carries `reason` plus a stable `tool_call_id` for control-plane bookkeeping.** The plan suggested storing `source_agent`, `target_agent`, `reason`, and `tool_call_id`. We kept all of these because:
   - `tool_call_id` is the precise key for stripping the matching `ToolMessage` in `_messages_for_selected_agent` (rather than relying on `name == "hand_off"` alone, which would also strip a hypothetical second handoff in the same turn).
   - `source_agent` and `reason` are useful for trace/telemetry consumers; cheap to store.

3. **Handoff stripping uses both `name` and `tool_call_id`.** The filter drops any `ToolMessage` whose `name == "hand_off"` *or* whose `tool_call_id` matches the recorded handoff id. The `name` check is the primary guard; the `tool_call_id` check protects against handoff tool implementations whose ToolMessage was built without a `name`.

4. **Lightweight, type-permissive helper.** `_messages_for_selected_agent` returns the existing `_get_current_turn_messages` slice unchanged whenever no handoff is active or the target does not match. This keeps the change surgical for non-handoff turns and matches FR-013 (existing rendering remains compatible).

5. **RAG `tool_context` filter is targeted, not state-aware.** Rather than threading the handoff metadata into `_rag_node`, we simply drop `ToolMessage(name="hand_off")` from the `tool_context` list. RAG's existing `original_query` extraction is unchanged.

6. **Test for `test_recover_terminal_response_ignores_handoff_narration` kept as guardrail.** The current code already handles this correctly (skips AIMessages with `tool_calls`), but the plan asked for the test as regression coverage — added without code change, per FR-008.

7. **Did not delete `_compact_checkpoint_after_terminal_response`.** Per FR-005 and Finding 1A, removal would require an explicit A/B test proving checkpoint messages don't accumulate without it. Out of scope for this patch.

### Files Touched

- Modified: [app/ai/graph.py](../app/ai/graph.py) — `tracing_context` import; trace-suppressing `with` around `aupdate_state`; handoff context metadata; `_messages_for_selected_agent` helper; nodes wired to helper; `_rag_node` hand_off filter; `last_emitted_agent` + handoff-driven `agent_selected` events in both stream functions.
- Modified: [demo.py](../demo.py) — replaced inline helper with re-export from `app.ui.stream_markdown`; both live `markdown(...)` call sites now route through `normalize_stream_markdown_text`.
- Created: [app/ui/stream_markdown.py](../app/ui/stream_markdown.py).
- Modified: [tests/test_message_history_pipeline.py](../tests/test_message_history_pipeline.py) — added `test_checkpoint_compaction_disables_langsmith_tracing`.
- Modified: [tests/test_graph_planning_subagents.py](../tests/test_graph_planning_subagents.py) — added three handoff-scoping regressions.
- Created: [tests/test_graph_handoff_streaming.py](../tests/test_graph_handoff_streaming.py).
- Created: [tests/test_demo_stream_rendering.py](../tests/test_demo_stream_rendering.py).
