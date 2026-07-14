# Dynamic Handoff Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a model-issued `hand_off(target_agent)` a single, validated graph transition from every tool loop, including agentic RAG.

**Architecture:** The workflow computes a live roster from graph state and supplies the same dynamic handoff tool to model binding and tool execution. A shared interpreter validates and rewrites handoff outputs before normal `ToolMessage` persistence, then each continuation edge uses the selected target node.

**Tech Stack:** Python 3.13, LangChain Core, LangGraph, Pydantic settings, pytest.

---

### Task 1: Canonical tool and delegation configuration

**Files:**
- Modify: `app/ai/hand_off_tool.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/prompts.py`
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Modify: `README.md`
- Test: `tests/test_base_agent_dynamic_handoff.py`
- Test: `tests/test_planning_subagents.py`

- [x] **Step 1: Write failing target-only schema tests**

```python
def test_hand_off_tool_accepts_only_target_agent():
    schema = create_hand_off_tool(["search_agent"]).args_schema
    assert schema.model_validate({"target_agent": "search_agent"}).target_agent == "search_agent"
    with pytest.raises(ValidationError):
        schema.model_validate({"target_agent": "search_agent", "reason": "old"})
```

- [x] **Step 2: Remove the static handoff and add settings**

```python
class HandOffInput(BaseModel):
    target_agent: str = Field(..., description="The agent id to delegate to.")

def _hand_off_impl(target_agent: str) -> str:
    return json.dumps({"hand_off": target_agent})
```

Add `max_handoff_delegation_depth: int = Field(default=5, ...)` to `Settings`, the existing positive-integer validator, `MAX_HANDOFF_DELEGATION_DEPTH=5` to `.env.example`, and the README handoff section.

- [x] **Step 3: Run the contract test**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests\\test_base_agent_dynamic_handoff.py`

Expected: PASS.

### Task 2: Live roster and scoped binding

**Files:**
- Modify: `app/ai/workflow/custom_agents.py`
- Modify: `app/ai/graph.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/workflow/planning_loop.py`
- Modify: `app/ai/workflow/rag_loop.py`
- Modify: `app/ai/agents/rag_agent.py`
- Test: `tests/test_custom_agents_graph.py`
- Test: `tests/test_graph_planning_subagents.py`
- Test: `tests/test_rag_agent.py`
- Test: `tests/test_hitl_client_and_deferred.py`

- [x] **Step 1: Write failing binding tests**

```python
def test_base_roster_is_dynamic_without_custom_agents():
    kwargs = workflow._multi_agent_kwargs(state, "chat_agent")
    assert "chat_agent" not in kwargs["handoff_target_descriptions"]
    assert "search_agent" in kwargs["handoff_target_descriptions"]
```

- [x] **Step 2: Implement one live-roster helper**

```python
def _handoff_targets(self, state: GraphState, active_agent_id: str | None) -> list[str]:
    agent_ids = [*self.agents, *GraphStateView(state).custom_agents()]
    return [agent_id for agent_id in agent_ids if agent_id != active_agent_id]
```

Build descriptions from the same helper. Inject `create_hand_off_tool(...)` for every base agent, merge it with Planning's dispatch tool, and pass it through `RAGAgent.process_message` into RAG's binding and prompt.

- [x] **Step 3: Scope the execution map to the invocation**

```python
async def ensure_agent_tool_map(..., internal_tools: list[BaseTool] | None = None) -> dict[str, Any]:
    tools = agent._get_tools_for_binding(..., internal_tools=internal_tools)
```

Pass the same handoff tool to generic, Planning, and RAG tool execution, plus
the generic and RAG HITL approval/provenance paths.

- [x] **Step 4: Run binding tests**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests\\test_base_agent_dynamic_handoff.py tests\\test_custom_agents_graph.py tests\\test_graph_planning_subagents.py tests\\test_rag_agent.py`

Expected: PASS.

### Task 3: Shared interpreter and graph routing

**Files:**
- Modify: `app/ai/workflow/tool_loop.py`
- Modify: `app/ai/workflow/rag_loop.py`
- Modify: `app/ai/workflow/planning_loop.py`
- Modify: `app/ai/workflow/graph_builder.py`
- Test: `tests/test_custom_agents_graph.py`
- Test: `tests/test_graph_planning_subagents.py`
- Modify: `tests/test_rag_tool_loop_finalization.py`

- [x] **Step 1: Write failing RAG route and single-pair tests**

```python
@pytest.mark.asyncio
async def test_rag_handoff_routes_to_search_agent_in_same_turn(...):
    state = await workflow._rag_tools_node(rag_handoff_state)
    assert state["selected_agent"] == "search_agent"
    assert workflow._should_continue_rag(state) == "search_agent"
```

- [x] **Step 2: Implement canonical output validation**

```python
payload = json.loads(output["content"])
if set(payload) != {"hand_off"} or not isinstance(payload["hand_off"], str):
    return _reject(output, "Hand-off refused: canonical target is required.")
```

Reject zero/multiple calls, malformed payloads, unavailable targets, self/repeated targets, and depth exhaustion by replacing `output["content"]`. On success update selection, depth, trail, and `{active, source_agent, target_agent, tool_call_id}`.

- [x] **Step 3: Interpret before persistence and route targets**

Call the interpreter before normal tool-message writes in all three loops. Give `rag_tools` a routing map for every base target and `custom_agent`; short-circuit RAG continuation to `_route_target_for(...)` on a successful RAG-originated handoff.

- [x] **Step 4: Run protocol tests**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests\\test_rag_tool_loop_finalization.py tests\\test_custom_agents_graph.py tests\\test_graph_planning_subagents.py`

Expected: PASS.

### Task 4: Prompt scoping, metadata, and verification

**Files:**
- Modify: `app/ai/agent_metadata.py`
- Modify: `app/ai/workflow/custom_agents.py`
- Modify: `app/ai/agents/custom_agent.py`
- Modify: `app/ai/agents/planning_agent.py`
- Modify: `app/ai/workflow/planning_loop.py`
- Test: `tests/test_graph_handoff_streaming.py`
- Test: `tests/test_graph_planning_subagents.py`

- [x] **Step 1: Write failing no-rationale and Planning-scoping tests**

```python
def test_final_handoff_metadata_has_no_model_reason():
    assert "reason" not in response.metadata["handoff"]
```

- [x] **Step 2: Remove only model rationale**

Remove the model's `reason` from prompts, trail entries, and metadata; preserve stream `reason="handoff"`. Make Planning call `_messages_for_selected_agent` so it excludes the source handoff pair.

- [x] **Step 3: Run focused verification**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests\\test_base_agent_dynamic_handoff.py tests\\test_custom_agents_graph.py tests\\test_graph_handoff_streaming.py tests\\test_graph_planning_subagents.py tests\\test_rag_agent.py tests\\test_graph_refactor_contract.py tests\\test_graph_stream_projection.py`

Expected: PASS.

- [x] **Step 4: Run the AI workflow suite**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests\\test_graph_*.py tests\\test_rag_*.py tests\\test_router.py tests\\test_base_agent_dynamic_handoff.py tests\\test_custom_agents_graph.py`

Expected: PASS.
