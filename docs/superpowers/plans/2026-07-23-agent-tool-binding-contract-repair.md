# Agent Tool-Binding Contract Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore production execution for custom and planning agents by making their tool-binding overrides fully compatible with the `BaseAgent` exclusion contract.

**Architecture:** Keep `BaseAgent` as the authoritative method contract. Add the explicit optional parameter to each affected override, enforce exclusions after `CustomAgent` assembles its complete tool collection, and forward exclusions through `PlanningAgent` while preserving mandatory `write_todos` injection.

**Tech Stack:** Python 3.11, LangChain tools, pytest, pytest monkeypatch

---

## File Map

- Modify `tests/test_custom_agents_tools.py`: cover custom-agent acceptance and enforcement of tool-name exclusions.
- Modify `tests/test_planning_subagents.py`: cover both planning override paths and preservation of `write_todos`.
- Modify `app/ai/agents/custom_agent.py`: accept the base contract and filter the final custom-agent tool collection.
- Modify `app/ai/agents/planning_agent.py`: accept and forward the base contract through both planning overrides.

### Task 1: CustomAgent Contract and Exclusion Behavior

**Files:**
- Modify: `tests/test_custom_agents_tools.py`
- Modify: `app/ai/agents/custom_agent.py:172-276`

- [ ] **Step 1: Write the failing custom-agent regression test**

Add this test after `test_two_custom_agents_do_not_share_deferred_tool_state`:

```python
def test_custom_agent_binding_honors_excluded_tool_names():
    from app.ai.agents.custom_agent import CustomAgent

    agent = CustomAgent(_spec())
    agent.tools = []
    blocked = _FakeTool("blocked_tool", {})
    retained = _FakeTool("retained_tool", {})

    tools = agent._get_tools_for_binding(
        internal_tools=[blocked, retained],
        excluded_tool_names={"blocked_tool"},
    )
    names = {tool.name for tool in tools}

    assert "blocked_tool" not in names
    assert "retained_tool" in names
```

- [ ] **Step 2: Run the test and verify the current regression**

Run:

```powershell
conda run -n agents python -m pytest tests/test_custom_agents_tools.py::test_custom_agent_binding_honors_excluded_tool_names -v
```

Expected: FAIL with `TypeError: CustomAgent._get_tools_for_binding() got an unexpected keyword argument 'excluded_tool_names'`.

- [ ] **Step 3: Implement explicit compatibility and final-collection filtering**

Change the method signature in `app/ai/agents/custom_agent.py` to:

```python
def _get_tools_for_binding(
    self,
    conversation_id: str | None = None,
    internal_tools: list[BaseTool] | None = None,
    user_id: str | None = None,
    device_id: str | None = None,
    tool_scope: str | None = None,
    include_hand_off: bool | None = None,
    excluded_tool_names: set[str] | frozenset[str] | None = None,
) -> list[BaseTool]:
```

Replace the final deduplication loop with:

```python
excluded = set(excluded_tool_names or ())
seen: set[str] = set()
deduped: list[BaseTool] = []
for tool in tools:
    name = getattr(tool, "name", None)
    if name in excluded or name in seen:
        continue
    seen.add(name)
    deduped.append(tool)
return deduped
```

This applies the same name-based semantics as `BaseAgent` after all custom tool
sources have been assembled.

- [ ] **Step 4: Run the focused custom-agent test**

Run:

```powershell
conda run -n agents python -m pytest tests/test_custom_agents_tools.py::test_custom_agent_binding_honors_excluded_tool_names -v
```

Expected: PASS.

- [ ] **Step 5: Run the complete custom-agent tool suite**

Run:

```powershell
conda run -n agents python -m pytest tests/test_custom_agents_tools.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the custom-agent repair**

```powershell
git add -- app/ai/agents/custom_agent.py tests/test_custom_agents_tools.py
git commit -m "fix: restore custom agent tool binding contract"
```

### Task 2: PlanningAgent Contract Propagation

**Files:**
- Modify: `tests/test_planning_subagents.py`
- Modify: `app/ai/agents/planning_agent.py:72-115`

- [ ] **Step 1: Write the failing planning binding regression test**

Add this test after `test_planning_agent_binding_includes_graph_injected_hand_off`:

```python
def test_planning_agent_binding_honors_excluded_tool_names(monkeypatch):
    from app.ai.agents.planning_agent import PlanningAgent
    from app.ai.hand_off_tool import create_hand_off_tool

    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.get_available_skill_summaries",
        lambda **kwargs: [],
    )

    agent = PlanningAgent.__new__(PlanningAgent)
    agent.agent_config_key = "planning"
    agent.mcp_manager = None
    agent.tools = []

    tools = agent._get_tools_for_binding(
        conversation_id="conversation-1",
        internal_tools=[create_hand_off_tool(["search_agent"])],
        excluded_tool_names={"hand_off"},
    )
    tool_names = {tool.name for tool in tools}

    assert "write_todos" in tool_names
    assert "hand_off" not in tool_names
```

- [ ] **Step 2: Write the failing planning LLM-binding propagation test**

Add this test immediately after the test from Step 1:

```python
def test_planning_llm_binding_forwards_excluded_tool_names(monkeypatch):
    from app.ai.agents.base_agent import BaseAgent
    from app.ai.agents.planning_agent import PlanningAgent

    captured = {}

    def fake_get_llm_with_tools(self, **kwargs):
        captured.update(kwargs)
        return "bound-model"

    monkeypatch.setattr(BaseAgent, "_get_llm_with_tools", fake_get_llm_with_tools)
    agent = PlanningAgent.__new__(PlanningAgent)

    result = agent._get_llm_with_tools(
        model=object(),
        excluded_tool_names={"blocked_tool"},
    )

    assert result == "bound-model"
    assert captured["excluded_tool_names"] == {"blocked_tool"}
    assert "write_todos" in {tool.name for tool in captured["internal_tools"]}
```

- [ ] **Step 3: Run both tests and verify the current regression**

Run:

```powershell
conda run -n agents python -m pytest tests/test_planning_subagents.py::test_planning_agent_binding_honors_excluded_tool_names tests/test_planning_subagents.py::test_planning_llm_binding_forwards_excluded_tool_names -v
```

Expected: both FAIL with unexpected-keyword `TypeError` messages from the corresponding `PlanningAgent` overrides.

- [ ] **Step 4: Update both planning override signatures and forwarding**

Add the explicit parameter to `_get_llm_with_tools`:

```python
excluded_tool_names: set[str] | frozenset[str] | None = None,
```

Forward it in the superclass call:

```python
excluded_tool_names=excluded_tool_names,
```

Make the same two edits to `_get_tools_for_binding`. Its superclass call must
continue forwarding `tool_scope` and `include_hand_off` and additionally include:

```python
excluded_tool_names=excluded_tool_names,
```

- [ ] **Step 5: Run the focused planning tests**

Run:

```powershell
conda run -n agents python -m pytest tests/test_planning_subagents.py::test_planning_agent_binding_honors_excluded_tool_names tests/test_planning_subagents.py::test_planning_llm_binding_forwards_excluded_tool_names -v
```

Expected: both PASS.

- [ ] **Step 6: Run the existing planning tool-binding neighborhood**

Run:

```powershell
conda run -n agents python -m pytest tests/test_planning_subagents.py -k "binding or hand_off or write_todos" -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the planning-agent repair**

```powershell
git add -- app/ai/agents/planning_agent.py tests/test_planning_subagents.py
git commit -m "fix: propagate planning tool exclusions"
```

### Task 3: Cross-Agent Regression Verification

**Files:**
- Verify: `app/ai/agents/base_agent.py`
- Verify: `app/ai/agents/custom_agent.py`
- Verify: `app/ai/agents/planning_agent.py`
- Verify: `tests/test_custom_agents_graph.py`
- Verify: `tests/test_base_agent_dynamic_handoff.py`
- Verify: `tests/test_graph_planning_subagents.py`

- [ ] **Step 1: Check formatting and static syntax**

Run:

```powershell
conda run -n agents python -m ruff check app/ai/agents/custom_agent.py app/ai/agents/planning_agent.py tests/test_custom_agents_tools.py tests/test_planning_subagents.py
conda run -n agents python -m compileall -q app/ai/agents/custom_agent.py app/ai/agents/planning_agent.py
git diff --check
```

Expected: all commands exit successfully with no diagnostics.

- [ ] **Step 2: Run related agent and handoff tests**

Run:

```powershell
conda run -n agents python -m pytest tests/test_custom_agents_tools.py tests/test_custom_agents_graph.py tests/test_base_agent_dynamic_handoff.py tests/test_planning_subagents.py tests/test_graph_planning_subagents.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run the full automated test suite**

Run:

```powershell
conda run -n agents python -m pytest -q
```

Expected: all tests pass. If an environment-dependent integration test is
skipped, record the reported skip reason; do not treat unrelated unavailable
external services as evidence that this repair failed.

- [ ] **Step 4: Confirm the final diff is scoped**

Run:

```powershell
git status --short
git diff HEAD~2 -- app/ai/agents/custom_agent.py app/ai/agents/planning_agent.py tests/test_custom_agents_tools.py tests/test_planning_subagents.py
```

Expected: production edits are limited to the explicit contract repair and
tests are limited to the three regression cases.

- [ ] **Step 5: Record final verification if needed**

If verification required a test-only correction, commit it with:

```powershell
git add -- tests/test_custom_agents_tools.py tests/test_planning_subagents.py
git commit -m "test: harden agent tool binding regression coverage"
```

If no correction was needed, do not create an empty commit.
