# Agent Metadata And Custom-Agent Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a simple persisted metadata field that records which agent answered or worked on an assistant message, clean redundant metadata fields, verify custom-agent handoff/planning behavior, and show the working agent in Streamlit.

**Architecture:** Keep the graph and routing behavior unchanged except for missing custom-agent handoff fixes. Persist a compact `agent` metadata object on assistant messages from existing runtime state (`AgentResponse.agent_id`, custom-agent response metadata, handoff context, and Planning subagent results). Do not build a separate provenance ledger, query LangSmith, inject hidden context text, or infer attribution from answer content.

**Tech Stack:** FastAPI, SQLAlchemy JSONB message metadata, LangGraph, LangChain, existing `MessageService`, `MultiAgentWorkflow`, Streamlit `demo.py`, pytest.

---

## Scope

### What This Builds

- A single canonical assistant-message metadata field:

```json
{
  "agent": {
    "id": "custom_agent:uuid",
    "kind": "custom",
    "name": "Data Analyst",
    "custom_agent_id": "uuid",
    "source": "response"
  }
}
```

- Optional compact companion fields only when relevant:

```json
{
  "handoff": {
    "from_agent_id": "planning_agent",
    "to_agent_id": "custom_agent:uuid",
    "reason": "needs domain-specific analysis",
    "tool_call_id": "call-1"
  },
  "subagent_results": [
    {
      "id": "risk-review",
      "agent": "custom_agent:uuid",
      "agent_name": "Data Analyst",
      "agent_kind": "custom",
      "custom_agent_id": "uuid",
      "status": "completed",
      "summary": "Reviewed the migration risks."
    }
  ]
}
```

- Streamlit displays the agent name for live streams and persisted messages.
- Metadata cleanup removes redundant or deprecated custom-agent identity fields after compatibility handling.
- Custom-agent handoff and Planning-mode custom-agent assignment are verified and fixed where needed.

### Non-Goals

- No line-level attribution.
- No LangSmith API integration.
- No persisted local LangSmith run ledger.
- No tool that reads metadata back into the model.
- No prompt-history/context injection.
- No database migration; existing `messages.message_metadata` JSONB is enough.

---

## Current Code Findings

- `app/ai/agents/custom_agent.py` currently adds `runtime_agent_id`, `custom_agent_id`, and `custom_agent_name` to response metadata.
- `app/services/message_service.py::_agent_selected_event()` enriches streaming events with `agent_name` for custom agents, but final persisted messages do not have one simple canonical agent metadata field.
- `app/ai/graph.py::_apply_hand_off_if_present()` stores handoff control data in `state["context"]["handoff"]`.
- `app/ai/planning_subagents.py` already stores `subagent_results`, but results need custom-agent display identity normalized for UI.
- `app/ai/graph.py::_should_continue_planning()` has a base-agent-only handoff branch; Planning handoff should validate that the delegated target is a base agent or attached custom agent before routing through `_route_target_for()`.
- `demo.py` has live stream status and trace panels, but persisted messages do not show “worked by Data Analyst/Search Agent” as a first-class UI element.
- This repository's `pytest` executable is not on PATH in the observed shell; verification commands should use `python -m pytest`.

---

## Metadata Contract

### Canonical Field

Use `message_metadata["agent"]` for the final assistant message producer:

```python
{
    "id": str,                     # "chat_agent" or "custom_agent:<uuid>"
    "kind": "base" | "custom",
    "name": str,                   # display name
    "custom_agent_id": str | None,
    "source": "response" | "selected_agent" | "handoff",
}
```

### Companion Fields

Use `message_metadata["handoff"]` when the final answer came after a handoff:

```python
{
    "from_agent_id": str | None,
    "to_agent_id": str,
    "reason": str,
    "tool_call_id": str | None,
}
```

Keep the existing `message_metadata["subagent_results"]` field for Planning dispatch summaries and enrich each entry with display identity:

```python
[
    {
        "id": str,
        "agent": str,                    # existing worker target id, e.g. "search_agent" or "custom_agent:<uuid>"
        "agent_name": str,
        "agent_kind": "base" | "custom",
        "custom_agent_id": str | None,
        "status": str,
        "summary": str,
    }
]
```

### Deprecated / Compatibility Fields

These may be read as inputs during migration but should not be the preferred persisted contract:

- `runtime_agent_id`
- `custom_agent_id`
- `custom_agent_name`
- `subagent_results[].agent` without companion display identity

Keep them only if existing tests or UI still depend on them. The plan includes tests to confirm `agent` is canonical and Streamlit reads it first.

---

## File Map

Create:

- `app/ai/agent_metadata.py` — small helpers to normalize base/custom agent identity and fold existing metadata into the canonical `agent`, `handoff`, and enriched `subagent_results` fields.
- `tests/test_agent_metadata.py` — focused tests for metadata normalization and cleanup.

Modify:

- `app/ai/agents/custom_agent.py` — keep existing custom response metadata for compatibility, but rely on finalizer to create canonical `agent`.
- `app/ai/graph.py` — attach canonical `agent` metadata to final responses, carry handoff metadata, fix Planning handoff to custom agents.
- `app/ai/planning_subagents.py` — include display identity in Planning worker results.
- `app/core/response_constants.py` — cleanup redundant metadata and prefer canonical `agent`.
- `app/services/message_service.py` — ensure streaming, resume, interrupt, stopped, and error paths preserve the correct agent metadata where available.
- `app/ui/subagent_activity.py` — read `agent_name` first and keep `agent` as the worker id fallback.
- `demo.py` — show live and persisted agent labels.

Update tests:

- `tests/test_custom_agents_graph.py`
- `tests/test_custom_agents_message_service.py`
- `tests/test_custom_agents_planning.py`
- `tests/test_graph_handoff_streaming.py`
- `tests/test_graph_planning_subagents.py`
- `tests/test_message_service_subagent_streaming.py`
- `tests/test_demo_custom_agents.py`
- `tests/test_demo_subagent_activity.py`

---

## Task 1: Add Canonical Agent Metadata Helper

**Files:**
- Create: `app/ai/agent_metadata.py`
- Create: `tests/test_agent_metadata.py`

> **Progress (done 2026-06-02):** Implemented `app/ai/agent_metadata.py` and `tests/test_agent_metadata.py` exactly as specified. Verification: `python -m pytest tests/test_agent_metadata.py -v` → 5 passed. Design notes: the repo's pytest reporter (RTK) collapses output to `Pytest: N passed` and prints only `Pytest: No tests collected` on a collection/import error — so a "failing test" step shows up as "No tests collected" rather than a red failure. Confirmed the Step-2 failure with a direct `python -c "import app.ai.agent_metadata"` (ModuleNotFoundError) before implementing.

- [x] **Step 1: Write failing metadata helper tests**

Create `tests/test_agent_metadata.py`:

```python
from app.ai.agent_metadata import (
    agent_identity,
    attach_agent_metadata,
    normalize_subagent_metadata,
)


def test_agent_identity_resolves_custom_agent_name():
    custom_agents = {
        "custom_agent:abc": {
            "id": "abc",
            "runtime_agent_id": "custom_agent:abc",
            "name": "Data Analyst",
        }
    }

    assert agent_identity("custom_agent:abc", custom_agents) == {
        "id": "custom_agent:abc",
        "kind": "custom",
        "name": "Data Analyst",
        "custom_agent_id": "abc",
    }


def test_agent_identity_resolves_base_agent_name():
    assert agent_identity("search_agent", {}) == {
        "id": "search_agent",
        "kind": "base",
        "name": "Search Agent",
        "custom_agent_id": None,
    }


def test_attach_agent_metadata_prefers_response_agent_id():
    metadata = {}

    attach_agent_metadata(
        metadata,
        response_agent_id="search_agent",
        selected_agent_id="chat_agent",
        custom_agents={},
    )

    assert metadata["agent"]["id"] == "search_agent"
    assert metadata["agent"]["source"] == "response"


def test_attach_agent_metadata_uses_custom_response_compat_fields():
    metadata = {
        "runtime_agent_id": "custom_agent:abc",
        "custom_agent_id": "abc",
        "custom_agent_name": "Data Analyst",
    }

    attach_agent_metadata(
        metadata,
        response_agent_id=None,
        selected_agent_id=None,
        custom_agents={},
    )

    assert metadata["agent"] == {
        "id": "custom_agent:abc",
        "kind": "custom",
        "name": "Data Analyst",
        "custom_agent_id": "abc",
        "source": "response",
    }


def test_normalize_subagent_metadata_adds_display_names():
    custom_agents = {
        "custom_agent:abc": {
            "id": "abc",
            "runtime_agent_id": "custom_agent:abc",
            "name": "Data Analyst",
        }
    }
    raw = [{"id": "w1", "agent": "custom_agent:abc", "status": "completed", "summary": "ok"}]

    normalized = normalize_subagent_metadata(raw, custom_agents)

    assert normalized == [
        {
            "id": "w1",
            "agent": "custom_agent:abc",
            "agent_name": "Data Analyst",
            "agent_kind": "custom",
            "custom_agent_id": "abc",
            "status": "completed",
            "summary": "ok",
        }
    ]
```

- [x] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_agent_metadata.py -v
```

Expected: fails because `app.ai.agent_metadata` does not exist.

- [x] **Step 3: Implement `app/ai/agent_metadata.py`**

Create:

```python
from __future__ import annotations

from typing import Any

BASE_AGENT_DISPLAY_NAMES = {
    "chat_agent": "Chat Agent",
    "rag_agent": "RAG Agent",
    "search_agent": "Search Agent",
    "image_generator_agent": "Image Generator Agent",
    "planning_agent": "Planning Agent",
    "canvas_agent": "Canvas Agent",
}


def is_custom_agent_id(agent_id: str | None) -> bool:
    return isinstance(agent_id, str) and agent_id.startswith("custom_agent:")


def _fallback_name(agent_id: str) -> str:
    return agent_id.replace("_", " ").title()


def agent_identity(
    agent_id: str | None,
    custom_agents: dict[str, Any] | None,
    *,
    fallback_name: str | None = None,
    custom_agent_id: str | None = None,
) -> dict[str, Any] | None:
    if not agent_id:
        return None

    custom_agents = custom_agents if isinstance(custom_agents, dict) else {}
    entry = custom_agents.get(agent_id)
    if isinstance(entry, dict):
        resolved_custom_id = entry.get("id") or custom_agent_id
        if not resolved_custom_id and ":" in agent_id:
            resolved_custom_id = agent_id.split(":", 1)[1]
        return {
            "id": agent_id,
            "kind": "custom",
            "name": str(entry.get("name") or fallback_name or agent_id),
            "custom_agent_id": str(resolved_custom_id) if resolved_custom_id else None,
        }

    if is_custom_agent_id(agent_id):
        resolved_custom_id = custom_agent_id or agent_id.split(":", 1)[1]
        return {
            "id": agent_id,
            "kind": "custom",
            "name": fallback_name or agent_id,
            "custom_agent_id": str(resolved_custom_id) if resolved_custom_id else None,
        }

    return {
        "id": agent_id,
        "kind": "base",
        "name": BASE_AGENT_DISPLAY_NAMES.get(agent_id, fallback_name or _fallback_name(agent_id)),
        "custom_agent_id": None,
    }


def attach_agent_metadata(
    metadata: dict[str, Any],
    *,
    response_agent_id: str | None,
    selected_agent_id: str | None,
    custom_agents: dict[str, Any] | None,
) -> None:
    compat_runtime_id = metadata.get("runtime_agent_id")
    compat_name = metadata.get("custom_agent_name")
    compat_custom_id = metadata.get("custom_agent_id")

    if response_agent_id:
        source = "response"
        agent_id = response_agent_id
    elif isinstance(compat_runtime_id, str) and compat_runtime_id:
        source = "response"
        agent_id = compat_runtime_id
    else:
        source = "selected_agent"
        agent_id = selected_agent_id

    identity = agent_identity(
        agent_id,
        custom_agents,
        fallback_name=compat_name if isinstance(compat_name, str) else None,
        custom_agent_id=compat_custom_id if isinstance(compat_custom_id, str) else None,
    )
    if identity:
        metadata["agent"] = {**identity, "source": source}


def normalize_handoff_metadata(context: dict[str, Any] | None) -> dict[str, Any] | None:
    context = context if isinstance(context, dict) else {}
    handoff = context.get("handoff")
    if not isinstance(handoff, dict) or not handoff.get("target_agent"):
        return None
    return {
        "from_agent_id": handoff.get("source_agent"),
        "to_agent_id": handoff.get("target_agent"),
        "reason": handoff.get("reason") or "",
        "tool_call_id": handoff.get("tool_call_id"),
    }


def normalize_subagent_metadata(
    raw_results: Any,
    custom_agents: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_results, list):
        return []

    normalized: list[dict[str, Any]] = []
    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        agent_id = entry.get("agent_id") or entry.get("agent")
        identity = agent_identity(
            str(agent_id) if agent_id else None,
            custom_agents,
            fallback_name=entry.get("agent_name") if isinstance(entry.get("agent_name"), str) else None,
            custom_agent_id=entry.get("custom_agent_id")
            if isinstance(entry.get("custom_agent_id"), str)
            else None,
        )
        if not identity:
            continue
        item = {
            "id": str(entry.get("id") or entry.get("worker_id") or ""),
            "agent": identity["id"],
            "agent_name": identity["name"],
            "agent_kind": identity["kind"],
            "custom_agent_id": identity.get("custom_agent_id"),
            "status": str(entry.get("status") or "unknown"),
            "summary": str(entry.get("summary") or ""),
        }
        normalized.append({k: v for k, v in item.items() if v not in (None, "")})
    return normalized
```

- [x] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests/test_agent_metadata.py -v
```

Expected: all pass.

---

## Task 2: Attach Canonical Metadata To Final Responses

**Files:**
- Modify: `app/ai/graph.py`
- Modify: `app/core/response_constants.py`
- Test: `tests/test_custom_agents_graph.py`
- Test: `tests/test_agent_metadata.py`

> **Progress (done 2026-06-02):** Added `_attach_final_agent_metadata()` to `MultiAgentWorkflow` and wired it into `_finalize_agent_response()`, `_attach_context_outputs()` (so recovered terminal responses get metadata), and `_build_interrupt_agent_response()`. Added the `agent`-gated compat-field cleanup to `build_bot_metadata()`. Verification: `python -m pytest tests/test_agent_metadata.py tests/test_custom_agents_graph.py -v` → 21 passed. Design decisions: (1) Placed the new `from .agent_metadata import …` above the `.agents.*` block to satisfy isort (`agent_metadata` < `agents`). (2) Placed the compat-field cleanup right after the `live_widgets` derivation in `build_bot_metadata`, before the three `return` paths, so it runs regardless of the inline-rich-response feature flag.

- [x] **Step 1: Write failing graph finalization tests**

Add to `tests/test_custom_agents_graph.py`:

Update the existing schema import near the top of the file:

```python
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
```

```python
def test_finalize_response_adds_canonical_custom_agent_metadata():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])
    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id=rid,
        message=AgentMessage(role=MessageRole.ASSISTANT, content="answer"),
        metadata={"runtime_agent_id": rid, "custom_agent_name": "A0"},
    )

    wf._attach_final_agent_metadata(state, response)

    assert response.metadata["agent"] == {
        "id": rid,
        "kind": "custom",
        "name": "A0",
        "custom_agent_id": rid.split(":", 1)[1],
        "source": "response",
    }


def test_finalize_response_adds_handoff_metadata():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state("chat_agent", [rid])
    wf._apply_hand_off_if_present(state, _handoff_output(rid, tool_call_id="h1"))

    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id=rid,
        message=AgentMessage(role=MessageRole.ASSISTANT, content="answer"),
        metadata={},
    )

    wf._attach_final_agent_metadata(state, response)

    assert response.metadata["handoff"]["from_agent_id"] == "chat_agent"
    assert response.metadata["handoff"]["to_agent_id"] == rid


def test_recover_terminal_response_adds_canonical_agent_metadata():
    wf = _workflow()
    rid = f"custom_agent:{uuid4()}"
    state = _multi_custom_state(rid, [rid])
    state["messages"].append(AIMessage(content="answer"))

    response = wf._recover_terminal_response(state)

    assert response is not None
    assert response.metadata["agent"]["id"] == rid
    assert response.metadata["agent"]["name"] == "A0"
```

Add to `tests/test_agent_metadata.py`:

```python
from app.core.response_constants import build_bot_metadata
from app.schemas.workflow import WorkflowResponse, WorkflowResponseMessage


def test_build_bot_metadata_keeps_agent_and_removes_redundant_custom_fields():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="answer"),
        metadata={
            "agent": {
                "id": "custom_agent:abc",
                "kind": "custom",
                "name": "Data Analyst",
                "custom_agent_id": "abc",
                "source": "response",
            },
            "runtime_agent_id": "custom_agent:abc",
            "custom_agent_name": "Data Analyst",
            "custom_agent_id": "abc",
        },
    )

    metadata = build_bot_metadata(response)

    assert metadata["agent"]["name"] == "Data Analyst"
    assert "runtime_agent_id" not in metadata
    assert "custom_agent_name" not in metadata
    assert "custom_agent_id" not in metadata
```

- [x] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_custom_agents_graph.py::test_finalize_response_adds_canonical_custom_agent_metadata tests/test_custom_agents_graph.py::test_finalize_response_adds_handoff_metadata tests/test_custom_agents_graph.py::test_recover_terminal_response_adds_canonical_agent_metadata tests/test_agent_metadata.py::test_build_bot_metadata_keeps_agent_and_removes_redundant_custom_fields -v
```

Expected: failures because finalizer and cleanup do not exist yet.

- [x] **Step 3: Add final metadata attachment in `app/ai/graph.py`**

Import:

```python
from .agent_metadata import (
    attach_agent_metadata,
    normalize_handoff_metadata,
    normalize_subagent_metadata,
)
```

Add to `MultiAgentWorkflow`:

```python
    def _attach_final_agent_metadata(
        self,
        state: GraphState,
        response: AgentResponse,
    ) -> None:
        if response.metadata is None:
            response.metadata = {}

        custom_agents = GraphStateView(state).custom_agents()
        attach_agent_metadata(
            response.metadata,
            response_agent_id=response.agent_id,
            selected_agent_id=state.get("selected_agent"),
            custom_agents=custom_agents,
        )

        handoff = normalize_handoff_metadata(GraphStateView(state).context())
        if handoff:
            response.metadata["handoff"] = handoff

        subagent_results = normalize_subagent_metadata(
            response.metadata.get("subagent_results"),
            custom_agents,
        )
        if subagent_results:
            response.metadata["subagent_results"] = subagent_results
```

Call it at the start of `_finalize_agent_response()`:

```python
    def _finalize_agent_response(self, state: GraphState, response: AgentResponse) -> GraphState:
        self._attach_final_agent_metadata(state, response)
        state["response"] = response
```

Also call it from `_attach_context_outputs()` so recovered terminal responses built after streaming/checkpoint recovery get the same metadata:

```python
    def _attach_context_outputs(
        self, state: dict[str, Any], response: AgentResponse
    ) -> AgentResponse:
        state_view = GraphStateView(state)
        tool_artifacts = self._merge_unique_items(
            response.tool_artifacts, state_view.tool_artifacts()
        )
        response.tool_artifacts = tool_artifacts or None

        if response.metadata is None:
            response.metadata = {}

        images = self._merge_unique_items(response.metadata.get("images"), state_view.tool_images())
        if images:
            response.metadata["images"] = images

        self._attach_final_agent_metadata(state, response)
        return response
```

Attach metadata to interrupt placeholder responses in `_build_interrupt_agent_response()` before returning:

```python
        response = AgentResponse(
            agent_type=agent_type,
            agent_id=selected_agent or "search_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
            ),
            metadata={"interrupt": interrupt_response},
        )
        self._attach_final_agent_metadata(values, response)
        return response
```

- [x] **Step 4: Cleanup redundant metadata in `build_bot_metadata()`**

In `app/core/response_constants.py::build_bot_metadata()`, after metadata is copied and before return paths:

```python
    if "agent" in metadata:
        for redundant_key in (
            "runtime_agent_id",
            "custom_agent_id",
            "custom_agent_name",
        ):
            metadata.pop(redundant_key, None)
```

Do not remove `custom_agent_warnings`; it is still useful when selected client tools or skills are unavailable.

- [x] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_agent_metadata.py tests/test_custom_agents_graph.py -v
```

Expected: all pass.

---

## Task 3: Metadata Cleanup Audit

> **Progress (done 2026-06-02):** Added the two preservation tests; both pass against the Task-2 cleanup (no code change needed in Step 2). Ran the ripgrep audit — confirmed expected findings: `custom_agent.py` still emits compat fields, `agent_metadata.py` reads them, `response_constants.py` strips them only when canonical `agent` exists, UI reads `agent`. Updated the `_augment_response_metadata()` comment to mark the fields as compatibility-only. Verification: `python -m pytest tests/test_agent_metadata.py tests/test_custom_agents_message_service.py tests/test_custom_agents_graph.py -v` → 28 passed.

**Files:**
- Modify: `app/core/response_constants.py`
- Modify: `app/ai/agents/custom_agent.py`
- Modify: tests listed below
- Test: `tests/test_agent_metadata.py`
- Test: `tests/test_custom_agents_message_service.py`

- [x] **Step 1: Write explicit cleanup tests**

Add to `tests/test_agent_metadata.py`:

```python
def test_build_bot_metadata_preserves_custom_agent_warnings():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="answer"),
        metadata={
            "agent": {
                "id": "custom_agent:abc",
                "kind": "custom",
                "name": "Data Analyst",
                "custom_agent_id": "abc",
                "source": "response",
            },
            "custom_agent_warnings": ["Selected client tool is unavailable."],
        },
    )

    metadata = build_bot_metadata(response)

    assert metadata["custom_agent_warnings"] == ["Selected client tool is unavailable."]


def test_build_bot_metadata_does_not_remove_legacy_fields_when_agent_missing():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="answer"),
        metadata={
            "runtime_agent_id": "custom_agent:abc",
            "custom_agent_name": "Data Analyst",
            "custom_agent_id": "abc",
        },
    )

    metadata = build_bot_metadata(response)

    assert metadata["runtime_agent_id"] == "custom_agent:abc"
```

This preserves backward compatibility if a response path has not yet attached canonical `agent`.

- [x] **Step 2: Run cleanup tests**

Run:

```powershell
python -m pytest tests/test_agent_metadata.py::test_build_bot_metadata_preserves_custom_agent_warnings tests/test_agent_metadata.py::test_build_bot_metadata_does_not_remove_legacy_fields_when_agent_missing -v
```

Expected: pass after Task 2 cleanup is implemented correctly.

- [x] **Step 3: Audit metadata keys with ripgrep**

Run:

```powershell
rg -n "runtime_agent_id|custom_agent_id|custom_agent_name|agent_name|subagent_results|message_metadata\\[\"agent\"\\]|\\.get\\(\"agent\"\\)" app tests demo.py
```

Expected findings:

- `custom_agent.py` may still emit compatibility fields.
- `agent_metadata.py` reads compatibility fields.
- `response_constants.py` removes compatibility fields only when canonical `agent` exists.
- UI reads `agent` first, compatibility fields second.

- [x] **Step 4: Update comments to mark compatibility fields**

In `app/ai/agents/custom_agent.py`, update `_augment_response_metadata()` comment:

```python
        # Compatibility fields consumed by the final metadata normalizer.
        # Persisted assistant messages should prefer message_metadata["agent"].
```

- [x] **Step 5: Run targeted regression**

Run:

```powershell
python -m pytest tests/test_agent_metadata.py tests/test_custom_agents_message_service.py tests/test_custom_agents_graph.py -v
```

Expected: all pass.

---

## Task 4: Verify And Fix Custom-Agent Handoff

> **Progress (done 2026-06-02):** No code fix required — the four base/custom handoff direction tests already pass, confirming `_apply_hand_off_if_present()` validates targets via `_is_attached_custom_agent()`. Extended the base→custom and custom→base tests with `context.handoff` source/target assertions. Verification: `python -m pytest tests/test_custom_agents_graph.py tests/test_graph_handoff_streaming.py -v` → 17 passed.

**Files:**
- Modify: `app/ai/graph.py`
- Test: `tests/test_custom_agents_graph.py`
- Test: `tests/test_graph_handoff_streaming.py`

- [x] **Step 1: Confirm existing handoff tests cover base/custom directions**

Run:

```powershell
python -m pytest tests/test_custom_agents_graph.py::test_base_agent_can_handoff_to_custom_agent tests/test_custom_agents_graph.py::test_custom_agent_can_handoff_to_base_agent tests/test_custom_agents_graph.py::test_custom_agent_can_handoff_to_another_attached_custom_agent tests/test_custom_agents_graph.py::test_handoff_to_unattached_custom_agent_is_refused_with_tool_error -v
```

Expected: all pass before or after this task. If any fail, fix `_apply_hand_off_if_present()` target validation using `_is_attached_custom_agent()`.

- [x] **Step 2: Add metadata assertion to existing handoff tests**

Extend `test_base_agent_can_handoff_to_custom_agent()`:

```python
    assert new_state["context"]["handoff"]["source_agent"] == "chat_agent"
    assert new_state["context"]["handoff"]["target_agent"] == rid
```

Extend `test_custom_agent_can_handoff_to_base_agent()`:

```python
    assert new_state["context"]["handoff"]["source_agent"] == rid
    assert new_state["context"]["handoff"]["target_agent"] == "search_agent"
```

- [x] **Step 3: Run handoff tests**

Run:

```powershell
python -m pytest tests/test_custom_agents_graph.py tests/test_graph_handoff_streaming.py -v
```

Expected: all pass.

---

## Task 5: Fix Planning Handoff To Attached Custom Agents

> **Progress (done 2026-06-02):** Replaced the base-agent-only branch in `_should_continue_planning()` with one that accepts base agents OR attached custom agents (via `_is_attached_custom_agent()`) and routes through `_route_target_for()` so custom ids map to the `custom_agent` node. Added `from uuid import uuid4` to the test file. Verification: new test + `tests/test_graph_handoff_streaming.py` → 3 passed; full `tests/test_graph_planning_subagents.py` → 37 passed (no base-agent routing regression).

**Files:**
- Modify: `app/ai/graph.py`
- Test: `tests/test_graph_planning_subagents.py`

- [x] **Step 1: Write failing Planning custom handoff test**

Add to `tests/test_graph_planning_subagents.py`:

Add this import near the existing imports if it is not already present:

```python
from uuid import uuid4
```

```python
def test_should_continue_planning_routes_handoff_to_custom_agent():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    rid = f"custom_agent:{uuid4()}"
    workflow.agents = {
        "planning_agent": object(),
        "chat_agent": object(),
        "search_agent": object(),
    }
    state = {
        "selected_agent": rid,
        "custom_agents": {
            rid: {
                "id": rid.split(":", 1)[1],
                "runtime_agent_id": rid,
                "name": "Data Analyst",
            }
        },
        "planning_call_count": 1,
        "planning_phase": "executing",
        "context": {},
        "messages": [],
    }

    assert workflow._should_continue_planning(state) == "custom_agent"
```

- [x] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest tests/test_graph_planning_subagents.py::test_should_continue_planning_routes_handoff_to_custom_agent -v
```

Expected: fails if `_should_continue_planning()` only accepts delegated base agents.

- [x] **Step 3: Route delegated agents through `_route_target_for()`**

In `app/ai/graph.py::_should_continue_planning()`, replace the base-agent-only branch:

```python
        if (
            isinstance(delegated_agent, str)
            and delegated_agent != "planning_agent"
            and delegated_agent in self.agents
        ):
            logger.info("Planning hand_off detected: routing planning_tools → %s", delegated_agent)
            return delegated_agent
```

With a branch that accepts existing base agents and attached custom agents, but still rejects unknown strings:

```python
        if isinstance(delegated_agent, str) and delegated_agent != "planning_agent":
            is_base_agent = delegated_agent in self.agents
            is_attached_custom = self._is_attached_custom_agent(state, delegated_agent)
            if is_base_agent or is_attached_custom:
                delegated_node = self._route_target_for(state, delegated_agent)
                logger.info(
                    "Planning hand_off detected: routing planning_tools to %s via %s",
                    delegated_agent,
                    delegated_node,
                )
                return delegated_node
```

- [x] **Step 4: Run Planning handoff tests**

Run:

```powershell
python -m pytest tests/test_graph_planning_subagents.py::test_should_continue_planning_routes_handoff_to_custom_agent tests/test_graph_handoff_streaming.py -v
```

Expected: all pass.

---

## Task 6: Planning Mode Custom-Agent Assignment

> **Progress (done 2026-06-02):** Extended `PlanningSubagentResult` with `agent_name`, `agent_kind`, `custom_agent_id`; imported `agent_identity`; computed `default_identity` (from task target) and `response_identity` (from the answering worker's response metadata) in `run_one()`, threading `identity=` into the completed/approval/error `_result()` calls so timeout/exception paths fall back to `default_identity`. Added a `normalize_subagent_metadata` pass in `_attach_planning_state_metadata()`. Confirmed `include_hand_off=False` still passed to isolated workers (graph.py:2744). Updated `_normalize_result()` to fall back to `agent_id` and preserve identity fields. Verification: target tests → 9 passed; broader regression `test_planning_subagents.py + test_demo_subagent_activity.py + test_graph_planning_subagents.py` → 106 passed (base-agent results now carry agent_name/agent_kind without breaking existing field-level assertions).

**Files:**
- Modify: `app/ai/planning_subagents.py`
- Modify: `app/ai/graph.py`
- Modify: `app/ui/subagent_activity.py`
- Test: `tests/test_custom_agents_planning.py`
- Test: `tests/test_message_service_subagent_streaming.py`

- [x] **Step 1: Write failing custom worker identity test**

Add to `tests/test_custom_agents_planning.py`:

```python
@pytest.mark.asyncio
async def test_custom_subagent_result_includes_display_identity():
    from app.ai.planning_subagents import (
        DispatchSubagentsInput,
        PlanningSubagentDispatcher,
        PlanningSubagentTask,
    )
    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole

    rid = f"custom_agent:{uuid4()}"

    class _Workflow:
        async def _run_agent_in_isolated_context(self, **kwargs):
            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id=rid,
                message=AgentMessage(role=MessageRole.ASSISTANT, content="worker answer"),
                metadata={
                    "runtime_agent_id": rid,
                    "custom_agent_id": rid.split(":", 1)[1],
                    "custom_agent_name": "Data Analyst",
                },
            )

    dispatcher = PlanningSubagentDispatcher(workflow=_Workflow())
    result = await dispatcher.dispatch(
        DispatchSubagentsInput(tasks=[PlanningSubagentTask(id="w1", agent=rid, task="analyze")]),
        parent_state={
            "custom_agents": {
                rid: {
                    "id": rid.split(":", 1)[1],
                    "runtime_agent_id": rid,
                    "name": "Data Analyst",
                }
            }
        },
    )

    entry = result.results[0]
    assert entry.agent == rid
    assert entry.agent_name == "Data Analyst"
    assert entry.custom_agent_id == rid.split(":", 1)[1]


@pytest.mark.asyncio
async def test_failed_custom_subagent_result_keeps_display_identity():
    from app.ai.planning_subagents import (
        DispatchSubagentsInput,
        PlanningSubagentDispatcher,
        PlanningSubagentTask,
    )

    rid = f"custom_agent:{uuid4()}"

    class _Workflow:
        async def _run_agent_in_isolated_context(self, **kwargs):
            raise RuntimeError("boom")

    dispatcher = PlanningSubagentDispatcher(workflow=_Workflow())
    result = await dispatcher.dispatch(
        DispatchSubagentsInput(tasks=[PlanningSubagentTask(id="w1", agent=rid, task="analyze")]),
        parent_state={
            "custom_agents": {
                rid: {
                    "id": rid.split(":", 1)[1],
                    "runtime_agent_id": rid,
                    "name": "Data Analyst",
                }
            }
        },
    )

    entry = result.results[0]
    assert entry.status == "failed"
    assert entry.agent == rid
    assert entry.agent_name == "Data Analyst"
    assert entry.custom_agent_id == rid.split(":", 1)[1]
```

- [x] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest tests/test_custom_agents_planning.py::test_custom_subagent_result_includes_display_identity tests/test_custom_agents_planning.py::test_failed_custom_subagent_result_keeps_display_identity -v
```

Expected: fails until `PlanningSubagentResult` has display identity fields.

- [x] **Step 3: Extend `PlanningSubagentResult`**

In `app/ai/planning_subagents.py`:

```python
    agent_name: str | None = Field(default=None)
    agent_kind: Literal["base", "custom"] | None = None
    custom_agent_id: str | None = None
```

- [x] **Step 4: Fill identity from helper**

Import:

```python
from .agent_metadata import agent_identity
```

In `PlanningSubagentDispatcher.run_one()`, compute a default identity before `_result()` so timeout/exception paths also get display metadata:

```python
        custom_agents = parent_state.get("custom_agents") if isinstance(parent_state, dict) else {}
        default_identity = agent_identity(task.agent, custom_agents)

        def _result(
            status: Literal["completed", "failed", "timeout", "requires_approval"],
            answer: str,
            error: str | None = None,
            artifacts: list[dict[str, Any]] | None = None,
            resolved_model: dict[str, Any] | None = None,
            identity: dict[str, Any] | None = None,
        ) -> PlanningSubagentResult:
            resolved_identity = identity or default_identity or {}
            return PlanningSubagentResult(
                id=task.id,
                agent=task.agent,
                agent_name=resolved_identity.get("name"),
                agent_kind=resolved_identity.get("kind"),
                custom_agent_id=resolved_identity.get("custom_agent_id"),
                status=status,
                elapsed_ms=int((time.perf_counter() - wall_start) * 1000),
                summary=_activity_summary_from_answer(answer),
                answer=answer,
                related_todo_ids=list(task.related_todo_ids),
                error=error,
                artifacts=list(artifacts or []),
                requested_model=requested_model,
                resolved_model=resolved_model,
            )
```

After the worker response is available, prefer the actual response identity for completed/approval/error results:

```python
        response_metadata = response.metadata or {}
        response_agent_id = response_metadata.get("runtime_agent_id") or response.agent_id or task.agent
        response_identity = agent_identity(
            response_agent_id,
            custom_agents,
            fallback_name=response_metadata.get("custom_agent_name"),
            custom_agent_id=response_metadata.get("custom_agent_id"),
        ) or default_identity
```

Pass `identity=response_identity` to each `_result()` call after `response` exists:

```python
            return _result(
                "completed",
                response.message.content or "",
                artifacts=worker_artifacts,
                resolved_model=resolved_model,
                identity=response_identity,
            )
```

- [x] **Step 5: Normalize Planning context results on final metadata**

In `app/ai/graph.py::_attach_planning_state_metadata()`, after copying `subagent_dispatches` and `subagent_results` from context into `response.metadata`, normalize `subagent_results` against the final state custom-agent map:

```python
        subagent_results = normalize_subagent_metadata(
            response.metadata.get("subagent_results"),
            state_view.custom_agents(),
        )
        if subagent_results:
            response.metadata["subagent_results"] = subagent_results
```

This keeps the existing `message_metadata["subagent_results"]` UI contract while adding `agent_name`, `agent_kind`, and `custom_agent_id`.

- [x] **Step 6: Keep subagent handoff isolated**

Confirm `_run_agent_in_isolated_context()` still passes:

```python
                include_hand_off=False,
```

This avoids conflict between Planning subagent dispatch and top-level custom-agent handoff. A Planning-dispatched custom worker should finish its isolated task instead of rerouting the parent graph.

- [x] **Step 7: Normalize UI activity**

In `app/ui/subagent_activity.py::_normalize_result()`, preserve identity fields:

```python
    agent = str(entry.get("agent") or entry.get("agent_id") or "unknown_agent").strip() or "unknown_agent"

    for key in ("agent_name", "agent_kind", "custom_agent_id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
```

- [x] **Step 8: Run tests**

Run:

```powershell
python -m pytest tests/test_custom_agents_planning.py tests/test_message_service_subagent_streaming.py -v
```

Expected: all pass.

---

## Task 7: Streamlit Live And Persisted Agent Display

> **Progress (done 2026-06-02):** Added `get_message_agent_label()` after `get_agent_display_name()`; added the `Worked by <agent>` caption in `render_message_bubble()` before the trace panel; updated both `agent_selected` streaming handlers (first-pass + resume) to prefer `event['agent_name']` then `get_agent_display_name`; updated BOTH subagent-row renderers (the tool-render-payload renderer ~L3172 and `_render_subagent_activity_view` ~L5893) to prefer `item['agent_name']`. Verification: `tests/test_demo_custom_agents.py tests/test_demo_subagent_activity.py` → 17 passed; `python -m py_compile demo.py` → OK. Design decisions: (1) the plan's two new label tests referenced a module-level `demo` that does not exist in this file — adapted them to the file's `_import_demo_with_ui_stubs(monkeypatch)` fixture pattern, assertions unchanged. (2) Updated two subagent-row sites rather than one, since both render the worker's agent identity.

**Files:**
- Modify: `demo.py`
- Test: `tests/test_demo_custom_agents.py`
- Test: `tests/test_demo_subagent_activity.py`

- [x] **Step 1: Write failing UI helper tests**

Add to `tests/test_demo_custom_agents.py`:

```python
def test_message_agent_label_prefers_canonical_agent_metadata():
    metadata = {
        "agent": {
            "id": "custom_agent:abc",
            "kind": "custom",
            "name": "Data Analyst",
            "custom_agent_id": "abc",
            "source": "response",
        }
    }

    assert demo.get_message_agent_label(metadata) == "Data Analyst"


def test_message_agent_label_falls_back_to_legacy_custom_fields():
    metadata = {
        "runtime_agent_id": "custom_agent:abc",
        "custom_agent_name": "Data Analyst",
    }

    assert demo.get_message_agent_label(metadata) == "Data Analyst"
```

- [x] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_demo_custom_agents.py::test_message_agent_label_prefers_canonical_agent_metadata tests/test_demo_custom_agents.py::test_message_agent_label_falls_back_to_legacy_custom_fields -v
```

Expected: fails because `get_message_agent_label()` does not exist.

- [x] **Step 3: Add label helper in `demo.py`**

Add near `get_agent_display_name()`:

```python
def get_message_agent_label(message_metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(message_metadata, dict):
        return None
    agent = message_metadata.get("agent")
    if isinstance(agent, dict):
        name = agent.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        agent_id = agent.get("id")
        if isinstance(agent_id, str) and agent_id.strip():
            return get_agent_display_name(agent_id.strip())

    legacy_name = message_metadata.get("custom_agent_name")
    if isinstance(legacy_name, str) and legacy_name.strip():
        return legacy_name.strip()

    legacy_id = message_metadata.get("runtime_agent_id")
    if isinstance(legacy_id, str) and legacy_id.strip():
        return get_agent_display_name(legacy_id.strip())
    return None
```

- [x] **Step 4: Show persisted worked-by label**

In the message rendering path where `message_metadata = get_message_metadata(msg)` is available, add a compact caption before trace panels:

```python
    agent_label = get_message_agent_label(message_metadata)
    if agent_label:
        st.caption(f"Worked by {agent_label}")
```

Use the existing message container style; do not add a large card.

- [x] **Step 5: Show live worked-by label from stream events**

In both streaming loops handling `event_type == "agent_selected"`, replace raw formatting:

```python
selected_agent = event.get("agent", "unknown")
display_name = event.get("agent_name") or get_agent_display_name(selected_agent)
status.update(label=f"{display_name} is processing...", state="running")
```

Also keep:

```python
st.session_state.stream_selected_agent = selected_agent
```

- [x] **Step 6: Show custom subagent names in activity panel**

Where Streamlit renders subagent rows, prefer:

```python
agent_name = item.get("agent_name") or get_agent_display_name(str(item.get("agent") or "unknown_agent"))
```

- [x] **Step 7: Run UI tests**

Run:

```powershell
python -m pytest tests/test_demo_custom_agents.py tests/test_demo_subagent_activity.py -v
python -m py_compile demo.py
```

Expected: tests pass and `demo.py` compiles.

---

## Task 8: Message Service Paths And Resume/Interrupt Coverage

> **Progress (done 2026-06-02):** Added the static `_attach_selected_agent_metadata()` helper next to `_agent_selected_event()`. Preserved agent metadata on both partial-message branches (user_requested stop + first-pass disconnect) via the metadata-variable pattern. Extended `_persist_interrupt_bot_message()` with `selected_agent`/`custom_agents` params and wired the `_attach_selected_agent_metadata()` call before persistence. Threaded the selection through the first-pass streaming interrupt (`inflight.selected_agent` + `workflow_request.custom_agents`), the resume path (new `resume_selected_agent` + `resume_custom_agents`, reused for the `agent_selected` event and the post-resume interrupt persist), and the non-stream create-message interrupt path (changed `_execute_user_message_workflow` to return the `bot_response` instead of `None`, then passed `bot_response.agent_id` / `workflow_request.custom_agents`). Verification: `tests/test_custom_agents_message_service.py tests/test_message_service_subagent_streaming.py` → 10 passed; `tests/test_message_history_pipeline.py` → 8 passed (no regression from the signature/return changes). Design note: returning `bot_response` on interrupt is safe because the create_message flow `return`s the user_message_read before reaching `_persist_completed_workflow_response`, so the completed-response path is untouched.

**Files:**
- Modify: `app/services/message_service.py`
- Test: `tests/test_custom_agents_message_service.py`
- Test: `tests/test_message_service_subagent_streaming.py`

- [x] **Step 1: Add tests for message-service selected-agent metadata helper**

Add to `tests/test_custom_agents_message_service.py`:

```python
def test_agent_selected_event_still_adds_name_for_custom_only():
    from app.services.message_service import MessageService

    rid = f"custom_agent:{uuid4()}"
    custom = MessageService._agent_selected_event(rid, {rid: {"name": "Analyst"}})

    assert custom == {
        "type": "agent_selected",
        "agent": rid,
        "agent_name": "Analyst",
    }
    assert MessageService._agent_selected_event("chat_agent", {rid: {"name": "Analyst"}}) == {
        "type": "agent_selected",
        "agent": "chat_agent",
    }


def test_attach_selected_agent_metadata_adds_canonical_custom_agent():
    from app.services.message_service import MessageService

    rid = f"custom_agent:{uuid4()}"
    metadata = {}

    MessageService._attach_selected_agent_metadata(
        metadata,
        rid,
        {rid: {"id": rid.split(":", 1)[1], "runtime_agent_id": rid, "name": "Analyst"}},
    )

    assert metadata["agent"] == {
        "id": rid,
        "kind": "custom",
        "name": "Analyst",
        "custom_agent_id": rid.split(":", 1)[1],
        "source": "response",
    }


def test_attach_selected_agent_metadata_noops_without_selected_agent():
    from app.services.message_service import MessageService

    metadata = {"stopped": True}

    MessageService._attach_selected_agent_metadata(metadata, None, {})

    assert metadata == {"stopped": True}
```

- [x] **Step 2: Add a small message-service helper**

Add this static method near `_agent_selected_event()` in `app/services/message_service.py`:

```python
    @staticmethod
    def _attach_selected_agent_metadata(
        metadata: dict[str, Any],
        selected_agent: str | None,
        custom_agents: dict[str, Any] | None,
    ) -> None:
        if not selected_agent:
            return

        from app.ai.agent_metadata import attach_agent_metadata

        attach_agent_metadata(
            metadata,
            response_agent_id=selected_agent,
            selected_agent_id=selected_agent,
            custom_agents=custom_agents,
        )
```

- [x] **Step 3: Preserve agent metadata on stopped partial messages**

In the user-requested stop branch, replace the inline metadata literal with a variable before calling `_create_bot_response_message()`:

```python
metadata = {
    "stopped": True,
    "partial": True,
    "stop_reason": "user_requested",
    "persona_used": sanitized_persona,
    "reply_to_user_message_id": str(user_message_id),
}
self._attach_selected_agent_metadata(
    metadata,
    inflight.selected_agent,
    workflow_request.custom_agents,
)
bot_message = self._create_bot_response_message(
    conversation_id=message_create_data.conversation_id,
    content=partial,
    metadata=metadata,
    message_id=bot_message_id,
)
```

Apply the same pattern to the first-pass disconnect partial branch that uses `stop_reason: "disconnect"`.

- [x] **Step 4: Preserve agent metadata on interrupt placeholders**

Extend `_persist_interrupt_bot_message()` so persisted paused messages can receive selected-agent state:

```python
        selected_agent: str | None = None,
        custom_agents: dict[str, Any] | None = None,
```

Inside `_persist_interrupt_bot_message()`, after the base `metadata` dict and before `_create_bot_response_message()`:

```python
        self._attach_selected_agent_metadata(metadata, selected_agent, custom_agents)
```

For the first-pass streaming interrupt call, pass the in-flight selection:

```python
                            "message": self._persist_interrupt_bot_message(
                                conversation_id=message_create_data.conversation_id,
                                interrupt_payload=interrupt_response,
                                sanitized_persona=sanitized_persona,
                                pending_tool_calls=event.get("pending_tool_calls"),
                                thread_id=event.get("thread_id")
                                or str(message_create_data.conversation_id),
                                next_nodes=event.get("next"),
                                user_id=resolved_user_id,
                                message_id=bot_message_id,
                                tool_artifacts=stream_tool_artifacts or None,
                                selected_agent=inflight.selected_agent,
                                custom_agents=workflow_request.custom_agents,
                            ).model_dump(mode="json"),
```

For resume streaming, keep a `resume_selected_agent` variable and resolved custom-agent map:

```python
        resume_selected_agent: str | None = None
        resume_custom_agents = self._resolve_custom_agents_state(user_id, conversation_id)
```

Update it on `agent_selected` and reuse the same map for the event:

```python
                if event_type == "agent_selected":
                    resume_selected_agent = event.get("agent")
                    yield self._agent_selected_event(
                        resume_selected_agent,
                        resume_custom_agents,
                    )
```

Pass it when persisting a new interrupt after resume:

```python
                    persisted = self._persist_interrupt_bot_message(
                        conversation_id=conversation_id,
                        interrupt_payload=normalized_interrupt,
                        sanitized_persona=sanitized_persona,
                        pending_tool_calls=event.get("pending_tool_calls"),
                        thread_id=event.get("thread_id") or thread_id,
                        next_nodes=event.get("next"),
                        user_id=user_id,
                        message_id=bot_message_id,
                        tool_artifacts=resume_tool_artifacts or None,
                        selected_agent=resume_selected_agent,
                        custom_agents=resume_custom_agents,
                    )
```

For the non-stream create-message interrupt path, keep the workflow response instead of discarding it:

```python
        if bot_response and bot_response.metadata and "interrupt" in bot_response.metadata:
            self._set_plan_lifecycle(
                conversation_id,
                user_id,
                PlanLifecycle.paused,
            )
            return (
                bot_response,
                bot_response.metadata["interrupt"],
            )
```

Then pass its selected agent when persisting the interrupt message:

```python
                    self._persist_interrupt_bot_message(
                        conversation_id=message_create_data.conversation_id,
                        interrupt_payload=interrupt_payload,
                        sanitized_persona=sanitized_persona,
                        thread_id=getattr(interrupt_payload, "thread_id", None),
                        pending_tool_calls=(
                            [
                                r.model_dump(mode="json")
                                for r in getattr(interrupt_payload, "action_requests", [])
                            ]
                            if isinstance(interrupt_payload, InterruptResponse)
                            else None
                        ),
                        user_id=resolved_user_id,
                        selected_agent=bot_response.agent_id if bot_response else None,
                        custom_agents=workflow_request.custom_agents,
                    )
```

- [x] **Step 5: Run message service tests**

Run:

```powershell
python -m pytest tests/test_custom_agents_message_service.py tests/test_message_service_subagent_streaming.py -v
```

Expected: all pass.

---

## Task 9: Verification

> **Progress (done 2026-06-02):** All verification steps pass. Step 1 metadata → 8 passed; Step 2 graph+planning → 61 passed; Step 3 message-service+streaming → 18 passed; Step 4 demo → 17 passed + `demo.py` compiles; Step 5 focused custom-agent suite → 74 passed. Broader regression across affected modules (planning_subagents, multi_sidecar_hardening, demo rich/stream/plan, context/model metadata, graph streaming, rag artifacts) → 181 passed. **Full suite: 958 passed, 1 failed.** The single failure — `tests/client_backend/test_live_server_integration.py::test_live_document_upload_list_get_task_and_delete_flow` (`KeyError: 'document'` after a `local_mcp_manager` tool-load error) — is PRE-EXISTING and unrelated: confirmed it fails identically on a clean source tree (changes stashed) and it is a live-server document-upload integration test that touches none of the files in this plan.

**Files:**
- Existing tests only.

- [x] **Step 1: Run metadata tests**

Run:

```powershell
python -m pytest tests/test_agent_metadata.py -v
```

Expected: all pass.

- [x] **Step 2: Run custom-agent graph and planning tests**

Run:

```powershell
python -m pytest tests/test_custom_agents_graph.py tests/test_custom_agents_planning.py tests/test_graph_handoff_streaming.py tests/test_graph_planning_subagents.py -v
```

Expected: all pass.

- [x] **Step 3: Run message service and streaming tests**

Run:

```powershell
python -m pytest tests/test_custom_agents_message_service.py tests/test_message_service_subagent_streaming.py tests/test_message_history_pipeline.py -v
```

Expected: all pass.

- [x] **Step 4: Run Streamlit tests and compile**

Run:

```powershell
python -m pytest tests/test_demo_custom_agents.py tests/test_demo_subagent_activity.py -v
python -m py_compile demo.py
```

Expected: tests pass and `demo.py` compiles.

- [x] **Step 5: Run focused full custom-agent suite**

Run:

```powershell
python -m pytest tests/test_custom_agents_service.py tests/test_custom_agents_api.py tests/test_custom_agents_tools.py tests/test_custom_agents_graph.py tests/test_custom_agents_planning.py tests/test_custom_agents_message_service.py -v
```

Expected: all pass.

---

## Manual Verification

1. Attach a custom agent named `Data Analyst` to a conversation.
2. Ask it a direct question.
3. Confirm live Streamlit status says `Data Analyst is processing...`.
4. Confirm the persisted assistant message shows `Worked by Data Analyst`.
5. Confirm the assistant message metadata contains `agent.id == "custom_agent:<uuid>"`.
6. Trigger a handoff from a custom agent to `search_agent`.
7. Confirm the final message metadata contains `handoff.from_agent_id` and `handoff.to_agent_id`.
8. In Planning mode, dispatch a task to the custom agent.
9. Confirm persisted metadata contains `subagent_results[0].agent_name == "Data Analyst"`.

---

## Production Readiness Checks

- The canonical field is `message_metadata["agent"]`.
- Streamlit reads `agent` first and legacy custom fields second.
- Compatibility custom-agent fields are removed from final metadata only when `agent` exists.
- `custom_agent_warnings` remain available.
- No prompt-history/context injection is added.
- No LangSmith API dependency is introduced.
- Planning top-level handoff validates delegated targets before routing attached custom agents through `_route_target_for()`.
- Planning subagent workers keep `include_hand_off=False` so subagent dispatch does not conflict with top-level orchestration.
- Planning worker display identity enriches the existing `subagent_results` field; no parallel `subagents` metadata field is introduced.
