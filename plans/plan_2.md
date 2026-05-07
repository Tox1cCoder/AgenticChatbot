# DeepAgents Gap Follow-up And RAG Artifact Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RAG retrieval evidence visible and durable in the Streamlit demo, then add the DeepAgents-inspired capabilities that are valuable for this repo: large tool-result offloading, one-shot context-overflow retry, optional user memory tools, and HITL `respond`.

**Architecture:** Keep the existing LangGraph orchestrator, sidecar MCP execution, and `hand_off` routing. Add narrow backend services and UI helpers around the current graph instead of migrating to DeepAgents. Treat RAG `search_documents` outputs as first-class tool artifacts, expose them consistently through workflow responses and persisted message metadata, and render them in `demo.py` as retrieved evidence rather than hiding them inside generic trace internals.

**Tech Stack:** Python 3.10+, FastAPI, LangGraph, LangChain, SQLAlchemy, Alembic, Streamlit, pytest, pytest-asyncio, Redis-backed client runtime store, MCP tool catalogs.

---

## References Checked

- Deep Agents overview: `https://docs.langchain.com/oss/python/deepagents/overview`
- Deep Agents context engineering: `https://docs.langchain.com/oss/python/deepagents/context-engineering`
- Deep Agents human-in-the-loop: `https://docs.langchain.com/oss/python/deepagents/human-in-the-loop`
- Deep Agents memory: `https://docs.langchain.com/oss/python/deepagents/memory`
- Deep Agents profiles: `https://docs.langchain.com/oss/python/deepagents/profiles`

These references are used only to decide which DeepAgents capabilities are worth copying into the current LangGraph architecture. The plan intentionally does not migrate the chatbot to `create_deep_agent`.

---

## Current Audit

### RAG Artifact Path

- `app/ai/graph.py::_rag_tools_node()` creates `search_documents` tool artifacts and stores them in `state["context"]["tool_artifacts"]`.
- `app/ai/graph.py::_recover_terminal_response()` now calls `_attach_context_outputs()`, which can attach context artifacts to the final `AgentResponse`.
- `app/ai/graph.py::_rag_node()` still differs from other agent nodes: it assigns `state["response"] = response` directly and does not call `_merge_tool_artifacts()` before appending the assistant message.
- `tests/test_rag_tool_loop_finalization.py::test_rag_document_tool_results_are_recorded_as_response_artifacts` currently checks `_rag_tools_node()` plus `_recover_terminal_response()`, but it does not check `_rag_node()`, `AIService`, message persistence, or `demo.py`.

### Streamlit Demo Path

- `demo.py` can render message metadata `tool_artifacts` through `render_message_trace()`.
- `render_message_trace()` shows artifacts under a generic `Thought Process -> Tool Activity` expander.
- `render_tool_artifacts()` is defined but has no call sites.
- There is no RAG-specific "Retrieved Evidence" section, so chunk/document outputs are easy to miss.
- During live streaming, `app/ai/graph.py` emits `tool_end` for only the last `ToolMessage` in a node update. A RAG tools node can append multiple `ToolMessage` objects in one update, so earlier `search_documents` chunk results can be absent from the live trace until message history is reloaded.

### DeepAgents-Inspired Gaps

- Filesystem capability through Desktop Commander MCP is acceptable for now; no general virtual filesystem is required.
- Large tool results are truncated for the model and artifacts, but the full output is not reliably preserved.
- Conversation history summarization exists, but there is no automatic retry when a single turn exceeds model context because of tool output.
- `hand_off` is graph rerouting, not context-isolated subagents. Leave it unchanged.
- Async/parallel subagents, declarative filesystem permissions, sandbox execution, and provider profiles are not part of this implementation.
- Agent-editable long-term memory is not present. It is useful only if scoped, explicit, visible, and disabled by default.
- HITL supports `approve`, `edit`, and `reject`, but not `respond`.

## Non-Scope

- Do not migrate to DeepAgents.
- Do not replace Desktop Commander MCP.
- Do not add true subagents or parallel subagent workstreams.
- Do not add a declarative filesystem permission layer.
- Do not add sandbox execution.
- Do not add DeepAgents profiles/middleware.
- Do not rewrite RAG retrieval, indexing, OCR, Qdrant filtering, or document upload.

## Files To Modify

- `app/ai/graph.py`
- `app/ai/tool_execution.py`
- `app/ai/agents/base_agent.py`
- `app/ai/agents/rag_agent.py`
- `app/core/config.py`
- `app/core/container.py`
- `app/core/response_constants.py`
- `app/services/message_service.py`
- `app/services/ai_service.py`
- `app/schemas/workflow.py`
- `app/ai/schemas.py`
- `app/ai/utils.py`
- `app/models/tool_approval.py`
- `demo.py`

## Files To Add

- `app/ui/__init__.py`
- `app/ui/rag_artifacts.py`
- `app/ai/context_overflow.py`
- `app/models/tool_result_blob.py`
- `app/repositories/tool_result_blob.py`
- `app/services/tool_result_blob_service.py`
- `app/api/tool_result_blobs.py`
- `app/models/user_memory.py`
- `app/repositories/user_memory.py`
- `app/ai/user_memory_tools.py`
- Alembic migration for `tool_result_blobs`
- Alembic migration for `user_memories`
- Alembic migration extending HITL decision enums with `respond`
- `tests/test_rag_artifact_visibility.py`
- `tests/test_graph_streaming_tool_events.py`
- `tests/test_demo_rag_artifacts.py`
- `tests/test_tool_result_blob_service.py`
- `tests/test_context_overflow_retry.py`
- `tests/test_user_memory_tools.py`
- `tests/test_hitl_respond_decision.py`

---

## Task 1: RAG Artifact Backend Visibility

**Files:**
- Modify: `app/ai/graph.py`
- Test: `tests/test_rag_artifact_visibility.py`

- [x] **Step 1: Write a failing test that `_rag_node()` carries existing RAG artifacts into the response**

Create `tests/test_rag_artifact_visibility.py`:

```python
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage, ToolMessage

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


class FakeRAGAgent:
    async def process_message(self, message, conversation_id):
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="The retrieved chunks show revenue increased.",
            ),
            metadata={"agentic_mode": True},
        )


def _make_workflow():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.rag_agent = FakeRAGAgent()
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow._get_state_attachments = lambda _state: []
    return workflow


def test_rag_node_merges_existing_search_document_artifacts_into_response():
    workflow = _make_workflow()
    artifact = {
        "tool_call_id": "chunk-call",
        "tool": "search_documents",
        "args": {"action": "search_chunks", "query": "revenue"},
        "output": "SEARCH RESULTS:\n\n[1] report.pdf\nchunk evidence",
        "error": None,
        "status": "success",
    }
    state = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {"tool_artifacts": [artifact]},
        "messages": [
            HumanMessage(content="What changed in revenue?"),
            ToolMessage(
                content="SEARCH RESULTS:\n\n[1] report.pdf\nchunk evidence",
                tool_call_id="chunk-call",
                name="search_documents",
            ),
        ],
    }

    asyncio.run(workflow._rag_node(state))

    assert state["response"].tool_artifacts == [artifact]
    assert state["response"].message.content == "The retrieved chunks show revenue increased."
```

- [x] **Step 2: Run the failing test**

Run:

```powershell
python -m pytest tests\test_rag_artifact_visibility.py::test_rag_node_merges_existing_search_document_artifacts_into_response -q
```

Expected result before implementation:

- Fails because `_rag_node()` does not merge `state["context"]["tool_artifacts"]` into `state["response"]`.

- [x] **Step 3: Merge tool artifacts inside `_rag_node()`**

Modify `app/ai/graph.py::_rag_node()` after `response = await self.rag_agent.process_message(...)`:

```python
        response = await self.rag_agent.process_message(agent_msg, conversation_id)
        response = self._finalize_forced_final_response(state, response)
        self._merge_tool_artifacts(state, response)
        state["response"] = response
```

Keep the existing `AIMessage` append logic unchanged.

- [x] **Step 4: Verify the focused backend artifact test passes**

Run:

```powershell
python -m pytest tests\test_rag_artifact_visibility.py -q
```

Expected result:

- The new RAG artifact visibility test passes.

- [x] **Step 5: Run adjacent RAG graph tests**

Run:

```powershell
python -m pytest tests\test_rag_tool_loop_finalization.py tests\test_graph_tool_budget.py -q
```

Expected result:

- Existing RAG finalization and graph-budget tests pass.

---

## Task 2: Emit Every RAG Tool Result In Streaming

**Files:**
- Modify: `app/ai/graph.py`
- Test: `tests/test_graph_streaming_tool_events.py`

- [x] **Step 1: Add a failing helper-level streaming test**

Create `tests/test_graph_streaming_tool_events.py`:

```python
from langchain_core.messages import ToolMessage

from app.ai.graph import MultiAgentWorkflow


def test_tool_end_events_include_every_tool_message_from_one_node_update():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    emitted: set[str] = set()
    node_state = {
        "context": {
            "tool_render_results": {
                "chunk-call": {"type": "text", "text": "chunk evidence"},
                "doc-call": {"type": "text", "text": "full document text"},
            }
        },
        "messages": [
            ToolMessage(
                content="SEARCH RESULTS:\n\nchunk evidence",
                tool_call_id="chunk-call",
                name="search_documents",
            ),
            ToolMessage(
                content="DOCUMENT CONTENT:\n\nfull document text",
                tool_call_id="doc-call",
                name="search_documents",
            ),
        ],
    }

    events = list(
        workflow._tool_end_events_from_node_state(
            node_state=node_state,
            last_state_values=node_state,
            emitted_tool_result_ids=emitted,
        )
    )

    assert [event["tool_call_id"] for event in events] == ["chunk-call", "doc-call"]
    assert events[0]["render"]["text"] == "chunk evidence"
    assert events[1]["render"]["text"] == "full document text"


def test_tool_end_events_are_deduplicated_by_tool_call_id():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    emitted = {"chunk-call"}
    node_state = {
        "messages": [
            ToolMessage(
                content="SEARCH RESULTS:\n\nchunk evidence",
                tool_call_id="chunk-call",
                name="search_documents",
            ),
        ],
    }

    events = list(
        workflow._tool_end_events_from_node_state(
            node_state=node_state,
            last_state_values=node_state,
            emitted_tool_result_ids=emitted,
        )
    )

    assert events == []
```

- [x] **Step 2: Run the failing streaming helper tests**

Run:

```powershell
python -m pytest tests\test_graph_streaming_tool_events.py -q
```

Expected result before implementation:

- Fails because `_tool_end_events_from_node_state()` does not exist.

- [x] **Step 3: Add `_tool_end_events_from_node_state()` to `MultiAgentWorkflow`**

Add this method near `_lookup_tool_render_payload()` in `app/ai/graph.py`:

```python
    def _tool_end_events_from_node_state(
        self,
        *,
        node_state: dict[str, Any],
        last_state_values: dict[str, Any] | None,
        emitted_tool_result_ids: set[str],
    ):
        messages = node_state.get("messages", [])
        if not isinstance(messages, list):
            messages = [messages]

        for message in messages:
            if not isinstance(message, ToolMessage):
                continue

            tool_call_id = getattr(message, "tool_call_id", None)
            dedupe_key = str(tool_call_id or f"{getattr(message, 'name', 'unknown')}:{id(message)}")
            if dedupe_key in emitted_tool_result_ids:
                continue
            emitted_tool_result_ids.add(dedupe_key)

            event_payload = {
                "type": "tool_end",
                "name": getattr(message, "name", "unknown"),
                "tool_call_id": tool_call_id,
                "result": make_json_safe(message.content),
            }
            render_payload = self._lookup_tool_render_payload(
                last_state_values,
                tool_call_id,
            ) or self._lookup_tool_render_payload(node_state, tool_call_id)
            if render_payload:
                event_payload["render"] = render_payload
            yield event_payload
```

- [x] **Step 4: Use the helper in both stream loops**

In `execute_request_stream()` and `resume_with_decisions_stream()`:

1. Add this set beside `emitted_tool_call_ids`:

```python
        emitted_tool_result_ids = set()
```

2. Replace the `elif isinstance(last_msg, ToolMessage): ... yield event_payload` block with:

```python
                                        for event_payload in self._tool_end_events_from_node_state(
                                            node_state=node_state,
                                            last_state_values=last_state_values,
                                            emitted_tool_result_ids=emitted_tool_result_ids,
                                        ):
                                            yield event_payload
```

- [x] **Step 5: Verify streaming helper tests pass**

Run:

```powershell
python -m pytest tests\test_graph_streaming_tool_events.py -q
```

Expected result:

- Both streaming helper tests pass.

- [x] **Step 6: Verify existing streaming and RAG tests**

Run:

```powershell
python -m pytest tests\test_graph_streaming_summarization.py tests\test_rag_tool_loop_finalization.py -q
```

Expected result:

- No new failures from streaming event changes. Existing baseline failures in graph streaming fixtures must be recorded if still present before this task starts.

---

## Task 3: Render RAG Retrieved Evidence In `demo.py`

**Files:**
- Create: `app/ui/__init__.py`
- Create: `app/ui/rag_artifacts.py`
- Modify: `demo.py`
- Test: `tests/test_demo_rag_artifacts.py`

- [x] **Step 1: Add pure RAG artifact extraction tests**

Create `tests/test_demo_rag_artifacts.py`:

```python
from app.ui.rag_artifacts import extract_rag_artifact_views


def test_extract_rag_artifact_views_returns_search_documents_outputs():
    metadata = {
        "tool_artifacts": [
            {
                "tool_call_id": "chunk-call",
                "tool": "search_documents",
                "args": {"action": "search_chunks", "query": "revenue"},
                "output": "SEARCH RESULTS:\n\n[1] report.pdf\nchunk evidence",
                "status": "success",
            },
            {
                "tool_call_id": "other-call",
                "tool": "widget_create",
                "args": {},
                "output": "widget",
                "status": "success",
            },
        ]
    }

    views = extract_rag_artifact_views(metadata)

    assert len(views) == 1
    assert views[0].tool_call_id == "chunk-call"
    assert views[0].action == "search_chunks"
    assert views[0].title == "Search Chunks"
    assert "chunk evidence" in views[0].output
    assert views[0].preview.endswith("chunk evidence")
```

- [x] **Step 2: Run the failing pure helper test**

Run:

```powershell
python -m pytest tests\test_demo_rag_artifacts.py -q
```

Expected result before implementation:

- Fails because `app.ui.rag_artifacts` does not exist.

- [x] **Step 3: Create the pure helper module**

Create `app/ui/__init__.py` as an empty package marker.

Create `app/ui/rag_artifacts.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_ACTION_TITLES = {
    "scan_all": "Scan All Documents",
    "list_documents": "List Documents",
    "search_chunks": "Search Chunks",
    "read_document": "Read Document",
    "grep_document": "Grep Document",
    "view_images": "View Images",
}


@dataclass(frozen=True)
class RAGArtifactView:
    tool_call_id: str
    action: str
    title: str
    query: str | None
    document_id: str | None
    status: str
    output: str
    preview: str


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _preview(value: str, max_chars: int = 1200) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def extract_rag_artifact_views(metadata: dict[str, Any] | None) -> list[RAGArtifactView]:
    if not isinstance(metadata, dict):
        return []

    raw_artifacts = metadata.get("tool_artifacts")
    if not isinstance(raw_artifacts, list):
        return []

    views: list[RAGArtifactView] = []
    for index, artifact in enumerate(raw_artifacts, start=1):
        if not isinstance(artifact, dict):
            continue
        if artifact.get("tool") != "search_documents":
            continue

        args = artifact.get("args") if isinstance(artifact.get("args"), dict) else {}
        action = _string_value(args.get("action") or "search_chunks").strip() or "search_chunks"
        output = _string_value(
            artifact.get("output")
            if artifact.get("output") is not None
            else artifact.get("result")
        )
        if not output.strip() and artifact.get("error"):
            output = _string_value(artifact.get("error"))

        views.append(
            RAGArtifactView(
                tool_call_id=_string_value(artifact.get("tool_call_id") or f"rag-artifact-{index}"),
                action=action,
                title=_ACTION_TITLES.get(action, action.replace("_", " ").title()),
                query=_string_value(args.get("query")).strip() or None,
                document_id=_string_value(args.get("document_id")).strip() or None,
                status=_string_value(artifact.get("status") or "success").strip() or "success",
                output=output,
                preview=_preview(output),
            )
        )

    return views
```

- [x] **Step 4: Add a RAG evidence renderer to `demo.py`**

Add the import near the other app imports:

```python
from app.ui.rag_artifacts import RAGArtifactView, extract_rag_artifact_views
```

Add this function near `render_message_trace()`:

```python
def render_rag_retrieval_artifacts(message_metadata: dict[str, Any]) -> None:
    views = extract_rag_artifact_views(message_metadata)
    if not views:
        return

    label = "Retrieved Evidence"
    with st.expander(f"{label} ({len(views)} result{'s' if len(views) != 1 else ''})", expanded=False):
        for index, view in enumerate(views, start=1):
            st.markdown(f"**[{index}] {view.title}**")
            detail_parts = []
            if view.query:
                detail_parts.append(f"query: `{view.query}`")
            if view.document_id:
                detail_parts.append(f"document: `{view.document_id}`")
            if view.status:
                detail_parts.append(f"status: `{view.status}`")
            if detail_parts:
                st.caption(" | ".join(detail_parts))

            if view.preview:
                st.code(view.preview, language="text")
            if view.output and view.output != view.preview:
                with st.expander(f"Full output for {view.title}", expanded=False):
                    st.code(view.output, language="text")
```

Call it in `render_message_bubble()` for assistant messages immediately after `st.markdown(content_text)`:

```python
        st.markdown(content_text)

        if not is_user:
            render_rag_retrieval_artifacts(message_metadata)
```

Remove the old duplicate `st.markdown(content_text)` line instead of rendering the content twice.

- [x] **Step 5: Verify helper tests and demo compilation**

Run:

```powershell
python -m pytest tests\test_demo_rag_artifacts.py -q
python -m py_compile demo.py app\ui\rag_artifacts.py
```

Expected result:

- Helper tests pass.
- `demo.py` and `app/ui/rag_artifacts.py` compile successfully.

---

## Task 4: Large Tool-Result Offloading

**Files:**
- Create: `app/models/tool_result_blob.py`
- Create: `app/repositories/tool_result_blob.py`
- Create: `app/services/tool_result_blob_service.py`
- Create: `app/api/tool_result_blobs.py`
- Modify: `app/core/config.py`
- Modify: `app/core/container.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/graph.py`
- Modify: `demo.py`
- Test: `tests/test_tool_result_blob_service.py`

- [x] **Step 1: Add service tests for offloading large tool results**

Create `tests/test_tool_result_blob_service.py`:

```python
from uuid import uuid4

from app.services.tool_result_blob_service import ToolResultBlobService


class FakeRepository:
    def __init__(self):
        self.created = []

    def create(self, data):
        record = {"id": uuid4(), **data}
        self.created.append(record)
        return record


def test_offload_if_large_returns_inline_for_small_output(tmp_path):
    repo = FakeRepository()
    service = ToolResultBlobService(repo, storage_root=tmp_path, threshold_chars=20)

    result = service.offload_if_large(
        conversation_id=uuid4(),
        user_id=uuid4(),
        tool_call_id="call-1",
        tool_name="search_documents",
        output_text="short result",
    )

    assert result["output"] == "short result"
    assert result["blob_id"] is None
    assert repo.created == []


def test_offload_if_large_writes_full_output_and_returns_preview(tmp_path):
    repo = FakeRepository()
    service = ToolResultBlobService(repo, storage_root=tmp_path, threshold_chars=10)
    conversation_id = uuid4()
    user_id = uuid4()

    result = service.offload_if_large(
        conversation_id=conversation_id,
        user_id=user_id,
        tool_call_id="call-1",
        tool_name="search_documents",
        output_text="abcdefghijklmnopqrstuvwxyz",
    )

    assert result["output"] == "abcdefghij\n\n[Output offloaded: use blob_id to read the full result.]"
    assert result["blob_id"]
    assert result["size_bytes"] == 26
    assert repo.created[0]["conversation_id"] == conversation_id
    assert repo.created[0]["user_id"] == user_id
    assert repo.created[0]["tool_call_id"] == "call-1"
    assert (tmp_path / repo.created[0]["storage_path"]).read_text(encoding="utf-8") == "abcdefghijklmnopqrstuvwxyz"
```

- [x] **Step 2: Run the failing service tests**

Run:

```powershell
python -m pytest tests\test_tool_result_blob_service.py -q
```

Expected result before implementation:

- Fails because `ToolResultBlobService` does not exist.

- [x] **Step 3: Add config settings**

Modify `app/core/config.py`:

```python
    tool_result_offload_enabled: bool = Field(
        default=True,
        description="Persist full large tool outputs outside model-visible ToolMessages and return a preview plus blob_id.",
    )
    tool_result_offload_threshold_chars: int = Field(
        default=16000,
        description="Character count above which full tool output is offloaded.",
    )
    tool_result_offload_preview_chars: int = Field(
        default=4000,
        description="Preview characters kept inline after a tool result is offloaded.",
    )
    tool_result_blob_storage_dir: str = Field(
        default="data/tool_result_blobs",
        description="Directory for full offloaded tool result payloads.",
    )
```

- [x] **Step 4: Add model and repository**

Create `app/models/tool_result_blob.py` with columns:

```python
id UUID primary key
conversation_id UUID foreign key conversations.id, indexed, not null
user_id UUID foreign key users.id, indexed, not null
tool_call_id String(255), indexed, nullable
tool_name String(255), indexed, not null
storage_path String(1024), not null
sha256 String(64), not null
size_bytes Integer, not null
content_type String(128), default "text/plain"
created_at DateTime timezone, default func.now()
deleted_at DateTime timezone, nullable
```

Create `app/repositories/tool_result_blob.py` with:

```python
class ToolResultBlobRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, data: dict[str, Any]) -> ToolResultBlob:
        ...

    def get_for_user(self, blob_id: UUID, user_id: UUID) -> ToolResultBlob | None:
        ...
```

`get_for_user()` must filter by `ToolResultBlob.id`, `ToolResultBlob.user_id`, and `ToolResultBlob.deleted_at.is_(None)`.

- [x] **Step 5: Add `ToolResultBlobService`**

Create `app/services/tool_result_blob_service.py` with:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID, uuid4


class ToolResultBlobService:
    def __init__(self, repository, *, storage_root: str | Path, threshold_chars: int, preview_chars: int | None = None):
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.threshold_chars = max(1, int(threshold_chars))
        self.preview_chars = max(1, int(preview_chars or threshold_chars))

    def offload_if_large(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        tool_call_id: str | None,
        tool_name: str,
        output_text: str,
    ) -> dict[str, object]:
        if len(output_text) <= self.threshold_chars:
            return {"output": output_text, "blob_id": None, "size_bytes": len(output_text.encode("utf-8"))}

        blob_id = uuid4()
        relative_path = Path(str(conversation_id)) / f"{blob_id}.txt"
        absolute_path = self.storage_root / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text(output_text, encoding="utf-8")
        encoded = output_text.encode("utf-8")
        record = self.repository.create(
            {
                "id": blob_id,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "storage_path": str(relative_path),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size_bytes": len(encoded),
                "content_type": "text/plain",
            }
        )
        preview = output_text[: self.preview_chars].rstrip()
        return {
            "output": f"{preview}\n\n[Output offloaded: use blob_id to read the full result.]",
            "blob_id": str(record["id"] if isinstance(record, dict) else record.id),
            "size_bytes": len(encoded),
        }
```

- [x] **Step 6: Wire offloading into tool artifacts**

Modify `app/ai/tool_execution.py` so successful and error artifacts call an injected or globally resolved offload helper before `build_tool_artifact()` stores output.

Required artifact fields after offload:

```python
{
    "output": "<preview plus offload notice>",
    "blob_id": "<uuid string>",
    "blob_size_bytes": 12345,
    "output_truncated": True,
}
```

Modify `app/ai/graph.py::_rag_tools_node()` so `search_documents` artifacts use the same offload service for large RAG chunk/document outputs.

- [x] **Step 7: Add an authenticated read endpoint**

Create `app/api/tool_result_blobs.py`:

```python
@router.get("/tool-results/{blob_id}")
async def read_tool_result_blob(blob_id: UUID, current_user: User = Depends(get_current_user)):
    record = repository.get_for_user(blob_id, current_user.id)
    if record is None:
        raise HTTPException(status_code=404, detail="Tool result not found")
    return PlainTextResponse(service.read_text(record), media_type=record.content_type or "text/plain")
```

Register the router in the main API router file used by this project.

- [x] **Step 8: Add Streamlit fetch/render support**

Modify `demo.py`:

- When an artifact view has `blob_id`, show a button labeled `Load full result`.
- On click, request `GET /tool-results/{blob_id}` with the existing auth header.
- Render the returned text in a `st.code(..., language="text")` block.

- [x] **Step 9: Verify offload tests and compile**

Run:

```powershell
python -m pytest tests\test_tool_result_blob_service.py -q
python -m py_compile app\services\tool_result_blob_service.py app\repositories\tool_result_blob.py app\models\tool_result_blob.py app\api\tool_result_blobs.py demo.py
```

Expected result:

- Offload service tests pass.
- New Python files compile.

---

## Task 5: One-Shot Context-Overflow Retry

**Files:**
- Create: `app/ai/context_overflow.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `app/core/config.py`
- Test: `tests/test_context_overflow_retry.py`

- [x] **Step 1: Add context-overflow detector tests**

Create `tests/test_context_overflow_retry.py`:

```python
from app.ai.context_overflow import is_context_overflow_error, compact_tool_messages_for_retry
from langchain_core.messages import ToolMessage


def test_detects_common_context_limit_errors():
    assert is_context_overflow_error(Exception("maximum context length exceeded"))
    assert is_context_overflow_error(Exception("input token limit exceeded"))
    assert is_context_overflow_error(Exception("context window is too small"))
    assert not is_context_overflow_error(Exception("network timeout"))


def test_compact_tool_messages_replaces_large_tool_output():
    messages = [
        ToolMessage(
            content="x" * 200,
            tool_call_id="call-1",
            name="search_documents",
        )
    ]

    compacted = compact_tool_messages_for_retry(messages, max_chars=40)

    assert len(compacted) == 1
    assert isinstance(compacted[0], ToolMessage)
    assert len(compacted[0].content) < 120
    assert "Tool output compacted for context retry" in compacted[0].content
    assert "call-1" in compacted[0].content
```

- [x] **Step 2: Run the failing detector tests**

Run:

```powershell
python -m pytest tests\test_context_overflow_retry.py -q
```

Expected result before implementation:

- Fails because `app.ai.context_overflow` does not exist.

- [x] **Step 3: Add context overflow helpers**

> **Design note:** The plan's exact prefix produced 139 chars of content, but the test asserts `len < 120`. I shortened the format to `"Tool output compacted for context retry ({id} {name}):\n{preview}"` to satisfy the test contract while preserving the required substrings (`"Tool output compacted for context retry"` and the tool_call_id).

Create `app/ai/context_overflow.py`:

```python
from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage


_CONTEXT_ERROR_MARKERS = (
    "maximum context length",
    "context length exceeded",
    "context window",
    "input token limit",
    "token limit exceeded",
    "too many tokens",
)


def is_context_overflow_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CONTEXT_ERROR_MARKERS)


def compact_tool_messages_for_retry(messages: list[Any], *, max_chars: int) -> list[Any]:
    compacted: list[Any] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            compacted.append(message)
            continue
        content = str(message.content or "")
        if len(content) <= max_chars:
            compacted.append(message)
            continue
        preview = content[:max_chars].rstrip()
        compacted.append(
            ToolMessage(
                content=(
                    "Tool output compacted for context retry.\n"
                    f"tool_call_id: {message.tool_call_id}\n"
                    f"tool_name: {getattr(message, 'name', '') or 'unknown'}\n"
                    f"preview:\n{preview}"
                ),
                tool_call_id=message.tool_call_id,
                name=getattr(message, "name", None),
            )
        )
    return compacted
```

- [x] **Step 4: Add retry config**

Modify `app/core/config.py`:

```python
    context_overflow_retry_enabled: bool = Field(
        default=True,
        description="Retry one model call with compacted tool messages when a provider rejects the prompt for context length.",
    )
    context_overflow_retry_tool_preview_chars: int = Field(
        default=4000,
        description="Characters retained per ToolMessage during context-overflow retry.",
    )
```

- [x] **Step 5: Use retry in `BaseAgent.invoke_model_with_history()`**

> **Design note:** Wrapped only the FIRST `_ainvoke_with_retries` call in a nested try/except for context overflow, then let the outer fallback handler take over for non-context errors. This preserves the existing OpenAI reasoning summary and provider fallback paths instead of replacing them.

In `app/ai/agents/base_agent.py`, wrap the model call that invokes the tool-bound model:

```python
try:
    response = await self._ainvoke_with_retries(llm_with_tools, messages)
except Exception as exc:
    if not settings.context_overflow_retry_enabled or not is_context_overflow_error(exc):
        raise
    compacted_messages = compact_tool_messages_for_retry(
        messages,
        max_chars=settings.context_overflow_retry_tool_preview_chars,
    )
    response = await self._ainvoke_with_retries(llm_with_tools, compacted_messages)
    metadata["context_overflow_retry"] = True
```

Import:

```python
from ..context_overflow import compact_tool_messages_for_retry, is_context_overflow_error
```

- [x] **Step 6: Use retry in `RAGAgent._invoke_agentic_rag_model()`**

In `app/ai/agents/rag_agent.py`, apply the same catch around:

```python
response = await self._ainvoke_with_retries(llm_with_tools, messages)
```

Set:

```python
metadata["context_overflow_retry"] = True
```

on the final response metadata when the retry path succeeds.

- [x] **Step 7: Verify retry helper tests**

Run:

```powershell
python -m pytest tests\test_context_overflow_retry.py tests\test_rag_agent.py tests\test_graph_tool_budget.py -q
```

Expected result:

- Context overflow helper tests pass.
- Existing RAG and graph-budget tests pass.

---

## Task 6: Optional Agent-Editable User Memory

**Files:**
- Create: `app/models/user_memory.py`
- Create: `app/repositories/user_memory.py`
- Create: `app/ai/user_memory_tools.py`
- Modify: `app/core/config.py`
- Modify: `app/ai/agents/base_agent.py`
- Test: `tests/test_user_memory_tools.py`

- [x] **Step 1: Add memory tool tests**

Create `tests/test_user_memory_tools.py`:

```python
from uuid import uuid4

from app.ai.user_memory_tools import create_user_memory_tools


class FakeMemoryRepository:
    def __init__(self):
        self.rows = []

    def create(self, *, user_id, content, source):
        row = {"id": uuid4(), "user_id": user_id, "content": content, "source": source}
        self.rows.append(row)
        return row

    def list_for_user(self, user_id, limit=20):
        return [row for row in self.rows if row["user_id"] == user_id][:limit]

    def delete_for_user(self, memory_id, user_id):
        before = len(self.rows)
        self.rows = [row for row in self.rows if not (str(row["id"]) == str(memory_id) and row["user_id"] == user_id)]
        return len(self.rows) < before


def test_memory_tools_are_user_scoped():
    repo = FakeMemoryRepository()
    user_id = str(uuid4())
    other_user_id = str(uuid4())
    tools = create_user_memory_tools(repository=repo, user_id=user_id)
    remember = next(tool for tool in tools if tool.name == "remember_memory")
    list_memory = next(tool for tool in tools if tool.name == "list_memories")

    remember.invoke({"content": "User prefers concise answers.", "source": "explicit_user_request"})
    repo.create(user_id=other_user_id, content="Other user's memory", source="test")

    result = list_memory.invoke({"limit": 20})

    assert "User prefers concise answers." in result
    assert "Other user's memory" not in result
```

- [x] **Step 2: Run the failing memory tool test**

Run:

```powershell
python -m pytest tests\test_user_memory_tools.py -q
```

Expected result before implementation:

- Fails because `app.ai.user_memory_tools` does not exist.

- [x] **Step 3: Add memory config**

Modify `app/core/config.py`:

```python
    enable_user_memory_tools: bool = Field(
        default=False,
        description="Enable explicit agent-editable user memory tools.",
    )
    user_memory_max_prompt_items: int = Field(
        default=20,
        description="Maximum user memory items exposed to agents when memory tools are enabled.",
    )
```

- [x] **Step 4: Add memory model**

Create `app/models/user_memory.py` with columns:

```python
id UUID primary key
user_id UUID foreign key users.id, indexed, not null
content Text, not null
source String(255), not null
created_at DateTime timezone, default func.now()
updated_at DateTime timezone, default func.now(), onupdate func.now()
deleted_at DateTime timezone, nullable
```

- [x] **Step 5: Add memory repository**

Create `app/repositories/user_memory.py` with:

```python
class UserMemoryRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, *, user_id: str, content: str, source: str):
        ...

    def list_for_user(self, user_id: str, limit: int = 20):
        ...

    def delete_for_user(self, memory_id: str, user_id: str) -> bool:
        ...
```

All methods must filter by `user_id` and exclude rows where `deleted_at` is not null.

- [x] **Step 6: Add memory tools**

> **Design note:** The plan-provided closures lacked docstrings, which `StructuredTool.from_function` rejects. Added one-line docstrings to each closure (`remember_memory`, `list_memories`, `forget_memory`) describing the behavior so they bind without modification.

Create `app/ai/user_memory_tools.py`:

```python
from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool


def create_user_memory_tools(*, repository: Any, user_id: str | None):
    if not user_id:
        return []

    def remember_memory(content: str, source: str = "explicit_user_request") -> str:
        text = str(content or "").strip()
        if not text:
            return "Error: memory content is required."
        repository.create(user_id=user_id, content=text, source=source)
        return "Memory saved."

    def list_memories(limit: int = 20) -> str:
        rows = repository.list_for_user(user_id, limit=max(1, min(int(limit or 20), 50)))
        if not rows:
            return "No saved memories."
        return "\n".join(f"- {row['content'] if isinstance(row, dict) else row.content}" for row in rows)

    def forget_memory(memory_id: str) -> str:
        deleted = repository.delete_for_user(memory_id, user_id)
        return "Memory removed." if deleted else "Memory not found."

    return [
        StructuredTool.from_function(remember_memory),
        StructuredTool.from_function(list_memories),
        StructuredTool.from_function(forget_memory),
    ]
```

- [x] **Step 7: Bind memory tools only when enabled**

> **Design note:** Resolved `user_memory_repository` lazily through the DI container inside `_get_tools_for_binding` (the agent has no constructor injection point for it without breaking the existing constructor signature). Anonymous calls are guarded by the `user_id` check inside `create_user_memory_tools`.

Modify `BaseAgent._get_tools_for_binding()`:

- If `settings.enable_user_memory_tools` is true and `user_id` is present, append `remember_memory`, `list_memories`, and `forget_memory` to internal tools.
- Do not bind memory tools for anonymous requests.
- Add prompt text warning that memories must be written only when the user explicitly asks the assistant to remember something.

- [x] **Step 8: Verify memory tests**

Run:

```powershell
python -m pytest tests\test_user_memory_tools.py tests\test_client_tool_scope.py -q
```

Expected result:

- Memory tool tests pass.
- Existing client tool scope tests pass.

---

## Task 7: HITL `respond` Decision Type

**Files:**
- Modify: `app/schemas/workflow.py`
- Modify: `app/ai/schemas.py`
- Modify: `app/ai/utils.py`
- Modify: `app/models/tool_approval.py`
- Modify: `app/services/message_service.py`
- Add Alembic migration extending the HITL decision enum
- Test: `tests/test_hitl_respond_decision.py`

- [x] **Step 1: Add HITL respond tests**

Create `tests/test_hitl_respond_decision.py`:

```python
from app.ai.utils import apply_hitl_decisions


def test_hitl_respond_skips_tool_and_returns_human_response_as_tool_message():
    tool_calls = [
        {
            "id": "call-1",
            "name": "ask_user",
            "args": {"question": "Which file should I use?"},
        }
    ]
    decisions = [
        {
            "type": "respond",
            "tool_call_id": "call-1",
            "action": "ask_user",
            "args": {"response": "Use the quarterly report."},
        }
    ]

    approved, feedback = apply_hitl_decisions(tool_calls, decisions)

    assert approved == []
    assert feedback == {"call-1": "Use the quarterly report."}
```

- [x] **Step 2: Run the failing HITL respond test**

Run:

```powershell
python -m pytest tests\test_hitl_respond_decision.py -q
```

Expected result before implementation:

- Fails because `respond` is treated like reject.

- [x] **Step 3: Extend decision enums**

Modify both `app/schemas/workflow.py` and `app/ai/schemas.py`:

```python
class InterruptDecisionType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    RESPOND = "respond"
```

Modify `app/models/tool_approval.py`:

```python
class DecisionType(str, enum.Enum):
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"
    RESPOND = "respond"
```

- [x] **Step 4: Add the Alembic enum migration**

Create a migration that runs:

```python
op.execute("ALTER TYPE decision_type ADD VALUE IF NOT EXISTS 'respond'")
```

The downgrade should not attempt to remove the PostgreSQL enum value.

- [x] **Step 5: Implement `respond` behavior**

Modify `app/ai/utils.py::apply_hitl_decisions()`:

```python
        elif decision_type == "respond":
            response_text = ""
            if isinstance(decision, dict):
                args = decision.get("args") if isinstance(decision.get("args"), dict) else {}
                response_text = str(
                    args.get("response")
                    or args.get("message")
                    or decision.get("response")
                    or decision.get("message")
                    or ""
                ).strip()
            if not response_text:
                response_text = "The human responded without additional text."
            if tool_call_id:
                rejected_feedback[tool_call_id] = response_text
```

The returned mapping is still named `rejected_feedback` for compatibility with existing graph code, but the content is now a human-provided tool result.

- [x] **Step 6: Audit persistence mapping**

Modify `app/services/message_service.py`:

```python
decision_type_map = {
    InterruptDecisionType.APPROVE: DecisionType.ACCEPT,
    InterruptDecisionType.EDIT: DecisionType.EDIT,
    InterruptDecisionType.REJECT: DecisionType.REJECT,
    InterruptDecisionType.RESPOND: DecisionType.RESPOND,
}
```

- [x] **Step 7: Verify HITL tests**

Run:

```powershell
python -m pytest tests\test_hitl_respond_decision.py tests\test_hitl_config.py tests\test_multi_sidecar_hardening.py -q
```

Expected result:

- HITL respond tests pass.
- Existing HITL and sidecar hardening tests pass.

---

## Task 8: Full Verification

**Files:**
- No new files.

- [x] **Step 1: Run focused RAG and demo tests**

Run:

```powershell
python -m pytest tests\test_rag_artifact_visibility.py tests\test_graph_streaming_tool_events.py tests\test_demo_rag_artifacts.py tests\test_rag_tool_loop_finalization.py tests\test_rag_agent.py tests\test_rag_multi_user_isolation.py -q
```

Expected result:

- All focused RAG artifact, streaming, demo helper, and existing RAG tests pass.

- [x] **Step 2: Run DeepAgents-gap feature tests**

Run:

```powershell
python -m pytest tests\test_tool_result_blob_service.py tests\test_context_overflow_retry.py tests\test_user_memory_tools.py tests\test_hitl_respond_decision.py -q
```

Expected result:

- Large-result offload, context retry, user memory, and HITL respond tests pass.

- [x] **Step 3: Run adjacent existing suites**

> **Result:** 99 passed, 2 failed. The 2 failures (`test_deferred_tool_snapshot_round_trip_restores_aliases_and_client_scope`, `test_execute_agent_tool_calls_persists_deferred_snapshot_to_state_context`) are documented baseline failures in `plans/rag_fix.md` lines 70-71 and are unrelated to this work.

Run:

```powershell
python -m pytest tests\test_graph_tool_budget.py tests\test_client_tool_scope.py tests\test_client_tool_isolation.py tests\test_hitl_config.py tests\test_multi_sidecar_hardening.py tests\test_widget_runtime.py -q
```

Expected result:

- No new failures from this plan.
- If the two deferred client-tool snapshot failures remain, their failure messages must match the baseline already recorded in `plans/rag_fix.md`.

- [x] **Step 4: Compile changed Python files**

Run:

```powershell
python -m py_compile demo.py app\ai\graph.py app\ai\tool_execution.py app\ai\context_overflow.py app\ai\user_memory_tools.py app\services\tool_result_blob_service.py app\repositories\tool_result_blob.py app\repositories\user_memory.py app\models\tool_result_blob.py app\models\user_memory.py
```

Expected result:

- All listed files compile.

## Completion Criteria

- RAG final responses expose `search_documents` chunk/document artifacts through `AgentResponse.tool_artifacts`.
- Persisted assistant message metadata contains RAG `tool_artifacts`.
- `demo.py` shows RAG outputs in a visible `Retrieved Evidence` expander.
- Live streaming emits `tool_end` for every `ToolMessage` produced by a multi-call RAG tools node.
- Large tool results preserve full output behind a scoped blob id and keep only a preview in model-visible content.
- Context-overflow errors retry once with compacted tool messages.
- User memory tools exist behind `enable_user_memory_tools=False` by default.
- HITL supports `respond` and records the decision type in audit rows.
- No DeepAgents migration, subagent rewrite, sandbox rewrite, or provider profile abstraction is introduced.
