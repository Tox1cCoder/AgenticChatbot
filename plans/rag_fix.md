# RAG Refactor And Dead Path Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Each task has explicit validation commands and expected outcomes.

**Goal:** Refactor the RAG agent so it uses one maintained agentic execution path, matches the shared agent runtime where it matters, closes document authorization-scope gaps, guarantees a final synthesized answer after tool loops, and removes unused legacy code that still bloats the codebase.

**Architecture:** Keep `search_documents` as the single RAG tool exposed to the model and keep document action execution inside `MultiAgentWorkflow._rag_tools_node`, because that graph node owns server context, HITL flow, artifacts, and image context. Refactor `RAGAgent` around one agentic model invocation path that refreshes tools, builds the RAG prompt, applies shared runtime behavior, and can perform a no-tools final synthesis pass. Thread server-owned `user_id` and `conversation_id` through every document helper, while keeping those fields out of the model-facing tool schema.

**Tech Stack:** Python 3.10+, FastAPI backend, LangChain chat models, LangGraph `StateGraph`, SQLAlchemy repositories, Qdrant vector store, pytest, pytest-asyncio, unittest.mock.

---

## Current Implementation Audit

The current production RAG flow is `MultiAgentWorkflow._rag_node -> RAGAgent.process_message -> RAGAgent._process_message_agentic`, with tool calls handled by `_rag_tools_node` through `execute_search_documents_action`.

Important findings from the codebase:

- `app/ai/agents/rag_agent.py` still imports and uses `langchain.agents.create_agent` in legacy helper methods, even though the active graph path uses `_process_message_agentic`.
- `RAGAgent` bypasses several shared runtime behaviors used by other agents: consistent tool refresh, shared prompt suffixes, token instrumentation, forced-final response behavior, and standardized runtime metadata.
- RAG document actions have authorization-scope holes:
  - `SCAN_ALL` calls `rag_agent.scan_all_documents(conversation_id)` without `user_id`.
  - `VIEW_IMAGES` calls `rag_agent.get_document_images(document_id)` without `user_id` or `conversation_id`.
  - `scan_all_documents()` calls `list_conversation_documents(conversation_id)` and `get_document_preview(doc_id)` without carrying the user scope.
  - `get_document_images()` reads images by document id without joining back to the owning conversation.
- RAG can end after the maximum tool iteration budget without a final no-tools synthesis pass. `_should_continue_rag()` currently returns `end` when `agentic_max_iterations` is reached, so a tool-call message or empty response can become terminal.
- `RAGAgent.process_message()` refreshes tools only when `mcp_manager is None`, while other agents refresh tools on each invocation.
- Old helper methods are not called by the active graph flow and should be removed once tests prove there are no supported external call sites.

## Requirements

- RAG must have exactly one supported model-generation path.
- RAG must not depend on `langchain.agents.create_agent`.
- All document reads reachable from `search_documents` must be scoped by server-owned `conversation_id` and, when available, `user_id`.
- Model-facing RAG tool schemas must not expose `user_id`, `conversation_id`, `device_id`, or other server-owned context fields.
- RAG must refresh available MCP/runtime tools for every message, matching the other agents.
- RAG must perform a final no-tools answer pass when the tool loop reaches the configured iteration budget.
- RAG must preserve current artifact, citation, document, image-context, and streaming behavior unless a task below explicitly changes it.
- Dead RAG paths must be deleted instead of kept behind compatibility branches.

## Non-Goals

- Do not migrate RAG to DeepAgents in this refactor.
- Do not change the external API shape for chat requests.
- Do not expose internal scope fields to the model tool schema.
- Do not rewrite vector search, document ingestion, OCR, or Qdrant storage.
- Do not fix unrelated deferred client-tool snapshot tests unless a RAG change directly touches that code.

## Files To Modify

- `app/ai/agents/rag_agent.py`
- `app/ai/rag_tool_actions.py`
- `app/ai/graph.py`
- `app/repositories/document_chunk.py`
- `app/repositories/document_image.py`
- `tests/test_rag_agent.py`
- `tests/test_rag_multi_user_isolation.py`
- `tests/test_graph_tool_budget.py`

## Files To Add

- `tests/test_rag_dead_code_cleanup.py`
- `tests/test_rag_tool_loop_finalization.py`

## Phase 1: Baseline And Dead-Code Guard

### Task 1.1: Record The Current Failing Baseline ✅ DONE 2026-05-06

**Baseline result (matches plan expectation):**
- `tests\test_client_tool_isolation.py::test_deferred_tool_snapshot_round_trip_restores_aliases_and_client_scope` — FAIL (unrelated, deferred snapshot)
- `tests\test_client_tool_isolation.py::test_execute_agent_tool_calls_persists_deferred_snapshot_to_state_context` — FAIL (unrelated, deferred snapshot)
- All other tests pass (34 passed, 2 failed in 4.98s).

These two failures are documented baseline; they are NOT addressed by this refactor.

Run:

```powershell
python -m pytest tests\test_rag_agent.py tests\test_rag_multi_user_isolation.py tests\test_graph_tool_budget.py tests\test_client_tool_scope.py tests\test_client_tool_isolation.py tests\test_hitl_config.py -q
```

Expected result before this refactor:

- RAG and graph-budget tests pass.
- Two existing deferred client-tool snapshot tests may fail:
  - `test_deferred_tool_snapshot_round_trip_restores_aliases_and_client_scope`
  - `test_execute_agent_tool_calls_persists_deferred_snapshot_to_state_context`

Do not hide those failures in this refactor. Keep them documented as baseline failures if they remain unrelated.

### Task 1.2: Add A Source-Level Guard For Removed RAG Legacy Paths ✅ DONE 2026-05-06

Created `tests/test_rag_dead_code_cleanup.py` exactly as specified. Both tests fail at this baseline (legacy helpers and `create_agent` import still present).

Create `tests/test_rag_dead_code_cleanup.py`:

```python
from pathlib import Path


RAG_AGENT_SOURCE = Path("app/ai/agents/rag_agent.py")


def test_rag_agent_does_not_use_langchain_create_agent() -> None:
    source = RAG_AGENT_SOURCE.read_text(encoding="utf-8")

    assert "from langchain.agents import create_agent" not in source
    assert "create_agent(" not in source


def test_rag_agent_has_no_legacy_generation_helpers() -> None:
    source = RAG_AGENT_SOURCE.read_text(encoding="utf-8")

    legacy_helpers = [
        "def _create_agent_executor(",
        "def _generate_with_tools_stream(",
        "def _generate_with_tools(",
        "def _generate_with_vision(",
        "def _generate(",
        "def _verify_citations(",
        "def _augment_prompt_with_tool_context(",
    ]

    for helper in legacy_helpers:
        assert helper not in source
```

Run:

```powershell
python -m pytest tests\test_rag_dead_code_cleanup.py -q
```

Expected result before cleanup:

- This new test fails because legacy helpers still exist.

Expected result after cleanup:

- Both tests pass.

## Phase 2: Close RAG Document Scope Gaps ✅ DONE 2026-05-06

**Phase 2 result:** All 27 RAG-scoped tests pass.

**Design decisions:**
- `scan_all_documents` keeps return type `str` (not `Dict[str, Any]` as the plan suggested) because `app/ai/rag_tool_actions.py` consumes the formatted string directly as the tool message content. Changing the return type would require an unrelated formatter refactor in the action handler. The plan's representative test (`assert result["total_documents"] == 1`) is therefore not a binding requirement; the focused tests in `tests/test_rag_agent.py` only assert the helper-call propagation, which works with either return type.
- All scope parameters added as keyword-only (`*, user_id`, `*, conversation_id`) for consistency with existing helper signatures (`get_document_full_content`, `grep_document`, `list_conversation_documents`).
- `get_document_preview` retains its existing keyword-only `max_chars` parameter for backward compatibility; only the new scope params are added.
- `DocumentImageRepository.get_by_document_for_scope` mirrors `DocumentChunkRepository.get_by_document_for_scope` exactly: SQL-layer auth join via `Document` and (optionally) `Conversation`, with UUID coercion that returns `[]` on bad inputs.

### Task 2.1: Add Tests For Server Context Propagation

Add tests to `tests/test_rag_agent.py` or `tests/test_rag_multi_user_isolation.py` that prove:

- `RAGAgent.scan_all_documents(conversation_id, user_id=user_id)` passes `user_id` into `list_conversation_documents`.
- `RAGAgent.scan_all_documents(conversation_id, user_id=user_id)` passes both `user_id` and `conversation_id` into `get_document_preview`.
- `RAGAgent.get_document_preview(document_id, user_id=user_id, conversation_id=conversation_id)` passes both fields into `get_document_full_content`.
- `RAGAgent.get_document_images(document_id, user_id=user_id, conversation_id=conversation_id)` uses a scoped image repository lookup.

Representative test shape:

```python
@pytest.mark.asyncio
async def test_scan_all_documents_preserves_user_scope() -> None:
    agent = object.__new__(RAGAgent)
    agent.list_conversation_documents = AsyncMock(
        return_value=[{"id": "doc-1", "filename": "a.pdf"}]
    )
    agent.get_document_preview = AsyncMock(return_value="preview")

    result = await agent.scan_all_documents("conv-1", user_id="user-1")

    agent.list_conversation_documents.assert_awaited_once_with(
        "conv-1",
        user_id="user-1",
    )
    agent.get_document_preview.assert_awaited_once_with(
        "doc-1",
        user_id="user-1",
        conversation_id="conv-1",
    )
    assert result["total_documents"] == 1
```

Add action tests for `execute_search_documents_action`:

```python
@pytest.mark.asyncio
async def test_search_documents_scan_all_passes_server_scope() -> None:
    rag_agent = Mock()
    rag_agent.scan_all_documents = AsyncMock(return_value={"total_documents": 0})

    await execute_search_documents_action(
        SearchDocumentsInput(query="", action="scan_all"),
        rag_agent,
        conversation_id="conv-1",
        user_id="user-1",
    )

    rag_agent.scan_all_documents.assert_awaited_once_with(
        "conv-1",
        user_id="user-1",
    )
```

Run:

```powershell
python -m pytest tests\test_rag_agent.py tests\test_rag_multi_user_isolation.py -q
```

Expected result before implementation:

- The new scope propagation tests fail.

### Task 2.2: Add Scoped Document Image Lookup

Modify `app/repositories/document_image.py`.

Add a repository method with this signature:

```python
def get_by_document_for_scope(
    self,
    document_id: UUID,
    user_id: str | None = None,
    conversation_id: str | None = None,
) -> List[DocumentImage]:
```

Implementation requirements:

- Start from `DocumentImage`.
- Join to `Document`.
- Filter by `DocumentImage.document_id == document_id`.
- If `conversation_id` is provided, filter by `Document.conversation_id == UUID(conversation_id)`.
- If `user_id` is provided, join to `Conversation` and filter by `Conversation.owner_id == user_id`.
- Return an empty list if the document id or conversation id is not a valid UUID.
- Preserve the existing unscoped `get_by_document_id()` method for internal call sites that genuinely have no scope, but do not use it from the RAG tool action path.

Add a focused repository test if the existing test fixtures can create `DocumentImage` rows cheaply. If repository fixtures are heavy, keep the repository method covered through the RAG helper tests with a mocked repository.

### Task 2.3: Thread Scope Through RAG Helpers

Modify `app/ai/agents/rag_agent.py`.

Change signatures:

```python
async def get_document_preview(
    self,
    document_id: str,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> str:
```

```python
async def scan_all_documents(
    self,
    conversation_id: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
```

```python
async def get_document_images(
    self,
    document_id: str,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
```

Implementation requirements:

- `get_document_preview()` calls `get_document_full_content(document_id, user_id=user_id, conversation_id=conversation_id)`.
- `scan_all_documents()` calls `list_conversation_documents(conversation_id, user_id=user_id)`.
- `scan_all_documents()` calls `get_document_preview(doc_id, user_id=user_id, conversation_id=conversation_id)`.
- `get_document_images()` calls `DocumentImageRepository.get_by_document_for_scope()` when either scope field is present.
- `get_document_images()` may call `get_by_document_id()` only when both `user_id` and `conversation_id` are absent.

### Task 2.4: Thread Scope Through RAG Tool Actions

Modify `app/ai/rag_tool_actions.py`.

Implementation requirements:

- `SCAN_ALL` calls:

```python
await rag_agent.scan_all_documents(conversation_id, user_id=user_id)
```

- `VIEW_IMAGES` calls:

```python
await rag_agent.get_document_images(
    document_id,
    user_id=user_id,
    conversation_id=conversation_id,
)
```

- Existing `READ_DOCUMENT`, `SEARCH_CHUNKS`, `GREP_DOCUMENT`, and `LIST_DOCUMENTS` scope behavior remains intact.

Run:

```powershell
python -m pytest tests\test_rag_agent.py tests\test_rag_multi_user_isolation.py -q
```

Expected result:

- New and existing RAG scope tests pass.

## Phase 3: Make RAG Runtime Behavior Consistent ✅ DONE 2026-05-06

**Phase 3 result:** 26 RAG tests + 11 message-history tests pass.

**Design decisions:**
- The shared runtime metadata keys are `provider`, `model`, `key_source`, `config_source` (set by `BaseAgent._apply_runtime_metadata`). The plan's representative example used `runtime_provider`/`runtime_model`, which do not exist; tests assert against the real keys.
- `_invoke_agentic_rag_model` keeps the per-invocation provider-fallback loop that the inline implementation had, so transient provider errors keep working unchanged.
- `disable_tools=True` skips `ModelFactory.bind_tools_to_model` entirely (passes the bare LLM). Tools are also skipped when the tool list is empty (no schema spam).
- `DELEGATION_SUFFIX` is now appended to the agentic RAG system prompt for parity with `BaseAgent._build_system_prompt`. The `hand_off` tool exists in the shared MCP catalog, so the model needs the prompt language to know to use it for off-topic requests.
- `_process_message_agentic` reads `rag_force_final_response` and `rag_tool_budget_notice` from the inbound `AgentMessage.metadata` and forwards `disable_tools` into `_invoke_agentic_rag_model`. This is the wiring Phase 4 then leverages.

### Task 3.1: Test Tool Refresh On Every RAG Invocation

Add to `tests/test_rag_agent.py`:

```python
@pytest.mark.asyncio
async def test_rag_process_message_refreshes_tools_every_invocation() -> None:
    agent = object.__new__(RAGAgent)
    agent.mcp_manager = object()
    agent._init_tools = AsyncMock()
    agent._process_message_agentic = AsyncMock(
        return_value=AgentResponse(
            content="done",
            agent_type=AgentType.RAG,
            metadata={},
        )
    )

    message = AgentMessage(
        content="read the uploaded document",
        user_id="user-1",
        metadata={},
    )

    await agent.process_message(message, "conv-1")

    agent._init_tools.assert_awaited_once()
```

Run:

```powershell
python -m pytest tests\test_rag_agent.py -q
```

Expected result before implementation:

- The new test fails because RAG refreshes tools only when `mcp_manager is None`.

### Task 3.2: Refresh Tools Unconditionally

Modify `RAGAgent.process_message()`.

Implementation requirement:

- Call `await self._init_tools()` for every invocation before `_process_message_agentic()`.
- Preserve all existing error handling and response shape.
- Do not duplicate initialization if `_init_tools()` already handles idempotence.

Run:

```powershell
python -m pytest tests\test_rag_agent.py -q
```

Expected result:

- The tool refresh test passes.

### Task 3.3: Add Shared Runtime Coverage For RAG

Add focused tests that prove active RAG invocation preserves these shared behaviors:

- Runtime metadata includes the selected model/provider fields already used by other agents.
- RAG prompt includes `DELEGATION_SUFFIX`.
- RAG can bind model tools through the shared model factory path.
- RAG can run with tools disabled for final synthesis.

Use mocks for model invocation. The tests should assert method calls and metadata content; they do not need to call external providers.

Representative assertions:

```python
assert "Delegate to" in rendered_prompt or DELEGATION_SUFFIX in rendered_prompt
assert response.metadata["runtime_model"] == "..."
assert response.metadata["runtime_provider"] == "..."
assert bound_tools == []
```

The exact metadata key names must match the existing shared runtime methods in `BaseAgent`.

### Task 3.4: Extract One Active Agentic Invocation Path

Modify `app/ai/agents/rag_agent.py`.

Create one internal method for active RAG model calls:

```python
async def _invoke_agentic_rag_model(
    self,
    *,
    message: AgentMessage,
    conversation_id: str,
    system_prompt: str,
    chat_history: list[BaseMessage],
    tools: list[Any],
    disable_tools: bool,
    stream_callback: Optional[Callable[[str], None]],
) -> AgentResponse:
```

Implementation requirements:

- Use the same runtime model resolution path as other agents:
  - `_resolve_runtime_model_config`
  - `_create_langchain_model_from_runtime`
  - `ModelFactory.bind_tools_to_model`
  - `_ainvoke_with_retries`
  - `_apply_runtime_metadata`
- Bind no tools when `disable_tools=True`.
- Preserve active RAG metadata:
  - `agentic_rag`
  - `tool_calls`
  - `artifacts`
  - `documents_used`
  - `image_context`
  - token usage fields already present in RAG responses
- Keep RAG-specific prompt sections:
  - `AGENTIC_RAG_SYSTEM_PROMPT`
  - `TOOL_EXPLORATION_SUFFIX`
  - persona prompt
  - memory context
  - uploaded document context
  - `DELEGATION_SUFFIX`
- Keep graph-owned tool execution. This method only asks the model for the next assistant message or final answer.

Run:

```powershell
python -m pytest tests\test_rag_agent.py tests\test_message_history_pipeline.py -q
```

Expected result:

- Existing message-history behavior remains unchanged.
- New runtime behavior tests pass.

## Phase 4: Guarantee Final Synthesis After RAG Tool Budget ✅ DONE 2026-05-06

**Phase 4 result:** 4 new finalization tests + existing tool-budget tests pass.

**Design decisions:**
- `_should_continue_rag` now sets two context keys when budget exhausted: `rag_force_final_response=True` and `rag_tool_budget_notice` (the prompt-side notice). On the second visit (when the flag is already set) it returns `end` so the loop cannot run forever.
- `_rag_node` forwards both keys from `state["context"]` into `agent_msg.metadata` so the agent layer (which has no graph-state visibility) can honour them. The flags stay in context only for the remainder of this turn — each new user request starts with an empty `agentic_rag_iteration` and clears the flags implicitly.
- The budget notice text mirrors the wording used by `BaseAgent` for non-RAG tool-budget exhaustion (`TOOL BUDGET NOTICE: ...`), so prompt observers see consistent output.

### Task 4.1: Add Tests For RAG Tool-Loop Finalization

Create `tests/test_rag_tool_loop_finalization.py`.

Test cases:

- When the last message is a RAG tool result and `agentic_rag_iteration >= agentic_max_iterations`, `_should_continue_rag()` routes back to `rag_agent` for one final no-tools pass.
- That final pass marks context so RAG disables tool binding.
- The final pass produces a normal assistant response with non-empty content.
- If the last RAG response already has no tool calls, the graph still ends normally.

Representative test shape:

```python
def test_rag_budget_routes_to_final_no_tool_pass() -> None:
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agentic_max_iterations = 2

    state = {
        "agentic_rag_iteration": 2,
        "messages": [ToolMessage(content="chunk text", tool_call_id="call-1")],
        "context": {},
    }

    route = workflow._should_continue_rag(state)

    assert route == "rag_agent"
    assert state["context"]["rag_force_final_response"] is True
```

Run:

```powershell
python -m pytest tests\test_rag_tool_loop_finalization.py tests\test_graph_tool_budget.py -q
```

Expected result before implementation:

- The new finalization tests fail because RAG ends immediately at the budget limit.

### Task 4.2: Route Budget Exhaustion To A Final No-Tools Pass

Modify `app/ai/graph.py`.

Implementation requirements:

- In `_should_continue_rag()`, when the RAG iteration budget is reached and the latest message is a tool result or the latest assistant message contains tool calls, return `rag_agent` one more time instead of `end`.
- Set server context flags:

```python
context["rag_force_final_response"] = True
context["rag_tool_budget_notice"] = (
    "The RAG tool budget is exhausted. Produce the final answer from the "
    "retrieved document evidence already available. Do not call tools."
)
```

- If `rag_force_final_response` is already true and the latest assistant message still asks for tools, route to `end` and use the existing terminal recovery fallback to avoid an infinite loop.
- Do not change non-RAG routing.

### Task 4.3: Disable RAG Tool Binding During Final Synthesis

Modify `app/ai/agents/rag_agent.py`.

Implementation requirements:

- Read `rag_force_final_response` and `rag_tool_budget_notice` from `message.metadata` or graph-provided context.
- Pass `disable_tools=True` into `_invoke_agentic_rag_model()` when `rag_force_final_response` is true.
- Append the budget notice to the system prompt.
- Ensure response metadata includes:

```python
"rag_force_final_response": True
```

when the final pass was used.

Modify `_rag_node()` if needed so the graph forwards these context flags into `AgentMessage.metadata`.

Run:

```powershell
python -m pytest tests\test_rag_tool_loop_finalization.py tests\test_graph_tool_budget.py tests\test_rag_agent.py -q
```

Expected result:

- RAG budget exhaustion now produces one final no-tools model pass.
- Existing tool-budget tests still pass.

## Phase 5: Delete Legacy RAG Paths ✅ DONE 2026-05-06

**Phase 5 result:** Dead-code guard tests pass (`tests/test_rag_dead_code_cleanup.py`); 43 RAG-focused tests pass.

**Call-site audit (Task 5.1):**
- `get_status` — KEPT. Production caller: `app/workers/cleanup_tasks.py:78` (Celery cleanup task).
- `delete_document_vectors` — KEPT. No production caller found, but the plan's explicit delete list does not include it. Treated as a public maintenance method (documents/admin pipeline) rather than legacy LLM plumbing.
- `_rerank_results` — KEPT. Called from `_search` (still active).
- `_fetch_images_for_chunks` — KEPT. Called from `app/ai/rag_tool_actions.py` SEARCH_CHUNKS.

**Methods deleted (Task 5.2):**
- `_create_agent_executor`
- `_generate_with_tools_stream`
- `_generate_with_tools`
- `_generate_with_vision`
- `_generate`
- `_generate_stream` (orphaned; unused even before this refactor)
- `_verify_citations`
- `_augment_prompt_with_tool_context`
- `_get_media_resolution` (orphaned by `_generate_with_vision` removal)
- `_extract_referenced_document_numbers` (orphaned by `_verify_citations` removal)

**Imports removed (Task 5.3):**
- `asyncio`, `json`, `queue`, `threading` (only used by deleted streaming/threaded code)
- `from google.genai import types` (only used by deleted vision code)
- `from langchain.agents import create_agent` (only used by deleted `_create_agent_executor`)
- `from ..agent_config import build_gemini_generate_config` (only used by deleted `_generate*` paths)
- `extract_agent_execution_info` from `..utils` (only used by deleted `_generate_with_tools`)

`ruff` is not installed in this environment, so `python -m compileall` was used as the lint substitute (per the plan's fallback instruction). All four touched production files compile.

### Task 5.1: Verify The Candidate Dead Paths Have No Supported Call Sites

Run:

```powershell
rg -n "create_agent|_create_agent_executor|_generate_with_tools_stream|_generate_with_tools\(|_generate_with_vision|_generate\(|_augment_prompt_with_tool_context|_verify_citations|delete_document_vectors|get_status" app tests
```

Classify results:

- Delete `create_agent` import and all helper definitions that only reference each other.
- Delete citation helper code if its only caller is `_verify_citations`.
- Delete vector/status helper methods only if the search shows no production or test call sites outside the method definitions.
- If `delete_document_vectors` or `get_status` has a production call site, keep it and add a short comment in this plan execution notes explaining why it is not dead.

### Task 5.2: Remove Legacy Helper Methods

Modify `app/ai/agents/rag_agent.py`.

Delete these methods when Task 5.1 confirms they are unreferenced:

- `_create_agent_executor`
- `_generate_with_tools_stream`
- `_generate_with_tools`
- `_generate_with_vision`
- `_generate`
- `_verify_citations`
- `_augment_prompt_with_tool_context`

Delete imports that become unused, including:

- `from langchain.agents import create_agent`
- prompt or citation imports used only by removed code
- typing imports used only by removed code

Preserve active public RAG helper methods:

- `process_message`
- `_process_message_agentic`
- `_search`
- `get_document_full_content`
- `get_document_preview`
- `list_conversation_documents`
- `grep_document`
- `scan_all_documents`
- `get_document_images`

Run:

```powershell
python -m pytest tests\test_rag_dead_code_cleanup.py tests\test_rag_agent.py -q
```

Expected result:

- Dead-code cleanup tests pass.
- RAG agent tests pass.

### Task 5.3: Clean Imports And Type Noise

Run:

```powershell
python -m ruff check app\ai\agents\rag_agent.py app\ai\rag_tool_actions.py app\ai\graph.py app\repositories\document_image.py tests\test_rag_agent.py tests\test_rag_multi_user_isolation.py tests\test_graph_tool_budget.py tests\test_rag_dead_code_cleanup.py tests\test_rag_tool_loop_finalization.py
```

Expected result:

- No unused imports remain in modified files.
- No broad lint churn appears outside the touched files.

If `ruff` is not installed, run:

```powershell
python -m compileall app\ai\agents\rag_agent.py app\ai\rag_tool_actions.py app\ai\graph.py app\repositories\document_image.py
```

Expected result:

- All modified production files compile successfully.

## Phase 6: Full Focused Verification ✅ DONE 2026-05-06

**Final verification results:**
- Focused RAG suite (`test_rag_agent`, `test_rag_multi_user_isolation`, `test_graph_tool_budget`, `test_rag_dead_code_cleanup`, `test_rag_tool_loop_finalization`) — **43 passed**.
- Adjacent agent-runtime suite (`test_message_history_pipeline`, `test_search_agent_time_context`, `test_tool_search_prompt_guidance`, `test_hitl_config`) — **18 passed**.
- Baseline-compare suite (the same set used in Task 1.1) — **48 passed, 2 failed**. The two failures are exactly the same baseline failures recorded in Task 1.1: `test_deferred_tool_snapshot_round_trip_restores_aliases_and_client_scope` and `test_execute_agent_tool_calls_persists_deferred_snapshot_to_state_context`. They are unrelated to this refactor.

All Completion Criteria from the plan are met:
- `app/ai/agents/rag_agent.py` has no `create_agent` import or call.
- `tests/test_rag_dead_code_cleanup.py` passes.
- Every `search_documents` action that reads documents, chunks, previews, images, or listings receives server-owned scope from the graph.
- `SearchDocumentsInput` remains free of `user_id`, `conversation_id`, and `device_id`.
- RAG refreshes tools on every invocation.
- RAG budget exhaustion performs a final no-tools synthesis pass.
- Focused RAG tests pass.
- The two remaining failing tests are documented baseline failures (Task 1.1) unrelated to the RAG refactor.

Run the focused RAG suite:

```powershell
python -m pytest tests\test_rag_agent.py tests\test_rag_multi_user_isolation.py tests\test_graph_tool_budget.py tests\test_rag_dead_code_cleanup.py tests\test_rag_tool_loop_finalization.py -q
```

Expected result:

- All focused RAG tests pass.

Run the adjacent agent-runtime suite:

```powershell
python -m pytest tests\test_message_history_pipeline.py tests\test_search_agent_time_context.py tests\test_tool_search_prompt_guidance.py tests\test_hitl_config.py -q
```

Expected result:

- Adjacent agent-runtime tests pass.

Run the known baseline suite again:

```powershell
python -m pytest tests\test_rag_agent.py tests\test_rag_multi_user_isolation.py tests\test_graph_tool_budget.py tests\test_client_tool_scope.py tests\test_client_tool_isolation.py tests\test_hitl_config.py -q
```

Expected result:

- No new RAG failures.
- If the two deferred client-tool snapshot tests still fail, their failure messages match the baseline from Phase 1.

## Phase 7: Staff Review Follow-up Fixes ✅ DONE 2026-05-06

**Review result:** Three follow-up issues were found after the initial implementation and have been fixed with regression coverage.

**Issues fixed:**
- `SEARCH_CHUNKS` now re-checks server-owned scope during SQL chunk hydration. Qdrant payload filtering alone is not trusted; `RAGAgent._search()` calls the new `DocumentChunkRepository.get_by_ids_for_scope()` whenever `user_id` or `conversation_id` is available.
- Forced-final RAG responses that still contain tool calls no longer route to `rag_tools`. `_should_call_rag_tools()` now ends the graph when `rag_force_final_response=True`, preventing an extra tool execution after the no-tools final pass.
- The final no-tools synthesis prompt no longer tells the model to use `search_documents`; the tool-use instruction is only appended during normal tool-enabled RAG passes.

**Regression tests added:**
- `tests/test_rag_multi_user_isolation.py::test_rag_search_rehydrates_chunks_with_server_scope`
- `tests/test_rag_tool_loop_finalization.py::test_forced_final_assistant_tool_calls_do_not_route_to_rag_tools`
- Extended `tests/test_rag_tool_loop_finalization.py::test_process_message_agentic_disables_tools_when_force_final_response_flag_set` to assert the final pass prompt omits the `search_documents` instruction.

**Verification after follow-up fixes:**
- Focused RAG suite (`test_rag_agent`, `test_rag_multi_user_isolation`, `test_graph_tool_budget`, `test_rag_dead_code_cleanup`, `test_rag_tool_loop_finalization`) — **45 passed**.
- Adjacent agent-runtime suite (`test_message_history_pipeline`, `test_search_agent_time_context`, `test_tool_search_prompt_guidance`, `test_hitl_config`) — **18 passed**.
- Production compile check (`rag_agent`, `rag_tool_actions`, `graph`, `document_chunk`, `document_image`) — **passed**.
- Baseline-compare suite (`test_rag_agent`, `test_rag_multi_user_isolation`, `test_graph_tool_budget`, `test_client_tool_scope`, `test_client_tool_isolation`, `test_hitl_config`) — **49 passed, 2 failed**. The two failures are the documented deferred client-tool snapshot baseline failures from Task 1.1.

## Completion Criteria

- `app/ai/agents/rag_agent.py` has no `create_agent` import or call.
- `tests/test_rag_dead_code_cleanup.py` passes.
- Every `search_documents` action that reads documents, chunks, previews, images, or listings receives server-owned scope from the graph, and `SEARCH_CHUNKS` revalidates that scope during SQL chunk hydration.
- `SearchDocumentsInput` remains free of `user_id`, `conversation_id`, and `device_id`.
- RAG refreshes tools on every invocation.
- RAG budget exhaustion performs a final no-tools synthesis pass.
- Focused RAG tests pass.
- Any remaining failing tests are documented baseline failures unrelated to the RAG refactor.
