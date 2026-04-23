# Chatbot Memory Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make chatbot history handling deterministic, durable, budgeted, non-duplicative, and safe across non-streaming chat, streaming chat, AI SDK routes, RAG fast paths, tool loops, HITL resume, cancellation, and conversation deletion.

**Architecture:** PostgreSQL `messages` remains the canonical user-visible transcript. A new durable conversation-memory summary stores compacted long-term memory with a database-message cursor, while LangGraph checkpoints are treated as transient execution state for ReAct loops and HITL resume. Agent prompts are assembled through one history provider that returns a rolling summary plus recent unsummarized messages, with stable exclusion of the current user message by ID.

**Tech Stack:** FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2, LangGraph, LangChain messages, PostgreSQL JSONB, pytest, pytest-asyncio.

---

## Current Implementation Audit

### What Is Already Good

- User-visible messages are persisted in PostgreSQL before workflow execution in `app/services/message_service.py`.
- Assistant messages invalidate the workflow history cache through `AIService.invalidate_history_cache`.
- Agent-specific history budgets exist in `app/core/config.py`: `chat_history_max_*`, `rag_history_max_*`, `search_history_max_*`, and `planning_history_max_*`.
- Tool and HITL flows persist enough metadata for UI recovery.
- Summarization code is fail-closed on model errors and suppresses internal summarization output from streams.

### Production Gaps To Fix

- Memory has two sources of truth: PostgreSQL messages and LangGraph checkpoint `messages`. Current rolling summaries are checkpoint-backed, but prompt history is loaded from PostgreSQL, so summaries can overlap with DB history.
- `summary_cursor_message_id` is a LangChain message ID, not a database message ID. It cannot safely prevent duplicate prompt context.
- Graph checkpoint compaction depends on `RemoveMessage` IDs, but some `HumanMessage` and fast-path persisted checkpoint messages are created without stable IDs.
- `_get_conversation_history()` uses `exclude_last=1`, which assumes the newest DB row is always the current user message. That is brittle for direct workflow calls, retries, interrupted turns, cancellation, and future ingestion paths.
- `MemoryManager` has an unbounded process-global `_memories` dict and reaches around dependency injection by calling `get_db()` directly.
- `MessageCRUDStrategy.get_by_conversation_id()` does not filter `Message.deleted_at`, and count queries materialize whole result sets instead of using SQL `COUNT`.
- Empty paused assistant messages and error/cancellation artifacts can enter model history as normal assistant turns.
- Some legacy agent methods build a prompt containing history and then pass the same history to `BaseAgent.invoke_model_with_history()`, making duplicate context possible outside the graph path.
- Streaming summarization tests currently fail against the implementation:
  - Command: `python -m pytest tests/test_graph_streaming_summarization.py -q`
  - Result: `2 failed, 1 passed`
  - The tests expect deferred streaming summarization, but `execute_request_stream()` no longer marks state for deferred summarization or calls `_persist_deferred_stream_summarization()`.
- Conversation deletion soft-deletes the conversation but does not clear in-process memory caches or LangGraph checkpoint state.

---

## Target Behavior

### Canonical Memory Rules

1. PostgreSQL `messages` is the only canonical long-term transcript.
2. LangGraph checkpoints are execution state only. They may hold current-turn tool state and HITL resume state, but should not be relied on for long-term memory.
3. Prompt memory is built from:
   - the latest durable summary for the conversation, if present;
   - recent DB messages after the summary cursor;
   - no copy of the current user message in prior history.
4. The durable summary cursor is a `messages.id` value, not a LangChain generated ID.
5. Deleted messages, empty paused assistant placeholders, and hidden system artifacts are not included as normal prompt history.
6. Every graph invocation receives stable message IDs for user-visible messages whenever a DB row exists.
7. Summarization never blocks token streaming on the hot path. Summary refresh runs after successful assistant persistence or as a bounded background job.
8. History budgets are enforced after combining summary and recent messages.

### Non-Goals

- Do not add semantic/vector long-term memory in this refactor.
- Do not change the public response shapes for `/messages/*`, `/api/chat/{conversation_id}`, or `/ai/chat/{conversation_id}`.
- Do not rewrite the agent routing or tool execution architecture.

---

## File Structure

Create:

- `app/models/conversation_memory_summary.py` - SQLAlchemy model for durable rolling memory.
- `app/repositories/conversation_memory_summary.py` - upsert and lookup repository.
- `app/ai/history.py` - prompt-history assembly, DB row normalization, budget trimming, and cache ownership.
- `app/ai/conversation_summarizer.py` - provider wrapper around the existing summary prompt/model call.
- `app/alembic/versions/p9q0r1s2t3u4_add_conversation_memory_summaries.py` - migration.
- `tests/test_history_provider.py` - focused prompt-history unit tests.
- `tests/test_conversation_memory_summary_repository.py` - repository tests.
- `tests/test_message_history_pipeline.py` - workflow/message-service pipeline tests.

Modify:

- `app/ai/schemas.py` - add workflow message ID fields and optional memory context metadata.
- `app/schemas/workflow.py` - mirror service-facing workflow message ID fields.
- `app/ai/graph.py` - consume `ConversationHistoryProvider`, set stable message IDs, remove graph-level long-term summarization from the request hot path, compact checkpoint state after completion.
- `app/ai/memory.py` - replace or reduce to compatibility wrapper around `app/ai/history.py`.
- `app/ai/summarization_middleware.py` - keep only checkpoint-local utilities if needed, or delegate summary generation to `conversation_summarizer.py`.
- `app/services/message_service.py` - pass user/assistant message IDs into workflow requests, schedule summary refresh after assistant persistence, invalidate cache on all transcript writes.
- `app/services/ai_service.py` - pass through new workflow fields and expose memory invalidation hooks.
- `app/repositories/message.py` - add efficient filtered history queries and SQL `COUNT`.
- `app/models/__init__.py` - import the new model if this package is used for metadata discovery.
- `app/core/container.py` - inject the history provider and summary repository into the workflow/service layer.
- `app/services/conversation_service.py` - clear memory/checkpoint state on conversation deletion.
- `README.md` and `.env.example` - document the new memory flow and settings.
- `tests/test_graph_streaming_summarization.py` - replace stale checkpoint-summary expectations with the new durable-memory behavior.

---

## Data Model

Add table `conversation_memory_summaries`:

```text
id uuid primary key
conversation_id uuid not null unique references conversations(id)
user_id uuid not null references users(id)
summary_text text not null default ''
last_summarized_message_id uuid null references messages(id)
source_message_count integer not null default 0
estimated_tokens integer not null default 0
summary_version integer not null default 1
created_at timestamptz not null default now()
updated_at timestamptz not null default now()
```

Indexes:

```text
ux_conversation_memory_summaries_conversation_id unique(conversation_id)
ix_conversation_memory_summaries_user_id(user_id)
ix_conversation_memory_summaries_last_message(last_summarized_message_id)
```

---

## Task 1: Add Durable Conversation Summary Storage

**Files:**
- Create: `app/models/conversation_memory_summary.py`
- Create: `app/repositories/conversation_memory_summary.py`
- Create: `app/alembic/versions/p9q0r1s2t3u4_add_conversation_memory_summaries.py`
- Modify: `app/models/__init__.py`
- Test: `tests/test_conversation_memory_summary_repository.py`

- [ ] **Step 1: Write the repository tests**

Create `tests/test_conversation_memory_summary_repository.py` with tests that assert:

```python
def test_upsert_creates_summary_for_conversation(session_factory, user, conversation, message):
    repo = ConversationMemorySummaryRepository(session_factory)

    saved = repo.upsert(
        conversation_id=conversation.id,
        user_id=user.id,
        summary_text="- User asked about invoices",
        last_summarized_message_id=message.id,
        source_message_count=2,
        estimated_tokens=16,
    )

    assert saved.conversation_id == conversation.id
    assert saved.last_summarized_message_id == message.id
    assert saved.summary_text == "- User asked about invoices"
```

```python
def test_upsert_updates_existing_summary_cursor(session_factory, user, conversation, message, later_message):
    repo = ConversationMemorySummaryRepository(session_factory)
    repo.upsert(
        conversation_id=conversation.id,
        user_id=user.id,
        summary_text="- Old summary",
        last_summarized_message_id=message.id,
        source_message_count=2,
        estimated_tokens=8,
    )

    saved = repo.upsert(
        conversation_id=conversation.id,
        user_id=user.id,
        summary_text="- Updated summary",
        last_summarized_message_id=later_message.id,
        source_message_count=6,
        estimated_tokens=20,
    )

    assert saved.summary_text == "- Updated summary"
    assert saved.last_summarized_message_id == later_message.id
    assert saved.source_message_count == 6
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_conversation_memory_summary_repository.py -q
```

Expected: FAIL because `ConversationMemorySummaryRepository` does not exist.

- [ ] **Step 3: Add the SQLAlchemy model**

Implement `ConversationMemorySummary` with the fields listed in the Data Model section and `relationship()` links to `Conversation`, `User`, and `Message`.

- [ ] **Step 4: Add the Alembic migration**

Generate or write a migration that creates the table and indexes. Use `sa.text("now()")` or `func.now()` consistently with the existing migration style.

- [ ] **Step 5: Add the repository**

Implement:

```python
class ConversationMemorySummaryRepository:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def get_by_conversation_id(self, conversation_id: UUID) -> ConversationMemorySummary | None:
        with self.session_factory() as session:
            statement = select(ConversationMemorySummary).where(
                ConversationMemorySummary.conversation_id == conversation_id
            )
            result = session.execute(statement).scalar_one_or_none()
            if result is not None:
                session.expunge(result)
            return result

    def upsert(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        summary_text: str,
        last_summarized_message_id: UUID | None,
        source_message_count: int,
        estimated_tokens: int,
    ) -> ConversationMemorySummary:
        with self.session_factory() as session:
            statement = select(ConversationMemorySummary).where(
                ConversationMemorySummary.conversation_id == conversation_id
            )
            row = session.execute(statement).scalar_one_or_none()
            if row is None:
                row = ConversationMemorySummary(
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                session.add(row)
            row.summary_text = summary_text
            row.last_summarized_message_id = last_summarized_message_id
            row.source_message_count = source_message_count
            row.estimated_tokens = estimated_tokens
            row.summary_version = (row.summary_version or 0) + 1
            session.commit()
            session.refresh(row)
            session.expunge(row)
            return row
```

Use one transaction and return a detached refreshed model instance, matching repository patterns in this repo.

- [ ] **Step 6: Run repository tests**

Run:

```bash
python -m pytest tests/test_conversation_memory_summary_repository.py -q
```

Expected: PASS.

---

## Task 2: Add Canonical Prompt History Queries

**Files:**
- Modify: `app/repositories/message.py`
- Test: `tests/test_history_provider.py`

- [ ] **Step 1: Write message-query tests**

Add tests that create:

- an old user message;
- an old assistant message;
- a soft-deleted user message;
- an empty paused assistant message with `message_metadata={"paused": True}`;
- the current user message.

Assert the history query:

```python
rows = message_repo.get_prompt_history(
    conversation_id=conversation.id,
    before_message_id=current_message.id,
    after_message_id=None,
    limit=20,
)

assert [row.id for row in rows] == [old_user.id, old_assistant.id]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_history_provider.py -q
```

Expected: FAIL because `get_prompt_history()` does not exist.

- [ ] **Step 3: Replace materialized counts with SQL counts**

In `MessageCRUDStrategy.count_by_conversation_id()` and `count_by_user_id()`, use `select(func.count(Message.id))` and filter `Message.deleted_at.is_(None)`.

- [ ] **Step 4: Add filtered prompt-history method**

Add `MessageRepository.get_prompt_history()` and strategy support:

```python
def get_prompt_history(
    self,
    conversation_id: UUID,
    *,
    before_message_id: UUID | None = None,
    after_message_id: UUID | None = None,
    limit: int = 50,
) -> list[Message]:
    with self.session_factory() as session:
        return self._crud_strategy.get_prompt_history(
            session,
            conversation_id,
            before_message_id=before_message_id,
            after_message_id=after_message_id,
            limit=limit,
        )
```

Rules:

- Always filter `Message.deleted_at.is_(None)`.
- Always order final returned rows ascending by `created_at`, then `id`.
- If `before_message_id` is provided, find that row's `(created_at, id)` and return only earlier rows.
- If `after_message_id` is provided, find that row's `(created_at, id)` and return only later rows.
- Exclude assistant rows where `content == ""` and `message_metadata.paused == true`.
- Exclude assistant rows where `message_metadata.interrupt` exists and `content == ""`.
- Do not exclude partial assistant rows with non-empty content; they are visible transcript history.

- [ ] **Step 5: Run history query tests**

Run:

```bash
python -m pytest tests/test_history_provider.py -q
```

Expected: prompt-query tests PASS.

---

## Task 3: Build One History Provider For All Agents

**Files:**
- Create: `app/ai/history.py`
- Modify: `app/ai/memory.py`
- Modify: `app/core/container.py`
- Test: `tests/test_history_provider.py`

- [ ] **Step 1: Add failing provider tests**

Test cases:

```python
async def test_history_provider_returns_summary_plus_recent_without_overlap(
    provider, conversation, user, current_user_message, after_cursor_user, after_cursor_assistant
):
    context = await provider.build_context(
        conversation_id=conversation.id,
        user_id=user.id,
        current_message_id=current_user_message.id,
        agent_key="chat",
    )

    assert context.summary == "- Earlier billing discussion"
    assert [m.metadata["message_id"] for m in context.messages] == [
        str(after_cursor_user.id),
        str(after_cursor_assistant.id),
    ]
```

```python
async def test_history_provider_trims_by_agent_budget(
    provider, conversation, user, current_user_message, settings
):
    context = await provider.build_context(
        conversation_id=conversation.id,
        user_id=user.id,
        current_message_id=current_user_message.id,
        agent_key="rag",
    )

    assert len(context.messages) <= settings.rag_history_max_messages
    assert context.budget.agent_key == "rag"
```

- [ ] **Step 2: Implement provider types**

Create concrete dataclasses:

```python
@dataclass(frozen=True)
class HistoryBudget:
    agent_key: str
    max_messages: int
    max_tokens: int

@dataclass(frozen=True)
class ConversationHistoryContext:
    conversation_id: str
    user_id: str
    summary: str | None
    summary_message_id: str | None
    messages: list[AgentMessage]
    budget: HistoryBudget
```

Create `ConversationHistoryProvider`:

```python
class ConversationHistoryProvider:
    def __init__(
        self,
        message_repository: MessageRepository,
        summary_repository: ConversationMemorySummaryRepository,
        settings: Settings,
    ):
        self.message_repository = message_repository
        self.summary_repository = summary_repository
        self.settings = settings
        self._cache = TTLCache(maxsize=settings.memory_cache_max_conversations, ttl=settings.memory_cache_ttl_seconds)

    async def build_context(
        self,
        *,
        conversation_id: UUID | str,
        user_id: UUID | str,
        current_message_id: UUID | str | None,
        agent_key: str,
    ) -> ConversationHistoryContext:
        conversation_uuid = UUID(str(conversation_id))
        user_uuid = UUID(str(user_id))
        current_uuid = UUID(str(current_message_id)) if current_message_id else None
        summary = self.summary_repository.get_by_conversation_id(conversation_uuid)
        rows = self.message_repository.get_prompt_history(
            conversation_uuid,
            before_message_id=current_uuid,
            after_message_id=summary.last_summarized_message_id if summary else None,
            limit=self._budget_for(agent_key).max_messages or 50,
        )
        messages = [msg for row in rows if (msg := self._db_message_to_agent_message(row))]
        return self._build_context(conversation_uuid, user_uuid, agent_key, summary, messages)

    def invalidate(self, conversation_id: UUID | str) -> None:
        prefix = f"{conversation_id}:"
        for key in list(self._cache.keys()):
            if str(key).startswith(prefix):
                self._cache.pop(key, None)
```

Cache key must include `conversation_id`, `user_id`, `current_message_id`, `agent_key`, and summary cursor/version. Use a bounded TTL cache; do not keep an unbounded `dict`.

- [ ] **Step 3: Normalize DB messages once**

Add a private mapper:

```python
def _db_message_to_agent_message(message: Message) -> AgentMessage | None:
    if message.deleted_at is not None:
        return None
    if message.sender == MessageRole.user.value:
        return AgentMessage(
            role=MessageRole.USER,
            content=message.content,
            metadata={"message_id": str(message.id), "created_at": message.created_at.isoformat()},
        )
    if message.sender == MessageRole.assistant.value and message.content.strip():
        return AgentMessage(
            role=MessageRole.ASSISTANT,
            content=message.content,
            metadata={"message_id": str(message.id), "created_at": message.created_at.isoformat()},
        )
    return None
```

Keep attachment base64 out of prior prompt history. If attachments exist, include a short metadata note only when needed:

```text
[User attached 2 image(s) in this earlier turn.]
```

- [ ] **Step 4: Make `app/ai/memory.py` a compatibility shim**

Keep `get_memory_manager()` only if other code still imports it, but route new workflow calls through `ConversationHistoryProvider`. Remove process-global long-lived conversation objects from the new path.

- [ ] **Step 5: Run provider tests**

Run:

```bash
python -m pytest tests/test_history_provider.py -q
```

Expected: PASS.

---

## Task 4: Pass Stable Message IDs Through The Workflow Boundary

**Files:**
- Modify: `app/schemas/workflow.py`
- Modify: `app/ai/schemas.py`
- Modify: `app/services/message_service.py`
- Modify: `app/services/ai_service.py`
- Modify: `app/ai/graph.py`
- Test: `tests/test_message_history_pipeline.py`

- [ ] **Step 1: Add failing workflow-ID tests**

Test:

```python
async def test_message_service_passes_user_and_assistant_ids_to_workflow(
    service, message_create, user, assistant_id
):
    events = []
    fake_ai_service.execute_request_stream = capture_request(events)

    async for _ in service.create_message_stream(message_create, user.id, bot_message_id=assistant_id):
        pass

    request = events[0]
    assert request.user_message_id == str(created_user_message_id)
    assert request.assistant_message_id == str(assistant_id)
```

- [ ] **Step 2: Extend request schemas**

Add to both workflow request classes:

```python
user_message_id: str | None = None
assistant_message_id: str | None = None
```

- [ ] **Step 3: Pass IDs from message service**

In `create_message_stream()`:

- use the persisted `created_message.id` as `user_message_id`;
- pass `bot_message_id` as `assistant_message_id`;
- if `bot_message_id` is missing, generate one before workflow execution.

In non-streaming `create_message()`:

- generate an assistant UUID before workflow execution;
- pass it into the workflow request;
- persist the assistant response with that same ID.

- [ ] **Step 4: Add stable LangChain IDs in graph input**

In `_build_initial_state_from_request()`:

```python
initial_state["messages"] = [
    HumanMessage(content=request.message, id=request.user_message_id)
]
initial_state["user_message_id"] = request.user_message_id
initial_state["assistant_message_id"] = request.assistant_message_id
```

Add the corresponding optional keys to `GraphState`.

- [ ] **Step 5: Use assistant ID only for final assistant messages**

In `_finalize_agent_response()`:

```python
ai_kwargs = {"content": response.message.content}
if response.message.tool_calls:
    ai_kwargs["tool_calls"] = response.message.tool_calls
else:
    assistant_message_id = state.get("assistant_message_id")
    if assistant_message_id:
        ai_kwargs["id"] = assistant_message_id
state.setdefault("messages", []).append(AIMessage(**ai_kwargs))
```

This avoids reusing the final assistant DB ID for intermediate tool-calling AI messages.

- [ ] **Step 6: Run workflow-ID tests**

Run:

```bash
python -m pytest tests/test_message_history_pipeline.py -q
```

Expected: PASS.

---

## Task 5: Replace Graph History Loading With The Provider

**Files:**
- Modify: `app/ai/graph.py`
- Modify: `app/ai/agents/chat_agent.py`
- Modify: `app/ai/agents/search_agent.py`
- Modify: `app/ai/agents/rag_agent.py`
- Modify: `app/ai/agents/planning_agent.py`
- Modify: `app/ai/agents/canvas_agent.py`
- Modify: `app/ai/agents/image_generator_agent.py`
- Test: `tests/test_message_history_pipeline.py`

- [ ] **Step 1: Add tests proving no current-turn duplication**

Test:

```python
async def test_chat_node_uses_history_provider_and_excludes_current_message(monkeypatch):
    captured = {}
    provider = FakeHistoryProvider(
        messages=[AgentMessage(role=MessageRole.USER, content="previous")],
        summary="- earlier summary",
    )
    workflow.history_provider = provider
    workflow.chat_agent.invoke_model_with_history = capture_args(captured)

    await workflow._chat_node(state_with_current_user_message_id)

    assert [m.content for m in captured["conversation_history"]] == ["previous"]
    assert captured["history_summary"] == "- earlier summary"
    assert captured["messages"][-1].content == "current"
```

- [ ] **Step 2: Inject provider into `MultiAgentWorkflow`**

Update constructor signature:

```python
def __init__(
    self,
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    checkpointer: BaseCheckpointSaver | None = None,
    document_repository: Optional["DocumentRepository"] = None,
    runtime_model_resolver: IRuntimeModelResolver | None = None,
    history_provider: ConversationHistoryProvider | None = None,
):
    self.history_provider = history_provider
```

Keep a fallback provider construction only for tests that instantiate `MultiAgentWorkflow.__new__`.

- [ ] **Step 3: Replace `_get_conversation_history()` return type**

Change `_get_conversation_history()` to call:

```python
context = await self.history_provider.build_context(
    conversation_id=conversation_id,
    user_id=user_id,
    current_message_id=state.get("user_message_id"),
    agent_key=agent_key,
)
```

Return `ConversationHistoryContext`, or add a new `_get_history_context()` and migrate all nodes to it.

- [ ] **Step 4: Update every agent node**

Each node should pass:

```python
conversation_history=history_context.messages
history_summary=history_context.summary
```

Do this for chat, RAG, search, image generator, canvas, and planning.

- [ ] **Step 5: Remove duplicate legacy history prompt construction**

In `ChatAgent.invoke_model()` and `SearchAgent.invoke_model()`, stop building a user prompt that embeds `conversation_history` before calling `invoke_model_with_history()`.

Use:

```python
response = await self.invoke_model_with_history(
    [HumanMessage(content=message_content)],
    conversation_history,
    persona,
    conversation_id,
    user_id=request_user_id,
    device_id=request_device_id,
    model_request=model_request,
    history_summary=history_summary,
)
```

Leave `build_rag_prompt()` history handling in place for traditional RAG because that path does not call `BaseAgent.invoke_model_with_history()`.

- [ ] **Step 6: Run pipeline tests**

Run:

```bash
python -m pytest tests/test_message_history_pipeline.py tests/test_history_provider.py -q
```

Expected: PASS.

---

## Task 6: Move Long-Term Summarization Out Of The Streaming Hot Path

**Files:**
- Create: `app/ai/conversation_summarizer.py`
- Modify: `app/ai/summarization_middleware.py`
- Modify: `app/services/message_service.py`
- Modify: `app/ai/graph.py`
- Modify: `tests/test_graph_streaming_summarization.py`
- Test: `tests/test_message_history_pipeline.py`

- [ ] **Step 1: Replace stale streaming summarization tests**

Replace current tests with:

```python
async def test_streaming_does_not_call_summary_model_before_first_token(monkeypatch):
    summary_calls = []
    service.memory_summary_service = FakeSummaryService(summary_calls)

    events = [event async for event in service.create_message_stream(message_create, user.id)]

    assert events[0]["type"] == "user_message_created"
    assert any(event["type"] == "token" for event in events)
    assert summary_calls == ["scheduled_after_assistant_persist"]
```

```python
async def test_history_provider_uses_existing_summary_on_next_turn(
    provider, conversation, user, next_user_message
):
    context = await provider.build_context(
        conversation_id=conversation.id,
        user_id=user.id,
        current_message_id=next_user_message.id,
        agent_key="chat",
    )

    assert context.summary == "- Durable summary from previous turn"
```

- [ ] **Step 2: Create summarizer service wrapper**

Move model-call logic from `summarization_middleware.generate_summary()` into `ConversationSummarizer`:

```python
class ConversationSummarizer:
    async def summarize(
        self,
        *,
        existing_summary: str | None,
        messages: list[AgentMessage],
        user_id: str | None,
    ) -> str:
        config = self._get_config()
        return await asyncio.wait_for(
            self._generate_summary(
                existing_summary=existing_summary,
                messages=messages,
                user_id=user_id,
                config=config,
            ),
            timeout=config.timeout_seconds,
        )
```

Keep timeout and fail-closed behavior. Use per-user model credentials when available; otherwise use system Gemini config as the fallback.

- [ ] **Step 3: Add summary refresh method**

Add a service method:

```python
async def refresh_summary_after_turn(
    self,
    *,
    conversation_id: UUID,
    user_id: UUID,
    through_message_id: UUID,
) -> None:
    existing = self.summary_repository.get_by_conversation_id(conversation_id)
    rows = self.message_repository.get_summarization_window(
        conversation_id=conversation_id,
        after_message_id=existing.last_summarized_message_id if existing else None,
        through_message_id=through_message_id,
    )
    summarized_rows = rows[: -self.settings.summarization_keep_messages]
    if not self._should_summarize(summarized_rows):
        return
    agent_messages = [
        msg for row in summarized_rows if (msg := self.history_provider.db_message_to_agent_message(row))
    ]
    summary_text = await self.summarizer.summarize(
        existing_summary=existing.summary_text if existing else None,
        messages=agent_messages,
        user_id=str(user_id),
    )
    self.summary_repository.upsert(
        conversation_id=conversation_id,
        user_id=user_id,
        summary_text=summary_text,
        last_summarized_message_id=summarized_rows[-1].id,
        source_message_count=len(summarized_rows),
        estimated_tokens=estimate_tokens(summary_text),
    )
```

Rules:

- Load existing summary row.
- Load DB messages after `last_summarized_message_id` and up to `through_message_id`.
- Keep the newest `settings.summarization_keep_messages` out of the summary.
- Summarize only if message or token thresholds are reached.
- Upsert the new summary with `last_summarized_message_id` set to the newest summarized DB message ID.
- On timeout or model error, leave the previous summary unchanged.

- [ ] **Step 4: Schedule summary refresh after assistant persistence**

In `_persist_completed_workflow_response()` and resume completion:

```python
await self.memory_summary_service.refresh_summary_after_turn(
    conversation_id=conversation_id,
    user_id=user_id,
    through_message_id=bot_message.id,
)
```

If this is too slow for the request path, enqueue a Celery task and make the direct method available for tests. The request path must not wait longer than `settings.summarization_timeout_seconds`.

- [ ] **Step 5: Remove graph long-term summarization from START**

Remove the START -> `summarize` -> `route` edge for long-term memory. The graph should start at `route`. If checkpoint-local compaction is still needed, run it after completion, not before routing.

- [ ] **Step 6: Run tests**

Run:

```bash
python -m pytest tests/test_graph_streaming_summarization.py tests/test_message_history_pipeline.py -q
```

Expected: PASS.

---

## Task 7: Compact Checkpoint State After Terminal Completion

**Files:**
- Modify: `app/ai/graph.py`
- Modify: `app/workers/cleanup_tasks.py`
- Test: `tests/test_message_history_pipeline.py`

- [ ] **Step 1: Add checkpoint compaction tests**

Test:

```python
async def test_checkpoint_compaction_runs_after_complete_not_after_interrupt(monkeypatch):
    workflow.checkpointer = object()
    workflow.graph = FakeGraphWithState(messages=[m1, m2, m3], next=[])

    await workflow._compact_checkpoint_after_terminal_response(config, thread_id="conv-1")

    assert workflow.graph.updated_state["messages"] == [
        RemoveMessage(id=m1.id),
        RemoveMessage(id=m2.id),
        RemoveMessage(id=m3.id),
    ]
```

Test that compaction is skipped when `snapshot.next` is non-empty.

- [ ] **Step 2: Implement graph compaction helper**

Add:

```python
async def _compact_checkpoint_after_terminal_response(
    self,
    *,
    config: dict[str, Any] | None,
    thread_id: str | None,
) -> None:
    if not self.checkpointer or not config or not thread_id:
        return
    snapshot = await self.graph.aget_state(config)
    if snapshot.next:
        return
    messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    removals = [
        RemoveMessage(id=message.id)
        for message in messages
        if getattr(message, "id", None)
    ]
    if removals:
        await self.graph.aupdate_state(config, {"messages": removals})
```

Rules:

- No-op if no checkpointer or no thread ID.
- Load graph state.
- If interrupted, return without changes.
- Build `RemoveMessage` for every checkpoint message with an ID.
- Keep non-message state only if it is needed for active planning metadata; do not keep old `messages` as transcript memory.

- [ ] **Step 3: Call compaction after final response recovery**

Call after final `complete` response is yielded/persisted, never before response recovery and never on interrupt.

- [ ] **Step 4: Extend cleanup task**

`app/workers/cleanup_tasks.py` already deletes expired threads. Ensure conversation deletion and expired HITL cleanup call the checkpoint manager for thread IDs linked to the deleted/expired conversation.

- [ ] **Step 5: Run checkpoint tests**

Run:

```bash
python -m pytest tests/test_message_history_pipeline.py tests/test_checkpoint_serializer.py -q
```

Expected: PASS.

---

## Task 8: Make Cache Invalidation Complete And Explicit

**Files:**
- Modify: `app/services/message_service.py`
- Modify: `app/services/ai_service.py`
- Modify: `app/ai/history.py`
- Modify: `app/services/conversation_service.py`
- Test: `tests/test_message_history_pipeline.py`

- [ ] **Step 1: Add invalidation tests**

Assert invalidation happens after:

- user message creation;
- assistant message creation;
- partial assistant persistence on cancellation;
- interrupt assistant placeholder persistence;
- message update;
- message delete;
- conversation delete.

- [ ] **Step 2: Add one invalidation method**

Expose:

```python
def invalidate_conversation_memory(self, conversation_id: str) -> None:
    self.workflow.invalidate_history_cache(conversation_id)
```

In workflow, delegate to `history_provider.invalidate(conversation_id)`.

- [ ] **Step 3: Call invalidation on every transcript mutation**

Add invalidation after successful repository writes in `create_message`, `create_message_stream`, `_create_bot_response_message`, `update_message`, and `delete_message`.

- [ ] **Step 4: Clear memory on conversation delete**

In `ConversationService.delete_conversation()`, after repository delete succeeds:

- invalidate prompt-history cache;
- delete checkpoint thread state for that conversation if a checkpoint manager is available;
- leave DB messages soft-deleted behavior unchanged unless a separate product decision says to cascade.

- [ ] **Step 5: Run invalidation tests**

Run:

```bash
python -m pytest tests/test_message_history_pipeline.py -q
```

Expected: PASS.

---

## Task 9: Update API SDK And Sidecar History Assumptions

**Files:**
- Modify: `app/api/ai_sdk.py`
- Modify: `client_backend/services/server_api.py`
- Modify: `tests/client_backend/test_messages.py`
- Test: `tests/test_message_history_pipeline.py`

- [ ] **Step 1: Add AI SDK latest-message tests**

Test that `/api/chat/{conversation_id}`:

- extracts only the latest user message from the AI SDK payload;
- persists exactly one new user DB row;
- relies on server-side DB memory for previous turns;
- does not duplicate the AI SDK-provided history in the prompt.

- [ ] **Step 2: Fix sidecar `stream_message()` payload**

`client_backend/services/server_api.py::stream_message()` currently sends `{"content": content}` to `/api/chat/{conversation_id}`, but the server route expects a `messages` array payload.

Change it to:

```python
payload = {"messages": [{"role": "user", "content": content}]}
```

Preserve `device_id` injection.

- [ ] **Step 3: Run sidecar tests**

Run:

```bash
python -m pytest tests/client_backend/test_messages.py tests/test_message_history_pipeline.py -q
```

Expected: PASS.

---

## Task 10: Documentation And Production Settings

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `app/core/config.py`
- Test: config validation tests if present, otherwise import settings in a focused unit test.

- [ ] **Step 1: Add memory settings**

Add or document:

```text
MEMORY_CACHE_TTL_SECONDS=60
MEMORY_CACHE_MAX_CONVERSATIONS=256
MEMORY_SUMMARY_MIN_UNSUMMARIZED_MESSAGES=60
MEMORY_SUMMARY_MIN_UNSUMMARIZED_TOKENS=18000
MEMORY_SUMMARY_KEEP_MESSAGES=8
MEMORY_SUMMARY_MAX_TOKENS=1500
MEMORY_SUMMARY_TIMEOUT_SECONDS=30
```

If the existing `SUMMARIZATION_*` names remain, document that they now drive durable DB conversation memory instead of checkpoint memory.

- [ ] **Step 2: Update README flow**

Replace the AI workflow memory section with:

```text
1. Persist current user message.
2. Build prompt memory from durable summary plus recent DB messages after the summary cursor.
3. Execute graph for the current turn and tool/HITL state.
4. Persist assistant/interrupt/partial response.
5. Refresh durable summary after the turn when thresholds are exceeded.
6. Compact completed checkpoint state.
```

- [ ] **Step 3: Document operational behavior**

Document:

- summaries are best-effort and fail-closed;
- prompt history uses DB message IDs as cursors;
- deleted messages are excluded from future prompts;
- checkpoint state is not long-term memory;
- AI SDK clients may send full UI history, but the server uses the latest user message plus server-side memory.

- [ ] **Step 4: Run docs/config checks**

Run:

```bash
python -m pytest tests/test_config_redis.py tests/test_graph_streaming_summarization.py -q
```

Expected: PASS.

---

## Final Verification Matrix

Run this full set before claiming the refactor complete:

```bash
python -m pytest tests/test_history_provider.py -q
python -m pytest tests/test_conversation_memory_summary_repository.py -q
python -m pytest tests/test_message_history_pipeline.py -q
python -m pytest tests/test_graph_streaming_summarization.py -q
python -m pytest tests/client_backend/test_messages.py -q
python -m pytest tests/test_checkpoint_serializer.py -q
python -m pytest tests/test_router.py -q
```

Then run the broader suite if the local environment can support it:

```bash
python -m pytest
```

Manual smoke test:

1. Create a conversation.
2. Send three normal chat turns.
3. Confirm the fourth turn receives only prior turns in history, not the current user message twice.
4. Trigger a tool approval interrupt.
5. Refresh conversation messages and confirm the pending approval is recoverable from DB.
6. Resume approval and confirm the final assistant message is persisted once.
7. Send enough long turns to trigger durable summary refresh.
8. Confirm the prompt contains the durable summary plus recent messages after the summary cursor, with no overlap.
9. Delete the conversation and confirm future access fails and checkpoint cleanup is invoked.

---

## Acceptance Criteria

- Current user messages are excluded by stable `messages.id`, never by `exclude_last=1`.
- Prompt history is assembled in one place for all agents.
- Durable summaries use DB message cursors and do not duplicate recent prompt messages.
- Checkpoint state can support HITL resume but no longer acts as long-term transcript memory.
- Deleted and empty paused messages do not enter normal model history.
- Streaming no longer waits for a summary model call before the first user-facing token.
- AI SDK and internal message routes produce equivalent memory behavior.
- Cache invalidation is explicit for every transcript mutation.
- Targeted tests pass, including the currently failing streaming summarization tests after they are rewritten for the new architecture.
