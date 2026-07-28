# Demo Refactor, Conversation Search, and Stream Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep `demo.py` as one file while making conversation search complete and relevance-ranked, preserving the first Manage Conversations statistics element, keeping live status expanded, and removing only proven-redundant internal code.

**Architecture:** The existing paginated conversation route gains an optional `search` query that is forwarded through the sidecar and implemented as an owner-scoped SQLAlchemy query across titles and all non-deleted messages. `demo.py` keeps unfiltered lazy paging and adds separate server-backed search state, a stable loading placeholder, and a shared stream-status update policy; the cleanup remains inside the same file and is guarded by characterization tests.

**Tech Stack:** Python 3.10+, Streamlit, FastAPI, SQLAlchemy 2/PostgreSQL, Pydantic v2, pytest, Ruff.

## Global Constraints

- `demo.py` remains one file; no Streamlit UI logic is moved into new modules.
- Existing supported API routes, stored conversation formats, authentication behavior, message rendering, HITL behavior, and streaming event semantics remain compatible.
- Compatibility code is removed only when repository evidence and characterization tests show that it is redundant or unreachable.
- Search is scoped to conversations owned by the authenticated user and excludes soft-deleted conversations and messages.
- Blank search input behaves exactly like the existing unfiltered conversation list.
- Search ranking is exact title, title prefix, title substring, message content, then `updated_at DESC` and `id ASC` for deterministic ties.
- No database migration or external search dependency is introduced.

## File Structure

- `app/api/conversations.py`: accepts and forwards the optional canonical `search` query.
- `app/interfaces/conversation_service_interface.py`: records the updated service contract.
- `app/services/conversation_service.py`: normalizes search text and forwards it to the repository.
- `app/repositories/conversation.py`: owns the owner-scoped search predicate, rank expression, count, ordering, and preview loading.
- `client_backend/api/conversations.py`: forwards `search` through the local sidecar.
- `demo.py`: remains the one-file Streamlit app; owns encoded API calls, manager search state, the stable loading slot, status expansion policy, and scoped cleanup.
- `tests/test_conversation_search.py`: unit contract and compiled-SQL search coverage.
- `tests/client_backend/test_conversation_routes.py`: sidecar query-forwarding coverage.
- `tests/test_demo_conversation_manager.py`: Streamlit API/search/loading regressions.
- `tests/test_demo_live_ui_state.py`: live status/trace expansion regressions.
- `tests/test_demo_refactor_contract.py`: characterization and dead-code cleanup guardrails.

---

### Task 1: Add the canonical search contract and sidecar forwarding

**Files:**
- Create: `tests/test_conversation_search.py`
- Modify: `tests/client_backend/test_conversation_routes.py`
- Modify: `app/api/conversations.py:76-108`
- Modify: `app/interfaces/conversation_service_interface.py:34-49`
- Modify: `app/services/conversation_service.py:110-156`
- Modify: `client_backend/api/conversations.py:14-45`

**Interfaces:**
- Consumes: existing `ConversationPaginationParams`, `Paginator`, and conversation list route.
- Produces: `IConversationService.get_by_user_id(..., search: str | None = None)` and sidecar/canonical `GET /conversations/?search={query}` forwarding.

- [ ] **Step 1: Write failing route and service contract tests.**

```python
# tests/test_conversation_search.py
from __future__ import annotations

from uuid import uuid4

import pytest

from app.api.conversations import get_conversations
from app.repositories.utils.pagination import Paginator
from app.schemas.pagination import ConversationPaginationParams
from app.services.conversation_service import ConversationService


class _RecordingRepository:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def get_by_owner_id(self, owner_id, **kwargs):
        self.kwargs = {"owner_id": owner_id, **kwargs}
        return Paginator.create([], 0, kwargs["page"], kwargs["limit"])


class _UserValidation:
    def validate_user_exists(self, _owner_id) -> None:
        return None


class _ConversationValidation:
    pass


@pytest.mark.asyncio
async def test_conversation_route_forwards_search_to_service() -> None:
    owner_id = uuid4()

    class Service:
        def __init__(self) -> None:
            self.kwargs: dict = {}

        def get_by_user_id(self, user_id, **kwargs):
            self.kwargs = {"user_id": user_id, **kwargs}
            return Paginator.create([], 0, kwargs["page"], kwargs["limit"])

    service = Service()
    await get_conversations(
        conversation_service=service,
        user_id=owner_id,
        pagination=ConversationPaginationParams(page=2, limit=25),
        include=["messages"],
        latest_messages=3,
        search="  Roadmap  ",
    )

    assert service.kwargs["search"] == "  Roadmap  "


def test_conversation_service_normalizes_and_forwards_search() -> None:
    owner_id = uuid4()
    repository = _RecordingRepository()
    service = ConversationService(
        conversation_repository=repository,
        user_validation_utils=_UserValidation(),
        conversation_validation_utils=_ConversationValidation(),
    )

    service.get_by_user_id(owner_id, page=1, limit=10, search="  RoadMap  ")

    assert repository.kwargs["search"] == "roadmap"


def test_conversation_service_treats_blank_search_as_unfiltered() -> None:
    repository = _RecordingRepository()
    service = ConversationService(
        conversation_repository=repository,
        user_validation_utils=_UserValidation(),
        conversation_validation_utils=_ConversationValidation(),
    )

    service.get_by_user_id(uuid4(), search="   ")

    assert repository.kwargs["search"] is None
```

Extend the sidecar test request and expected params exactly as follows:

```python
response = client.get(
    "/conversations/"
    "?page=2&limit=5&orderBy=createdAt&orderDirection=asc"
    "&latestMessages=7&include=messages&include=feedback"
    "&search=Roadmap%20Q3"
)

assert calls[0]["params"] == {
    "page": 2,
    "limit": 5,
    "orderBy": "createdAt",
    "orderDirection": "asc",
    "include": ["messages", "feedback"],
    "latestMessages": 7,
    "search": "Roadmap Q3",
}
```

Keep the existing no-query request in `test_conversation_routes_use_proxy_server_request` and assert its params remain exactly `{"page": 1, "limit": 20, "include": [], "latestMessages": 3}` with no synthetic empty `search` key.

- [ ] **Step 2: Run the contract tests and verify RED.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_conversation_search.py tests\client_backend\test_conversation_routes.py
```

Expected: failures show that `search` is not accepted by the canonical route/service and is absent from sidecar params.

- [ ] **Step 3: Add the optional query through the canonical and sidecar layers.**

Use these exact signatures:

```python
# app/api/conversations.py
search: str | None = Query(default=None, max_length=200),

# app/interfaces/conversation_service_interface.py and app/services/conversation_service.py
def get_by_user_id(
    self,
    owner_id: UUID,
    page: int = 1,
    limit: int = 10,
    order_by: str = "updated_at",
    order_direction: str = "desc",
    include: list[str] = None,
    latest_messages: int = 3,
    search: str | None = None,
) -> Paginator[ConversationRead]:
```

Normalize in the service before the repository call:

```python
normalized_search = search.strip().lower() if isinstance(search, str) else ""
search = normalized_search or None
```

Add `search=search` to the repository call. In `client_backend/api/conversations.py`, accept the same optional query and add it to `params` only when `search is not None` so an absent parameter remains absent.

- [ ] **Step 4: Run the contract tests and verify GREEN.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_conversation_search.py tests\client_backend\test_conversation_routes.py
```

Expected: all route, normalization, blank-search, and proxy tests pass.

- [ ] **Step 5: Commit the contract increment.**

```powershell
git add app/api/conversations.py app/interfaces/conversation_service_interface.py app/services/conversation_service.py client_backend/api/conversations.py tests/test_conversation_search.py tests/client_backend/test_conversation_routes.py
git commit -m "feat: add conversation search contract"
```

### Task 2: Implement complete and deterministic repository search

**Files:**
- Modify: `tests/test_conversation_search.py`
- Modify: `app/repositories/conversation.py:1-155,157-225`

**Interfaces:**
- Consumes: normalized `search: str | None` from `ConversationService`.
- Produces: `ConversationRepository.get_by_owner_id(..., search: str | None = None)` returning a `Paginator[Conversation]` ranked across all owned titles and messages.

- [ ] **Step 1: Add failing compiled-query and pagination tests.**

```python
# append to tests/test_conversation_search.py
from sqlalchemy.dialects import postgresql

from app.repositories.conversation import _build_owned_conversation_queries


def _postgres_sql(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    ).lower()


def test_search_query_covers_titles_and_all_live_messages_with_stable_rank() -> None:
    owner_id = uuid4()
    count_statement, page_statement = _build_owned_conversation_queries(
        owner_id=owner_id,
        page=2,
        limit=10,
        order_by="updated_at",
        order_direction="desc",
        search="roadmap",
    )

    count_sql = _postgres_sql(count_statement)
    page_sql = _postgres_sql(page_statement)

    assert "conversations.owner_id" in page_sql
    assert "conversations.deleted_at is null" in page_sql
    assert "exists (select messages.id" in page_sql
    assert "messages.deleted_at is null" in page_sql
    assert "lower(messages.content)" in page_sql
    assert "case when" in page_sql
    assert "lower(conversations.title) = 'roadmap'" in page_sql
    assert "conversations.updated_at desc" in page_sql
    assert "conversations.id asc" in page_sql
    assert "offset 10" in page_sql
    assert "limit 10" in page_sql
    assert "count(conversations.id)" in count_sql


def test_blank_search_query_preserves_requested_sorting() -> None:
    _, page_statement = _build_owned_conversation_queries(
        owner_id=uuid4(),
        page=1,
        limit=20,
        order_by="created_at",
        order_direction="asc",
        search=None,
    )

    page_sql = _postgres_sql(page_statement)
    assert "case when" not in page_sql
    assert "conversations.created_at asc" in page_sql
```

Add this fake-session repository test to catch accidental message joins that duplicate conversations:

```python
from contextlib import contextmanager
from types import SimpleNamespace

from app.repositories.conversation import ConversationRepository


def test_repository_uses_filtered_count_and_keeps_distinct_page_order() -> None:
    conversations = [
        SimpleNamespace(id=uuid4(), title="Exact"),
        SimpleNamespace(id=uuid4(), title="Prefix"),
        SimpleNamespace(id=uuid4(), title="Message"),
    ]

    class ScalarResult:
        def __init__(self, items):
            self.items = items

        def scalars(self):
            return self

        def all(self):
            return self.items

    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _statement):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(scalar=lambda: 3)
            return ScalarResult(conversations)

    session = Session()

    @contextmanager
    def session_factory():
        yield session

    result = ConversationRepository(session_factory).get_by_owner_id(
        uuid4(), page=2, limit=3, search="roadmap"
    )

    assert result.items == conversations
    assert result.meta.total == 3
    assert result.meta.current_page == 2
```

- [ ] **Step 2: Run the repository tests and verify RED.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_conversation_search.py
```

Expected: collection or assertion failure because `_build_owned_conversation_queries` and repository `search` support do not exist.

- [ ] **Step 3: Implement a correlated-message search without duplicate rows.**

Add a module-level helper with this signature:

```python
def _build_owned_conversation_queries(
    *,
    owner_id: UUID,
    page: int,
    limit: int,
    order_by: str,
    order_direction: str,
    search: str | None,
) -> tuple[Any, Any]:
```

Build the base predicate from owner and `Conversation.deleted_at.is_(None)`. For nonblank search, construct a correlated `exists(select(Message.id)...)` requiring matching `conversation_id`, `Message.deleted_at.is_(None)`, and case-insensitive content containment. Build a SQLAlchemy `case` rank using exact title, title prefix, title containment, and message match. Use expression methods with `autoescape=True` so `%` and `_` in user input remain literal search characters. Count `Conversation.id` from the same predicate, then order the page query by rank, `Conversation.updated_at.desc()`, and `Conversation.id.asc()`.

For blank search, preserve the existing validated requested sort and direction. Apply offset and limit only to the page statement.

Update `ConversationCRUDStrategy.get_by_owner_id`, `get_with_recent_messages`, `count_by_owner_id`, and `ConversationRepository.get_by_owner_id` to accept `search`. Both include paths must use the same filtered count. Keep recent-message preview loading limited to `latest_messages`; it is display data and must not constrain the search predicate.

- [ ] **Step 4: Run repository tests and verify GREEN.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_conversation_search.py
```

Expected: all compiled-query, deduplication, pagination, service, and API contract tests pass.

- [ ] **Step 5: Run existing conversation and database regressions.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_database_session_provider.py tests\client_backend\test_conversation_routes.py tests\client_backend\test_live_server_integration.py
```

Expected: all selected tests pass; live-server tests may skip only under their existing environment guard.

- [ ] **Step 6: Commit the repository search.**

```powershell
git add app/repositories/conversation.py tests/test_conversation_search.py
git commit -m "feat: rank conversations across full history"
```

### Task 3: Make Streamlit search server-backed and paginated

**Files:**
- Create: `tests/test_demo_conversation_manager.py`
- Modify: `demo.py:2833-2854,3645-3734,9965-10187`

**Interfaces:**
- Consumes: `GET /conversations/?search={query}` from Tasks 1-2.
- Produces: `get_conversations(..., search: str | None = None)`, `_deduplicate_conversations(...)`, `_load_manager_page(page: int, *, search: str | None = None)`, and separate `manager_search_*` session state.

- [ ] **Step 1: Write failing encoded-request and manager-state tests.**

Use the existing Streamlit stub pattern from `tests/test_demo_plan_widget.py`, adding a placeholder object with `container()` and `empty()` call recording.

```python
def test_get_conversations_encodes_search_and_preview_params(monkeypatch):
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)
    endpoints: list[str] = []

    monkeypatch.setattr(
        demo,
        "make_api_request",
        lambda _method, endpoint, *_args: endpoints.append(endpoint)
        or {
            "success": True,
            "data": {
                "items": [],
                "meta": {"total": 0, "currentPage": 1, "lastPage": 1},
            },
        },
    )

    demo.get_conversations(
        page=1,
        limit=100,
        include_messages=True,
        latest_messages=3,
        search="Q3 roadmap & budget",
    )

    assert endpoints == [
        "/conversations/?page=1&limit=100&include=messages&latestMessages=3&search=Q3+roadmap+%26+budget"
    ]


def test_manager_search_state_is_separate_from_unfiltered_cache(monkeypatch):
    demo, streamlit = _import_demo_with_ui_stubs(monkeypatch)
    streamlit.session_state.manager_conversations = [{"id": "cached"}]
    streamlit.session_state.manager_search_conversations = []
    streamlit.session_state.manager_search_page = 0
    streamlit.session_state.manager_search_has_more = False
    streamlit.session_state.manager_search_total = 0
    calls: list[dict] = []
    monkeypatch.setattr(
        demo,
        "get_conversations",
        lambda **kwargs: calls.append(kwargs)
        or {
            "success": True,
            "data": {
                "items": [{"id": "matched"}],
                "meta": {"total": 1, "currentPage": 1, "lastPage": 1},
            },
        },
    )

    demo._load_manager_page(1, search="roadmap")

    assert calls[0]["search"] == "roadmap"
    assert streamlit.session_state.manager_conversations == [{"id": "cached"}]
    assert streamlit.session_state.manager_search_conversations == [{"id": "matched"}]


def test_failed_search_keeps_unfiltered_cache_and_remains_retryable(monkeypatch):
    demo, streamlit = _import_demo_with_ui_stubs(monkeypatch)
    streamlit.session_state.manager_conversations = [{"id": "cached"}]
    streamlit.session_state.manager_search_conversations = []
    streamlit.session_state.manager_search_page = 0
    streamlit.session_state.manager_search_has_more = False
    streamlit.session_state.manager_search_total = 0
    monkeypatch.setattr(demo, "get_conversations", lambda **_kwargs: {})

    demo._load_manager_page(1, search="roadmap")

    assert streamlit.session_state.manager_conversations == [{"id": "cached"}]
    assert streamlit.session_state.manager_search_page == 0
```

Add a test that changing `manager_search_query` from `alpha` to `beta` resets only the `manager_search_*` fields before fetching page 1, and a test that duplicate IDs from later pages are merged once in first-seen order.

- [ ] **Step 2: Run the Streamlit manager tests and verify RED.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_demo_conversation_manager.py
```

Expected: failures because `search`, top-level manager helpers, and search session state do not exist.

- [ ] **Step 3: Encode conversation query parameters centrally.**

Extend `get_conversations` with `search: str | None = None`. Replace string concatenation with an ordered list of pairs passed to `urlencode(..., doseq=True)`:

```python
params: list[tuple[str, Any]] = [("page", current_page), ("limit", limit)]
if include_messages:
    params.extend([("include", "messages"), ("latestMessages", latest_messages)])
if isinstance(search, str) and search.strip():
    params.append(("search", search.strip()))
endpoint = f"/conversations/?{urlencode(params, doseq=True)}"
```

Keep `fetch_all_pages` behavior for the sidebar refresh path, but search calls use ordinary server pagination.

- [ ] **Step 4: Move manager helpers to top-level and add search state.**

Move `_deduplicate_conversations` and `_load_manager_page` out of the dialog so they are independently testable. `_load_manager_page(page: int, *, search: str | None = None)` chooses unfiltered keys when `search is None` and `manager_search_*` keys otherwise. Add a `_reset_manager_search_state(query: str = "") -> None` helper and clear these keys in `open_conversation_manager` and `close_conversation_manager`.

In `render_manage_modal`, normalize the text input with `search_term.strip()`. When it differs from `manager_search_query`, reset search results and load page 1. Render search results from `manager_search_conversations`; otherwise render `manager_conversations`. Show Load more for either active result set using its own page, total, and `has_more` values. After loading another page, call `st.rerun()` so newly appended rows appear immediately.

Delete the current local title/latest-three-message filtering loop because the server result is authoritative and complete.

- [ ] **Step 5: Run manager tests and verify GREEN.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_demo_conversation_manager.py tests\test_demo_sidecar_auth.py
```

Expected: encoded URL, cache isolation, query reset, deduplication, and existing sign-out behavior pass.

- [ ] **Step 6: Commit the Streamlit search increment.**

```powershell
git add demo.py tests/test_demo_conversation_manager.py
git commit -m "fix: search every owned conversation"
```

### Task 4: Preserve the first conversation element across loading reruns

**Files:**
- Modify: `tests/test_demo_conversation_manager.py`
- Modify: `demo.py:9965-10187`

**Interfaces:**
- Consumes: top-level `_load_manager_page(page: int, *, search: str | None = None)` from Task 3.
- Produces: `_load_manager_page(page: int, *, loading_slot: Any, search: str | None = None)` plus one stable `manager_loading_slot` allocated on every dialog render and cleared after each fetch.

- [ ] **Step 1: Add failing stable-placeholder tests.**

```python
class _RecordingContext:
    def __init__(self, on_enter=None) -> None:
        self.on_enter = on_enter

    def __enter__(self):
        if self.on_enter is not None:
            self.on_enter()
        return self

    def __exit__(self, *_args) -> bool:
        return False


class _Placeholder:
    def __init__(self) -> None:
        self.empty_calls = 0
        self.container_calls = 0
        self.rendered_labels: list[str] = []

    def container(self):
        return _RecordingContext(
            lambda: setattr(self, "container_calls", self.container_calls + 1)
        )

    def empty(self) -> None:
        self.empty_calls += 1


def test_manager_loader_always_clears_transient_loading_content(monkeypatch):
    demo, streamlit = _import_demo_with_ui_stubs(monkeypatch)
    placeholder = _Placeholder()
    streamlit.spinner = lambda label: _RecordingContext(
        lambda: placeholder.rendered_labels.append(label)
    )
    monkeypatch.setattr(
        demo,
        "get_conversations",
        lambda **_kwargs: {
            "success": True,
            "data": {
                "items": [{"id": "first", "title": "Actual first conversation"}],
                "meta": {"total": 1, "currentPage": 1, "lastPage": 1},
            },
        },
    )

    demo._load_manager_page(1, search=None, loading_slot=placeholder)

    assert placeholder.empty_calls == 1
    assert placeholder.container_calls == 1
    assert placeholder.rendered_labels == ["Loading conversations..."]


def test_manager_dialog_has_no_durable_loading_status():
    source = Path("demo.py").read_text(encoding="utf-8")
    manager_source = source[source.index("def render_manage_modal") : source.index("def render_chunk_preview_modal")]
    assert 'st.status("Loading conversations..."' not in manager_source
    assert "manager_loading_slot = st.empty()" in manager_source
```

The `_Placeholder` test double must return a context manager from `container()` and record `empty()` calls. Extend the Streamlit stub's `spinner` to record the label and behave as a context manager.

- [ ] **Step 2: Run the loading regressions and verify RED.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_demo_conversation_manager.py -k "loading or placeholder or durable"
```

Expected: failure because the dialog still renders conditional `st.status` and the loader does not accept a stable slot.

- [ ] **Step 3: Implement the stable structural slot.**

At the beginning of `manage_dialog`, before any conditional load, allocate:

```python
manager_loading_slot = st.empty()
```

Pass this same slot to initial, search, and load-more fetches. `_load_manager_page` renders `st.spinner("Loading conversations...")` inside `loading_slot.container()` and calls `loading_slot.empty()` from `finally`, including failed requests. The placeholder allocation remains unconditional on every dialog rerun, so later elements retain the same relative positions; only its child content disappears.

- [ ] **Step 4: Run the complete manager test file and verify GREEN.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_demo_conversation_manager.py
```

Expected: all tests pass and no durable loading status remains in the manager source.

- [ ] **Step 5: Commit the identity fix.**

```powershell
git add demo.py tests/test_demo_conversation_manager.py
git commit -m "fix: preserve conversation manager element identity"
```

### Task 5: Keep live send and resume status expanded

**Files:**
- Modify: `tests/test_demo_live_ui_state.py`
- Modify: `demo.py:5934-5947,6852-6880,9096-9213,9777-9921`

**Interfaces:**
- Consumes: Streamlit status objects and current stream event loops.
- Produces: `_update_stream_status(status: Any, *, label: str, state: str = "running") -> None` with explicit expansion policy.

- [ ] **Step 1: Write failing helper and source-contract tests.**

```python
def test_running_stream_status_update_remains_expanded(monkeypatch):
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)

    class Status:
        def __init__(self) -> None:
            self.updates: list[dict[str, Any]] = []

        def update(self, **kwargs: Any) -> None:
            self.updates.append(kwargs)

    status = Status()
    demo._update_stream_status(status, label="Working...", state="running")
    assert status.updates == [
        {"label": "Working...", "state": "running", "expanded": True}
    ]


def test_token_transition_does_not_force_live_trace_closed():
    source = Path("demo.py").read_text(encoding="utf-8")
    resume_source = source[
        source.index("def _submit_interrupt_decisions") : source.index("def _render_plan_progress_widget")
    ]
    send_source = source[
        source.index("def render_chat_view") : source.index("def render_manage_modal")
    ]
    assert "st.session_state.stream_trace_expanded = False" not in resume_source
    assert "st.session_state.stream_trace_expanded = False" not in send_source
```

Extend the existing successful HITL resume test's `Status` double to record updates and assert the terminal call also includes `expanded=True`. Add a small streamed-send source contract asserting direct `status.update(` calls no longer occur inside `_submit_interrupt_decisions` or `render_chat_view`; all label changes use `_update_stream_status`.

- [ ] **Step 2: Run live UI tests and verify RED.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_demo_live_ui_state.py
```

Expected: failure because the helper is missing, direct updates omit `expanded`, and token paths assign `False`.

- [ ] **Step 3: Centralize status updates and remove forced trace collapse.**

Add:

```python
def _update_stream_status(
    status: Any,
    *,
    label: str,
    state: str = "running",
) -> None:
    status.update(label=label, state=state, expanded=True)
```

Replace every `status.update` in the send and resume streaming loops with this helper, preserving all existing labels and states exactly. Remove both token-event assignments that set `stream_trace_expanded = False`; retain `_reset_stream_trace_state(expanded=True)` and thinking-event assignments to `True`. Do not change stream event ordering, partial content, HITL reconciliation, image finalization, or terminal metadata.

- [ ] **Step 4: Run focused streaming regressions and verify GREEN.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_demo_live_ui_state.py tests\test_demo_stop_generation.py tests\test_hitl_demo_panel.py tests\test_demo_rich_response.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the expansion fix.**

```powershell
git add demo.py tests/test_demo_live_ui_state.py
git commit -m "fix: keep live chat status expanded"
```

### Task 6: Perform the scoped single-file cleanup under characterization tests

**Files:**
- Create: `tests/test_demo_refactor_contract.py`
- Modify: `demo.py:2722-2943,3645-3734,9096-9921,9965-10187`

**Interfaces:**
- Consumes: passing behavior tests from Tasks 3-5 and current `demo.py` public helper names.
- Produces: a less duplicated single-file implementation with no unused `fallback_conversation` parameter, no redundant initial `selected_agent`, and no nested duplicate manager helpers.

- [ ] **Step 1: Add characterization and cleanup guard tests before moving code.**

```python
# tests/test_demo_refactor_contract.py
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from tests.test_demo_plan_widget import _import_demo_with_ui_stubs


def test_refresh_conversations_list_has_no_unused_fallback_parameter(monkeypatch):
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)
    assert tuple(inspect.signature(demo.refresh_conversations_list).parameters) == ()


def test_manager_helpers_are_top_level_and_defined_once():
    tree = ast.parse(Path("demo.py").read_text(encoding="utf-8"))
    top_level = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert top_level.count("_deduplicate_conversations") == 1
    assert top_level.count("_load_manager_page") == 1


def test_supported_persisted_compatibility_reads_remain_present():
    source = Path("demo.py").read_text(encoding="utf-8")
    assert 'message_metadata.get("citations", [])' in source
    assert 'message_metadata.get("custom_agent_name")' in source
    assert "view.use_legacy_image_gallery" in source
```

Also add characterization tests for `_deduplicate_conversations` first-seen ordering and `get_conversations(fetch_all_pages=True)` aggregation, because those paths are touched by the cleanup.

- [ ] **Step 2: Run the characterization tests and verify the intended RED condition.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_demo_refactor_contract.py
```

Expected: the fallback-signature and top-level-helper assertions fail, while compatibility-retention and existing-behavior characterizations pass.

- [ ] **Step 3: Remove only proven redundancy and consolidate adjacent code.**

Make these bounded edits:

- remove `fallback_conversation` from `refresh_conversations_list` and delete its unreachable conditional block; repository-wide search already proves there are no callers supplying it;
- remove the unused `selected_agent = None` initialization in the send path while retaining the event-local value and `stream_selected_agent` session field;
- retain the now-top-level manager helpers from Task 3 and delete their old nested definitions;
- consolidate repeated manager pagination metadata assignment in `_load_manager_page` behind one key-prefix branch;
- keep `fetch_all_pages=True` for sidebar/full cache refresh and keep persisted-data compatibility reads for legacy citations, custom-agent labels, and rich image galleries;
- add concise section comments around conversation state/API, stream trace, and dialog helpers without moving code across unrelated subsystems.

Do not remove provider environment fallback, persisted citation/image readers, API error fallbacks, or supported v1 rich-response readers; the characterization test makes this boundary explicit.

- [ ] **Step 4: Run the refactor contract and all affected demo tests.**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_demo_refactor_contract.py tests\test_demo_conversation_manager.py tests\test_demo_live_ui_state.py tests\test_demo_plan_widget.py tests\test_demo_sidecar_auth.py tests\test_demo_stop_generation.py tests\test_demo_rich_response.py tests\test_hitl_demo_panel.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the behavior-preserving cleanup.**

```powershell
git add demo.py tests/test_demo_refactor_contract.py
git commit -m "refactor: simplify single-file Streamlit demo"
```

### Task 7: Verify the complete change and perform local UI acceptance

**Files:**
- Modify only if verification reveals a scoped defect: files already listed in Tasks 1-6.

**Interfaces:**
- Consumes: all search, manager, streaming, and compatibility tests.
- Produces: evidence that the implementation meets the approved specification.

- [ ] **Step 1: Run syntax, focused lint, and format checks.**

```powershell
.\.venv\Scripts\python.exe -m py_compile demo.py app\api\conversations.py app\interfaces\conversation_service_interface.py app\services\conversation_service.py app\repositories\conversation.py client_backend\api\conversations.py
.\.venv\Scripts\python.exe -m ruff check demo.py app\api\conversations.py app\interfaces\conversation_service_interface.py app\services\conversation_service.py app\repositories\conversation.py client_backend\api\conversations.py tests\test_conversation_search.py tests\test_demo_conversation_manager.py tests\test_demo_live_ui_state.py tests\test_demo_refactor_contract.py tests\client_backend\test_conversation_routes.py
.\.venv\Scripts\python.exe -m ruff format --check demo.py app\api\conversations.py app\interfaces\conversation_service_interface.py app\services\conversation_service.py app\repositories\conversation.py client_backend\api\conversations.py tests\test_conversation_search.py tests\test_demo_conversation_manager.py tests\test_demo_live_ui_state.py tests\test_demo_refactor_contract.py tests\client_backend\test_conversation_routes.py
```

Expected: compilation succeeds; Ruff reports no errors and no formatting changes required.

- [ ] **Step 2: Run the complete focused regression suite.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_conversation_search.py tests\client_backend\test_conversation_routes.py tests\test_demo_conversation_manager.py tests\test_demo_live_ui_state.py tests\test_demo_refactor_contract.py tests\test_demo_plan_widget.py tests\test_demo_sidecar_auth.py tests\test_demo_stop_generation.py tests\test_demo_rich_response.py tests\test_hitl_demo_panel.py tests\test_streamlit_width_deprecation.py
```

Expected: all selected tests pass with no unexpected warnings.

- [ ] **Step 3: Run broader conversation/API regressions.**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\client_backend\test_live_server_integration.py tests\test_custom_agents_api.py tests\test_message_history_pipeline.py tests\test_message_service_event_streaming.py
```

Expected: all runnable tests pass; environment-gated integration tests may report their existing skips.

- [ ] **Step 4: Perform browser acceptance when a browser surface is available.**

With the local server, sidecar, and Streamlit services running on ports 8000, 8100, and 8501:

1. Open Manage Conversations and confirm the first card keeps its real title and Message/Created/Persona statistics after dialog reruns.
2. Search for a unique phrase in a message older than the three-message preview and in a conversation beyond the first 100; confirm both are found.
3. Search an exact title also present only inside another conversation's message; confirm the exact-title result appears first.
4. Send a message that emits agent selection, thinking, a tool event, and tokens; confirm the status remains expanded through each label change.
5. Trigger and approve a HITL tool; confirm the resume status also remains expanded.

If no browser surface is available, record that limitation and rely on the automated component-state and request-contract tests without claiming visual acceptance.

- [ ] **Step 5: Inspect the final diff and working tree.**

```powershell
git diff --check HEAD~4..HEAD
git status --short --branch
```

Expected: no whitespace errors and no unintended files. Preserve any pre-existing user changes.
