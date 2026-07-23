# AI SDK Stream Context Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every AI SDK event source is created, iterated, and closed in one asyncio task context while retaining heartbeat, error, and cancellation behavior.

**Architecture:** Replace the per-event `create_task(anext(source))` loop with one producer task that owns the async source and sends events through an internal queue. The adapter remains the queue consumer, emitting heartbeats on timed-out reads and propagating producer failures when it receives a completion sentinel.

**Tech Stack:** Python 3.11+, asyncio, ContextVar, pytest, pytest-asyncio

---

## File Structure

- Modify `app/services/event_streaming/ai_sdk_v6.py`: make one producer task own each event source for its complete lifecycle.
- Modify `tests/test_ai_sdk_v6_stream_contract.py`: add transport-level regressions for normal ContextVar cleanup and early stream closure.
- Verify `tests/client_backend/test_sse_keepalive.py`: retain existing heartbeat behavior across slow, fast, and interrupting sources.

### Task 1: Reproduce normal-completion context corruption

**Files:**
- Modify: `tests/test_ai_sdk_v6_stream_contract.py`
- Test: `tests/test_ai_sdk_v6_stream_contract.py`

- [ ] **Step 1: Add imports for a real ContextVar-backed scope**

Add these imports after `import json`:

```python
from contextlib import contextmanager
from contextvars import ContextVar
```

- [ ] **Step 2: Write the failing plain-text regression test**

Add this test after `test_text_stream_maps_to_ui_message_chunks`:

```python
@pytest.mark.asyncio
async def test_text_stream_preserves_source_context_until_cleanup():
    chat_scope = ContextVar("test_chat_scope", default=None)

    @contextmanager
    def bind_chat_scope():
        token = chat_scope.set("bound")
        try:
            yield
        finally:
            chat_scope.reset(token)

    async def source():
        with bind_chat_scope():
            yield make_event("message_delta", sequence=1, data={"text": "hello"})
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = await _collect_payloads(source)

    assert not any(
        payload != "[DONE]" and payload.get("type") == "error" for payload in payloads
    )
    assert any(
        payload != "[DONE]"
        and payload.get("type") == "text-delta"
        and payload.get("delta") == "hello"
        for payload in payloads
    )
    assert payloads[-1] == "[DONE]"
```

- [ ] **Step 3: Run the regression and verify the expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_v6_stream_contract.py::test_text_stream_preserves_source_context_until_cleanup -q
```

Expected: FAIL because the payload contains an `error` event whose `errorText`
includes `was created in a different Context`.

### Task 2: Reproduce early-close context corruption

**Files:**
- Modify: `tests/test_ai_sdk_v6_stream_contract.py`
- Test: `tests/test_ai_sdk_v6_stream_contract.py`

- [ ] **Step 1: Write the failing early-close regression test**

Add this test after the normal-completion regression:

```python
@pytest.mark.asyncio
async def test_heartbeat_stream_closes_source_in_its_own_context():
    chat_scope = ContextVar("test_close_chat_scope", default=None)
    cleaned = asyncio.Event()
    block_forever = asyncio.Event()

    async def source():
        token = chat_scope.set("bound")
        try:
            yield make_event("message_delta", sequence=1, data={"text": "hello"})
            await block_forever.wait()
        finally:
            chat_scope.reset(token)
            cleaned.set()

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
        heartbeat_interval_seconds=1,
    )
    events = adapter._events_with_heartbeats()

    first = await anext(events)
    await events.aclose()

    assert first.type == "message_delta"
    await asyncio.wait_for(cleaned.wait(), timeout=0.1)
```

- [ ] **Step 2: Add the required asyncio import**

Add this import above `import json`:

```python
import asyncio
```

- [ ] **Step 3: Run the early-close regression and verify the expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_v6_stream_contract.py::test_heartbeat_stream_closes_source_in_its_own_context -q
```

Expected: FAIL with `TimeoutError` because the old response-task `aclose()`
suppresses the different-context reset error before `cleaned.set()` executes.

### Task 3: Give the async source one lifecycle owner

**Files:**
- Modify: `app/services/event_streaming/ai_sdk_v6.py`
- Test: `tests/test_ai_sdk_v6_stream_contract.py`

- [ ] **Step 1: Replace the per-item task loop with a producer and queue**

Replace `AISDKV6StreamAdapter._events_with_heartbeats` with:

```python
    async def _events_with_heartbeats(
        self,
    ) -> AsyncGenerator[V3StreamEvent, None]:
        queue: asyncio.Queue[V3StreamEvent | object] = asyncio.Queue()
        source_complete = object()

        async def produce_events() -> None:
            source = self._event_source_factory()
            try:
                async for event in source:
                    await queue.put(event)
            finally:
                aclose = getattr(source, "aclose", None)
                if callable(aclose):
                    with contextlib.suppress(Exception):
                        await aclose()
                await queue.put(source_complete)

        producer = asyncio.create_task(produce_events())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=self._heartbeat_interval,
                    )
                except asyncio.TimeoutError:
                    yield make_event("heartbeat", sequence=0)
                    continue

                if item is source_complete:
                    await producer
                    break
                yield item
        finally:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer
```

This is the complete lifecycle change. Delete the old `source`, `pending_next`,
per-event `create_task(anext(source))`, `asyncio.wait`, and response-task
`aclose()` branches rather than retaining compatibility code around them.

- [ ] **Step 2: Run both context lifecycle regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_v6_stream_contract.py::test_text_stream_preserves_source_context_until_cleanup tests/test_ai_sdk_v6_stream_contract.py::test_heartbeat_stream_closes_source_in_its_own_context -q
```

Expected: `2 passed`.

- [ ] **Step 3: Run the complete AI SDK stream-contract module**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_v6_stream_contract.py -q
```

Expected: all tests pass with no warnings caused by unclosed tasks or async
generators.

- [ ] **Step 4: Commit the tested root fix**

```powershell
git add app/services/event_streaming/ai_sdk_v6.py tests/test_ai_sdk_v6_stream_contract.py
git commit -m "fix: preserve AI SDK stream task context"
```

### Task 4: Verify heartbeat and integration contracts

**Files:**
- Verify: `tests/client_backend/test_sse_keepalive.py`
- Verify: `tests/test_ai_sdk_v6_stream_contract.py`

- [ ] **Step 1: Run the focused heartbeat suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/client_backend/test_sse_keepalive.py -q
```

Expected: all available tests pass; dependency-gated tests may be reported as
skipped only when their documented server dependencies are unavailable.

- [ ] **Step 2: Run both affected suites together**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_v6_stream_contract.py tests/client_backend/test_sse_keepalive.py -q
```

Expected: all available tests pass, with no `different Context`, pending-task,
or unclosed-async-generator warnings.

- [ ] **Step 3: Check formatting and static analysis for touched files**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check app/services/event_streaming/ai_sdk_v6.py tests/test_ai_sdk_v6_stream_contract.py
.\.venv\Scripts\python.exe -m ruff format --check app/services/event_streaming/ai_sdk_v6.py tests/test_ai_sdk_v6_stream_contract.py
```

Expected: both commands exit successfully.

- [ ] **Step 4: Confirm the diff contains no unrelated cleanup**

Run:

```powershell
git diff HEAD^ --check
git diff HEAD^ -- app/services/event_streaming/ai_sdk_v6.py tests/test_ai_sdk_v6_stream_contract.py
```

Expected: the diff contains only the two regressions and the replacement of the
per-item heartbeat task lifecycle.
