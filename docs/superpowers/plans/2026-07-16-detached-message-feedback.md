# Detached Message Feedback Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure a newly persisted and detached `Message` can be serialized as `MessageRead` without lazy-loading `feedback`.

**Architecture:** Preserve the existing repository and service boundaries. Mark the necessarily empty `feedback` relationship as loaded when constructing a brand-new message, before the repository commits and detaches it.

**Tech Stack:** Python, SQLAlchemy 2.x ORM, Pydantic 2.x, pytest

---

## File Structure

- Modify `tests/test_conversation_compaction_repository.py`: add a focused repository regression test using a minimal session stand-in that applies SQLAlchemy's real detached-instance state transition.
- Modify `app/repositories/conversation_compaction.py`: initialize the new message's `feedback` relationship to `None` at construction.

### Task 1: Preserve the No-Feedback State Across Detachment

**Files:**
- Modify: `tests/test_conversation_compaction_repository.py`
- Modify: `app/repositories/conversation_compaction.py:180-210`

- [ ] **Step 1: Write the failing regression test**

Add these imports to `tests/test_conversation_compaction_repository.py`:

```python
from sqlalchemy import inspect
from sqlalchemy.orm import make_transient_to_detached

from app.models.enums import MessageRole
from app.schemas.message import MessageRead
```

Add the following focused test support and regression test:

```python
class _AllocatedSequenceResult:
    def scalar_one_or_none(self) -> int:
        return 1


class _DetachingPersistenceSession:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, _statement):
        return _AllocatedSequenceResult()

    def add(self, message) -> None:
        self.message = message

    def flush(self) -> None:
        now = datetime.now(timezone.utc)
        self.message.created_at = now
        self.message.updated_at = now
        self.message.deleted_at = None

    def commit(self) -> None:
        pass

    def expunge(self, message) -> None:
        make_transient_to_detached(message)


def test_persisted_message_serializes_feedback_after_session_detaches_it() -> None:
    session = _DetachingPersistenceSession()
    repository = ConversationCompactionRepository(lambda: session)

    message = repository.persist_message(
        {
            "id": uuid4(),
            "conversation_id": uuid4(),
            "sender": MessageRole.user.value,
            "content": "hello",
            "message_metadata": {},
        }
    )

    assert inspect(message).detached
    assert MessageRead.model_validate(message).feedback is None
```

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_compaction_repository.py::test_persisted_message_serializes_feedback_after_session_detaches_it -v
```

Expected: `FAIL`; Pydantic reports a `DetachedInstanceError` while extracting the `feedback` attribute. The fixture explicitly initializes scalar fields so `feedback` is the only failing field.

- [ ] **Step 3: Implement the minimal fix**

In `ConversationCompactionRepository.persist_message()`, replace the message construction with:

```python
message = Message(
    **data,
    sequence=int(allocated),
    feedback=None,
)
```

This is valid because feedback cannot exist before the message itself is inserted. Assignment records the relationship's empty value in SQLAlchemy's instance state, so expunging the object does not leave a lazy load pending.

- [ ] **Step 4: Run the regression test and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_compaction_repository.py::test_persisted_message_serializes_feedback_after_session_detaches_it -v
```

Expected: `1 passed`.

- [ ] **Step 5: Run related repository and message-path tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_compaction_repository.py tests/test_conversation_compaction_message_paths.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Exercise PostgreSQL integration discovery**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_conversation_compaction_postgres.py -q
```

Expected: tests pass when `TEST_DATABASE_URL` is configured; otherwise the module's PostgreSQL tests are reported as skipped, with no failures.

- [ ] **Step 7: Run focused static and diff checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check app/repositories/conversation_compaction.py tests/test_conversation_compaction_repository.py
git diff --check
```

Expected: both commands exit successfully with no lint or whitespace errors.

- [ ] **Step 8: Review and commit only the fix files**

Run:

```powershell
git diff -- app/repositories/conversation_compaction.py tests/test_conversation_compaction_repository.py
git add -- app/repositories/conversation_compaction.py tests/test_conversation_compaction_repository.py
git commit -m "fix: preserve message feedback state after detach"
```

Confirm the diff contains only the regression test and the relationship initialization before committing.
