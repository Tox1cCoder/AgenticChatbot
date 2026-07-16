# Detached Message Feedback Serialization Fix

## Problem

`ConversationCompactionRepository.persist_message()` creates and commits a
`Message`, expunges it from its SQLAlchemy session, and returns it. The new
instance has never loaded its `feedback` relationship. When the service later
calls `MessageRead.model_validate(message)`, Pydantic reads the relationship
from the detached object and SQLAlchemy raises `DetachedInstanceError`.

## Design

Initialize the `feedback` relationship to `None` when constructing a newly
persisted message. A message cannot have feedback before its initial insert, so
this accurately represents its state while marking the relationship as loaded
before the object is detached. This avoids an unnecessary follow-up query and
does not change repository or service interfaces.

Do not change the global relationship loading strategy or session lifetime.
Those broader changes would affect unrelated query paths and would only mask
the persistence-boundary defect.

## Data Flow

1. The repository constructs the new `Message` with `feedback=None`.
2. It flushes and commits the message and any compaction job in the existing
   transaction.
3. It expunges and returns the message as before.
4. `MessageRead.model_validate()` reads the already-initialized relationship
   and emits `feedback: null` without attempting lazy loading.

## Error Handling

Existing transaction and error behavior remains unchanged. This fix only
ensures the valid no-feedback state survives detachment.

## Testing

Add a regression test that persists a new message through the real repository
path, verifies the returned ORM instance is detached, and validates it as a
`MessageRead` with `feedback is None`. Run the focused regression test, then the
related message/compaction tests.
