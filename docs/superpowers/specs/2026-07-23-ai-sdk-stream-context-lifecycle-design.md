# AI SDK Stream Context Lifecycle Design

## Problem

The AI SDK heartbeat adapter advances its async event source by creating a new
`asyncio.Task` for each `anext(source)` call. An async generator may therefore
enter a `ContextVar`-backed scope in one task context and leave it in another.
When the scope resets its token, Python raises:

```text
ValueError: <Token ...> was created in a different Context
```

The image-loader scope exposed the defect, but the defect is transport-wide.
Plain-text chat follows the same AI SDK stream path and can fail during stream
cleanup. The resulting error can then be persisted as assistant content and
displayed by another client such as Streamlit.

## Scope

Fix the shared AI SDK heartbeat stream lifecycle for all response types.
Remove the obsolete per-item task and cross-context close machinery from that
path. Preserve the existing AI SDK wire protocol, heartbeat behavior, error
projection, and unrelated event-stream compatibility fallbacks.

Previously persisted error responses are not modified. Their missing assistant
content cannot be reconstructed safely.

## Considered Approaches

### Dedicated source-owner task (selected)

Create one producer task that constructs, iterates, and closes the async source.
Send produced events to the response task through a queue. The response task
waits with a timeout and emits heartbeat events while the queue is idle.

This gives the source one stable task context, retains keepalives, and provides
one deterministic cleanup owner.

### Reused explicit context

Create each `anext()` task with the same explicit `contextvars.Context`.
Although the code change is smaller, it depends on task-context APIs and keeps
the fragmented per-item lifecycle and close paths.

### Inline source iteration without heartbeats

Consume the source directly in the response task. This naturally preserves
context but removes keepalives, exposing long-running generations to proxy and
client timeouts.

## Architecture and Data Flow

`AISDKV6StreamAdapter._events_with_heartbeats()` creates a queue and starts one
producer task. The source factory is invoked inside that producer task. The
producer iterates the source to exhaustion and places each canonical event on
the queue. It then places a private completion sentinel on the queue.

The response task only consumes the queue. A timed-out queue read produces the
existing heartbeat event. A regular item is yielded unchanged. When the
completion sentinel arrives, the response task awaits the producer so any
source exception propagates through the existing AI SDK exception projection.

The current `create_task(anext(source))` loop and response-task `aclose()` call
are removed. They are redundant once the producer owns the full source
lifecycle.

## Error Handling and Cancellation

- Normal exhaustion signals completion and preserves the existing
  `finish`/`[DONE]` sequence.
- Source exceptions propagate from the producer and retain the existing
  AI SDK error payload mapping.
- On early response termination or disconnect, the response task cancels and
  awaits the producer.
- The producer finalizes the source in its own task context, suppressing only
  secondary close failures so they do not replace a primary stream exception.
- Heartbeats remain transient transport events and are never passed into the
  source or persisted as message content.

## Verification

Add a regression test whose plain-text source enters a `ContextVar` scope,
yields message and completion events, and resets the token during cleanup. The
stream must finish without an AI SDK error event.

Retain and run coverage for:

- slow sources emitting heartbeat events;
- fast sources completing without unnecessary heartbeats;
- source exceptions retaining status and error-code metadata;
- interrupt terminal behavior;
- early cancellation closing the source in its owning context;
- the broader AI SDK v6 stream contract.

The regression test must fail against the current per-item task implementation
with the reported different-context error before the production change is
made.
