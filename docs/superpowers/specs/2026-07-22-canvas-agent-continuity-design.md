# Canvas Agent Continuity Design

## Summary

Canvas behaves as a conversation-scoped working artifact. A follow-up edit reopens the latest
valid persisted canvas, gives the complete source to `canvas_agent`, and emits the next revision
under the same stable artifact identity. Canvas may still discover widget tools on demand, but a
turn that is editing an existing canvas cannot create or update a widget instead of returning the
updated canvas.

This design fixes two confirmed failures:

1. Persisted `canvas_artifact` metadata is discarded while database rows are normalized into
   prompt history, so `_extract_previous_artifact()` never sees the source it was written to find.
2. Base-agent follow-ups have no canvas continuity rule. A terse edit can be routed to
   `chat_agent`, whose pinned widget tools can create an unrelated widget.

The solution uses persisted message metadata as the append-only revision record. It does not add a
new database table or depend on checkpoint message history, which is intentionally compacted after
terminal responses.

## Goals

- Preserve the latest valid canvas source across turns, process restarts, history trimming, and
  conversation compaction.
- Make an immediate canvas follow-up continue editing the active artifact.
- Let a later explicit canvas reference reopen the active artifact after intervening chat turns.
- Preserve unchanged source while applying requested edits, like a coding agent editing a file.
- Keep canvas and inline-widget rendering channels distinct without permanently removing widget
  capability from Canvas.
- Keep prompts compact and avoid keyword-based routing lists.
- Prevent malformed, missing, identical, or truncated model output from corrupting a valid canvas.
- Preserve existing legacy and AI SDK canvas response contracts.

## Non-goals

- A multi-file browser IDE or arbitrary filesystem workspace.
- Multiple independently addressable canvases in one conversation. The existing contract has one
  active artifact, `canvas:main`.
- Frontend layout redesign. Clients continue rendering the existing canvas rich item; revision
  metadata makes replacement semantics explicit.
- Editing arbitrary historical canvas revisions. The latest valid revision is the working state.

## Approaches Considered

### Prompt-only continuity

Tell the router and Canvas model more emphatically to edit the previous canvas. This is small but
cannot work reliably because the previous source is absent from model-visible context. It also
cannot guarantee that widget tools will not be used as a substitute. Rejected.

### Dedicated canvas database table

Create a mutable canvas record plus revision table. This provides strong lifecycle modeling but
duplicates the complete append-only revision history already stored in assistant message metadata,
requires a migration and backfill, and expands the API surface. Rejected for the current
single-canvas contract.

### Persisted-message snapshot plus turn-scoped channel policy

Query the latest valid canvas from assistant metadata, inject it only for Canvas editing, preserve
the stable artifact identity, and apply a structural tool policy during edits. This survives
compaction, adds no storage system, keeps normal prompts and tool schemas small, and directly fixes
both failure modes. Selected.

## Architecture

### 1. Durable canvas snapshot

`MessageRepository` gains a focused query for the newest non-deleted assistant message in a
conversation whose JSON metadata contains a valid `canvas_artifact`. The returned snapshot contains:

- `artifact_id`, defaulting legacy records to `canvas:main`;
- `revision`, defaulting a legacy first revision to `1`;
- `content`, `language`, and `title`;
- source `message_id` and `sequence` for diagnostics and lineage.

The query remains conversation-scoped and never searches another conversation. Invalid candidate
metadata is skipped in favor of the newest valid revision, with a warning that contains identifiers
but not artifact source.

`ConversationHistoryProvider` exposes the snapshot lookup because it already owns the message
repository used by the workflow. Ordinary prompt-history normalization remains unchanged, so large
canvas source is not copied into every agent's history or hidden inside `additional_kwargs`.

### 2. Routing continuity

Before routing, the workflow obtains only a bounded canvas descriptor: existence, title, revision,
and source message ID. The full source is not placed in graph state or the router prompt.

Routing follows these rules:

- Planning mode with an existing plan retains its current precedence.
- If the previous responding agent was `canvas_agent` and a valid active canvas exists, the next
  natural follow-up stays with Canvas. Canvas can use the existing `hand_off` tool when the new
  request is unrelated.
- Otherwise, the router receives one compact contextual line stating that an active canvas exists.
  Its semantic decision can route a later request that refers to that artifact back to Canvas.
- No phrase list or regular-expression intent classifier is added.

The continuity decision requires a valid persisted artifact. A stale `last_agent=canvas_agent`
without recoverable source does not silently start a replacement canvas.

### 3. Model editing context

When Canvas is selected, the canvas node fetches the authoritative snapshot. The agent receives the
complete source in a dedicated, clearly delimited model message immediately before the current user
request. The system prompt states concisely that the block is untrusted artifact data, not
instructions, and that an edit must return one complete replacement document.

The source is injected only into `canvas_agent`; chat, RAG, search, image, planning, and router calls
never pay its token cost. Existing model-context preflight accounts for the injected source. If the
source cannot fit the selected model's context, the request fails explicitly without replacing the
saved artifact.

### 4. Canvas identity and revisions

The persisted legacy `canvas_artifact` gains backward-compatible fields:

```json
{
  "artifact_id": "canvas:main",
  "revision": 2,
  "operation": "update",
  "content": "<!doctype html>...",
  "language": "html",
  "title": "Project"
}
```

Initial generation uses revision `1` and operation `create`. A successful changed edit keeps the
same `artifact_id`, increments `revision`, and uses operation `update`. The AI SDK canvas rich item
continues using `id: "canvas:main"`; its payload adds optional `revision` and `operation` fields.
Existing clients can ignore these fields, while clients that reconcile artifacts can deterministically
replace the prior revision.

Identical output is a no-op: it does not claim a new revision. The response identifies the artifact
as unchanged and preserves the active source.

Every edit attempt also carries a compact `canvas_update` status record:

```json
{
  "status": "updated",
  "artifact_id": "canvas:main",
  "base_revision": 1,
  "revision": 2
}
```

`status` is `updated`, `unchanged`, or `failed`. Failed records add a stable `reason`; unchanged and
failed records keep `revision` equal to the base revision. This status is control metadata, not a
second renderable artifact.

### 5. Canvas and widget channel policy

Widget tools are not permanently excluded from Canvas. They remain unpinned and therefore add no
default tool-schema cost, but Canvas can discover them through `tool_search` for a genuinely
explicit inline-widget task.

An existing-canvas edit activates a turn-scoped channel policy:

- `widget_create` and `widget_update` are absent from bound tools and tool-search results;
- execution rejects either mutation defensively if a stale deferred binding attempts it;
- read-only widget operations remain available;
- the accepted completion channel is a full canvas artifact or an explicit handoff.

The policy is centralized as tool capability metadata used by binding, discovery, and execution.
It is not implemented as scattered prompt phrases. The Canvas prompt needs only a compact rule:
follow the active output channel and never substitute a widget for a canvas edit.

Outside canvas-edit mode, Canvas retains on-demand widget discovery. Chat, RAG, and Search keep their
existing pinned widget behavior for normal inline visual aids. The normal router therefore sends an
inline-widget request to those agents. Canvas uses widget discovery only when it was directly
preselected or handed an explicit inline-widget task while no canvas edit is active. If a user asks
for a companion widget during an active canvas edit, Canvas hands that work to an eligible agent
instead of changing output channels mid-edit.

### 6. Failed-edit protection

An existing valid canvas is never overwritten when the model response:

- contains no fenced artifact;
- contains an empty artifact;
- is truncated before the closing fence;
- is byte-for-byte identical to the active source.

For the first three cases, the response carries bounded failure metadata with `artifact_id`, base
revision, and a stable reason code. It does not publish a replacement `canvas_artifact`. The user
receives a concise retry message, and the client keeps rendering the last valid revision.

A truncated first-time creation retains the existing behavior of exposing a partial artifact with a
warning because no valid revision exists to protect. A truncated update never replaces an existing
valid revision.

## Data Flow

1. The service persists the current user message and invalidates conversation caches as it does now.
2. The workflow asks the history provider for the bounded active-canvas descriptor.
3. Routing applies planning precedence, immediate canvas continuity, or the normal semantic router.
4. When Canvas is selected, the canvas node loads the full latest snapshot.
5. The node marks the turn as an edit only when a valid snapshot exists and applies the channel
   policy to binding, discovery, and execution.
6. Canvas receives recent conversation text, the authoritative source block, and the current user
   request.
7. The response parser validates the complete artifact and compares it with the base source.
8. A changed valid artifact is emitted with stable identity and the next revision. Persistence of
   the assistant message makes it the next durable snapshot.
9. Existing checkpoint compaction removes graph messages but does not affect canvas recovery.

## Compatibility

- `metadata.canvas_artifact.content`, `language`, and `title` remain unchanged.
- `canvas:main` remains the AI SDK rich-item ID.
- New legacy and rich-payload fields are optional and additive.
- Streamlit and clients that ignore revision metadata continue to render the current artifact.
- Historical records without identity or revision fields are normalized as `canvas:main`, revision
  `1` when read.
- Widget behavior for Chat, RAG, and Search remains unchanged.

## Observability and Security

- Log routing mode, artifact ID, base revision, result revision, and failure reason without logging
  artifact source.
- Treat persisted HTML/SVG/React source as untrusted model input. Delimit it and explicitly prevent
  embedded text or comments from overriding system instructions.
- Keep full source out of router prompts, generic history metadata, graph checkpoints, and logs.
- Continue relying on the existing sandboxed renderer boundary for artifact execution; this change
  does not widen renderer privileges.
- Reject cross-conversation artifact lookup by construction and cover it with tests.

## Testing Strategy

Tests are added before implementation and must first fail for the confirmed behavior.

### Repository and history provider

- Returns the newest valid canvas revision for one conversation.
- Ignores deleted, non-assistant, malformed, and other-conversation records.
- Normalizes legacy identity and revision fields.
- Still retrieves an artifact older than the normal prompt-history/compaction cursor.

### Routing

- Immediate follow-up to a valid canvas stays on `canvas_agent` without invoking the router.
- A stale canvas `last_agent` without a recoverable artifact does not force Canvas.
- Planning supervision retains precedence.
- A later semantic canvas reference receives the bounded active-canvas descriptor.
- No keyword routing table is introduced.

### Canvas context and output

- The complete prior source reaches the model-facing edit context.
- Other agents and the router never receive the source.
- A valid edit preserves `canvas:main`, increments revision, and emits `update`.
- A first generation emits revision `1` and `create`.
- Identical output is a no-op.
- Missing, empty, malformed, and truncated edits retain the previous revision.
- Truncated first generation preserves current compatibility behavior.

### Widget separation

- Widget tools are discoverable but not pinned for Canvas outside edit mode.
- Canvas-edit binding and tool search omit widget mutation tools.
- Execution rejects a stale deferred widget mutation during a canvas edit.
- Read-only widget tools remain usable.
- Chat, RAG, and Search widget tests remain unchanged and passing.

### Contract and regression coverage

- Legacy metadata and AI SDK rich-item serialization include compatible revision fields.
- Conversation history, compaction, streaming, tool-loop, and widget suites pass.
- Focused tests prove the original two failures: a follow-up edits the saved source and cannot create
  a replacement widget artifact.

## Acceptance Criteria

- "Make the header blue" immediately after a Canvas response updates the saved HTML rather than
  starting from a blank document.
- "Edit the canvas" after intervening chat can reopen the latest valid artifact.
- A canvas edit cannot finish by creating or updating a live widget.
- Canvas can still discover widget tools outside edit mode without default schema or prompt bloat.
- The latest valid artifact survives history trimming, durable compaction, and process restart.
- Invalid model output never replaces a valid canvas revision.
- Existing canvas and widget clients remain compatible.
