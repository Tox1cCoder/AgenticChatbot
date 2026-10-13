# Fork conversation: frontend contract

A fork copies a conversation's history up to and including one completed assistant reply into a
new conversation. The child is independent: later turns in either conversation do not affect the
other. No new SSE protocol is involved.

## Route

`POST /conversations/{sourceConversationId}/fork` (sidecar: also `/api/conversations/{id}/fork`).
There is no `/ai/conversations/.../fork` alias; AI SDK clients call the route above.

```json
{
  "messageId": "uuid of the assistant reply to fork at",
  "requestId": "client-generated uuid for this fork action",
  "title": "optional, 1..255 chars"
}
```

- Response is `ApiResponse<ConversationRead>`. `201` for a new child, `200` when the same
  `requestId` is replayed (same child returned).
- Omitting `title` yields `"<source title> (fork)"`; no LLM is involved.
- `ConversationRead` carries lineage: `forkedFromConversationId`, `forkedFromMessageId`,
  `forkedThroughSequence`. Deletion is soft, so the first two may still be set after the source
  or message is deleted, pointing at a resource that answers 404: link only when the source is in
  the user's conversation list. `forkedThroughSequence` stays set, so
  `forkedThroughSequence != null` means "this is a fork".

## Request id rules

1. Generate one `requestId` per user action (per source + message pair), not per HTTP attempt.
2. Keep it across retries after a network error, a 5xx, or `409 conversation_turn_conflict`.
3. Disable the control while a request is in flight; a double click must not create two forks.
4. Drop it (mint a new one on the next click) after success or after `FORK_RESULT_DELETED` /
   `FORK_REQUEST_CONFLICT`. A new `requestId` intentionally creates another fork.

## Errors

| Status | Code | Meaning / UI action |
| --- | --- | --- |
| 404 | `FORK_MESSAGE_NOT_FOUND` | Reply missing, deleted, or not in this conversation. Refresh history. |
| 409 | `FORK_POINT_NOT_COMPLETED` | User row, empty placeholder, or unfinished/continued reply. |
| 409 | `conversation_turn_conflict` | A turn is running or the lock timed out. Retriable with the same `requestId`. |
| 409 | `FORK_REQUEST_CONFLICT` | `requestId` already used with a different message/title. Mint a new id. |
| 410 | `FORK_RESULT_DELETED` | The child from this `requestId` was deleted. Mint a new id to fork again. |
| 409 | `FORK_DOCUMENT_CONTEXT_UNSUPPORTED` | Source has conversation-scoped documents (Phase 1 limit). |
| 413 | `FORK_HISTORY_TOO_LARGE` | Prefix exceeds the configured row/byte bounds. |
| 404/403 | existing | Source missing, deleted, or not owned. |

## Navigation and cache

1. Call the fork route once.
2. Put the returned conversation in your conversation list (insert at the top).
3. Switch the selected conversation id to `data.id`.
4. Load the child's persisted messages from the server (`GET /conversations/{id}/messages`, or
   `GET /ai/conversations/{id}/messages` for AI SDK clients). Do not build child history by
   slicing the source transcript you hold in memory; it may be paginated and the server applies a
   snapshot policy to copied rows.
5. Send every later request (`/messages`, `/ai/...` chat, stop, continue) to the child id.
6. Do not show source-scoped client state in the child. Discard pending approval/interrupt UI,
   mounted live widgets and their subscriptions, composer attachments, and any message cache
   keyed by the source. A generation snapshot (Continue/Stop) may be kept for the user's return,
   but render and send it only in the conversation named by its `conversation_id`. The demo does
   this with `generation_snapshot_for`. Do not send a stop/cancel for
   the source as part of forking: the server rejects a fork while a source turn is active, so a
   stop could only alter a paused or stale source generation.

## Copied rows are history

Every copied message has `metadata.fork_history = true`. Copied rows are inert: execution handles
(interrupts, thread ids, continuation ids) and live widgets are not copied, and widget markers are
replaced by a text label. Clients must suppress interactive controls on rows with
`fork_history = true`: no Approve/Reject, no Continue, no live-widget mount, no re-run.
`metadata.fork_cutoff_eligible` (boolean) is the server's record of whether a copied reply was a
valid fork point in the source. A nested fork re-validates server-side; hide the fork control on a
copied row whose `fork_cutoff_eligible` is `false`, since the server answers 409 for it.
These fields reach AI SDK clients through the message metadata projection unchanged.

## Lineage display

Show "Forked from <source title>" and link to the source only when `forkedFromConversationId` is in
the user's conversation list. The id may still be set after the source is deleted, so do not link
on its presence alone. When it is `null` or not in the list but `forkedThroughSequence` is set,
show a non-linked "Forked from a deleted conversation" label; the child remains fully usable.

## Limits and measured cost

Measured with `scripts/benchmark_conversation_fork.py` against a local PostgreSQL 18.3 on one
Windows 11 host (32 logical CPUs, Python 3.13.7), through the real fork service and repository,
no model calls, about 9 KiB of metadata per assistant row; median of 3 timed forks after a warm-up.
These are single local host numbers; re-measure on the deployment target before raising
the 5,000-row / 32 MiB fork limits.

| Messages | Latency median (min-max) | Peak Python alloc | Rows copied | Bytes copied | SQL statements |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 78 ms (77-79) | 1.3 MiB | 100 | 0.5 MiB | 16 (1 executemany) |
| 1,000 | 215 ms (211-318) | 11.3 MiB | 1,000 | 5.2 MiB | 16 (1 executemany) |
| 5,000 | 895 ms (879-942) | 56.7 MiB | 5,000 | 26.0 MiB | 16 (1 executemany) |

The statement count does not grow with the prefix; latency and allocation grow linearly (about
0.17 ms and 11 KiB per row). Concurrent near-limit forks, a remote database and container memory
limits were not measured.

`tool_render` items of type `mcp_app` are copied as inert static results; any future iframe
mounting must respect `fork_history`.

## Streamlit demo

`demo.py` implements this flow: `fork_conversation_from_message`, `render_fork_control` (shown only
beside persisted assistant replies) and `render_fork_lineage`. Pure rules live in
`app/ui/conversation_fork.py`.
