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
  `forkedThroughSequence`. The first two become `null` if the source or message is later deleted;
  `forkedThroughSequence` stays set, so `forkedThroughSequence != null` means "this is a fork".

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
6. Discard source-scoped client state when switching: the generation snapshot (Continue/Stop),
   pending approval/interrupt UI, mounted live widgets and their subscriptions, composer
   attachments, and any message cache keyed by the source. Capture the source stream's
   conversation/generation id *before* discarding if you still need to cancel it; a stop must
   target the source id, never the child.

## Copied rows are history

Every copied message has `metadata.fork_history = true`. Copied rows are inert: execution handles
(interrupts, thread ids, continuation ids) and live widgets are not copied, and widget markers are
replaced by a text label. Clients must suppress interactive controls on rows with
`fork_history = true`: no Approve/Reject, no Continue, no live-widget mount, no re-run.
`metadata.fork_cutoff_eligible` (boolean) is the server's record of whether a copied reply was a
valid fork point in the source; a nested fork re-validates server-side, so treat it as informational.
These fields reach AI SDK clients through the message metadata projection unchanged.

## Lineage display

Show "Forked from <source title>" and link to the source when `forkedFromConversationId` is in the
user's list. When it is `null` (or not in the list) but `forkedThroughSequence` is set, show a
non-linked "Forked from a deleted conversation" label; the child remains fully usable.

## Streamlit demo

`demo.py` implements this flow: `fork_conversation_from_message`, `render_fork_control` (shown only
beside persisted assistant replies) and `render_fork_lineage`. Pure rules live in
`app/ui/conversation_fork.py`.
