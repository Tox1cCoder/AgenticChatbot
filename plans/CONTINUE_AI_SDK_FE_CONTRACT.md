# Continue — AI SDK Frontend Contract

Continue resumes the same generation, agent, and saved evidence. It creates no
user message. Each run can save a new assistant message; keep earlier partials.

## Endpoint

`POST /ai/continue` — available on the server and client-backend proxy.
Use the same API base URL and authentication as chat. Send JSON:

```json
{
  "conversationId": "11111111-1111-1111-1111-111111111111",
  "generationId": "33333333-3333-3333-3333-333333333333",
  "continuationId": "44444444-4444-4444-4444-444444444444",
  "expectedVersion": 4,
  "idempotencyKey": "continue-unique-command-id",
  "inlineRichResponseV1": true,
  "streamWebAnswerV1": true
}
```

- Required: three UUIDs, `expectedVersion` (integer >= 1), and `idempotencyKey`
  (8–160 characters). Copy IDs/version from the latest server snapshot.
- Generate one key per command; reuse it for retries of that identical command.
- Flags are optional, default `false`; match chat capabilities on every request.
  `streamWebAnswerV1` requires handling `data-answer-reconcile` to replace drafts.
- No chat history, synthetic user prompt, or checkpoint/thread ID is needed.

## When to show Continue

Track `data-generation` parts from chat and continued streams. Their payload uses
**snake_case**, even though request bodies use camelCase:

```json
{
  "type": "data-generation",
  "id": "33333333-3333-3333-3333-333333333333",
  "data": {
    "phase": "continuation_available",
    "generation_id": "33333333-3333-3333-3333-333333333333",
    "conversation_id": "11111111-1111-1111-1111-111111111111",
    "status": "continuable",
    "version": 4,
    "execution_epoch": 0,
    "continuation_id": "44444444-4444-4444-4444-444444444444",
    "continuation_available": true
  }
}
```

- `phase`: `start`, `status`, or `continuation_available`. Parts are persistent.
  Merge present fields per generation; reset for a new one; reject older versions.
- Enable Continue only for the current conversation when status is `continuable`
  or `stopped`, `continuation_available === true`, and `continuation_id` exists.
- Disable during the request. Show `continuation_block_reason` when unavailable.
- `metadata.execution_paused` or stream closure alone does not enable Continue.

## Consume the response

- Response: SSE, `Content-Type: text/event-stream`,
  `x-vercel-ai-ui-message-stream: v1`; same AI SDK UI Message Stream as chat.
- POST through a continuation action/custom transport; use the chat stream parser.
  Accept the new `start.messageId` for this run; preserve previous messages.
- Keep AI SDK `resume: false`. Do not implement this with `resumeStream`,
  `regenerate`, or a normal `sendMessage` call.
- Handle text, reasoning, tools, rich items, sources, and reconciliation as chat.
  Route `data-interrupt` to approval UI and `POST /ai/resume-interrupt`.
- Another budget pause produces a new Continue offer. Each ID is single-use.
- `finish` / `[DONE]` ends the response, including paused responses. Read lifecycle
  state when the final generation status is needed.

## Restore after reopening the app

Paused partials and continued replies already persist `metadata.generation_id`;
AI SDK history preserves this **snake_case** metadata key. Read it from the
latest relevant assistant message returned by
`GET /ai/conversations/{conversationId}/messages`, then fetch the generation
snapshot below. Its camelCase `data.continuationAvailable`, `continuationId`,
`status`, and `version` determine the current offer. Saved `continuable` metadata
is historical and must not enable the button without that fresh read. Older
messages without a generation ID cannot be restored by guessing an ID.

## Stop during Continue

Use `POST /messages/stop` with `conversationId`, `generationId`,
`expectedVersion` from the latest generation snapshot, and `idempotencyKey`.
Generate a new key for each new Stop command; reuse it only for an identical
retry. Continue keeps the same generation and logical turn. Obtain the current
version from its `data-generation` events or the snapshot GET.

The legacy `userMessageId` form resolves the original logical turn, but currently
reuses a backend Stop key across epochs and can replay an earlier Stop. New FE
integrations must use the generation-based form. HTTP `202` means pending: poll
the snapshot until the worker confirms its state.

## Errors and recovery

Canonical HTTP JSON failures use `{ "success": false, "code": "...",
"message": "..." }`; validation errors use `code: "invalid_input"` and an
`error` field with details. A proxy may wrap this under `detail`, so keeping
`detail.code` as a compatibility fallback is fine.

HTTP `200` can still contain a streamed refusal: read
`data-stream-error.data.errorCode` (`statusCode` when supplied), then
`error.errorText`. Do not derive control-flow decisions from message text.

| Error code | Frontend action |
|---|---|
| `stale_command` | Refresh state; the saved version is outdated. |
| `continuation_unavailable` | Refresh state; the ID may be spent or blocked. |
| `illegal_transition` | Refresh state; command/key usage conflicts. |
| `generation_not_found` | Clear the unavailable offer. |

After refusal or disconnect, read
`GET /messages/generations/{generationId}?conversation_id={conversationId}`
and reload history as needed. Its `{ "success": true, "data": { ... } }` envelope
uses **camelCase** snapshot fields; normalize before merging. Redeemed requests
do not replay streams or start another run. Never auto-Continue on reconnect.

When a fresh snapshot allows Continue, paused UI copy should say
“Generation paused. Click Continue this answer to resume.” A new user message
starts another turn. If continuation is blocked, show the block reason instead
of instructions to click Continue.

## Streamlit compatibility check

- Core flow uses `POST /messages/continue`: same backend service and request
  schema. Keep this internal JSON SSE route; Streamlit expects `token`,
  `generation_start`, `continuation_available`, `message_end`, and `complete`.
- Confirmed gaps in `demo.py`: `build_continue_request` omits `inlineRichResponseV1`
  used by Send/Resume; `consume_continue_stream` ignores live `rich_items` and
  approval `interrupt` events, including their partial messages. Full UI parity
  needs these handlers; this documentation does not fix them.

See [the full AI SDK contract](AI_SDK_FE_CONTRACT.md) for shared stream behavior
and [the rich-item contract](AI_SDK_FE_RICH_ITEM_CONTRACT.md) for rendering.
