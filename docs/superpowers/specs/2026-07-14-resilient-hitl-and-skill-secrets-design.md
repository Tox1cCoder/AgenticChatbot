# Resilient HITL and Skill Secrets Design

## Purpose

Make a claimed HITL resume recoverable for both the Streamlit operator UI and AI SDK web clients, while keeping resume execution first-write-wins and skill credentials device-local.

## Shared interrupt lifecycle

The existing `pending -> resolving` conditional update remains the sole authorization to resume a graph. No client retries a resume after it receives a duplicate conflict.

An interrupt reaches one terminal outcome after it is claimed:

- `resolved` when the resumed graph completes or pauses on a new interrupt.
- `failed` when the claimed graph emits an error, raises, or is cancelled before it reaches a terminal graph outcome.
- `expired` when it times out or becomes invalid because its execution scope changed.

`GET /hitl/interrupts/{interrupt_id}` is authenticated, owner-filtered, and exposes only `interruptId`, `conversationId`, `status`, `expiresAt`, and `updatedAt`. Its status set includes `pending`, `resolving`, `resolved`, `failed`, and `expired`.

## Error transport

The canonical API response envelope has a serialized optional `code` field. The local sidecar preserves that code when it relays non-stream HTTP errors.

Both stream protocols retain structured known-domain failures:

- Internal Streamlit SSE uses `status_code` and `error_code`.
- AI SDK SSE uses `statusCode` and `errorCode` alongside its existing `errorText` field.

Unknown exceptions retain the existing minimal error shapes. Error metadata is descriptive only; the durable state endpoint determines reconciliation.

## Client reconciliation

On submit, both clients set an interrupt-id-scoped lock before opening the stream and disable every action that can resume it. A duplicate code (`INTERRUPT_ALREADY_RESOLVED` or `INTERRUPT_CONFLICT`) triggers one uncached state read, never a second resume request.

- `pending`: remove the lock and restore approval controls.
- `resolving`: retain a stale-interrupt suppression marker and show only a status check / return-to-chat affordance.
- `resolved`: clear the paused UI, refresh history, and retain suppression until the stale paused message is no longer the latest recoverable interrupt.
- `failed` or `expired`: clear paused UI, present a new-message-required notice, and retain suppression for that stale interrupt.

State reads used for reconciliation bypass the Streamlit GET cache.

## Skill credentials

The existing `SkillSecretStore` remains the sole credential store: per signed-in user, local profile root, server identity, and skill. The UI accepts only user-supplied environment variable names and password-style values; it displays configured names only and clears the value widget after a successful write and on logout.

## Verification

Tests cover the complete lifecycle, including a claimed resume that emits an error or is cancelled, canonical API and sidecar error-envelope propagation, both stream error shapes, uncached reconciliation, and both clients' duplicate-submission state transitions. The AI SDK frontend contract documents its new state endpoint, error metadata, and no-replay behavior.
