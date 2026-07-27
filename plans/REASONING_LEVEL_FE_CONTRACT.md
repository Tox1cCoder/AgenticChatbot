# Reasoning Level — Frontend Change Contract

## Scope

This is the delta contract for provider-native reasoning levels. It supplements
the existing frontend contracts; it does not redefine chat streaming, widgets,
or general model/provider management.

All routes require the normal authenticated `Authorization: Bearer <jwt>`
header. Responses use the standard envelope:

```json
{ "success": true, "message": "...", "data": {} }
```

## What changed

- Reasoning is model-specific, provider-native, and persisted per primary
  agent (`chat`, `rag`, `search`, `planning`).
- `Provider default` is represented on the wire by `null`. Do not send a
  string such as `"default"`.
- The available values must come from the selected model's
  `reasoningControl.levels`; do not maintain a frontend list or translate
  values between providers.
- The display label is model/provider supplied: for example `Thinking level`
  (Gemini) or `Reasoning effort` (OpenAI).
- Unknown or unsupported models intentionally expose no explicit levels. The
  UI must show only `Provider default` in that case.

## Read selectable models and reasoning levels

```http
GET /model-config/options
```

Use this as the canonical source for both the model picker and its reasoning
selector. Refresh it after a provider/model change or after the user requests a
catalog refresh.

Relevant response shape:

```json
{
  "success": true,
  "data": {
    "providers": [
      {
        "providerType": "gemini",
        "configured": true,
        "models": [
          {
            "id": "gemini-3.6-flash",
            "displayName": "Gemini 3.6 Flash",
            "providerType": "gemini",
            "supportsReasoning": true,
            "reasoningControl": {
              "supported": true,
              "parameterName": "thinking_level",
              "displayLabel": "Thinking level",
              "levels": ["minimal", "low", "medium", "high"],
              "defaultLevel": "medium",
              "source": "official_registry"
            }
          }
        ]
      }
    ],
    "agentConfig": {
      "chat": {
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "reasoningEffort": "medium"
      }
    }
  }
}
```

### `reasoningControl`

| Field | Type | Frontend rule |
|---|---|---|
| `supported` | boolean | Informational. Explicit selection is allowed only when `levels` is non-empty. |
| `displayLabel` | string | Use as the selector label. |
| `levels` | string[] | Render exactly, in this order, after `Provider default`. |
| `defaultLevel` | string or null | Informational provider default; do not preselect it when the saved value is null. |
| `parameterName` | string or null | Backend transport detail. Do not send it from the frontend. |
| `source` | string | Informational (`official_registry`, `provider_api`, or `unknown`). Do not branch UI behavior on it. |

`reasoningEffort` is nullable. When it is `null`, the UI selection is
`Provider default`, regardless of `defaultLevel`.

## Save a primary-agent setting

```http
PATCH /model-config
Content-Type: application/json
```

Send the exact selected native value, or `null` to reset that agent to provider
default. The API accepts `reasoningEffort` (preferred frontend casing); it also
accepts `reasoning_effort` for compatibility.

```json
{
  "chat": {
    "provider": "openai",
    "model": "gpt-5.6-sol",
    "reasoningEffort": "max"
  },
  "planning": {
    "reasoningEffort": null
  }
}
```

The response's `data` returns the effective per-agent configuration. Read its
`reasoning_effort` value as the persisted result; `null` means provider
default. Re-fetch `/model-config/options` if the model/provider changed so the
selector uses its current descriptor.

### Validation failure

An unsupported explicit value returns HTTP `400`:

```json
{
  "detail": "Thinking level 'max' is unsupported for gemini:gemini-3.6-flash. Accepted: minimal, low, medium, high."
}
```

Show the returned message, retain the user's unsaved selection if useful, then
refresh options. Never silently substitute another level.

## Reset all primary-agent settings

```http
POST /model-config/reset
```

No request body is required. This resets persisted model settings, including
`reasoning_effort`, to the application defaults. Use the returned effective
configuration and refresh `/model-config/options` before rendering selectors.

## Custom-agent create and edit

Reasoning level is now an optional field on custom agents.

```http
POST /custom-agents
PATCH /custom-agents/{customAgentId}
```

`/ai/custom-agents` and `/ai/custom-agents/{customAgentId}` are equivalent
aliases. Use `reasoningEffort` in frontend payloads:

```json
{
  "name": "Research analyst",
  "prompt": "...",
  "providerType": "openai",
  "model": "gpt-5.6-sol",
  "reasoningEffort": "high"
}
```

For an update, send `"reasoningEffort": null` to clear the saved override.
Use the model descriptor returned by `GET /custom-agents/options` (its
provider-model entries carry the same `reasoningControl` shape) to populate
this selector. Invalid values return HTTP `400`; do not save a guessed value.

## Per-message override

The existing message request accepts an optional `modelConfig` object. Its
agent override values use the service's snake-case field name:

```json
{
  "conversationId": "<conversation-uuid>",
  "content": "Compare the alternatives.",
  "modelConfig": {
    "chat": {
      "provider": "gemini",
      "model": "gemini-3.6-flash",
      "reasoning_effort": "high"
    }
  }
}
```

Omit `reasoning_effort` to use the persisted setting. Send it as `null` to use
the provider default for that one request. This override is not persisted.

## Selector algorithm

1. Read `/model-config/options`.
2. Find the selected model in the selected provider's `models` array.
3. Render `[Provider default, ...reasoningControl.levels]` using
   `reasoningControl.displayLabel`.
4. Select `Provider default` when the saved value is `null` or absent.
5. When the provider or model changes, clear an incompatible selected value to
   `null`, update the options immediately, and save only a value present in the
   new `levels` array.
6. Do not render a reasoning selector with guessed values for an unknown model.

## Not a frontend responsibility

- Do not map Gemini values to OpenAI values, or vice versa.
- Do not send `parameterName`; the backend chooses the provider request shape.
- Do not infer that a reasoning level guarantees a visible Thinking trace.
  Public reasoning summaries are provider-controlled and may be absent.
