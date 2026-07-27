# Reasoning Selector Compatibility Design

## Problem

The Streamlit reasoning selector can show only `Provider default` even for a
reasoning-capable model. Two conditions cause this:

- the running FastAPI process predates the backend capability metadata; and
- stable provider aliases such as `gemini-pro-latest` do not match an exact
  model registry rule.

The UI must offer explicit levels only when the backend can validate and send
those values safely.

## Design

The backend remains the single source of truth for reasoning controls. Exact
model IDs receive their documented provider-native levels. Stable aliases may
inherit a documented compatible family rule when the alias contract identifies
that family. Unknown or ambiguous models continue to expose only
`Provider default`.

Streamlit renders the descriptor returned with each catalog model and does not
maintain a second compatibility table. `Provider default` remains the nullable
choice and explicit values remain unchanged: Gemini uses thinking-level names;
OpenAI uses reasoning-effort names.

The initial compatibility expansion covers active reasoning families already
present in the synced catalog, including documented Gemini 2.5 and latest
aliases and OpenAI o-series models. Runtime request construction must support
the corresponding native control before a level is advertised. It must never
offer a value that would later be discarded or translated incompatibly.

The local FastAPI process is restarted after implementation so its catalog
responses include the new descriptors. Streamlit's cached model snapshot is
then refreshed through its existing reload behavior.

## Error Handling

- Unknown models offer only `Provider default`.
- An explicit value that becomes incompatible is rejected during save or
  omitted with an existing runtime-fallback warning when the model changes.
- Provider API failures do not replace the last safe compatibility descriptor
  with guessed values.

## Testing

- Exact model IDs return their documented levels.
- Stable aliases return only levels compatible with their documented family.
- Unknown aliases remain default-only.
- Provider/model-config response schemas preserve the descriptor.
- Streamlit renders every level supplied by the descriptor.
- Runtime constructors receive the selected native value for each supported
  provider family.

## Acceptance Criteria

- A supported catalog model shows a selectable compatible list in Streamlit.
- Changing provider or model updates that list immediately.
- Exact IDs, stable aliases, save validation, and runtime execution use one
  backend policy.
- No universal cross-provider list or prompt expansion is introduced.
