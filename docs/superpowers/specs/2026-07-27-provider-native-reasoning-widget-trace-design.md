# Provider-Native Reasoning, Widget State, and Trace Design

## Problem

Three related gaps remain after the streaming interaction repair:

1. Gemini can emit final thinking content as structured content blocks, but the
   non-streaming/final response path only inspects a top-level `thinking`
   attribute. A thought summary visible during streaming can therefore be absent
   from the persisted message and its Streamlit or AI SDK rerender. Gemini's
   dynamic thinking can also legitimately produce no summary for a turn.
2. `widget_create` and `widget_update` ask the model to encode dynamic HTML state
   as a JSON string inside the outer tool-call JSON. Large HTML/CSS/JavaScript
   payloads require fragile double escaping and can fail before the tool runs.
3. Reasoning intensity exists as a request-only, provider-agnostic field with a
   fixed enum and lossy mappings. It is not persisted for the four primary
   agents, and the same model/level pair is not validated consistently across
   model settings, custom agents, message overrides, and planning subagents.

## Goals

- Preserve real provider-emitted thinking summaries through terminal persistence
  for both Streamlit and AI SDK clients without inventing an empty trace.
- Accept widget state as a native object while keeping generated interactions
  fully dynamic and the model guidance compact.
- Let users persist a reasoning control per primary agent from the Models page.
- Display and store the provider's exact values and terminology.
- Never send a reasoning value that is known to be unsupported by the selected
  model or a runtime fallback model.
- Apply one capability and validation policy to every endpoint or internal
  contract that accepts a provider, model, and reasoning setting.

## Non-Goals

- Guaranteeing a thought summary for every request. Providers may reason without
  returning a summary, especially under dynamic thinking.
- Exposing private chain-of-thought or fabricating reasoning text.
- Scraping provider documentation at runtime.
- Treating a numeric Gemini 2.5 `thinking_budget` as a named thinking level.
- Constraining the HTML, CSS, JavaScript, or subject matter generated for a
  widget beyond the existing validation and sandbox boundaries.

## Considered Approaches

### Fixed cross-provider levels

A shared list such as `disabled`, `medium`, and `high` is easy to render, but
those are not valid for every model. Mapping `none` to Gemini `minimal`, or
OpenAI `xhigh` to `high`, hides behavior changes and can still produce invalid
requests.

### Provider endpoint discovery only

The Gemini Models API exposes whether a model supports thinking, but not its
accepted level enum. OpenAI's Models API exposes only model identity and
ownership. Endpoint-only discovery cannot safely build the option list.

### Endpoint catalog plus official capability registry (selected)

Keep endpoint discovery as the source of available model IDs and provider
metadata, then enrich known IDs/families through a small, tested registry derived
from official model documentation. Unknown models remain usable with `Provider
default`, but the application does not guess a reasoning parameter for them.

## Architecture

### Shared reasoning-control descriptor

A provider-neutral resolver returns a descriptor for a concrete provider/model:

```text
supported
parameter_name       # thinking_level | reasoning.effort | null
display_label        # Thinking level | Reasoning effort
levels               # exact provider-native string values
default_level        # documented default when known
source               # provider_api | official_registry | unknown
```

The provider catalog keeps `supports_reasoning` for compatibility and adds the
descriptor to each normalized model entry. Provider API metadata wins when it
contains a relevant capability. The official registry supplies exact levels
when the API does not. Matching supports dated aliases and documented model
families while preferring an exact model rule over a family rule.

Initial registry coverage follows the official active catalogs, including the
Gemini 3.x and 2.5 matrices and OpenAI reasoning families. It includes newer
levels such as OpenAI `max` instead of truncating them to the old fixed set.
Models that do not support a named control, including a Gemini transport that
only accepts numeric `thinking_budget`, expose no named levels.

The UI always prepends `Provider default`. This is an application choice whose
stored value is `null`; it is not sent to the provider. All other option values
are shown and stored exactly as documented. Gemini uses the label `Thinking
level`; OpenAI uses `Reasoning effort`.

### Persistence and API contracts

`agent_model_configs` gains a nullable `reasoning_effort` column. The existing
field name is retained for backward compatibility, but its value is now the
provider-native value rather than a provider-agnostic intensity. Model-config
patches, effective snapshots, and the Models page read and write it per primary
agent.

Each catalog model returned by both provider-status and model-config options
endpoints includes the same reasoning-control descriptor. This prevents the two
public catalog representations from drifting.

The shared resolver and validator also cover:

- custom-agent create and update payloads;
- per-message `modelConfig` overrides used by Streamlit and AI SDK endpoints;
- planning subagent model overrides;
- persisted primary-agent settings; and
- runtime provider/model fallback.

`null` always means provider default. A non-null value must be one of the
resolved descriptor's exact levels. For a custom/unknown model, only `null` is
accepted unless a known family rule applies. Existing stored values are checked
when read so a registry or catalog change cannot cause an invalid request.

If runtime fallback changes the provider or model, capability resolution runs
again. An incompatible saved/requested value is omitted and a configuration
warning records the reason; it is never translated silently.

### Provider request construction

Runtime configuration carries the validated native value unchanged. Gemini 3
requests send it as `thinking_level`. OpenAI reasoning models send it as
`reasoning.effort`. When the value is `null`, unsupported, or unknown, request
construction omits the parameter and lets the provider choose its default.

Thought-summary inclusion remains independent from reasoning intensity. The app
continues requesting summaries where supported, but a high level does not imply
that a summary must be returned.

### Thinking-summary extraction and trace parity

A provider-neutral response helper extracts a public thought summary from:

- the existing top-level `response.thinking` compatibility attribute; and
- structured final content blocks whose type is `thinking` or provider-standard
  `reasoning` and whose public `thinking`, summary, or text field is present.

It does not include signatures, encrypted reasoning state, or ordinary answer
text. The extracted summary is added to the same terminal message metadata used
by the streaming trace. Because persistence happens before client-specific
adaptation, Streamlit and AI SDK history receive the same result.

The trace panel remains absent when no thought summary, reasoning summary, or
tool artifact exists. Tool-only turns still show an execution trace. Dynamic
thinking without a returned summary is treated as normal provider behavior.

### Native widget state

The model-facing widget tools accept `initial_state`/`state` as native JSON
objects rather than JSON strings. The object still uses the existing dynamic
shape, including generated `html`, bounded `height`, and optional metadata.
Internal parsing temporarily tolerates a legacy string from direct callers, but
the generated MCP schema advertises only an object so models do not double
encode it.

Prompt guidance is reduced to one compact rule: provide one self-contained
state object and do not wrap it in Markdown or additional prose. Structural and
security constraints stay in the tool schema, runtime validator, and sandbox,
not repeated in the agent prompt.

## Error Handling

- Invalid reasoning values return a model-specific validation message listing
  the accepted native values and `Provider default`.
- Unknown/custom models remain selectable, but explicit reasoning overrides are
  rejected rather than guessed.
- Runtime fallback drops an incompatible value, records a warning, and proceeds
  with the fallback provider default.
- Malformed legacy widget strings retain the existing actionable JSON error;
  native object calls avoid that parsing layer.
- Missing or signature-only thought blocks do not create an empty trace or an
  application error.

## Testing Strategy

- Unit-test exact and family capability resolution, documented defaults,
  non-reasoning models, custom models, and model aliases for both providers.
- Contract-test identical reasoning descriptors on provider and model-config
  catalog endpoints.
- Test save/read/reset and migration behavior for per-agent reasoning settings.
- Test valid and invalid levels through model-config patches, custom-agent
  create/update, message overrides, and planning subagent overrides.
- Test fallback across providers/models and assert incompatible levels are
  omitted with a warning rather than remapped.
- Assert Gemini and OpenAI model constructors receive the exact validated native
  value, including OpenAI `max` where supported.
- Test final Gemini `thinking` content blocks, top-level compatibility content,
  signature-only blocks, no-summary turns, and Streamlit/AI SDK persisted trace
  parity.
- Assert the widget MCP schema requires an object, and create/update a large
  quote-heavy HTML state successfully without double encoding. Retain legacy
  string parsing and sandbox validation tests.

## Acceptance Criteria

- The Models page offers a per-agent provider-native reasoning selector whose
  values change with the selected model and persist across sessions.
- Every endpoint and runtime path accepting reasoning uses the same model-aware
  validation policy.
- No unsupported or guessed reasoning value is sent to either provider.
- A Gemini final thinking block survives persistence and appears in both client
  histories; a genuine no-summary turn remains trace-free.
- Large dynamic widgets can be created on the first tool attempt with native
  object state, without expanding the main prompt materially.
