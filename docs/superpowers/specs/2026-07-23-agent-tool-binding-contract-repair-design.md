# Agent Tool-Binding Contract Repair Design

## Problem

`BaseAgent` added an optional `excluded_tool_names` parameter to its tool-binding
methods. `CustomAgent` and `PlanningAgent` override those methods but were not
updated to accept the expanded contract. Because `BaseAgent` always supplies the
keyword, affected agents fail before model invocation with an unexpected-keyword
`TypeError`. The normal error wrapper then replaces that exception with a generic
assistant error message.

The selection and handoff mechanisms are not the source of the failure. Direct
selection and handoff both eventually invoke the same incompatible custom-agent
method.

## Goals

- Restore direct and handed-off custom-agent execution.
- Restore planning-agent execution affected by the same contract drift.
- Preserve explicit, statically inspectable method signatures.
- Apply tool exclusions consistently rather than merely accepting and ignoring
  the parameter.
- Add regression coverage for method compatibility and exclusion behavior.

## Non-Goals

- Changing routing or handoff semantics.
- Changing the custom-agent persistence model or capability resolver.
- Changing frontend or streaming event contracts.
- Refactoring unrelated tool-binding behavior.

## Design

### CustomAgent

Add `excluded_tool_names` to `CustomAgent._get_tools_for_binding` with the same
type and default as `BaseAgent`. After the custom agent has assembled and
deduplicated its restricted internal and external tools, omit any tool whose name
appears in the exclusion set.

Filtering the final collection provides defense across every tool source:
restricted internal tools, graph-injected tools, server tools, and client tools.
It also matches the base implementation's name-based exclusion semantics.

### PlanningAgent

Add `excluded_tool_names` to both planning overrides:

- `_get_llm_with_tools`
- `_get_tools_for_binding`

Forward the value explicitly to the corresponding superclass method while
preserving the existing mandatory `write_todos` injection. This keeps the
planning specialization compatible with the base contract and avoids silently
discarding exclusions.

### Contract Discipline

Do not add a catch-all `**kwargs`. Explicit parameters allow type checking,
inspection, and tests to detect future base/subclass signature drift. Do not
conditionally omit the keyword in `BaseAgent`; subclasses must honor the full
base contract even when a particular call currently supplies `None`.

## Testing

Use test-driven development:

1. Add a custom-agent test that calls the override with
   `excluded_tool_names` and verifies an excluded internal tool is absent while
   a non-excluded tool remains.
2. Add planning-agent tests that verify both overrides accept and forward
   `excluded_tool_names` while retaining the existing `write_todos` behavior.
3. Run the new tests before implementation and confirm they fail because the
   overrides reject the keyword.
4. Apply the minimal production changes and rerun the focused tests.
5. Run the existing custom-agent, planning-agent, graph, and tool-binding suites
   to detect regressions.

## Error Handling and Compatibility

No new exception handling is introduced. The repair prevents the programming
error at its source instead of masking it. Existing callers remain compatible
because the new parameter is optional and defaults to `None`.

No schema, migration, API, or client update is required.

## Acceptance Criteria

- Custom agents reach model invocation without an unexpected-keyword error.
- Custom-agent tool exclusions are enforced by name.
- Planning-agent tool-binding overrides accept and propagate exclusions.
- Existing handoff behavior remains unchanged.
- Focused and related automated tests pass.
