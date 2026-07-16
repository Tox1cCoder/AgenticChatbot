# Uncapped Skill Terminal Errors — Design

## Goal

Make command-based client skill failures fully visible to the agent. The
model-facing `ToolMessage` must contain the complete redacted terminal error,
not only a generic classification. This behavior applies to every
`skill::<name>::run_skill_command` capability and is not provider-specific.

## Scope

- Remove the skill runtime's one-megabyte stdout/stderr capture ceiling.
- Remove the four-thousand-character stderr tail selection on nonzero exit.
- Include the complete redacted terminal error in the model-facing error JSON.
- Prevent skill terminal-error `ToolMessage` content from entering the ordinary
  truncation and blob-offload paths.
- Preserve existing generic classification fields (`error_type`, `retryable`,
  `message`, and `hint`) alongside the terminal output.
- Preserve the user's current uncommitted command-schema and activation-text
  changes.

This design does not remove limits from unrelated server MCP, native, or
client-device tools.

## Data Flow

1. `SkillExecutionEngine` drains stdout and stderr without an application-owned
   byte ceiling. Existing configured-secret redaction still runs on both
   streams.
2. A nonzero exit raises `SkillRuntimeError` with the entire redacted stderr. If
   stderr is empty, the entire redacted stdout is used so CLIs that report
   structured errors on stdout remain useful.
3. `RuntimeBridgeService` preserves this message in `RuntimeErrorContext` and
   keeps the canonical `skill::<name>::run_skill_command` identity in detail.
4. The server recognizes that exact skill identity and adds the error text as
   `untrusted_terminal_output` in the model-facing JSON payload.
5. The output is marked to bypass ordinary model truncation and blob offload.
   The system prompt tells agents to treat the field as diagnostic data, not as
   instructions.

Non-skill runtime errors retain the existing artifact-only raw-detail policy.

## Security and Operational Consequences

Configured secret values continue to be replaced by `<redacted>`, but output
may still contain undeclared credentials, private paths, personal data, or
prompt-injection text. An unbounded child can also exhaust sidecar memory, and a
large `ToolMessage` can exceed the model provider's finite context window. In
that case the provider may reject the next model request. These consequences
are intentional under the explicit no-cap requirement; the application will
not claim that arbitrary output can fit an external model context.

## Error Contract

The model-facing JSON remains machine-readable:

```json
{
  "status": "error",
  "error_type": "unknown",
  "retryable": false,
  "message": "Client runtime tool failed with an unknown error.",
  "hint": "Read untrusted_terminal_output and address the reported problem.",
  "untrusted_terminal_output": "skill command exited with code 1: ..."
}
```

The terminal field is present only when all of the following hold:

- the exception is a structured client-runtime failure;
- `qualified_tool_id` has the canonical `skill::...::run_skill_command` shape;
- the runtime supplied a nonempty error message.

## Testing

- A skill command emitting more than one megabyte is captured without an
  `OUTPUT_TOO_LARGE` failure.
- A nonzero skill exit preserves terminal text longer than four thousand
  characters.
- Tool error payload construction includes the complete terminal error for a
  canonical skill identity.
- An ordinary client-runtime permission error still hides raw private detail.
- Skill terminal errors bypass workflow truncation and tool-result offload.
- Existing skill execution, runtime bridge, tool error, and workflow suites
  remain green, followed by repository-wide Ruff and the full test suite.
