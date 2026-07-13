# Streamlit Skill HITL Controls Design

## Goal

Provide a streamlined way to configure human approval for each device-local
skill command from the Streamlit Skills tab. The interaction should match the
existing per-tool MCP approval control and reuse the existing HITL policy API.

## Scope

This change adds one approval control to each command-capable skill card in
`demo.py`. It does not add a new policy type, database table, endpoint, or
standalone HITL page.

Each current command-capable skill exposes one fixed command tool:

```text
skill::<skill-name>::run_skill_command
```

That exact qualified tool ID is already supported by the tool-scoped rules in
`GET`, `POST`, and `DELETE /hitl/settings`.

## User Experience

The Skills tab fetches HITL settings once alongside the skill list. When the
global HITL master switch is disabled, it displays the same informational
notice used by the MCP controls.

Each skill card contains a **Human approval** radio control with three modes:

- **Inherit**: remove the exact tool rule. The skill command falls back to its
  mutation metadata and therefore requires approval by default.
- **Require**: store an exact tool rule with `requireApproval: true`.
- **Skip**: store an exact tool rule with `requireApproval: false`, which
  preapproves that skill command at the server HITL gate.

Changing the mode writes through the existing HITL helpers and reruns the page
after a successful response. A failed settings read displays a warning and
does not render editable controls. A failed update displays the existing API
error rather than changing the visible policy state.

Disabled skills may retain and edit their rule. The rule has no execution
effect while the skill is disabled, but remains available if the skill is
enabled again.

## Data Flow

1. `render_skills_tab()` calls `get_skills_list()` and `get_hitl_settings()`.
2. Tool-scoped settings are indexed by `scopeValue`.
3. For each skill, Streamlit constructs
   `skill::<name>::run_skill_command` and maps its rule to Inherit, Require, or
   Skip.
4. A changed selection calls `clear_hitl_setting("tool", qualified_id)` for
   Inherit, or `set_hitl_setting("tool", qualified_id, bool)` for Require and
   Skip.
5. The existing per-turn HITL policy resolves the exact qualified ID before
   falling back to mutation gating.

Settings remain per-user because that is the behavior of the existing MCP
HITL controls. Actual command routing, runtime state, secrets, catalog entries,
and execution authorization remain device- and session-bound.

## Error and Safety Behavior

- Skill commands continue to require approval by default.
- Only an explicit Skip rule bypasses the server approval prompt.
- The sidecar command runner still validates the device, session, catalog,
  tool instance, source hash, and mutation approval context before spawning.
- Settings values contain only qualified tool IDs and booleans; no secrets or
  command arguments are persisted.
- If the global master switch is disabled, controls remain visible but are
  described as inactive, matching the MCP UI.

## Testing

Extend the Streamlit static guard to prove that the Skills tab:

- constructs the exact `skill::<name>::run_skill_command` ID;
- renders the same Inherit, Require, and Skip modes;
- calls the existing set and clear HITL helpers with tool scope; and
- displays the global-disabled notice.

Run the focused Streamlit HITL tests, the skill HITL policy tests, formatting
checks, and the broader skill/HITL regression matrix.
