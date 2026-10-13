# Projects production review — 2026-10-09

Scope: the chatbot application's Projects settings, default custom-agent
inheritance, prompt composition, skills, MCP access, and the associated API,
sidecar, runtime, database, and demo paths.

## Findings and repairs

| Priority | Finding | Result |
| --- | --- | --- |
| P1 | Async project handlers directly ran synchronous SQLAlchemy operations on the HTTP event loop. | Project handlers now run in FastAPI's worker pool. An HTTP regression verifies the service call executes outside the event loop. |
| P1 | Attach and conversation creation committed membership before copying default agents. | Membership and inheritance now share one transaction. Forced insert failures leave the previous membership intact or leave no new conversation. |
| P1 | A project could be deleted after validation but before membership was written. | Live ownership is checked under the project lock inside the write transaction; deletion uses the same project-first ordering. |
| P2 | Detach could clear a newer move to another project. | The update requires the expected project, owner, and live conversation. A concurrent move produces the existing wrong-membership error. |
| P2 | Concurrent seeding could create duplicate attachment errors and conflicting order numbers. | Conversation locks serialize seeding and attachment replacement; existing agent order and insert-if-absent behavior are retained. |
| P1 | Disabled inherited custom agents were still executable. | Runtime projection suppresses disabled agents; explicit disabled entries are also rejected by handoff and construction. Saved defaults and attachments remain intact. |
| P2 | Project agent responses lacked device-specific skill/MCP availability. | Project GET and agent GET/PUT accept deviceId and reuse the custom-agent capability projection with one consistent device snapshot. Proxy schemas and demo helpers carry the context. |
| P2 | Legacy server-source skills could report ready while being filtered out at runtime. | Runtime skill filtering now uses the same compatibility matcher as management availability. |
| P2 | Names containing only whitespace were accepted. | Create and update trim names and reject blanks; null PATCH names remain invalid. |
| P1 | The ongoing fork implementation locked the source conversation before acquiring its project's FK lock, reversing deletion's lock order. | Fork now locks the live owned project before the source conversation. A real PostgreSQL concurrency regression reproduced the deadlock before the fix and verifies both operations complete afterward. |

Attachment writers lock live agent definitions in UUID order before changing child
rows. Agent deletion locks the definition before deleting its links. This avoids
reintroducing deleted definitions between validation and insertion. Project updates
and default replacement coordinate with project deletion as well.

The independently edited fork implementation's agent-copy path already handles
concurrent agent deletion correctly: PostgreSQL raises a serialization failure
under repeatable read, and the fork retries from a fresh snapshot. A concurrency
test verifies this behavior. No agent-copy repair was needed.

## Effective settings and inheritance

- Project instructions are read on each turn, ownership-checked against the
  conversation, and combined with its persona and scoped memory. The existing
  8,000-character limits, ordering, and non-project behavior remain covered.
- Default agents are an ordered set copied at conversation creation or attach.
  Attach appends missing defaults while keeping existing attachments and order.
  Changing defaults affects subsequent joins; it does not synchronize existing
  conversations. Detach and project deletion retain conversation attachments.
- (Changed 2026-10-13) The set is also an allowlist. A member conversation uses
  only attachments the project lists; out-of-set rows are kept but hidden from
  listing, PUT, and the runtime roster. An empty set allows no agents. Attaching
  an out-of-set agent is refused with 403. Like a disabled agent, a hidden agent
  is left out of the next turn's roster; a turn already running is not cancelled.
- Agent definitions remain live: configuration edits apply when a subsequent
  turn builds its runtime roster. Disabling suppresses future execution while
  preserving saved links; it does not cancel a turn already running.
- Skills are client-owned. Selected skills and client MCP tools resolve against
  the authenticated user's requested device and current session/catalog. Missing
  capabilities degrade availability without substituting another user's device.
- A project has no separate skills or MCP permission set. Those selections belong
  to its custom agents. Refresh availability when device sessions/catalogs change.

## Backend MCP policy decision

Saved backend MCP selections currently act as hints. Custom agents can discover
the globally enabled backend catalog, currently widgets, Tavily, time, and Brave
image search. Client MCP and skill selections remain restricted.

That existing policy was retained while the optional preference question is
unanswered. The frontend contract now explicitly distinguishes hints from
permissions. If selected backend tools are meant to define permissions, enforce
the same allowlist in discovery, binding, and execution and revise the existing
full-catalog tests together. This is a behavior change, not an existing enforced
security boundary.

## Verification

All database checks used a dedicated local PostgreSQL database, separate from the
application database. Its full Alembic chain upgraded to head successfully. Tests
disabled external tracing and used isolated sidecar profiles and test-only keys.

- Final project/custom-agent/skills/MCP/device/routing-and-projection selection:
  **1,035 passed, 7 skipped, 2 warnings** in 53.60 seconds.
- Additional fork, fork/project-deletion coordination, and membership selection:
  **87 passed**. The two coordination regressions and updated proxy/demo/OpenAPI
  checks also passed together (**30 passed**) after the fixture/schema updates.
- All changed Python files passed **Ruff**; **git diff --check** passed.
- All database-backed project tests ran. Six skips concern Windows FIFO/symlink
  constraints; one requires a reference skill archive absent from this checkout.
  The two warnings come from LangGraph's experimental v3 streaming protocol.

The broad selection can be repeated with a dedicated, migrated TEST_DATABASE_URL:

```powershell
$reviewTests = @(rg --files tests | Where-Object {
    $_ -match '(^|\\)test_.*(project|custom_agent|skills|skill_|mcp_|client_device|tool_scope|device_isolation|routing_inventory|agent_transitions).*\.py$'
})
& .\.venv\Scripts\python.exe -m pytest @reviewTests -q -rs
```

The review includes simulated device/provider collaborators, real PostgreSQL
transactions, HTTP contract tests, and MCP/skill runtime regressions. It does not
constitute a deployment or a live external-provider canary. Production credentials,
external service reachability, and deployment load behavior still require checks
in the target environment.

Existing sandbox edits and ongoing conversation-fork work were preserved. The
fork change in this review is limited to project lock coordination. Changes are
left uncommitted for review alongside the user's ongoing work.
