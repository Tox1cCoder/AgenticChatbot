# Projects production remediation

The review covers the chatbot's Projects feature, including default agent inheritance,
instructions, skills, and MCP access. Preserve existing user changes and seed-on-join
semantics: editing defaults does not rewrite existing conversation attachments.

## Implementation and verification

- Reproduce partial attach/create commits with PostgreSQL insertion failures, and
  stale membership writes with an interleaved move or project deletion.
- Make project membership and default-agent seeding one transaction. Lock live,
  owner-checked projects before conversations; use that order for deletion and
  replacement. Detach with an owner/live/expected-project conditional update.
- Run synchronous project handlers in FastAPI's worker pool. Verify the actual
  HTTP service calls have no running event loop.
- Propagate the persisted enabled switch into runtime state and suppress disabled
  agents in routing, handoff, and construction while retaining stored attachments.
- Reuse custom-agent capability projection for project agent reads, with optional
  deviceId and one consistent device snapshot per response.
- Confirm client skill and MCP isolation. Resolve the backend MCP hint versus
  permission policy explicitly before changing it.
- Run project, custom-agent, skills, MCP, proxy, routing, and lifecycle regressions
  against a dedicated migrated PostgreSQL database; run changed-file lint and an
  independent review. Record exact results and limits in a review report.
