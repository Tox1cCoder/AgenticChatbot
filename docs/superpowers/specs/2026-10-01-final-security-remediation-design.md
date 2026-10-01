# Final security remediation design

The user's renewed request lifts the earlier Task 6 deferral and authorizes finishing
the remaining credential work. Preserve all completed lifecycle and answer-preview edits.

Server MCP mutation and direct testing require an authenticated user whose UUID is
listed in `ADMIN_USER_IDS`. An empty list grants nobody administrator rights. Read
catalog routes remain authenticated and redact secrets. `MCP_ALLOW_API_STDIO=false`
rejects direct stdio configuration and commands parsed through the from-URL route
before persistence or process startup. The operator can explicitly enable it for
administrators. Bundled `mcp_config.json` startup keeps using the existing manager.

Use the existing isolated worktree and observe failing request-level security tests
before implementing. Cover every POST/PATCH/DELETE route, authenticated reads,
explicit admin/stdio opt-in, URL-command bypasses, and existing bundled stdio behavior.

For the exposed service passwords, discover authenticated account controls using the
encrypted local skill bindings. Never emit passwords, tokens, or sensitive HTTP bodies.
Before any supported remote rotation, generate and durably encrypt the replacement,
then change through the site's actual account control and verify a new login. Keep
the existing binding until success, then replace it and invalidate obsolete sessions.
Do not guess write endpoints. If a current attendance site or recovery capability is
missing, request that specific information while proceeding with independent work.

Configure this installation's admin UUID only after confirming the existing signed-in
profile against its canonical server; never auto-promote the first registered user.
