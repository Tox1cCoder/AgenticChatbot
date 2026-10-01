# Server MCP API access

All server MCP routes require a valid bearer token. Signed-in users can read the
server and tool catalog; server credentials in headers, environment variables,
and URL query values are redacted.

Adding a server (including `/mcp/servers/from-url`), deleting a server, toggling
its enabled state, and calling `/mcp/tools/{tool_name}/execute` additionally require
the signed-in user's UUID to appear in `ADMIN_USER_IDS`. Configure this as a JSON
array of verified user UUIDs in the server environment. The default `[]` grants
nobody administrator rights; registration does not grant administrative access.
Denied requests return `403` with code `ADMIN_REQUIRED`.

`MCP_ALLOW_API_STDIO=false` is the default. Even administrators cannot add stdio
configuration, add an `npx` command through the from-URL route, or enable an existing
stdio server through the API. These requests return `403` with code
`API_STDIO_DISABLED` before configuration persistence or process startup. Existing
configuration uses the manager's transport normalization, including its stdio
default when transport is omitted. Administrators can still disable or remove
stdio servers. Set `MCP_ALLOW_API_STDIO=true` explicitly to permit these operations.

HTTP, SSE, and streamable HTTP server management remains available to administrators.
Bundled server startup through `MCPManager` continues to load the operator's
`app/ai/mcp_config.json` configuration, including its stdio servers. The API setting
does not change bundled startup or ordinary chat tool execution.
