# Client Backend Bundle

This is the source-bundle distribution for the local client sidecar. Copy one
folder to another machine, run the startup script, and let the script maintain
its bundle-owned `.venv`.

## What Ships

- `client_backend/` source package
- `shared/` contract modules used by both server and client runtime
- minimal shared `app/` helpers required by the runtime bridge and local MCP manager
- bundled schema-v2 MCP registry and portable MCP server scripts
- `requirements-client.txt`
- `.env.client.example`
- `start-client-backend.ps1`
- `start-client-backend.bat`

## Fast Start On Another PC

1. Install Python 3.10 or newer.
2. Copy this folder or unzip `client-backend-bundle.zip`.
3. Copy `.env.client.example` to `.env.client`.
4. Set at least `CLIENT_SERVER_API_BASE_URL`.
5. Run `start-client-backend.bat`.

The launcher finds a compatible Python, creates or repairs `.venv`, installs
dependencies only when the Python version or requirements content changes, and
then runs:

```powershell
python -m client_backend run --config .env.client
```

If bootstrap fails, the launcher returns a nonzero exit code and does not write
a success marker. It never deletes or replaces `.env.client`.

## CLI

Useful commands inside the bundle:

```powershell
python -m client_backend run --config .env.client
python -m client_backend doctor --config .env.client
python -m client_backend mcp migrate
python -m client_backend mcp doctor --servers widgets,tavily,time
```

## Rebuild From This Repo

```powershell
.\scripts\build-client-backend-bundle.ps1
```

or:

```bash
python scripts/build_client_backend_bundle.py
```

Both builders consume the same canonical handoff files in
`scripts/client-backend-bundle/`.
