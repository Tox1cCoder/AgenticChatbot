# One-command local Docker stack

Approved in chat on 2026-10-28: run the complete local application with
`docker compose up -d --build`, including the containerized client sidecar.

## Services and startup

The default `compose.yaml` runs PostgreSQL 17, authenticated Redis 7.4,
Qdrant, the FastAPI API, separate Celery parse/index/summary workers, one
Celery Beat scheduler, a persistent MinerU 3.4.4 pipeline API, the client
sidecar, and Streamlit. Infrastructure health checks gate the API. The API
applies its existing Alembic migrations and initializes server secrets before
workers start. Workers and Beat wait for the healthy API; the parse worker
also waits for MinerU. Streamlit waits for the sidecar.

Application services share a Python 3.12 image with CPU PyTorch, MinerU's
pipeline extra, Node.js for local MCP subprocesses, and demo dependencies.
Use the authoritative `pyproject.toml`, not the Windows CUDA requirements
snapshot. Run application processes as an unprivileged user.

## Networking and persistence

Use Compose DNS for PostgreSQL, Redis, Qdrant, API, and MinerU. The demo
shares the sidecar's network namespace and reaches it over loopback, preserving
the existing Host allowlist. The sidecar writes its launch token into a
persistent profile volume, mounted read-only by the demo. Trust checks remain
enabled. Published API, sidecar, and demo ports bind to host loopback only.

Persist databases, uploaded content, parse artifacts, temporary staged files,
sidecar profile, separate `/home/chatbot/workspace`, model caches, and Beat
schedule in named volumes. Keep the workspace outside the credential-bearing
profile so the existing file-tool credential guard permits ordinary file access.
Keep existing host databases and containers untouched. The old Redis-only
Compose file remains available for native development. The new stack uses
separate infrastructure volumes and does not silently migrate existing data.

## Configuration and user experience

Use optional `.env` input for existing provider credentials and app settings;
explicit container environment values override host-only URLs and paths.
No env file is required just to start services. AI requests still require a
provider key. CPU document parsing downloads model weights on first use.
The container sidecar operates on its Linux workspace; it does not execute
commands in the Windows host environment. Document the startup command,
URLs, rebuilds, logs, persistence, and how to stop the stack without deleting data.

## Verification

Validate the Compose model with and without a dotenv file, build the image,
start an isolated test project on unused ports, and inspect health status.
Probe API, Redis, Qdrant, MinerU, sidecar token-authenticated routes, and
Streamlit. Check worker queue registration and Beat startup. Smoke-test
registration/login through the sidecar and document processing when feasible.
Stop only the isolated test project's containers after verification, preserving
its volumes and existing user services. Report any external download or runtime
limitations precisely.
