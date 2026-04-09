# Chatbot

The repository contains:

- A canonical server backend in `app/` for auth, persistence, AI orchestration, document processing, provider management, planning, MCP integration, and widget sessions.
- A local client backend in `client_backend/` for loopback-safe desktop/runtime flows such as local sessions, local shell/filesystem/MCP access, skill discovery, and device-side tool execution.

## Architecture

```text
Frontend / Demo UI / Desktop UI
            |
            | HTTP
            v
  client_backend/ (local loopback runtime)
    - local auth/session wrapper
    - local shell + filesystem services
    - local MCP + local skills registry
    - runtime bridge + device catalog sync
    - compatibility proxy for server APIs
            |
            | HTTP + WebSocket runtime bridge
            v
      app/ (canonical server backend)
    - JWT auth + persistence
    - LangGraph agent workflow
    - planning + HITL interrupts
    - MCP registry + tool execution
    - widget session minting
    - document ingestion + RAG
            |
            +--> PostgreSQL
            +--> Redis
            +--> Celery workers
            +--> Qdrant
            +--> External model providers / MCP servers
```

### Server backend

`app/` is the source of truth for:

- users, conversations, messages, feedback, and documents
- agent orchestration and streaming responses
- task-plan generation and planning mode
- provider credentials and model configuration
- server-managed MCP servers and tool execution
- client-device registration and runtime dispatch
- widget connection tokens and widget state recovery

Key folders:

```text
app/
  ai/            LangGraph workflow, agents, prompts, tool binding, HITL
  api/           FastAPI route modules
  core/          config, DI container, auth helpers, exceptions
  database/      SQLAlchemy session/engine wiring
  models/        ORM models
  repositories/  persistence layer
  schemas/       Pydantic schemas and API contracts
  services/      business logic and orchestration
  workers/       Celery worker entrypoints and tasks
```

### Client backend

`client_backend/` is the local sidecar/runtime service. It keeps a desktop or local UI from talking directly to privileged local resources while still exposing those resources to the server when a device session is active.

It owns:

- local-session aware auth wrappers around upstream server auth
- runtime lifecycle endpoints such as `/runtime/connect` and `/runtime/status`
- local shell execution and filesystem services
- local MCP server discovery and execution
- local skill registry loading/toggling
- device registration, heartbeats, catalog sync, and runtime WebSocket handling
- compatibility routes at both `/...` and `/api/...` for local consumers

Key folders:

```text
client_backend/
  api/       loopback HTTP API and compatibility proxies
  core/      client config, logging, path and security helpers
  schemas/   runtime and skills payload models
  services/  runtime bridge, server API client, shell runner, local MCP
  cli.py     codex-client-backend entrypoint
  main.py    FastAPI application entrypoint
```

## Core Runtime Flows

### AI workflow

The main workflow is implemented in `app/ai/graph.py` and fronts several specialized agents. The router decides when to use:

- chat responses
- document-aware RAG responses
- web/search-style responses
- planning/task decomposition
- image generation
- canvas/artifact responses

The workflow also supports:

- streaming token and reasoning events
- deferred tool loading and MCP tool search
- HITL approval interrupts and resume flows
- conversation summarization and history-budget trimming
- client-device aware tool routing when a local runtime is connected

### Document pipeline

Document uploads are staged to temp storage, queued through Celery, parsed and chunked, then indexed into Qdrant for retrieval. The pipeline also persists document metadata and can recover extracted images/captions for richer RAG responses.

### Client runtime bridge

The server can register client devices and dispatch tool calls to a connected local runtime over WebSocket. The client backend:

- authenticates to the server
- registers a device session
- syncs tool and skill catalogs
- executes approved local tools
- returns structured tool results back to the server

### Widgets

Live widgets now have a dedicated HTTP handshake and WebSocket channel. The server mints a short-lived widget token via `POST /widgets/{widget_id}/connection`, and the actual widget state stream continues over `WS /widgets/{widget_id}/connect`.

## Prerequisites

- Python 3.10+
- PostgreSQL
- Redis
- Qdrant if you want document retrieval / RAG
- One or more model-provider credentials for real AI execution

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

For the Streamlit demo:

```bash
pip install -r demo_requirements.txt
```

### 2. Create environment files

- Copy `.env.example` to `.env`
- Copy `.env.client.example` to `.env.client` if you plan to run the local client backend

Important notes:

- `DATABASE_URL` should point to PostgreSQL.
- Redis is strongly recommended for Celery, HITL timeouts, client runtime state, and live widgets.
- `QDRANT_URL` is needed for document indexing/retrieval.
- `MODEL_ENCRYPTION_KEY` must be set if you want to store per-user provider API keys through `/providers`.
- `GEMINI_API_KEY` is still the default direct provider path from environment variables, but provider records can also be managed per user through the API.

To generate an encryption key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 3. Start infrastructure

PostgreSQL is required. Redis and Qdrant are optional at startup, but large parts of the platform will be degraded without them.

Redis helper:

```bash
docker compose -f docker-compose.redis.yml up -d redis
```

Qdrant helper:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 4. Run database migrations

```bash
alembic upgrade head
```

## Running The Services

### Canonical server backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Celery worker

```bash
python -m app.workers.start_worker
```

### Local client backend

```bash
codex-client-backend run --config .env.client
```

Equivalent module entrypoint:

```bash
python -m client_backend
```

### Demo UI

```bash
streamlit run demo.py
```

## API Surface Summary

### Canonical server routes

The server FastAPI app currently exposes these major route groups:

- `/auth/*` for signup, login, refresh, logout
- `/users/*` for user lookups
- `/conversations/*` for CRUD, title generation, and paginated message listing
- `/messages/*` for non-streaming, internal SSE streaming, stop, and resume-interrupt flows
- `/api/chat/{conversation_id}` and `/ai/*` for Vercel AI SDK style chat + conversation APIs
- `/documents/*` for upload, task status, document CRUD, and conversation document listings
- `/messages/{message_id}/feedbacks*` for feedback CRUD and stats
- `/conversations/{conversation_id}/task-plans*` and `/task-plans/*` for planning mode
- `/providers/*` and `/model-config*` for provider/model management
- `/mcp/*` for server-managed MCP servers and tool testing
- `/skills/*` for server skill discovery and toggling
- `/client-devices/*` and `/device-runtime/*` for local-device registration and connectivity
- `/widgets/{widget_id}/connection` for widget session handshake
- `/health*` for health and dependency checks

### Client backend routes

The local client backend provides:

- local auth/session routes under `/auth/*`
- runtime lifecycle routes under `/runtime/*`
- local MCP management under `/mcp/*`
- local skills inspection under `/skills/*`
- local message and AI SDK proxy routes under `/messages*` and `/api/chat/*`
- compatibility proxy routes for selected upstream resources under both `/...` and `/api/...`

## OpenAPI And Postman

- Server OpenAPI: `http://localhost:8000/docs`
- Client backend OpenAPI: `http://127.0.0.1:8100/docs`
- Postman collection: `Chatbot API.postman_collection.json`

The Postman collection is maintained for the canonical server HTTP API. It includes the newer planning, provider, model-config, MCP, skills, client-device, widget-token, and AI SDK surfaces.

It does not attempt to model the WebSocket-only flows:

- `/device-runtime/{device_id}/connect`
- `/widgets/{widget_id}/connect`

## Tests

Run the full suite:

```bash
pytest
```

Some tests exercise only the server, while `tests/client_backend/` covers the local runtime and compatibility layer.
