# Sample Chatbot

A production-grade, multi-agent chatbot platform built on **FastAPI**, **LangGraph**, **PostgreSQL**, **Redis**, **Celery**, and **Qdrant**. The repository ships two complementary services:

- **`app/`** — the canonical **server backend**: authentication, persistence, multi-agent AI orchestration, document ingestion + RAG, task planning, HITL interrupts, per-user provider credentials, MCP tool registry, live widgets, and a client-device runtime bridge.
- **`client_backend/`** — a **local sidecar runtime** that lets a desktop or embedded UI expose privileged local resources (shell, filesystem, local MCP servers, local skills) to the server through a signed, device-authenticated WebSocket.

A Streamlit **demo UI** ([`demo.py`](demo.py)) and a ready-to-import **Postman collection** ([`Chatbot API.postman_collection.json`](Chatbot%20API.postman_collection.json)) are provided out of the box.

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Database Migrations](#database-migrations)
- [Running The Services](#running-the-services)
- [AI Workflow](#ai-workflow)
- [Document Pipeline & RAG](#document-pipeline--rag)
- [Provider & Model Configuration](#provider--model-configuration)
- [MCP Integration](#mcp-integration)
- [Skills System](#skills-system)
- [Planning Mode & Task Plans](#planning-mode--task-plans)
- [Human-in-the-Loop (HITL)](#human-in-the-loop-hitl)
- [Client Runtime Bridge](#client-runtime-bridge)
- [Live Widgets](#live-widgets)
- [API Reference](#api-reference)
- [Streaming, SSE & WebSocket Endpoints](#streaming-sse--websocket-endpoints)
- [OpenAPI & Postman](#openapi--postman)
- [Testing](#testing)
- [Observability](#observability)
- [Packaging & Distribution](#packaging--distribution)
- [Troubleshooting](#troubleshooting)

---

## Highlights

| Capability | Summary |
|---|---|
| **Multi-agent workflow** | LLM-driven router dispatches to specialised agents: `chat`, `rag`, `search`, `image_generator`, `planning`, `canvas`. Implemented as a LangGraph state machine in [`app/ai/graph.py`](app/ai/graph.py). |
| **Streaming-first API** | SSE streaming with 1-second heartbeats for [`/messages/stream`](app/api/messages.py), [`/messages/resume-interrupt`](app/api/messages.py), and full **Vercel AI SDK** compatibility at [`/api/chat/{conversation_id}`](app/api/ai_sdk.py) and [`/ai/chat/{conversation_id}`](app/api/ai_sdk.py). |
| **Document RAG** | Server-owned ingestion with MinerU parsing, structure-aware chunking, PostgreSQL as the canonical chunk store, Qdrant as the vector lookup index, optional cross-encoder re-ranking, and image captions indexed before embedding. |
| **Agentic RAG** | The only runtime RAG path. The agent uses `search_documents` actions to scan, read, grep, search chunks, list documents, and load images for long-document exploration. |
| **Planning mode** | End-to-end `TaskPlan` lifecycle (`draft` → `ready` → `executing` → `paused` → `completed`) with both AI-generated and manually-authored plans. |
| **Human-in-the-Loop** | Configurable per-tool approval interrupts with resume/reject semantics, persisted `ToolApproval` records, and Redis-backed timeout cleanup. |
| **Multi-provider** | Per-user, Fernet-encrypted API keys for **Google Gemini**, **OpenAI**, and **Anthropic**, with per-agent overrides (`agent_model_configs` table). |
| **MCP-native** | Server-managed MCP registry ([`/mcp/*`](app/api/mcp.py)) plus deferred tool search ([`tool_search`](app/ai/tool_search_tool.py)) to keep agent schemas small at prompt time. |
| **Custom agents** | Per-user agents with their own prompt, model, tools, and skills. MCP/skill selections are stored as **account-wide desired capabilities** (keyed by stable logical identity) but **execute device-locally**: on each request they resolve against the requesting device's live catalog, rebinding to its current session identity. Capabilities missing on the active device are reported as non-blocking `degraded`/`device_unavailable` availability and skipped at runtime — the agent still runs — while another device is never scanned for a match. |
| **Client runtime bridge** | Devices register, heartbeat, sync tool/skill catalogs, and receive WebSocket-dispatched tool calls — enabling local shell/filesystem/MCP execution without exposing them to the public network. |
| **Skills** | Markdown-defined skills with YAML frontmatter, owned by each client device. The sidecar scans one skills root (`CLIENT_SKILLS_ROOT`, or a per-user directory under the profile), syncs a per-device catalog to the server, and serves skill content over the runtime bridge ([`client_backend/services/local_skills_registry.py`](client_backend/services/local_skills_registry.py)). The server has no skills of its own. |
| **Live widgets** | Token-minted handshake (`POST /widgets/{id}/connection`) followed by a stateful WebSocket (`/widgets/{id}/connect`) for interactive, server-driven UI components. |
| **Durable conversation compaction** | PostgreSQL-backed compacted memory with sequence cursors, leased Celery jobs, request-budget preflight, and a bounded emergency path. |
| **Auto-continue** | Automatic continuation rounds when an agent hits iteration limits (`auto_continue_enabled`), with absolute wall-clock and iteration safety caps. |
| **Thinking / reasoning** | First-class support for Gemini 3 thinking levels (`minimal` / `low` / `medium` / `high`) and Gemini 2.5 thinking budgets, surfaced as streaming `reasoning` events. |
| **Gemini code execution** | Optional native tool for agentic vision + computation across agents (`enable_gemini_code_execution`). |
| **Observability** | Opt-in LangSmith tracing, centralised exception handler, per-service health endpoints (`/health`, `/health/celery`, `/health/redis`, `/health/qdrant`, `/health/all`), and structured stream events. |

---

## Architecture

```text
        ┌──────────────────────────────────────────────────────────────┐
        │   Frontend / Demo UI / Desktop UI / Vercel AI SDK client     │
        └───────────────┬──────────────────────────┬──────────────────┘
                        │ HTTP                     │ HTTP + WebSocket
                        ▼                          ▼
        ┌──────────────────────────┐   ┌─────────────────────────────┐
        │  client_backend/         │   │  app/  (canonical server)   │
        │  loopback sidecar        │──▶│                             │
        │  ─ local auth wrapper    │   │  ─ FastAPI + JWT auth       │
        │  ─ local shell/FS        │   │  ─ LangGraph multi-agent    │
        │  ─ local MCP manager     │   │  ─ Planning + HITL          │
        │  ─ local skills registry │   │  ─ MCP registry + execution │
        │  ─ runtime bridge (WS)   │◀──│  ─ Widget session minting   │
        │  ─ compatibility proxy   │   │  ─ Document ingestion + RAG │
        └──────────────────────────┘   │  ─ Client-device bridge     │
                                       └──┬─────────┬─────────┬──────┘
                                          │         │         │
                                          ▼         ▼         ▼
                                  ┌───────────┐ ┌───────┐ ┌────────┐
                                  │PostgreSQL │ │ Redis │ │ Qdrant │
                                  └───────────┘ └───┬───┘ └────────┘
                                                    │
                                            ┌───────▼────────┐
                                            │ Celery workers │
                                            └────────────────┘
```

Both services speak the same schemas (`app/schemas/`). The **client backend** exposes every upstream route under both `/...` and `/api/...` prefixes so legacy consumers continue to work without modification.

---

## Tech Stack

- **Runtime**: Python 3.10+, FastAPI, Uvicorn, asyncio
- **AI**: LangChain 1.x, LangGraph 1.x, LangSmith, langchain-google-genai, langchain-openai, langchain-mcp-adapters, Tavily
- **Persistence**: SQLAlchemy 2.x + Alembic, PostgreSQL 14+, psycopg driver
- **Background**: Celery 5.x + Redis 7.x
- **Vector search**: Qdrant, Gemini Embedding API (`gemini-embedding-2`) with optional sentence-transformers fallback for offline development, HF cross-encoder re-rankers
- **Documents**: MinerU (pipeline / hybrid / VLM backends), pdfplumber, python-docx, openpyxl, Pillow, pypdf
- **Security**: PyJWT, bcrypt, Fernet (cryptography) for provider key encryption
- **DI**: `dependency-injector` with auto-injection decorators (`AppAutoInjector`)
- **Demo**: Streamlit + fastapi-radar for live debugging

---

## Repository Layout

```text
.
├── app/                              Canonical server backend (FastAPI)
│   ├── ai/                           LangGraph workflow, agents, MCP, skills, tools
│   │   ├── agents/                   chat / rag / search / image / planning / canvas / router
│   │   ├── mcp_servers/              Built-in MCP servers (calculator, tavily, brave_image_search, time, widgets)
│   │   ├── graph.py                  MultiAgentWorkflow + streaming + HITL
│   │   ├── history.py                Canonical compacted-memory + transcript assembly
│   │   ├── conversation_compactor.py Durable compaction orchestration
│   │   ├── token_counter.py          Provider-aware request token accounting
│   │   ├── tool_search_tool.py       Deferred tool loading
│   │   ├── deferred_tool_*.py        Deferred binding + state machine
│   │   ├── skills_*.py               Skill registry / resolver / snapshot / tool
│   │   └── model_factory.py          Provider-agnostic LLM instantiation
│   ├── api/                          FastAPI route modules (15 routers)
│   ├── core/                         config, DI container, auth, exceptions, runtime modelling
│   ├── database/                     session / engine / migration bootstrap
│   │   └── qdrant/config/config.yaml Local Qdrant server config (binary and
│   │                                 storage/ state are gitignored)
│   ├── factories/                    Pydantic/domain factories
│   ├── interfaces/                   Service interface contracts (ABCs)
│   ├── models/                       15 SQLAlchemy ORM models
│   ├── repositories/                 persistence + query strategy + command strategy
│   ├── schemas/                      Pydantic schemas and API contracts
│   ├── services/                     business logic, orchestration, event listeners
│   ├── storage/                      user content: document images, chat images,
│   │                                 parse artifacts (gitignored, not distributed)
│   ├── utils/                        exception handlers, helpers
│   └── workers/                      Celery app, document processor, cleanup tasks
├── client_backend/                   Local sidecar runtime
│   ├── api/                          auth, conversations, messages, documents, runtime,
│   │                                 mcp, skills, proxy, health
│   ├── core/                         client config, logging, paths, security
│   ├── schemas/                      runtime + skills payload models
│   ├── services/                     runtime_bridge, server_api, local_mcp_manager,
│   │                                 local_skills_registry, upstream_auth
│   ├── storage/                      per-user profile storage root
│   ├── cli.py                        kani-client-backend entrypoint (run / doctor)
│   └── main.py                       FastAPI app factory
├── shared/skills/                    Shared skill parsing helpers (front matter)
├── skills/                           Optional user-local skills (ignored, not distributed)
├── tests/                            Unit + integration tests (server and client_backend)
├── scripts/                          Bundle builders, benchmarks, reindex and
│                                     verification helpers
├── dist/client-backend-bundle/       Generated client bundle output (not tracked)
├── docker-compose.redis.yml          Local Redis with persistence + auth
├── alembic.ini                       Alembic runtime config
├── demo.py                           Streamlit demo UI
├── demo_requirements.txt             Demo-only dependencies
├── upload_support.py                 Streamlit document upload helper
├── pyproject.toml                    Project metadata, deps, scripts (kani-client-backend)
│                                     — the authoritative dependency declaration
├── requirements.txt                  Frozen Windows / Python 3.11 / CUDA 13.0
│                                     snapshot for the GPU MinerU pipeline
├── environment.yml                   Conda form of the same frozen snapshot
├── Chatbot API.postman_collection.json
└── README.md
```

---

## Prerequisites

| Component | Requirement | Notes |
|---|---|---|
| Python | **3.10+** | 3.13 is the current development runtime. Use 3.11 only for the frozen CUDA environment (`requirements.txt` / `environment.yml`), which pins packages without 3.13 wheels |
| PostgreSQL | **14+** | Required; holds auth, conversations, plans, feedback, HITL state, LangGraph checkpoints |
| Redis | **7+** | Strongly recommended. Required for Celery, live widgets, client runtime state, HITL timeouts |
| Qdrant | latest | Required for document retrieval / RAG |
| Node.js | optional | Only for consumers using the `@ai-sdk` client |
| Docker | optional | Helpers provided for Redis + Qdrant |

At least one **LLM provider credential** is required for real AI execution:

- `GEMINI_API_KEY` — the default provider, wired via `langchain-google-genai`
- per-user OpenAI / Anthropic keys managed through [`/providers`](app/api/providers.py) once `MODEL_ENCRYPTION_KEY` is set

Optional: `TAVILY_API_KEY` for web search agent, `BRAVE_SEARCH_API_KEY` for image search, `SMITHERY_API_KEY` for hosted MCP servers, `LANGSMITH_API_KEY` for tracing.

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .[dev]
```

For the Streamlit demo only:

```bash
pip install -r demo_requirements.txt
```

A conda environment snapshot is available as [`environment.yml`](environment.yml).
It declares `name: agents`, so that is the environment to activate:

```bash
conda env create -f environment.yml
conda activate agents
```

`requirements.txt` and `environment.yml` are **not** the recommended way to run
the server. They are frozen snapshots of the Windows / Python 3.11 / CUDA 13.0
environment used for the GPU document pipeline, and they carry the full MinerU
stack (`torch+cu130`, `doclayout_yolo`, `ultralytics`, OpenCV). Use them only
when you need that pipeline locally; use `pip install -e .[dev]` above
otherwise.

Validate the complete frozen dependency graph from a Windows Python 3.11
interpreter with:

```bash
python scripts/verify_frozen_requirements.py
```

The verifier exits with code 2 on any other platform or Python version, so it
cannot run on a machine that has no 3.11 interpreter — check `py -0p` before
relying on it. On macOS or Linux, use the editable install instead of these
frozen files.

### 2. Create environment files

```bash
cp .env.example .env
cp .env.client.example .env.client   # if running the local client backend
```

### 3. Generate a provider-key encryption key

Per-user provider API keys are encrypted at rest with Fernet. Generate and set `MODEL_ENCRYPTION_KEY`:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 4. Start infrastructure

**PostgreSQL** — create a database, then set `DATABASE_URL` in `.env`.

**Redis** (Docker, with auth enabled by default):

```bash
docker compose -f docker-compose.redis.yml up -d redis
```

**Qdrant**:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

To run Qdrant as a native Windows process instead, download `qdrant.exe` from
[the Qdrant releases page](https://github.com/qdrant/qdrant/releases) into
`app/database/qdrant/` and launch it with the tracked
[`config/config.yaml`](app/database/qdrant/config/config.yaml). The binary and
its `storage/` state are gitignored, so they are not part of a fresh clone.

---

## Environment Variables

The full schema lives in [`app/core/config.py`](app/core/config.py). Selected highlights:

### Core

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://localhost:5432/chatbot` | SQLAlchemy DSN |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Uvicorn bind |
| `ENVIRONMENT` | `development` | `development` / `staging` / `production` |
| `SECRET_KEY` | *(ephemeral in dev)* | Must be set in production |
| `JWT_ALGORITHM` | `HS256` | |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `600` | |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | |
| `CORS_ORIGINS` | `[]` | JSON list; `["*"]` allows any origin |

### Providers & models

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Default provider; still supported via env |
| `TAVILY_API_KEY` | — | Tavily ranked-source Search plus Extract, Map, and Crawl |
| `BRAVE_SEARCH_API_KEY` | — | Brave Image Search (provider-confidence visual references) |
| `BRAVE_IMAGE_SEARCH_DEFAULT_COUNT` | `6` | Default image results per call |
| `BRAVE_IMAGE_SEARCH_MAX_COUNT` | `10` | Hard cap on image results per call |
| `BRAVE_IMAGE_SEARCH_TIMEOUT_SECONDS` | `2.5` | Per-request timeout for image search |
| `BRAVE_IMAGE_SEARCH_DEFAULT_SAFESEARCH` | `strict` | Brave safesearch level (`off` or `strict`) |
| `REMOTE_IMAGE_ENRICHMENT_ENABLED` | `true` | Enables optional Brave-backed remote image enrichment for rich responses |
| `SMITHERY_API_KEY` | — | Hosted MCP registry |
| `MODEL_ENCRYPTION_KEY` | — | Fernet key for per-user provider credentials |
| `RAG_AGENT_MODEL` | `gemini-3.1-pro-preview` | |
| `CHAT_AGENT_MODEL` | `gemini-3-flash-preview` | |
| `SEARCH_AGENT_MODEL` | `gemini-3-flash-preview` | |
| `ROUTER_MODEL` | `gemini-3-flash-preview` | Request routing |
| `IMAGE_GENERATOR_TOOL_MODEL` | `gemini-3-flash-preview` | Image-agent tool calling |
| `CANVAS_AGENT_MODEL` | `gemini-3.1-pro-preview` | Canvas agent default |
| `SUGGESTION_MODEL` | `gemini-3-flash-preview` | Follow-up suggestions |
| `TITLE_GENERATOR_MODEL` | `gemini-3-flash-preview` | Conversation titles |
| `IMAGE_GENERATOR_MODEL` | `gemini-3-pro-image` | Image generation |
| `IMAGE_CAPTION_MODEL` | `gemini-3-flash-preview` | |
| `MEDIA_RESOLUTION` | `high` | `low` / `medium` / `high` (Gemini 3 per-part) |
| `ENABLE_THINKING` | `true` | |
| `THINKING_LEVEL` | `high` | `minimal` / `low` / `medium` / `high` (Gemini 3) |
| `THINKING_BUDGET` | `-1` | Token budget for Gemini 2.5 (-1 dynamic, 0 off) |
| `ENABLE_GEMINI_CODE_EXECUTION` | `true` | Native code-execution tool |

### Model usage analytics

These variables are discovered by `app/core/config.py::Settings`. The values
below are safe development defaults. See the
[model-usage analytics operations runbook](docs/operations/model-usage-analytics.md)
for schema-first rollout, maintenance, monitoring, investigation, and rollback
guidance.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_USAGE_TRACKING_ENABLED` | `true` | Record provider attempts and rollups |
| `MODEL_USAGE_UI_ENABLED` | `true` | Expose usage analytics endpoints and UI |
| `MODEL_USAGE_RAW_RETENTION_DAYS` | `90` | Raw-event retention |
| `MODEL_USAGE_ROLLUP_RETENTION_DAYS` | `730` | Minute-rollup retention |
| `MODEL_USAGE_RECONCILE_MINUTES` | `2880` | Trailing reconciliation window |
| `MODEL_USAGE_RECONCILE_CHUNK_MINUTES` | `60` | Maximum transaction span |
| `MODEL_USAGE_CLEANUP_BATCH_SIZE` | `5000` | Rows deleted per cleanup batch |
| `MODEL_USAGE_RETRY_MAX_ATTEMPTS` | `5` | Failed-write retry limit |
| `MODEL_USAGE_RETRY_BASE_SECONDS` | `10` | Exponential retry base delay |
| `MODEL_USAGE_USER_HASH_SECRET` | `blank` | User-hash key; development may leave blank |
| `MODEL_USAGE_HEALTH_LOOKBACK_MINUTES` | `60` | Health snapshot lookback |
| `MODEL_USAGE_HEALTH_UNATTRIBUTED_DEGRADED_RATIO` | `0.1` | Degraded unattributed-attempt threshold |
| `MODEL_USAGE_HEALTH_ROLLUP_LAG_DEGRADED_MINUTES` | `2` | Degraded rollup-lag threshold |
| `MODEL_USAGE_HEALTH_ROLLUP_LAG_UNHEALTHY_MINUTES` | `5` | Unhealthy rollup-lag threshold |
| `MODEL_USAGE_HEALTH_FAILURE_WINDOW_SECONDS` | `300` | Shared failure-health window |
| `MODEL_USAGE_FAILURE_STORE_TTL_SECONDS` | `900` | Redis failure-bucket TTL |
| `MODEL_USAGE_FAILURE_STORE_TIMEOUT_SECONDS` | `0.25` | Best-effort Redis timeout |

Cross-field constraints: reconcile window must be shorter than raw retention;
rollup retention must be at least raw retention; failure-store TTL must cover
the health window plus 60 seconds; unhealthy rollup lag must be at least
degraded rollup lag. MODEL_USAGE_USER_HASH_SECRET must be set in production
when LangSmith tracing is enabled.

Per-field ranges: retention, reconciliation, cleanup, retry, lookback,
failure-window, and TTL integers must be positive; unattributed ratio must be
between 0 and 1; rollup lag thresholds must be nonnegative; failure window
cannot exceed 3600 seconds; failure-store TTL must be between 60 and 86400
seconds; failure-store timeout must be positive.

Authenticated clients discover availability at `/usage/capabilities`, query
account-wide analytics at `/usage/dashboard`, and query one owned conversation
at `/usage/conversations/{conversation_id}`. Operators use the aggregate-only
`/health/model-usage` and `/metrics/model-usage` endpoints. By default, cleanup
retains 90 days of raw events and 730 days of minute rollups. Analytics reports
tokens, requests, images, coverage, and operational health; it does not
calculate or track monetary cost. Frontend consumers should follow the
[AI SDK usage analytics contract](plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md).

### Vector store / RAG

| Variable | Default |
|---|---|
| `QDRANT_URL` | `http://localhost:6333` |
| `QDRANT_COLLECTION_NAME` | `documents_gemini_embedding_2_3072` |
| `RAG_EMBEDDING_PROVIDER` | `gemini` (alt: `sentence_transformers`) |
| `RAG_EMBEDDING_MODEL` | `gemini-embedding-2` |
| `RAG_EMBEDDING_DIMENSION` | `3072` |
| `RAG_EMBEDDING_QUERY_TASK` | `search result` (or `question answering`) |
| `RAG_MULTIMODAL_IMAGE_EMBEDDINGS_ENABLED` | `false` |
| `RAG_TOP_K` | `15` |
| `RAG_SCORE_THRESHOLD` | `0.2` |
| `ENABLE_RERANKING` | `true` |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANK_TOP_K` | `10` |
| `RAG_CHUNK_TARGET_TOKENS` | `400` |
| `RAG_CHUNK_OVERLAP_TOKENS` | `40` |
| `RAG_CHUNK_MAX_TOKENS` | `800` |
| `RAG_EMBEDDING_BATCH_SIZE` | `32` |
| `RAG_EMBEDDING_MAX_CONCURRENCY` | `4` |
| `QDRANT_UPSERT_BATCH_SIZE` | `1000` |

`GEMINI_API_KEY` is required when `RAG_EMBEDDING_PROVIDER=gemini`. The
`sentence_transformers` fallback is for offline development; switching
providers requires a deliberate Qdrant collection cutover (see "RAG
embedding migration" below).

### Conversation memory & history budgets

`MEMORY_MAX_MESSAGES`, `CHAT_HISTORY_MAX_MESSAGES` / `_TOKENS`, `RAG_HISTORY_MAX_*`, `SEARCH_HISTORY_MAX_*`, `PLANNING_HISTORY_MAX_*`.

### Durable conversation compaction

Prompt memory is built in one place — `app.ai.history.ConversationHistoryProvider` — and combines a durable per-conversation compacted-memory record in PostgreSQL with transcript rows after its numeric message-sequence cursor. Request preflight uses the same provider-aware token counter as the background compactor and can request durable work or, at the hard boundary, run one bounded synchronous compaction attempt.

| Variable | Default | Notes |
|---|---|---|
| `MEMORY_CACHE_TTL_SECONDS` | `60` | TTL for the in-process prompt-history cache. |
| `MEMORY_CACHE_MAX_CONVERSATIONS` | `256` | LRU cap before older conversations are evicted. |
| `CONVERSATION_SUMMARY_ENABLED` | `true` | Enables durable and emergency compaction. |
| `CONVERSATION_SUMMARY_TRIGGER_MESSAGES` | `60` | Pending-message trigger; pair with the token trigger. |
| `CONVERSATION_SUMMARY_TRIGGER_TOKENS` | `18000` | Pending-token trigger. |
| `CONVERSATION_SUMMARY_KEEP_RECENT_TURNS` | `4` | Complete recent turns excluded from compaction. |
| `CONVERSATION_SUMMARY_MAX_TOKENS` | `1500` | Positive output cap for compacted memory. |
| `CONVERSATION_SUMMARY_TIMEOUT_SECONDS` | `30` | Model-call timeout. |

Operational behaviour:

- Compaction is idempotent and leased; a timeout or model error leaves the previous valid memory in place and retries with bounded backoff.
- Prompt history is cached by conversation, user, current message, agent, cursor, and memory version; transcript writes invalidate the cache.
- Soft-deleted messages and empty paused/interrupt assistant placeholders are excluded from prompt history.
- Checkpoint state is **not** long-term memory. After the terminal assistant response is persisted, the service issues `RemoveMessage` for every checkpoint message id; PostgreSQL is the canonical transcript.
- AI SDK clients may post their full UI history at `POST /api/chat/{conversation_id}`; the server only consumes the latest user message and rebuilds prior memory from the database.

Deployment, migration, rollback, backfill, monitoring, and credential-rotation procedures are documented in [`docs/operations/conversation-compaction.md`](docs/operations/conversation-compaction.md).

### Document processing

`MINERU_TIMEOUT`, `MINERU_API_URL`, `MINERU_BACKEND` (`pipeline` / `hybrid-*` / `vlm-*`), `MINERU_METHOD` (`auto` / `txt` / `ocr`), `MINERU_LANG`, `MINERU_EXTRA_ARGS`, `EXTRACT_FORMULAS_FROM_PDF`, `EXTRACT_TABLES_FROM_PDF`, `TABLE_FORMAT`, `MAX_FILE_SIZE_MB`, `TEMP_STORAGE_PATH`, `DOCUMENT_IMAGES_STORAGE_PATH`, `IMAGE_CAPTION_MODEL`, `IMAGE_CAPTION_MAX_RETRY_ATTEMPTS`, `IMAGE_CAPTION_RETRY_DELAY_SECONDS`.

### MCP tool search

`MCP_TOOL_SEARCH_ENABLED`, `MCP_TOOL_SEARCH_DEFAULT_TOP_K`, `MCP_TOOL_SEARCH_AUTOLOAD_TOP_K`, `MCP_TOOL_SEARCH_PINNED_TOOLS`, `MCP_TOOL_SEARCH_MAX_LOADED_TOOLS_PER_CONVERSATION`, `MCP_TOOL_SEARCH_LOADED_TOOLS_TTL_MINUTES`, `MCP_TOOL_SEARCH_MIN_RELEVANCE_SCORE`, `MCP_TOOL_SEARCH_AUTOLOAD_MIN_RELEVANCE_SCORE`.

### Tool execution policy

Interactive tools resolve an origin-aware policy from the canonical
`tool_origin` plus `qualified_tool_id`, trusted internal metadata, and layered
deployment rules. Unknown tools keep the 30-second soft timeout and one attempt;
automatic retries require both a transient failure and explicit safe-repeat
policy. Client execution and bridge-response deadlines finish before the server
soft deadline.

| Variable | Default |
|---|---:|
| `TOOL_EXECUTION_TIMEOUT` | `30` |
| `TOOL_EXECUTION_POLICIES` | `{}` |
| `TOOL_EXECUTION_MAX_INTERACTIVE_TIMEOUT_SECONDS` | `120` |
| `TOOL_EXECUTION_CANCELLATION_GRACE_SECONDS` | `2` |
| `TOOL_EXECUTION_CLIENT_EXECUTION_GRACE_SECONDS` | `2` |
| `TOOL_EXECUTION_CLIENT_RESPONSE_GRACE_SECONDS` | `1` |

Matching, JSON configuration, safe rollout, incident caps, cancellation
semantics, and the sole `dispatch_subagents` exception are documented in
[`docs/operations/tool-execution-policy.md`](docs/operations/tool-execution-policy.md).

### HITL & planning

`ENABLE_HUMAN_IN_THE_LOOP`, `HITL_TOOLS_REQUIRE_APPROVAL`, `HITL_APPROVAL_TIMEOUT_MINUTES`, `MAX_AUTO_PLAN_TASKS`, `EXECUTION_CALL_BUDGET`, `PLANNING_MAX_ITERATIONS`, `PLANNING_CONSECUTIVE_ERRORS_LIMIT`.

### Planning-mode subagents

`PLANNING_SUBAGENTS_ENABLED`.

While Planning mode is active, the Planning Agent can call the internal `dispatch_subagents` tool to fan out independent worker tasks to other graph agents (`chat_agent`, `rag_agent`, `search_agent`, `image_generator_agent`, `canvas_agent`). Workers run concurrently in the same chat turn — there is no background queue and the dispatch call blocks until every worker completes, fails, or signals it needs human approval. Workers run with isolated message state, inherit scoped identifiers (`conversation_id`, `user_id`, `device_id`) and runtime model overrides, and return a full `answer` to the Planning Agent plus a compact `summary` for activity UI/metadata. The dispatcher does not apply `TOOL_RESULT_MAX_CHARS` or tool-result offload to the worker `answer`; workers are prompted to answer concisely but with enough detail for supervisor reconciliation. Workers cannot mutate todos directly: the Planning Agent reads each `answer` and reconciles the plan with `write_todos`. This is distinct from `hand_off`, which re-routes the entire turn to a single top-level agent rather than fanning out parallel research/build work.

**Per-task model override.** Each entry in `dispatch_subagents.tasks[]` accepts an optional `model_override` so the Planning Agent can assign a concrete model to one worker without affecting parent or sibling routing. The override carries `provider`, `model`, optional `temperature`, `allow_custom_model`, and a provider-agnostic `reasoning_effort` (`none`/`minimal`/`low`/`medium`/`high`/`xhigh`) that is normalized to OpenAI `reasoning.effort` or a Gemini 3 `thinking_level` for the worker call only. The override is request-scoped — it is never persisted to `agent_model_configs`.

```json
{
  "id": "w1",
  "agent": "search_agent",
  "task": "Research the migration risk.",
  "model_override": {
    "provider": "openai",
    "model": "gpt-5.4",
    "reasoning_effort": "high"
  }
}
```

### Client runtime bridge

`ENABLE_CLIENT_RUNTIME_BRIDGE`, `CLIENT_RUNTIME_WS_TIMEOUT_SECONDS`, `CLIENT_RUNTIME_CATALOG_CACHE_TTL_SECONDS`, `CLIENT_RUNTIME_REQUIRE_CONNECTED_DEVICE_FOR_LOCAL_TOOLS`, `CLIENT_RUNTIME_HEARTBEAT_INTERVAL_SECONDS`, `CLIENT_RUNTIME_MAX_TOOL_RESULT_SIZE_BYTES`.

### Redis

The server accepts either a fully-formed URL (`REDIS_URL`) or a hostname + convenience `REDIS_PASSWORD`. Loopback hosts are automatically normalised to `127.0.0.1` on Windows to avoid async-client `localhost` issues. Missing `REDIS_URL` falls back to `CELERY_BROKER_URL`.

### Client backend (`.env.client`)

| Variable | Default |
|---|---|
| `CLIENT_SERVER_API_BASE_URL` | `http://localhost:8000` |
| `CLIENT_SERVER_API_TIMEOUT_SECONDS` | `60` |
| `CLIENT_BACKEND_HOST` / `CLIENT_BACKEND_PORT` | `127.0.0.1` / `8100` |
| `CLIENT_ENVIRONMENT` | `development` |
| `CLIENT_PROFILE_ROOT` | *(OS-default — `%LOCALAPPDATA%\KaniDesktop` on Windows, `~/.config/kani-desktop` elsewhere)* |
| `CLIENT_DEVICE_NAME` | — |
| `CLIENT_SKILLS_ROOT` | one absolute path holding every skill; unset means `<profile>/<server-hash>/<user-id>/skills/installed` |
| `CLIENT_WORKSPACE_ROOTS` | comma-separated absolute paths for local filesystem tools |
| `CLIENT_MCP_CONFIG_PATH` | optional legacy source path for the one-time `mcp migrate` command |
| `CLIENT_MCP_STARTUP_TIMEOUT_SECONDS` | `30` |
| `CLIENT_TOOL_CALL_TIMEOUT_SECONDS` | `60` |
| `CLIENT_HEARTBEAT_INTERVAL_SECONDS` | `30` |
| `CLIENT_RECONNECT_DELAY_SECONDS` / `CLIENT_MAX_RECONNECT_ATTEMPTS` | `5` / `10` |
| `CLIENT_LOCAL_SESSION_SECRET` / `CLIENT_LOCAL_SESSION_EXPIRE_MINUTES` | auto-generated / `1440` |
| `CLIENT_LOG_LEVEL` / `CLIENT_LOG_TO_FILE` | `INFO` / `true` |

---

## Database Migrations

Migrations are Alembic-managed (single head, currently `c3d4e5f6a7b8`). They are applied **automatically** at application startup via `app.database.migrations.upgrade_database` inside the lifespan hook, so manual migration is only required for dev or out-of-process tooling:

```bash
alembic upgrade head

# Generate a new revision
alembic revision --autogenerate -m "add something"
```

Key schemas:

| Table | Role |
|---|---|
| `users` | Accounts + auth |
| `conversations` | Threads, plan lifecycle (`plan_lifecycle`), persona prompt, title |
| `messages` | User/assistant turns + JSONB metadata (tool calls, artifacts, interrupts) |
| `feedbacks` | 1-5 rating, categorical tags, text |
| `documents` | Uploaded files, status, file type, Celery task id |
| `document_chunks` | Canonical parsed chunk content, page spans, provenance, and index status |
| `document_images` | Extracted images, generated captions, and optional linked chunk IDs |
| `document_parse_artifacts` | Source and parser output artifacts such as MinerU markdown / JSON metadata |
| `task_plans` | Plan steps, order, status, metadata |
| `tool_approvals` | HITL approval decisions, persisted for audit |
| `hitl_interrupts` | Suspended workflow snapshots |
| `model_providers` | Fernet-encrypted per-user keys, per-provider metadata |
| `agent_model_configs` | Per-agent model/provider/temperature overrides |
| `client_devices` | Device registration, session, heartbeat, catalogs |
| `skill_settings` | Per-user skill toggles |

LangGraph checkpoints are kept in the same database under `CHECKPOINT_SCHEMA` (default `public`) by `langgraph-checkpoint-postgres`.

### Schema ownership and cleanup

> **Historical migration warning:** the original `6c6598a9eb26` revision could drop LangGraph checkpoint tables and `mcp_oauth_tokens`. For a deployment that may have run that revision, inspect these tables before upgrading and take a verified database backup. The forward `b5c6d7e8f9a0` repair cannot reconstruct deleted checkpoint or MCP OAuth data. Restore the affected tables from a pre-upgrade backup when available. Without a checkpoint backup, reinitialize LangGraph only after accepting the loss of resumable workflow/HITL state; users must reauthenticate affected MCP servers when OAuth rows cannot be restored.
>
> The same repair normalizes legacy uppercase approval enum labels to lowercase.
> Drain API and worker approval writers before applying `b5c6d7e8f9a0`, then
> deploy code whose SQLAlchemy mapping persists `DecisionType.value`. Its
> downgrade intentionally does not restore uppercase labels; rollback to an
> older uppercase-mapped release requires a compatible backport or restoration
> of the verified pre-upgrade backup.

- **Alembic owns the application tables only.** Current autogenerate is filtered (`app/alembic/autogenerate_filters.py`) so it does not touch the LangGraph checkpoint tables or `alembic_version`.
- **LangGraph owns the checkpoint tables** (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`). Corrected migration history and new application migrations leave them unchanged; pruning their rows is operational cleanup, not a schema migration.
- Migrations are the **only** schema-mutation path — `Base.metadata.create_all()` is not called at application startup (the old `Database.create_database()` helper was removed).
- `conversation_device_bindings` was dropped (migration `v1w2x3y4z5a6`): conversation ownership is **user-based**, not device-bound.
- **Checkpoint retention** (expiring abandoned HITL interrupts and reaping the checkpoint threads of expired interrupts and soft-deleted conversations) is handled by `CheckpointRetentionService`, invoked from the `cleanup_abandoned_interrupts` Celery beat task.

---

## Running The Services

### Canonical server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Startup performs, in order: DB migration, checkpoint-table setup, agent pre-warming, skills pre-scan, Redis availability check, and launches the periodic client-runtime session cleanup task when `ENABLE_CLIENT_RUNTIME_BRIDGE=true`.

### Celery worker

```bash
python -m app.workers.start_worker
```

Handles:

- `app.workers.document_processor` — ingest → parse → caption images → chunk → persist SQL chunks → embed → upsert Qdrant lookup points
- `app.workers.cleanup_tasks` — expired tokens, orphaned files, stale device sessions

Parallelism is config-driven (see [`.env.example`](.env.example)):

| Setting | Default | Notes |
|---|---|---|
| `CELERY_WORKER_POOL` | `auto` | `auto` resolves to `threads` on Windows and `prefork` on Linux. `solo` is a single-task debug pool — do not use for batch uploads. |
| `CELERY_WORKER_CONCURRENCY` | `2` | Number of tasks running in parallel. Increase for more upload throughput; lower if model/embedding providers rate-limit. |
| `CELERY_WORKER_PREFETCH_MULTIPLIER` | `1` | Keep at 1 unless you understand Celery prefetch semantics. |
| `CELERY_WORKER_MAX_TASKS_PER_CHILD` | `10` | Recycles worker process every N tasks to bound memory growth. |
| `CELERY_WORKER_TIME_LIMIT` / `CELERY_WORKER_SOFT_TIME_LIMIT` | `300` / `240` | Per-task wall-clock limits (seconds). |
| `CELERY_WORKER_CANCEL_LONG_RUNNING_TASKS_ON_CONNECTION_LOSS` | `true` | Cancels late-acknowledged running tasks if Redis disconnects so redelivery does not run a duplicate copy concurrently. |
| `CELERY_BROKER_HEALTH_CHECK_INTERVAL` | `30` | Redis broker socket health-check interval in seconds. |
| `CELERY_BROKER_VISIBILITY_TIMEOUT` | `3600` | Redis late-ack visibility timeout; keep above expected max document-processing time. |
| `CELERY_BROKER_SOCKET_KEEPALIVE` / `CELERY_BROKER_RETRY_ON_TIMEOUT` | `true` / `true` | Keep Redis sockets alive and retry timeout-level broker operations. |

On startup the worker prints a banner: `Starting Celery worker: pool=threads concurrency=2 prefetch=1`. If the banner shows `pool=solo` while you are expecting parallel processing, override `CELERY_WORKER_POOL` to `threads` (Windows) or `prefork` (Linux).

### Local client backend

```bash
kani-client-backend run --config .env.client
# or
python -m client_backend
```

Additional CLI:

```bash
kani-client-backend doctor --config .env.client        # validate configuration
kani-client-backend doctor --config .env.client --json # machine-readable diagnostics
```

### Demo UI

`demo.py` is the development client UI; it talks to the local sidecar (default `http://127.0.0.1:8100`) via `CHATBOT_API_BASE_URL`. Full three-process dev setup:

1. **Canonical server** (port 8000):
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
2. **Sidecar** (port 8100) — set `CLIENT_SKILLS_ROOT` to the absolute path of `<repo>/skills` so the bundled examples are served (uploads then install into that same folder):
   ```bash
   kani-client-backend run --config .env.client
   ```
3. **Demo UI** (talks to the sidecar):
   ```bash
   streamlit run demo.py
   ```

---

## AI Workflow

The agent workflow is a **LangGraph state machine** defined in [`app/ai/graph.py`](app/ai/graph.py). High-level steps:

1. **Persist current user message** — `MessageService` writes the row to PostgreSQL and reserves the assistant message id. Both ids ride into the workflow so prompt history can exclude the current turn by id (not by tail position) and the final assistant `AIMessage` carries the same id later persisted to the DB.
2. **Hydrate prompt memory** — `ConversationHistoryProvider` (`app/ai/history.py`) returns the durable summary plus recent unsummarized DB messages after the summary cursor. Soft-deleted rows and empty paused/interrupt placeholders are filtered. Per-agent budgets (`chat_history_max_messages` / `_tokens`, …) trim the result.
3. **Router** — [`Router`](app/ai/agents/router.py) invokes Gemini with `ROUTER_SYSTEM_PROMPT` plus server-generated runtime time context and returns one of `chat_agent` / `rag_agent` / `search_agent` / `image_generator_agent` / `planning_agent` / `canvas_agent`. `START` connects directly to `route`.
4. **Agent execution** — the selected agent runs a ReAct-style loop with deferred tool binding, HITL gating, streaming, and the same runtime time context in its system prompt. Configure the local time anchor with `RUNTIME_TIME_CONTEXT_TIMEZONE`; UTC is always included.
5. **Tool execution** — `tool_execution.execute_tool_calls` resolves an origin-aware soft/hard/total deadline policy, shares its attempt budget across retries and MCP reconnect, emits sanitized attempt diagnostics, and keeps compact model-facing errors separate from full UI artifacts.
6. **Auto-continue** — on hitting iteration limits, continuation rounds run until user-configured caps (`auto_continue_max_rounds`, `auto_continue_max_total_iterations`, `auto_continue_timeout_seconds`).
7. **Stream** — every token, reasoning chunk, tool call, artifact, and interrupt is serialized as a structured SSE event.
8. **Persist assistant reply, refresh durable summary, compact checkpoint** — once the terminal `complete` event is received, the service persists the assistant message, schedules a non-blocking durable summary refresh, then compacts the checkpoint transcript with `RemoveMessage` so it does not drift from DB truth.

### Agent cheat-sheet

| Agent | Responsibility | Key tools |
|---|---|---|
| `chat_agent` | General chat + tool use | `web_research`, MCP tools, `tool_search`, skills, handoff |
| `rag_agent` | Document-grounded QA with citation verification | `search_documents`, optional reranker, agentic RAG phases |
| `search_agent` | Web/news answers | `web_research`, `tool_search`, time-context helpers |
| `image_generator_agent` | Gemini image generation | Aspect-ratio / count controls |
| `planning_agent` | Creates / edits task plans | `write_todos`, plan tools |
| `canvas_agent` | Produces canvas/artifact replies | Custom canvas writers |

### Hand-off & delegation

`hand_off` supports graph-validated, LLM-directed delegation. Its per-turn loop guard is
configured with `MAX_HANDOFF_DELEGATION_DEPTH` (default `5`); tool, recursion, and
wall-clock limits remain independent safeguards.

### Deferred tool search

When `MCP_TOOL_SEARCH_ENABLED=true`, only lightweight discovery, pinned tools, already-loaded deferred tools, internal tools, and device-scoped loaded client tools are bound at start. Agents should use a clearly matching bound tool directly and use [`tool_search`](app/ai/tool_search_tool.py) when capability is missing, ambiguous, or tied to an unknown integration/server identifier — with intent-aware scoring in [`tool_search_scoring.py`](app/ai/tool_search_scoring.py), conservative autoload of the single high-confidence recommended tool, TTL eviction, and a per-conversation loaded-tools cap.

Run the deterministic tool-search accuracy checks before changing ranking:

```bash
python scripts/evaluate_tool_search_accuracy.py
python -m pytest tests/test_tool_search_accuracy.py tests/test_unified_tool_search.py -q
```

### Confidence & hallucination controls

`confidence_threshold_abstain`, `confidence_weight_tool_success` / `_completeness` / `_retrieval`, and `enable_citation_verification` gate RAG and tool-driven responses; see [`app/ai/schemas.py`](app/ai/schemas.py) and [`agents/rag_agent.py`](app/ai/agents/rag_agent.py).

---

## Document Pipeline & RAG

1. **Upload** — `POST /documents/uploads` (multipart, plural) accepts one or more files in a single request. Each file is staged under `TEMP_STORAGE_PATH`, a `Document` row is created, and a Celery task is enqueued per file. Duplicate filenames in the same conversation are rejected per-file (case-insensitive) without blocking siblings. The single-file `POST /documents/upload` route remains as a thin compatibility wrapper around the batch path.
2. **Parse** — MinerU handles rich document formats (`.pdf`, `.docx`, `.pptx`, `.html`, `.md`) through the server pipeline. Excel workbooks (`.xlsx`) are parsed server-side with `openpyxl` into markdown tables, and plain `.txt` files are loaded directly. Parser output may include markdown, structured content blocks, tables, formulas, page spans, and extracted image files.
3. **Caption images before indexing** — extracted page images are copied to `DOCUMENT_IMAGES_STORAGE_PATH`; when a Gemini key is available, the image captioning model describes each image. Captions are appended to the matching chunk text before embedding so questions about image-only content can be retrieved semantically.
4. **Chunk** — `DocumentChunkBuilder` creates token-aware chunks from normalized blocks using `RAG_CHUNK_TARGET_TOKENS`, `RAG_CHUNK_OVERLAP_TOKENS`, and `RAG_CHUNK_MAX_TOKENS`. Tables stay atomic when possible, large tables split on row groups, page spans are preserved, and tiny orphan text merges with neighbors.
5. **Persist & index** — `document_chunks` rows are the canonical content store. `DocumentIndexService` replaces chunks idempotently by `document_id`, embeds chunk content through `GeminiRAGEmbeddingService` (`gemini-embedding-2`, `output_dimensionality=3072`) using the document-format prompt `title: {filename} | text: {content}`, and upserts Qdrant points containing lookup metadata only (`document_id`, `chunk_id`, `conversation_id`, `user_id`, page/source metadata, `embedding_provider="gemini"`, `modality="text"`). Queries are embedded with the matching `task: {query_task} | query: ...` prefix.
6. **Retrieval** — the RAG agent uses Qdrant for vector candidate IDs, then hydrates chunk text, filenames, page metadata, and linked image captions/files from PostgreSQL. If a Qdrant point references a missing SQL chunk, it is treated as an index consistency error and skipped rather than serving raw Qdrant payload content.
7. **Agentic RAG** — document-aware chat always uses the agentic `search_documents` tool path. Available actions include `SCAN_ALL`, `READ_DOCUMENT`, `SEARCH_CHUNKS`, `GREP_DOCUMENT`, `LIST_DOCUMENTS`, and `VIEW_IMAGES`.

### RAG embedding migration

Phase 11 swapped the embedding provider from the local `google/embeddinggemma-300m`
SentenceTransformer to the Gemini Embeddings API (`gemini-embedding-2`).
Vectors from the two providers live in different spaces — they are not
portable. Treat the cutover as a cold migration:

1. Set `GEMINI_API_KEY` (required when `RAG_EMBEDDING_PROVIDER=gemini`).
2. Deploy with the new defaults — `QDRANT_COLLECTION_NAME=documents_gemini_embedding_2_3072`
   and `RAG_EMBEDDING_MODEL=gemini-embedding-2`. The startup hook in
   `app/main.py` calls `DocumentIndexService.ensure_collection()` to create
   the collection at the configured `RAG_EMBEDDING_DIMENSION`.
3. Re-embed the existing chunks into the new collection — chunk text is
   already in PostgreSQL, so no re-parse is required:

   ```bash
   python scripts/reindex_embeddings.py --dry-run
   python scripts/reindex_embeddings.py --all --continue-on-error
   ```

4. Verify `qdrant_client.get_collection(name).points_count` equals the count
   of `index_status='indexed'` rows in `document_chunks`. Optionally drop
   the legacy `documents_gemma` collection after a soak window.

The current `RAG_EMBEDDING_DIMENSION` is `3072`. Any future dimension change
should be planned as a separate cold migration with a new
`QDRANT_COLLECTION_NAME` namespace. Mixing Gemma (`google/embeddinggemma-300m`)
and Gemini vectors, or Gemini vectors of different dimensions, in the same
collection is unsupported and rejected by `ensure_collection`.

Raw multimodal image embeddings are disabled by default
(`RAG_MULTIMODAL_IMAGE_EMBEDDINGS_ENABLED=false`); caption-augmented chunks
remain the primary image-retrieval path.

Document lifecycle events (`UPLOAD_STARTED`, `PROCESSING_STARTED`, `PROCESSING_COMPLETED`, `PROCESSING_FAILED`, `DELETED`) are published on an in-process event bus and logged by [`DocumentEventLogger`](app/services/document_event_listener.py).

---

## Provider & Model Configuration

### Per-user credentials

`POST /providers` stores an API key for a provider type (`gemini`, `openai`, `anthropic`). Keys are encrypted with `MODEL_ENCRYPTION_KEY` before persisting in `model_providers` and never returned in plaintext. Supporting endpoints:

- `GET /providers` — list provider records
- `GET /providers/{provider_type}`
- `DELETE /providers/{provider_type}`
- `POST /providers/{provider_type}/validate` — live credential check
- `GET /providers/{provider_type}/models` — provider-curated model catalog

### Per-agent model configuration

`PATCH /model-config` sets provider + model + temperature per agent (chat / rag / search / planning). Defaults fall back to environment-configured models. `GET /model-config/options` returns available combinations. `POST /model-config/reset` restores defaults.

---

## MCP Integration

Server-managed MCP servers are registered under [`/mcp/*`](app/api/mcp.py):

- `GET /mcp/servers`, `GET /mcp/servers/{name}`
- `POST /mcp/servers` — add from JSON spec
- `POST /mcp/servers/from-url` — add a hosted MCP server from a Smithery-style URL
- `DELETE /mcp/servers/{name}` · `PATCH /mcp/servers/{name}/toggle`
- `GET /mcp/tools`, `GET /mcp/tools/{name}`
- `POST /mcp/tools/{name}/execute`

The bundled in-process MCP servers are under [`app/ai/mcp_servers/`](app/ai/mcp_servers/):

| Server | Purpose |
|---|---|
| `calculator_server.py` | Arithmetic |
| `time_server.py` | Current time with timezone handling |
| `tavily_server.py` | Tavily Search, Extract, Map, and Crawl web retrieval tools |
| `brave_image_search_server.py` | Brave Image Search adapter — normalized inline image candidates |
| `widgets_server.py` | Emits interactive widget state + mints tokens |

The same endpoints are exposed by `client_backend` at `/mcp/*` so a desktop UI can configure MCP both globally (server) and per-device (client).

**Global default tools.** Enabled servers in [`app/ai/mcp_config.json`](app/ai/mcp_config.json) are by definition part of the global server catalog, visible to every client (currently `time`, `tavily`, `widgets`, `brave_image_search` — enforced by `tests/test_mcp_global_allowlist.py`). Their raw provider tools remain discoverable through `tool_search`; Tavily and Brave are not system-pinned for the chat or search agent. Those agents instead receive the internal `web_research` tool directly, which centrally runs Tavily retrieval and optional Brave enrichment under one budget and fallback contract. Operators may still opt a raw provider tool into the configurable pinned set. Machine-specific servers (for example, desktop-commander or Excel) belong to the sidecar schema-v2 profile at `<profile>/<server-hash>/<user-id>/devices/<device-identifier>/mcp/config.v2.json`; credentials are stored separately in encrypted form. Use `python -m client_backend mcp migrate` once for an authenticated session, then verify with `python -m client_backend mcp doctor --servers widgets,tavily,time`.

`tavily` is one global server with multiple retrieval tools. `web_research` resolves `tavily_search` internally for ranked factual retrieval; no raw Tavily tool is system-pinned for the search agent. `tavily_search`, `tavily_extract`, `tavily_map`, and `tavily_crawl` remain discoverable through `tool_search` when needed. `tavily_search` returns ranked sources and query-aligned content; it requests no provider-generated answer by default (`include_answer=false`) because the answer model performs final synthesis.

Tavily defaults keep broad search cheap and site-level operations bounded.
Use `TAVILY_SEARCH_DEFAULT_DEPTH=basic` unless you need advanced search by
default. Use existing HITL settings or per-user approval policy to require
approval for `tavily::tavily_crawl` in production deployments where crawl cost
or external traffic needs review.

The model-facing `web_research` tool exposes two optional Tavily controls:

- `topic`: `general`, `news`, or `finance`;
- `time_range`: `day`, `week`, `month`, or `year`.

Its Tavily search call passes `timeout=10` directly to the Tavily SDK/API. This
fixed provider timeout has no environment setting and is not implemented by a
local timeout wrapper.

### Deferred tool binding

`DeferredToolBinding` + `DeferredToolState` ([`app/ai/deferred_tool_*.py`](app/ai/)) record discovery decisions per conversation, enforce TTL, and survive turn boundaries through the LangGraph checkpoint.

### Tool result rendering contract

The backend preserves rich tool render metadata in two places:

- Persisted assistant message metadata: `metadata.tool_artifacts[].render`
- AI SDK streams: `tool-output-available.render`

The model-facing tool message remains compact text. Frontends should render from
`render` when present and fall back to `output` when it is absent.

Supported backend render types:

- `mcp_app`: MCP/App result with a UI template URI such as `_meta["openai/outputTemplate"]`
- `live_widget`: in-repo live widget created by the `widgets` MCP server
- `chart`: structured chart payload
- `table`: structured table payload
- `image`: image content block
- `resource`: MCP resource without an app template
- `json`: structured payload without a richer type
- `text`: plain text payload
- `error`: failed tool result

The frontend owns component rendering. Unknown render types must fall back to JSON or text.

---

## Skills System

Skills are Markdown files with YAML frontmatter describing a capability. The
current parser accepts `name`, `description`, `category`, comma-separated
`tags`, and comma-separated `secrets` (environment-variable names the skill
needs, at most 20, each dropped if the secret store would refuse it);
unsupported metadata is not treated as a security policy. They are loaded by:

- **Client (only source of skills)** — [`LocalSkillsRegistry`](client_backend/services/local_skills_registry.py), scanning the one skills root; synced per-device to the server and resolved at chat time by [`skill_resolver.py`](app/ai/skill_resolver.py) strictly for the originating device.
- To serve this repo's `skills/` folder during development, set the sidecar's `CLIENT_SKILLS_ROOT` to its absolute path.

**One skills root.** `CLIENT_SKILLS_ROOT` is the single directory the sidecar
reads *and* writes: scanning, ZIP installation, guarded updates, and uninstall
all act on it. Leave it unset and it resolves to
`<profile>/<server-hash>/<user-id>/skills/installed`, which keeps two users on
one machine from sharing a catalog. Set it and the sidecar uses that path
verbatim — including installing uploads into it — so pointing it at a working
copy means the installer writes `install.json`, staging directories, and updated
bundles there. It is not a read-only view of someone else's folder.

Frontmatter parsing is shared in [`shared/skills/front_matter.py`](shared/skills/front_matter.py).
Optional user-local examples such as `skills/playwright-cli/` and
`skills/take100/` may be added under the ignored `skills/` directory; they are
not part of the distributed repository.

API (sidecar only — the server has no skills endpoints of its own):

- `/skills`, `/skills/{name}`, `/skills/{name}/toggle`, `/skills/reload`
- `POST /skills/uploads`, `GET /skills/installations/{operationId}` (browser ZIP installation)

When a skill is resolved to a tool (`skills_tool.py`), the agent's skill summaries are injected into its system prompt so it knows *what* is available without paying the schema cost for every skill.

### Executable skills (skill runtime)

An executable skill uses the standard Agent Skills layout: `SKILL.md` plus optional bundle-owned `bin/`, `scripts/`, or `pyproject.toml` assets. Every ready executable skill publishes one fixed client tool, `skill::<skill>::run_skill_command`, which accepts an argv array and runs on the selected sidecar without a shell or global command lookup.

Highlights:

- **Three install paths** — writing a bundle into the skills root by hand; hash-bound `POST /skills/install/preview` plus `POST /skills/install` for local tooling; or a browser ZIP upload (below). All three land in the same directory.
- **Scoped execution** — argv zero resolves only from that skill's bundle or prepared Python environment; no global `PATH` mutation and no arbitrary system-command fallback.
- **Readiness** — `ready` / `not_ready` / `instruction_only`, with explicit setup and rebuild hints; unsafe bundles are rejected or omitted.
- **Python setup** — approved projects are installed into staged, per-skill virtual environments and atomically promoted.
- **Secrets** — a skill declares the environment-variable names it needs in its front matter (`secrets: TAVILY_API_KEY, ACCOUNT_ID`), so the credential form can name them instead of asking the operator to remember them; `GET /skills/{name}/secrets` marks each name `declared` and `configured`. The operator supplies the password-style value. Values are encrypted, per-skill, per-user, and per-machine; they are injected only at execution time and redacted from output/audit. They never synchronize to the server or another device, and never appear in chat history.
- **Permissions & HITL** — every skill command is treated as mutating and passes the existing approval-policy path before secrets or process creation (human confirmation by default, with explicit per-tool preapproval supported).
- **Audit** — one JSONL record per execution under the profile (no secrets or raw output).

This is command confinement, not an OS sandbox: an approved skill still runs with the local sidecar user's host privileges, matching the trust model of coding-agent commands.

Extra sidecar endpoints: `POST /skills/install/preview`, `POST /skills/install`, `POST /skills/{name}/setup`, `POST /skills/uninstall`, `GET /skills/installed`, and per-skill secret GET/POST/DELETE routes.

### Installing a skill from a ZIP

A browser (Streamlit or the AI SDK frontend) installs a skill by uploading one
ZIP to the sidecar. Uploads never reach the canonical server: a skill bundle and
its secrets are device-local.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/skills/uploads` | Upload, validate, extract, and preview one ZIP (`201`) |
| `DELETE` | `/skills/uploads/{uploadId}` | Discard a staged upload |
| `POST` | `/skills/uploads/{uploadId}/install` | Start a new install or guarded update (`202`) |
| `GET` | `/skills/installations/{operationId}` | Poll installation state |
| `DELETE` | `/skills/installations/{operationId}` | Cancel before the commit boundary |

The flow is deliberately two-step. Uploading validates and previews; it never
runs setup code. Installing requires the `expectedSourceHash` from that preview,
plus `approveSetup` when the bundle declares a Python project, plus
`replaceSourceHash` when a skill of the same name already exists — including one
written into the root by hand, since the sidecar owns the whole root. The one
exception is shape: a colliding skill nested below a direct child of the root
reports `preview.existingSkill.replaceable: false` and is never overwritten,
because promotion and rollback move direct children only.

Installation runs asynchronously because a dependency build outlasts an HTTP
request. Poll the returned `statusUrl` with bounded backoff; a failed
installation is reported as HTTP `200` with `state: "failed"`, since the
*operation* was retrieved successfully. Cancelling after the commit boundary
returns `409 SKILL_OPERATION_COMMITTED` — keep polling to the real outcome.

Archives are accepted only as ZIP, and only within the configured limits
(`CLIENT_SKILL_UPLOAD_*`): 25 MiB uploaded, 100 MiB expanded, 50 MiB per file,
2,000 entries, a 200:1 compression ratio, 20 path components, and 240 path
characters. Path traversal, absolute and UNC names, reserved device names,
case- and Unicode-colliding paths, encrypted members, and device, socket, or pipe
entries are rejected before anything is written.

Symbolic links are dropped rather than extracted, and reported as
`archive.skippedLinkCount`. Source downloads of real repositories routinely carry
one, and materializing it is the actual hazard -- on POSIX it would later be
followed out of the bundle, and on Windows it becomes a plain file whose contents
are the target path.

**One archive may install a whole library.** A downloaded skill repository holding
many skills installs as one unit: the preview lists every skill it found, one
approval covers the set, all members are staged before promotion, and a durable
journal restores the previous complete set after a failure or interrupted
process. Catalog refresh and uninstall share the same mutation lock, so stage,
backup, or partially promoted members are never published. The library's preview
name and version come from its plugin manifest (`.claude-plugin/plugin.json` and
the equivalents for other harnesses); collection-level provenance is not yet a
persisted lifecycle object.

**Skills can disclose their own files.** A skill in the current convention keeps
`SKILL.md` short and points at companion documents. Activation lists those files
by relative path and the model reads one with `read_skill_resource`, confined to
that skill's own folder: links refused, regular UTF-8 text files only, size
capped. Binary assets still ship and run; they are not readable as text.

Every catalog response carries `catalogGeneration` and `catalogSyncStatus`. Do
not replace a cached catalog with a lower generation. A `pending` sync means the
skill is installed and working locally while publication to the server is still
outstanding; it is not a failure.

**Frontend contract: [`plans/SKILL_INSTALLATION_FE_CONTRACT.md`](plans/SKILL_INSTALLATION_FE_CONTRACT.md)** — request/response
shapes, the full error-code table, polling policy, and cache rules.

The credential UI returns configured names only, never values. Bindings live in
the local sidecar profile for the current user and device; switching users or
devices does not share them.

**Full guide: [`docs/skill-runtime.md`](docs/skill-runtime.md)** — bundle layout, setup, command confinement, device isolation, secret setup, and troubleshooting.

---

## Planning Mode & Task Plans

Conversations can be put into **planning mode** (`planning_mode_enabled`) to materialise a `TaskPlan` that orchestrates multi-step work.

| Endpoint | Purpose |
|---|---|
| `POST /conversations/{id}/task-plans` | AI-generated plan from a prompt |
| `POST /conversations/{id}/task-plans/manual` | Manually-authored plan |
| `GET /conversations/{id}/task-plans` | List tasks |
| `GET /task-plans/{task_id}` | Retrieve one |
| `PATCH /task-plans/{task_id}` | Update status / order / metadata |
| `POST /task-plans/{task_id}/complete` | Mark complete |
| `DELETE /task-plans/{task_id}` | Remove |
| `GET /conversations/{id}/planning-status` | Plan lifecycle snapshot |

Plan lifecycle: `draft` → `ready` → `executing` → `paused` / `completed`. The planning agent's execution-call budget is capped by `EXECUTION_CALL_BUDGET` and max tasks by `MAX_AUTO_PLAN_TASKS`.

### Planning Rubric Grading

Planning mode includes a native rubric grader for generated and modified task plans.
For each planning attempt, the runtime resolves a context-specific rubric from the
user request, existing plan state, candidate todos, and optional caller-supplied
rubric metadata. The grader evaluates candidate `write_todos` output against that
rubric and returns actionable feedback. If the grader returns `needs_revision`,
the Planning Agent receives the feedback and revises the plan until it is satisfied
or `planning_rubric_max_iterations` is reached.

Rubric results are exposed on assistant message metadata under `planning_rubric`:

    {
      "status": "satisfied",
      "iterations": 1,
      "evaluations": [
        {
          "iteration": 0,
          "result": "satisfied",
          "explanation": "All criteria passed.",
          "criteria": [{"name": "request_fit", "passed": true}]
        }
      ]
    }

Toggle with `planning_rubric_enabled` (default on); cap grader passes with
`planning_rubric_max_iterations` (default 3, minimum 1). The rubric pass cap is
independent of the `planning_max_iterations` graph-loop budget.

---

## Human-in-the-Loop (HITL)

HITL is global (`ENABLE_HUMAN_IN_THE_LOOP=true`) with a per-tool opt-in list (`HITL_TOOLS_REQUIRE_APPROVAL`). When the agent attempts an approvable tool:

1. Execution suspends via `langgraph.types.interrupt(...)`.
2. A `ToolApproval` record is persisted and the message response returns with HTTP **202** plus an interrupt payload.
3. The UI presents the payload to the operator, collects a decision (`approve` / `reject` / edit), and calls `POST /messages/resume-interrupt` (or the equivalent AI-SDK route) with the signed decision.
4. The graph resumes, applies the decision to pending tool calls, and continues streaming.

Timeout handling is Redis-backed; after `HITL_APPROVAL_TIMEOUT_MINUTES` the interrupt is auto-rejected or cleaned up.

### Durable resume recovery

Resume is first-write-wins: when two clients submit decisions for the same
interrupt, exactly one claim can continue. Every client locks all
interrupt-scoped submit controls before opening its resume stream. A duplicate
reported during streaming is terminal, but it is a `200` SSE response with the
typed duplicate code (`INTERRUPT_ALREADY_RESOLVED` or `INTERRUPT_CONFLICT`),
not a second successful resume; pre-stream validation errors remain ordinary
JSON HTTP errors. The internal Streamlit stream uses optional `status_code` /
`error_code` metadata; the AI SDK stream projects the same values as optional
`statusCode` / `errorCode`.

On either duplicate code, make exactly one owner-filtered lifecycle read with
`GET /hitl/interrupts/{interrupt_id}` using `cache: "no-store"`; never
automatically replay the POST. The endpoint exposes only `pending`,
`resolving`, `resolved`, `failed`, or `expired` and returns `404
INTERRUPT_NOT_FOUND` for missing or foreign IDs. `pending` restores the form;
`resolving` shows a non-submittable processing panel with one manual **Check
status** action (no polling); `resolved` refreshes history and returns to chat.
For `failed`, `expired`, or an unavailable lifecycle row, clear the paused UI,
retain an exact-interrupt suppression marker so a stale paused message cannot
rehydrate, and require a new chat message. A continuation that fails after its
claim becomes terminal durable `failed` (`INTERRUPT_FAILED`). Failed and
expired interrupts cannot be resumed by either client.

**Human-in-the-loop approval (per-user).** Beyond the global `HITL_TOOLS_REQUIRE_APPROVAL` floor, approval is governed by a per-user policy stored server-side (`tool_approval_settings`). A rule is either **server-scoped** (gates every tool from an MCP server) or **tool-scoped** (a `"<server>::<tool>"` rule that overrides its server). Precedence: tool rule > server rule > the legacy global floor `hitl_tools_require_approval`; the global `enable_human_in_the_loop` switch is the master kill-switch. The gate resolves each pending call's provenance (client tools from their `client__<server>__<tool>` name, server tools via the MCP manager) so it works for client-sidecar and deferred (search-loaded) tools alike. Manage it from the demo's MCP panel (per-server "Approval" toggle; per-tool Inherit/Require/Skip), which calls `GET/POST/DELETE /hitl/settings` through the sidecar proxy.

---

## Client Runtime Bridge

The bridge lets a trusted local device execute privileged tools without opening its network surface.

**Device lifecycle** — server side ([`/client-devices/*`](app/api/client_devices.py)):

| Endpoint | Purpose |
|---|---|
| `POST /client-devices/register` | Register device, receive `session_id` |
| `POST /client-devices/heartbeat` | Keep-alive |
| `GET /client-devices/me` | List this user's devices |
| `PUT /client-devices/{id}/tool-catalog` | Sync locally-available tools |
| `PUT /client-devices/{id}/skill-catalog` | Sync locally-available skills |
| `GET /client-devices/{id}` | Device details |

**Runtime WebSocket** — [`/device-runtime/{device_id}/connect`](app/api/device_runtime.py) is a bidirectional channel where the server dispatches tool calls and the device responds with structured results. `GET /device-runtime/connected-devices` lists currently-connected devices. Results exceeding `CLIENT_RUNTIME_MAX_TOOL_RESULT_SIZE_BYTES` (1 MB default) are truncated with a warning.

**Client side** — [`client_backend/services/runtime_bridge.py`](client_backend/services/runtime_bridge.py):

- authenticates against the server
- opens and maintains the runtime WebSocket with exponential reconnect
- services tool dispatches through `LocalMCPManager` and the local skills registry
- enforces per-tool timeouts and permission scopes
- periodically syncs tool + skill catalogs back to the server

Configured by `CLIENT_HEARTBEAT_INTERVAL_SECONDS`, `CLIENT_TOOL_CALL_TIMEOUT_SECONDS`, `CLIENT_RECONNECT_*`.

**Device identity** — each installation has a random identifier generated on
first run and persisted as `device_identity.json` in the profile directory
(`CLIENT_PROFILE_ROOT`). Two installations on the same machine are therefore
separate, fully independent devices.

> **Migration note (2026-06):** the identifier used to be derived from machine
> attributes (hostname/MAC). Existing installations re-register as a *new*
> device on next start; previously registered device rows become inert and can
> be cleaned up at any time. Deleting `device_identity.json` likewise
> re-registers the installation as a new device.

> **Migration note (2026-08):** the default profile directory moved from
> `CodexDesktop` / `codex-desktop` to `KaniDesktop` / `kani-desktop`. Nothing is
> copied across: a sidecar started after this change finds an empty profile, so
> it mints a new device identity and no longer sees the previous installation's
> installed skills, MCP configuration, or stored secrets. Point
> `CLIENT_PROFILE_ROOT` at the old directory, or move it to the new name, to keep
> an existing profile. The console script was renamed in the same pass —
> `codex-client-backend` is now `kani-client-backend`, so reinstall the package
> (`pip install -e .`) for the new command to appear and rebuild any client
> bundle in `dist/`.

---

## Live Widgets

Widgets are interactive UI elements rendered by the frontend but driven by the agent.

1. The agent triggers a widget by calling the `widgets` MCP server.
2. The server mints a short-lived widget token via `POST /widgets/{widget_id}/connection`.
3. The frontend opens a stateful WebSocket at `/widgets/{widget_id}/connect`.
4. The server streams widget state, collects user input, and emits terminal events.

Widget state is Redis-backed (see startup banner `"Widget runtime: Redis-backed storage active"`). Without Redis, widget flows degrade; a warning is logged at startup.

### Interactive HTML contract

Each live widget is a self-contained, sandboxed-iframe HTML micro-app. Agents
create it through the `widgets` MCP server with a minimal state envelope shared
by the AI SDK frontend path and the Streamlit `demo.py` path:

```json
{ "html": "<!doctype html>...", "height": 620, "caption": "Optional short caption" }
```

Frontends render the state **only** as a sandboxed iframe from `state.html` — never inject
`state.html` into the main chat DOM. `state.html` is untrusted, executable content.

Contract validation happens at widget-tool time (`app/services/widget_contract.py`). The
checks are shape, not editorial quality: non-object state, empty `html`, or a missing / non-numeric / out-of-range
`height` (260–960) all block create/update with a clear, model-readable error. There is no
`quality_guidance`.

Action resolution endpoint: `POST /widgets/{widget_id}/actions/{action_key}` applies an
optional `state_patch` (re-validated against the HTML contract before it is stored), renders
an action `message_template` from the widget state, records `last_action`, and returns
`{widget_id, session_id, action_key, content}`. Frontends submit the returned `content`
through the normal chat stream — the endpoint does not invoke the assistant directly.

Test commands:

```bash
pytest tests/test_widget_contract.py tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_widget_actions_api.py
pytest tests/test_demo_meaningful_widgets.py tests/test_demo_plan_widget.py tests/test_demo_rich_response.py
pytest tests/client_backend/test_widget_action_proxy.py
```

See [`plans/live-widgets-frontend-integration.md`](plans/live-widgets-frontend-integration.md) §§ 5 and 10 for the full AI SDK HTML-widget renderer contract and reference snippets.

---

## API Reference

The server mounts 15 routers; the client backend mirrors most of them and proxies everything else. URL paths below are server-side.

### Authentication ([`/auth`](app/api/auth.py))

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/auth/signup` | Create account |
| `POST` | `/auth/login` | Issue access + refresh tokens |
| `POST` | `/auth/refresh` | Refresh access token |
| `POST` | `/auth/logout` | Invalidate session |

### Users ([`/users`](app/api/users.py))

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/users/{user_id}` | Profile lookup |

### Conversations ([`/conversations`](app/api/conversations.py))

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/conversations/generate-title` | LLM title synthesis |
| `POST` | `/conversations/` | Create |
| `GET` | `/conversations/` | Paginated list |
| `GET` | `/conversations/{id}` | Retrieve |
| `GET` | `/conversations/{id}/messages` | Paginated messages |
| `PATCH` | `/conversations/{id}` | Update title / persona / planning mode |
| `DELETE` | `/conversations/{id}` | Delete |

### Messages ([`/messages`](app/api/messages.py))

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/messages/` | Non-streaming create; returns **202** if interrupted |
| `POST` | `/messages/stream` | **SSE stream** with heartbeats |
| `POST` | `/messages/stop` | Cancel in-flight generation |
| `POST` | `/messages/resume-interrupt` | **SSE stream** resuming after HITL decision |
| `GET` | `/messages/{id}` | Retrieve |
| `GET` | `/messages/` | Paginated |

### Feedback ([`/messages/{id}/feedbacks`](app/api/feedback.py))

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/messages/{id}/feedbacks` | Create / upsert |
| `GET` | `/messages/{id}/feedbacks` | List |
| `GET` | `/messages/{id}/feedbacks/user` | Current user's record |
| `GET` | `/messages/{id}/feedbacks/stats` | Aggregated rating counts |
| `PUT` | `/messages/{id}/feedbacks/{fid}` | Update |

### Documents ([`/documents`](app/api/documents.py))

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents/uploads` | Canonical batch multipart upload + enqueue (one task per file) |
| `POST` | `/documents/upload` | Legacy single-file wrapper around `/documents/uploads` |
| `GET` | `/documents/task/{task_id}` | Celery task status |
| `GET` | `/documents/{id}` | Document details |
| `GET` | `/documents/conversation/{conversation_id}` | Documents for a conversation |
| `PUT` | `/documents/{id}` | Update metadata |
| `DELETE` | `/documents/{id}` | Delete + Qdrant purge |

Batch upload semantics:

- Send one or more `files` fields plus a `conversation_id` form field.
- Files are processed independently: an invalid or duplicate file is rejected per-file without preventing siblings from being staged.
- Duplicate filename detection is per-conversation and case-insensitive (the server stores a normalized `filename_key` with a unique constraint on `(conversation_id, filename_key)`).
- Status codes: `201` (all accepted), `207` (mixed accepted/rejected), `409` (all rejected as duplicates), `400` (all rejected for validation reasons or empty batch).
- Response shape mirrors input order:

```json
{
  "data": {
    "conversation_id": "...",
    "total_count": 3,
    "accepted_count": 2,
    "rejected_count": 1,
    "files": [
      {"filename": "alpha.pdf", "status": "accepted", "document": {...}, "processing": {"task_id": "..."}},
      {"filename": "alpha.pdf", "status": "rejected", "error_code": "DUPLICATE_FILENAME", "message": "..."}
    ]
  }
}
```

### Task Plans ([`/task-plans` + `/conversations/{id}/task-plans`](app/api/task_plans.py))

See [Planning Mode](#planning-mode--task-plans).

### Providers ([`/providers`](app/api/providers.py))

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/providers` | Store encrypted key |
| `GET` | `/providers` | List |
| `GET` | `/providers/{type}` | Retrieve |
| `DELETE` | `/providers/{type}` | Delete |
| `POST` | `/providers/{type}/validate` | Verify key |
| `GET` | `/providers/{type}/models` | Available models |

### Model Configuration ([`/model-config`](app/api/model_config.py))

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/model-config` | Current per-agent configuration |
| `GET` | `/model-config/options` | Available providers + models |
| `PATCH` | `/model-config` | Update per-agent |
| `POST` | `/model-config/reset` | Restore defaults |

### MCP ([`/mcp`](app/api/mcp.py))

See [MCP Integration](#mcp-integration).

### Client Devices ([`/client-devices`](app/api/client_devices.py))

See [Client Runtime Bridge](#client-runtime-bridge).

### Device Runtime ([`/device-runtime`](app/api/device_runtime.py))

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/device-runtime/connected-devices` | Currently-online devices |
| `WS`  | `/device-runtime/{device_id}/connect` | Runtime WebSocket |

### Widgets ([`/widgets`](app/api/widgets.py))

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/widgets/{widget_id}/connection` | Mint short-lived handshake token |
| `POST` | `/widgets/{widget_id}/actions/{action_key}` | Resolve an action template into a chat message |
| `WS`  | `/widgets/{widget_id}/connect` | Widget state WebSocket |

### AI SDK surface ([`/ai/*` + `/api/chat/*`](app/api/ai_sdk.py))

Vercel AI SDK compatible streaming + UIMessage-format routes:

| Method | Path |
|---|---|
| `POST` | `/ai/conversations` |
| `GET` | `/ai/conversations` |
| `GET` | `/ai/conversations/{conversation_id}` |
| `PATCH` | `/ai/conversations/{conversation_id}` |
| `DELETE` | `/ai/conversations/{conversation_id}` |
| `GET` | `/ai/conversations/{conversation_id}/messages` |
| `POST` | `/api/chat/{conversation_id}` |
| `POST` | `/ai/chat/{conversation_id}` |
| `POST` | `/ai/resume-interrupt` |

Responses set `X-Vercel-Ai-UI-Message-Stream: v1` (whitelisted as a CORS expose-header).

### Health

`/health`, `/health/celery`, `/health/redis`, `/health/qdrant`, `/health/all` — the aggregate endpoint returns `degraded` if any dependency is unhealthy.

### Client backend routes

In addition to proxying most server routes under both `/...` and `/api/...`, the client backend adds:

- `/runtime/status`, `/runtime/connect`, `/runtime/disconnect`, `/runtime/refresh-catalogs`
- `/skills`, `/skills/{name}`, `/skills/{name}/toggle`, `/skills/reload`
- `POST /skills/uploads`, `GET /skills/installations/{operationId}` (browser ZIP installation)
- `/mcp/*` — local MCP lifecycle
- `/auth/restore`, `/auth/session`, `/auth/verify-local-token` — local-session wrappers
- `/status`, `/device` — local-runtime state

---

## Streaming, SSE & WebSocket Endpoints

| API namespace | Endpoint | Stream protocol | Primary events |
| --- | --- | --- | --- |
| assistant-ui / AI SDK v6 | `POST /api/chat/{conversation_id}` and `POST /ai/chat/{conversation_id}` | Vercel AI SDK UI Message Stream over SSE | `start`, `start-step`, `text-start`, `text-delta`, `reasoning-start`, `reasoning-delta`, `tool-input-start`, `tool-input-available`, `tool-output-available`, `data-interrupt`, `data-rich-items`, `data-image-preview`, `finish-step`, `finish`, `[DONE]` |
| assistant-ui / AI SDK v6 | `POST /ai/resume-interrupt` | As above | As above |
| Streamlit internal client | `POST /messages/stream` and `POST /messages/resume-interrupt` | Backend JSON SSE compatibility stream | `user_message_created`, `agent_selected`, `token`, `thinking`, `tool`, `rich_items`, `image_preview`, `interrupt`, `title_updated`, `complete`, `error`, `heartbeat` |
| Backend internal | service layer | Canonical v3 event model (`V3StreamEvent`, `schema_version="v3"`) | `message_delta`, `reasoning_delta`, `tool_call_available`, `tool_execution_end`, `image_preview`, `subagent_start`, `subagent_end`, `interrupt`, `complete`, `error` |
| Device runtime | WS `/device-runtime/{device_id}/connect` | WebSocket | Tool dispatch + results |
| Live widgets | WS `/widgets/{widget_id}/connect` | WebSocket | Widget state streaming |

Heartbeat interval for SSE: **1 s**. `SUPPRESS_INTERNAL_STREAM_CHUNKS=true` drops internal events (e.g. summarisation output) before they reach clients.

---

## Inline Rich Response (v1)

The backend supports an **opt-in inline-rich-response contract** that lets agents place selected images, live widgets, tool renders, canvas artifacts, citations, and resource links at specific positions inside the markdown answer. Enabled by default with `INLINE_RICH_RESPONSE_ENABLED=true`; keep the flag as a server-side kill switch and have each client advertise the per-request capability before receiving marker-bearing content.

### Marker syntax

Rich items are placed using a standalone block-level HTML comment:

```markdown
The pressure differential causes lift over the upper wing surface.

<!--rich:image:tool:call_7:0-->

This pattern helps explain why the pressure is lower above the wing.
```

Marker rules:

- Must appear on its own line, optionally with up to three leading spaces and trailing whitespace.
- `<id>` may contain ASCII letters, digits, `_`, `-`, `.`, and `:` and is at most 128 characters.
- Markers inside fenced (` ``` ` or `~~~`) or indented (4-space) code blocks are treated as literal markdown.
- Unknown ids produce a neutral unavailable-content block plus a validation warning in `metadata.rich_reference_warnings`.

### Stable item IDs

| Origin | ID format |
|---|---|
| Tool result render | `tool:<tool_call_id>` |
| Image from a tool result | `image:tool:<tool_call_id>:<zero_based_index>` |
| Image group from a tool result | `imagegroup:tool:<tool_call_id_or_query_digest>` |
| RAG document image | `image:document:<document_image_id>` |
| Generated image | `image:generated:<assistant_message_id>:<zero_based_index>` |
| Live widget | `widget:<widget_id>` |
| Canvas artifact | `canvas:main` (stable across conversation revisions) |
| Structured citation item | `citation:<assistant_message_id>:<zero_based_index>` |

### Per-message metadata

Capable assistant messages persist:

```json
{
  "rich_items_version": 1,
  "rich_items": [
    {
      "id": "image:tool:call_7:0",
      "type": "image",
      "source": "web_search",
      "display_policy": "inline_only",
      "alt_text": "Airflow around an airfoil",
      "payload": {
        "url": "/web-images/55d170b5-b0f0-44fc-9155-af8af484513d",
        "mime_type": "image/png",
        "source_url": "https://example.org/airfoil-study",
        "width": 1200,
        "height": 800
      }
    }
  ],
  "rich_reference_warnings": []
}
```

Item types emitted today: `image`, `image_group`, `live_widget`, `tool_render`, `canvas_artifact`. The schema also reserves `citation` and `resource_link` for forward compatibility, but no backend path constructs them yet. Payloads are type-validated through a Pydantic discriminated union with `extra="forbid"`; only renderer-consumed fields are accepted, and items are serialized with null-valued keys omitted. An `image` payload accepts exactly one of `url` (https / dev-only http) or `data` (base64 of an allowed raster MIME); an `image_group` contains an ordered `items` list of URL-backed image cells.

### Display policy

- `inline_only` — render only at its marker. Both `image` and `image_group` items use this policy and are **never** appended implicitly. Unreferenced image candidates are dropped entirely.
- `inline_or_append` — render inline when referenced, otherwise append below the body. Used by `live_widget`, `tool_render`, `canvas_artifact` (and the reserved `citation`/`resource_link` types).

For capable responses, widget placement is authored dynamically in the response body: a `<!--rich:widget:<widget_id>-->` marker selects its position. The server does not insert a marker when the response omits one.

For v1 messages, the legacy `metadata["images"]` gallery is suppressed in the Streamlit renderer and only selected images surface as AI SDK `file` parts. Pre-v1 messages (no `rich_items_version`) retain their existing gallery.

### Capability negotiation

Clients opt in per-request by setting `inline_rich_response_v1: true` (camelCase or snake_case) on:

- `POST /messages` and `POST /messages/stream` body (`MessageCreate.inline_rich_response_v1`).
- `POST /ai/chat/{conversation_id}` body extras (`inlineRichResponseV1` / `inline_rich_response_v1`).
- `POST /messages/resume-interrupt` and `POST /ai/resume-interrupt` body (`InterruptResumeRequest.inline_rich_response_v1`).
- `GET /ai/conversations/{conversation_id}/messages` query (`inlineRichResponseV1=true` / `inline_rich_response_v1=true`) when refetching marker-bearing history.

Capability is then ANDed with `INLINE_RICH_RESPONSE_ENABLED` server-side. AI SDK history reads without the read capability receive a legacy projection: standalone marker lines are stripped from `content`, and `rich_items` / `rich_items_version` / `rich_reference_warnings` are removed from metadata.

### Transient stream events

Capable AI SDK streams emit additive `data-rich-items` parts as safe non-image records become available:

```json
{
  "type": "data-rich-items",
  "data": {
    "operation": "upsert",
    "items": [{"id": "widget:2f13", "type": "live_widget", "payload": { ... }}]
  },
  "transient": true
}
```

Image candidates are **never** streamed transiently — they only surface in the final `data-assistant-message.data.message.metadata.rich_items` after marker selection. Canvas source is excluded from transient upserts. Transient data parts ride in `useChat({ onData })`, not in `message.parts`. The AI SDK response retains the `x-vercel-ai-ui-message-stream: v1` header.

### Rich image provider, latency, and failure policy

Brave Image Search is the preferred adapter for focused visual discovery because it supplies native confidence, rank, and dedicated thumbnail metadata. Eligible high-confidence results are selected in provider order; medium-confidence results are considered only when no high-confidence candidate survives. Selection is confidence-based metadata processing: the path does not inspect image pixels or run an image-quality model. The Brave-proxied thumbnail is the preferred display URL and remains eligible when no original image URL is available; original image URLs are never exposed in public message metadata. `tavily_search` returns text and sources only — it never returns images, so `brave_image_search` is the only source of web images.

Candidate filtering is deterministic, but retrieval and presentation have separate bounds. `BRAVE_IMAGE_SEARCH_DEFAULT_COUNT` requests `6` raw provider results by default (with `BRAVE_IMAGE_SEARCH_MAX_COUNT=10` as the request ceiling). Provider-native discovery inspects that bounded response before confidence-tier selection and original-image deduplication; it intentionally does not use the generic raw-tool harvesting cap. `RICH_IMAGE_CANDIDATE_MAX_COUNT` (default `8`) bounds generic/raw tool-result candidate harvesting. Both paths apply `RICH_IMAGE_MIN_WIDTH_PX` (`320`) and `RICH_IMAGE_MIN_HEIGHT_PX` (`180`) when original dimensions are known.

Presentation is bounded separately: `RICH_AUTO_PLACE_MAX_IMAGES` (default `2`) caps figure-mode image items per answer, while `RICH_IMAGE_GALLERY_MAX_ITEMS` (default `6`, hard maximum `8`) caps the ordered cells inside one provider-native `image_group`. The legacy raw-Brave grouping path remains capped by `RICH_IMAGE_GROUP_MAX_ITEMS=3`. An image group counts as one model-facing inventory item and one marker, but the renderer loads and displays every selected cell. An image the model did not place itself is anchored on its own image-search query — there is a single image-placement path, with no description-matching alternative and no rollback flag. `RICH_AUTO_PLACE_MIN_SCORE` governs widget auto-placement only; `RICH_IMAGE_ANCHOR_MIN_SCORE` (`0.34`) is the image-query threshold. See the [rich image rendering contract](docs/frontend/rich-image-rendering.md) for the per-origin anchoring rules.

Creating `/web-images/{id}` references is a persistence-time database-only operation. Upstream image bytes are fetched later, only when an authenticated client requests the media route, so the configured connect/read timeouts do not extend text time-to-first-token or assistant completion. A fetch failure affects only the optional figure; reference-registration failure removes the item and its exact marker while preserving the complete text answer.

Operational metrics are exposed at `GET /metrics/rich-images`. Labels are bounded to provider and fixed outcome codes; queries, URLs, captions, tenant IDs, and other user content are never labels. `REMOTE_IMAGE_ENRICHMENT_ENABLED` disables Brave discovery independently, while `INLINE_RICH_RESPONSE_ENABLED` remains the complete server-side rich-response rollback switch. Frontend teams should implement the full [rich image rendering contract](docs/frontend/rich-image-rendering.md), including Bearer fetch, object-URL cleanup, file-part deduplication, one-footer ownership, and whole-figure failure replacement.

### Client renderer algorithm

1. Opt in with the per-request capability flag.
2. In `useChat({ onData })`, maintain a map of safe non-image `rich_items` from transient `data-rich-items` upserts.
3. Accumulate `text-delta` content normally.
4. Split the accumulated text only on standalone complete markers into blocks.
5. Render a known typed item at that block position; while a streamed marker is waiting for its widget upsert, render a lightweight inline placeholder there.
6. On final `data-assistant-message`, replace transient registry data with persisted `metadata.rich_items` when present. Refetch `GET /ai/conversations/{conversation_id}/messages?inlineRichResponseV1=true` after `finish` for the authoritative final content layout when auto-placement may have inserted markers during persistence.
7. Append only unreferenced items whose `display_policy == "inline_or_append"`. Never build an image gallery from unreferenced image candidates or legacy `images`.

### Migration rules

- Legacy non-image metadata (`tool_artifacts`, `live_widgets`, `canvas_artifact`, citations) is retained for fallback clients.
- `rich_items` is the new placement contract.
- For new v1 messages the automatic appended images gallery is disabled; legacy messages keep their gallery until an explicit migration/backfill decision.
- External AI SDK clients that want the report layout must declare `inline_rich_response_v1`, implement the HTML-comment marker resolver, and consume `data-rich-items` in `onData`. Non-opt-in clients receive a marker-free legacy projection and no `data-rich-items` parts.
- Custom backends/proxies must keep the `x-vercel-ai-ui-message-stream: v1` header required by the AI SDK UI Message Stream protocol.

---

## OpenAPI & Postman

- Server OpenAPI: `http://localhost:8000/docs` (ReDoc at `/redoc`)
- Client backend OpenAPI: `http://127.0.0.1:8100/docs`
- Postman collection: [`Chatbot API.postman_collection.json`](Chatbot%20API.postman_collection.json)

The Postman collection covers every HTTP route — planning, providers, model config, MCP, skills, client-device registration, widget-token handshake, AI SDK — but **does not** model the WebSocket flows (`/device-runtime/{device_id}/connect`, `/widgets/{widget_id}/connect`). Use a WebSocket client (e.g. `wscat`, Postman's WebSocket workspace) for those.

---

## Testing

Run the suite:

```bash
pytest
```

Curated subsets:

```bash
pytest tests/test_graph_streaming_summarization.py   # summarisation + streaming
pytest tests/test_router.py                          # router LLM selection
pytest tests/test_tool_search_scoring.py \
       tests/test_tool_search_prompt_guidance.py \
       tests/test_unified_tool_search.py             # deferred tool search
pytest tests/test_hitl_config.py tests/test_hitl_decision_mapping.py
pytest tests/test_widget_runtime.py tests/test_widgets_api.py
pytest tests/test_skills_*                           # skills architecture + parity
pytest tests/client_backend                          # local sidecar
```

Coverage spans:

- multi-agent streaming + summarisation
- HITL decision mapping and config wiring
- checkpoint serialisation / tool execution recovery / multi-sidecar hardening
- tool scope isolation, per-agent tool allowlists
- MCP adapter utilities and config-Redis wiring
- widget API + runtime
- skills architecture, parity, and snapshot
- client-backend bundle, auth, CORS, conversations, SSE keepalive, runtime bridge, local MCP manager

Fixtures live under [`tests/fixtures/`](tests/fixtures/).

The client/backend live-server suite is intentionally opt-in so a normal test
run never waits on or mutates a developer-specific API instance. Start a
disposable API server, set `RUN_LIVE_SERVER_TESTS=1`, and set
`LIVE_SERVER_TEST_URL` when the server is not at `http://127.0.0.1:8000`:

```powershell
$env:RUN_LIVE_SERVER_TESTS = "1"
$env:LIVE_SERVER_TEST_URL = "http://127.0.0.1:8000"
pytest tests/client_backend/test_live_server_integration.py
```

---

## Observability

- **LangSmith** — set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` to stream traces to `LANGSMITH_PROJECT` (default `sample-chatbot`). The config module translates these into the `LANGCHAIN_*` variables LangChain expects.
- **Centralised exceptions** — `register_exception_handlers` wires `APIException` → structured JSON with `{status, error_code, message, details}`.
- **Health endpoints** — see above.
- **Event bus** — `app.core.events.get_event_bus()` publishes `DocumentEvent` values consumed by `DocumentEventLogger`.

---

## Packaging & Distribution

The client backend can be bundled for desktop distribution:

```bash
pwsh -File scripts/build-client-backend-bundle.ps1
```

The bundle scripts generate artifacts under the ignored
`dist/client-backend-bundle/` directory. The PowerShell and Python builders copy
the same tracked templates from `scripts/client-backend-bundle/`; edit those
templates instead of embedding launcher text in either builder.

On Windows, run `start-client-backend.bat` (Explorer/cmd) or
`./start-client-backend.ps1` (PowerShell). The launcher supports Windows
PowerShell 5.1 and PowerShell 7 without relying on `Get-FileHash`. It selects
Python 3.10+, creates or repairs the bundle-owned `.venv`, bootstraps missing pip
with `ensurepip`, and installs `requirements-client.txt` only when its SHA-256 or
the interpreter major/minor changes. The marker is written only after a
successful install. If the venv is corrupt, only the bundle-owned `.venv` is
replaced; `.env.client` is preserved. Launcher and sidecar failures propagate as
non-zero exit codes.

`pyproject.toml` defines the console script:

```toml
[project.scripts]
kani-client-backend = "client_backend.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["app", "client_backend"]
```

Wheels can be built with `python -m build`.

---

## MinerU Persistent Service

By default, every document parse cold-starts a temporary MinerU process, incurring 30–90 s GPU model load overhead. To eliminate this, run the persistent `mineru-api` FastAPI service:

```powershell
pwsh -File scripts/start_mineru_service.ps1
```

This binds the service to `localhost:8765` and loads GPU models once on startup. Then configure your `.env`:

```env
MINERU_API_URL=http://localhost:8765
```

The service must be running before starting Celery workers or submitting document parsing tasks.

To register as a Windows service (persistent across reboots), use NSSM:

```powershell
nssm install MinerUService "C:\path\to\Scripts\mineru-api.exe" "--host 0.0.0.0 --port 8765"
$projectRoot = (Resolve-Path ".").Path
nssm set MinerUService AppDirectory $projectRoot
nssm start MinerUService
```

See [`scripts/start_mineru_service.ps1`](scripts/start_mineru_service.ps1) for full usage options and NSSM commands.

### Verifying warm parse (no cold start)

Run the same PDF through the upload API twice and compare parse times:

1. Start the MinerU service: `scripts/start_mineru_service.ps1`
2. Set `MINERU_API_URL=http://localhost:8765` in `.env`, restart Celery workers
3. Upload any multi-page PDF via the API
4. Note `parse_s` in the worker logs (look for: `MinerU completed for ... in X.XXs`)
5. Upload the **same PDF** again immediately
6. Compare: second `parse_s` should be 80-90% lower (no model load overhead)

Expected results:
| Run | parse_s | model load? |
|-----|---------|-------------|
| Cold (first) | ~40-90 s | Yes — models load from disk |
| Warm (second+) | ~3-10 s | No — models already in memory |

If both runs show similar times, the service is restarting between requests
(check NSSM service status: `nssm status MinerUService`).

---

## Troubleshooting

| Symptom | Likely cause & fix |
|---|---|
| `Widget runtime: Redis unavailable` warning | Start Redis (`docker compose -f docker-compose.redis.yml up -d redis`) and set `REDIS_URL`. Widget flows degrade without it. |
| `Router Gemini client not initialized` | Missing `GEMINI_API_KEY`. Set it or register a Gemini provider via `POST /providers`. |
| SSE disconnects after 60 s | Some proxies buffer; deploy with HTTP/2 or disable proxy buffering. The in-app heartbeat is 1 s. |
| Checkpoint table errors on startup | Ensure `langgraph-checkpoint-postgres` migrations run by not disabling `ENABLE_LANGGRAPH_CHECKPOINTS` before first boot. |
| Provider key decryption fails | `MODEL_ENCRYPTION_KEY` changed. Re-create provider records or restore the prior key. |
| Windows + async + `localhost` Redis | The config normaliser rewrites `localhost` → `127.0.0.1` automatically on `win32`. |
| Document uploads stuck in `processing` | Celery worker not running: `python -m app.workers.start_worker`. Check `/health/celery`. |
| Multiple uploads process one at a time | Check the worker startup banner. On Windows, pool must be `threads` (or another parallel pool); `solo` is single-task debug mode. Set `CELERY_WORKER_POOL=threads` or leave at `auto`. |
| `consumer: Connection to broker lost` / Redis `WinError 10054` | Redis was restarted or the TCP connection was reset. Confirm `docker ps` shows `sample_chatbot_redis` healthy, then restart the worker. The worker config enables reconnects and cancels late-ack tasks on broker loss to avoid duplicate concurrent document processing after redelivery. |
| Batch upload returns 207 | Mixed accepted/rejected response. Inspect `data.files` for per-file status and `error_code` (e.g. `DUPLICATE_FILENAME`). Do not treat 207 as a hard failure. |
| Reranker download slow / `ReadTimeoutError` from `huggingface.co` | The reranker loads offline-first from the local HF cache, so a cached model never blocks on the hub. The error means the model isn't cached yet (first run) or the one-time download timed out. Pre-fetch it with `python scripts/download_reranker.py`, then it loads with no network. Or disable with `ENABLE_RERANKING=false`. |
| Client-device tool calls fail | Device offline or `CLIENT_RUNTIME_REQUIRE_CONNECTED_DEVICE_FOR_LOCAL_TOOLS=true`. Inspect `GET /device-runtime/connected-devices`. |
