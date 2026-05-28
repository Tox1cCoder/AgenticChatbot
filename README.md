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
| **Client runtime bridge** | Devices register, heartbeat, sync tool/skill catalogs, and receive WebSocket-dispatched tool calls — enabling local shell/filesystem/MCP execution without exposing them to the public network. |
| **Skills** | Markdown-defined skills with YAML frontmatter, resolved to tools at runtime. Shared registry spans server, client, and device ([`app/ai/skills_registry.py`](app/ai/skills_registry.py), [`client_backend/services/local_skills_registry.py`](client_backend/services/local_skills_registry.py)). |
| **Live widgets** | Token-minted handshake (`POST /widgets/{id}/connection`) followed by a stateful WebSocket (`/widgets/{id}/connect`) for interactive, server-driven UI components. |
| **Summarization middleware** | Context-budget-aware rolling summarisation ([`summarization_middleware.py`](app/ai/summarization_middleware.py)) with token/message/fraction triggers, hard summary caps, and fail-closed timeouts. |
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
- **Persistence**: SQLAlchemy 2.x + Alembic (28 migrations), PostgreSQL 14+, psycopg driver
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
│   │   ├── mcp_servers/              Built-in MCP servers (calculator, tavily, time, widgets, form_filler, boring_reader)
│   │   ├── graph.py                  MultiAgentWorkflow + streaming + HITL
│   │   ├── memory.py                 Conversation memory manager
│   │   ├── summarization_middleware.py   Rolling summarisation
│   │   ├── token_instrumentation.py  History-budget + token truncation
│   │   ├── tool_search_tool.py       Deferred tool loading
│   │   ├── deferred_tool_*.py        Deferred binding + state machine
│   │   ├── skills_*.py               Skill registry / resolver / snapshot / tool
│   │   └── model_factory.py          Provider-agnostic LLM instantiation
│   ├── api/                          FastAPI route modules (15 routers)
│   ├── core/                         config, DI container, auth, exceptions, runtime modelling
│   ├── database/                     session / engine / migration bootstrap
│   ├── factories/                    Pydantic/domain factories
│   ├── interfaces/                   Service interface contracts (ABCs)
│   ├── models/                       15 SQLAlchemy ORM models
│   ├── repositories/                 persistence + query strategy + command strategy
│   ├── schemas/                      Pydantic schemas and API contracts
│   ├── services/                     business logic, orchestration, event listeners
│   ├── storage/                      document image storage
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
│   ├── cli.py                        codex-client-backend entrypoint (run / doctor)
│   └── main.py                       FastAPI app factory
├── shared/skills/                    Shared skill parsing helpers (front matter)
├── skills/                           Bundled skills (playwright-cli, take100)
├── tests/                            Unit + integration tests (server and client_backend)
├── scripts/build-client-backend-bundle.ps1   Client bundle builder
├── dist/client-backend-bundle/       Pre-built client distribution
├── docker-compose.redis.yml          Local Redis with persistence + auth
├── alembic.ini                       Alembic runtime config
├── demo.py                           Streamlit demo UI
├── demo_requirements.txt             Demo-only dependencies
├── upload_support.py                 Streamlit document upload helper
├── pyproject.toml                    Project metadata, deps, scripts (codex-client-backend)
├── environment.yml                   Conda environment snapshot
├── Chatbot API.postman_collection.json
└── README.md
```

---

## Prerequisites

| Component | Requirement | Notes |
|---|---|---|
| Python | **3.10+** | 3.11 or 3.12 recommended |
| PostgreSQL | **14+** | Required; holds auth, conversations, plans, feedback, HITL state, LangGraph checkpoints |
| Redis | **7+** | Strongly recommended. Required for Celery, live widgets, client runtime state, HITL timeouts |
| Qdrant | latest | Required for document retrieval / RAG |
| Node.js | optional | Only for consumers using the `@ai-sdk` client |
| Docker | optional | Helpers provided for Redis + Qdrant |

At least one **LLM provider credential** is required for real AI execution:

- `GEMINI_API_KEY` — the default provider, wired via `langchain-google-genai`
- per-user OpenAI / Anthropic keys managed through [`/providers`](app/api/providers.py) once `MODEL_ENCRYPTION_KEY` is set

Optional: `TAVILY_API_KEY` for web search agent, `SMITHERY_API_KEY` for hosted MCP servers, `LANGSMITH_API_KEY` for tracing.

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

A conda environment snapshot is available as [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
conda activate sample-chatbot
```

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
| `TAVILY_API_KEY` | — | Web search |
| `SMITHERY_API_KEY` | — | Hosted MCP registry |
| `MODEL_ENCRYPTION_KEY` | — | Fernet key for per-user provider credentials |
| `RAG_AGENT_MODEL` | `gemini-3.1-pro-preview` | |
| `CHAT_AGENT_MODEL` | `gemini-3-flash-preview` | |
| `SEARCH_AGENT_MODEL` | `gemini-3-flash-preview` | |
| `IMAGE_GENERATOR_MODEL` | `gemini-3-pro-image-preview` | |
| `IMAGE_CAPTION_MODEL` | `gemini-3-flash-preview` | |
| `MEDIA_RESOLUTION` | `high` | `low` / `medium` / `high` (Gemini 3 per-part) |
| `ENABLE_THINKING` | `true` | |
| `THINKING_LEVEL` | `high` | `minimal` / `low` / `medium` / `high` (Gemini 3) |
| `THINKING_BUDGET` | `-1` | Token budget for Gemini 2.5 (-1 dynamic, 0 off) |
| `ENABLE_GEMINI_CODE_EXECUTION` | `true` | Native code-execution tool |

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
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANK_TOP_K` | `10` |
| `RAG_CHUNK_TARGET_TOKENS` | `400` |
| `RAG_CHUNK_OVERLAP_TOKENS` | `40` |
| `RAG_CHUNK_MAX_TOKENS` | `800` |
| `RAG_INDEX_BATCH_SIZE` | `16` |

`GEMINI_API_KEY` is required when `RAG_EMBEDDING_PROVIDER=gemini`. The
`sentence_transformers` fallback is for offline development; switching
providers requires a deliberate Qdrant collection cutover (see "RAG
embedding migration" below).

### Conversation memory & history budgets

`MEMORY_MAX_MESSAGES`, `CHAT_HISTORY_MAX_MESSAGES` / `_TOKENS`, `RAG_HISTORY_MAX_*`, `SEARCH_HISTORY_MAX_*`, `PLANNING_HISTORY_MAX_*`.

### Durable conversation memory (memory refactor 2026-04-29)

Prompt memory is built in one place — `app.ai.history.ConversationHistoryProvider` — and combines a durable per-conversation summary stored in PostgreSQL (`conversation_memory_summaries`) with the recent unsummarized messages from the `messages` table. The summary cursor is a database `messages.id`, never a LangChain message id, so prompt history can never overlap with the summary text.

| Variable | Default | Notes |
|---|---|---|
| `MEMORY_CACHE_TTL_SECONDS` | `60` | TTL for the in-process prompt-history cache. |
| `MEMORY_CACHE_MAX_CONVERSATIONS` | `256` | LRU cap before older conversations are evicted. |
| `MEMORY_SUMMARY_MIN_UNSUMMARIZED_MESSAGES` | `60` | Refresh threshold; `0` disables. |
| `MEMORY_SUMMARY_MIN_UNSUMMARIZED_TOKENS` | `18000` | Token threshold; `0` disables. |
| `MEMORY_SUMMARY_KEEP_MESSAGES` | `8` | Newest messages to keep out of the summary. |
| `MEMORY_SUMMARY_MAX_TOKENS` | `1500` | Hard cap on the generated summary text. |
| `MEMORY_SUMMARY_TIMEOUT_SECONDS` | `30` | Fail-closed timeout for the summarizer call. |

Operational behaviour:

- Summaries are best-effort and fail-closed — a timeout or model error leaves the previous summary in place.
- Prompt history is cached by `(conversation_id, user_id, current_message_id, agent_key, summary_cursor, summary_version)`; transcript writes invalidate the cache.
- Soft-deleted messages and empty paused/interrupt assistant placeholders are excluded from prompt history.
- Checkpoint state is **not** long-term memory. After the terminal assistant response is persisted, the service issues `RemoveMessage` for every checkpoint message id; PostgreSQL is the canonical transcript.
- AI SDK clients may post their full UI history at `POST /api/chat/{conversation_id}`; the server only consumes the latest user message and rebuilds prior memory from the database.

### Summarization middleware (rolling, off the hot path)

`ENABLE_SUMMARIZATION`, `SUMMARIZATION_TRIGGER_TOKENS`, `SUMMARIZATION_TRIGGER_MESSAGES`, `SUMMARIZATION_TRIGGER_FRACTION`, `SUMMARIZATION_MODEL_CONTEXT_SIZE`, `SUMMARIZATION_KEEP_MESSAGES`, `SUMMARIZATION_MODEL`, `SUMMARIZATION_MAX_SUMMARY_TOKENS`, `SUMMARIZATION_TIMEOUT_SECONDS`.

After the memory refactor, summarization no longer runs on the streaming hot path. `MessageService.refresh_summary_after_turn` schedules the durable refresh via `asyncio.create_task` once the assistant message is persisted. The `_summarization_node` graph node was reduced to a no-op; `START` now connects directly to `route` so streaming yields the first user-facing token without waiting on a summary call.

### Document processing

`MINERU_TIMEOUT`, `MINERU_API_URL`, `MINERU_BACKEND` (`pipeline` / `hybrid-*` / `vlm-*`), `MINERU_METHOD` (`auto` / `txt` / `ocr`), `MINERU_LANG`, `MINERU_EXTRA_ARGS`, `EXTRACT_FORMULAS_FROM_PDF`, `EXTRACT_TABLES_FROM_PDF`, `TABLE_FORMAT`, `MAX_FILE_SIZE_MB`, `TEMP_STORAGE_PATH`, `DOCUMENT_IMAGES_STORAGE_PATH`, `IMAGE_CAPTION_MODEL`, `IMAGE_CAPTION_MAX_RETRY_ATTEMPTS`, `IMAGE_CAPTION_RETRY_DELAY_SECONDS`.

### MCP tool search

`MCP_TOOL_SEARCH_ENABLED`, `MCP_TOOL_SEARCH_DEFAULT_TOP_K`, `MCP_TOOL_SEARCH_AUTOLOAD_TOP_K`, `MCP_TOOL_SEARCH_PINNED_TOOLS`, `MCP_TOOL_SEARCH_MAX_LOADED_TOOLS_PER_CONVERSATION`, `MCP_TOOL_SEARCH_LOADED_TOOLS_TTL_MINUTES`, `MCP_TOOL_SEARCH_MIN_RELEVANCE_SCORE`, `MCP_TOOL_SEARCH_AUTOLOAD_MIN_RELEVANCE_SCORE`.

### HITL & planning

`ENABLE_HUMAN_IN_THE_LOOP`, `HITL_TOOLS_REQUIRE_APPROVAL`, `HITL_APPROVAL_TIMEOUT_MINUTES`, `MAX_AUTO_PLAN_TASKS`, `EXECUTION_CALL_BUDGET`, `PLANNING_MAX_ITERATIONS`, `PLANNING_CONSECUTIVE_ERRORS_LIMIT`.

### Planning-mode subagents

`PLANNING_SUBAGENTS_ENABLED`.

While Planning mode is active, the Planning Agent can call the internal `dispatch_subagents` tool to fan out independent worker tasks to other graph agents (`chat_agent`, `rag_agent`, `search_agent`, `image_generator_agent`, `canvas_agent`). Workers run concurrently in the same chat turn — there is no background queue and the dispatch call blocks until every worker completes, fails, or signals it needs human approval. Workers run with isolated message state, inherit scoped identifiers (`conversation_id`, `user_id`, `device_id`) and runtime model overrides, and return summaries to the Planning Agent using the same generic tool-result size controls as other tools. Workers cannot mutate todos directly: the Planning Agent reads each result and reconciles the plan with `write_todos`. This is distinct from `hand_off`, which re-routes the entire turn to a single top-level agent rather than fanning out parallel research/build work.

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
| `CLIENT_ENVIRONMENT` | `production` |
| `CLIENT_PROFILE_ROOT` | *(OS-default — `%LOCALAPPDATA%\CodexDesktop` on Windows)* |
| `CLIENT_DEVICE_NAME` | — |
| `CLIENT_SKILLS_ROOTS` | comma-separated absolute paths for local skill scanning |
| `CLIENT_MCP_CONFIG_PATH` | optional explicit path |
| `CLIENT_TOOL_CALL_TIMEOUT_SECONDS` | `60` |
| `CLIENT_HEARTBEAT_INTERVAL_SECONDS` | `30` |
| `CLIENT_RECONNECT_DELAY_SECONDS` / `CLIENT_MAX_RECONNECT_ATTEMPTS` | `5` / `10` |
| `CLIENT_LOG_LEVEL` / `CLIENT_LOG_TO_FILE` | `INFO` / `true` |

---

## Database Migrations

Migrations are Alembic-managed (28 revisions tracked). They are applied **automatically** at application startup via `app.database.migrations.upgrade_database` inside the lifespan hook, so manual migration is only required for dev or out-of-process tooling:

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
codex-client-backend run --config .env.client
# or
python -m client_backend
```

Additional CLI:

```bash
codex-client-backend doctor --config .env.client        # validate configuration
codex-client-backend doctor --config .env.client --json # machine-readable diagnostics
```

### Demo UI

```bash
streamlit run demo.py
```

---

## AI Workflow

The agent workflow is a **LangGraph state machine** defined in [`app/ai/graph.py`](app/ai/graph.py). High-level steps:

1. **Persist current user message** — `MessageService` writes the row to PostgreSQL and reserves the assistant message id. Both ids ride into the workflow so prompt history can exclude the current turn by id (not by tail position) and the final assistant `AIMessage` carries the same id later persisted to the DB.
2. **Hydrate prompt memory** — `ConversationHistoryProvider` (`app/ai/history.py`) returns the durable summary plus recent unsummarized DB messages after the summary cursor. Soft-deleted rows and empty paused/interrupt placeholders are filtered. Per-agent budgets (`chat_history_max_messages` / `_tokens`, …) trim the result.
3. **Router** — [`Router`](app/ai/agents/router.py) invokes Gemini with `ROUTER_SYSTEM_PROMPT` plus server-generated runtime time context and returns one of `chat_agent` / `rag_agent` / `search_agent` / `image_generator_agent` / `planning_agent` / `canvas_agent`. (The legacy `summarize` node is kept as a no-op; `START` connects directly to `route`.)
4. **Agent execution** — the selected agent runs a ReAct-style loop with deferred tool binding, HITL gating, streaming, and the same runtime time context in its system prompt. Configure the local time anchor with `RUNTIME_TIME_CONTEXT_TIMEZONE`; UTC is always included.
5. **Tool execution** — `tool_execution.execute_tool_calls` runs each tool with per-tool timeout, retries, validation, and truncated `ToolMessage` bodies (full artifacts preserved for the UI).
6. **Auto-continue** — on hitting iteration limits, continuation rounds run until user-configured caps (`auto_continue_max_rounds`, `auto_continue_max_total_iterations`, `auto_continue_timeout_seconds`).
7. **Stream** — every token, reasoning chunk, tool call, artifact, and interrupt is serialized as a structured SSE event.
8. **Persist assistant reply, refresh durable summary, compact checkpoint** — once the terminal `complete` event is received, the service persists the assistant message, schedules a non-blocking durable summary refresh, then compacts the checkpoint transcript with `RemoveMessage` so it does not drift from DB truth.

### Agent cheat-sheet

| Agent | Responsibility | Key tools |
|---|---|---|
| `chat_agent` | General chat + tool use | MCP tools, `tool_search`, skills, handoff |
| `rag_agent` | Document-grounded QA with citation verification | `search_documents`, optional reranker, agentic RAG phases |
| `search_agent` | Web/news answers | Tavily, time-context helpers |
| `image_generator_agent` | Gemini image generation | Aspect-ratio / count controls |
| `planning_agent` | Creates / edits task plans | `write_todos`, plan tools |
| `canvas_agent` | Produces canvas/artifact replies | Custom canvas writers |

### Hand-off & delegation

`hand_off_tool` supports inter-agent delegation with `MAX_DELEGATION_DEPTH` safety to prevent loops.

### Deferred tool search

When `MCP_TOOL_SEARCH_ENABLED=true`, only the lightweight [`tool_search`](app/ai/tool_search_tool.py) tool and `MCP_TOOL_SEARCH_PINNED_TOOLS` are bound at start. The agent discovers further tools semantically, with scoring in [`tool_search_scoring.py`](app/ai/tool_search_scoring.py), automatic autoload of the top-`N` above a stricter relevance threshold, TTL eviction, and a per-conversation loaded-tools cap.

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

After upgrading from the older direct-Qdrant index format, run the reindex utility so existing Qdrant points reference SQL chunks:

```bash
python scripts/reindex_documents.py --dry-run
python scripts/reindex_documents.py --all --continue-on-error
```

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
| `tavily_server.py` | Web search adapter |
| `widgets_server.py` | Emits interactive widget state + mints tokens |
| `form_filler_server.py` | Structured-form population |
| `boring_servers/boring_reader_server.py` | Local PDF/image/OCR exploration (ships its own Tesseract + YOLO artifacts under `boring_servers/`) |

The same endpoints are exposed by `client_backend` at `/mcp/*` so a desktop UI can configure MCP both globally (server) and per-device (client).

### Deferred tool binding

`DeferredToolBinding` + `DeferredToolState` ([`app/ai/deferred_tool_*.py`](app/ai/)) record discovery decisions per conversation, enforce TTL, and survive turn boundaries through the LangGraph checkpoint.

### Tool result rendering contract

The backend preserves rich tool render metadata in two places:

- Persisted assistant message metadata: `messageMetadata.tool_artifacts[].render`
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

Skills are markdown files with YAML frontmatter describing a capability (name, description, allowed-tools, optional arguments). They are loaded by:

- **Server** — [`SkillsRegistry`](app/ai/skills_registry.py) (+ [`skill_resolver.py`](app/ai/skill_resolver.py), [`skills_snapshot.py`](app/ai/skills_snapshot.py))
- **Client** — [`LocalSkillsRegistry`](client_backend/services/local_skills_registry.py), scanning `CLIENT_SKILLS_ROOTS`

Both registries share frontmatter parsing in [`shared/skills/front_matter.py`](shared/skills/front_matter.py). Bundled examples:

- [`skills/playwright-cli/`](skills/playwright-cli/) — Playwright CLI browser automation
- [`skills/take100/`](skills/take100/) — HTTP-based timesheet automation

API:

- Server: `/skills/*` (list / detail / toggle / reload)
- Client: `/skills`, `/skills/{name}`, `/skills/{name}/toggle`, `/skills/reload`

When a skill is resolved to a tool (`skills_tool.py`), the agent's skill summaries are injected into its system prompt so it knows *what* is available without paying the schema cost for every skill.

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

---

## Human-in-the-Loop (HITL)

HITL is global (`ENABLE_HUMAN_IN_THE_LOOP=true`) with a per-tool opt-in list (`HITL_TOOLS_REQUIRE_APPROVAL`). When the agent attempts an approvable tool:

1. Execution suspends via `langgraph.types.interrupt(...)`.
2. A `ToolApproval` record is persisted and the message response returns with HTTP **202** plus an interrupt payload.
3. The UI presents the payload to the operator, collects a decision (`approve` / `reject` / edit), and calls `POST /messages/resume-interrupt` (or the equivalent AI-SDK route) with the signed decision.
4. The graph resumes, applies the decision to pending tool calls, and continues streaming.

Timeout handling is Redis-backed; after `HITL_APPROVAL_TIMEOUT_MINUTES` the interrupt is auto-rejected or cleaned up.

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

---

## Live Widgets

Widgets are interactive UI elements rendered by the frontend but driven by the agent.

1. The agent triggers a widget by calling the `widgets` MCP server.
2. The server mints a short-lived widget token via `POST /widgets/{widget_id}/connection`.
3. The frontend opens a stateful WebSocket at `/widgets/{widget_id}/connect`.
4. The server streams widget state, collects user input, and emits terminal events.

Widget state is Redis-backed (see startup banner `"Widget runtime: Redis-backed storage active"`). Without Redis, widget flows degrade; a warning is logged at startup.

### Meaningful Widgets contract

Live widgets read like article-quality inline visuals. The shared state envelope is identical between the AI SDK frontend path and the Streamlit `demo.py` path; agents create widgets through the `widgets` MCP server using a unified shape:

- `presentation` — `title`, `caption`, `x_label`, `y_label`, `unit`, `x_kind` (`time`/`ordered`/`sequence`), and `annotations[]` for callouts. Rendered inline near the chart body, not as a heavy header.
- `controls` + `control_values` — local interaction (sliders, segmented controls, filters, chart-type pickers) that only update widget state.
- `views` / `variants` — pre-computed scenario payloads keyed by control values.
- `actions[]` — assistant-triggering controls. Each action of `type: "assistant_message"` declares a `message_template` that can reference `{{control_values.<key>}}`, `{{input_values.<key>}}`, `{{state.<path>}}`, or `{{presentation.<key>}}`.

Quality enforcement happens at widget-tool time (`app/services/widget_quality.py`). Objective failures — empty chart labels, line/area charts without an ordered `x_kind`, donut charts with negative values, html widgets with empty content or out-of-range height — block the create/update. Soft guidance is returned in the response as `quality_guidance: [...]` so the model can iterate.

Action resolution endpoint: `POST /widgets/{widget_id}/actions/{action_key}` applies an optional `state_patch`, renders the action template, records `last_action` on widget state, and returns `{widget_id, session_id, action_key, content}`. Frontends submit the returned `content` through the normal chat stream — the endpoint does not invoke the assistant directly.

Test commands:

```bash
pytest tests/test_widget_quality.py tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_widget_actions_api.py
pytest tests/test_demo_meaningful_widgets.py tests/test_demo_plan_widget.py tests/test_demo_rich_response.py
pytest tests/client_backend/test_widget_action_proxy.py
```

See [`plans/live-widgets-frontend-integration.md`](plans/live-widgets-frontend-integration.md) § 11 for the full AI SDK renderer contract and reference snippets.

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
- `/mcp/*` — local MCP lifecycle
- `/auth/restore`, `/auth/session`, `/auth/verify-local-token` — local-session wrappers
- `/status`, `/device` — local-runtime state

---

## Streaming, SSE & WebSocket Endpoints

| Transport | Endpoint | Notes |
|---|---|---|
| SSE | `POST /messages/stream` | `token`, `reasoning`, `tool_call`, `tool_result`, `interrupt`, `heartbeat`, `complete`, `error`, `rich_items` |
| SSE | `POST /messages/resume-interrupt` | Same event vocabulary; resumes a suspended graph |
| SSE | `POST /api/chat/{conversation_id}` | Vercel AI SDK wire format (`text`, `tool-call`, `tool-result`, `finish`, `error`, optional `data-rich-items`) |
| SSE | `POST /ai/chat/{conversation_id}` | As above |
| SSE | `POST /ai/resume-interrupt` | As above |
| WS  | `/device-runtime/{device_id}/connect` | Tool dispatch + results |
| WS  | `/widgets/{widget_id}/connect` | Widget state streaming |

Heartbeat interval for SSE: **1 s**. `SUPPRESS_INTERNAL_STREAM_CHUNKS=true` drops internal events (e.g. summarisation output) before they reach clients.

---

## Inline Rich Response (v1)

The backend supports an **opt-in inline-rich-response contract** that lets agents place selected images, live widgets, tool renders, canvas artifacts, citations, and resource links at specific positions inside the markdown answer. Disabled by default — set `INLINE_RICH_RESPONSE_ENABLED=true` and have the client advertise the per-request capability to receive marker-bearing content.

### Marker syntax

Rich items are placed using a standalone block-level HTML comment:

```markdown
The pressure differential causes lift over the upper wing surface.

<!--rich:image:tool:call_7:0-->

*Figure 1. Streamlines around an airfoil.*

This pattern helps explain why the pressure is lower above the wing.
```

Marker rules:

- Must appear on its own line, optionally with up to three leading spaces and trailing whitespace.
- `<id>` may contain ASCII letters, digits, `_`, `-`, `.`, and `:` and is at most 128 characters.
- Markers inside fenced (` ``` ` or `~~~`) or indented (4-space) code blocks are treated as literal markdown.
- Unknown ids produce a neutral unavailable-content block plus a validation warning in `messageMetadata.rich_reference_warnings`.

### Stable item IDs

| Origin | ID format |
|---|---|
| Tool result render | `tool:<tool_call_id>` |
| Image from a tool result | `image:tool:<tool_call_id>:<zero_based_index>` |
| RAG document image | `image:document:<document_image_id>` |
| Generated image | `image:generated:<assistant_message_id>:<zero_based_index>` |
| Live widget | `widget:<widget_id>` |
| Canvas artifact | `canvas:<assistant_message_id>` |
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
      "payload": {"url": "https://example.org/airfoil.png", "mime_type": "image/png"}
    }
  ],
  "rich_reference_warnings": []
}
```

Item types: `image`, `live_widget`, `tool_render`, `canvas_artifact`, `citation`, `resource_link`. Payloads are type-validated through a Pydantic discriminated union with `extra="forbid"`; only renderer-consumed fields are accepted. Image payloads accept exactly one of `url` (https / dev-only http) or `data` (base64 of an allowed raster MIME).

### Display policy

- `inline_only` — render only at its marker. Image items use this policy and are **never** appended as a gallery. Unreferenced image candidates are dropped entirely.
- `inline_or_append` — render inline when referenced, otherwise append below the body. Used by `live_widget`, `tool_render`, `canvas_artifact`, `resource_link`.

For capable responses, widget placement is authored dynamically in the response body: a `<!--rich:widget:<widget_id>-->` marker selects its position. The server does not insert a marker when the response omits one.

For v1 messages, the legacy `metadata["images"]` gallery is suppressed in the Streamlit renderer and only selected images surface as AI SDK `file` parts. Pre-v1 messages (no `rich_items_version`) retain their existing gallery.

### Capability negotiation

Clients opt in per-request by setting `inline_rich_response_v1: true` (camelCase or snake_case) on:

- `POST /messages` and `POST /messages/stream` body (`MessageCreate.inline_rich_response_v1`).
- `POST /ai/chat/{conversation_id}` body extras (`inlineRichResponseV1` / `inline_rich_response_v1`).
- `POST /messages/resume-interrupt` and `POST /ai/resume-interrupt` body (`InterruptResumeRequest.inline_rich_response_v1`).

Capability is then ANDed with `INLINE_RICH_RESPONSE_ENABLED` server-side. Non-capable AI SDK reads of a persisted v1 message receive a legacy projection: standalone marker lines are stripped from `content`, and `rich_items` / `rich_items_version` / `rich_reference_warnings` are removed from metadata.

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

Image candidates are **never** streamed transiently — they only surface in the final `data-assistant-message.data.message.messageMetadata.rich_items` after marker selection. Canvas source is excluded from transient upserts. Transient data parts ride in `useChat({ onData })`, not in `message.parts`. The AI SDK response retains the `x-vercel-ai-ui-message-stream: v1` header.

### Client renderer algorithm

1. Opt in with the per-request capability flag.
2. In `useChat({ onData })`, maintain a map of safe non-image `rich_items` from transient `data-rich-items` upserts.
3. Accumulate `text-delta` content normally.
4. Split the accumulated text only on standalone complete markers into blocks.
5. Render a known typed item at that block position; while a streamed marker is waiting for its widget upsert, render a lightweight inline placeholder there.
6. On final `data-assistant-message`, replace transient registry data with persisted `messageMetadata.rich_items` and final content (authoritative).
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

Pre-built artifacts are kept under [`dist/client-backend-bundle/`](dist/client-backend-bundle/). `pyproject.toml` defines the console script:

```toml
[project.scripts]
codex-client-backend = "client_backend.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["app", "client_backend"]
```

Wheels can be built with `python -m build`.

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
| Reranker download slow | First run fetches the HF model. Pin `RERANKER_MODEL` or disable with `ENABLE_RERANKING=false`. |
| Client-device tool calls fail | Device offline or `CLIENT_RUNTIME_REQUIRE_CONNECTED_DEVICE_FOR_LOCAL_TOOLS=true`. Inspect `GET /device-runtime/connected-devices`. |

---

## License

Internal / unspecified. Add a `LICENSE` file before distribution.
