# Sample Chatbot

Backend for a multi-agent chatbot (FastAPI + LangGraph) with PostgreSQL persistence, optional Qdrant RAG, and Celery workers for document processing.

## Project Structure

```
app/
  ai/                 # LangGraph agent graph, prompts, HITL, memory
  api/                # FastAPI route handlers
  core/               # Settings, DI container, auth dependencies, security utils
  database/           # DB session/engine wiring
  models/             # SQLAlchemy ORM models
  repositories/       # Data access layer (Repository pattern)
  schemas/            # Pydantic request/response models
  services/           # Business logic (AI, messages, documents, auth, etc.)
  workers/            # Celery background tasks
  main.py             # FastAPI application entry point
```

## Quickstart

### 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

For the demo UI:

```bash
pip install -r demo_requirements.txt
streamlit run demo.py
```

### 2. External Services

#### PostgreSQL

```sql
CREATE DATABASE chatbot;
\c chatbot;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

#### Qdrant (optional, for RAG)

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

#### Redis (optional, for Celery + HITL timeouts)

```bash
docker run -d -p 6379:6379 redis
```

### 3. External Services

#### PostgreSQL

```sql
CREATE DATABASE chatbot;
\c chatbot;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

#### Qdrant (optional, for RAG)

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

#### Redis (optional, for Celery + HITL timeouts)

```bash
docker run -d -p 6379:6379 redis
```

### 4. DB migrations

```bash
alembic upgrade head
```

### 5. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Celery worker (for document processing):

```bash
python -m app.workers.start_worker
```

### Access Points

- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Authentication

Authenticated endpoints expect a JWT access token via `Authorization: Bearer <token>`.

- `POST /auth/signup`
- `POST /auth/login`
- `POST /auth/refresh` (expects the **refresh** token in `Authorization` header)
- `POST /auth/logout`

---

## API Endpoints

### Health

- `GET /health` (basic status)
- `GET /health/celery`
- `GET /health/redis`
- `GET /health/qdrant`
- `GET /health/all`

### Users

- `GET /users/{user_id}`

### Conversations (require authentication)

- `POST /conversations/`
- `GET /conversations/{conversation_id}`
- `GET /conversations/` (paginated: `page`, `limit`)
- `GET /conversations/{conversation_id}/messages` (paginated: `page`, `limit`)
- `PATCH /conversations/{conversation_id}`
- `DELETE /conversations/{conversation_id}`

### Messages (require authentication)

- `POST /messages/`
- `POST /messages/stream` (SSE of internal events)
- `POST /messages/resume-interrupt` (resume HITL approvals)
- `GET /messages/{message_id}`
- `GET /messages/` (paginated: `page`, `limit`)

### AI Chat (assistant-ui / Vercel AI SDK)

UI Message Stream (SSE) compatible with `@ai-sdk/react` / assistant-ui defaults:

- `POST /api/chat` (alias: `POST /ai/chat`)
  - Requires `Authorization: Bearer <access_token>`
  - URL: `/api/chat/{conversation_id}`
  - Body: `{ "messages": [...] }`
  - Response includes `x-vercel-ai-ui-message-stream: v1`
  - Emits custom data parts:
    - `data-user-message` (DB-persisted user message)
    - `data-assistant-message` (DB-persisted assistant message)
    - `data-agent-selected` (which agent answered)
    - `data-interrupt` (HITL approval needed)

### Documents (require authentication)

- `POST /documents/upload` (multipart: `file`, `conversation_id`)
- `GET /documents/task/{task_id}`
- `GET /documents/{document_id}`
- `GET /documents/conversation/{conversation_id}` (paginated: `page`, `page_size`)
- `PUT /documents/{document_id}`
- `DELETE /documents/{document_id}`

### Feedbacks

- `POST /messages/{message_id}/feedbacks` (auth)
- `GET /messages/{message_id}/feedbacks/user` (auth)
- `PUT /messages/{message_id}/feedbacks/{feedback_id}` (auth)
- `GET /messages/{message_id}/feedbacks` (public)
- `GET /messages/{message_id}/feedbacks/stats` (public)
