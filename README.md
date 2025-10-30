# Sample Chatbot

## Project Structure

```
app/
├── ai/                # LangGraph agent, prompts, memory
│   ├── agents/        # RAG agent implementations
│   ├── graph.py       # LangGraph workflow
│   ├── memory.py      # Conversation memory
│   ├── prompts.py     # AI prompt templates
│   └── schemas.py     # AI-related schemas
├── api/               # FastAPI route handlers
│   ├── auth.py        # Authentication endpoints
│   ├── conversations.py
│   ├── messages.py
│   ├── documents.py   # Document upload endpoints
│   ├── feedback.py
│   └── users.py
├── core/              # Configuration, security, DI
│   ├── config.py      # Settings & environment
│   ├── container.py   # Dependency injection
│   ├── auth.py        # Auth dependencies
│   └── security/      # Password hashing, tokens
├── database/          # Database layer
│   ├── session.py     # DB session management
│   ├── base.py        # Base model
│   └── qdrant/        # Vector DB connection
├── models/            # SQLAlchemy ORM models
│   ├── user.py
│   ├── conversation.py
│   ├── message.py
│   ├── document.py
│   └── feedback.py
├── repositories/      # Data access layer (Repository pattern)
├── schemas/           # Pydantic models (request/response)
├── services/          # Business logic
│   ├── ai_service.py
│   ├── auth_service.py
│   ├── document_service.py
│   ├── document_processing_service.py
│   └── ...
├── workers/           # Celery background tasks
│   ├── celery_app.py  # Celery configuration
│   ├── document_processor.py  # Document processing tasks
│   └── cleanup_tasks.py       # Maintenance tasks
├── main.py            # FastAPI application entry point
└── utils/             # Helpers and utilities
```

## Quickstart

### 1. Clone & Setup

```bash
git clone <repository-url>
cd sample-chatbot
python -m venv .venv
.venv\Scripts\activate
pip install -e .  # Installs dependencies from pyproject.toml
```

For the demo UI:

```bash
pip install -r demo_requirements.txt
streamlit run demo.py
```

### 2. External Services Setup

#### PostgreSQL Database

```sql
CREATE DATABASE chatbot;
\c chatbot;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

#### Qdrant Vector Database

**Option 1: Docker (Recommended)**

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Option 2: Cloud**
Sign up at [cloud.qdrant.io](https://cloud.qdrant.io) and update `QDRANT_URL` and `QDRANT_API_KEY` in `.env`

#### Redis Server

- Download from [redis.io](https://redis.io/download) or use [Memurai](https://www.memurai.com/)
- Or use Docker: `docker run -d -p 6379:6379 redis`

### 3. Database Migrations

```bash
alembic upgrade head
```

### 4. Start Services

#### Terminal 1: Start API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2: Start Celery Worker (for document processing)

```bash
python -m app.workers.start_worker
```

Or directly with Celery:

```bash
celery -A app.workers.celery_app worker --loglevel=info --pool=solo
```

#### Terminal 3: Start Streamlit Demo

```bash
streamlit run demo.py
```

### Access Points

- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **Streamlit Demo**: [http://localhost:8501](http://localhost:8501)

---

## API Endpoints

### Health & Status

- `GET /health/` — Health check
- `GET /health/db` — Database health check
- `GET /health/celery` — Check Celery worker status
- `GET /health/redis` — Check Redis connection
- `GET /health/qdrant` — Check Qdrant connection
- `GET /health/all` — Check all services at once

### Authentication

- `POST /auth/signup` — Register new user
- `POST /auth/login` — Login and get JWT tokens
- `POST /auth/refresh` — Refresh access token

### Users

- `POST /users/` — Create user
- `GET /users/{user_id}` — Get user by ID
- `GET /users/` — List users (paginated, requires authentication)

### Conversations (all require authentication)

- `POST /conversations/` — Create conversation for current user
- `GET /conversations/{conversation_id}` — Get conversation by ID
- `GET /conversations/` — List current user's conversations (paginated: `page`, `limit`)
- `PATCH /conversations/{conversation_id}` — Update conversation (user must own conversation)
- `DELETE /conversations/{conversation_id}` — Delete conversation (user must own conversation)

### Messages

- `POST /messages/` — Create message (auto bot reply if role is 'user')
- `GET /messages/{message_id}` — Get message by ID
- `GET /messages/conversation/{conversation_id}` — Get messages for a conversation (requires user_id, paginated)
- `GET /messages/conversation/{conversation_id}/thread` — Get conversation thread (requires user_id)

### Documents (RAG Knowledge Base - all require authentication)

Upload Document

- `POST /documents/upload` — Upload document for processing
  - Supported formats: PDF, DOCX, TXT
  - Max size: 200MB (configurable)
  - Requires: conversation ownership
  - Returns: document metadata + task_id for status tracking
  - Events: Emits UPLOAD_STARTED event

Get Task Status

- `GET /documents/task/{task_id}` — Get Celery task status
  - Returns: task state (PENDING, STARTED, SUCCESS, FAILURE)
  - Use for polling upload progress

Get Document

- `GET /documents/{document_id}` — Get document details
  - Requires: conversation ownership
  - Returns: document metadata with processing status

List Documents

- `GET /documents/conversation/{conversation_id}` — Get documents for conversation
  - Requires: conversation ownership
  - Supports pagination: `page`, `page_size`

Update Document

- `PUT /documents/{document_id}` — Update document metadata
  - Requires: conversation ownership

Delete Document

- `DELETE /documents/{document_id}` — Delete document and embeddings
  - Requires: conversation ownership
  - Removes from both database and Qdrant vector store
  - Events: Emits DELETED event

---

### Feedbacks

- `POST /messages/{message_id}/feedbacks` — Create feedback for a message
- `GET /messages/{message_id}/feedbacks/{feedback_id}` — Get specific feedback for a message
- `GET /messages/{message_id}/feedbacks/user/{user_id}` — Get user's feedback for a message
- `GET /messages/{message_id}/feedbacks/stats` — Get rating stats for a message
- `PUT /messages/{message_id}/feedbacks/{feedback_id}` — Update feedback (requires user ownership)
