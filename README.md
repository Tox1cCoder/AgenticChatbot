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
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On Linux/Mac
pip install -e .  # Installs dependencies from pyproject.toml
```

For the demo UI:

```bash
pip install -r demo_requirements.txt
streamlit run demo.py
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/chatbot

# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true

# AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional, leave empty for local instance
QDRANT_COLLECTION_NAME=chatbot_documents

# Celery & Redis (Background Tasks)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# JWT Authentication
SECRET_KEY=your-secret-key-min-32-characters-long
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS (Optional)
CORS_ORIGINS=["http://localhost:3000","http://localhost:8501"]

# Environment
ENVIRONMENT=development
```

See `app/core/config.py` for all available configuration options.

### 3. External Services Setup

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

**On Windows:**
- Download from [redis.io](https://redis.io/download) or use [Memurai](https://www.memurai.com/)
- Or use Docker: `docker run -d -p 6379:6379 redis`

**On Linux:**
```bash
sudo apt-get install redis-server
sudo service redis-server start
```

### 4. Database Migrations

```bash
alembic upgrade head
```

### 5. Start Services

#### Terminal 1: Start API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2: Start Celery Worker (for document processing)
```bash
celery -A app.workers.celery_app worker --loglevel=info --pool=solo
```

Note: Use `--pool=solo` on Windows. On Linux/Mac, you can omit this flag.

#### Terminal 3: Start Streamlit Demo (Optional)
```bash
streamlit run demo.py
```


### 🔗 Access Points

- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **Streamlit Demo**: [http://localhost:8501](http://localhost:8501)

---

## 📡 API Endpoints

### Health & Status

- `GET /health/` — Health check
- `GET /health/db` — Database health check

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
- `PUT /conversations/{conversation_id}` — Update conversation (user must own conversation)
- `DELETE /conversations/{conversation_id}` — Delete conversation (user must own conversation)

### Messages

- `POST /messages/` — Create message (auto bot reply if role is 'user')
- `GET /messages/{message_id}` — Get message by ID
- `GET /messages/conversation/{conversation_id}` — Get messages for a conversation (requires user_id, paginated)
- `GET /messages/conversation/{conversation_id}/thread` — Get conversation thread (requires user_id)

### Documents (RAG Knowledge Base)

- `POST /documents/upload` — Upload document for processing (PDF, DOCX, etc.)
- `GET /documents/` — List all documents (paginated)
- `GET /documents/{document_id}` — Get document details
- `GET /documents/{document_id}/status` — Check processing status
- `PUT /documents/{document_id}` — Update document metadata
- `DELETE /documents/{document_id}` — Delete document and embeddings

### Feedbacks

- `POST /messages/{message_id}/feedbacks` — Create feedback for a message
- `GET /messages/{message_id}/feedbacks/{feedback_id}` — Get specific feedback for a message
- `GET /messages/{message_id}/feedbacks/user/{user_id}` — Get user's feedback for a message
- `GET /messages/{message_id}/feedbacks/stats` — Get rating stats for a message
- `PUT /messages/{message_id}/feedbacks/{feedback_id}` — Update feedback (requires user ownership)

---

## 💡 Example API Usage

### Authentication Flow

#### 1. Register New User

```http
POST /auth/signup
Content-Type: application/json

{
  "username": "testuser",
  "email": "test@example.com",
  "password": "secure123",
  "fullName": "Test User"
}
```

#### 2. Login

```http
POST /auth/login
Content-Type: application/json

{
  "email": "test@example.com",
  "password": "secure123"
}
```

**Response:**
```json
{
  "accessToken": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refreshToken": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "tokenType": "bearer",
  "userId": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Document Upload & RAG

#### Upload Document

```http
POST /documents/upload
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

file: <your-pdf-or-docx-file>
userId: <user-id>
```

**Response:**
```json
{
  "id": "doc-uuid",
  "filename": "research.pdf",
  "status": "pending",
  "uploadedAt": "2024-01-01T12:00:00Z"
}
```

The document will be processed asynchronously by Celery workers:
1. Extract text content
2. Split into chunks
3. Generate embeddings
4. Store in Qdrant vector database

#### Check Processing Status

```http
GET /documents/{document_id}/status
Authorization: Bearer <access_token>
```

### Chat with RAG

```http
POST /messages/
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "conversationId": "<conversation-uuid>",
  "content": "What does the uploaded document say about AI?",
  "role": "user"
}
```

The AI will automatically:
- Retrieve relevant document chunks from Qdrant
- Use LangGraph agent to generate context-aware responses
- Return answer based on uploaded documents

### Create User (Legacy)

```http
POST /users/
{
   "username": "testuser",
   "email": "test@example.com",
   "password": "secure123",
   "full_name": "Test User",
}
```

### Create Conversation

```http
POST /conversations/
{
   "title": "My First Chat"
}
```

### Send Message

```http
POST /messages/
{
   "conversation_id": "<uuid>",
   "content": "Hello, how are you?",
   "role": "user"
}
```

### Threaded Reply

```http
POST /messages/
{
   "conversation_id": "<uuid>",
   "content": "Reply to previous",
   "role": "user",
   "parent_message_id": "<uuid>"
}
```

### Rate Message

```http
POST /feedback/user/{user_id}
{
   "message_id": "<uuid>",
   "rating": 5,
   "comment": "Great!"
}
```

---

## Database Schema

**User**

- id: UUID (PK)
- username: VARCHAR(50), unique, required
- email: VARCHAR(255), unique, required
- password_hash: TEXT, required
- avatar_url: VARCHAR(2048), optional

**Conversation**

- id: UUID (PK)
- user_id: UUID (FK to user.id), required
- title: VARCHAR(255), required

**Message**

- id: UUID (PK)
- conversation_id: UUID (FK to conversation.id), required
- role: ENUM (user/assistant/system), required
- content: TEXT, required
- parent_message_id: UUID (FK to message.id), optional (for threading)
- Index: (conversation_id, created_at)

**Document** (New)

- id: UUID (PK)
- user_id: UUID (FK to user.id), required
- filename: VARCHAR(255), required
- file_path: VARCHAR(512), optional
- status: ENUM (pending/processing/completed/failed), required
- chunks_count: INTEGER, optional
- error_message: TEXT, optional
- Index: (user_id, status)

**Feedback**

- id: UUID (PK)
- message_id: UUID (FK to message.id), required, unique, indexed
- user_id: UUID (FK to user.id), required, indexed
- rating: SMALLINT (1-5), required
- comment: TEXT, optional
- Index: (message_id, user_id)

---

## 🚀 API Testing with Postman

This section provides a comprehensive workflow for testing all API endpoints using Postman. Follow these steps to demo the complete chatbot functionality.

### Prerequisites

1. **Install Postman**: Download from [postman.com](https://www.postman.com/)
2. **Start the API**: Run `uvicorn app.main:app --reload`
3. **Base URL**: `http://localhost:8000`
4. **Database**: Ensure PostgreSQL is running with proper configuration

### Postman Collection Setup

Create a new Postman collection called "Sample Chatbot API" and add the following environment variables:

```json
{
  "baseUrl": "http://localhost:8000",
  "accessToken": "",
  "refreshToken": "",
  "userId": "",
  "conversationId": "",
  "messageId": ""
}
```

### 1. Authentication Flow

#### 1.1 User Registration (Signup)

```http
POST {{baseUrl}}/auth/signup
Content-Type: application/json

{
  "username": "testuser",
  "email": "test@example.com",
  "fullName": "Test User",
  "password": "password123"
}
```

**Expected Response (201 Created):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "testuser",
  "email": "test@example.com",
  "fullName": "Test User",
  "createdAt": "2024-01-01T12:00:00Z",
  "updatedAt": "2024-01-01T12:00:00Z"
}
```

#### 1.2 User Login

```http
POST {{baseUrl}}/auth/login
Content-Type: application/json

{
  "email": "test@example.com",
  "password": "password123"
}
```

**Expected Response (200 OK):**

```json
{
  "accessToken": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refreshToken": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "tokenType": "bearer",
  "expiresIn": 1800,
  "userId": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Post-Response Script** (Save tokens to environment):

```javascript
const response = pm.response.json();
pm.environment.set("accessToken", response.accessToken);
pm.environment.set("refreshToken", response.refreshToken);
pm.environment.set("userId", response.userId);
```

#### 1.3 Token Refresh

```http
POST {{baseUrl}}/auth/refresh
Authorization: Bearer {{refreshToken}}
```

### 2. User Management

#### 2.1 Get Current User

```http
GET {{baseUrl}}/users/{{userId}}
Authorization: Bearer {{accessToken}}
```

#### 2.2 Get All Users (Admin/Testing)

```http
GET {{baseUrl}}/users?page=1&limit=10
Authorization: Bearer {{accessToken}}
```

### 3. Conversation Management

#### 3.1 Create New Conversation

```http
POST {{baseUrl}}/conversations
Authorization: Bearer {{accessToken}}
Content-Type: application/json

{
  "title": "My First Chat"
}
```

**Post-Response Script** (Save conversation ID):

```javascript
const response = pm.response.json();
pm.environment.set("conversationId", response.id);
```

#### 3.2 Get User's Conversations

```http
GET {{baseUrl}}/conversations?page=1&limit=10
Authorization: Bearer {{accessToken}}
```

#### 3.3 Get Specific Conversation

```http
GET {{baseUrl}}/conversations/{{conversationId}}
Authorization: Bearer {{accessToken}}
```

### 4. Message Flow

#### 4.1 Send User Message

```http
POST {{baseUrl}}/messages
Authorization: Bearer {{accessToken}}
Content-Type: application/json

{
  "conversationId": "{{conversationId}}",
  "content": "Hello! Can you help me with Python programming?"
}
```

**Post-Response Script** (Save message ID):

```javascript
const response = pm.response.json();
pm.environment.set("messageId", response.id);
```

#### 4.2 Get Conversation Messages

```http
GET {{baseUrl}}/messages/conversation/{{conversationId}}?page=1&limit=50
Authorization: Bearer {{accessToken}}
```

#### 4.3 Get Specific Message

```http
GET {{baseUrl}}/messages/{{messageId}}
Authorization: Bearer {{accessToken}}
```

### 5. Feedback System

#### 5.1 Rate Assistant Response

```http
POST {{baseUrl}}/messages/{{messageId}}/feedbacks
Authorization: Bearer {{accessToken}}
Content-Type: application/json

{
  "rating": 5,
  "comment": "Very helpful response!"
}
```

#### 5.2 Get Message Feedback

```http
GET {{baseUrl}}/messages/{{messageId}}/feedbacks/user/{{userId}}
Authorization: Bearer {{accessToken}}
```

#### 5.3 Update Feedback

```http
PUT {{baseUrl}}/messages/{{messageId}}/feedbacks/{{feedbackId}}
Authorization: Bearer {{accessToken}}
Content-Type: application/json

{
  "rating": 4,
  "comment": "Good response, but could be more detailed"
}
```

### 6. Testing Scenarios

#### Scenario 1: Complete Chat Session

1. Register new user
2. Login to get tokens
3. Create conversation
4. Send multiple messages
5. Rate responses
6. View conversation history

#### Scenario 2: Multi-User Chat

1. Register multiple users
2. Create separate conversations
3. Test message isolation
4. Verify access controls

#### Scenario 3: Error Handling

1. Test invalid authentication
2. Test access to unauthorized resources
3. Test malformed requests
4. Test rate limiting (if implemented)

### 7. Validation Tests

#### Authentication Errors

- Login with wrong password → 401 Unauthorized
- Access protected route without token → 401 Unauthorized
- Use expired token → 401 Unauthorized

#### Authorization Errors

- Access another user's conversation → 403 Forbidden
- Modify another user's message → 403 Forbidden

#### Validation Errors

- Send empty message → 422 Unprocessable Entity
- Invalid email format → 422 Unprocessable Entity
- Missing required fields → 422 Unprocessable Entity

### 8. Health Check

#### API Health Status

```http
GET {{baseUrl}}/health
```

**Expected Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

### 9. Advanced Testing

#### Performance Testing

- Create multiple concurrent conversations
- Send rapid message sequences
- Test with large message content

#### Data Integrity

- Verify conversation ownership
- Check message ordering
- Validate feedback associations

### 10. Cleanup Operations

#### Delete Test Data

```http
DELETE {{baseUrl}}/conversations/{{conversationId}}
Authorization: Bearer {{accessToken}}
```

### Expected Response Formats

All successful API responses return **camelCase** JSON:

- ✅ `userId`, `accessToken`, `createdAt`
- ❌ `user_id`, `access_token`, `created_at`

All timestamps are in ISO 8601 format (UTC).

### Notes for Testing

1. **Authentication Required**: Most endpoints require valid JWT token
2. **Rate Limiting**: Some endpoints may have rate limits
3. **Data Validation**: All inputs are validated according to Pydantic schemas
4. **Error Responses**: Consistent error format with status codes
5. **CORS**: Enabled for frontend integration

This comprehensive testing workflow ensures all chatbot functionality works correctly and demonstrates the complete user journey from registration to conversation management.

---

## 🛠️ Development

### Project Dependencies

Install all dependencies:
```bash
pip install -e .
```

Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Code Structure Principles

- **Repository Pattern**: Data access abstraction
- **Dependency Injection**: Using `dependency-injector`
- **Service Layer**: Business logic separation
- **Clean Architecture**: Separation of concerns
- **Type Hints**: Full Python type annotations
- **Async/Await**: Async operations throughout

### Database Migrations

Create a new migration:
```bash
alembic revision --autogenerate -m "Description of changes"
```

Apply migrations:
```bash
alembic upgrade head
```

Rollback migration:
```bash
alembic downgrade -1
```

### Running Tests (if configured)

```bash
pytest
pytest -v  # Verbose
pytest --cov=app  # With coverage
```

---

## 🐛 Troubleshooting

### Common Issues

**Celery Worker Not Starting**
- Ensure Redis is running: `redis-cli ping` should return `PONG`
- On Windows, use `--pool=solo` flag
- Check `CELERY_BROKER_URL` in `.env`

**Qdrant Connection Failed**
- Verify Qdrant is running: Visit `http://localhost:6333/dashboard`
- Check `QDRANT_URL` in `.env`
- Docker: `docker ps` should show qdrant container

**Database Connection Error**
- Verify PostgreSQL is running
- Check `DATABASE_URL` format: `postgresql://user:pass@host:port/dbname`
- Ensure database exists and UUID extension is enabled

**Document Processing Stuck**
- Check Celery worker logs
- Verify document format is supported (PDF, DOCX, TXT)
- Check file size limits
- Inspect document status: `GET /documents/{id}/status`

**JWT Token Expired**
- Use refresh token: `POST /auth/refresh`
- Check token expiration settings in `.env`

**Import Errors**
- Reinstall package: `pip install -e .`
- Check Python version: `python --version` (should be >=3.10)

### Logs & Debugging

**API Server Logs**: Check terminal running `uvicorn`
**Celery Worker Logs**: Check terminal running celery worker
**Redis Logs**: `redis-cli monitor`
**Database Queries**: Set `API_DEBUG=true` in `.env` for SQL logging

---

## 📦 Deployment

### Environment Variables for Production

```env
ENVIRONMENT=production
API_DEBUG=false
SECRET_KEY=<strong-random-secret-key>
DATABASE_URL=<production-database-url>
QDRANT_URL=<production-qdrant-url>
CELERY_BROKER_URL=<production-redis-url>
```

### Docker Deployment (Example)

```dockerfile
# Example Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
