# Sample Chatbot

## Project Structure

```
app/
├── api/           # FastAPI routes
├── core/          # Config, security, DI
├── database/      # DB session, base, connection
├── factories/     # Test/data factories
├── models/        # SQLAlchemy models
├── repositories/  # Data access layer
├── schemas/       # Pydantic schemas
├── services/      # Business logic
├── main.py        # FastAPI app entrypoint
```

## Quickstart

### 1. Clone & Setup

```bash
git clone https://tk-itteam.backlog.com/git/AI202508/ai_training.git
git checkout Thai-Postgre-FastAPI
cd <repo-folder>
python -m venv .venv
.venv\Scripts\activate
pip install -e .  # Installs dependencies from pyproject.toml
```

For the demo UI:

```bash
pip install -r demo_requirements.txt
streamlit run demo.py
```

### 2. Environment Variables

Create a `.env` file in the project root directory (or edit the existing one) and set:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/chatbot
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true
GEMINI_API_KEY=your_gemini_api_key_here
```

Other optional variables (see `app/core/config.py`):

- SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS, CORS_ORIGINS, ENVIRONMENT

### 3. Database Setup

Create DB and enable UUID extension:

```sql
CREATE DATABASE chatbot;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### 4. Migrations

```bash
alembic upgrade head
```

### 5. Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
Health: [http://localhost:8000/health](http://localhost:8000/health)

---

## Demo UI

The demo UI uses Streamlit. To run it:

```bash
pip install -r demo_requirements.txt
streamlit run demo.py
```

Demo: [http://localhost:8501](http://localhost:8501)

---

## API Endpoints

### Health

- `GET /health/` — Health check
- `GET /health/db` — Database health check

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

### Feedbacks

- `POST /messages/{message_id}/feedbacks` — Create feedback for a message
- `GET /messages/{message_id}/feedbacks/{feedback_id}` — Get specific feedback for a message
- `GET /messages/{message_id}/feedbacks/user/{user_id}` — Get user's feedback for a message
- `GET /messages/{message_id}/feedbacks/stats` — Get rating stats for a message
- `PUT /messages/{message_id}/feedbacks/{feedback_id}` — Update feedback (requires user ownership)

---

## Example API Usage

### Create User

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
- sender: ENUM (user/assistant/system), required
- content: TEXT, required
- Index: (conversation_id, created_at)

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
