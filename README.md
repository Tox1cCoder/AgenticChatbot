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

Create a `.env` file in `app/core/` (or edit the existing one) and set:

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

### Feedback

- `POST /feedback/user/{user_id}` — Create/update feedback for a message
- `GET /feedback/{feedback_id}` — Get feedback by ID
- `GET /feedback/message/{message_id}` — Get all feedback for a message (paginated)
- `GET /feedback/user/{user_id}` — Get all feedback by a user (paginated)
- `GET /feedback/message/{message_id}/user/{user_id}` — Get user's feedback for a message
- `GET /feedback/message/{message_id}/stats` — Get rating stats for a message
- `PUT /feedback/{feedback_id}` — Update feedback (requires user_id)

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
   "avatar_url": "https://example.com/avatar.jpg"
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
