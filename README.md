# Sample Chatbot - FastAPI & PostgreSQL

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Layer     │    │  Service Layer  │    │ Repository Layer│    │  Database Layer │
│                 │    │                 │    │                 │    │                 │
│ FastAPI Routes  │───▶│ Business Logic  │───▶│  Data Access   │───▶│   PostgreSQL   │
│ Request/Response│    │ Validation      │    │  CRUD Operations│    │   SQLAlchemy    │
│ Pydantic Schemas│    │ Domain Rules    │    │  Query Building │    │   Alembic       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://tk-itteam.backlog.com/git/AI202508/ai_training.git
   cd ai-training
   git checkout Thai-Postgre-FastAPI
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -e .
   ```

   Required packages:

   - FastAPI
   - SQLAlchemy 2.0+
   - PostgreSQL driver (psycopg2-binary)
   - Alembic
   - Pydantic v2
   - bcrypt
   - uvicorn

4. **Set up environment variables**

   ```bash
   copy .env.example .env
   ```

   Edit `.env` file with your database configuration:

   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/chatbot
   API_HOST=0.0.0.0
   API_PORT=8000
   API_DEBUG=true
   ```

5. **Set up the database**

   Create a PostgreSQL database and enable UUID extension:

   ```sql
   CREATE DATABASE chatbot;
   \c chatbot;
   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
   ```

   Run database migrations:

   ```bash
   alembic upgrade head
   ```

## Running the Application

1. **Start the development server**

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc
   - Health Check: http://localhost:8000/health

## API Endpoints

### Health Endpoints

- `GET /health/` - Basic health check
- `GET /health/db` - Database health check

### User Management

- `POST /users/` - Create a new user (with password hashing)
- `GET /users/{user_id}` - Get user by UUID
- `GET /users/` - List all users (paginated)
- `GET /users/email/{email}` - Get user by email
- `GET /users/username/{username}` - Get user by username
- `PUT /users/{user_id}` - Update user
- `DELETE /users/{user_id}` - Delete user

### Conversation Management

- `POST /conversations/` - Create a new conversation
- `GET /conversations/{conversation_id}` - Get conversation by UUID
- `GET /conversations/user/{user_id}` - Get user's conversations
- `GET /conversations/{conversation_id}/messages` - Get conversation with messages
- `PUT /conversations/{conversation_id}` - Update conversation
- `DELETE /conversations/{conversation_id}` - Delete conversation

### Message Management

- `POST /messages/` - Create a new message (auto-generates bot response)
- `GET /messages/{message_id}` - Get message by UUID
- `GET /messages/conversation/{conversation_id}` - Get conversation messages
- `GET /messages/conversation/{conversation_id}/thread` - Get threaded conversation
- `GET /messages/{parent_message_id}/replies` - Get message replies
- `PUT /messages/{message_id}` - Update message
- `DELETE /messages/{message_id}` - Delete message

### Feedback Management

- `POST /feedback/` - Create feedback for a message (rating 1-5)
- `GET /feedback/{feedback_id}` - Get feedback by UUID
- `GET /feedback/message/{message_id}` - Get all feedback for a message
- `GET /feedback/user/{user_id}` - Get user's feedback history
- `GET /feedback/message/{message_id}/user/{user_id}` - Get user's feedback for specific message
- `GET /feedback/message/{message_id}/stats` - Get message rating statistics
- `PUT /feedback/{feedback_id}` - Update feedback
- `DELETE /feedback/{feedback_id}` - Delete feedback
- `GET /conversations/{conversation_id}` - Get conversation by ID
- `GET /conversations/user/{user_id}` - Get user's conversations
- `PUT /conversations/{conversation_id}` - Update conversation
- `DELETE /conversations/{conversation_id}` - Delete conversation

### Message Management

- `POST /messages/` - Create a new message (auto-generates bot response)
- `GET /messages/{message_id}` - Get message by ID
- `GET /messages/conversation/{conversation_id}` - Get conversation messages
- `GET /messages/conversation/{conversation_id}/history` - Get conversation history
- `GET /messages/user/{user_id}` - Get user's messages
- `PUT /messages/{message_id}` - Update message
- `DELETE /messages/{message_id}` - Delete message

## Testing with Postman

### 1. Create a User

```http
POST http://localhost:8000/users/
Content-Type: application/json

{
    "username": "testuser",
    "email": "test@example.com",
    "password": "secure123",
    "full_name": "Test User",
    "avatar_url": "https://example.com/avatar.jpg"
}
```

### 2. Create a Conversation

```http
POST http://localhost:8000/conversations/
Content-Type: application/json

{
    "title": "My First Chat"
}
```

Note: You'll need to pass the user_id as a query parameter or include it in the request context.

### 3. Send a Message (Triggers Bot Response)

```http
POST http://localhost:8000/messages/
Content-Type: application/json

{
    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "Hello, how are you?",
    "role": "user"
}
```

### 4. Send a Threaded Reply

```http
POST http://localhost:8000/messages/
Content-Type: application/json

{
    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
    "content": "This is a reply to the previous message",
    "role": "user",
    "parent_message_id": "660e8400-e29b-41d4-a716-446655440001"
}
```

### 5. Get Conversation Thread

```http
GET http://localhost:8000/messages/conversation/9775b267-2641-4fb1-974a-b04835f803c8/thread?user_id=d264183a-eb1a-4ade-93a3-3438548d632a
```

### 6. Rate a Message

```http
POST http://localhost:8000/feedback/
Content-Type: application/json

{
    "message_id": "660e8400-e29b-41d4-a716-446655440001",
    "rating": 5,
    "comment": "Very helpful response!"
}
```

### 7. Get Message Rating Statistics

```http
GET http://localhost:8000/feedback/message/660e8400-e29b-41d4-a716-446655440001/stats
```

## Database Schema

### Users Table

- `id` (UUID, Primary Key)
- `username` (VARCHAR, Unique)
- `email` (VARCHAR, Unique)
- `password_hash` (TEXT)
- `full_name` (VARCHAR, Optional)
- `avatar_url` (VARCHAR, Optional)
- `created_at` (TIMESTAMPTZ)
- `updated_at` (TIMESTAMPTZ)

### Conversations Table

- `id` (UUID, Primary Key)
- `user_id` (UUID, Foreign Key → Users)
- `title` (VARCHAR, Required)
- `created_at` (TIMESTAMPTZ)
- `updated_at` (TIMESTAMPTZ)

### Messages Table

- `id` (UUID, Primary Key)
- `conversation_id` (UUID, Foreign Key → Conversations)
- `parent_message_id` (UUID, Foreign Key → Messages, Optional)
- `content` (TEXT)
- `role` (ENUM: user/assistant/system)
- `created_at` (TIMESTAMPTZ)

### Feedback Table

- `id` (UUID, Primary Key)
- `message_id` (UUID, Foreign Key → Messages)
- `user_id` (UUID, Foreign Key → Users)
- `rating` (SMALLINT, 1-5)
- `comment` (TEXT, Optional)
- `created_at` (TIMESTAMPTZ)
- `updated_at` (TIMESTAMPTZ)
- **Unique Constraint**: (message_id, user_id)

## Development

### Project Structure

```
app/
├── api/                    # FastAPI route handlers
│   ├── users.py           # User management endpoints
│   ├── conversations.py   # Conversation endpoints
│   ├── messages.py        # Message endpoints
│   └── feedback.py        # Feedback endpoints
├── core/                  # Core utilities
│   ├── database.py        # Database connection
│   └── security.py        # Password hashing
├── models/                # SQLAlchemy models
│   ├── base.py           # Base model with UUID and timestamps
│   ├── user.py           # User model
│   ├── conversation.py   # Conversation model
│   ├── message.py        # Message model with threading
│   ├── feedback.py       # Feedback model
│   └── enums.py          # Message role enum
├── repositories/          # Data access layer
│   ├── base.py           # Base repository
│   ├── user.py           # User repository
│   ├── conversation.py   # Conversation repository
│   ├── message.py        # Message repository
│   └── feedback.py       # Feedback repository
├── schemas/               # Pydantic schemas
│   ├── user.py           # User validation schemas
│   ├── conversation.py   # Conversation schemas
│   ├── message.py        # Message schemas
│   └── feedback.py       # Feedback schemas
├── services/              # Business logic layer
│   ├── user.py           # User service
│   ├── conversation.py   # Conversation service
│   ├── message.py        # Message service
│   └── feedback.py       # Feedback service
└── main.py               # FastAPI application
```

### Database Migrations

Generate a new migration:

```bash
alembic revision --autogenerate -m "Description of changes"
```

Apply migrations:

```bash
alembic upgrade head
```

View migration history:

```bash
alembic history
```

Rollback to previous migration:

```bash
alembic downgrade -1
```