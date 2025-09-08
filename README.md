# Sample Chatbot - FastAPI & PostgreSQL

A modern, production-ready chatbot application built with FastAPI and PostgreSQL, featuring threaded conversations, user authentication with password hashing, and a comprehensive feedback system.

## 🚀 Features

- **User Authentication**: Secure password hashing with bcrypt
- **UUID Primary Keys**: All entities use UUID for better scalability and security
- **Threaded Conversations**: Support for message threading with parent-child relationships
- **Feedback System**: Users can rate and comment on messages (1-5 star ratings)
- **Message Roles**: Structured message types (user, assistant, system)
- **RESTful API**: Complete CRUD operations with FastAPI
- **Database Migrations**: Alembic for schema version control
- **Clean Architecture**: Layered design with separation of concerns
- **Type Safety**: Full type hints with Pydantic validation
- **Auto Documentation**: Interactive API docs with Swagger UI

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
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
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
GET http://localhost:8000/messages/conversation/550e8400-e29b-41d4-a716-446655440000/thread?user_id=770e8400-e29b-41d4-a716-446655440002
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

### Key Features Implemented

1. **UUID Primary Keys**: Better for distributed systems and security
2. **Password Hashing**: Secure bcrypt hashing for user passwords
3. **Message Threading**: Parent-child relationships for reply chains
4. **Feedback System**: 1-5 star ratings with optional comments
5. **Message Roles**: Enum-based role system (user/assistant/system)
6. **Clean Architecture**: Separation of concerns across layers
7. **Type Safety**: Full type hints throughout the codebase
8. **Validation**: Pydantic schemas for request/response validation

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

### Security Features

- **Password Hashing**: Uses bcrypt with salt for secure password storage
- **UUID Keys**: Prevents enumeration attacks on entity IDs
- **Input Validation**: Pydantic schemas validate all input data
- **SQL Injection Protection**: SQLAlchemy ORM prevents SQL injection
- **Access Control**: User ownership validation for conversations and messages

### Performance Considerations

- **Database Indexes**: Proper indexing on foreign keys and search fields
- **Pagination**: All list endpoints support skip/limit pagination
- **Lazy Loading**: Relationships loaded only when needed
- **Connection Pooling**: SQLAlchemy manages database connections efficiently

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
