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

## Prerequisites

- Python 3.10 or higher
- PostgreSQL 12 or higher
- Git (for cloning the repository)

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd sample-chatbot
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

   Create a PostgreSQL database:

   ```sql
   CREATE DATABASE chatbot;
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

- `POST /users/` - Create a new user
- `GET /users/{user_id}` - Get user by ID
- `GET /users/` - List all users (paginated)
- `GET /users/email/{email}` - Get user by email
- `GET /users/username/{username}` - Get user by username
- `PUT /users/{user_id}` - Update user
- `DELETE /users/{user_id}` - Delete user

### Conversation Management

- `POST /conversations/` - Create a new conversation
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
    "email": "test@example.com"
}
```

### 2. Create a Conversation

```http
POST http://localhost:8000/conversations/
Content-Type: application/json

{
    "user_id": 1,
    "title": "My First Chat"
}
```

### 3. Send a Message (Triggers Bot Response)

```http
POST http://localhost:8000/messages/
Content-Type: application/json

{
    "conversation_id": 1,
    "user_id": 1,
    "content": "Hello, how are you?",
    "role": "user"
}
```

### 4. Get Conversation History

```http
GET http://localhost:8000/messages/conversation/1/history?user_id=1
```

## Database Schema

### Users Table

- `id` (Primary Key)
- `username` (Unique)
- `email` (Unique)
- `created_at`
- `updated_at`

### Conversations Table

- `id` (Primary Key)
- `user_id` (Foreign Key → Users)
- `title`
- `created_at`
- `updated_at`

### Messages Table

- `id` (Primary Key)
- `conversation_id` (Foreign Key → Conversations)
- `user_id` (Foreign Key → Users)
- `content`
- `role` (user/assistant)
- `created_at`

## Development

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