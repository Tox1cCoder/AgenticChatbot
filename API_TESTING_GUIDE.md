# Chatbot API Testing Guide

## Overview

This guide provides step-by-step instructions for testing the Chatbot API using Postman.

## Prerequisites

1. FastAPI server running on `http://localhost:8000`
2. Postman installed
3. Import the `postman_collection.json` file into Postman

## Testing Workflow

### 1. Import Collection

1. Open Postman
2. Click "Import" button
3. Select `postman_collection.json` file
4. The collection "Chatbot API Test Collection" will be imported with all endpoints

### 2. Test Sequence

#### Step 1: Health Checks

- **API Health**: Verify the API is running
  - Expected: 200 OK with service status
- **Database Health**: Verify database connectivity
  - Expected: 200 OK with database status

#### Step 2: Authentication

- **Sign Up**: Create a new test user

  - Request body: `{"username": "test_user", "email": "test@example.com", "password": "testpassword123"}`
  - Expected: 201 Created with user data
  - Auto-saves: `user_id` to collection variables

- **Login**: Authenticate with created user
  - Request body: `{"email": "test@example.com", "password": "testpassword123"}`
  - Expected: 200 OK with access token
  - Auto-saves: `access_token` and `user_id` to collection variables

#### Step 3: User Management

- **Get All Users**: List all users (admin functionality)

  - Headers: `Authorization: Bearer {{access_token}}`
  - Expected: 200 OK with user list

- **Get Current User**: Get authenticated user details
  - Headers: `Authorization: Bearer {{access_token}}`
  - Expected: 200 OK with user data

#### Step 4: Conversation Management

- **Create Conversation**: Start a new conversation

  - Headers: `Authorization: Bearer {{access_token}}`
  - Request body: `{"title": "Test Conversation"}`
  - Expected: 201 Created with conversation data
  - Auto-saves: `conversation_id` to collection variables

- **Get All Conversations**: List user's conversations

  - Headers: `Authorization: Bearer {{access_token}}`
  - Expected: 200 OK with conversation list

- **Get Conversation by ID**: Get specific conversation
  - Headers: `Authorization: Bearer {{access_token}}`
  - Expected: 200 OK with conversation details

#### Step 5: Message Management

- **Send Message**: Send a user message

  - Headers: `Authorization: Bearer {{access_token}}`
  - Request body: `{"conversationId": "{{conversation_id}}", "content": "Hello, this is a test message!", "role": 1}`
  - Expected: 201 Created with message data (triggers bot response)
  - Auto-saves: `message_id` to collection variables

- **Get Conversation Thread**: Get all messages in conversation

  - Headers: `Authorization: Bearer {{access_token}}`
  - Expected: 200 OK with message thread (user + bot messages)

- **Get Message by ID**: Get specific message
  - Headers: `Authorization: Bearer {{access_token}}`
  - Expected: 200 OK with message details

#### Step 6: Feedback Management

- **Submit Feedback**: Rate a bot message

  - Headers: `Authorization: Bearer {{access_token}}`
  - Request body: `{"messageId": "{{message_id}}", "rating": 5, "comment": "Great response!"}`
  - Expected: 201 Created with feedback data

- **Get Message Feedback**: Get all feedback for a message

  - Headers: `Authorization: Bearer {{access_token}}`
  - Expected: 200 OK with feedback list

- **Get Feedback Stats**: Get aggregated feedback statistics
  - Headers: `Authorization: Bearer {{access_token}}`
  - Expected: 200 OK with statistics (average rating, count)

#### Step 7: Cleanup

- **Delete Conversation**: Clean up test data
  - Headers: `Authorization: Bearer {{access_token}}`
  - Expected: 200 OK with success message

## Environment Variables

The collection uses these variables (automatically managed):

- `base_url`: API base URL (http://localhost:8000)
- `access_token`: JWT token from login
- `user_id`: User ID from signup/login
- `conversation_id`: Created conversation ID
- `message_id`: Created message ID

## Error Scenarios to Test

### Authentication Errors

1. Login with invalid credentials
2. Access protected endpoints without token
3. Use expired token

### Validation Errors

1. Create conversation with empty title
2. Send message with invalid conversation ID
3. Submit feedback with invalid rating (< 1 or > 5)

### Authorization Errors

1. Access another user's conversation
2. Send message to conversation you don't own
3. Delete conversation you don't own

## Expected Response Format

All successful responses follow this structure:

```json
{
  "success": true,
  "message": "Success message",
  "data": {
    /* response data */
  }
}
```

Error responses:

```json
{
  "success": false,
  "message": "Error message",
  "errorCode": "ERROR_CODE",
  "details": {
    /* error details */
  }
}
```

## Bot Response Testing

When sending a user message, the system automatically generates a bot response using the Gemini API. Check the conversation thread to see both user and bot messages.

## Notes

- Run requests in sequence for the first time to set up dependencies
- Collection variables are shared across all requests
- The JWT token expires after the configured time (default: 60 minutes)
- Bot responses depend on Gemini API configuration in settings
