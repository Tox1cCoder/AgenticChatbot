from typing import Any, List
from uuid import UUID
from fastapi import APIRouter, status, Query
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.conversation_service_interface import IConversationService
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationRead,
)
from app.interfaces.message_service_interface import IMessageService
from app.schemas.message import MessageRead
from app.schemas.responses import ApiResponse
from app.schemas.responses.paginated_response import PaginatedApiResponse
from app.schemas.pagination import ConversationPaginationParams, MessagePaginationParams
from app.core.config import settings

router = APIRouter(prefix="/conversations", tags=["conversations"])


class GenerateTitleRequest(BaseModel):
    message: str


class GenerateTitleResponse(BaseModel):
    title: str


async def generate_title_from_message(user_message: str) -> str:
    """
    Generate a concise, descriptive title for a conversation based on the first user message.

    Args:
        user_message: The first message from the user

    Returns:
        A short, descriptive title (max 50 characters)
    """
    try:
        api_key = settings.gemini_api_key
        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=api_key,
            temperature=1
        )

        prompt = f"""Generate a very short, concise title (max 50 characters) for a conversation that starts with this user message:

"{user_message}"

Requirements:
- Maximum 50 characters
- Capture the main topic or intent
- No quotes, no punctuation at the end
- Should be clear and descriptive
- Use title case

Examples:
User: "How do I learn Python?" → Title: "Learning Python"
User: "What are the health benefits of exercise?" → Title: "Health Benefits of Exercise"
User: "Can you help me plan a trip to Japan?" → Title: "Japan Trip Planning"

Title:"""

        response = await llm.ainvoke(prompt)
        title = response.content.strip()

        # Clean up the title
        title = title.strip("\"'")  # Remove quotes
        title = title.rstrip(".")  # Remove trailing period

        # Ensure it's not too long
        if len(title) > 50:
            title = title[:47] + "..."

        # Fallback to truncated message if generation fails
        if not title or len(title) < 3:
            title = user_message[:50]
            if len(user_message) > 50:
                title = title[:47] + "..."

        return title

    except Exception as e:
        # Fallback: use truncated user message
        title = user_message[:50]
        if len(user_message) > 50:
            title = title[:47] + "..."
        return title


@router.post("/generate-title", response_model=ApiResponse[GenerateTitleResponse])
@AppAutoInjector.auto_inject()
async def generate_conversation_title(
    request: GenerateTitleRequest,
    user_id: UUID,
) -> ApiResponse[GenerateTitleResponse]:
    """Generate a concise title for a conversation based on the first message"""
    title = await generate_title_from_message(request.message)
    return ApiResponse(
        success=True,
        message="Title generated successfully",
        data=GenerateTitleResponse(title=title),
    )


@router.post(
    "/",
    response_model=ApiResponse[ConversationRead],
    status_code=status.HTTP_201_CREATED,
)
@AppAutoInjector.auto_inject()
async def create_conversation(
    conversation_data: ConversationCreate,
    conversation_service: IConversationService,
    user_id: UUID,
) -> ApiResponse[ConversationRead]:
    """Create a new conversation for authenticated user"""
    result = conversation_service.create_conversation(conversation_data, user_id)
    return ApiResponse(
        success=True, message="Conversation created successfully", data=result
    )


@router.get("/{conversation_id}", response_model=ApiResponse[ConversationRead])
@AppAutoInjector.auto_inject()
async def get_conversation(
    conversation_id: UUID,
    conversation_service: IConversationService,
    user_id: UUID,
) -> ApiResponse[ConversationRead]:
    """Get conversation by ID"""
    result = conversation_service.get_by_id_for_user(conversation_id, user_id)
    return ApiResponse(
        success=True, message="Conversation retrieved successfully", data=result
    )


@router.get("/", response_model=PaginatedApiResponse[ConversationRead])
@AppAutoInjector.auto_inject()
async def get_conversations(
    conversation_service: IConversationService,
    user_id: UUID,
    pagination: ConversationPaginationParams,
    include: List[str] = Query(
        default=[], description="Array of includes e.g. ['messages', 'feedback']"
    ),
    latest_messages: int = Query(
        3,
        alias="latestMessages",
        description="Number of latest messages to include",
    ),
) -> PaginatedApiResponse[ConversationRead]:
    """Get all conversations for authenticated user"""
    paginated_result = conversation_service.get_by_user_id(
        user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.to_snake_case(),
        order_direction=pagination.order_direction.value,
        include=include,
        latest_messages=latest_messages,
    )
    return PaginatedApiResponse.from_paginator(
        paginated_result, "Conversations retrieved successfully"
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=PaginatedApiResponse[MessageRead],
)
@AppAutoInjector.auto_inject()
async def get_conversation_messages(
    conversation_id: UUID,
    message_service: IMessageService,
    user_id: UUID,
    pagination: MessagePaginationParams,
    include: List[str] = Query(
        default=[], description="Array of includes e.g. ['feedback']"
    ),
) -> PaginatedApiResponse[MessageRead]:
    """Get conversation's messages (requires user ownership)"""
    include_feedback = "feedback" in include
    paginated_result = message_service.get_conversation_messages(
        conversation_id,
        user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.to_snake_case(),
        order_direction=pagination.order_direction.value,
        include_feedback=include_feedback,
    )
    return PaginatedApiResponse.from_paginator(
        paginated_result, "Conversation messages retrieved successfully"
    )


@router.patch("/{conversation_id}", response_model=ApiResponse[ConversationRead])
@AppAutoInjector.auto_inject()
async def update_conversation(
    conversation_id: UUID,
    conversation_data: ConversationUpdate,
    conversation_service: IConversationService,
    user_id: UUID,
) -> ApiResponse[ConversationRead]:
    """Update conversation (requires user ownership)"""
    result = conversation_service.update_conversation(
        conversation_id, user_id, conversation_data
    )
    return ApiResponse(
        success=True, message="Conversation updated successfully", data=result
    )


@router.delete("/{conversation_id}", response_model=ApiResponse[Any])
@AppAutoInjector.auto_inject()
async def delete_conversation(
    conversation_id: UUID,
    conversation_service: IConversationService,
    user_id: UUID,
) -> ApiResponse[Any]:
    """Delete conversation (requires user ownership)"""
    conversation_service.delete_conversation(conversation_id, user_id)
    return ApiResponse(success=True, message="Conversation deleted successfully")
