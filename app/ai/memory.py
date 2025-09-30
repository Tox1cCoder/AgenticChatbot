"""
Memory Management for Multi-Agent System

Provides both short-term (session-based) and long-term (PostgreSQL-based) memory
for the multi-agent system.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime
from collections import deque
import logging

from sqlalchemy.orm import Session

from ..database.session import get_db
from ..repositories.message import MessageCRUDStrategy
from ..repositories.conversation import ConversationCRUDStrategy
from ..models.message import Message
from ..models.conversation import Conversation
from .schemas import AgentMessage, MessageRole, MessageType

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages both short-term and long-term memory for a conversation.

    Short-term memory: In-memory cache of recent messages for fast access
    Long-term memory: PostgreSQL database for persistent storage
    """

    def __init__(
        self,
        conversation_id: UUID,
        user_id: UUID,
        max_short_term_messages: int = 20,
    ):
        """
        Initialize conversation memory.

        Args:
            conversation_id: ID of the conversation
            user_id: ID of the user
            max_short_term_messages: Maximum messages to keep in short-term memory
        """
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.max_short_term_messages = max_short_term_messages

        # Short-term memory: deque for efficient FIFO operations
        self._short_term: deque[AgentMessage] = deque(maxlen=max_short_term_messages)

        # Metadata and context
        self._context: Dict[str, Any] = {}
        self._initialized = False

        # Repository strategies for database access
        self._message_repo = MessageCRUDStrategy(Message)
        self._conversation_repo = ConversationCRUDStrategy(Conversation)

        logger.info(
            f"ConversationMemory initialized for conversation {conversation_id}"
        )

    async def initialize(self) -> None:
        """Initialize memory by loading recent messages from database."""
        if self._initialized:
            return

        try:
            # Load recent messages from database into short-term memory
            db = next(get_db())
            try:
                messages = self._message_repo.get_by_conversation_id(
                    db,
                    self.conversation_id,
                    page=1,
                    limit=self.max_short_term_messages,
                    order_by="created_at",
                    order_direction="desc",
                )

                # Convert database messages to AgentMessage format (reverse order)
                for msg in reversed(messages.items):
                    agent_msg = self._db_message_to_agent_message(msg)
                    if agent_msg:
                        self._short_term.append(agent_msg)

                logger.info(
                    f"Loaded {len(messages.items)} messages from database into short-term memory"
                )

            finally:
                db.close()

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize memory from database: {e}")
            self._initialized = True  # Continue without historical data

    def add_message(self, message: AgentMessage) -> None:
        """
        Add a message to short-term memory.

        Note: Message persistence to PostgreSQL is handled by MessageService.
        This method only manages the in-memory conversation context for agents.

        Args:
            message: The message to add to short-term memory
        """
        # Add to short-term memory for agent context
        self._short_term.append(message)

    def get_recent_messages(
        self, limit: Optional[int] = None, include_system: bool = False
    ) -> List[AgentMessage]:
        """
        Get recent messages from short-term memory.

        Args:
            limit: Maximum number of messages to return (None for all)
            include_system: Whether to include system messages

        Returns:
            List of recent messages
        """
        messages = list(self._short_term)

        if not include_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if limit:
            messages = messages[-limit:]

        return messages

    def get_conversation_history(
        self, limit: int = 100, offset: int = 0
    ) -> List[AgentMessage]:
        """
        Get conversation history from long-term memory (database).

        Args:
            limit: Maximum number of messages to return
            offset: Number of messages to skip

        Returns:
            List of historical messages
        """
        try:
            db = next(get_db())
            try:
                page = (offset // limit) + 1
                messages = self._message_repo.get_by_conversation_id(
                    db,
                    self.conversation_id,
                    page=page,
                    limit=limit,
                    order_by="created_at",
                    order_direction="asc",
                )

                return [
                    self._db_message_to_agent_message(msg)
                    for msg in messages.items
                    if self._db_message_to_agent_message(msg) is not None
                ]

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []

    def search_messages(self, query: str, limit: int = 10) -> List[AgentMessage]:
        """
        Search messages in long-term memory using database full-text search.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching messages ordered by relevance (most recent first)
        """
        if not query or not query.strip():
            return []

        try:
            db = next(get_db())
            try:
                # Use repository's search method for efficient database search
                db_messages = self._message_repo.search_by_content(
                    db, self.conversation_id, query.strip(), limit
                )

                # Convert to AgentMessage format
                return [
                    self._db_message_to_agent_message(msg)
                    for msg in db_messages
                    if self._db_message_to_agent_message(msg) is not None
                ]

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to search messages: {e}")
            # Fallback to in-memory search on short-term messages
            messages = list(self._short_term)
            query_lower = query.lower()
            return [msg for msg in messages if query_lower in msg.content.lower()][
                :limit
            ]

    def update_context(self, key: str, value: Any) -> None:
        """Update conversation context metadata."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get conversation context metadata."""
        return self._context.get(key, default)

    def get_all_context(self) -> Dict[str, Any]:
        """Get all conversation context."""
        return self._context.copy()

    def clear_short_term_memory(self) -> None:
        """Clear short-term memory (but not database)."""
        self._short_term.clear()
        logger.info(
            f"Cleared short-term memory for conversation {self.conversation_id}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the memory state."""
        return {
            "conversation_id": str(self.conversation_id),
            "user_id": str(self.user_id),
            "short_term_messages": len(self._short_term),
            "max_short_term_messages": self.max_short_term_messages,
            "context_keys": list(self._context.keys()),
            "initialized": self._initialized,
        }

    def _db_message_to_agent_message(
        self, db_message: Message
    ) -> Optional[AgentMessage]:
        """Convert database Message to AgentMessage."""
        try:
            sender_to_role_map = {
                1: MessageRole.USER,  # MessageRole.user = 1
                2: MessageRole.ASSISTANT,  # MessageRole.assistant = 2
            }

            sender_value = db_message.sender
            role = sender_to_role_map.get(sender_value, MessageRole.ASSISTANT)

            return AgentMessage(
                role=role,
                content=db_message.content,
                message_type=MessageType.TEXT,
                metadata={
                    "message_id": str(db_message.id),
                    "created_at": (
                        db_message.created_at.isoformat()
                        if db_message.created_at
                        else None
                    ),
                },
            )
        except Exception as e:
            logger.error(f"Failed to convert database message: {e}")
            return None


class MemoryManager:
    """
    Global memory manager for all conversations.
    Manages multiple ConversationMemory instances.
    """

    def __init__(self, max_short_term_messages: int = 20):
        """
        Initialize memory manager.

        Args:
            max_short_term_messages: Default max messages for short-term memory
        """
        self.max_short_term_messages = max_short_term_messages
        self._memories: Dict[str, ConversationMemory] = {}
        logger.info("MemoryManager initialized")

    async def get_memory(
        self, conversation_id: UUID, user_id: UUID, auto_initialize: bool = True
    ) -> ConversationMemory:
        """
        Get or create memory for a conversation.

        Args:
            conversation_id: ID of the conversation
            user_id: ID of the user
            auto_initialize: Whether to auto-initialize from database

        Returns:
            ConversationMemory instance
        """
        key = str(conversation_id)

        if key not in self._memories:
            memory = ConversationMemory(
                conversation_id, user_id, self.max_short_term_messages
            )
            self._memories[key] = memory

            if auto_initialize:
                await memory.initialize()

        return self._memories[key]

    def clear_memory(self, conversation_id: UUID) -> None:
        """Clear memory for a conversation."""
        key = str(conversation_id)
        if key in self._memories:
            del self._memories[key]
            logger.info(f"Cleared memory for conversation {conversation_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        return {
            "active_conversations": len(self._memories),
            "max_short_term_messages": self.max_short_term_messages,
            "total_short_term_messages": sum(
                len(mem._short_term) for mem in self._memories.values()
            ),
        }


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def reset_memory_manager() -> None:
    """Reset the global memory manager (for testing)."""
    global _memory_manager
    _memory_manager = None
