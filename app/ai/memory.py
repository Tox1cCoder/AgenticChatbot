import logging
from typing import Dict, List, Optional
from uuid import UUID
from collections import deque

from sqlalchemy.orm import Session

from ..database.session import get_db
from ..repositories.message import MessageCRUDStrategy
from ..models.message import Message
from .schemas import AgentMessage, MessageRole

logger = logging.getLogger(__name__)


class ConversationMemory:

    def __init__(self, conversation_id: UUID, user_id: UUID, max_messages: int = 20):
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.max_messages = max_messages
        self._messages: deque[AgentMessage] = deque(maxlen=max_messages)
        self._message_repo = MessageCRUDStrategy(Message)
        self._initialized = False

    async def initialize(self, force_refresh: bool = False):
        if self._initialized and not force_refresh:
            return

        try:
            db = next(get_db())
            try:
                messages = self._message_repo.get_by_conversation_id(
                    db,
                    self.conversation_id,
                    page=1,
                    limit=self.max_messages,
                    order_by="created_at",
                    order_direction="desc",
                )

                if force_refresh:
                    self._messages.clear()

                for msg in reversed(messages.items):
                    agent_msg = self._db_to_agent_message(msg)
                    if agent_msg:
                        self._messages.append(agent_msg)

                logger.info(
                    f"Loaded {len(messages.items)} messages from database for conversation {self.conversation_id}"
                )
            finally:
                db.close()

            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            self._initialized = True

    def add_message(self, message: AgentMessage):
        self._messages.append(message)

    def get_recent_messages(
        self,
        limit: Optional[int] = None,
        include_system: bool = False,
        exclude_last: int = 0,
    ) -> List[AgentMessage]:
        """
        Get recent messages from memory.

        Args:
            limit: Maximum number of messages to return
            include_system: Whether to include system messages
            exclude_last: Number of most recent messages to exclude
        """
        messages = list(self._messages)

        if not include_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if exclude_last > 0:
            messages = messages[:-exclude_last] if len(messages) > exclude_last else []

        if limit:
            messages = messages[-limit:]

        return messages

    def clear(self):
        self._messages.clear()
        logger.info(f"Cleared memory for conversation {self.conversation_id}")

    def _db_to_agent_message(self, db_message: Message) -> Optional[AgentMessage]:
        try:
            sender_to_role = {1: MessageRole.USER, 2: MessageRole.ASSISTANT}
            role = sender_to_role.get(db_message.sender, MessageRole.ASSISTANT)

            return AgentMessage(
                role=role,
                content=db_message.content,
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

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._memories: Dict[str, ConversationMemory] = {}
        logger.info("MemoryManager initialized")

    async def get_memory(
        self, conversation_id: UUID, user_id: UUID, force_refresh: bool = False
    ) -> ConversationMemory:
        key = str(conversation_id)

        if key not in self._memories:
            memory = ConversationMemory(conversation_id, user_id, self.max_messages)
            self._memories[key] = memory
            await memory.initialize()
        elif force_refresh:
            await self._memories[key].initialize(force_refresh=True)

        return self._memories[key]

    async def refresh_memory(self, conversation_id: UUID):
        """Refresh memory for a conversation by reloading from database"""
        key = str(conversation_id)
        if key in self._memories:
            await self._memories[key].initialize(force_refresh=True)
            logger.info(f"Refreshed memory for conversation {conversation_id}")

    def clear_memory(self, conversation_id: UUID):
        key = str(conversation_id)
        if key in self._memories:
            del self._memories[key]
            logger.info(f"Cleared memory for conversation {conversation_id}")


_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
