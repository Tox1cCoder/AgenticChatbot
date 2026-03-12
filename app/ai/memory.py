import logging
from typing import Dict, List, Optional, Tuple
from uuid import UUID
from collections import deque

from sqlalchemy.orm import Session

from ..database.session import get_db
from ..repositories.message import MessageCRUDStrategy
from ..models.message import Message
from .schemas import AgentMessage, MessageRole
from ..core.config import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(
        self,
        conversation_id: UUID,
        user_id: UUID,
        batch_size: int = 100,
        max_messages: int = 0,
    ):
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.batch_size = max(1, batch_size)
        self.max_messages = max(0, max_messages)
        self._messages: deque[AgentMessage] = deque()
        self._message_repo = MessageCRUDStrategy(Message)
        self._initialized = False

    async def initialize(self, force_refresh: bool = False):
        if self._initialized and not force_refresh:
            return

        try:
            db = next(get_db())
            try:
                loaded_messages, total_available = self._load_messages_from_db(db)

                self._messages.clear()
                self._messages.extend(loaded_messages)

            finally:
                db.close()

            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            self._initialized = True

    def _load_messages_from_db(self, db: Session) -> Tuple[List[AgentMessage], int]:
        collected: List[AgentMessage] = []
        total_available = 0
        page = 1
        limit = self.batch_size

        while True:
            paginator = self._message_repo.get_by_conversation_id(
                db,
                self.conversation_id,
                page=page,
                limit=limit,
                order_by="created_at",
                order_direction="desc",
            )

            if page == 1:
                total_available = paginator.meta.total if paginator.meta else 0

            if not paginator.items:
                break

            for msg in paginator.items:
                agent_msg = self._db_to_agent_message(msg)
                if agent_msg:
                    collected.append(agent_msg)

                # Stop early when memory_max_messages limit reached
                if self.max_messages > 0 and len(collected) >= self.max_messages:
                    break

            # Stop paging if we've hit the cap
            if self.max_messages > 0 and len(collected) >= self.max_messages:
                break

            if paginator.meta.last_page <= page:
                break

            page += 1

        collected.reverse()
        return collected, total_available

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
    def __init__(
        self,
        batch_size: int = 100,
        max_messages: int = 0,
    ):
        self.batch_size = max(1, batch_size)
        self.max_messages = max(0, max_messages)
        self._memories: Dict[str, ConversationMemory] = {}

    async def get_memory(
        self, conversation_id: UUID, user_id: UUID, force_refresh: bool = False
    ) -> ConversationMemory:
        key = str(conversation_id)

        if key not in self._memories:
            memory = ConversationMemory(
                conversation_id,
                user_id,
                self.batch_size,
                max_messages=self.max_messages,
            )
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

    def clear_memory(self, conversation_id: UUID):
        key = str(conversation_id)
        if key in self._memories:
            del self._memories[key]


_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(
            batch_size=settings.memory_load_batch_size,
            max_messages=settings.memory_max_messages,
        )
    return _memory_manager
