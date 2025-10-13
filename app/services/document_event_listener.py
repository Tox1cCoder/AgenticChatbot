import logging
from typing import Optional

from app.core.events import DocumentEvent, DocumentEventData, EventListener


class DocumentEventLogger(EventListener):
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    async def handle_event(
        self, event_type: DocumentEvent, data: DocumentEventData
    ) -> None:
        msg_ctx = {
            "document_id": str(data.document_id) if data.document_id else None,
            "conversation_id": (
                str(data.conversation_id) if data.conversation_id else None
            ),
            "user_id": str(data.user_id) if data.user_id else None,
            "filename": data.filename,
            "status": data.status,
            "metadata": data.metadata or {},
        }
