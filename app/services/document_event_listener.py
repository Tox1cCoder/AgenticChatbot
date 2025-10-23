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
            "metadata": data.metadata or {}
        }
        try:
            if event_type == DocumentEvent.UPLOAD_STARTED:
                self.logger.info(
                    "Document upload started",
                    extra={"event": str(event_type), **msg_ctx},
                )
            elif event_type == DocumentEvent.PROCESSING_STARTED:
                self.logger.info(
                    "Document processing started",
                    extra={"event": str(event_type), **msg_ctx},
                )
            elif event_type == DocumentEvent.PROCESSING_COMPLETED:
                self.logger.info(
                    "Document processing completed",
                    extra={"event": str(event_type), **msg_ctx},
                )
            elif event_type == DocumentEvent.PROCESSING_FAILED:
                # Include error details when available
                err = getattr(data, "error", None)
                self.logger.error(
                    "Document processing failed",
                    extra={"event": str(event_type), "error": err, **msg_ctx},
                )
            elif event_type == DocumentEvent.DELETED:
                self.logger.info(
                    "Document deleted", extra={"event": str(event_type), **msg_ctx}
                )
            else:
                self.logger.debug(
                    "Unhandled document event",
                    extra={"event": str(event_type), **msg_ctx},
                )
        except Exception:
            try:
                self.logger.exception("Failed while logging document event")
            except Exception:
                pass
