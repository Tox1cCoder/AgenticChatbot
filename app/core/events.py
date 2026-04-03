from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol
from uuid import UUID


class DocumentEvent(Enum):
    UPLOAD_STARTED = auto()
    PROCESSING_STARTED = auto()
    PROCESSING_COMPLETED = auto()
    PROCESSING_FAILED = auto()
    DELETED = auto()


@dataclass
class DocumentEventData:
    document_id: UUID | None = None
    conversation_id: UUID | None = None
    user_id: UUID | None = None
    filename: str | None = None
    status: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EventListener(Protocol):
    async def handle_event(self, event_type: DocumentEvent, data: DocumentEventData) -> None: ...


class EventBus:
    """Async event bus for document lifecycle events."""

    def __init__(self) -> None:
        self._listeners: defaultdict[DocumentEvent, list[EventListener]] = defaultdict(list)

    def register_listener(self, event_type: DocumentEvent, listener: EventListener) -> None:
        if listener not in self._listeners[event_type]:
            self._listeners[event_type].append(listener)

    def unregister_listener(self, event_type: DocumentEvent, listener: EventListener) -> None:
        listeners = self._listeners.get(event_type, [])
        if listener in listeners:
            listeners.remove(listener)

    async def emit(self, event_type: DocumentEvent, data: DocumentEventData) -> None:
        listeners = list(self._listeners.get(event_type, []))
        for listener in listeners:
            await listener.handle_event(event_type, data)


_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
