"""Service boundary for authenticated model-usage analytics."""

from abc import ABC, abstractmethod
from uuid import UUID

from app.schemas.model_usage import (
    ConversationUsage,
    ConversationUsageQuery,
    UsageDashboard,
    UsageDashboardQuery,
)


class IModelUsageService(ABC):
    """Build usage responses scoped to the authenticated user."""

    @abstractmethod
    def get_dashboard(self, *, user_id: UUID, query: UsageDashboardQuery) -> UsageDashboard:
        """Return account or owned-conversation usage for the requested range."""

    @abstractmethod
    def get_conversation_usage(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        query: ConversationUsageQuery,
    ) -> ConversationUsage:
        """Return usage details for one owned conversation."""


__all__ = ["IModelUsageService"]
