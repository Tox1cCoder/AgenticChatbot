"""Aggregate conversation-compaction health and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from app.core.config import settings
from app.observability.conversation_compaction import (
    ConversationCompactionHealthService,
    ConversationCompactionMetrics,
    conversation_compaction_metrics,
)


def _default_service() -> ConversationCompactionHealthService:
    from app.core.container import get_container

    return ConversationCompactionHealthService(
        get_container().conversation_compaction_repository(),
        queue_age_degraded_seconds=max(
            settings.conversation_summary_reconcile_seconds * 2,
            settings.conversation_summary_lease_seconds,
        ),
        lag_degraded_sequences=max(1, settings.conversation_summary_trigger_messages),
    )


def create_health_router(
    *,
    service: ConversationCompactionHealthService | None = None,
    metrics: ConversationCompactionMetrics | None = None,
) -> APIRouter:
    router = APIRouter(tags=["health"])
    selected_metrics = metrics or conversation_compaction_metrics

    @router.get("/health/conversation-compaction")
    def conversation_compaction_health():
        snapshot = (service or _default_service()).get_health()
        selected_metrics.update_health(snapshot)
        return snapshot

    @router.get("/metrics/conversation-compaction")
    def conversation_compaction_metrics_endpoint():
        return Response(
            content=selected_metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return router


router = create_health_router()
