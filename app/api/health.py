"""Aggregate, content-free internal health and metrics endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from app.core.config import settings
from app.observability.conversation_compaction import (
    ConversationCompactionHealthService,
    ConversationCompactionMetrics,
    conversation_compaction_metrics,
)
from app.observability.model_usage import (
    ModelUsageHealthService,
    ModelUsageMetrics,
)
from app.observability.model_usage import (
    model_usage_metrics as model_usage_metrics_singleton,
)

logger = logging.getLogger(__name__)


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


def _default_model_usage_service() -> ModelUsageHealthService:
    from app.core.container import get_container

    return ModelUsageHealthService(
        get_container().model_usage_repository(),
        metrics=model_usage_metrics_singleton,
        lookback_minutes=settings.model_usage_health_lookback_minutes,
        unattributed_degraded_ratio=(settings.model_usage_health_unattributed_degraded_ratio),
        rollup_lag_degraded_minutes=(settings.model_usage_health_rollup_lag_degraded_minutes),
        rollup_lag_unhealthy_minutes=(settings.model_usage_health_rollup_lag_unhealthy_minutes),
        persistence_failure_window_seconds=(settings.model_usage_health_failure_window_seconds),
    )


def create_health_router(
    *,
    service: ConversationCompactionHealthService | None = None,
    metrics: ConversationCompactionMetrics | None = None,
    model_usage_service: ModelUsageHealthService | None = None,
    model_usage_metrics: ModelUsageMetrics | None = None,
) -> APIRouter:
    router = APIRouter(tags=["health"])
    selected_metrics = metrics or conversation_compaction_metrics
    selected_usage_metrics = model_usage_metrics or model_usage_metrics_singleton

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

    @router.get("/health/model-usage")
    def model_usage_health():
        try:
            return (model_usage_service or _default_model_usage_service()).get_health()
        except Exception as exc:
            logger.warning(
                "model_usage health refresh failed: failure=%s",
                type(exc).__name__,
            )
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "data_available": False},
            )

    @router.get("/metrics/model-usage")
    def model_usage_metrics_endpoint():
        try:
            (model_usage_service or _default_model_usage_service()).get_health()
        except Exception as exc:
            selected_usage_metrics.mark_health_refresh_failure()
            logger.warning(
                "model_usage metrics refresh failed: failure=%s",
                type(exc).__name__,
            )
        return Response(
            content=selected_usage_metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return router


router = create_health_router()
