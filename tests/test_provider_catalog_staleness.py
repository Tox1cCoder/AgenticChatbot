"""Provider catalog TTL-staleness refresh (fixes stale model lists)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.services.provider_service import (
    PROVIDER_CATALOG_TTL_SECONDS,
    ProviderService,
    _catalog_is_stale,
)


def test_catalog_is_stale_cases():
    assert _catalog_is_stale(None) is True
    assert _catalog_is_stale("not-a-timestamp") is True
    assert _catalog_is_stale(datetime.now(timezone.utc).isoformat()) is False
    old = datetime.now(timezone.utc) - timedelta(seconds=PROVIDER_CATALOG_TTL_SECONDS + 60)
    assert _catalog_is_stale(old.isoformat()) is True


def _service_with_cached(status: dict):
    svc = ProviderService.__new__(ProviderService)
    svc._normalize_provider_type = lambda p: p  # type: ignore[assignment]
    svc.get_cached_provider_status = lambda user_id, provider_type: dict(status)  # type: ignore[assignment]
    return svc


@pytest.mark.asyncio
async def test_get_provider_status_refreshes_stale_but_populated_catalog():
    """A populated-but-stale catalog re-syncs so new models appear."""
    stale = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    svc = _service_with_cached(
        {
            "provider_type": "gemini",
            "configured": True,
            "models": [{"id": "gemini-2.5-flash"}],
            "last_synced_at": stale,
            "sync_status": "ready",
        }
    )
    fresh_catalog = {
        "provider_type": "gemini",
        "configured": True,
        "models": [{"id": "gemini-2.5-flash"}, {"id": "gemini-3-flash-preview"}],
        "sync_status": "ready",
    }
    svc.sync_provider_models = AsyncMock(return_value=fresh_catalog)  # type: ignore[assignment]

    result = await svc.get_provider_status(
        user_id=uuid4(), provider_type="gemini", refresh_if_missing=True
    )
    svc.sync_provider_models.assert_awaited_once()
    assert any(m["id"] == "gemini-3-flash-preview" for m in result["models"])


@pytest.mark.asyncio
async def test_get_provider_status_skips_refresh_when_fresh():
    fresh = datetime.now(timezone.utc).isoformat()
    svc = _service_with_cached(
        {
            "provider_type": "gemini",
            "configured": True,
            "models": [{"id": "gemini-2.5-flash"}],
            "last_synced_at": fresh,
            "sync_status": "ready",
        }
    )
    svc.sync_provider_models = AsyncMock()  # type: ignore[assignment]

    await svc.get_provider_status(user_id=uuid4(), provider_type="gemini", refresh_if_missing=True)
    svc.sync_provider_models.assert_not_awaited()
