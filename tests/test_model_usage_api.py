"""HTTP contract tests for authenticated model-usage read APIs."""

import inspect
from collections.abc import Iterator
from datetime import datetime, timezone
from uuid import UUID

import pytest
from dependency_injector import providers
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.api.model_usage import router as model_usage_router
from app.core.config import settings
from app.core.container import Container, container, setup_auto_injection
from app.core.dependency_injection import AppAutoInjector, AppContainerInjector
from app.core.exceptions import ResourceNotFoundException
from app.interfaces import IModelUsageService
from app.main import app
from app.repositories.model_usage import ModelUsageRepository
from app.schemas.model_usage import (
    ConversationUsage,
    ConversationUsageQuery,
    UsageCoverage,
    UsageDashboard,
    UsageDashboardQuery,
    UsageRange,
    UsageTotals,
)
from app.services.model_usage_service import ModelUsageService

# Importing app.main builds the global application and temporarily points the
# injector maps at the container instance. API routes have already captured
# their providers, so restore the declarative baseline for the rest of pytest.
setup_auto_injection(Container)

_USER_A = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_GENERATED_AT = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
_RANGE_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
_RANGE_END = datetime(2026, 1, 2, tzinfo=timezone.utc)
_CONVERSATION_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")


@pytest.fixture(autouse=True)
def _isolate_injector_wiring() -> Iterator[None]:
    setup_auto_injection(Container)
    try:
        yield
    finally:
        setup_auto_injection(Container)


def _usage_range() -> UsageRange:
    return UsageRange(
        from_=_RANGE_START,
        to=_RANGE_END,
        bucket="day",
        timezone="UTC",
    )


def _empty_dashboard() -> UsageDashboard:
    return UsageDashboard(
        totals=UsageTotals(),
        outcomes=[],
        series=[],
        by_provider=[],
        by_model=[],
        by_operation=[],
        by_agent=[],
        top_conversations=[],
        coverage=UsageCoverage(),
        range=_usage_range(),
        generated_at=_GENERATED_AT,
    )


def _empty_conversation_usage() -> ConversationUsage:
    return ConversationUsage(
        totals=UsageTotals(),
        by_provider=[],
        by_model=[],
        coverage=UsageCoverage(),
        latest_context_window=None,
        range=_usage_range(),
        generated_at=_GENERATED_AT,
    )


class UsageServiceSpy:
    def __init__(self) -> None:
        self.dashboard_calls: list[tuple[UUID, UsageDashboardQuery]] = []
        self.conversation_calls: list[tuple[UUID, UUID, ConversationUsageQuery]] = []
        self.conversation_error: ResourceNotFoundException | None = None

    def get_dashboard(self, *, user_id: UUID, query: UsageDashboardQuery) -> UsageDashboard:
        self.dashboard_calls.append((user_id, query))
        return _empty_dashboard()

    def get_conversation_usage(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        query: ConversationUsageQuery,
    ) -> ConversationUsage:
        self.conversation_calls.append((user_id, conversation_id, query))
        if self.conversation_error is not None:
            raise self.conversation_error
        return _empty_conversation_usage()


@pytest.fixture
def usage_service() -> UsageServiceSpy:
    spy = UsageServiceSpy()
    with Container.model_usage_service.override(providers.Object(spy)):
        yield spy


def _authorization(user_id: UUID) -> dict[str, str]:
    token = container.jwt_service().create_access_token({"sub": str(user_id)})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def validating_usage_service() -> ModelUsageService:
    service = ModelUsageService(
        repository=object(),
        conversation_repository=object(),
        clock=lambda: _GENERATED_AT,
    )
    with Container.model_usage_service.override(providers.Object(service)):
        yield service


def test_api_test_module_preserves_declarative_injector_wiring() -> None:
    assert AppAutoInjector.wiring_map[IModelUsageService] is Container.model_usage_service
    assert AppContainerInjector.wiring_map[IModelUsageService] is Container.model_usage_service
    assert AppContainerInjector.wiring_map[ModelUsageRepository] is (
        Container.model_usage_repository
    )


def test_usage_route_endpoints_are_synchronous_for_threadpool_execution() -> None:
    endpoints = [
        route.endpoint for route in model_usage_router.routes if isinstance(route, APIRoute)
    ]

    assert len(endpoints) == 2
    assert all(not inspect.iscoroutinefunction(endpoint) for endpoint in endpoints)


@pytest.mark.parametrize(
    "path",
    ["/usage/dashboard", f"/usage/conversations/{_CONVERSATION_ID}"],
)
def test_usage_endpoints_require_jwt(path: str) -> None:
    response = TestClient(app).get(path)

    assert response.status_code == 401
    assert response.json() == {
        "success": False,
        "code": "http_error",
        "message": "Not authenticated",
    }


@pytest.mark.parametrize(
    "path",
    ["/usage/dashboard", f"/usage/conversations/{_CONVERSATION_ID}"],
)
def test_usage_endpoints_reject_invalid_jwt(path: str) -> None:
    response = TestClient(app).get(
        path,
        headers={"Authorization": "Bearer not-a-valid-jwt"},
    )

    assert response.status_code == 401
    assert response.json() == {
        "success": False,
        "code": "INVALID_CREDENTIALS",
        "message": "Invalid authentication credentials",
    }


def test_dashboard_uses_authenticated_identity_and_returns_empty_camel_case_envelope(
    usage_service: UsageServiceSpy,
) -> None:
    response = TestClient(app).get(
        "/usage/dashboard",
        params={
            "userId": str(_USER_B),
            "user_id": str(_USER_B),
            "from": "2026-01-01T00:00:00+07:00",
            "to": "2026-01-01T02:00:00+07:00",
            "bucket": "hour",
            "timezone": "Asia/Bangkok",
            "conversationId": str(_CONVERSATION_ID),
        },
        headers=_authorization(_USER_A),
    )

    assert response.status_code == 200
    assert len(usage_service.dashboard_calls) == 1
    user_id, query = usage_service.dashboard_calls[0]
    assert user_id == _USER_A
    assert query == UsageDashboardQuery(
        from_="2026-01-01T00:00:00+07:00",
        to="2026-01-01T02:00:00+07:00",
        bucket="hour",
        timezone="Asia/Bangkok",
        conversation_id=_CONVERSATION_ID,
    )
    assert response.json() == {
        "success": True,
        "message": "Usage dashboard retrieved",
        "data": {
            "totals": {
                "inputTokens": 0,
                "outputTokens": 0,
                "totalTokens": 0,
                "reasoningTokens": 0,
                "cachedInputTokens": 0,
                "generatedImages": 0,
                "requestCount": 0,
            },
            "outcomes": [],
            "series": [],
            "byProvider": [],
            "byModel": [],
            "byOperation": [],
            "byAgent": [],
            "topConversations": [],
            "coverage": {
                "providerReportedRequests": 0,
                "mixedRequests": 0,
                "locallyEstimatedRequests": 0,
                "unavailableRequests": 0,
                "requestsWithKnownTotal": 0,
                "totalRequests": 0,
                "knownTotalRatio": 0.0,
            },
            "range": {
                "from": "2026-01-01T00:00:00Z",
                "to": "2026-01-02T00:00:00Z",
                "bucket": "day",
                "timezone": "UTC",
            },
            "generatedAt": "2026-01-02T03:04:00Z",
        },
        "error": None,
    }


def test_conversation_usage_returns_typed_empty_camel_case_envelope(
    usage_service: UsageServiceSpy,
) -> None:
    response = TestClient(app).get(
        f"/usage/conversations/{_CONVERSATION_ID}",
        params={
            "from": "2026-01-01T00:00:00+07:00",
            "to": "2026-01-01T02:00:00+07:00",
            "bucket": "hour",
            "timezone": "Asia/Bangkok",
        },
        headers=_authorization(_USER_A),
    )

    assert response.status_code == 200
    assert usage_service.conversation_calls == [
        (
            _USER_A,
            _CONVERSATION_ID,
            ConversationUsageQuery(
                from_="2026-01-01T00:00:00+07:00",
                to="2026-01-01T02:00:00+07:00",
                bucket="hour",
                timezone="Asia/Bangkok",
            ),
        )
    ]
    assert response.json() == {
        "success": True,
        "message": "Conversation usage retrieved",
        "data": {
            "totals": {
                "inputTokens": 0,
                "outputTokens": 0,
                "totalTokens": 0,
                "reasoningTokens": 0,
                "cachedInputTokens": 0,
                "generatedImages": 0,
                "requestCount": 0,
            },
            "byProvider": [],
            "byModel": [],
            "coverage": {
                "providerReportedRequests": 0,
                "mixedRequests": 0,
                "locallyEstimatedRequests": 0,
                "unavailableRequests": 0,
                "requestsWithKnownTotal": 0,
                "totalRequests": 0,
                "knownTotalRatio": 0.0,
            },
            "latestContextWindow": None,
            "range": {
                "from": "2026-01-01T00:00:00Z",
                "to": "2026-01-02T00:00:00Z",
                "bucket": "day",
                "timezone": "UTC",
            },
            "generatedAt": "2026-01-02T03:04:00Z",
        },
        "error": None,
    }


def test_foreign_conversation_is_an_ownership_safe_not_found(
    usage_service: UsageServiceSpy,
) -> None:
    usage_service.conversation_error = ResourceNotFoundException(
        detail="Conversation not found",
        error_code="CONVERSATION_NOT_FOUND",
    )

    response = TestClient(app).get(
        f"/usage/conversations/{_CONVERSATION_ID}",
        headers=_authorization(_USER_B),
    )

    assert response.status_code == 404
    assert usage_service.conversation_calls == [
        (_USER_B, _CONVERSATION_ID, ConversationUsageQuery())
    ]
    assert response.json() == {
        "success": False,
        "code": "CONVERSATION_NOT_FOUND",
        "message": "Conversation not found",
    }


@pytest.mark.parametrize(
    "path",
    [
        "/usage/dashboard",
        f"/usage/conversations/{_CONVERSATION_ID}",
    ],
)
def test_usage_ui_flag_hides_each_endpoint_without_changing_collection_flag(
    path: str,
    usage_service: UsageServiceSpy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_enabled = settings.model_usage_tracking_enabled
    monkeypatch.setattr(settings, "model_usage_ui_enabled", False)

    response = TestClient(app).get(path, headers=_authorization(_USER_A))

    assert response.status_code == 404
    assert response.json() == {
        "success": False,
        "code": "NOT_FOUND",
        "message": "Resource not found",
    }
    assert usage_service.dashboard_calls == []
    assert usage_service.conversation_calls == []
    assert settings.model_usage_tracking_enabled is tracking_enabled


def test_usage_ui_gate_is_evaluated_on_every_request(
    usage_service: UsageServiceSpy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    monkeypatch.setattr(settings, "model_usage_ui_enabled", False)

    hidden = client.get("/usage/dashboard", headers=_authorization(_USER_A))
    monkeypatch.setattr(settings, "model_usage_ui_enabled", True)
    visible = client.get("/usage/dashboard", headers=_authorization(_USER_A))

    assert hidden.status_code == 404
    assert visible.status_code == 200
    assert len(usage_service.dashboard_calls) == 1


def test_disabled_usage_ui_is_feature_hidden_before_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "model_usage_ui_enabled", False)

    response = TestClient(app).get("/usage/dashboard")

    assert response.status_code == 404
    assert response.json() == {
        "success": False,
        "code": "NOT_FOUND",
        "message": "Resource not found",
    }


def test_tracking_flag_does_not_hide_usage_read_api(
    usage_service: UsageServiceSpy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "model_usage_ui_enabled", True)
    monkeypatch.setattr(settings, "model_usage_tracking_enabled", False)

    response = TestClient(app).get(
        "/usage/dashboard",
        headers=_authorization(_USER_A),
    )

    assert response.status_code == 200
    assert len(usage_service.dashboard_calls) == 1


def test_usage_router_contract_has_no_ai_alias_or_identity_filter() -> None:
    schema = app.openapi()

    dashboard_operation = schema["paths"]["/usage/dashboard"]["get"]
    conversation_operation = schema["paths"]["/usage/conversations/{conversation_id}"]["get"]
    assert dashboard_operation["tags"] == ["usage"]
    assert conversation_operation["tags"] == ["usage"]
    assert {
        parameter["name"]
        for parameter in dashboard_operation["parameters"]
        if parameter["in"] == "query"
    } == {"from", "to", "bucket", "timezone", "conversationId"}
    assert {
        parameter["name"]
        for parameter in conversation_operation["parameters"]
        if parameter["in"] == "query"
    } == {"from", "to", "bucket", "timezone"}
    assert "/ai/usage/dashboard" not in schema["paths"]
    assert f"/ai/usage/conversations/{_CONVERSATION_ID}" not in schema["paths"]


@pytest.mark.parametrize(
    "params",
    [
        {"from": "2026-01-01T00:00:00+00:00"},
        {
            "from": "2026-01-01T00:00:00",
            "to": "2026-01-02T00:00:00",
        },
        {
            "from": "2026-01-01T00:00:00Z",
            "to": "2026-01-02T00:00:00Z",
        },
        {
            "from": "2026-01-01T00:00:01+00:00",
            "to": "2026-01-02T00:00:00+00:00",
        },
    ],
    ids=[
        "missing-paired-bound",
        "naive-no-offset",
        "z-is-not-numeric-offset",
        "non-zero-seconds",
    ],
)
@pytest.mark.parametrize(
    "path",
    ["/usage/dashboard", f"/usage/conversations/{_CONVERSATION_ID}"],
)
def test_query_model_validation_uses_invalid_input_envelope(
    path: str,
    params: dict[str, str],
    usage_service: UsageServiceSpy,
) -> None:
    response = TestClient(app).get(
        path,
        params=params,
        headers=_authorization(_USER_A),
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["success"] is False
    assert payload["code"] == "invalid_input"
    assert payload["message"] == "Invalid input"
    assert payload["error"]
    assert usage_service.dashboard_calls == []
    assert usage_service.conversation_calls == []


@pytest.mark.parametrize(
    ("params", "expected_code", "expected_message"),
    [
        (
            {"timezone": "Mars/Olympus"},
            "INVALID_USAGE_TIMEZONE",
            "Invalid IANA timezone: Mars/Olympus",
        ),
        (
            {
                "from": "2026-01-01T01:00:00+00:00",
                "to": "2026-01-02T00:00:00+00:00",
            },
            "INVALID_USAGE_ALIGNMENT",
            "Daily usage boundaries must be aligned to local midnight",
        ),
        (
            {
                "from": "2023-01-01T00:00:00+00:00",
                "to": "2026-01-01T00:00:00+00:00",
            },
            "USAGE_RANGE_TOO_LARGE",
            "Usage range cannot exceed two years (730 days)",
        ),
        (
            {
                "from": "2026-01-01T00:00:00+00:00",
                "to": "2026-02-02T00:00:00+00:00",
                "bucket": "hour",
            },
            "USAGE_HOURLY_RANGE_TOO_LARGE",
            "Hourly usage ranges cannot exceed 31 days",
        ),
    ],
    ids=["invalid-zone", "misaligned", "retention-cap", "hourly-cap"],
)
def test_service_query_validation_propagates_domain_422_envelope(
    params: dict[str, str],
    expected_code: str,
    expected_message: str,
    validating_usage_service: ModelUsageService,
) -> None:
    response = TestClient(app).get(
        "/usage/dashboard",
        params=params,
        headers=_authorization(_USER_A),
    )

    assert response.status_code == 422
    assert response.json() == {
        "success": False,
        "code": expected_code,
        "message": expected_message,
    }
