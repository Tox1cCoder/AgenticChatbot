"""Executable frontend contract checks for user-scoped model usage."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args
from unittest.mock import MagicMock
from uuid import uuid4

from app.ai.model_context import resolve_model_context_window
from app.api.ai_sdk import AISDKUIMessage, get_conversation_messages_ai_sdk
from app.repositories.utils.pagination import PaginationMeta
from app.schemas.model_usage import (
    ContextWindowMetadata,
    ConversationUsage,
    ConversationUsageResponse,
    UsageCapabilities,
    UsageDashboard,
)
from app.schemas.pagination import MessagePaginationParams
from app.schemas.responses import ApiResponse
from app.services.model_usage_service import (
    USAGE_CONVERSATION_DEFAULT_DAYS,
    USAGE_DASHBOARD_DEFAULT_DAYS,
)

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "plans" / "TOKEN_USAGE_AI_SDK_FE_CONTRACT.md"


def _contract() -> str:
    return CONTRACT_PATH.read_text(encoding="utf-8")


def _json_example(name: str) -> dict[str, Any]:
    match = re.search(
        rf"<!-- example:{re.escape(name)} -->\s*```json\s*(.*?)\s*```",
        _contract(),
        re.DOTALL,
    )
    assert match is not None, f"missing executable JSON example: {name}"
    value = json.loads(match.group(1))
    assert isinstance(value, dict)
    return value


def _typescript_context_fields() -> tuple[set[str], set[str]]:
    match = re.search(
        r"type ContextWindowMetadata\s*=\s*\{(.*?)\};",
        _contract(),
        re.DOTALL,
    )
    assert match is not None
    fields = re.findall(r"\b([a-z_][a-z0-9_]*)(\?)?\s*:", match.group(1))
    required = {name for name, optional in fields if not optional}
    optional = {name for name, optional in fields if optional}
    return required, optional


def test_dashboard_example_round_trips_through_real_response_schema() -> None:
    example = _json_example("usage-dashboard-response")

    parsed = ApiResponse[UsageDashboard].model_validate(example)

    assert parsed.model_dump(mode="json", by_alias=True) == example


def test_capability_example_round_trips_through_real_response_schema() -> None:
    example = _json_example("usage-capabilities-response")

    parsed = ApiResponse[UsageCapabilities].model_validate(example)

    assert parsed.model_dump(mode="json", by_alias=True) == example


def test_conversation_example_round_trips_through_real_response_schema() -> None:
    example = _json_example("conversation-usage-response")

    parsed = ApiResponse[ConversationUsageResponse].model_validate(example)

    assert parsed.model_dump(mode="json", by_alias=True) == example


def test_context_example_is_valid_ai_sdk_metadata_and_preserves_future_fields() -> None:
    context_window = _json_example("context-window-metadata")

    assert (
        ContextWindowMetadata.model_validate(context_window).model_dump(exclude_unset=True)
        == context_window
    )

    message = AISDKUIMessage.model_validate(
        {
            "id": "message-1",
            "role": "assistant",
            "content": "Done",
            "parts": [{"type": "text", "text": "Done"}],
            "metadata": {
                "context_window": context_window,
                "futureMetadata": {"must": "be ignored safely"},
            },
        }
    )

    assert message.metadata == {
        "context_window": context_window,
        "futureMetadata": {"must": "be ignored safely"},
    }


def test_static_registry_context_is_valid_without_dynamic_usage_fields() -> None:
    static_context = resolve_model_context_window("openai", "gpt-4o").to_dict()
    static_context["future_context_field"] = "ignored"

    parsed = ContextWindowMetadata.model_validate(static_context)
    serialized = parsed.model_dump(exclude_unset=True)

    assert serialized == {
        "provider": "openai",
        "model": "gpt-4o",
        "context_window_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "limit_type": "shared_context",
        "source": "registry",
        "known": True,
    }


def test_context_typescript_required_and_optional_fields_match_schema() -> None:
    json_schema = ContextWindowMetadata.model_json_schema()
    schema_fields = set(json_schema["properties"])
    schema_required = set(json_schema.get("required", []))
    ts_required, ts_optional = _typescript_context_fields()

    assert ts_required == schema_required
    assert ts_optional == schema_fields - schema_required
    assert ContextWindowMetadata in get_args(
        ConversationUsage.model_fields["latest_context_window"].annotation
    )


def test_real_usage_response_schemas_ignore_unknown_future_fields() -> None:
    dashboard = copy.deepcopy(_json_example("usage-dashboard-response"))
    dashboard["futureEnvelopeField"] = True
    dashboard["data"]["futureDashboardField"] = {"value": 1}
    dashboard["data"]["totals"]["futureTotalField"] = 1

    parsed = ApiResponse[UsageDashboard].model_validate(dashboard)

    assert parsed.data is not None
    assert parsed.data.totals.total_tokens == 1500
    assert "futureDashboardField" not in parsed.data.model_dump(by_alias=True)


def test_history_path_preserves_complete_additive_context_metadata() -> None:
    context_window = _json_example("context-window-metadata")
    message_service = MagicMock()
    message_service.get_conversation_messages.return_value = SimpleNamespace(
        items=[
            SimpleNamespace(
                id=uuid4(),
                sender=2,
                content="Done",
                created_at=datetime(2026, 7, 2, tzinfo=UTC),
                message_metadata={
                    "context_window": context_window,
                    "futureMetadata": True,
                },
            )
        ],
        meta=PaginationMeta.calculate(1, 100, 1),
    )

    response = asyncio.run(
        get_conversation_messages_ai_sdk(
            uuid4(),
            message_service,
            uuid4(),
            MessagePaginationParams(),
        )
    )

    assert response.data is not None
    assert response.data.messages[0].metadata == {
        "context_window": context_window,
        "futureMetadata": True,
    }


def test_contract_covers_endpoint_security_ranges_and_errors() -> None:
    contract = _contract()
    required_phrases = (
        "GET /usage/dashboard",
        "GET /usage/conversations/{conversationId}",
        "Authorization: Bearer <JWT>",
        "inclusive",
        "exclusive",
        "31 days",
        "730 days",
        "401",
        "404",
        "422",
        "conversationId",
        "from",
        "to",
        "bucket",
        "timezone",
    )

    for phrase in required_phrases:
        assert phrase in contract


def test_contract_default_ranges_track_service_defaults() -> None:
    contract = _contract()

    assert f"dashboard defaults to {USAGE_DASHBOARD_DEFAULT_DAYS} local calendar days" in contract
    assert (
        f"conversation defaults to the retained {USAGE_CONVERSATION_DEFAULT_DAYS} days" in contract
    )


def test_contract_covers_types_visualization_refresh_and_ui_states() -> None:
    contract = _contract()
    required_phrases = (
        "type UsageTotals",
        "type UsageSeriesPoint",
        "type UsageBreakdownItem",
        "type UsageCoverage",
        "type ConversationUsageItem",
        "type UsageDashboard",
        "type ConversationUsage",
        "type ContextWindowMetadata",
        "stacked input/output",
        "outcome rate",
        "provider/model/operation/agent",
        "top conversations",
        "filter change",
        "AI SDK `finish`",
        "Never poll during generation",
        "Loading",
        "Empty",
        "Error",
        "unknown future fields",
        "visual fill",
        "raw ratio",
        "unknown denominator",
    )

    for phrase in required_phrases:
        assert phrase in contract


def test_contract_covers_capability_gating_and_message_gauge_authority() -> None:
    contract = _contract()
    required_phrases = (
        "GET /usage/capabilities",
        "type UsageCapabilities",
        "latest non-deleted assistant message",
        "helper calls cannot replace",
        "No new SSE event",
        "AbortController",
        "separate_io",
        "unknown denominator",
        "Frontend acceptance checklist",
    )

    for phrase in required_phrases:
        assert phrase in contract


def test_contract_does_not_put_dashboard_aggregates_in_message_metadata() -> None:
    context_window = _json_example("context-window-metadata")

    assert "totals" not in context_window
    assert "series" not in context_window
    assert "topConversations" not in context_window
    assert "byProvider" not in context_window


def test_dashboard_outcomes_partition_every_total() -> None:
    dashboard = _json_example("usage-dashboard-response")["data"]

    for field, expected in dashboard["totals"].items():
        assert sum(item["totals"][field] for item in dashboard["outcomes"]) == expected
