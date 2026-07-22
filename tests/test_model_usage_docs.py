"""Documentation contracts for operating per-user model-usage analytics."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RUNBOOK = ROOT / "docs" / "operations" / "model-usage-analytics.md"
POSTMAN = ROOT / "Chatbot API.postman_collection.json"

MODEL_USAGE_CONTROLS = (
    "MODEL_USAGE_TRACKING_ENABLED",
    "MODEL_USAGE_UI_ENABLED",
    "MODEL_USAGE_RAW_RETENTION_DAYS",
    "MODEL_USAGE_ROLLUP_RETENTION_DAYS",
    "MODEL_USAGE_RECONCILE_MINUTES",
    "MODEL_USAGE_RECONCILE_CHUNK_MINUTES",
    "MODEL_USAGE_CLEANUP_BATCH_SIZE",
    "MODEL_USAGE_RETRY_MAX_ATTEMPTS",
    "MODEL_USAGE_RETRY_BASE_SECONDS",
    "MODEL_USAGE_USER_HASH_SECRET",
    "MODEL_USAGE_HEALTH_LOOKBACK_MINUTES",
    "MODEL_USAGE_HEALTH_UNATTRIBUTED_DEGRADED_RATIO",
    "MODEL_USAGE_HEALTH_ROLLUP_LAG_DEGRADED_MINUTES",
    "MODEL_USAGE_HEALTH_ROLLUP_LAG_UNHEALTHY_MINUTES",
    "MODEL_USAGE_HEALTH_FAILURE_WINDOW_SECONDS",
    "MODEL_USAGE_FAILURE_STORE_TTL_SECONDS",
    "MODEL_USAGE_FAILURE_STORE_TIMEOUT_SECONDS",
)


def _normalized(path: Path) -> str:
    assert path.is_file(), f"missing documentation file: {path.relative_to(ROOT)}"
    return " ".join(path.read_text(encoding="utf-8").split())


def _markdown_section(document: str, heading: str) -> str:
    """Return one Markdown section, stopping at the next equal/higher heading."""
    lines = document.splitlines()
    start = lines.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    end = len(lines)
    for index in range(start + 1, len(lines)):
        candidate = lines[index]
        candidate_level = len(candidate) - len(candidate.lstrip("#"))
        if candidate_level and candidate_level <= level and candidate.startswith("#"):
            end = index
            break
    return "\n".join(lines[start:end])


def _postman_request_url(request: dict) -> tuple[str, dict[str, list[str]]]:
    raw = request["url"]["raw"]
    parsed = urlsplit(raw.replace("{{HOST}}:{{PORT}}", "http://postman.example", 1))
    return parsed.path, parse_qs(parsed.query, strict_parsing=True)


def _aware_datetime(value: str) -> datetime:
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2}",
        value,
    )
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
    return parsed


def test_readme_advertises_the_complete_model_usage_contract() -> None:
    readme = _normalized(README)

    for name in MODEL_USAGE_CONTROLS:
        assert name in readme
    for route in (
        "/usage/capabilities",
        "/usage/dashboard",
        "/usage/conversations/{conversation_id}",
        "/health/model-usage",
        "/metrics/model-usage",
    ):
        assert route in readme
    assert "90 days of raw events" in readme
    assert "730 days of minute rollups" in readme
    assert "does not calculate or track monetary cost" in readme
    assert "docs/operations/model-usage-analytics.md" in readme
    assert "plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md" in readme


def test_runbook_covers_deployment_and_maintenance() -> None:
    runbook = _normalized(RUNBOOK)

    required = (
        "Schema-first deployment",
        "y2z3a4b5c6d7",
        "z3a4b5c6d7e8",
        "a4b5c6d7e8f9",
        "b5c6d7e8f9a0",
        "Drain old API/worker approval writers",
        "persists `DecisionType.value`",
        "No historical backfill",
        "MODEL_USAGE_TRACKING_ENABLED",
        "MODEL_USAGE_UI_ENABLED",
        "complete UTC minutes",
        "advisory lock",
        "minute `:15`",
        "`03:40`",
        "`summary` queue",
        "90 days",
        "730 days",
        "5,000",
        "strictly shorter than raw retention",
    )
    for phrase in required:
        assert phrase in runbook


def test_runbook_requires_safe_rollback_order_and_no_schema_normal_path() -> None:
    document = RUNBOOK.read_text(encoding="utf-8")
    rollback = _markdown_section(document, "## Rollback")
    rollback_text = " ".join(rollback.split())

    ordered_steps = (
        "disable new collection on API producers",
        "drain failed-write retries",
        "using the current release's migration artifacts",
        ".venv\\Scripts\\python.exe -m alembic downgrade z3a4b5c6d7e8",
        "Deploy the previous application",
    )
    for step in ordered_steps:
        assert step in rollback_text
    positions = [rollback_text.index(step) for step in ordered_steps]
    assert positions == sorted(positions)
    assert "normal rollback leaves the additive usage tables and timestamp indexes" in rollback_text
    assert "The normal rollback path performs no schema action" in rollback_text
    assert "optional index-only downgrade" in rollback_text
    assert "b5c6d7e8f9a0`'s safe no-op downgrade" in rollback_text
    assert "preserving repaired tool-approval schema and data" in rollback_text
    assert "isolated scratch database" in rollback_text
    assert "coordinated maintenance" in rollback_text


def test_runbook_covers_observability_investigation_and_replay() -> None:
    runbook = _normalized(RUNBOOK)

    required = (
        "/health/model-usage",
        "/metrics/model-usage",
        "aggregate-only",
        "shared Redis failure store",
        "availability",
        "freshness",
        "unattributed rate",
        "rollup lag",
        "Investigation",
        "event_key",
        "operation_id:attempt",
        "idempotent",
        "replay",
    )
    for phrase in required:
        assert phrase in runbook


def test_runbook_covers_api_privacy_and_frontend_contracts() -> None:
    runbook = _normalized(RUNBOOK)

    required = (
        "/usage/capabilities",
        "/usage/dashboard",
        "/usage/conversations/{conversation_id}",
        "/api/usage/capabilities",
        "/api/usage/dashboard",
        "/api/usage/conversations/{conversation_id}",
        "31 days",
        "730 days",
        "IANA timezone",
        "numeric UTC offset",
        "DST",
        "No prompt or response content",
        "user-scoped identifiers",
        "cascades",
        "No monetary cost tracking",
        "plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md",
    )
    for phrase in required:
        assert phrase in runbook


def test_runbook_documents_account_deletion_across_all_data_stores() -> None:
    document = RUNBOOK.read_text(encoding="utf-8")
    deletion = _markdown_section(document, "## Data classification and deletion")
    deletion_text = " ".join(deletion.split())

    required = (
        "user, conversation, message, and document IDs",
        "normalized usage",
        "no analytics-specific expiry or TTL",
        "broker visibility timeout is not a privacy TTL",
        "no dedicated dead-letter queue",
        "drain or purge",
        "ModelUsageReferenceError",
        "cannot recreate the user or tenant data",
        "LangSmith",
        "external traces",
        "provider project/API",
        "keyed user hash or correlation",
        "PostgreSQL",
        "broker and any deployment-managed DLQ",
    )
    for phrase in required:
        assert phrase in deletion_text

    verification = _markdown_section(deletion, "### Account-deletion verification")
    verification_text = " ".join(verification.split())
    assert "`model_usage_events`" in verification_text
    assert "`model_usage_minute`" in verification_text
    assert "broker and any deployment-managed DLQ" in verification_text
    assert "LangSmith" in verification_text


def test_postman_timestamp_parser_requires_a_numeric_offset() -> None:
    with pytest.raises(AssertionError):
        _aware_datetime("2026-07-15T00:00:00Z")


def test_postman_has_authenticated_dashboard_and_conversation_examples() -> None:
    collection = json.loads(POSTMAN.read_text(encoding="utf-8"))
    folder = next(item for item in collection["item"] if item["name"] == "Model Usage Analytics")
    requests = {item["name"]: item["request"] for item in folder["item"]}

    dashboard = requests["Dashboard - Bangkok Daily"]
    assert dashboard["method"] == "GET"
    dashboard_headers = {header["key"].lower(): header["value"] for header in dashboard["header"]}
    assert dashboard_headers["authorization"] == "Bearer {{ACCESS_TOKEN}}"
    dashboard_path, dashboard_query = _postman_request_url(dashboard)
    assert dashboard_path == "/usage/dashboard"
    assert dashboard_query.keys() == {"from", "to", "bucket", "timezone"}
    assert dashboard_query["bucket"] == ["day"]
    assert dashboard_query["timezone"] == ["Asia/Bangkok"]
    dashboard_from = _aware_datetime(dashboard_query["from"][0])
    dashboard_to = _aware_datetime(dashboard_query["to"][0])
    bangkok = ZoneInfo("Asia/Bangkok")
    dashboard_from_local = dashboard_from.astimezone(bangkok)
    dashboard_to_local = dashboard_to.astimezone(bangkok)
    assert dashboard_from_local.time().isoformat() == "00:00:00"
    assert dashboard_to_local.time().isoformat() == "00:00:00"
    assert dashboard_from.utcoffset() == dashboard_from_local.utcoffset()
    assert dashboard_to.utcoffset() == dashboard_to_local.utcoffset()
    assert dashboard_to_local.date() - dashboard_from_local.date() == timedelta(days=7)

    conversation = requests["Conversation Usage - Hourly 7 Days"]
    assert conversation["method"] == "GET"
    conversation_headers = {
        header["key"].lower(): header["value"] for header in conversation["header"]
    }
    assert conversation_headers["authorization"] == "Bearer {{ACCESS_TOKEN}}"
    conversation_path, conversation_query = _postman_request_url(conversation)
    assert conversation_path == "/usage/conversations/{{conversation_id}}"
    assert conversation_query.keys() == {"from", "to", "bucket", "timezone"}
    assert conversation_query["bucket"] == ["hour"]
    assert conversation_query["timezone"] == ["Asia/Bangkok"]
    conversation_from = _aware_datetime(conversation_query["from"][0])
    conversation_to = _aware_datetime(conversation_query["to"][0])
    assert conversation_to - conversation_from == timedelta(days=7)
    assert (
        conversation_from.minute == conversation_from.second == conversation_from.microsecond == 0
    )
    assert conversation_to.minute == conversation_to.second == conversation_to.microsecond == 0
