"""Documentation contracts for operating per-user model-usage analytics."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_runbook_covers_deployment_maintenance_and_rollback() -> None:
    runbook = _normalized(RUNBOOK)

    required = (
        "Schema-first deployment",
        "y2z3a4b5c6d7",
        "z3a4b5c6d7e8",
        "a4b5c6d7e8f9",
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
        "downgrade z3a4b5c6d7e8",
        "isolated scratch database",
        "disable collection",
        "drain failed-write retries",
        "coordinated maintenance",
    )
    for phrase in required:
        assert phrase in runbook


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


def test_postman_has_authenticated_dashboard_and_conversation_examples() -> None:
    collection = json.loads(POSTMAN.read_text(encoding="utf-8"))
    folder = next(item for item in collection["item"] if item["name"] == "Model Usage Analytics")
    requests = {item["name"]: item["request"] for item in folder["item"]}

    dashboard = requests["Dashboard - Bangkok Daily"]
    assert dashboard["method"] == "GET"
    assert dashboard["header"] == [
        {
            "key": "Authorization",
            "value": "Bearer {{ACCESS_TOKEN}}",
            "type": "text",
        }
    ]
    dashboard_url = dashboard["url"]["raw"]
    assert "/usage/dashboard?" in dashboard_url
    assert "bucket=day" in dashboard_url
    assert "timezone=Asia/Bangkok" in dashboard_url
    assert "%2B07:00" in dashboard_url

    conversation = requests["Conversation Usage - Hourly 7 Days"]
    assert conversation["method"] == "GET"
    assert conversation["header"] == dashboard["header"]
    conversation_url = conversation["url"]["raw"]
    assert "/usage/conversations/{{conversation_id}}?" in conversation_url
    assert "bucket=hour" in conversation_url
    assert "timezone=Asia/Bangkok" in conversation_url
    assert "from=2026-07-15T00:00:00%2B07:00" in conversation_url
    assert "to=2026-07-22T00:00:00%2B07:00" in conversation_url
