from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "docs" / "operations" / "conversation-compaction.md"

EXPECTED_ENV = {
    "CONVERSATION_SUMMARY_ENABLED": "true",
    "CONVERSATION_SUMMARY_PROVIDER": "gemini",
    "CONVERSATION_SUMMARY_MODEL": "gemini-2.5-flash",
    "CONVERSATION_SUMMARY_TRIGGER_MESSAGES": "60",
    "CONVERSATION_SUMMARY_TRIGGER_TOKENS": "18000",
    "CONVERSATION_SUMMARY_SOFT_CONTEXT_RATIO": "0.70",
    "CONVERSATION_SUMMARY_HARD_CONTEXT_RATIO": "0.85",
    "CONVERSATION_SUMMARY_KEEP_RECENT_TURNS": "4",
    "CONVERSATION_SUMMARY_MAX_TOKENS": "1500",
    "CONVERSATION_SUMMARY_TIMEOUT_SECONDS": "30",
    "CONVERSATION_SUMMARY_MAX_ATTEMPTS": "5",
    "CONVERSATION_SUMMARY_LEASE_SECONDS": "120",
    "CONVERSATION_SUMMARY_RETRY_BASE_SECONDS": "5",
    "CONVERSATION_SUMMARY_RETRY_MAX_SECONDS": "900",
    "CONVERSATION_SUMMARY_RECONCILE_SECONDS": "60",
    "CONVERSATION_SUMMARY_SAFETY_MARGIN_TOKENS": "1024",
    "CONVERSATION_SUMMARY_DEFAULT_RESERVED_OUTPUT_TOKENS": "4096",
}


def test_example_environment_and_runbook_defaults_match() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    runbook = OPS.read_text(encoding="utf-8")

    for name, default in EXPECTED_ENV.items():
        assert f"{name}={default}" in example
        assert f"`{name}` | `{default}`" in runbook


def test_runbook_covers_deployment_and_operations_contract() -> None:
    runbook = OPS.read_text(encoding="utf-8")
    required = (
        "celery -A app.workers.celery_app:celery_app worker -Q summary",
        "celery -A app.workers.celery_app:celery_app beat",
        "alembic upgrade x1y2z3a4b5c6",
        "alembic downgrade w7x8y9z0a1b2",
        "backfill_conversation_summaries_task",
        "/health/conversation-compaction",
        "/metrics/conversation-compaction",
        "conversation_compaction_oldest_actionable_age_seconds",
        "conversation_compaction_sequence_lag",
        "credential rotation",
        "stable model",
        "Rollback limitations",
        "Rollout checklist",
    )
    for phrase in required:
        assert phrase in runbook


def test_legacy_names_are_confined_to_release_mapping() -> None:
    runbook = OPS.read_text(encoding="utf-8")
    before, mapping = runbook.split("<!-- legacy-map:start -->", maxsplit=1)
    mapping, after = mapping.split("<!-- legacy-map:end -->", maxsplit=1)

    legacy_prefixes = (
        "MEMORY_" + "SUMMARY_",
        "SUMMARIZATION" + "_",
        "ENABLE_" + "SUMMARIZATION",
    )
    for prefix in legacy_prefixes:
        assert prefix not in before
        assert prefix not in after
        assert prefix in mapping
