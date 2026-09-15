import pytest
from prometheus_client import CollectorRegistry

from app.ai.research_budget import ResearchBudget
from app.ai.web_research.contracts import ResearchRequest, ResearchScope
from app.ai.web_research.service import WebResearchService
from app.observability.web_research import WebResearchMetrics


def test_metric_labels_collapse_untrusted_values() -> None:
    metrics = WebResearchMetrics(CollectorRegistry())
    metrics.record(
        operation="https://secret.example/query?q=user",
        mode="tenant-123",
        outcome="raw exception text",
        visual_intent="private title",
    )

    rendered = metrics.render().decode()

    assert 'operation="other"' in rendered
    assert 'mode="other"' in rendered
    assert "secret.example" not in rendered
    assert "tenant-123" not in rendered


@pytest.mark.asyncio
async def test_session_records_real_operation_outcome() -> None:
    metrics = WebResearchMetrics(CollectorRegistry())
    session = WebResearchService(metrics=metrics).new_session(
        ResearchScope(conversation_id="c", user_id="u", logical_turn_id="t"),
        ResearchBudget(),
        mode="quick",
    )

    await session.search(ResearchRequest(query="release notes", objective="verify release"))

    rendered = metrics.render().decode()
    assert 'operation="search"' in rendered
    assert 'outcome="error"' in rendered
