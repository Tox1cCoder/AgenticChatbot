from __future__ import annotations

from types import SimpleNamespace

from app.ai.research_budget import ResearchBudget
from app.ai.web_research.policy import ResearchLimits, enforce_web_requirement
from app.core.config import Settings


def test_modes_bound_admission_but_not_search_count() -> None:
    assert ResearchLimits.for_mode("quick").max_sources == 5
    assert ResearchLimits.for_mode("quick").max_page_opens == 2
    assert ResearchLimits.for_mode("agentic").max_sources == 8
    assert ResearchLimits.for_mode("agentic").max_page_opens == 4
    assert not hasattr(ResearchLimits.for_mode("agentic"), "max_searches")


def test_gallery_only_raises_the_model_image_limit() -> None:
    normal = ResearchLimits.for_mode("agentic", visual_intent="figure")
    gallery = ResearchLimits.for_mode("agentic", visual_intent="gallery")

    assert normal.max_model_images == 4
    assert gallery.max_model_images == 6
    assert gallery.max_sources == normal.max_sources
    assert gallery.max_page_opens == normal.max_page_opens


def test_current_model_driven_search_contract_remains_intact() -> None:
    budget = ResearchBudget()
    for query in ("alpha release", "beta release", "gamma release", "delta release"):
        assert budget.reserve_search(query) is None
        budget.record_search(query, query)

    assert "research_max_search_calls_per_turn" not in Settings.model_fields


def test_explicit_url_and_current_fact_cannot_be_downgraded() -> None:
    decision = SimpleNamespace(
        requires_web=False,
        research_mode="none",
        model_copy=lambda *, update: SimpleNamespace(**update),
    )

    for message in ("Review https://example.test/page", "What is the current price?"):
        enforced = enforce_web_requirement(decision, message)
        assert enforced.requires_web is True
        assert enforced.research_mode == "quick"
