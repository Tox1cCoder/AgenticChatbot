"""Stable visual-grounding instructions shared by answer agents."""

from app.ai import prompts
from app.ai.web_tools import WEB_SEARCH_DESCRIPTION

WEB_CAPABLE_PROMPTS = (
    prompts.CHAT_SYSTEM_PROMPT,
    prompts.SEARCH_SYSTEM_PROMPT,
    prompts.SEARCH_WITH_RESULTS_SYSTEM_PROMPT,
)
ANSWER_PROMPTS = (
    *WEB_CAPABLE_PROMPTS,
    prompts.RAG_SYSTEM_PROMPT,
    prompts.AGENTIC_RAG_SYSTEM_PROMPT,
)


def test_media_snippet_is_compact_and_has_one_image_selection_contract() -> None:
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET

    assert len(snippet) < 2100
    assert "[[image:I#]]" in snippet
    assert "actual images" in snippet
    assert "No image token means no web image" in snippet
    assert "image_search" not in snippet


def test_media_snippet_requires_visible_inspection_and_allows_zero_images() -> None:
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "inspect the visible pixels" in snippet
    assert "selecting none is normal" in snippet
    assert "never invent an id or url" in snippet


def test_web_search_description_owns_visual_query_mechanics() -> None:
    description = WEB_SEARCH_DESCRIPTION.lower()

    assert "visual_intent" in description
    assert "image_query" in description
    assert "part, screen" in description
    assert "version or year" in description
    assert "freshness='recent'" in description
    assert "freshness='as_of'" in description


def test_web_capable_prompts_name_integrated_search_not_legacy_image_search() -> None:
    for prompt in WEB_CAPABLE_PROMPTS:
        assert "web_search" in prompt
        assert "image_search" not in prompt


def test_rag_prompts_do_not_advertise_unbound_web_tools() -> None:
    for prompt in (prompts.RAG_SYSTEM_PROMPT, prompts.AGENTIC_RAG_SYSTEM_PROMPT):
        for tool_name in ("web_research", "image_search", "web_search", "web_open"):
            assert tool_name not in prompt


def test_widget_marker_mechanics_still_reach_every_answer_prompt() -> None:
    for prompt in ANSWER_PROMPTS:
        assert "<!--rich:<id>-->" in prompt
        assert "Never invent an ID" in prompt


def test_non_answer_prompts_stay_free_of_media_instructions() -> None:
    for prompt in (
        prompts.ROUTER_SYSTEM_PROMPT,
        prompts.TITLE_GENERATION_PROMPT,
        prompts.PLANNING_EXECUTION_PROMPT,
        prompts.IMAGE_GENERATOR_SYSTEM_PROMPT,
    ):
        assert "Media and visuals:" not in prompt
