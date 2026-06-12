"""Answer-producing prompts must state the inline media capability.

Guards against the model claiming it "cannot send images": every prompt that
produces user-facing answers carries one shared capability snippet.
"""

from app.ai import prompts

ANSWER_PROMPTS = (
    prompts.CHAT_SYSTEM_PROMPT,
    prompts.RAG_SYSTEM_PROMPT,
    prompts.AGENTIC_RAG_SYSTEM_PROMPT,
    prompts.SEARCH_SYSTEM_PROMPT,
    prompts.SEARCH_WITH_RESULTS_SYSTEM_PROMPT,
)


def test_snippet_defined_once_and_compact():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET
    assert "You CAN display images inline" in snippet
    assert len(snippet) < 1200, "media snippet must stay compact — do not bloat prompts"


def test_all_answer_prompts_carry_media_capability():
    for prompt in ANSWER_PROMPTS:
        assert "You CAN display images inline" in prompt
        assert "Never tell the user you cannot" in prompt


def test_non_answer_prompts_unchanged():
    for prompt in (
        prompts.ROUTER_SYSTEM_PROMPT,
        prompts.TITLE_GENERATION_PROMPT,
        prompts.PLANNING_EXECUTION_PROMPT,
        prompts.IMAGE_GENERATOR_SYSTEM_PROMPT,
    ):
        assert "You CAN display images inline" not in prompt
