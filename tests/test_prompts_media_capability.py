"""The shared media snippet must stay compact, capability-oriented, and free of
hardcoded visual-topic routing (image_search.md Phase 5).

It tells the model it can place provided rich items inline using available IDs
only, to call an image/search tool when a visual would materially help, and not
to add decorative media. It must not enumerate a visual-topic taxonomy.
"""

from app.ai import prompts

ANSWER_PROMPTS = (
    prompts.CHAT_SYSTEM_PROMPT,
    prompts.RAG_SYSTEM_PROMPT,
    prompts.AGENTIC_RAG_SYSTEM_PROMPT,
    prompts.SEARCH_SYSTEM_PROMPT,
    prompts.SEARCH_WITH_RESULTS_SYSTEM_PROMPT,
)

# Topic words that would signal a reintroduced hardcoded taxonomy.
TAXONOMY_WORDS = ("architecture", "fashion", "cuisine", "brutalist", "gothic")


def test_snippet_defined_once_and_compact():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET
    assert "Media and visuals:" in snippet
    # Raised from 1300 when the snippet was rewritten to describe the
    # server-orchestrated web_research tool: image_intent="gallery", the
    # verified-before-injection guarantee, and the gallery item-count rule
    # (vision-verified-image-injection Task 6).
    assert len(snippet) < 1700, "media snippet must stay compact — do not bloat prompts"


def test_snippet_uses_available_ids_only_and_forbids_invention():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()
    assert "available ids" in snippet
    assert "never invent" in snippet


def test_snippet_has_no_hardcoded_visual_topic_list():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()
    for word in TAXONOMY_WORDS:
        assert word not in snippet, f"snippet must not hardcode visual topics: found {word!r}"


def test_all_answer_prompts_carry_media_capability():
    for prompt in ANSWER_PROMPTS:
        assert "Media and visuals:" in prompt
        assert "available IDs" in prompt


def test_non_answer_prompts_unchanged():
    for prompt in (
        prompts.ROUTER_SYSTEM_PROMPT,
        prompts.TITLE_GENERATION_PROMPT,
        prompts.PLANNING_EXECUTION_PROMPT,
        prompts.IMAGE_GENERATOR_SYSTEM_PROMPT,
    ):
        assert "Media and visuals:" not in prompt
