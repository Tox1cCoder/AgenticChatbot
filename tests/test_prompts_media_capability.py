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
    """The snippet must name where IDs come from, not just say "available".

    "use only available IDs" left the source of an ID unstated, so on a turn
    with no inventory the instruction had no referent at all.
    """
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()
    assert "available rich items" in snippet
    assert "never invent" in snippet


def test_media_guidance_describes_automatic_visual_enrichment_without_taxonomy():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "automatically considers" in snippet
    assert "skip_images" in snippet
    assert "product, device" not in snippet
    assert "code, math" not in snippet


def test_snippet_has_no_hardcoded_visual_topic_list():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()
    for word in TAXONOMY_WORDS:
        assert word not in snippet, f"snippet must not hardcode visual topics: found {word!r}"


def test_all_answer_prompts_carry_media_capability():
    for prompt in ANSWER_PROMPTS:
        assert "Media and visuals:" in prompt
        # The exact header the inventory block emits, so the reference resolves.
        assert "AVAILABLE RICH ITEMS" in prompt


def test_non_answer_prompts_unchanged():
    for prompt in (
        prompts.ROUTER_SYSTEM_PROMPT,
        prompts.TITLE_GENERATION_PROMPT,
        prompts.PLANNING_EXECUTION_PROMPT,
        prompts.IMAGE_GENERATOR_SYSTEM_PROMPT,
    ):
        assert "Media and visuals:" not in prompt


def test_snippet_forbids_inventing_an_ID_not_only_a_URL():
    """The standing snippet is the only marker guidance on a no-inventory turn.

    ``build_rich_response_guidance`` — which carries "Never invent an ID" —
    returns "" when the turn produced no candidates, which the design says is
    the common case. So on those turns the model was told the marker syntax
    with no inventory and no prohibition, and invented one
    (`<!--rich:widget:t1_roster_2026-->`), which rendered as
    "rich item ... is unavailable".
    """
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "never invent an id" in snippet


def test_snippet_says_what_an_absent_inventory_means():
    """Absence of an inventory must be stated, not left to inference."""
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "available rich items" in snippet
    assert "no rich items" in snippet


def test_snippet_forbids_claiming_the_assistant_cannot_show_images():
    """The trace's actual failure text was a fabricated capability limit.

    With no image available the model wrote that it "cannot send image files
    through this chat window" — inventing a limitation of a system that renders
    images fine. It needed to be told what an empty turn means for the user,
    not only what it means for the marker.
    """
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "unable to show images" in snippet
    assert "no suitable one" in snippet
