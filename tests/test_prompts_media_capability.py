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


def test_media_guidance_describes_provider_native_selection_without_false_assurance():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "provider-native" in snippet
    assert "selected" in snippet
    assert "visual " + "verifier" not in snippet
    assert "images are " + "verified" not in snippet
    assert "verified " + "image" not in snippet


def test_media_guidance_scopes_tavily_controls_to_recency_and_finance():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "current events" in snippet
    assert 'topic="news"' in snippet
    assert "time_range" in snippet
    assert 'topic="finance"' in snippet
    assert "general factual research" in snippet


def test_snippet_has_no_hardcoded_visual_topic_list():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()
    for word in TAXONOMY_WORDS:
        assert word not in snippet, f"snippet must not hardcode visual topics: found {word!r}"


def test_all_answer_prompts_carry_media_capability():
    for prompt in ANSWER_PROMPTS:
        assert "Media and visuals:" in prompt
        # The exact header the inventory block emits, so the reference resolves.
        assert "AVAILABLE RICH ITEMS" in prompt


def test_rag_prompts_omit_research_controls_they_cannot_use():
    """``web_research`` is internal and bound only for the chat and search agents,
    and internal tools never surface through ``tool_search`` — so naming it in a
    RAG prompt advertises a tool that agent can never call."""
    for prompt in (prompts.RAG_SYSTEM_PROMPT, prompts.AGENTIC_RAG_SYSTEM_PROMPT):
        assert "web_research" not in prompt
        assert "skip_images" not in prompt
        assert "image_intent" not in prompt


def test_research_capable_prompts_keep_research_controls():
    for prompt in (
        prompts.CHAT_SYSTEM_PROMPT,
        prompts.SEARCH_SYSTEM_PROMPT,
        prompts.SEARCH_WITH_RESULTS_SYSTEM_PROMPT,
    ):
        assert "web_research" in prompt
        assert "skip_images" in prompt


def test_placement_mechanics_reach_every_answer_prompt():
    """Widgets are rich items too, so an agent with no image path still needs the
    marker contract to place what it built."""
    for prompt in ANSWER_PROMPTS:
        assert "<!--rich:<id>-->" in prompt
        assert "Never invent an ID" in prompt


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
    returns "" when the turn produced no candidates, so without this the model
    is told the marker syntax with no inventory and no prohibition.
    """
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "never invent an id" in snippet


def test_snippet_says_what_an_absent_inventory_means():
    """Absence of an inventory must be stated, not left to inference."""
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "available rich items" in snippet
    assert "no rich items" in snippet


def test_snippet_forbids_claiming_the_assistant_cannot_show_images():
    """An empty turn must not read as a fabricated capability limit.

    The model has to be told what "no image this turn" means for the user, not
    only what it means for the marker.
    """
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "unable to show images" in snippet
    assert "no suitable one" in snippet
