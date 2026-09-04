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
    """The bound guards against unbounded growth, not against content.

    It was 1700 while the snippet was purely mechanical. It has since taken on
    four load-bearing behaviours: image_search is the sole path an image can
    take, the recency qualifier that is the only way to ask the provider for a
    current picture, the form word that decides whether "what is X" returns
    identity art or an in-use shot, and one figure per call.

    The time_range and intent bullets have since moved out to the image_search
    description, which is where argument mechanics belong: both
    texts sit in context on every call, and the description is what the model
    reads while choosing arguments. The prompt keeps what is behavioural —
    when a visual is worth having, which form to ask for, and how to place it.
    """
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET
    assert "Media and visuals:" in snippet
    assert len(snippet) < 2100, "media snippet must stay compact — do not bloat prompts"


def test_snippet_uses_available_ids_only_and_forbids_invention():
    """The snippet must name where IDs come from, not just say "available".

    "use only available IDs" left the source of an ID unstated, so on a turn
    with no inventory the instruction had no referent at all.
    """
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()
    assert "available rich items" in snippet
    assert "never invent" in snippet


def test_media_guidance_describes_visual_acquisition_without_taxonomy():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "image_search" in snippet
    assert "only path an image can take" in snippet
    assert "product, device" not in snippet
    assert "code, math" not in snippet


def test_media_guidance_says_a_web_search_does_not_fetch_a_picture():
    """Splitting the tools created a new way to be wrong: assuming a text
    search already looked for an image, and never calling for one."""
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "not something a `web_search` does for you" in snippet


def test_media_guidance_describes_provider_native_selection_without_false_assurance():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "provider-native" in snippet
    assert "selected" in snippet
    assert "visual " + "verifier" not in snippet
    assert "images are " + "verified" not in snippet
    assert "verified " + "image" not in snippet


def test_freshness_controls_live_in_the_search_tool_description():
    """Argument mechanics live in the tool description, not the system prompt.

    Both texts sit in context on every call, and the description is what the
    model reads while choosing arguments — so duplicating them into the prompt
    bought nothing but length.
    """
    from app.ai.web_tools import WEB_SEARCH_DESCRIPTION

    description = WEB_SEARCH_DESCRIPTION.lower()

    assert "freshness='recent'" in description
    assert "freshness='as_of'" in description
    assert "never guess" in description
    assert "objective" in description


def test_snippet_has_no_hardcoded_visual_topic_list():
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()
    for word in TAXONOMY_WORDS:
        assert word not in snippet, f"snippet must not hardcode visual topics: found {word!r}"


def test_all_answer_prompts_carry_media_capability():
    for prompt in ANSWER_PROMPTS:
        assert "Media and visuals:" in prompt
        # The exact header the inventory block emits, so the reference resolves.
        assert "AVAILABLE RICH ITEMS" in prompt


def _query_guidance() -> str:
    """Everything the model reads while writing an image query.

    Query construction is argument mechanics, so it lives in the tool
    description rather than the system prompt — see
    test_snippet_defined_once_and_compact.
    """
    from app.ai.web_tools import IMAGE_SEARCH_DESCRIPTION

    return IMAGE_SEARCH_DESCRIPTION.lower()


def test_image_query_guidance_ties_the_form_word_to_what_was_asked():
    """ "What is X" was returning in-use screenshots because the form words on
    offer were all depiction words (photo, diagram, map). A thing's identity
    image — its logo, key art, cover — is what answers "what is this", and the
    query has to name it. Keyed to the kind of question, not to the subject.
    """
    guidance = _query_guidance()

    assert "logo" in guidance
    assert "key art" in guidance


def test_image_query_guidance_asks_for_the_part_the_answer_is_about():
    """Naming the product returns the most-photographed view of it — a press
    shot of a whole scooter for a question about its warning lights. The image
    has to show the thing the reader must look at, which is usually a component,
    a screen or a panel rather than the product that contains it.
    """
    guidance = _query_guidance()

    assert "part" in guidance
    assert "look at" in guidance


def test_image_query_guidance_requires_resolving_what_the_user_referred_to():
    """ "my scooter", "this game", "it" cannot be searched. The named thing has
    to be recovered from the conversation before it becomes a query."""
    guidance = _query_guidance()

    assert "my scooter" in guidance or "pronoun" in guidance
    assert "conversation" in guidance


def test_image_query_guidance_covers_the_language_the_subject_lives_in():
    """A scooter sold mainly in Vietnam is photographed on Vietnamese sites. An
    English query cannot reach them, and the answer's own language is the wrong
    signal too — a global subject is best served in English whatever language
    the user writes in."""
    guidance = _query_guidance()

    assert "language" in guidance


def test_image_query_guidance_covers_subjects_whose_look_changes():
    """Brave's image endpoint has no freshness parameter, so the query text is
    the only way to ask for a current picture."""
    guidance = _query_guidance()

    assert "query" in guidance
    assert "year" in guidance
    assert "current" in guidance


def test_the_image_tool_opens_by_saying_it_is_the_only_path():
    """A model deciding whether to call a tool reads its description first. The
    description opened by promising sources to synthesize, so a question the
    model could already answer ("pokemon unite là gì") resolved to "no sources
    needed" and the only image path in the product was never entered. The
    picture job has to be in the opening line, not the third paragraph.
    """
    from app.ai.web_tools import IMAGE_SEARCH_DESCRIPTION

    opening = IMAGE_SEARCH_DESCRIPTION.split("\n\n")[0].lower()

    assert "only way an image reaches your answer" in opening
    assert "not only when you need sources" in opening


def test_media_guidance_names_the_tool_before_saying_there_is_no_inventory():
    """On a turn that has called nothing, the inventory is always absent — so
    "no rich items this turn" is the first thing the model reads about media
    unless acquisition comes first. Stating the dead end before the way out
    reads as a capability limit."""
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET

    assert snippet.index("image_search") < snippet.index("no rich items")


def test_media_guidance_never_tells_the_model_not_to_ask_for_pictures():
    """ "you never ask for pictures" was meant as "there is no separate image
    tool". It reads as an instruction to stay passive about images."""
    snippet = prompts.MEDIA_CAPABILITY_SNIPPET.lower()

    assert "never ask for pictures" not in snippet
    assert "you never ask" not in snippet


def test_image_tool_description_covers_the_same_recency_lever():
    """The tool description is read at call time and is where the argument is
    actually chosen."""
    from app.ai.web_tools import IMAGE_SEARCH_DESCRIPTION

    description = IMAGE_SEARCH_DESCRIPTION.lower()

    assert "year" in description
    assert "current" in description
    assert "time_range" in description


def test_rag_prompts_omit_web_tools_they_cannot_use():
    """The product web tools are internal and bound only for the chat and
    search agents, and internal tools never surface through ``tool_search`` — so
    naming one in a RAG prompt advertises a tool that agent can never call."""
    for prompt in (prompts.RAG_SYSTEM_PROMPT, prompts.AGENTIC_RAG_SYSTEM_PROMPT):
        for tool_name in ("web_research", "image_search", "web_search", "web_open"):
            assert tool_name not in prompt


def test_web_capable_prompts_name_the_image_tool():
    for prompt in (
        prompts.CHAT_SYSTEM_PROMPT,
        prompts.SEARCH_SYSTEM_PROMPT,
        prompts.SEARCH_WITH_RESULTS_SYSTEM_PROMPT,
    ):
        assert "image_search" in prompt
        assert "web_research" not in prompt


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
