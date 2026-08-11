"""The visual-strategy block must tell every answering agent *when* a visual is
worth having, and must stay a decision procedure rather than a topic lookup.

Before this block existed, the only visual guidance was mechanical (marker
syntax, tool arguments), so an agent that already knew the answer never
researched, never received an image candidate, and never built a widget. The
tests below pin the two halves of that fix: a positive trigger to go and fetch a
visual, and the prose-only exceptions that keep it from decorating everything.
"""

from app.ai import prompts

ANSWER_PROMPTS = (
    prompts.CHAT_SYSTEM_PROMPT,
    prompts.RAG_SYSTEM_PROMPT,
    prompts.AGENTIC_RAG_SYSTEM_PROMPT,
    prompts.SEARCH_SYSTEM_PROMPT,
    prompts.SEARCH_WITH_RESULTS_SYSTEM_PROMPT,
)

# Words that would signal visual routing hardcoded to subject matter rather than
# to what the reader needs to see. Capability words a library is chosen by
# ("physics", "math") are deliberately absent: those describe what the widget has
# to do, not a school subject the answer belongs to.
TAXONOMY_WORDS = ("chemistry", "biology", "anatomy", "recipe", "sports", "finance")


def test_block_is_compact():
    assert len(prompts.VISUAL_STRATEGY_SNIPPET) < 2800, (
        "visual strategy block must stay compact — it rides on every answering prompt"
    )


def test_all_answer_prompts_carry_visual_strategy():
    for prompt in ANSWER_PROMPTS:
        assert "Show, don't only tell:" in prompt


def test_visual_need_alone_justifies_research():
    """The gap that kept images out of answers: knowing the facts suppressed the
    research call, and images only arrive through research."""
    block = prompts.VISUAL_STRATEGY_SNIPPET.lower()

    assert "reason enough" in block
    assert "even when you already know the facts" in block


def test_block_keeps_prose_only_exceptions():
    """A trigger without exceptions turns every answer into a slideshow."""
    block = prompts.VISUAL_STRATEGY_SNIPPET.lower()

    assert "stay in prose" in block
    assert "no visual at all" in block


def test_image_trigger_requires_a_concrete_subject():
    block = prompts.VISUAL_STRATEGY_SNIPPET.lower()

    assert "concrete subject" in block
    assert "abstract" in block


def test_widget_guidance_reaches_every_widget_capable_agent():
    """Widget tools are pinned for chat, rag and search, but only the chat prompt
    used to describe them — so rag and search never built one."""
    for prompt in (
        prompts.CHAT_SYSTEM_PROMPT,
        prompts.RAG_SYSTEM_PROMPT,
        prompts.SEARCH_SYSTEM_PROMPT,
    ):
        lowered = prompt.lower()
        assert "live widget" in lowered
        assert "micro-app" in lowered
        assert "slider" in lowered


def test_widget_guidance_is_not_duplicated_in_the_chat_prompt():
    assert prompts.CHAT_SYSTEM_PROMPT.lower().count("micro-app") == 1


def test_widget_library_guidance_stays_capability_keyed():
    """Libraries are allowed, but chosen by what the widget must do — the prompt
    must not name a library the model is expected to reach for by default."""
    block = prompts.VISUAL_STRATEGY_SNIPPET.lower()

    assert "vanilla javascript handles most widgets" in block
    assert "public cdn" in block
    for capability in ("physics", "3d scene", "axes and scales", "math", "tweened"):
        assert capability in block, f"library guidance must name the need: {capability!r}"


def test_prompt_states_the_sandbox_limits_that_break_libraries():
    """The frame is sandboxed without same-origin, so storage-backed libraries
    fail at runtime and an unreachable CDN renders an empty box."""
    block = prompts.VISUAL_STRATEGY_SNIPPET.lower()

    assert "no storage" in block
    assert "fails to load" in block


def test_widget_tool_description_carries_the_full_library_contract():
    """The tool description is what the model reads while writing the html, so
    the concrete named libraries and sandbox limits live there, not in every
    system prompt."""
    from app.ai.mcp_servers.widgets_server import widget_create

    # Collapsed: the docstring is hard-wrapped, so phrases straddle line breaks.
    doc = " ".join((widget_create.__doc__ or "").lower().split())

    assert "public cdn" in doc
    assert "matter.js" in doc
    assert "three.js" in doc
    assert "localstorage" in doc
    assert "readable message" in doc


def test_block_has_no_hardcoded_visual_topic_list():
    block = prompts.VISUAL_STRATEGY_SNIPPET.lower()
    for word in TAXONOMY_WORDS:
        assert word not in block, f"block must not hardcode visual topics: found {word!r}"


def test_non_answer_prompts_unchanged():
    for prompt in (
        prompts.ROUTER_SYSTEM_PROMPT,
        prompts.TITLE_GENERATION_PROMPT,
        prompts.PLANNING_EXECUTION_PROMPT,
        prompts.IMAGE_GENERATOR_SYSTEM_PROMPT,
    ):
        assert "Show, don't only tell:" not in prompt
