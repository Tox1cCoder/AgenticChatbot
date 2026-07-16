from __future__ import annotations

import pytest

from app.ai.mcp_tool_catalog import ToolDescriptor
from app.ai.tool_search_profiles import infer_query_intent, infer_tool_profile
from app.ai.tool_search_scoring import rank_tool_candidates


def _tool(
    name: str,
    description: str,
    args: list[str],
    required: list[str],
    *,
    server: str = "synthetic_server",
) -> ToolDescriptor:
    return ToolDescriptor(
        tool_name=name,
        server_name=server,
        description=description,
        arg_names=args,
        required_arg_names=required,
        schema_fingerprint=f"fp-{server}-{name}",
    )


@pytest.mark.parametrize(
    "query",
    [
        "open browser url",
        "open link",
        "open application or URL on desktop",
        "browser open",
        "open youtube in browser",
        "open url in browser play youtube video",
    ],
)
def test_external_action_queries_infer_external_open(query: str):
    intent = infer_query_intent(query)

    assert "external_open" in intent.capabilities
    assert "web_extract" not in intent.capabilities


@pytest.mark.parametrize(
    "query",
    [
        "extract this URL",
        "read this web page",
        "summarize this article",
        "inspect content at this URL",
    ],
)
def test_content_queries_infer_web_extract(query: str):
    intent = infer_query_intent(query)

    assert "web_extract" in intent.capabilities
    assert "external_open" not in intent.capabilities


def test_url_without_action_does_not_guess_extract_or_open():
    intent = infer_query_intent("https://example.com")

    assert "web_extract" not in intent.capabilities
    assert "external_open" not in intent.capabilities


def test_tool_profiles_do_not_depend_on_server_brand():
    first = infer_tool_profile(
        tool_name="extract_url",
        server_name="alpha",
        description="Extract page content from known URLs.",
        arg_names=["urls"],
        required_arg_names=["urls"],
    )
    second = infer_tool_profile(
        tool_name="extract_url",
        server_name="beta",
        description="Extract page content from known URLs.",
        arg_names=["urls"],
        required_arg_names=["urls"],
    )

    assert first.capabilities == second.capabilities
    assert "web_extract" in first.capabilities


def _action_candidates(*, include_direct_opener: bool) -> list[ToolDescriptor]:
    candidates = [
        _tool(
            "start_process",
            "Start a shell command or local process.",
            ["command", "timeout_ms", "shell"],
            ["command"],
            server="local_runtime",
        ),
        _tool(
            "extract_url",
            "Extract page content from known URLs.",
            ["urls"],
            ["urls"],
            server="web_content",
        ),
        _tool(
            "web_search",
            "Search the web for current sources.",
            ["query"],
            ["query"],
            server="web_content",
        ),
    ]
    if include_direct_opener:
        candidates.append(
            _tool(
                "open_url",
                "Open a URL in the default browser.",
                ["url"],
                ["url"],
                server="local_runtime",
            )
        )
    return candidates


def test_direct_opener_outranks_process_fallback():
    ranked = rank_tool_candidates(
        query="open URL in browser",
        candidates=_action_candidates(include_direct_opener=True),
    )

    assert ranked[0].tool.tool_name == "open_url"
    assert ranked[0].confidence == "high"
    assert ranked[0].autoload_eligible is True
    assert "direct external opener" in ranked[0].match_reasons
    assert ranked[1].tool.tool_name == "start_process"


@pytest.mark.parametrize(
    "query",
    [
        "open browser url",
        "open link",
        "open application or URL on desktop",
        "browser open",
        "open youtube in browser",
        "open url in browser play youtube video",
    ],
)
def test_process_executor_is_high_confidence_external_open_fallback(query: str):
    ranked = rank_tool_candidates(
        query=query,
        candidates=_action_candidates(include_direct_opener=False),
    )

    assert ranked[0].tool.tool_name == "start_process"
    assert ranked[0].confidence == "high"
    assert ranked[0].autoload_eligible is True
    assert "can execute launch command" in ranked[0].match_reasons
    assert all(item.tool.tool_name != "extract_url" for item in ranked[:2])


def test_extract_query_still_prefers_extractor():
    ranked = rank_tool_candidates(
        query="extract content from this URL",
        candidates=_action_candidates(include_direct_opener=True),
    )

    assert ranked[0].tool.tool_name == "extract_url"
    assert ranked[0].confidence == "high"
    assert ranked[0].autoload_eligible is True
