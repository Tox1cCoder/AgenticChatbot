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
