"""The typed web-search intent and its temporal normalization.

Every case here is about one thing: the server, not the model, decides what
"now" means. A model that writes "2024" into a query about the present is
repaired; a model that explicitly asks about 2024 is left alone.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from app.ai.mcp_servers.tavily_server import TAVILY_QUERY_MAX_LENGTH
from app.ai.web_query_contract import (
    WEB_QUERY_MAX_CHARS,
    WebQueryError,
    WebSearchRequest,
    normalize_web_search,
    tavily_search_args,
)

NOW = datetime(2026, 9, 4, 12, tzinfo=ZoneInfo("Asia/Bangkok"))


def _normalize(request: WebSearchRequest, *, now: datetime = NOW, configured: int = 8):
    return normalize_web_search(request, now=now, configured_max_results=configured)


def test_recent_query_repairs_stale_year_from_authoritative_now():
    request = WebSearchRequest(
        query="best local LLMs in 2024",
        objective="Find the currently strongest local models",
        freshness="recent",
    )
    normalized = normalize_web_search(
        request,
        now=datetime(2026, 9, 4, 12, tzinfo=ZoneInfo("Asia/Bangkok")),
        configured_max_results=8,
    )
    assert normalized.query == "best local LLMs in 2026"
    assert normalized.end_date == date(2026, 9, 4)


def test_as_of_query_preserves_historical_year():
    request = WebSearchRequest(
        query="Python packaging guidance in 2024",
        objective="Report what the guidance said at the end of 2024",
        freshness="as_of",
        end_date=date(2024, 12, 31),
    )
    normalized = normalize_web_search(
        request,
        now=datetime(2026, 9, 4, tzinfo=timezone.utc),
        configured_max_results=8,
    )
    assert "2024" in normalized.query
    assert normalized.end_date == date(2024, 12, 31)


def test_timeless_query_leaves_years_untouched():
    normalized = _normalize(
        WebSearchRequest(
            query="why the 1929 crash happened",
            objective="Explain the causes of the 1929 crash",
        )
    )

    assert normalized.query == "why the 1929 crash happened"
    assert normalized.end_date is None
    assert normalized.start_date is None


def test_recent_repairs_every_stale_year_and_ignores_bare_digits():
    normalized = _normalize(
        WebSearchRequest(
            query="RTX 5090 vs 2024 cards  2025 benchmarks",
            objective="Compare current GPU benchmarks",
            freshness="recent",
        )
    )

    assert normalized.query == "RTX 5090 vs 2026 cards 2026 benchmarks"


def test_recent_leaves_the_current_year_alone():
    normalized = _normalize(
        WebSearchRequest(
            query="2026 tax brackets",
            objective="Find this year's tax brackets",
            freshness="recent",
        )
    )

    assert normalized.query == "2026 tax brackets"


def test_recent_refuses_a_future_year_rather_than_rewriting_it():
    """Silently rewriting 2030 to 2026 would answer a different question than
    the one asked. The model gets a corrective error instead."""
    with pytest.raises(WebQueryError, match="future year"):
        _normalize(
            WebSearchRequest(
                query="projected 2030 grid capacity",
                objective="Find projections for 2030",
                freshness="recent",
            )
        )


def test_whitespace_is_collapsed_in_query_and_objective():
    normalized = _normalize(
        WebSearchRequest(
            query="  best   local\tLLMs \n now ",
            objective="  find the  strongest  ",
        )
    )

    assert normalized.query == "best local LLMs now"
    assert normalized.objective == "find the strongest"


def test_domains_are_lowercased_idna_normalized_and_deduplicated():
    normalized = _normalize(
        WebSearchRequest(
            query="release notes",
            objective="Find the release notes",
            include_domains=[
                "HTTPS://Example.COM/path?q=1",
                "example.com",
                "  münchen.de  ",
            ],
        )
    )

    assert normalized.include_domains == ("example.com", "xn--mnchen-3ya.de")


def test_blank_domains_are_dropped_not_forwarded_as_empty_filters():
    normalized = _normalize(
        WebSearchRequest(
            query="release notes",
            objective="Find the release notes",
            include_domains=["", "   ", "example.com"],
        )
    )

    assert normalized.include_domains == ("example.com",)


def test_max_results_is_clamped_to_the_configured_ceiling():
    normalized = _normalize(
        WebSearchRequest(
            query="release notes",
            objective="Find the release notes",
            max_results=500,
        ),
        configured=8,
    )

    assert normalized.max_results == 8


def test_max_results_below_the_ceiling_is_preserved():
    normalized = _normalize(
        WebSearchRequest(
            query="release notes",
            objective="Find the release notes",
            max_results=3,
        ),
        configured=8,
    )

    assert normalized.max_results == 3


def test_an_end_date_in_the_future_is_rejected():
    with pytest.raises(WebQueryError, match="future"):
        _normalize(
            WebSearchRequest(
                query="quarterly results",
                objective="Find the latest quarterly results",
                freshness="as_of",
                end_date=date(2027, 1, 1),
            )
        )


def test_a_start_date_in_the_future_is_rejected():
    with pytest.raises(WebQueryError, match="future"):
        _normalize(
            WebSearchRequest(
                query="quarterly results",
                objective="Find the latest quarterly results",
                freshness="as_of",
                start_date=date(2027, 1, 1),
                end_date=date(2027, 2, 1),
            )
        )


def test_a_reversed_range_is_rejected():
    with pytest.raises(WebQueryError, match="start_date"):
        _normalize(
            WebSearchRequest(
                query="quarterly results",
                objective="Find the results for that window",
                freshness="as_of",
                start_date=date(2024, 6, 1),
                end_date=date(2024, 1, 1),
            )
        )


def test_as_of_without_an_end_date_is_rejected():
    with pytest.raises(WebQueryError, match="end_date"):
        _normalize(
            WebSearchRequest(
                query="Python packaging guidance",
                objective="Report what the guidance said then",
                freshness="as_of",
            )
        )


def test_a_short_query_is_rejected_at_the_schema_boundary():
    with pytest.raises(ValidationError):
        WebSearchRequest(query="ai", objective="Find something about AI")


def test_a_missing_objective_is_rejected_at_the_schema_boundary():
    with pytest.raises(ValidationError):
        WebSearchRequest(query="best local LLMs")


def test_a_whitespace_only_objective_is_rejected():
    with pytest.raises((ValidationError, WebQueryError)):
        _normalize(WebSearchRequest(query="best local LLMs", objective="   "))


def test_the_normalized_value_is_immutable():
    normalized = _normalize(
        WebSearchRequest(query="release notes", objective="Find the release notes")
    )

    with pytest.raises(ValidationError):
        normalized.query = "something else"


def test_locale_is_stripped_and_lowercased():
    normalized = _normalize(
        WebSearchRequest(
            query="release notes",
            objective="Find the release notes",
            locale="  EN-GB ",
        )
    )

    assert normalized.locale == "en-gb"


def test_tavily_args_map_a_recent_search_to_the_news_topic():
    normalized = _normalize(
        WebSearchRequest(
            query="LCK roster changes",
            objective="Find the latest roster changes",
            freshness="recent",
            max_results=4,
        )
    )

    assert tavily_search_args(normalized) == {
        "query": "LCK roster changes",
        "max_results": 4,
        "search_depth": "advanced",
        "include_raw_content": False,
        "topic": "news",
        "end_date": "2026-09-04",
    }


def test_tavily_args_map_a_timeless_search_to_the_general_topic():
    normalized = _normalize(
        WebSearchRequest(
            query="how b-trees work",
            objective="Explain b-tree structure",
            max_results=5,
        )
    )

    assert tavily_search_args(normalized) == {
        "query": "how b-trees work",
        "max_results": 5,
        "search_depth": "advanced",
        "include_raw_content": False,
        "topic": "general",
    }


def test_tavily_args_forward_an_explicit_range_and_domains():
    normalized = _normalize(
        WebSearchRequest(
            query="packaging guidance",
            objective="Report the guidance as of that window",
            freshness="as_of",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            include_domains=["Packaging.Python.ORG"],
        )
    )

    args = tavily_search_args(normalized)

    assert args["start_date"] == "2024-01-01"
    assert args["end_date"] == "2024-12-31"
    assert args["include_domains"] == ["packaging.python.org"]
    assert args["topic"] == "general"


def test_tavily_args_never_request_raw_page_content():
    """Raw content is what made search results unbounded. Extraction is a
    separate, question-focused call."""
    normalized = _normalize(
        WebSearchRequest(query="release notes", objective="Find the release notes")
    )

    assert tavily_search_args(normalized)["include_raw_content"] is False


def test_the_public_query_bound_is_the_one_the_provider_actually_accepts():
    """Two boundaries for one value is a rejection the model cannot see coming.

    A query the schema advertises as valid, refused later by the provider
    wrapper, surfaces as a provider error after the turn's search slot has
    already been reserved -- so the corrected retry is refused for budget.
    """
    assert WEB_QUERY_MAX_CHARS == TAVILY_QUERY_MAX_LENGTH
    schema = WebSearchRequest.model_json_schema()["properties"]["query"]
    assert schema["maxLength"] == TAVILY_QUERY_MAX_LENGTH


def test_a_query_longer_than_the_provider_accepts_is_rejected_at_the_schema():
    with pytest.raises(ValidationError):
        WebSearchRequest(
            query="q" * (WEB_QUERY_MAX_CHARS + 1),
            objective="Find the published documentation",
        )


def test_normalization_rejects_an_overlong_query_from_a_caller_without_the_schema():
    request = WebSearchRequest.model_construct(
        query="q" * (WEB_QUERY_MAX_CHARS + 1),
        objective="Find the published documentation",
        freshness="timeless",
        start_date=None,
        end_date=None,
        locale=None,
        include_domains=[],
        max_results=5,
    )

    with pytest.raises(WebQueryError, match="characters"):
        _normalize(request)
