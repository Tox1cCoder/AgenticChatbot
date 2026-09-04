"""Typed web-search intent and the temporal repair applied to it.

The model states what kind of question it is asking — timeless, current, or as
of a stated date — and the server decides what "now" means. Nothing here calls
a clock: ``now`` is passed in, so the same request normalizes identically in a
test, a replay, and production.

Two failure modes this exists to prevent:

- A model trained before the current year writes that stale year into a query
  about the present, and the provider faithfully returns last year's answer.
  ``freshness="recent"`` repairs those years against the server's date.
- A model asking a genuine historical question has its year "corrected" out
  from under it. ``freshness="as_of"`` preserves every year as written.

A future year under ``recent`` is neither: rewriting it would answer a
different question, so it raises and the model receives a corrective error.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

Freshness = Literal["timeless", "recent", "as_of"]

#: Four-digit years only, on word boundaries. A bare ``5090`` inside a product
#: name is not a year, so the range is anchored to plausible calendar years.
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b")
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_DOMAINS = 10


class WebQueryError(ValueError):
    """A web-search intent the server refuses to normalize.

    Raised rather than repaired: each case names a request the model can fix
    itself, and a silent repair would answer a question nobody asked.
    """


class WebSearchRequest(BaseModel):
    """What the model asks for, before the server anchors it in time."""

    query: str = Field(
        min_length=3,
        max_length=500,
        description="The search query, written as you would type it into a search engine.",
    )
    objective: str = Field(
        min_length=3,
        max_length=500,
        description=(
            "The concrete fact or answer this search must produce. Used to rank "
            "evidence, so state the goal, not the topic."
        ),
    )
    freshness: Freshness = Field(
        default="timeless",
        description=(
            "'recent' for current/latest information, 'as_of' with an explicit "
            "end_date for a historical cutoff, 'timeless' otherwise."
        ),
    )
    start_date: date | None = Field(
        default=None, description="Optional inclusive start of an explicit date range."
    )
    end_date: date | None = Field(
        default=None,
        description="Inclusive end of the date range. Required when freshness is 'as_of'.",
    )
    locale: str | None = Field(
        default=None,
        max_length=32,
        description="Optional BCP-47 locale hint for the requested sources.",
    )
    include_domains: list[str] = Field(
        default_factory=list,
        max_length=_MAX_DOMAINS,
        description="Optional domains to restrict the search to.",
    )
    max_results: int = Field(
        default=5, ge=1, description="Results to request. Clamped to the configured ceiling."
    )


class NormalizedWebSearch(BaseModel):
    """A request the server has anchored, bounded, and made provider-ready."""

    model_config = ConfigDict(frozen=True)

    query: str
    objective: str
    freshness: Freshness
    start_date: date | None
    end_date: date | None
    locale: str | None
    include_domains: tuple[str, ...]
    max_results: int


def normalize_web_search(
    request: WebSearchRequest,
    *,
    now: datetime,
    configured_max_results: int,
) -> NormalizedWebSearch:
    """Anchor, validate, and bound one search intent. Pure: no clock, no I/O."""

    today = now.date()
    query = _collapse(request.query)
    objective = _collapse(request.objective)
    if len(objective) < 3:
        raise WebQueryError("objective must state the fact the search has to produce")
    if len(query) < 3:
        raise WebQueryError("query must not be blank")

    start_date = request.start_date
    end_date = request.end_date
    _reject_future(start_date, field="start_date", today=today)
    _reject_future(end_date, field="end_date", today=today)
    if start_date and end_date and start_date > end_date:
        raise WebQueryError("start_date must not be later than end_date")

    if request.freshness == "as_of":
        if end_date is None:
            raise WebQueryError("freshness='as_of' requires an explicit end_date")
    elif request.freshness == "recent":
        query = _repair_stale_years(query, current_year=today.year)
        if end_date is None:
            end_date = today

    return NormalizedWebSearch(
        query=query,
        objective=objective,
        freshness=request.freshness,
        start_date=start_date,
        end_date=end_date,
        locale=_normalize_locale(request.locale),
        include_domains=_normalize_domains(request.include_domains),
        max_results=max(1, min(int(request.max_results), max(1, int(configured_max_results)))),
    )


def tavily_search_args(value: NormalizedWebSearch) -> dict[str, Any]:
    """Build the provider call for one normalized search.

    Centralized on purpose: a second place that assembles these arguments is a
    second place the date anchoring can be forgotten.
    """

    args: dict[str, Any] = {
        "query": value.query,
        "max_results": value.max_results,
        "search_depth": "advanced",
        "include_raw_content": False,
        "topic": "news" if value.freshness == "recent" else "general",
    }
    if value.start_date:
        args["start_date"] = value.start_date.isoformat()
    if value.end_date:
        args["end_date"] = value.end_date.isoformat()
    if value.include_domains:
        args["include_domains"] = list(value.include_domains)
    return args


def _collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def _reject_future(value: date | None, *, field: str, today: date) -> None:
    if value is not None and value > today:
        raise WebQueryError(f"{field} is in the future; the web has no results past {today}")


def _repair_stale_years(query: str, *, current_year: int) -> str:
    """Rewrite past years to the current one; refuse a future one."""

    for match in _YEAR_RE.finditer(query):
        if int(match.group()) > current_year:
            raise WebQueryError(
                f"query names the future year {match.group()} but asks for recent "
                "information; use freshness='timeless' for a projection"
            )
    return _YEAR_RE.sub(str(current_year), query)


def _normalize_locale(locale: str | None) -> str | None:
    cleaned = _collapse(locale or "").lower()
    return cleaned or None


def _normalize_domains(domains: list[str]) -> tuple[str, ...]:
    """Reduce each entry to a bare, comparable, ASCII host."""

    normalized: list[str] = []
    for raw in domains:
        host = _bare_host(raw)
        if host and host not in normalized:
            normalized.append(host)
    return tuple(normalized[:_MAX_DOMAINS])


def _bare_host(raw: str) -> str:
    candidate = unicodedata.normalize("NFKC", str(raw or "")).strip().lower()
    if not candidate:
        return ""
    if "//" in candidate:
        candidate = urlsplit(candidate).netloc or candidate
    candidate = candidate.split("/", 1)[0].split("?", 1)[0].split("@")[-1].strip("[]")
    candidate = candidate.rsplit(":", 1)[0] if candidate.count(":") == 1 else candidate
    if not candidate:
        return ""
    try:
        return candidate.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        # A host the IDNA codec rejects (empty or overlong label) is passed
        # through as written: the provider is the authority on what it accepts,
        # and dropping it silently would widen the search the model narrowed.
        return candidate


__all__ = [
    "Freshness",
    "NormalizedWebSearch",
    "WebQueryError",
    "WebSearchRequest",
    "normalize_web_search",
    "tavily_search_args",
]
