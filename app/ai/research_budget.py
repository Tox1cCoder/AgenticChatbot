"""Turn-local accounting for factual and image research calls.

A model that receives a thin result reaches for another search. Bounding that
within a turn keeps one broad question from spending three provider calls on
near-identical queries. Nothing here outlives the turn: the store is reset when
a new user turn builds its initial graph state.

Tokens keep short identifiers such as ``T1``, ``F1`` and ``3M`` because those
are frequently the only word that identifies the subject. No stopword list is
used: the corpus is multilingual and a per-language table would buy nothing the
threshold does not already provide.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..core.config import settings

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_MAX_QUERY_CHARS = 2048
_MAX_TOKENS = 128
_MAX_TRACKED_CONVERSATIONS = 256


def normalize_query_tokens(query: str) -> frozenset[str]:
    """Return the comparable token set for a research query."""

    bounded = str(query or "")[:_MAX_QUERY_CHARS]
    normalized = unicodedata.normalize("NFKC", bounded).casefold()
    return frozenset(_TOKEN_PATTERN.findall(normalized)[:_MAX_TOKENS])


def near_duplicate(a: frozenset[str], b: frozenset[str], *, threshold: float) -> bool:
    """Return whether two token sets overlap by at least ``threshold``."""

    smaller = min(len(a), len(b))
    if smaller == 0:
        return False
    return len(a & b) / smaller >= float(threshold)


@dataclass
class ResearchBudget:
    """One turn's research accounting for a single conversation."""

    max_search_calls: int = 2
    near_duplicate_threshold: float = 0.75
    _searches: list[tuple[frozenset[str], str]] = field(default_factory=list)
    _image_searched: bool = False
    _image_candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def search_calls(self) -> int:
        return len(self._searches)

    def find_reuse(self, query: str) -> str | None:
        """Return an existing result for an exact or near-duplicate query."""

        tokens = normalize_query_tokens(query)
        for recorded_tokens, result_text in self._searches:
            if recorded_tokens == tokens or near_duplicate(
                recorded_tokens, tokens, threshold=self.near_duplicate_threshold
            ):
                return result_text
        return None

    def may_search(self, query: str) -> bool:
        if self.find_reuse(query) is not None:
            return False
        return self.search_calls < max(1, int(self.max_search_calls))

    def record_search(self, query: str, result_text: str) -> None:
        self._searches.append((normalize_query_tokens(query), result_text))

    def accumulated(self) -> list[str]:
        return [result_text for _, result_text in self._searches]

    def may_image_search(self) -> bool:
        return not self._image_searched

    def record_image_search(self, candidates: list[dict[str, Any]]) -> None:
        self._image_searched = True
        self._image_candidates = list(candidates)

    def image_result(self) -> list[dict[str, Any]]:
        return list(self._image_candidates)


_lock = threading.Lock()
_budgets: OrderedDict[str, ResearchBudget] = OrderedDict()


def _key(conversation_id: str | None) -> str:
    return str(conversation_id or "__no_conversation__")


def get_research_budget(conversation_id: str | None) -> ResearchBudget:
    """Return the live budget for a conversation, creating it on first use."""

    key = _key(conversation_id)
    with _lock:
        budget = _budgets.get(key)
        if budget is None:
            budget = ResearchBudget(
                max_search_calls=max(1, int(settings.research_max_search_calls_per_turn)),
                near_duplicate_threshold=float(settings.research_near_duplicate_threshold),
            )
            _budgets[key] = budget
        _budgets.move_to_end(key)
        while len(_budgets) > _MAX_TRACKED_CONVERSATIONS:
            _budgets.popitem(last=False)
        return budget


def reset_research_budget(conversation_id: str | None) -> None:
    """Drop a conversation's budget so the next turn starts clean."""

    with _lock:
        _budgets.pop(_key(conversation_id), None)
