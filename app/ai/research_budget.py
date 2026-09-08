"""Turn-local accounting for factual and image research calls.

A model that receives a thin result reaches for another search. Bounding that
within a turn keeps one broad question from spending three provider calls on
near-identical queries.

**Keyed by logical turn, not by conversation (R4).** Conversation keying had two
faults that only appear once a turn can span more than one epoch: a Continue
served by a different worker saw no accounting at all, and one served by *this*
worker shared a single entry with any other turn in flight for the same
conversation. The logical turn is the unit the accounting is about, so it is
the key.

**What a Continue replenishes, and what it does not.** Per-epoch call caps
reset: a continued turn gets a fresh allowance, which is the point of
continuing. Cross-epoch *deduplication* does not — the "already searched this"
memory carries forward, or Continue would become a way to re-run the identical
query the budget had just refused. That memory is what
:meth:`ResearchBudget.to_state` persists onto the generation row and
:func:`research_budget_from_state` restores, so any worker can serve the next
epoch.

Only the token sets travel, never the result text. Task 4's ``carried_messages``
already hands the next epoch the previous epoch's ``ToolMessage``s, so the
evidence is in the transcript; what the row needs to carry is only enough to
refuse a repeat.

Tokens keep short identifiers such as ``T1``, ``F1`` and ``3M`` because those
are frequently the only word that identifies the subject. No stopword list is
used: the corpus is multilingual and a per-language table would buy nothing the
threshold does not already provide.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..core.config import settings

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_MAX_QUERY_CHARS = 2048
_MAX_TOKENS = 128
# A turn that happens to fetch many *other* conversations' budgets between two
# lookups of its own can push its own entry out of this LRU, silently
# resetting its dedup memory and call count mid-turn. Bounded storage is still
# the right tradeoff for a process that must not leak memory across an
# unbounded number of conversations; the debug log on eviction below is how
# the next reader learns this failure mode exists instead of hitting it blind.
_MAX_TRACKED_CONVERSATIONS = 256
SearchScope = tuple[str | int | None, ...]

#: Bumped only for a change a reader of an older payload cannot absorb. An
#: unknown version is refused rather than guessed at, because a
#: misinterpreted dedup memory silently grants a full new quota.
RESEARCH_ACCOUNTING_SCHEMA_VERSION = 1

#: How many prior queries the dedup memory carries across epochs. The row is
#: not a place for unbounded growth, and the oldest entries are the least
#: likely to be repeated. Exceeding it drops the oldest, which can only make
#: the guard more permissive -- never less.
_MAX_PERSISTED_SEARCHES = 64


class ResearchAccountingUnreadable(ValueError):
    """A persisted accounting payload could not be restored.

    Raised rather than degrading to an empty budget, because an empty budget is
    indistinguishable from a fresh turn's full quota: the Continue would
    silently re-run every search the previous epoch had already paid for. R4
    requires the Continue to fail instead.
    """


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
    max_image_searches: int = 1
    _searches: list[tuple[frozenset[str], SearchScope, str]] = field(default_factory=list)
    _in_flight: list[tuple[frozenset[str], SearchScope]] = field(default_factory=list)
    _image_searches: list[frozenset[str]] = field(default_factory=list)
    _image_candidates: list[dict[str, Any]] = field(default_factory=list)
    # Queries an *earlier epoch* of this turn already made. They carry no
    # result text -- the next epoch receives the previous one's ToolMessages
    # through `carried_messages`, so all the row has to remember is enough to
    # refuse a repeat. These do not count against this epoch's call cap; only
    # `_searches` does, which is what makes the cap per-epoch.
    _prior_searches: list[tuple[frozenset[str], SearchScope]] = field(default_factory=list)
    _prior_image_searches: list[frozenset[str]] = field(default_factory=list)
    #: How many epochs have contributed to this accounting, this one included.
    epochs_recorded: int = 1
    _instance_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def search_calls(self) -> int:
        return len(self._searches)

    def find_reuse(self, query: str, *, scope: SearchScope = ()) -> str | None:
        """Return a result for a matching query made with the same controls.

        Only this epoch's searches, because only they carry a result. A match
        against an earlier epoch is reported by :meth:`searched_in_prior_epoch`
        instead: there is nothing to hand back, and the answer to it is a
        refusal rather than a reuse.
        """

        tokens = normalize_query_tokens(query)
        for recorded_tokens, recorded_scope, result_text in self._searches:
            if recorded_scope != scope:
                continue
            if recorded_tokens == tokens or near_duplicate(
                recorded_tokens, tokens, threshold=self.near_duplicate_threshold
            ):
                return result_text
        return None

    def searched_in_prior_epoch(self, query: str, *, scope: SearchScope = ()) -> bool:
        """Whether an earlier epoch of this turn already ran this query.

        The guard that stops Continue being a way to re-run the search the
        previous epoch's budget had just refused. The evidence it produced is
        already in the model's transcript, so repeating the call would spend a
        provider request to learn nothing.
        """

        tokens = normalize_query_tokens(query)
        for recorded_tokens, recorded_scope in self._prior_searches:
            if recorded_scope != scope:
                continue
            if recorded_tokens == tokens or near_duplicate(
                recorded_tokens, tokens, threshold=self.near_duplicate_threshold
            ):
                return True
        return False

    def reserve_search(self, query: str, *, scope: SearchScope = ()) -> bool:
        """Atomically claim a search slot, or refuse if none remain.

        A caller checks the budget, awaits a network call, then records the
        result — two steps with an await in between. The model can emit
        parallel tool calls, so two concurrent callers could both pass a plain
        boolean check before either records its result, silently letting the
        turn exceed its cap. Comparing completed searches plus in-flight
        reservations to the cap inside one lock closes that window: the check
        and the claim happen as a single step.
        """

        with self._instance_lock:
            tokens = normalize_query_tokens(query)
            if self.find_reuse(query, scope=scope) is not None:
                return False
            if self.searched_in_prior_epoch(query, scope=scope):
                # Refused across the epoch boundary as well as within it,
                # which is what stops a Continue re-running the query the
                # previous epoch's cap had already turned down.
                return False
            for reserved_tokens, reserved_scope in self._in_flight:
                if reserved_scope != scope:
                    continue
                if reserved_tokens == tokens or near_duplicate(
                    reserved_tokens,
                    tokens,
                    threshold=self.near_duplicate_threshold,
                ):
                    return False
            if self.search_calls + len(self._in_flight) >= max(1, int(self.max_search_calls)):
                return False
            self._in_flight.append((tokens, scope))
            return True

    def record_search(self, query: str, result_text: str, *, scope: SearchScope = ()) -> None:
        """Append a completed result and release the reservation it used.

        A failed search never reaches this method, so its reservation is never
        released: a provider error must not buy the model a second attempt at
        the same broken query within the same turn.
        """

        with self._instance_lock:
            tokens = normalize_query_tokens(query)
            self._searches.append((tokens, scope, result_text))
            for index, reservation in enumerate(self._in_flight):
                if reservation == (tokens, scope):
                    self._in_flight.pop(index)
                    break

    def accumulated(self) -> list[str]:
        return [result_text for _, _, result_text in self._searches]

    def may_image_search(self) -> bool:
        with self._instance_lock:
            return len(self._image_searches) < max(1, int(self.max_image_searches))

    def reserve_image_search(self, image_query: str) -> bool:
        """Atomically claim an image-discovery slot for one visual subject.

        An answer may want more than one picture — a thing's own identity art
        and a shot of it in use are different subjects — so slots are per
        subject rather than per turn. A repeat of a subject already searched is
        refused: it would return the same picture again, which is what a deeper
        slice of one query already did.

        Like ``reserve_search``, the slot is consumed on claim, so a provider
        failure does not buy a second attempt inside the same turn.
        """

        with self._instance_lock:
            tokens = normalize_query_tokens(image_query)
            for searched in (*self._image_searches, *self._prior_image_searches):
                if searched == tokens or near_duplicate(
                    searched, tokens, threshold=self.near_duplicate_threshold
                ):
                    return False
            if len(self._image_searches) >= max(1, int(self.max_image_searches)):
                return False
            self._image_searches.append(tokens)
            return True

    def record_image_search(self, candidates: list[dict[str, Any]]) -> None:
        """Accumulate one search's selections alongside earlier subjects'."""
        with self._instance_lock:
            self._image_candidates.extend(candidates)

    def image_result(self) -> list[dict[str, Any]]:
        with self._instance_lock:
            return list(self._image_candidates)

    def to_state(self) -> dict[str, Any]:
        """The dedup memory, in a shape the generation row can hold.

        This epoch's searches and every earlier epoch's are folded into one
        list: from the next epoch's point of view they are all "already done".
        Result text is deliberately absent — see the module docstring.

        Token sets are sorted so the payload is stable, which keeps a row from
        churning on every write for no semantic change.
        """

        with self._instance_lock:
            searched = [
                {"tokens": sorted(tokens), "scope": list(scope)}
                for tokens, scope, _ in self._searches
            ]
            searched.extend(
                {"tokens": sorted(tokens), "scope": list(scope)}
                for tokens, scope in self._prior_searches
            )
            subjects = [sorted(tokens) for tokens in self._image_searches]
            subjects.extend(sorted(tokens) for tokens in self._prior_image_searches)
            return {
                "schema_version": RESEARCH_ACCOUNTING_SCHEMA_VERSION,
                # Newest first, so the bound below drops the oldest entries --
                # the ones least likely to be asked again.
                "searched": searched[:_MAX_PERSISTED_SEARCHES],
                "image_subjects": subjects[:_MAX_PERSISTED_SEARCHES],
                "epochs_recorded": int(self.epochs_recorded),
            }


def research_budget_from_state(payload: Any) -> ResearchBudget:
    """Rebuild a turn's accounting for its next epoch.

    The result starts a *fresh* epoch: the call caps are unspent and every
    persisted query sits in the cross-epoch dedup memory. That is the whole
    contract — replenish the allowance, keep the memory.

    Raises :class:`ResearchAccountingUnreadable` for anything it cannot read.
    Degrading to an empty budget is the one thing this must not do, because an
    empty budget looks exactly like a fresh turn with a full quota.
    """

    if not isinstance(payload, dict):
        raise ResearchAccountingUnreadable(
            f"research accounting must be an object, got {type(payload).__name__}"
        )

    version = payload.get("schema_version")
    if version != RESEARCH_ACCOUNTING_SCHEMA_VERSION:
        raise ResearchAccountingUnreadable(
            f"unsupported research accounting schema version {version!r}"
        )

    prior_searches: list[tuple[frozenset[str], SearchScope]] = []
    raw_searched = payload.get("searched")
    if not isinstance(raw_searched, list):
        raise ResearchAccountingUnreadable("research accounting is missing its searched list")
    for entry in raw_searched:
        if not isinstance(entry, dict):
            raise ResearchAccountingUnreadable("a searched entry is not an object")
        tokens = entry.get("tokens")
        scope = entry.get("scope")
        if not isinstance(tokens, list) or not isinstance(scope, list):
            raise ResearchAccountingUnreadable("a searched entry has a malformed shape")
        # Back to a tuple: `find_reuse` and `reserve_search` compare scopes by
        # equality, and a list would never match a caller's tuple.
        prior_searches.append((frozenset(str(token) for token in tokens), tuple(scope)))

    prior_images: list[frozenset[str]] = []
    raw_subjects = payload.get("image_subjects")
    if raw_subjects is not None:
        if not isinstance(raw_subjects, list):
            raise ResearchAccountingUnreadable("image_subjects must be a list")
        for entry in raw_subjects:
            if not isinstance(entry, list):
                raise ResearchAccountingUnreadable("an image subject is not a list of tokens")
            prior_images.append(frozenset(str(token) for token in entry))

    try:
        epochs_recorded = int(payload.get("epochs_recorded") or 1)
    except (TypeError, ValueError) as exc:
        raise ResearchAccountingUnreadable("epochs_recorded is not an integer") from exc

    return ResearchBudget(
        max_search_calls=max(1, int(settings.research_max_search_calls_per_turn)),
        near_duplicate_threshold=float(settings.research_near_duplicate_threshold),
        max_image_searches=max(1, int(settings.research_max_image_searches_per_turn)),
        _prior_searches=prior_searches,
        _prior_image_searches=prior_images,
        epochs_recorded=max(1, epochs_recorded) + 1,
    )


_lock = threading.Lock()
_budgets: OrderedDict[str, ResearchBudget] = OrderedDict()


def _key(logical_turn_id: str | None, conversation_id: str | None = None) -> str:
    """The store key for one turn's accounting.

    The logical turn is the unit, and it is the same value as the checkpoint
    thread's turn segment and the ``generations.logical_turn_id`` column, so
    one turn has one key wherever it is looked up.

    The conversation is a *fallback* only, for a caller with no turn identity
    to hand. Such a caller gets conversation-scoped behaviour — which is what
    it had before R4 — rather than a shared global bucket that would leak dedup
    memory between unrelated turns.
    """

    turn = str(logical_turn_id or "").strip()
    if turn:
        return f"turn:{turn}"
    conversation = str(conversation_id or "").strip()
    if conversation:
        return f"conversation:{conversation}"
    return "__no_turn__"


def get_research_budget(
    *,
    logical_turn_id: str | None = None,
    conversation_id: str | None = None,
) -> ResearchBudget:
    """Return the live budget for a logical turn, creating it on first use.

    Keyword-only, and deliberately so. Before R4 the single positional
    parameter was the conversation id; making the turn id positional instead
    would have let every existing call keep working while reading and writing a
    *different* key than it intended -- a silently wrong cache key, which is
    worse than a crash. A stale positional call is now a ``TypeError``.
    """

    key = _key(logical_turn_id, conversation_id)
    with _lock:
        budget = _budgets.get(key)
        if budget is None:
            budget = ResearchBudget(
                max_search_calls=max(1, int(settings.research_max_search_calls_per_turn)),
                near_duplicate_threshold=float(settings.research_near_duplicate_threshold),
                max_image_searches=max(1, int(settings.research_max_image_searches_per_turn)),
            )
            _budgets[key] = budget
        _budgets.move_to_end(key)
        while len(_budgets) > _MAX_TRACKED_CONVERSATIONS:
            evicted_key, _ = _budgets.popitem(last=False)
            logger.debug(
                "Evicted research budget for %r; a turn still in progress for "
                "it would silently lose its dedup memory and call count.",
                evicted_key,
            )
        return budget


def install_research_budget(
    payload: Any,
    *,
    logical_turn_id: str | None = None,
    conversation_id: str | None = None,
) -> ResearchBudget:
    """Restore a turn's accounting from its persisted state, for a new epoch.

    Used by both Continue paths, deliberately: a Continue served by *this*
    worker rehydrates from the row exactly as one served by another worker
    does, rather than reusing whatever the in-memory entry happens to hold. One
    path means the two cannot disagree about how much quota the next epoch has.

    Propagates :class:`ResearchAccountingUnreadable` so the caller can fail the
    Continue.
    """

    budget = research_budget_from_state(payload)
    key = _key(logical_turn_id, conversation_id)
    with _lock:
        _budgets[key] = budget
        _budgets.move_to_end(key)
    return budget


def snapshot_research_budget(
    *,
    logical_turn_id: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any] | None:
    """The persistable accounting for a turn, or ``None`` if it never searched.

    ``None`` rather than an empty payload: a turn that made no research calls
    has nothing for the next epoch to be refused against, and writing an empty
    object would make "never searched" indistinguishable from "restored from a
    payload with no entries".
    """

    key = _key(logical_turn_id, conversation_id)
    with _lock:
        budget = _budgets.get(key)
    if budget is None:
        return None
    state = budget.to_state()
    if not state["searched"] and not state["image_subjects"]:
        return None
    return state


def reset_research_budget(
    *,
    logical_turn_id: str | None = None,
    conversation_id: str | None = None,
) -> None:
    """Drop a turn's budget so the next turn starts clean.

    Largely belt-and-braces since R4: a new turn has a new key, so it starts
    clean whether or not this is called. It still runs at turn start to keep
    the store from holding entries no epoch will ask for again.
    """

    with _lock:
        _budgets.pop(_key(logical_turn_id, conversation_id), None)
