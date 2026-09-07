"""Bounded, deterministic retrieval of the passages that answer one objective.

A large tool result used to reach the model as a paged blob: read 8000 chars,
get a ``next_offset``, read again. That loop spends a turn discovering where
the answer is. This module answers the question directly — given the payload
and the fact still needed, return the few passages that carry it, each
addressable back to its position in the original payload.

Deliberately stdlib only. No embedding call, no model call: retrieval has to
stay cheap, reproducible, trace-light, and available when the providers this
system depends on are degraded — which is exactly when a large result is most
likely to be all the evidence there is.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field

#: Longest single candidate considered. Longer leaves are split into chunks so
#: a whole page arriving as one JSON string cannot hide its tail, while no
#: single candidate can dominate memory.
_CANDIDATE_MAX_CHARS = 1_200
#: Chunks one oversized leaf may contribute. Beyond this the leaf's remainder
#: is counted as omitted rather than silently dropped.
_MAX_CHUNKS_PER_LEAF = 60
#: Total candidates scored. A payload larger than this reports the remainder in
#: ``omitted_candidates``.
_MAX_CANDIDATES = 4_000
#: Shortest text worth returning as evidence.
_MIN_CANDIDATE_CHARS = 3
#: Floor below which a trimmed excerpt is dropped instead of shortened further.
_MIN_TRIMMED_CHARS = 60
#: Guards the float branch of scalar-fact rendering; JSON has no infinity.
_INF = float("inf")

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_NON_WORD_RE = re.compile(r"[\W_]+", flags=re.UNICODE)
_URL_RE = re.compile(r"^\s*(?:https?|ftp|data|blob)://\S*\s*$", flags=re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")

#: Tokens that carry a passage's claim rather than its subject. Two passages
#: that differ on one of these are stating different facts however similar
#: their wording, so they are compared before any similarity measure.
_FACT_WORDS = frozenset(
    # fmt: off
    [
        "no", "not", "never", "none", "nor", "without", "cannot", "cant",
        "unavailable", "unsupported", "unchanged", "false", "denied",
        "excluded", "removed", "revoked", "rejected",
    ]
    # fmt: on
)

#: Keys whose value labels a source rather than states a fact. They are lifted
#: onto every excerpt drawn from the same object, so scoring them as evidence
#: too would let a result's own title outrank the sentence that answers the
#: question — the title names the subject, which is most of what an objective
#: mentions.
_METADATA_KEYS = frozenset({"title", "url", "source_url"})

#: Small, static, and English-only on purpose. Coverage is measured as a share
#: of the objective's own terms, so an objective in another language simply
#: keeps all of its tokens and scores on the same scale.
_STOP_WORDS = frozenset(
    # fmt: off
    [
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
        "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "their", "there", "they", "this", "to", "was", "were", "what",
        "when", "where", "which", "who", "why", "will", "with",
    ]
    # fmt: on
)

_NO_MATCH_NOTE = (
    "No passage in the stored result matched this objective. Restate the exact "
    "fact you need, or accept that the result does not contain it — re-reading "
    "with the same objective returns this same answer."
)
_EMPTY_NOTE = "The stored result held no readable text."


class FocusedExcerpt(BaseModel):
    """One passage, addressed back to where it sits in the original payload."""

    source_path: str = Field(description="Position in the payload, e.g. $.results[2].content")
    text: str
    score: float
    url: str | None = None
    title: str | None = None


class FocusedResult(BaseModel):
    """What one focused read returned, and what it left behind."""

    objective: str
    excerpts: list[FocusedExcerpt] = Field(default_factory=list)
    total_candidates: int = 0
    omitted_candidates: int = 0
    truncated: bool = False
    note: str = ""


class _Candidate:
    """A scored passage. Mutable and internal; never leaves this module."""

    __slots__ = (
        "order",
        "path",
        "text",
        "normalized",
        "tokens",
        "facts",
        "url",
        "title",
        "score",
    )

    def __init__(
        self,
        *,
        order: int,
        path: str,
        text: str,
        url: str | None,
        title: str | None,
    ) -> None:
        self.order = order
        self.path = path
        self.text = text
        self.normalized = _normalize(text)
        self.tokens = tuple(_TOKEN_RE.findall(self.normalized))
        self.facts = frozenset(
            token
            for token in self.tokens
            if token in _FACT_WORDS or _DIGIT_RE.search(token)
        )
        self.url = url
        self.title = title
        self.score = 0.0


def select_focused_excerpts(
    payload: str,
    *,
    objective: str,
    max_excerpts: int,
    max_chars: int,
    fallback_to_leading: bool = False,
) -> FocusedResult:
    """Return the passages of ``payload`` that best answer ``objective``.

    Bounded twice over: at most ``max_excerpts`` passages, and a serialized
    result no longer than ``max_chars``.

    ``fallback_to_leading`` decides what "nothing matched" means. Term matching
    is exact, so a question phrased as "which release date is stated" scores
    zero against a passage that says "released on 14 March". Set it when
    something upstream already ranked these passages for this same question —
    a page extractor handed the query, say — because there the leading
    passages are the provider's answer and dropping them all loses evidence
    that was already retrieved. Leave it off for a stored blob nobody ranked:
    there, returning arbitrary text in place of a match is noise dressed as
    evidence.
    """

    objective = str(objective or "").strip()
    excerpt_limit = max(1, int(max_excerpts))
    char_limit = max(1, int(max_chars))

    candidates, overflow = _extract_candidates(payload)
    total = len(candidates) + overflow
    if not candidates:
        return _bounded(
            FocusedResult(objective=objective, total_candidates=total, note=_EMPTY_NOTE),
            char_limit,
        )

    terms = _objective_terms(objective)
    for candidate in candidates:
        candidate.score = _score(candidate, terms)

    ordered = sorted(candidates, key=lambda c: (-c.score, c.order))
    ranked = [item for item in ordered if item.score > 0.0]
    if not ranked and fallback_to_leading:
        ranked = sorted(candidates, key=lambda c: c.order)
    if not ranked:
        return _bounded(
            FocusedResult(
                objective=objective,
                total_candidates=total,
                omitted_candidates=total,
                note=_NO_MATCH_NOTE,
            ),
            char_limit,
        )

    selected = _take_distinct(ranked, limit=excerpt_limit)
    result = FocusedResult(
        objective=objective,
        excerpts=[
            FocusedExcerpt(
                source_path=item.path,
                text=item.text,
                score=round(item.score, 4),
                url=item.url,
                title=item.title,
            )
            for item in selected
        ],
        total_candidates=total,
        omitted_candidates=max(0, total - len(selected)),
    )
    return _bounded(result, char_limit)


def _bounded(result: FocusedResult, char_limit: int) -> FocusedResult:
    """Trim the result until its serialized form honors the exact budget.

    Serialization happens here and only here: scoring never pays for it. The
    lowest-ranked excerpt is shortened first and dropped once shortening it
    further would leave a fragment too small to be evidence.
    """

    for _ in range(len(result.excerpts) * 2 + 8):
        _recount(result)
        serialized = len(result.model_dump_json())
        if serialized <= char_limit:
            return result
        result.truncated = True
        if not result.excerpts:
            # Only the envelope is left. A note is the compressible part.
            if result.note:
                result.note = ""
                continue
            return result
        excess = serialized - char_limit
        last = result.excerpts[-1]
        keep = len(last.text) - excess
        if keep < _MIN_TRIMMED_CHARS:
            result.excerpts.pop()
            continue
        last.text = last.text[:keep].rstrip()
    _recount(result)
    return result


def _recount(result: FocusedResult) -> None:
    """Report what this response actually left behind, not what ranking did.

    The count has to be derived after trimming: an excerpt the budget dropped
    is as absent as one ranking never selected, and these numbers are what the
    operational log reports and what tells the model whether re-reading with a
    sharper objective could find more.
    """

    result.omitted_candidates = max(0, result.total_candidates - len(result.excerpts))


def _objective_terms(objective: str) -> tuple[str, ...]:
    tokens = _TOKEN_RE.findall(_normalize(objective))
    meaningful = tuple(token for token in tokens if token not in _STOP_WORDS)
    return meaningful or tuple(tokens)


def _score(candidate: _Candidate, terms: tuple[str, ...]) -> float:
    """Term coverage, lifted by the longest contiguous run of those terms.

    Coverage answers "does this passage mention what I asked about"; the run
    bonus answers "does it mention it the way I asked", which is what separates
    a passage about the subject from the passage that states the fact.
    """

    if not terms or not candidate.tokens:
        return 0.0
    present = set(candidate.tokens)
    covered = sum(1 for term in set(terms) if term in present)
    if not covered:
        return 0.0
    coverage = covered / len(set(terms))
    run = _longest_run(candidate.tokens, terms) / len(terms)
    return 0.7 * coverage + 0.3 * run


def _longest_run(tokens: tuple[str, ...], terms: tuple[str, ...]) -> int:
    """Longest contiguous slice of ``terms`` that appears in ``tokens``."""

    best = 0
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(tokens):
        positions.setdefault(token, []).append(index)
    for start in range(len(terms)):
        for origin in positions.get(terms[start], ()):
            length = 0
            while (
                start + length < len(terms)
                and origin + length < len(tokens)
                and tokens[origin + length] == terms[start + length]
            ):
                length += 1
            best = max(best, length)
        if best == len(terms):
            break
    return best


def _take_distinct(ranked: list[_Candidate], *, limit: int) -> list[_Candidate]:
    """Take the top candidates, skipping ones that repeat an earlier passage.

    Exact and normalized duplicates are already gone; this catches the case
    where two kept passages say the same thing in different words, which is
    only worth the O(n^2) comparison across the handful actually returned.
    """

    selected: list[_Candidate] = []
    for candidate in ranked:
        if any(_near_duplicate(candidate, chosen) for chosen in selected):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _near_duplicate(candidate: _Candidate, other: _Candidate) -> bool:
    """Two passages repeat each other only if they also carry the same facts.

    Overlap alone equates "revenue was 100 million" with "revenue was 200
    million", and "is enforced" with "is not enforced": the one or two tokens
    that separate them are exactly the ones the objective asked about, and
    dropping the second passage removes the only evidence that the sources
    disagree. Numbers, dates, versions and negations are therefore compared
    first; the similarity test then still catches a genuine restatement.
    """

    if candidate.facts != other.facts:
        return False
    left, right = set(candidate.tokens), set(other.tokens)
    smaller = min(len(left), len(right))
    if smaller == 0:
        return False
    return len(left & right) / smaller >= 0.9


def _extract_candidates(payload: str) -> tuple[list[_Candidate], int]:
    """Split the payload into addressable passages, JSON-aware where possible."""

    text = str(payload or "")
    if not text.strip():
        return [], 0
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        return _walk_json(parsed)
    return _paragraph_candidates(text)


def _paragraph_candidates(text: str) -> tuple[list[_Candidate], int]:
    candidates: list[_Candidate] = []
    overflow = 0
    for index, block in enumerate(_PARAGRAPH_RE.split(text)):
        chunks, dropped = _chunks(f"paragraph[{index}]", block)
        overflow += dropped
        for path, chunk in chunks:
            if len(candidates) >= _MAX_CANDIDATES:
                overflow += 1
                continue
            candidates.append(
                _Candidate(order=len(candidates), path=path, text=chunk, url=None, title=None)
            )
    return candidates, overflow


def _walk_json(root: Any) -> tuple[list[_Candidate], int]:
    candidates: list[_Candidate] = []
    overflow = 0
    stack: list[tuple[Any, str, str | None, str | None, str]] = [(root, "$", None, None, "")]
    while stack:
        node, path, url, title, label = stack.pop()
        if isinstance(node, dict):
            url = _string_field(node, "url") or _string_field(node, "source_url") or url
            title = _string_field(node, "title") or title
            # Reversed so the stack yields keys in declaration order, which is
            # what makes source paths and tie-breaks reproducible.
            for key in reversed(list(node.keys())):
                if key in _METADATA_KEYS and isinstance(node[key], str):
                    continue
                stack.append((node[key], f"{path}.{key}", url, title, str(key)))
            continue
        if isinstance(node, list):
            # The label does not advance: a list index names a position, not a
            # fact, so ``$.latencies_ms[1]`` still reads as a latency.
            for index in reversed(range(len(node))):
                stack.append((node[index], f"{path}[{index}]", url, title, label))
            continue
        if isinstance(node, str):
            if _URL_RE.match(node):
                continue
            text = node
        else:
            fact = _scalar_fact(label, node)
            if fact is None:
                continue
            text = fact
        chunks, dropped = _chunks(path, text)
        overflow += dropped
        for chunk_path, chunk in chunks:
            if len(candidates) >= _MAX_CANDIDATES:
                overflow += 1
                continue
            candidates.append(
                _Candidate(
                    order=len(candidates), path=chunk_path, text=chunk, url=url, title=title
                )
            )
    return candidates, overflow


def _scalar_fact(label: str, node: Any) -> str | None:
    """Render a non-string scalar as a key-addressed fact, or skip it.

    A number is evidence. The offset pager this reader replaced could see
    ``"annual_revenue": 42000000`` because it returned raw text; a reader that
    only walks string leaves cannot, and answers "no passage matched" to a
    payload that holds the answer.

    The key travels with the value because the value alone is unmatchable:
    nothing in the objective "annual revenue" matches ``42000000``. Without a
    key there is nothing to match against, so an unlabelled scalar — a bare
    element of a top-level list — is skipped rather than returned as noise.
    """

    if node is None or isinstance(node, str) or not isinstance(node, (bool, int, float)):
        return None
    if isinstance(node, float) and (node != node or node in (_INF, -_INF)):
        return None
    name = str(label or "").strip()
    if not name:
        return None
    return f"{name}: {json.dumps(node)}"


def _string_field(node: dict[str, Any], key: str) -> str | None:
    value = node.get(key)
    return value.strip() or None if isinstance(value, str) else None


def _chunks(path: str, raw: str) -> tuple[list[tuple[str, str]], int]:
    """Split one leaf into bounded pieces, each with its own address.

    Returns the pieces and how many were dropped past the per-leaf ceiling. A
    leaf short enough to stand alone keeps its plain path, so the common case
    stays citable as ``$.results[2].content``.
    """

    text = str(raw or "").strip()
    if len(text) < _MIN_CANDIDATE_CHARS:
        return [], 0
    if len(text) <= _CANDIDATE_MAX_CHARS:
        return [(path, text)], 0

    pieces: list[str] = []
    for block in _PARAGRAPH_RE.split(text):
        block = block.strip()
        while len(block) > _CANDIDATE_MAX_CHARS:
            # ``rfind`` answers -1 when the window holds no space at all, and
            # -1 is truthy: unspaced text — CJK, one machine-generated token —
            # used to cut at the last character, producing a chunk far over the
            # bound and silently losing the tail. There is no word boundary to
            # respect there, so the bound itself is the split.
            cut = block.rfind(" ", 0, _CANDIDATE_MAX_CHARS)
            if cut <= 0:
                cut = _CANDIDATE_MAX_CHARS
            pieces.append(block[:cut].strip())
            block = block[cut:].strip()
        if len(block) >= _MIN_CANDIDATE_CHARS:
            pieces.append(block)

    kept = pieces[:_MAX_CHUNKS_PER_LEAF]
    return [(f"{path}#{index}", piece) for index, piece in enumerate(kept)], len(pieces) - len(kept)


def _normalize(text: str) -> str:
    collapsed = _NON_WORD_RE.sub(" ", unicodedata.normalize("NFKC", str(text or "")).casefold())
    return collapsed.strip()


__all__ = ["FocusedExcerpt", "FocusedResult", "select_focused_excerpts"]
