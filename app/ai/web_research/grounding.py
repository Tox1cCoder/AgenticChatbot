"""Resolve model-authored web grounding tokens against one turn session."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SEGMENT = re.compile(
    r"(?P<fence>^ {0,3}(?P<fence_mark>`{3,}|~{3,})[^\n]*\n.*?"
    r"^ {0,3}(?P=fence_mark)[ \t]*(?:\n|$))"
    r"|(?P<inline>(?P<tick>`+)[^\n]*?(?P=tick))"
    r"|(?P<indent>^(?: {4}|\t).*?$)"
    r"|(?P<token>\[\[(?P<kind>source|image):(?P<identifier>[^\]]*)\]\])",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class GroundingResolution:
    text: str
    source_ids: tuple[str, ...]
    selected_image_ids: tuple[str, ...]
    rich_items: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, str], ...]


class GroundingParser:
    """One parser for terminal text and arbitrarily split streamed chunks."""

    def __init__(self, session: Any, *, max_warnings: int = 20) -> None:
        self._session = session
        self._max_warnings = max(1, int(max_warnings))
        self._chunks: list[str] = []

    def feed(self, chunk: str) -> str:
        self._chunks.append(str(chunk or ""))
        return ""

    def flush(self) -> GroundingResolution:
        text = "".join(self._chunks)
        self._chunks.clear()
        return self.resolve(text)

    def resolve(self, text: str) -> GroundingResolution:
        source_ids: list[str] = []
        image_ids: list[str] = []
        rich_items: list[dict[str, Any]] = []
        warnings: list[dict[str, str]] = []

        def warn(code: str, identifier: str) -> None:
            if len(warnings) < self._max_warnings:
                warnings.append({"code": code, "id": identifier[:64]})

        def replace(match: re.Match[str]) -> str:
            token = match.group("token")
            if token is None:
                return match.group(0)
            kind = str(match.group("kind") or "")
            identifier = str(match.group("identifier") or "").strip()
            expected = r"S[1-9][0-9]*" if kind == "source" else r"I[1-9][0-9]*"
            if identifier != str(match.group("identifier") or "") or not re.fullmatch(
                expected, identifier
            ):
                warn("malformed_grounding_token", identifier)
                return ""

            if kind == "source":
                source = self._session.source_registry.resolve(identifier)
                if source is None:
                    warn("unknown_source_id", identifier)
                    return ""
                if identifier not in source_ids:
                    source_ids.append(identifier)
                number = source_ids.index(identifier) + 1
                return f"[{number}]({source.url})"

            prepared = self._session.prepared_images.get(identifier)
            if prepared is None:
                warn("unknown_image_id", identifier)
                return ""
            if identifier in image_ids:
                return ""
            image_ids.append(identifier)
            rich_items.append(dict(prepared.rich_item))
            return f"<!--rich:{prepared.rich_item['id']}-->"

        resolved = _SEGMENT.sub(replace, str(text or ""))
        return GroundingResolution(
            text=resolved,
            source_ids=tuple(source_ids),
            selected_image_ids=tuple(image_ids),
            rich_items=tuple(rich_items),
            warnings=tuple(warnings),
        )


__all__ = ["GroundingParser", "GroundingResolution"]
