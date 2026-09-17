"""Canonical public URLs and stable source IDs for one research session."""

from __future__ import annotations

import ipaddress
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .contracts import ProviderSource, SourceRecord

_TRACKING_KEYS = frozenset(
    {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "msclkid",
        "ref_src",
    }
)


def canonicalize_public_url(url: str) -> str | None:
    """Return a stable public HTTP URL, or ``None`` for an unsafe shape."""

    try:
        parsed = urlsplit(str(url or "").strip())
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username or parsed.password:
            return None
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        if host == "localhost" or host.endswith((".localhost", ".local")):
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if host.replace(".", "").isdigit():
                return None
        else:
            if not address.is_global:
                return None
        port = parsed.port
    except (UnicodeError, ValueError):
        return None

    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_KEYS
        ),
        doseq=True,
    )
    return urlunsplit((scheme, netloc, path, query, ""))


class SourceRegistry:
    def __init__(self, *, max_sources: int) -> None:
        self._max_sources = max(0, int(max_sources))
        self._by_url: OrderedDict[str, SourceRecord] = OrderedDict()

    @property
    def records(self) -> tuple[SourceRecord, ...]:
        return tuple(self._by_url.values())

    @property
    def capacity(self) -> int:
        return self._max_sources

    @property
    def free_slots(self) -> int:
        return max(0, self._max_sources - len(self._by_url))

    def grow_capacity(self, max_sources: int) -> None:
        """Raise the admission ceiling. Never lowers it.

        A session learns its visual intent from the first search that declares
        one, after the registry already exists. Growing is safe because admitted
        records and their IDs are append-only; shrinking would orphan IDs the
        model has already been shown, so it is refused.
        """

        self._max_sources = max(self._max_sources, max(0, int(max_sources)))

    def admit(self, candidates: Sequence[ProviderSource]) -> tuple[SourceRecord, ...]:
        for candidate in candidates:
            url = canonicalize_public_url(candidate.url)
            if url is None:
                continue
            existing = self._by_url.get(url)
            if existing is not None:
                continue
            if len(self._by_url) >= self._max_sources:
                continue
            record = SourceRecord(
                source_id=f"S{len(self._by_url) + 1}",
                url=url,
                title=candidate.title,
                snippet=candidate.snippet,
                status="search_result",
                published_at=candidate.published_at,
                provider=candidate.provider,
                query_index=candidate.query_index,
            )
            self._by_url[url] = record
        return self.records

    def resolve(self, source_id_or_url: str) -> SourceRecord | None:
        value = str(source_id_or_url or "").strip()
        if value.startswith("S"):
            return next(
                (record for record in self._by_url.values() if record.source_id == value),
                None,
            )
        url = canonicalize_public_url(value)
        return self._by_url.get(url) if url else None

    def mark_opened(self, source_id: str, *, snippet: str | None = None) -> SourceRecord:
        record = self.resolve(source_id)
        if record is None:
            raise KeyError(source_id)
        updated = record.model_copy(
            update={
                "status": "opened",
                "snippet": snippet if snippet is not None else record.snippet,
            }
        )
        self._by_url[str(record.url)] = updated
        return updated

    def import_records(self, records: Iterable[SourceRecord]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for record in records:
            self.admit(
                (
                    ProviderSource(
                        provider=record.provider,
                        url=str(record.url),
                        title=record.title,
                        snippet=record.snippet,
                        rank=1,
                        query_index=record.query_index,
                        published_at=record.published_at,
                    ),
                )
            )
            imported = self.resolve(str(record.url))
            if imported is not None:
                mapping[record.source_id] = imported.source_id
        return mapping


__all__ = ["SourceRegistry", "canonicalize_public_url"]
