from __future__ import annotations

from app.ai.web_research.contracts import ProviderSource
from app.ai.web_research.source_registry import SourceRegistry, canonicalize_public_url


def test_registry_deduplicates_tracking_variants_and_keeps_first_rank() -> None:
    registry = SourceRegistry(max_sources=5)

    admitted = registry.admit(
        (
            ProviderSource(
                provider="first",
                url="https://EXAMPLE.com/x/?utm_source=newsletter&b=2&a=1#top",
                title="First",
                snippet="one",
                rank=1,
                query_index=1,
            ),
            ProviderSource(
                provider="second",
                url="https://example.com/x?a=1&b=2",
                title="Second",
                snippet="two",
                rank=2,
                query_index=1,
            ),
        )
    )

    assert [item.source_id for item in admitted] == ["S1"]
    assert str(admitted[0].url) == "https://example.com/x?a=1&b=2"
    assert admitted[0].title == "First"


def test_opened_status_updates_without_changing_identity() -> None:
    registry = SourceRegistry(max_sources=5)
    source = registry.admit(
        (
            ProviderSource(
                provider="test",
                url="https://example.com/release",
                rank=1,
                query_index=1,
            ),
        )
    )[0]

    opened = registry.mark_opened(source.source_id)

    assert opened.source_id == "S1"
    assert opened.status == "opened"
    assert registry.resolve("S1") == opened


def test_canonicalizer_rejects_credentials_and_non_web_schemes() -> None:
    assert canonicalize_public_url("file:///etc/passwd") is None
    assert canonicalize_public_url("https://user:pass@example.com/x") is None


def test_canonicalizer_rejects_non_public_hosts() -> None:
    for url in (
        "https://localhost/admin",
        "https://127.0.0.1/admin",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.1/private",
        "https://[::1]/admin",
    ):
        assert canonicalize_public_url(url) is None


def _provider_source(url: str) -> ProviderSource:
    return ProviderSource(provider="test", url=url, rank=1, query_index=1)


def test_admit_returns_only_the_current_delta() -> None:
    registry = SourceRegistry(max_sources=3)
    first = registry.admit((_provider_source("https://example.com/a"),))
    second = registry.admit(
        (
            _provider_source("https://example.com/a"),
            _provider_source("https://example.com/b"),
        )
    )

    assert [record.source_id for record in first] == ["S1"]
    assert [record.source_id for record in second] == ["S2"]
    assert [record.source_id for record in registry.records] == ["S1", "S2"]


def test_admit_returns_nothing_when_every_candidate_is_already_known() -> None:
    registry = SourceRegistry(max_sources=3)
    registry.admit((_provider_source("https://example.com/a"),))

    assert registry.admit((_provider_source("https://example.com/a"),)) == ()
