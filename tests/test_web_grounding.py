from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.web_research.grounding import GroundingParser


def _session():
    sources = {
        "S1": SimpleNamespace(source_id="S1", url="https://example.test/one", title="One"),
        "S2": SimpleNamespace(source_id="S2", url="https://example.test/two", title="Two"),
    }
    images = {
        candidate_id: SimpleNamespace(
            rich_item={
                "id": f"image:web:{candidate_id.lower()}",
                "type": "image",
                "source": "image_search",
                "payload": {"url": f"/web-images/{candidate_id.lower()}"},
            }
        )
        for candidate_id in ("I1", "I2", "I3")
    }
    return SimpleNamespace(
        source_registry=SimpleNamespace(resolve=lambda value: sources.get(value)),
        prepared_images=images,
    )


def test_resolve_uses_server_urls_and_preserves_authored_image_order() -> None:
    resolution = GroundingParser(_session()).resolve(
        "Claim [[source:S2]]. Compare [[image:I2]] then [[image:I1]] and [[image:I2]]."
    )

    assert resolution.source_ids == ("S2",)
    assert resolution.selected_image_ids == ("I2", "I1")
    assert "[1](https://example.test/two)" in resolution.text
    assert resolution.text.count("<!--rich:image:web:i2-->") == 1
    assert [item["id"] for item in resolution.rich_items] == [
        "image:web:i2",
        "image:web:i1",
    ]


def test_no_image_token_selects_no_image() -> None:
    resolution = GroundingParser(_session()).resolve("Answer [[source:S1]].")

    assert resolution.selected_image_ids == ()
    assert resolution.rich_items == ()


def test_unknown_and_malformed_tokens_are_removed_with_bounded_warnings() -> None:
    resolution = GroundingParser(_session()).resolve(
        "A [[source:S99]] B [[image:I99]] C [[image: I1]]."
    )

    assert "[[" not in resolution.text
    assert {warning["code"] for warning in resolution.warnings} == {
        "unknown_source_id",
        "unknown_image_id",
        "malformed_grounding_token",
    }


def test_tokens_inside_inline_fenced_and_indented_code_are_not_interpreted() -> None:
    text = (
        "`[[image:I1]]`\n\n"
        "```text\n[[source:S1]]\n```\n\n"
        "    [[image:I2]]\n\n"
        "Use [[image:I3]]."
    )
    resolution = GroundingParser(_session()).resolve(text)

    assert resolution.selected_image_ids == ("I3",)
    assert "`[[image:I1]]`" in resolution.text
    assert "[[source:S1]]" in resolution.text
    assert "    [[image:I2]]" in resolution.text


@pytest.mark.parametrize("split", range(1, 47))
def test_every_chunk_split_resolves_like_whole_text(split: int) -> None:
    text = "Current [[source:S1]] view [[image:I2]], not [[image:I1]]."
    expected = GroundingParser(_session()).resolve(text)
    parser = GroundingParser(_session())

    parser.feed(text[:split])
    parser.feed(text[split:])

    assert parser.flush() == expected
