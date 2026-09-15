"""Placement keeps widget convenience but never chooses an image."""

from types import SimpleNamespace

from app.core.config import settings
from app.core.rich_placement import auto_place_rich_items, finalize_article_content
from app.core.rich_response import parse_inline_rich_references


def test_widget_is_placed_after_its_best_paragraph() -> None:
    content = "Revenue rose this quarter.\n\nWeather remains mild in Bangkok."
    placed, ids = auto_place_rich_items(
        content,
        items=[("widget:revenue", "quarterly revenue")],
        min_score=0.5,
    )

    assert ids == ["widget:revenue"]
    assert "Revenue rose this quarter.\n\n<!--rich:widget:revenue-->" in placed


def test_widget_placement_ignores_fenced_code() -> None:
    content = "```\nquarterly revenue\n```"
    assert auto_place_rich_items(
        content,
        items=[("widget:revenue", "quarterly revenue")],
        min_score=0.1,
    ) == (content, [])


def test_finalize_never_inserts_unselected_image(monkeypatch) -> None:
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    content = "This paragraph describes the candidate image in exact detail."
    response = SimpleNamespace(
        message=SimpleNamespace(content=content),
        metadata={
            "_inline_rich_response_v1": True,
            "_rich_item_candidates": [
                {
                    "id": "image:web:unguarded",
                    "type": "image",
                    "source": "web_search",
                    "payload": {"url": "https://example.test/random.jpg"},
                }
            ],
        },
        tool_artifacts=None,
    )

    assert finalize_article_content(response, content) == content
    assert "<!--rich:image:" not in response.message.content


def test_finalize_keeps_only_server_authorized_image_markers(monkeypatch) -> None:
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    content = "Chosen.\n\n<!--rich:image:web:I2-->\n\n<!--rich:image:web:unknown-->"
    response = SimpleNamespace(
        message=SimpleNamespace(content=content),
        metadata={
            "_inline_rich_response_v1": True,
            "_rich_item_candidates": [{"id": "image:web:I2", "type": "image"}],
        },
        tool_artifacts=None,
    )

    result = finalize_article_content(response, content)

    assert parse_inline_rich_references(result) == ["image:web:I2"]
