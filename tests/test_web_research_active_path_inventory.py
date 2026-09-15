from pathlib import Path

from app.ai.deferred_tool_binding import RAW_WEB_TOOL_NAMES, ordinary_excluded_tool_names

FORBIDDEN = (
    "selected_image_sink",
    "offer_selected_images",
    "select_brave_candidates",
    "anchor_image_items_by_query",
    "create_image_search_tool",
    "build_image_candidates_from_tool_result",
    "extract_images_from_tool_result",
)


def test_superseded_web_image_symbols_are_absent_from_production() -> None:
    root = Path(__file__).parents[1] / "app"
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
        if "alembic" not in path.parts
    )

    assert not {symbol for symbol in FORBIDDEN if symbol in production}


def test_raw_provider_tools_are_excluded_from_ordinary_binding() -> None:
    assert ordinary_excluded_tool_names() >= RAW_WEB_TOOL_NAMES


def test_interrupted_specialists_release_uncheckpointed_image_state() -> None:
    specialists = (
        Path(__file__).parents[1] / "app" / "ai" / "workflow" / "specialists.py"
    ).read_text(encoding="utf-8")

    assert "web_research_session.suspend" not in specialists


def test_pending_web_image_expiry_is_wired_to_periodic_cleanup() -> None:
    cleanup = (Path(__file__).parents[1] / "app" / "workers" / "cleanup_tasks.py").read_text(
        encoding="utf-8"
    )

    assert "web_image_repo.arelease_expired(now)" in cleanup
