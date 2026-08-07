"""Schema contract for selected remote-image references."""

from app.models.web_image_reference import WebImageReference


def test_web_image_reference_columns_are_bounded_and_owned():
    columns = WebImageReference.__table__.columns

    assert WebImageReference.__tablename__ == "web_image_references"
    assert columns["conversation_id"].nullable is False
    assert columns["user_id"].nullable is False
    assert columns["upstream_url"].type.length == 4096
    assert columns["expected_mime"].type.length == 128
    assert columns["provider"].type.length == 32
    assert columns["deleted_at"].nullable is True


def test_web_image_reference_caches_only_verified_bytes():
    """The row may hold bytes this service itself fetched and decoded.

    Superseded the original "stores no image bytes" contract: visual
    verification already downloads and validates every image it approves, so
    fetching those same bytes a second time at render is both a wasted request
    and a second chance to fail after the image is already placed.

    All three columns stay nullable — a reference registered without bytes
    renders exactly as it did before — and there is still no on-disk path:
    Postgres is the only store.
    """
    columns = WebImageReference.__table__.columns

    assert "data" not in columns
    assert "storage_path" not in columns
    assert columns["content"].nullable is True
    assert columns["cached_width"].nullable is True
    assert columns["cached_height"].nullable is True
