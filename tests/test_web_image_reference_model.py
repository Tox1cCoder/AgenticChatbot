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


def test_web_image_reference_stores_no_image_bytes():
    columns = WebImageReference.__table__.columns

    assert "data" not in columns
    assert "content" not in columns
    assert "storage_path" not in columns
