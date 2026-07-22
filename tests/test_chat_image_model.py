from app.models.chat_image import ChatImage


def test_chat_image_table_and_columns():
    assert ChatImage.__tablename__ == "chat_images"
    cols = ChatImage.__table__.columns
    for name in (
        "id",
        "conversation_id",
        "user_id",
        "sha256",
        "size_bytes",
        "content_type",
        "storage_path",
        "created_at",
        "deleted_at",
    ):
        assert name in cols, f"missing column {name}"
    assert cols["storage_path"].nullable is False
    assert cols["deleted_at"].nullable is True
