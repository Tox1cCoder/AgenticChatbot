from app.core.config import get_settings


def test_chat_image_settings_defaults():
    s = get_settings()
    assert s.chat_images_storage_path == "app/storage/chat_images"
    assert s.chat_image_max_bytes == 10 * 1024 * 1024
    assert s.chat_image_history_rehydrate_limit == 4
