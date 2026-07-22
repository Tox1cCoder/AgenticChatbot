import base64

from app.core.response_constants import externalize_metadata_images


def test_generated_images_externalized():
    def store(*, mime, data_b64, name, **_):
        return {"name": name, "mime": mime, "image_id": "g1", "url": "/chat-images/g1"}

    imgs = [{"name": "gen.png", "mime_type": "image/png", "data": base64.b64encode(b"X").decode()}]
    out = externalize_metadata_images(imgs, store=store)
    assert out[0]["url"] == "/chat-images/g1"
    assert out[0]["image_id"] == "g1"
    assert "data" not in out[0] and "b64_data" not in out[0]


def test_b64_data_field_is_externalized():
    def store(*, mime, data_b64, name, **_):
        assert data_b64 == "WQ=="
        return {"name": name, "mime": mime, "image_id": "g2", "url": "/chat-images/g2"}

    imgs = [{"mime_type": "image/png", "b64_data": "WQ=="}]
    out = externalize_metadata_images(imgs, store=store)
    assert out[0]["url"] == "/chat-images/g2"
    assert "b64_data" not in out[0]


def test_remote_images_left_untouched():
    out = externalize_metadata_images(
        [{"url": "https://x/y.png", "mime_type": "image/png"}], store=None
    )
    assert out[0]["url"] == "https://x/y.png"


def test_store_failure_keeps_inline_entry():
    def boom(**_):
        raise RuntimeError("down")

    original = {"mime_type": "image/png", "data": "WQ=="}
    out = externalize_metadata_images([original], store=boom)
    assert out == [original]


def test_none_and_empty_are_safe():
    assert externalize_metadata_images(None, store=None) is None
    assert externalize_metadata_images([], store=None) == []
