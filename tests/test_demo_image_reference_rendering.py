"""Frontend rendering resolves stored chat-image references to displayable src."""

from types import SimpleNamespace

import demo


def _patch_st(monkeypatch, auth_token="tok"):
    monkeypatch.setattr(demo, "st", SimpleNamespace(session_state={"auth_token": auth_token}))


def test_absolute_url_passthrough(monkeypatch):
    _patch_st(monkeypatch)
    out = demo._normalize_image_for_gallery({"url": "https://x/y.png"}, "img")
    assert out == {"src": "https://x/y.png", "name": "img"}


def test_relative_reference_resolved_to_data_uri(monkeypatch):
    _patch_st(monkeypatch)
    seen = {}

    def fake_fetch(url, token):
        seen["url"] = url
        seen["token"] = token
        return "data:image/png;base64,QUJD"

    monkeypatch.setattr(demo, "_fetch_chat_image_data_uri", fake_fetch)
    out = demo._normalize_image_for_gallery(
        {"url": "/chat-images/abc", "name": "shot"}, "img"
    )
    assert out == {"src": "data:image/png;base64,QUJD", "name": "shot"}
    assert seen == {"url": "/chat-images/abc", "token": "tok"}


def test_relative_reference_falls_back_to_inline_data(monkeypatch):
    _patch_st(monkeypatch)
    monkeypatch.setattr(demo, "_fetch_chat_image_data_uri", lambda *_: None)
    out = demo._normalize_image_for_gallery(
        {"url": "/chat-images/abc", "data": "QUJD", "mime": "image/png"}, "img"
    )
    assert out == {"src": "data:image/png;base64,QUJD", "name": "img"}


def test_b64_data_field_supported(monkeypatch):
    _patch_st(monkeypatch)
    out = demo._normalize_image_for_gallery(
        {"b64_data": "QUJD", "mime_type": "image/webp"}, "img"
    )
    assert out == {"src": "data:image/webp;base64,QUJD", "name": "img"}
