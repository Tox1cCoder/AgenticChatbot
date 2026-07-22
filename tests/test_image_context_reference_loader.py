from app.ai.image_context import build_multimodal_content, has_image_parts, use_chat_image_loader


def test_reference_attachment_resolved_via_loader():
    calls = []

    def loader(image_id):
        calls.append(image_id)
        return "data:image/png;base64,QUJD"

    att = {"name": "a.png", "mime": "image/png", "image_id": "img-9", "url": "/chat-images/img-9"}
    with use_chat_image_loader(loader):
        parts = build_multimodal_content("hi", [att])

    assert calls == ["img-9"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_reference_dropped_when_no_loader_installed():
    att = {"name": "a.png", "mime": "image/png", "image_id": "img-9", "url": "/chat-images/img-9"}
    parts = build_multimodal_content("hi", [att])
    # internal /chat-images url is NOT model-fetchable; without a loader it must not leak
    assert not has_image_parts(parts)


def test_inline_data_still_works_without_loader():
    att = {"name": "a.png", "mime": "image/png", "data": "QUJD"}
    parts = build_multimodal_content("", [att])
    assert has_image_parts(parts)
    assert parts[0]["image_url"]["url"] == "data:image/png;base64,QUJD"
