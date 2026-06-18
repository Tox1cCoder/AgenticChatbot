"""Tests for harvesting inline images from multimodal LLM responses.

Image-capable models (e.g. Gemini image models) return generated images as
content blocks. ``coerce_response_text`` deliberately drops them, which made the
image-generator agent emit "No response generated" while discarding the images.
``extract_inline_images_from_content`` recovers those images so callers keep them.
"""

import base64

from app.ai.utils import coerce_response_text, extract_inline_images_from_content

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-bytes"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("utf-8")


def test_extracts_standard_image_blocks_and_skips_thinking():
    content = [
        {"type": "thinking", "thinking": "Designing a cat illustration..."},
        {"type": "image", "source_type": "base64", "data": PNG_B64, "mime_type": "image/png"},
        {"type": "image", "source_type": "base64", "data": PNG_B64, "mime_type": "image/jpeg"},
    ]
    images = extract_inline_images_from_content(content)
    assert [img["data"] for img in images] == [PNG_B64, PNG_B64]
    assert [img["mime"] for img in images] == ["image/png", "image/jpeg"]
    # The text coercion still yields nothing — that's the bug this guards against.
    assert coerce_response_text(content) == ""


def test_extracts_image_url_data_uri():
    content = [
        {"type": "text", "text": "Here you go:"},
        {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{PNG_B64}"}},
    ]
    images = extract_inline_images_from_content(content)
    assert len(images) == 1
    assert images[0]["data"] == PNG_B64
    assert images[0]["mime"] == "image/webp"


def test_extracts_gemini_inline_data_with_bytes():
    content = [
        {"inline_data": {"mime_type": "image/png", "data": PNG_BYTES}},
    ]
    images = extract_inline_images_from_content(content)
    assert len(images) == 1
    assert images[0]["data"] == PNG_B64  # bytes encoded to base64
    assert images[0]["mime"] == "image/png"


def test_extracts_media_block():
    content = [{"type": "media", "data": PNG_B64, "mime_type": "image/png"}]
    images = extract_inline_images_from_content(content)
    assert images == [{"data": PNG_B64, "mime": "image/png"}]


def test_skips_remote_image_urls():
    content = [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]
    assert extract_inline_images_from_content(content) == []


def test_text_only_and_empty_inputs_are_safe():
    assert extract_inline_images_from_content("just text") == []
    assert extract_inline_images_from_content([{"type": "text", "text": "hi"}]) == []
    assert extract_inline_images_from_content(None) == []
    assert extract_inline_images_from_content([]) == []
