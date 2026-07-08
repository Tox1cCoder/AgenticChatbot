import base64

from app.ai.image_context import (
    attachment_memory_lines,
    build_multimodal_content,
    has_image_parts,
    image_url_part,
    normalize_image_attachment,
)

PNG_B64 = base64.b64encode(b"fake-png").decode("ascii")


def test_normalize_base64_attachment_to_data_url():
    normalized = normalize_image_attachment(
        {"name": "screen.png", "mime": "image/png", "data": PNG_B64}
    )

    assert normalized == {
        "name": "screen.png",
        "mime": "image/png",
        "url": f"data:image/png;base64,{PNG_B64}",
    }


def test_normalize_data_url_keeps_single_data_prefix():
    url = f"data:image/webp;base64,{PNG_B64}"

    normalized = normalize_image_attachment({"name": "clip.webp", "data": url})

    assert normalized["url"] == url
    assert normalized["mime"] == "image/webp"


def test_normalize_remote_url_and_rejects_blob_url():
    normalized = normalize_image_attachment(
        {"name": "remote.png", "mime": "image/png", "url": "https://example.test/a.png"}
    )

    assert normalized == {
        "name": "remote.png",
        "mime": "image/png",
        "url": "https://example.test/a.png",
    }
    assert normalize_image_attachment({"name": "clip.png", "url": "blob:http://app/123"}) is None


def test_normalize_skips_local_path_and_non_image_mime():
    assert normalize_image_attachment({"path": "C:\\fakepath\\x.png"}) is None
    assert normalize_image_attachment({"path": "C:/fakepath/x.png"}) is None
    assert normalize_image_attachment({"mime": "text/plain", "data": PNG_B64}) is None


def test_image_url_part_uses_single_langchain_shape():
    assert image_url_part("data:image/png;base64,abc") == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc"},
    }


def test_build_multimodal_content_adds_text_before_images():
    content = build_multimodal_content(
        "describe this",
        [{"name": "screen.png", "mime": "image/png", "data": PNG_B64}],
    )

    assert content == [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
    ]
    assert has_image_parts(content) is True


def test_attachment_memory_lines_do_not_include_raw_image_data():
    lines = attachment_memory_lines(
        [
            {"name": "screen.png", "mime": "image/png", "data": PNG_B64},
            {"name": "notes.txt", "mime": "text/plain", "data": PNG_B64},
        ]
    )

    assert lines == ["[Attached image: screen.png, image/png]"]
    assert PNG_B64 not in "\n".join(lines)
