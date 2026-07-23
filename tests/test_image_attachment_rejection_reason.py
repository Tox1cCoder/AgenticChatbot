from app.ai.image_context import (
    describe_attachment_rejections,
    normalize_image_attachment,
    normalize_image_attachment_result,
)


def test_blob_url_reports_reason():
    result, reason = normalize_image_attachment_result(
        {"name": "x", "mime": "image/png", "url": "blob:http://app/abc"}
    )
    assert result is None
    assert reason == "blob_url"


def test_local_path_reports_reason():
    result, reason = normalize_image_attachment_result(
        {"name": "x", "mime": "image/png", "url": "C:/Users/me/pic.png"}
    )
    assert result is None
    assert reason == "local_path"


def test_valid_data_url_has_no_reason():
    result, reason = normalize_image_attachment_result(
        {"name": "x", "mime": "image/png", "data": "QUJD"}
    )
    assert result is not None
    assert reason is None


def test_thin_wrapper_matches_result_value():
    att = {"name": "x", "mime": "image/png", "url": "blob:http://app/abc"}
    assert normalize_image_attachment(att) == normalize_image_attachment_result(att)[0]


def test_describe_rejections_only_reports_fixable():
    attachments = [
        {"name": "good", "mime": "image/png", "data": "QUJD"},
        {"name": "blobby", "mime": "image/png", "url": "blob:http://app/z"},
        {"name": "ondisk", "mime": "image/png", "url": "./pic.png"},
    ]
    messages = describe_attachment_rejections(attachments)
    assert len(messages) == 2
    assert any("blobby" in m and "blob" in m.lower() for m in messages)
    assert any("ondisk" in m for m in messages)
