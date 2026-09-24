"""Tool results from a device are bounded before they cross the runtime bridge.

Text is cut in the middle, because command output carries its error or summary
at the end. Images are kept whole or dropped whole: half an image is corrupt
data, not a smaller image.
"""

from __future__ import annotations

import base64
import json

from shared.runtime_results import cap_tool_result

TEXT_BUDGET = 2_000
MEDIA_BUDGET = 3_000


def _wire_size(value) -> int:
    # The bridge sends json.dumps with the default ensure_ascii=True.
    return len(json.dumps(value))


def _image(decoded_bytes: int, mime: str = "image/png") -> dict:
    data = base64.b64encode(b"\x89" * decoded_bytes).decode("ascii")
    return {"type": "image", "data": data, "mimeType": mime}


def _cap(result):
    return cap_tool_result(result, max_text_bytes=TEXT_BUDGET, max_media_bytes=MEDIA_BUDGET)


def test_result_within_budget_is_returned_unchanged():
    result = [{"type": "text", "text": "ok"}, _image(1_000)]

    capped, truncated = _cap(result)

    assert capped == result
    assert truncated is False


def test_long_text_keeps_its_beginning_and_end():
    text = "BEGIN " + ("x" * 50_000) + " END"

    capped, truncated = _cap([{"type": "text", "text": text}])

    kept = capped[0]["text"]
    assert truncated is True
    assert kept.startswith("BEGIN ")
    assert kept.endswith(" END")
    assert "omitted" in kept
    assert _wire_size(capped) <= TEXT_BUDGET


def test_text_budget_counts_the_escaped_size_of_non_ascii_text():
    # Each "đ" is sent as the six characters đ.
    capped, truncated = _cap([{"type": "text", "text": "đ" * 5_000}])

    assert truncated is True
    assert _wire_size(capped) <= TEXT_BUDGET


def test_image_within_media_budget_survives_text_truncation():
    image = _image(2_000)
    result = [{"type": "text", "text": "y" * 50_000}, image]

    capped, truncated = _cap(result)

    assert truncated is True
    assert image in capped


def test_image_over_media_budget_is_dropped_whole_with_a_note():
    result = [{"type": "text", "text": "screenshot taken"}, _image(10_000)]

    capped, truncated = _cap(result)

    assert truncated is True
    assert not [block for block in capped if block.get("type") == "image"]
    notes = [block["text"] for block in capped if block.get("type") == "text"]
    assert any("image/png" in note and "omitted" in note for note in notes)


def test_images_are_kept_in_order_until_the_media_budget_is_spent():
    first, second, third = _image(1_400), _image(1_400, "image/jpeg"), _image(1_400)

    capped, _ = _cap([first, second, third])

    images = [block for block in capped if block.get("type") == "image"]
    assert images == [first, second]


def test_plain_string_result_keeps_its_beginning_and_end():
    capped, truncated = _cap("start-" + ("z" * 50_000) + "-finish")

    assert truncated is True
    assert capped.startswith("start-")
    assert capped.endswith("-finish")
    assert _wire_size(capped) <= TEXT_BUDGET


def test_oversized_structured_result_becomes_bounded_text():
    capped, truncated = _cap({"rows": [{"n": index} for index in range(5_000)]})

    assert truncated is True
    assert isinstance(capped, str)
    assert capped.startswith('{"rows": [{"n": 0}')
    assert _wire_size(capped) <= TEXT_BUDGET
