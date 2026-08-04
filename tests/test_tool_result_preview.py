from __future__ import annotations

import json

from app.services.tool_result_preview import build_tool_result_preview


def _payload(result_count: int, content_chars: int, *, answer_chars: int = 400) -> str:
    return json.dumps(
        {
            "results": [
                {
                    "index": index,
                    "title": f"Title {index}",
                    "url": f"https://example.com/{index}",
                    "content": "c" * content_chars,
                    "score": 0.9,
                }
                for index in range(1, result_count + 1)
            ],
            "total_results": result_count,
            "answer": "a" * answer_chars,
            "provider": "tavily",
            "operation": "search",
            "query": "t1 league of legends",
            "images": [{"url": f"https://cdn.example/{i}.jpg"} for i in range(24)],
            "usage": {"credits": 1},
        }
    )


def test_structured_preview_keeps_every_result_and_drops_the_array():
    preview = build_tool_result_preview(_payload(5, 3000), budget_chars=4000)

    assert preview.structured is True
    parsed = json.loads(preview.text)
    assert [entry["title"] for entry in parsed["results"]] == [
        "Title 1",
        "Title 2",
        "Title 3",
        "Title 4",
        "Title 5",
    ]
    assert all(entry["url"].startswith("https://example.com/") for entry in parsed["results"])
    assert "images" not in parsed
    assert preview.omitted_arrays == (("images", 24),)
    assert preview.omitted_results == 0
    assert len(preview.text) <= 4000


def test_answer_is_capped_at_its_configured_share():
    # answer_chars=3000 exceeds the 4000 * 0.25 = 1000 cap, so this fails if the
    # answer-slicing code is removed (unlike a 400-char answer, which is under
    # the cap regardless and can't tell a working cap from a missing one).
    preview = build_tool_result_preview(
        _payload(3, 1000, answer_chars=3000), budget_chars=4000, answer_share=0.25
    )

    assert len(json.loads(preview.text)["answer"]) == 1000


def test_titles_and_urls_are_never_truncated_when_content_is():
    preview = build_tool_result_preview(_payload(4, 5000), budget_chars=1200)

    parsed = json.loads(preview.text)
    assert parsed["results"][0]["title"] == "Title 1"
    assert parsed["results"][0]["url"] == "https://example.com/1"
    assert len(parsed["results"][0]["content"]) < 5000


def test_later_results_are_dropped_whole_below_the_content_floor():
    preview = build_tool_result_preview(
        _payload(10, 4000), budget_chars=1000, min_result_content_chars=200
    )

    parsed = json.loads(preview.text)
    assert 1 <= len(parsed["results"]) < 10
    assert preview.omitted_results == 10 - len(parsed["results"])
    assert all(len(entry["content"]) >= 200 for entry in parsed["results"])


def test_budget_below_shell_size_falls_back_to_character_prefix():
    # identity keys + capped answer alone already exceed a 50-char budget, so
    # no valid structured JSON can fit — this must not slice the JSON dump
    # mid-string and claim structured=True over invalid text.
    payload = _payload(3, 500)

    preview = build_tool_result_preview(payload, budget_chars=50, min_result_content_chars=200)

    assert preview.structured is False
    assert preview.text == payload[:50].rstrip()


def test_structured_previews_are_always_valid_json():
    payload = _payload(5, 3000)
    budgets_and_floors = [(4000, 200), (1200, 200), (1000, 200), (100, 50), (50, 200), (10, 1)]

    for budget_chars, floor in budgets_and_floors:
        preview = build_tool_result_preview(
            payload, budget_chars=budget_chars, min_result_content_chars=floor
        )
        if preview.structured:
            json.loads(preview.text)  # must not raise


def test_non_json_payload_keeps_character_prefix():
    preview = build_tool_result_preview("x" * 100, budget_chars=10)

    assert preview.structured is False
    assert preview.text == "x" * 10
    assert preview.omitted_arrays == ()


def test_json_without_a_results_list_keeps_character_prefix():
    preview = build_tool_result_preview(json.dumps({"provider": "x", "note": "y" * 100}), budget_chars=20)

    assert preview.structured is False
    assert len(preview.text) <= 20
