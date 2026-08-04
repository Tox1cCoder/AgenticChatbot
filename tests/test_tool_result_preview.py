from __future__ import annotations

import json

from app.services.tool_result_preview import build_tool_result_preview


def _payload(result_count: int, content_chars: int) -> str:
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
            "answer": "a" * 400,
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
    preview = build_tool_result_preview(_payload(3, 1000), budget_chars=4000, answer_share=0.25)

    assert len(json.loads(preview.text)["answer"]) <= 1000


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


def test_non_json_payload_keeps_character_prefix():
    preview = build_tool_result_preview("x" * 100, budget_chars=10)

    assert preview.structured is False
    assert preview.text == "x" * 10
    assert preview.omitted_arrays == ()


def test_json_without_a_results_list_keeps_character_prefix():
    preview = build_tool_result_preview(json.dumps({"provider": "x", "note": "y" * 100}), budget_chars=20)

    assert preview.structured is False
    assert len(preview.text) <= 20
