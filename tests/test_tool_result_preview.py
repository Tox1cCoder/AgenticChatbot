from __future__ import annotations

import json
import random
import time

from app.services import tool_result_preview
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
    preview = build_tool_result_preview(
        json.dumps({"provider": "x", "note": "y" * 100}), budget_chars=20
    )

    assert preview.structured is False
    assert len(preview.text) <= 20


def _extract_payload(pages: int, body_chars: int) -> str:
    """A ``tavily_extract`` payload: page text under ``raw_content``, no ``content``."""

    return json.dumps(
        {
            "provider": "tavily",
            "operation": "extract",
            "urls": [f"https://example.com/page/{index}" for index in range(pages)],
            "results": [
                {
                    "url": f"https://example.com/page/{index}",
                    "raw_content": f"PAGE-{index}-FACT " + "extracted body text. " * body_chars,
                    "images": [f"https://cdn.example/{index}-{n}.jpg" for n in range(6)],
                }
                for index in range(pages)
            ],
            "failed_results": [],
            "usage": {"credits": 2},
            "request_id": "req-9",
            "response_time": 1.2,
        }
    )


def test_extract_payload_keeps_page_text_instead_of_only_urls():
    # The whitelist that kept index/title/url/score/content turned a 41 KB
    # extract into 189 characters of URLs with zero page text, because extract
    # carries its body under raw_content.
    preview = build_tool_result_preview(_extract_payload(2, 200), budget_chars=4000)

    assert preview.structured is True
    assert len(preview.text) <= 4000
    parsed = json.loads(preview.text)
    assert len(parsed["results"]) >= 1
    for entry in parsed["results"]:
        assert len(entry["raw_content"]) >= 200
        assert "extracted body text" in entry["raw_content"]
    assert "PAGE-0-FACT" in parsed["results"][0]["raw_content"]


def test_extract_preview_reports_what_it_stripped_from_inside_each_result():
    preview = build_tool_result_preview(_extract_payload(2, 200), budget_chars=4000)

    assert "images" in preview.omitted_result_keys
    assert "raw_content" in preview.shortened_keys
    assert "request_id" in preview.omitted_keys
    assert "response_time" in preview.omitted_keys
    assert "usage" in preview.omitted_keys
    assert ("failed_results", 0) not in preview.omitted_arrays


def test_map_payload_keeps_its_plain_string_entries():
    # 400 URL strings became 18 empty objects: total information loss.
    payload = json.dumps(
        {
            "provider": "tavily",
            "operation": "map",
            "base_url": "https://example.com",
            "results": [f"https://example.com/docs/page-{index}" for index in range(400)],
            "total_results": 400,
            "usage": {"credits": 1},
        }
    )

    preview = build_tool_result_preview(payload, budget_chars=4000)

    assert preview.structured is True
    parsed = json.loads(preview.text)
    assert len(parsed["results"]) >= 10
    assert all(entry.startswith("https://example.com/docs/page-") for entry in parsed["results"])
    assert preview.omitted_results == 400 - len(parsed["results"])


def test_unknown_shape_keeps_every_scalar_and_its_long_text():
    payload = json.dumps(
        {
            "tool": "run_sql",
            "results": [
                {
                    "row": 1,
                    "customer": "Acme Manufacturing",
                    "notes": "MARKER-NOTES " + "escalation detail. " * 100,
                    "active": True,
                    "balance": 1234.5,
                },
                "trailing summary row that is a bare string",
                42,
            ],
            "columns": ["row", "customer", "notes", "active", "balance"],
            "elapsed_ms": 12,
        }
    )

    preview = build_tool_result_preview(payload, budget_chars=4000)

    assert preview.structured is True
    parsed = json.loads(preview.text)
    first = parsed["results"][0]
    assert first["row"] == 1
    assert first["customer"] == "Acme Manufacturing"
    assert first["active"] is True
    assert first["balance"] == 1234.5
    assert first["notes"].startswith("MARKER-NOTES ")
    assert len(first["notes"]) >= 200
    assert parsed["results"][1] == "trailing summary row that is a bare string"
    assert parsed["results"][2] == 42


def _escape_heavy_payload(result_count: int) -> str:
    snippet = 'He said "hello".\nPath: C:\\temp\\file\tdone. ' * 100
    return json.dumps(
        {
            "results": [
                {
                    "index": index,
                    "title": f"Title {index}",
                    "url": f"https://example.com/{index}",
                    "content": snippet,
                    "score": 0.9,
                }
                for index in range(1, result_count + 1)
            ],
            "total_results": result_count,
            "answer": 'The answer is "42".\nUse C:\\bin for the path.',
            "provider": "tavily",
            "operation": "search",
            "query": "escapes",
        }
    )


def test_escape_heavy_content_stays_on_the_structured_path():
    # Budgeting raw characters while json.dumps expands " \\ \n \t to two meant
    # ordinary scraped prose overflowed and fell back at every budget, so the
    # structured path did nothing on realistic input.
    payload = _escape_heavy_payload(5)

    for budget in (4000, 2000, 1200, 800):
        preview = build_tool_result_preview(payload, budget_chars=budget)

        assert preview.structured is True, f"fell back at budget {budget}"
        assert len(preview.text) <= budget
        parsed = json.loads(preview.text)
        assert parsed["results"]
        assert all('"hello"' in entry["content"] for entry in parsed["results"])
        assert parsed["answer"].startswith('The answer is "42".')


def test_large_result_count_builds_with_a_bounded_number_of_dumps(monkeypatch):
    # 8000 entries took 25.9 s of blocked event loop because the fit loop walked
    # down from len(results), re-dumping the whole scaffold each step.
    payload = _payload(8000, 300)
    dumps: list[int] = []
    real_dump = tool_result_preview._dump

    def counting_dump(payload_obj):
        dumps.append(1)
        return real_dump(payload_obj)

    monkeypatch.setattr(tool_result_preview, "_dump", counting_dump)

    started = time.perf_counter()
    preview = build_tool_result_preview(payload, budget_chars=4000)
    elapsed = time.perf_counter() - started

    assert preview.structured is True
    assert len(dumps) <= 12, f"{len(dumps)} scaffold dumps for 8000 results"
    assert elapsed < 2.0, f"took {elapsed:.2f}s"
    assert 1 <= len(json.loads(preview.text)["results"]) <= 20


_FUZZ_ALPHABET = 'abc def."\\' + "\n\t\r\x01é中"


def _fuzz_payload(rnd: random.Random) -> str:
    def text(length: int) -> str:
        return "".join(rnd.choice(_FUZZ_ALPHABET) for _ in range(length))

    entries: list[object] = []
    for index in range(rnd.randint(0, 10)):
        roll = rnd.random()
        if roll < 0.6:
            entry: dict[str, object] = {
                "index": index,
                "title": text(rnd.randint(0, 40)),
                "url": "https://e.example/" + text(rnd.randint(0, 20)),
                "content": text(rnd.randint(0, 900)),
                "score": rnd.random(),
            }
            if rnd.random() < 0.5:
                entry["raw_content"] = text(rnd.randint(0, 2000))
            if rnd.random() < 0.3:
                entry["images"] = [text(20) for _ in range(rnd.randint(0, 5))]
            entries.append(entry)
        elif roll < 0.85:
            entries.append(text(rnd.randint(0, 300)))
        else:
            entries.append(rnd.choice([42, 3.5, True, None, [1, 2], {"a": text(50)}]))

    payload: dict[str, object] = {
        "results": entries,
        "provider": "tavily",
        "operation": text(8),
        "query": text(rnd.randint(0, 60)),
        "total_results": len(entries),
        "answer": text(rnd.randint(0, 1500)),
    }
    if rnd.random() < 0.5:
        payload["images"] = [text(10) for _ in range(rnd.randint(0, 20))]
    return json.dumps(payload, ensure_ascii=False)


def test_structured_text_never_exceeds_its_budget_on_escape_heavy_payloads():
    # The fit guarantee is what makes the structured path usable: it is measured
    # in serialized characters, so escapes must not push the dump over budget
    # for any shape or any budget.
    rnd = random.Random(20260804)

    for _ in range(400):
        payload = _fuzz_payload(rnd)
        budget = rnd.choice([400, 800, 1200, 2000, 4000, 16000])
        preview = build_tool_result_preview(
            payload,
            budget_chars=budget,
            answer_share=rnd.choice([0.0, 0.25, 0.5]),
            min_result_content_chars=rnd.choice([1, 50, 200, 500]),
        )

        assert len(preview.text) <= budget
        if budget >= 1200:
            # Identity keys plus an answer capped at half the budget always fit
            # 1200 characters, so the only way to fall back here is a content
            # allocation that overflowed.
            assert preview.structured is True, f"fell back at budget {budget}"
        if preview.structured:
            json.loads(preview.text)  # must not raise
