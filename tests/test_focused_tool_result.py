"""Bounded, deterministic evidence selection from a large tool payload.

This utility is the difference between "the model can read the blob" and "the
model can page the blob forever". Everything it returns is addressable back to
a position in the original payload and bounded before it is serialized.
"""

from __future__ import annotations

import json

from app.ai.focused_tool_result import FocusedResult, select_focused_excerpts


def _select(payload: str, objective: str, *, max_excerpts: int = 8, max_chars: int = 16_000):
    return select_focused_excerpts(
        payload,
        objective=objective,
        max_excerpts=max_excerpts,
        max_chars=max_chars,
    )


def _search_payload() -> str:
    return json.dumps(
        {
            "results": [
                {
                    "title": "Unrelated ornithology digest",
                    "url": "https://birds.example/digest",
                    "content": "Migration patterns of the arctic tern span pole to pole. "
                    * 40,
                },
                {
                    "title": "Weather almanac",
                    "url": "https://weather.example/almanac",
                    "content": "Rainfall totals for the region remained steady. " * 40,
                },
                {
                    "title": "Aurora 4.2 release notes",
                    "url": "https://vendor.example/aurora/4.2",
                    "content": "Aurora 4.2 was released on 14 March 2026.",
                },
            ],
            "total_results": 3,
            "provider": "tavily",
        }
    )


def test_json_leaves_keep_a_stable_addressable_path():
    result = _select(_search_payload(), "Which release date is stated for Aurora 4.2?")

    assert result.excerpts[0].source_path == "$.results[2].content"


def test_the_matching_passage_outranks_a_longer_unrelated_one():
    result = _select(_search_payload(), "Which release date is stated for Aurora 4.2?")

    assert "14 March 2026" in result.excerpts[0].text
    assert result.excerpts[0].score > 0


def test_urls_and_titles_next_to_the_content_survive_as_source_metadata():
    result = _select(_search_payload(), "Which release date is stated for Aurora 4.2?")
    top = result.excerpts[0]

    assert top.url == "https://vendor.example/aurora/4.2"
    assert top.title == "Aurora 4.2 release notes"


def test_a_bare_url_is_never_returned_as_evidence_text():
    result = _select(_search_payload(), "Which release date is stated for Aurora 4.2?")

    assert not any(item.text.startswith("http") for item in result.excerpts)


def test_plain_text_is_split_into_addressable_paragraphs():
    payload = "\n\n".join(
        [
            "Introductory matter with no facts in it.",
            "A second paragraph about unrelated tooling.",
            "A third paragraph about packaging in general.",
            "The migration deadline is 30 June 2026 for every tenant.",
        ]
    )

    result = _select(payload, "What is the migration deadline?")

    assert result.excerpts[0].source_path == "paragraph[3]"
    assert "30 June 2026" in result.excerpts[0].text


def test_malformed_json_falls_back_to_paragraph_handling():
    payload = '{"results": [{"content": "The deadline is 30 June 2026."'

    result = _select(payload, "What is the deadline?")

    assert result.excerpts
    assert result.excerpts[0].source_path.startswith("paragraph[")
    assert "30 June 2026" in result.excerpts[0].text


def test_exact_duplicates_are_returned_once():
    line = "The deadline is 30 June 2026."
    payload = json.dumps({"a": line, "b": line})

    result = _select(payload, "What is the deadline?")

    assert len(result.excerpts) == 1
    assert result.excerpts[0].source_path == "$.a"


def test_normalized_near_duplicates_are_returned_once():
    payload = json.dumps(
        {
            "a": "The deadline is 30 June 2026.",
            "b": "the   DEADLINE is 30 june 2026!!",
        }
    )

    result = _select(payload, "What is the deadline?")

    assert len(result.excerpts) == 1


def test_the_excerpt_count_never_exceeds_the_requested_maximum():
    payload = json.dumps({f"k{index}": f"deadline detail number {index}" for index in range(50)})

    result = _select(payload, "What is the deadline?", max_excerpts=3)

    assert len(result.excerpts) == 3
    assert result.total_candidates == 50
    assert result.omitted_candidates == 47


def test_the_serialized_result_never_exceeds_the_character_budget():
    payload = json.dumps(
        {f"k{index}": f"deadline paragraph {index}: " + ("evidence " * 200) for index in range(20)}
    )

    result = _select(payload, "What is the deadline?", max_excerpts=8, max_chars=1_500)

    assert len(result.model_dump_json()) <= 1_500
    assert result.truncated is True


def test_a_budget_too_small_for_any_excerpt_still_returns_a_valid_bounded_object():
    payload = json.dumps({"a": "deadline " * 500})

    result = _select(payload, "What is the deadline?", max_chars=200)

    assert isinstance(result, FocusedResult)
    assert len(result.model_dump_json()) <= 200


def test_an_empty_payload_returns_a_bounded_explanation():
    result = _select("", "What is the deadline?")

    assert result.excerpts == []
    assert result.total_candidates == 0
    assert result.note


def test_a_payload_with_nothing_matching_returns_an_explanation_not_the_payload():
    payload = json.dumps({"content": "Arctic tern migration spans pole to pole. " * 50})

    result = _select(payload, "quarterly revenue for the Osaka subsidiary")

    assert result.excerpts == []
    assert result.note
    assert "Arctic tern" not in result.model_dump_json()


def test_the_objective_is_echoed_so_a_repeat_call_is_visible():
    result = _select(_search_payload(), "Which release date is stated for Aurora 4.2?")

    assert result.objective == "Which release date is stated for Aurora 4.2?"


def test_equal_scores_break_ties_on_original_payload_order():
    payload = json.dumps(
        {
            "results": [
                {"content": "deadline alpha"},
                {"content": "deadline beta"},
                {"content": "deadline gamma"},
            ]
        }
    )

    result = _select(payload, "deadline", max_excerpts=3)

    assert [item.source_path for item in result.excerpts] == [
        "$.results[0].content",
        "$.results[1].content",
        "$.results[2].content",
    ]


def test_an_oversized_leaf_is_chunked_so_late_evidence_stays_reachable():
    """A whole page arriving as one JSON string must not hide its tail. The
    chunk keeps an addressable suffix so the model can still cite the spot."""
    payload = json.dumps(
        {"results": [{"raw_content": ("filler sentence. " * 900) + "TAIL-MARKER present."}]}
    )

    result = _select(payload, "Find the TAIL-MARKER value")

    assert result.excerpts
    assert "TAIL-MARKER" in result.excerpts[0].text
    assert result.excerpts[0].source_path.startswith("$.results[0].raw_content")


def test_a_nested_list_of_strings_keeps_index_addressing():
    payload = json.dumps({"chunks": ["nothing here", "the deadline is 30 June 2026"]})

    result = _select(payload, "What is the deadline?")

    assert result.excerpts[0].source_path == "$.chunks[1]"


def test_selection_is_deterministic_across_repeated_calls():
    first = _select(_search_payload(), "Which release date is stated for Aurora 4.2?")
    second = _select(_search_payload(), "Which release date is stated for Aurora 4.2?")

    assert first.model_dump_json() == second.model_dump_json()
