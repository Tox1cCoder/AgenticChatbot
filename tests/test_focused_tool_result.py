"""Bounded, deterministic evidence selection from a large tool payload.

This utility is the difference between "the model can read the blob" and "the
model can page the blob forever". Everything it returns is addressable back to
a position in the original payload and bounded before it is serialized.
"""

from __future__ import annotations

import json

import pytest

from app.ai.focused_tool_result import (
    _CANDIDATE_MAX_CHARS,
    FocusedResult,
    _chunks,
    select_focused_excerpts,
)


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


def test_two_sources_disagreeing_on_a_number_both_survive():
    """Lexical similarity is not factual equivalence.

    Two sources that state the same fact with different numbers are the whole
    reason to read both. Collapsing them into one excerpt destroys the only
    evidence that they disagree.
    """
    payload = json.dumps(
        [
            {
                "url": f"https://{letter}.example",
                "content": "The official annual revenue for the entire northern business "
                f"division in fiscal year 2026 was {amount} million dollars.",
            }
            for letter, amount in (("a", 100), ("b", 200))
        ]
    )

    result = _select(payload, "annual revenue 2026")

    assert len(result.excerpts) == 2
    combined = " ".join(item.text for item in result.excerpts)
    assert "100 million" in combined
    assert "200 million" in combined


def test_a_negated_restatement_is_not_collapsed_into_its_opposite():
    payload = json.dumps(
        {
            "a": "The migration deadline is enforced for every tenant on 30 June.",
            "b": "The migration deadline is not enforced for every tenant on 30 June.",
        }
    )

    result = _select(payload, "Is the migration deadline enforced?")

    assert len(result.excerpts) == 2


def test_two_dates_for_the_same_event_both_survive():
    payload = json.dumps(
        {
            "a": "Aurora 4.2 was released on 14 March 2026 according to the vendor.",
            "b": "Aurora 4.2 was released on 21 March 2026 according to the vendor.",
        }
    )

    result = _select(payload, "Which release date is stated for Aurora 4.2?")

    assert len(result.excerpts) == 2


def test_a_leaf_without_spaces_is_chunked_within_the_bound_and_keeps_every_character():
    """Unspaced text has no split point; it must still be bounded, not mangled."""
    body = "\u754c" * 5_000

    chunks, omitted = _chunks("$.content", body)

    assert omitted == 0
    assert all(len(text) <= _CANDIDATE_MAX_CHARS for _, text in chunks)
    assert "".join(text for _, text in chunks) == body


def test_an_unbroken_machine_generated_token_keeps_its_final_character():
    body = "A" * 2_500 + "Z"

    chunks, omitted = _chunks("$.token", body)

    assert omitted == 0
    assert "".join(text for _, text in chunks) == body
    assert chunks[-1][1].endswith("Z")


def test_numeric_json_facts_are_readable_through_their_key():
    """A number is a fact. The reader replaced raw paging, which could see it."""
    payload = json.dumps({"annual_revenue": 42_000_000, "year": 2026})

    result = _select(payload, "annual revenue")

    assert result.excerpts
    assert result.excerpts[0].source_path == "$.annual_revenue"
    assert "42000000" in result.excerpts[0].text


def test_a_boolean_json_fact_is_readable_through_its_key():
    payload = json.dumps({"audit_passed": False, "note": "unrelated prose about terns"})

    result = _select(payload, "audit passed")

    assert result.excerpts
    assert "false" in result.excerpts[0].text.lower()


def test_a_scalar_inside_a_list_is_addressed_and_labelled_by_its_key():
    payload = json.dumps({"latencies_ms": [12, 4210]})

    result = _select(payload, "latencies ms")

    assert [item.source_path for item in result.excerpts] == [
        "$.latencies_ms[0]",
        "$.latencies_ms[1]",
    ]


def test_a_null_leaf_is_never_reported_as_a_fact():
    payload = json.dumps({"published_date": None, "summary": "the deadline is 30 June 2026"})

    result = _select(payload, "published date")

    assert not any("published_date" in item.text for item in result.excerpts)


def test_omitted_candidates_counts_what_the_budget_actually_dropped():
    """Omissions are recorded after trimming, not before it."""
    payload = json.dumps(
        {f"k{index}": f"deadline detail {index} " + str(index) * 800 for index in range(4)}
    )

    result = _select(payload, "deadline", max_chars=1_000)

    assert result.total_candidates == 4
    assert result.omitted_candidates > 0
    assert result.omitted_candidates == result.total_candidates - len(result.excerpts)


def test_an_opposite_sign_is_not_treated_as_the_same_number():
    """Normalization strips the minus; the claim is the opposite of the other."""
    payload = json.dumps(
        {
            "a": "The operating margin for the northern division was 10 percent.",
            "b": "The operating margin for the northern division was -10 percent.",
        }
    )

    result = _select(payload, "operating margin")

    assert len(result.excerpts) == 2


def test_a_reordered_version_is_not_treated_as_the_same_version():
    payload = json.dumps(
        {
            "a": "The supported runtime for this connector is version 4.2 exactly.",
            "b": "The supported runtime for this connector is version 2.4 exactly.",
        }
    )

    result = _select(payload, "supported runtime version")

    assert len(result.excerpts) == 2


def test_two_different_months_are_not_treated_as_the_same_date():
    payload = json.dumps(
        {
            "a": "The tenant migration window closes on 14 March 2026 for everyone.",
            "b": "The tenant migration window closes on 14 April 2026 for everyone.",
        }
    )

    result = _select(payload, "when does the migration window close")

    assert len(result.excerpts) == 2


def test_a_percentage_written_with_a_symbol_still_distinguishes_the_claim():
    payload = json.dumps(
        {
            "a": "Retrieval coverage across the whole corpus reached 95% last quarter.",
            "b": "Retrieval coverage across the whole corpus reached 59% last quarter.",
        }
    )

    result = _select(payload, "retrieval coverage")

    assert len(result.excerpts) == 2


@pytest.mark.parametrize("length", [1_201, 1_202, 2_401, 2_402])
def test_a_short_split_remainder_is_kept_rather_than_dropped(length):
    """The end of a split passage can hold the final digit of the fact.

    A fragment this short is not evidence standing alone, which is why it is
    dropped when it *is* the whole leaf. As the tail of a passage the splitter
    cut, dropping it silently loses characters and records no omission.
    """
    body = "A" * length

    chunks, omitted = _chunks("$.token", body)

    assert omitted == 0
    assert "".join(text for _, text in chunks) == body
    assert all(len(text) <= _CANDIDATE_MAX_CHARS for _, text in chunks)


def test_a_leaf_too_short_to_be_evidence_is_still_dropped():
    """The standalone case keeps its floor: two characters answer nothing."""
    assert _chunks("$.tiny", "ok") == ([], 0)
