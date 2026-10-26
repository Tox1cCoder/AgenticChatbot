"""Bounded static answer visuals: validation, evidence binding, and fallback.

The fallback Markdown is derived from the same validated content as the public
payload, so these tests pin both shapes from one input.
"""

from __future__ import annotations

import json
import math

import pytest

from app.core.answer_visuals import (
    AnswerVisualError,
    ResolvedEvidence,
    build_answer_visual,
)
from app.core.rich_response import validate_public_rich_item

_SOURCES = {
    "S1": ResolvedEvidence(title="Vendor spec sheet", url="https://vendor.example/spec"),
    "S2": ResolvedEvidence(title="Bench review", url="https://www.bench.example/review"),
}


def _resolve(ref):
    return _SOURCES.get(ref.id)


def _build(spec: dict, *, turn_key: str = "turn-1") -> dict:
    return build_answer_visual(spec, resolve_evidence=_resolve, turn_key=turn_key)


def _public(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "_internal"}


def _steps_spec(**overrides) -> dict:
    spec = {
        "variant": "steps",
        "title": "Make Google the default search engine",
        "steps": [
            {"title": "Open Edge settings"},
            {"title": "Go to Privacy, search, and services"},
            {
                "title": "Choose Google under Search engine used in the address bar",
                "note": "The list only shows engines you have visited.",
            },
        ],
    }
    spec.update(overrides)
    return spec


def _bars_spec(**overrides) -> dict:
    spec = {
        "variant": "bars",
        "title": "Average FPS",
        "unit": "FPS",
        "basis": "measured",
        "bars": [
            {"label": "Stock", "value": 100, "evidence": [{"kind": "web", "id": "S2"}]},
            {"label": "Overclocked", "value": 101, "evidence": [{"kind": "web", "id": "S2"}]},
        ],
    }
    spec.update(overrides)
    return spec


def _table_spec(**overrides) -> dict:
    spec = {
        "variant": "comparison_table",
        "title": "Tweak assessment",
        "columns": ["Effect"],
        "rows": [
            {
                "label": "Disable HPET",
                "cells": [{"text": "No measurable gain"}],
                "assessment": {
                    "status": "caution",
                    "label": "Skip",
                    "reason": "Benchmarks show no change | and possible timer issues.",
                    "evidence": [{"kind": "web", "id": "S1"}],
                },
            },
        ],
    }
    spec.update(overrides)
    return spec


# ---------------------------------------------------------------------------
# Variants and exact fallback
# ---------------------------------------------------------------------------


def test_steps_visual_is_a_valid_inline_only_rich_item_with_exact_fallback():
    item = _build(_steps_spec())

    validated = validate_public_rich_item(_public(item))
    assert validated.type == "answer_visual"
    assert item["display_policy"] == "inline_only"
    assert item["payload"]["variant"] == "steps"
    assert item["payload"]["fallback_markdown"] == (
        "**Make Google the default search engine**\n"
        "\n"
        "1. Open Edge settings\n"
        "2. Go to Privacy, search, and services\n"
        "3. Choose Google under Search engine used in the address bar"
        " — The list only shows engines you have visited."
    )


def test_steps_payload_carries_no_execution_state():
    item = _build(_steps_spec())

    serialized = json.dumps(item["payload"])
    assert "done" not in serialized
    assert "status" not in serialized


def test_bars_visual_uses_a_shared_zero_based_scale_and_exact_fallback():
    item = _build(_bars_spec())

    payload = item["payload"]
    assert payload["scale"] == {"min": 0, "max": 101}
    assert [bar["value"] for bar in payload["bars"]] == [100, 101]
    assert payload["evidence"] == [
        {
            "title": "Bench review",
            "url": "https://www.bench.example/review",
            "label": "bench.example",
        }
    ]
    assert payload["bars"][0]["citations"] == [1]
    assert payload["fallback_markdown"] == (
        "**Average FPS**\n"
        "\n"
        "| Item | Value (FPS) | Sources |\n"
        "| --- | --- | --- |\n"
        "| Stock | 100 | [bench.example](https://www.bench.example/review) |\n"
        "| Overclocked | 101 | [bench.example](https://www.bench.example/review) |\n"
        "\n"
        "Basis: measured. Scale: 0 to 101 FPS."
    )


def test_illustrative_bars_disclose_their_assumptions():
    item = _build(
        _bars_spec(
            basis="illustrative",
            assumptions="Assumes FPS scales linearly with core clock.",
            bars=[{"label": "Stock", "value": 100}, {"label": "+1% clock", "value": 101}],
        )
    )

    fallback = item["payload"]["fallback_markdown"]
    assert "Basis: illustrative — Assumes FPS scales linearly with core clock." in fallback


def test_all_zero_bars_get_a_unit_scale():
    item = _build(
        _bars_spec(
            basis="illustrative",
            assumptions="Placeholder values.",
            bars=[{"label": "A", "value": 0}, {"label": "B", "value": 0}],
        )
    )

    assert item["payload"]["scale"] == {"min": 0, "max": 1}


def test_comparison_table_escapes_cells_and_carries_badge_text():
    item = _build(_table_spec())

    payload = item["payload"]
    assessment = payload["rows"][0]["assessment"]
    assert assessment["status"] == "caution"
    assert assessment["label"] == "Skip"
    assert payload["fallback_markdown"] == (
        "**Tweak assessment**\n"
        "\n"
        "| Item | Effect | Assessment |\n"
        "| --- | --- | --- |\n"
        "| Disable HPET | No measurable gain | Caution: Skip — Benchmarks show no change \\|"
        " and possible timer issues. [vendor.example](https://vendor.example/spec) |"
    )


def test_fallback_neutralizes_rich_markers_in_model_text():
    item = _build(_steps_spec(steps=[{"title": "Use <!--rich:other--> here"}]))

    fallback = item["payload"]["fallback_markdown"]
    assert "<!--rich:" not in fallback


def test_caption_and_its_citations_appear_in_fallback():
    item = _build(
        _steps_spec(caption="Edge 120 menus.", caption_evidence=[{"kind": "web", "id": "S1"}])
    )

    assert item["payload"]["fallback_markdown"].endswith(
        "\n\nEdge 120 menus. [vendor.example](https://vendor.example/spec)"
    )
    assert item["payload"]["caption_citations"] == [1]


# ---------------------------------------------------------------------------
# Bounds and numeric integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-1, math.inf, math.nan])
def test_bars_reject_negative_and_nonfinite_values(value):
    spec = _bars_spec(basis="illustrative", assumptions="x", bars=[{"label": "A", "value": value}])

    with pytest.raises(AnswerVisualError):
        _build(spec)


def test_bars_reject_a_maximum_below_a_value():
    with pytest.raises(AnswerVisualError, match="max_value"):
        _build(_bars_spec(max_value=50))


def test_bars_reject_an_explicit_zero_maximum():
    spec = _bars_spec(
        basis="illustrative", assumptions="x", max_value=0, bars=[{"label": "A", "value": 0}]
    )

    with pytest.raises(AnswerVisualError, match="max_value"):
        _build(spec)


def test_calculated_bars_require_assumptions():
    with pytest.raises(AnswerVisualError, match="assumptions"):
        _build(_bars_spec(basis="calculated"))


def test_measured_bars_require_evidence_on_every_bar():
    spec = _bars_spec(bars=[{"label": "Stock", "value": 100}])

    with pytest.raises(AnswerVisualError, match="evidence"):
        _build(spec)


def test_bars_require_a_unit():
    with pytest.raises(AnswerVisualError):
        _build(_bars_spec(unit=None))


def test_at_most_twelve_steps():
    with pytest.raises(AnswerVisualError):
        _build(_steps_spec(steps=[{"title": f"Step {n}"} for n in range(13)]))


def test_at_most_six_columns():
    row = {"label": "r", "cells": [{"text": "x"} for _ in range(7)]}
    with pytest.raises(AnswerVisualError):
        _build(_table_spec(columns=[f"c{n}" for n in range(7)], rows=[row]))


def test_at_most_twelve_rows():
    row = {"label": "r", "cells": [{"text": "x"}]}
    with pytest.raises(AnswerVisualError):
        _build(_table_spec(rows=[row] * 13))


def test_row_cells_must_match_columns():
    with pytest.raises(AnswerVisualError, match="cells"):
        _build(_table_spec(rows=[{"label": "r", "cells": [{"text": "a"}, {"text": "b"}]}]))


def test_labels_are_bounded_not_truncated():
    with pytest.raises(AnswerVisualError):
        _build(_steps_spec(title="x" * 161))


def test_fields_of_another_variant_are_rejected():
    with pytest.raises(AnswerVisualError):
        _build(_steps_spec(unit="FPS"))


def test_serialized_public_item_is_bounded(monkeypatch):
    import app.core.answer_visuals as answer_visuals

    monkeypatch.setattr(answer_visuals, "ANSWER_VISUAL_MAX_ITEM_BYTES", 600)

    with pytest.raises(AnswerVisualError, match="bytes"):
        _build(_table_spec())


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_unknown_evidence_is_rejected_with_the_offending_id():
    spec = _bars_spec(
        bars=[{"label": "Stock", "value": 100, "evidence": [{"kind": "web", "id": "S9"}]}]
    )

    with pytest.raises(AnswerVisualError, match="S9"):
        _build(spec)


def test_public_payload_never_carries_turn_local_source_tokens():
    item = _build(_bars_spec())

    public = json.dumps(item["payload"])
    assert '"S2"' not in public
    assert "S2" not in item["payload"]["fallback_markdown"]


def test_visual_only_source_ids_are_recorded_privately():
    item = _build(_bars_spec())

    assert item["_internal"]["web_source_ids"] == ["S2"]


def test_internal_record_is_stripped_before_public_validation():
    validate_public_rich_item(_public(_build(_bars_spec())))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_replayed_call_reuses_its_id_and_descriptor():
    assert _build(_steps_spec()) == _build(_steps_spec())


def test_revised_payload_receives_a_new_id():
    first = _build(_steps_spec())
    revised = _build(_steps_spec(title="Make Google the default"))

    assert first["id"] != revised["id"]


def test_same_payload_in_another_turn_receives_a_new_id():
    assert _build(_steps_spec())["id"] != _build(_steps_spec(), turn_key="turn-2")["id"]
