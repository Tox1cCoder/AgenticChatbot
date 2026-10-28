"""Readable citation labels survive both public presentation paths."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.ai.web_research.grounding import GroundingParser
from app.services.event_streaming.ai_sdk_projection import project_ai_sdk_message_for_capability
from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event
from app.ui.stream_markdown import normalize_display_markdown_text, normalize_stream_markdown_text


def _grounded_answer():
    sources = [
        SimpleNamespace(
            source_id="S12", url="https://www.example.com/article", title="Untrusted [title]"
        ),
        SimpleNamespace(source_id="S45", url="https://docs.example.org/guide", title=None),
    ]
    session = SimpleNamespace(
        source_registry=SimpleNamespace(
            resolve=lambda value: next(
                (source for source in sources if value in (source.source_id, source.url)), None
            )
        ),
        prepared_images={},
    )
    return GroundingParser(session).resolve(
        "Claim [[source:S12]][[source:S45]]. Again [[source:S12]]."
    )


EXPECTED = (
    "Claim ([example.com](https://www.example.com/article))"
    "([docs.example.org](https://docs.example.org/guide)). "
    "Again ([example.com](https://www.example.com/article))."
)


def test_sparse_source_ids_render_as_stable_domain_labels_in_streamlit():
    result = _grounded_answer()

    assert result.text == EXPECTED
    assert result.source_ids == ("S12", "S45")
    assert normalize_stream_markdown_text(result.text) == EXPECTED
    assert normalize_display_markdown_text(result.text) == EXPECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_web_answer", [False, True])
async def test_domain_labels_match_ai_sdk_stream_completion_and_history(stream_web_answer):
    final = _grounded_answer().text
    persisted = {"id": "m-1", "role": "assistant", "content": final}

    async def events():
        yield make_event(
            "answer_preview", sequence=0, data={"operation": "replace", "text": "Claim "}
        )
        yield make_event("message_delta", sequence=1, data={"text": final})
        yield make_event("complete", sequence=2, data={"message": persisted})

    adapter = AISDKV6StreamAdapter(
        events,
        AISDKV6StreamState(
            message_id="m-1",
            text_id="t-1",
            reasoning_id="r-1",
            stream_web_answer_v1=stream_web_answer,
        ),
    )
    payloads = [
        json.loads(line[6:])
        for chunk in [part async for part in adapter.iter_sse()]
        for line in chunk.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    if stream_web_answer:
        assert [p["data"]["content"] for p in payloads if p["type"] == "data-answer-reconcile"] == [
            EXPECTED
        ]
    else:
        assert "".join(p["delta"] for p in payloads if p["type"] == "text-delta") == EXPECTED
    history = project_ai_sdk_message_for_capability(persisted, inline_rich_response_v1=True)
    assert history["content"] == EXPECTED
