from __future__ import annotations

from app.ai.utils import extract_public_thinking_summary


def test_extracts_gemini_thinking_block() -> None:
    content = [
        {"type": "thinking", "thinking": "Checking the constraints"},
        {"type": "text", "text": "Answer"},
    ]
    assert extract_public_thinking_summary(content) == "Checking the constraints"


def test_ignores_signature_only_thinking_block() -> None:
    assert (
        extract_public_thinking_summary(
            [{"type": "thinking", "signature": "encrypted"}]
        )
        is None
    )


def test_extracts_openai_summary_without_answer_text() -> None:
    content = [
        {
            "type": "reasoning",
            "summary": [{"type": "text", "text": "Compared options"}],
        },
        {"type": "text", "text": "Answer"},
    ]
    assert extract_public_thinking_summary(content) == "Compared options"


def test_can_filter_provider_block_type() -> None:
    content = [
        {"type": "thinking", "thinking": "Gemini"},
        {"type": "reasoning", "summary": "OpenAI"},
    ]
    assert extract_public_thinking_summary(content, block_types={"thinking"}) == "Gemini"
