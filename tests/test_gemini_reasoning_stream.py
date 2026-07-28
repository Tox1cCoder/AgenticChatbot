"""Gemini thought summaries must survive the canonical streaming pipeline.

``langchain-google-genai`` emits thoughts as ``{"type": "thinking",
"thinking": ...}``. ``langchain-core`` 1.5 classifies that as a
``non_standard`` content block, and the v3 protocol bridge emits **no**
``content-block-delta`` for non-standard blocks (it also overwrites, rather
than concatenates, per-index values). The result was zero ``reasoning_delta``
events for every Gemini turn, so no live thinking reached Streamlit or the
AI SDK — while OpenAI (standard ``reasoning`` blocks) worked.

Normalizing at the model boundary is what makes the rest of the chain work.
"""

from __future__ import annotations

from langchain_core.language_models._compat_bridge import chunks_to_events
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from app.ai.gemini_content import normalize_gemini_reasoning_blocks
from app.services.event_streaming.langchain_v3 import V3ProtocolTranslator


def _protocol_events(contents):
    chunks = (ChatGenerationChunk(message=AIMessageChunk(content=c)) for c in contents)
    return list(chunks_to_events(chunks))


def _canonical(raw_events):
    translator = V3ProtocolTranslator()
    out = []
    for raw in raw_events:
        envelope = {
            "type": "event",
            "method": "messages",
            "params": {"data": [raw, {"langgraph_node": "chat_agent"}], "namespace": []},
        }
        out.extend(translator.translate(envelope))
    return out


class TestNormalizeGeminiReasoningBlocks:
    def test_thinking_block_becomes_standard_reasoning_block(self):
        content = [{"type": "thinking", "thinking": "step one"}]
        assert normalize_gemini_reasoning_blocks(content) == [
            {"type": "reasoning", "reasoning": "step one"}
        ]

    def test_thought_signature_is_preserved_under_extras(self):
        """langchain-google-genai reads a reasoning block's signature from extras."""
        content = [{"type": "thinking", "thinking": "s", "signature": "YWJj"}]
        assert normalize_gemini_reasoning_blocks(content) == [
            {"type": "reasoning", "reasoning": "s", "extras": {"signature": "YWJj"}}
        ]

    def test_text_and_tool_blocks_are_untouched(self):
        content = [
            {"type": "text", "text": "answer"},
            {"type": "tool_call_chunk", "id": "c1", "name": "t", "args": "{}"},
        ]
        assert normalize_gemini_reasoning_blocks(content) == content

    def test_plain_string_content_is_untouched(self):
        assert normalize_gemini_reasoning_blocks("hello") == "hello"

    def test_already_standard_reasoning_block_is_untouched(self):
        content = [{"type": "reasoning", "reasoning": "x"}]
        assert normalize_gemini_reasoning_blocks(content) == content

    def test_empty_thinking_block_is_dropped(self):
        assert normalize_gemini_reasoning_blocks([{"type": "thinking", "thinking": ""}]) == []


class TestCanonicalStreamCarriesGeminiThinking:
    def test_unnormalized_gemini_thinking_is_recovered_by_the_backstop(self):
        """Un-normalized thinking degrades to one late delta, never to nothing.

        The bridge emits no ``content-block-delta`` for a non-standard block, so
        this text can only come from the terminal ``content-block-finish``.
        """
        events = _canonical(_protocol_events([[{"type": "thinking", "thinking": "abc"}]]))
        assert [e.data["text"] for e in events if e.type == "reasoning_delta"] == ["abc"]

    def test_unnormalized_multi_chunk_thinking_loses_early_text(self):
        """Why normalization at the model boundary matters, not just a backstop.

        The bridge overwrites non-standard blocks per index, so only the last
        chunk survives — normalized blocks keep every delta (see below).
        """
        contents = [
            [{"type": "thinking", "thinking": "part A "}],
            [{"type": "thinking", "thinking": "part B"}],
        ]
        events = _canonical(_protocol_events(contents))
        assert [e.data["text"] for e in events if e.type == "reasoning_delta"] == ["part B"]

    def test_normalized_gemini_thinking_streams_incrementally(self):
        contents = [
            normalize_gemini_reasoning_blocks([{"type": "thinking", "thinking": "part A "}]),
            normalize_gemini_reasoning_blocks([{"type": "thinking", "thinking": "part B"}]),
            [{"type": "text", "text": "Answer!"}],
        ]
        events = _canonical(_protocol_events(contents))

        reasoning = [e.data["text"] for e in events if e.type == "reasoning_delta"]
        assert reasoning == ["part A ", "part B"]

        message = [e.data["text"] for e in events if e.type == "message_delta"]
        assert message == ["Answer!"]

    def test_openai_reasoning_blocks_still_stream(self):
        events = _canonical(_protocol_events([[{"type": "reasoning", "reasoning": "summary"}]]))
        assert [e.data["text"] for e in events if e.type == "reasoning_delta"] == ["summary"]


class TestNonStandardFallbackInTranslator:
    """Defense in depth: a provider that still emits a non-standard thinking
    block must not lose its summary entirely."""

    def test_non_standard_finish_block_yields_reasoning_delta(self):
        translator = V3ProtocolTranslator()
        envelope = {
            "type": "event",
            "method": "messages",
            "params": {
                "data": [
                    {
                        "event": "content-block-finish",
                        "index": 0,
                        "content": {
                            "type": "non_standard",
                            "value": {"type": "thinking", "thinking": "recovered"},
                        },
                    },
                    {"langgraph_node": "chat_agent"},
                ],
                "namespace": [],
            },
        }
        events = list(translator.translate(envelope))
        assert [(e.type, e.data["text"]) for e in events] == [("reasoning_delta", "recovered")]
