"""Normalize Gemini thought content into standard reasoning blocks.

``langchain-google-genai`` emits Gemini thought parts as the provider-specific
block ``{"type": "thinking", "thinking": "..."}``. ``langchain-core`` 1.5 only
recognizes ``{"type": "reasoning", ...}`` as a standard reasoning block, so a
``thinking`` block is classified ``non_standard``. Two consequences follow, and
together they erased Gemini reasoning from every stream:

* the v3 protocol bridge emits **no** ``content-block-delta`` for a
  non-standard block, so there is nothing to translate into a
  ``reasoning_delta``; and
* it merges non-standard blocks per index by overwriting, so the terminal
  ``content-block-finish`` carries only the *last* chunk's text.

Rewriting the block at the model boundary — rather than special-casing
``non_standard`` in every downstream projector — lets the whole canonical
pipeline (protocol bridge → ``V3ProtocolTranslator`` → SSE/AI SDK adapters)
work as designed, and keeps incremental deltas instead of one truncated blob.

The rewrite is round-trip safe: ``langchain-google-genai`` already accepts a
``reasoning`` block on the request side and reads its thought signature from
``extras.signature``, which is where this module puts it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

_THINKING_TEXT_KEYS = ("thinking", "text")


def _normalize_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one provider ``thinking`` block to a standard reasoning block.

    Returns ``None`` for a signature-only or empty thought block so it does not
    create an empty reasoning delta.
    """
    text = ""
    for key in _THINKING_TEXT_KEYS:
        value = block.get(key)
        if isinstance(value, str) and value:
            text = value
            break

    if not text:
        return None

    normalized: dict[str, Any] = {"type": "reasoning", "reasoning": text}

    signature = block.get("signature")
    if isinstance(signature, str) and signature:
        # ``_convert_to_parts`` restores the thought signature from here when
        # the message is sent back to Gemini on a later turn.
        normalized["extras"] = {"signature": signature}

    return normalized


def normalize_gemini_reasoning_blocks(content: Any) -> Any:
    """Rewrite ``thinking`` blocks in message content as ``reasoning`` blocks.

    Plain-string content and every non-thinking block are returned unchanged.
    """
    if not isinstance(content, list):
        return content

    if not any(isinstance(block, dict) and block.get("type") == "thinking" for block in content):
        return content

    normalized: list[Any] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "thinking":
            normalized.append(block)
            continue
        converted = _normalize_block(block)
        if converted is not None:
            normalized.append(converted)
    return normalized


def _normalize_message(message: BaseMessage) -> BaseMessage:
    normalized_content = normalize_gemini_reasoning_blocks(message.content)
    if normalized_content is message.content:
        return message
    message.content = normalized_content
    return message


def _normalize_chunk(chunk: ChatGenerationChunk) -> ChatGenerationChunk:
    if chunk.message is not None:
        _normalize_message(chunk.message)
    return chunk


def _normalize_result(result: ChatResult) -> ChatResult:
    for generation in result.generations:
        if generation.message is not None:
            _normalize_message(generation.message)
    return result


class ReasoningNormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """``ChatGoogleGenerativeAI`` that speaks standard reasoning blocks.

    Applied to both streaming and non-streaming paths so persisted content and
    live deltas agree, and so ``AIMessage.content_blocks`` reports ``reasoning``
    instead of ``non_standard``.
    """

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        for chunk in super()._stream(*args, **kwargs):
            yield _normalize_chunk(chunk)

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in super()._astream(*args, **kwargs):
            yield _normalize_chunk(chunk)

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return _normalize_result(super()._generate(*args, **kwargs))

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return _normalize_result(await super()._agenerate(*args, **kwargs))
