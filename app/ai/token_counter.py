"""Provider-aware token accounting for complete model requests."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import tiktoken
from langchain_core.messages import ToolMessage

TokenSource = Literal["local", "provider", "reported"]
NativeCounter = Callable[..., int | Awaitable[int]]

_IMAGE_FALLBACK_TOKENS = 1_200
_MESSAGE_ENVELOPE_TOKENS = 4
_REQUEST_ENVELOPE_TOKENS = 2
_TOOL_ENVELOPE_TOKENS = 20
_TOOL_CALL_ENVELOPE_TOKENS = 10
_TOOL_RESULT_ENVELOPE_TOKENS = 8


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    strategy: str
    source: TokenSource = "local"


@dataclass(frozen=True)
class RequestTokenCount:
    input_tokens: int
    message_tokens: int
    tool_tokens: int
    attachment_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    total_tokens: int
    strategy: str
    source: TokenSource
    content_class: Literal["text", "tools", "multimodal", "mixed"]


@dataclass(frozen=True)
class ReportedTokenUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    source: Literal["reported"] = "reported"


class TokenCounter:
    """Count request components with provider/model-specific strategies."""

    def __init__(self, native_counters: Mapping[str, NativeCounter] | None = None):
        self._native_counters = {
            self._normalize_provider(provider): callback
            for provider, callback in (native_counters or {}).items()
        }

    @staticmethod
    def canonical_json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def count_text(self, *, provider: str, model: str, text: str) -> TokenCount:
        provider_key = self._normalize_provider(provider)
        raw_text = str(text or "")
        strategy, encoder = self._text_strategy(provider_key, str(model or ""))
        if not raw_text:
            return TokenCount(tokens=0, strategy=strategy)
        if encoder is not None:
            tokens = len(encoder.encode(raw_text))
        elif provider_key in {"gemini", "anthropic"}:
            tokens = math.ceil(len(raw_text.encode("utf-8")) / 3)
        else:
            tokens = len(raw_text.encode("utf-8"))
        return TokenCount(tokens=max(1, tokens), strategy=strategy)

    def count_messages(
        self,
        *,
        provider: str,
        model: str,
        messages: Sequence[Any] | None,
    ) -> TokenCount:
        total = _REQUEST_ENVELOPE_TOKENS if messages else 0
        strategy = self.count_text(provider=provider, model=model, text="").strategy
        for message in messages or ():
            total += _MESSAGE_ENVELOPE_TOKENS
            content = self._message_value(message, "content")
            total += self.count_text(
                provider=provider,
                model=model,
                text=self._stringify_content(content),
            ).tokens

            tool_calls = self._message_value(message, "tool_calls")
            if tool_calls:
                total += self.count_text(
                    provider=provider,
                    model=model,
                    text=self.canonical_json(tool_calls),
                ).tokens
                total += _TOOL_CALL_ENVELOPE_TOKENS * len(tool_calls)

            if isinstance(message, ToolMessage) or self._message_role(message) == "tool":
                identity = {
                    "name": self._message_value(message, "name") or "",
                    "tool_call_id": self._message_value(message, "tool_call_id") or "",
                }
                total += self.count_text(
                    provider=provider,
                    model=model,
                    text=self.canonical_json(identity),
                ).tokens
                total += _TOOL_RESULT_ENVELOPE_TOKENS

        return TokenCount(tokens=total, strategy=f"{strategy}:messages")

    def count_tools(
        self,
        *,
        provider: str,
        model: str,
        tools: Sequence[Any] | None,
    ) -> TokenCount:
        strategy = self.count_text(provider=provider, model=model, text="").strategy
        total = 0
        for tool in tools or ():
            payload = self._tool_payload(tool)
            total += self.count_text(
                provider=provider,
                model=model,
                text=self.canonical_json(payload),
            ).tokens
            total += _TOOL_ENVELOPE_TOKENS
        return TokenCount(tokens=total, strategy=f"{strategy}:tools")

    def count_attachments(
        self,
        *,
        provider: str,
        model: str,
        attachments: Sequence[Any] | None,
    ) -> TokenCount:
        provider_key = self._normalize_provider(provider)
        strategy = self.count_text(provider=provider, model=model, text="").strategy
        total = 0
        for raw_attachment in attachments or ():
            attachment = self._attachment_payload(raw_attachment)
            mime_type = str(
                attachment.get("mime_type")
                or attachment.get("content_type")
                or attachment.get("type")
                or ""
            ).lower()
            is_image = mime_type.startswith("image/") or mime_type == "image"
            if is_image:
                total += self._image_tokens(provider_key, attachment)
                continue
            safe_descriptor = {
                key: value
                for key, value in attachment.items()
                if key not in {"data", "content", "base64", "bytes"}
            }
            total += self.count_text(
                provider=provider,
                model=model,
                text=self.canonical_json(safe_descriptor),
            ).tokens
        return TokenCount(tokens=total, strategy=f"{strategy}:attachments")

    def estimate_request(
        self,
        *,
        provider: str,
        model: str,
        messages: Sequence[Any] | None,
        tools: Sequence[Any] | None = None,
        attachments: Sequence[Any] | None = None,
        reserved_output_tokens: int = 0,
        safety_margin_tokens: int = 0,
    ) -> RequestTokenCount:
        reserved = self._non_negative(reserved_output_tokens, "reserved_output_tokens")
        safety = self._non_negative(safety_margin_tokens, "safety_margin_tokens")
        message_count = self.count_messages(
            provider=provider,
            model=model,
            messages=messages,
        )
        tool_count = self.count_tools(provider=provider, model=model, tools=tools)
        attachment_count = self.count_attachments(
            provider=provider,
            model=model,
            attachments=attachments,
        )
        input_tokens = message_count.tokens + tool_count.tokens + attachment_count.tokens
        strategy = self.count_text(provider=provider, model=model, text="").strategy
        return RequestTokenCount(
            input_tokens=input_tokens,
            message_tokens=message_count.tokens,
            tool_tokens=tool_count.tokens,
            attachment_tokens=attachment_count.tokens,
            reserved_output_tokens=reserved,
            safety_margin_tokens=safety,
            total_tokens=input_tokens + reserved + safety,
            strategy=strategy,
            source="local",
            content_class=self._content_class(
                has_messages=bool(messages),
                has_tools=bool(tools),
                has_attachments=bool(attachments),
            ),
        )

    async def count_request(
        self,
        *,
        provider: str,
        model: str,
        messages: Sequence[Any] | None,
        tools: Sequence[Any] | None = None,
        attachments: Sequence[Any] | None = None,
        reserved_output_tokens: int = 0,
        safety_margin_tokens: int = 0,
        authoritative: bool = False,
    ) -> RequestTokenCount:
        local = self.estimate_request(
            provider=provider,
            model=model,
            messages=messages,
            tools=tools,
            attachments=attachments,
            reserved_output_tokens=reserved_output_tokens,
            safety_margin_tokens=safety_margin_tokens,
        )
        provider_key = self._normalize_provider(provider)
        native_counter = self._native_counters.get(provider_key)
        if not authoritative or native_counter is None:
            return local

        native_result = native_counter(
            model=model,
            messages=messages or (),
            tools=tools or (),
            attachments=attachments or (),
        )
        if inspect.isawaitable(native_result):
            native_result = await native_result
        native_tokens = self._non_negative(native_result, "native token count")
        return replace(
            local,
            input_tokens=native_tokens,
            total_tokens=(
                native_tokens + local.reserved_output_tokens + local.safety_margin_tokens
            ),
            strategy=f"{provider_key}:native_count",
            source="provider",
        )

    def extract_reported_usage(
        self,
        *,
        provider: str,
        response: Any,
    ) -> ReportedTokenUsage | None:
        del provider  # Key aliases below cover standardized and provider-native shapes.
        envelopes = [
            self._raw_value(response, "usage_metadata"),
            self._raw_value(response, "usage"),
        ]
        response_metadata = self._raw_value(response, "response_metadata")
        if response_metadata is not None:
            envelopes.extend(
                [
                    self._raw_value(response_metadata, "usage"),
                    self._raw_value(response_metadata, "token_usage"),
                    response_metadata,
                ]
            )

        for envelope in envelopes:
            if envelope is None:
                continue
            input_tokens = self._first_usage_int(
                envelope,
                ("input_tokens", "prompt_tokens", "prompt_token_count", "input_token_count"),
            )
            output_tokens = self._first_usage_int(
                envelope,
                (
                    "output_tokens",
                    "completion_tokens",
                    "candidates_token_count",
                    "output_token_count",
                ),
            )
            total_tokens = self._first_usage_int(
                envelope,
                ("total_tokens", "total_token_count"),
            )
            reasoning_tokens = self._first_usage_int(
                envelope,
                ("reasoning_tokens", "thoughts_token_count"),
            )
            if input_tokens is None and output_tokens is None and total_tokens is None:
                continue
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            return ReportedTokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
            )
        return None

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        return str(provider or "unknown").strip().lower() or "unknown"

    @staticmethod
    def _non_negative(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a non-negative integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a non-negative integer") from exc
        if parsed < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return parsed

    @staticmethod
    def _raw_value(data: Any, key: str) -> Any:
        if isinstance(data, dict):
            return data.get(key)
        return getattr(data, key, None)

    @classmethod
    def _first_usage_int(cls, data: Any, keys: tuple[str, ...]) -> int | None:
        for key in keys:
            value = cls._raw_value(data, key)
            if value is None or isinstance(value, bool):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                return parsed
        return None

    @staticmethod
    def _message_value(message: Any, key: str) -> Any:
        if isinstance(message, dict):
            return message.get(key)
        return getattr(message, key, None)

    @classmethod
    def _message_role(cls, message: Any) -> str:
        return str(
            cls._message_value(message, "role") or cls._message_value(message, "type") or "message"
        ).lower()

    @classmethod
    def _stringify_content(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return cls.canonical_json(content)

    @classmethod
    def _tool_payload(cls, tool: Any) -> dict[str, Any]:
        if isinstance(tool, dict):
            return tool
        args_schema = getattr(tool, "args_schema", None)
        if hasattr(args_schema, "model_json_schema"):
            args_schema = args_schema.model_json_schema()
        elif args_schema is None and hasattr(tool, "get_input_schema"):
            input_schema = tool.get_input_schema()
            if hasattr(input_schema, "model_json_schema"):
                args_schema = input_schema.model_json_schema()
        return {
            "name": getattr(tool, "name", "") or "",
            "description": getattr(tool, "description", "") or "",
            "parameters": args_schema or {},
        }

    @staticmethod
    def _attachment_payload(attachment: Any) -> dict[str, Any]:
        if isinstance(attachment, dict):
            return attachment
        if hasattr(attachment, "model_dump"):
            return attachment.model_dump(exclude_none=True)
        if hasattr(attachment, "__dict__"):
            return dict(vars(attachment))
        return {"descriptor": str(attachment)}

    @staticmethod
    def _image_tokens(provider: str, attachment: dict[str, Any]) -> int:
        detail = str(attachment.get("detail") or "high").lower()
        width = attachment.get("width")
        height = attachment.get("height")
        try:
            width_value = int(width) if width is not None else 0
            height_value = int(height) if height is not None else 0
        except (TypeError, ValueError):
            width_value = height_value = 0

        if provider == "openai" and detail == "low":
            return 85
        if width_value <= 0 or height_value <= 0:
            return _IMAGE_FALLBACK_TOKENS
        area = width_value * height_value
        if provider == "openai":
            tiles = math.ceil(width_value / 512) * math.ceil(height_value / 512)
            return 85 + 170 * tiles
        if provider == "anthropic":
            return min(1_600, max(1, math.ceil(area / 750)))
        if provider == "gemini":
            return min(2_048, max(258, math.ceil(area / 768)))
        return max(_IMAGE_FALLBACK_TOKENS, math.ceil(area / 512))

    @staticmethod
    def _content_class(
        *,
        has_messages: bool,
        has_tools: bool,
        has_attachments: bool,
    ) -> Literal["text", "tools", "multimodal", "mixed"]:
        active = sum((has_messages, has_tools, has_attachments))
        if active > 1:
            return "mixed"
        if has_attachments:
            return "multimodal"
        if has_tools:
            return "tools"
        return "text"

    @staticmethod
    def _text_strategy(provider: str, model: str) -> tuple[str, Any | None]:
        if provider == "openai":
            try:
                encoding = tiktoken.encoding_for_model(model)
                return f"openai:tiktoken:{encoding.name}", encoding
            except KeyError:
                encoding_name = (
                    "o200k_base"
                    if any(family in model.lower() for family in ("gpt-4o", "gpt-4.1", "o1", "o3"))
                    else "cl100k_base"
                )
                return (
                    f"openai:tiktoken:{encoding_name}:fallback",
                    tiktoken.get_encoding(encoding_name),
                )
        if provider in {"gemini", "anthropic"}:
            return f"{provider}:utf8_bytes_div_3", None
        return f"{provider}:utf8_byte_upper_bound", None


__all__ = [
    "ReportedTokenUsage",
    "RequestTokenCount",
    "TokenCount",
    "TokenCounter",
]
