"""Provider-aware token accounting for complete model requests."""

from __future__ import annotations

import inspect
import json
import math
import secrets
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import tiktoken
from langchain_core.messages import ToolMessage

TokenSource = Literal["local", "provider", "reported"]
NativeCounter = Callable[..., int | Awaitable[int]]
NativeTextCounter = Callable[..., int]

_IMAGE_FALLBACK_TOKENS = 1_200
_MESSAGE_ENVELOPE_TOKENS = 4
_REQUEST_ENVELOPE_TOKENS = 2
_TOOL_ENVELOPE_TOKENS = 20
_TOOL_CALL_ENVELOPE_TOKENS = 10
_TOOL_RESULT_ENVELOPE_TOKENS = 8

# Canonical usage-field aliases and nested-detail paths, shared by
# TokenCounter.extract_reported_usage and app.usage.normalizers.
# normalize_provider_usage. Previously each function hand-maintained its own
# copy and drifted (missing output-image and cache-fallback paths in one but
# not the other); every alias/path below now has exactly one definition, used
# by both call sites, so they cannot drift again. The two functions still
# differ in their per-function *algorithm* — extract_reported_usage
# synthesizes total_tokens = input + output when the provider omits a total
# and normalize_provider_usage deliberately does not — but that difference
# lives in each function's body, not in these shared constants.
_USAGE_INPUT_ALIASES = (
    "input_tokens",
    "prompt_tokens",
    "prompt_token_count",
    "input_token_count",
)
_USAGE_OUTPUT_ALIASES = (
    "output_tokens",
    "completion_tokens",
    "candidates_token_count",
    "output_token_count",
)
_USAGE_TOTAL_ALIASES = ("total_tokens", "total_token_count")
_USAGE_REASONING_ALIASES = ("reasoning_tokens", "thoughts_token_count")
# Exactly extract_reported_usage's pre-Task-4 3 nested paths (singular
# "output_token_details" / "completion_tokens_details") — do not widen this.
# normalize_provider_usage needs a 4th, provider-raw path
# ("output_tokens_details", "reasoning_tokens", plural) that
# extract_reported_usage never recognized; that extra path is a
# normalizer-local extension of this base tuple (see
# app/usage/normalizers.py's _NORMALIZE_REASONING_NESTED_PATHS), not part of
# the shared set, so extract_reported_usage's behavior stays byte-for-byte
# unchanged from before Task 4.
_USAGE_REASONING_NESTED_PATHS = (
    ("output_token_details", "reasoning"),
    ("output_token_details", "reasoning_tokens"),
    ("completion_tokens_details", "reasoning_tokens"),
)
# Cached-input-token sources, tried in order by _extract_cached_input_tokens:
# flat top-level aliases (summed — Anthropic's cache_read/cache_creation are
# additive sub-quantities, not aliases for the same value), LangChain's
# standardized input_token_details.{cache_read,cache_creation} (also summed),
# then first-match provider-raw nested shapes.
_USAGE_CACHED_INPUT_FLAT_KEYS = (
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cached_content_token_count",
)
_USAGE_CACHED_INPUT_DETAIL_KEYS = ("cache_read", "cache_creation")
_USAGE_CACHED_INPUT_NESTED_PATHS = (
    ("input_tokens_details", "cached_tokens"),
    ("prompt_tokens_details", "cached_tokens"),
)
_USAGE_INPUT_TEXT_NESTED_PATHS = (("input_tokens_details", "text_tokens"),)
_USAGE_INPUT_IMAGE_NESTED_PATHS = (("input_tokens_details", "image_tokens"),)
_USAGE_OUTPUT_TEXT_NESTED_PATHS = (("output_tokens_details", "text_tokens"),)
_USAGE_OUTPUT_IMAGE_NESTED_PATHS = (("output_tokens_details", "image_tokens"),)


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
    cost_amount: float | None = None
    cost_currency: str | None = None
    cached_input_tokens: int | None = None
    input_text_tokens: int | None = None
    input_image_tokens: int | None = None
    output_text_tokens: int | None = None
    output_image_tokens: int | None = None
    source: Literal["reported"] = "reported"


class EphemeralTokenCounterStore:
    """Bounded one-shot transport for counters that must never enter graph state.

    Checkpointed state carries only the opaque reference plus provider/model
    identity. A missing/expired reference is expected after process restart and
    callers must fall back to a deterministic local :class:`TokenCounter`.
    """

    def __init__(self, *, max_entries: int = 32, ttl_seconds: float = 300.0):
        self._max_entries = max(1, int(max_entries))
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._entries: OrderedDict[str, tuple[float, TokenCounter]] = OrderedDict()

    def put(self, counter: TokenCounter) -> str:
        self.prune()
        reference = secrets.token_urlsafe(18)
        self._entries[reference] = (time.monotonic() + self._ttl_seconds, counter)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return reference

    def take(self, reference: str | None) -> TokenCounter | None:
        self.prune()
        if not reference:
            return None
        entry = self._entries.pop(str(reference), None)
        return entry[1] if entry is not None else None

    def discard(self, reference: str | None) -> None:
        if reference:
            self._entries.pop(str(reference), None)

    def prune(self) -> None:
        now = time.monotonic()
        expired = [key for key, (deadline, _) in self._entries.items() if deadline <= now]
        for key in expired:
            self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        self.prune()
        return len(self._entries)


class TokenCounter:
    """Count request components with provider/model-specific strategies."""

    def __init__(
        self,
        native_counters: Mapping[str, NativeCounter] | None = None,
        native_text_counters: Mapping[str, NativeTextCounter] | None = None,
    ):
        self._native_counters = {
            self._normalize_provider(provider): callback
            for provider, callback in (native_counters or {}).items()
        }
        self._native_text_counters = {
            self._normalize_provider(provider): callback
            for provider, callback in (native_text_counters or {}).items()
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
        native_counter = self._native_text_counters.get(provider_key)
        if native_counter is not None:
            try:
                tokens = self._non_negative(
                    native_counter(model=str(model or ""), text=raw_text),
                    "native text token count",
                )
                return TokenCount(
                    tokens=tokens,
                    strategy=f"{provider_key}:native_text",
                    source="provider",
                )
            except Exception:
                # Provider tokenization is optional. The provider-specific local
                # strategy below is deliberately conservative and explicit.
                strategy = f"{strategy}:conservative_fallback"
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
        if self._native_text_counters:
            # Request estimation must remain one bounded local operation. Native
            # text callbacks are reserved for exact rendered evidence and the
            # single full-request native call in ``count_request``.
            return TokenCounter().estimate_request(
                provider=provider,
                model=model,
                messages=messages,
                tools=tools,
                attachments=attachments,
                reserved_output_tokens=reserved_output_tokens,
                safety_margin_tokens=safety_margin_tokens,
            )
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

    @classmethod
    def canonical_request_text(
        cls,
        *,
        messages: Sequence[Any],
        tools: Sequence[Any],
        attachments: Sequence[Any],
    ) -> str:
        """Stable full protocol payload used by provider-native tokenizers."""
        message_payloads = []
        for message in messages:
            payload = {
                "role": cls._message_role(message),
                "content": cls._message_value(message, "content"),
            }
            tool_calls = cls._message_value(message, "tool_calls")
            if tool_calls:
                payload["tool_calls"] = tool_calls
            if cls._message_role(message) == "tool" or isinstance(message, ToolMessage):
                payload["name"] = cls._message_value(message, "name") or ""
                payload["tool_call_id"] = (
                    cls._message_value(message, "tool_call_id") or ""
                )
            message_payloads.append(payload)
        return cls.canonical_json(
            {
                "attachments": [cls._attachment_payload(item) for item in attachments],
                "messages": message_payloads,
                "tools": [cls._tool_payload(tool) for tool in tools],
            }
        )

    def extract_reported_usage(
        self,
        *,
        provider: str,
        response: Any,
    ) -> ReportedTokenUsage | None:
        del provider  # Key aliases below cover standardized and provider-native shapes.
        for envelope in self._iter_usage_envelopes(response):
            if envelope is None:
                continue
            input_tokens = self._first_usage_int(envelope, _USAGE_INPUT_ALIASES)
            output_tokens = self._first_usage_int(envelope, _USAGE_OUTPUT_ALIASES)
            total_tokens = self._first_usage_int(envelope, _USAGE_TOTAL_ALIASES)
            reasoning_tokens = self._first_usage_int(envelope, _USAGE_REASONING_ALIASES)
            if reasoning_tokens is None:
                reasoning_tokens = self._first_nested_usage_int(
                    envelope, _USAGE_REASONING_NESTED_PATHS
                )
            if input_tokens is None and output_tokens is None and total_tokens is None:
                continue
            # Deliberate difference from normalize_provider_usage: this
            # function synthesizes a missing total. Keep that here, in the
            # per-function algorithm, not in the shared constants above.
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            cached_input_tokens = self._extract_cached_input_tokens(envelope)
            input_text_tokens = self._first_nested_usage_int(
                envelope, _USAGE_INPUT_TEXT_NESTED_PATHS
            )
            input_image_tokens = self._first_nested_usage_int(
                envelope, _USAGE_INPUT_IMAGE_NESTED_PATHS
            )
            output_text_tokens = self._first_nested_usage_int(
                envelope, _USAGE_OUTPUT_TEXT_NESTED_PATHS
            )
            output_image_tokens = self._first_nested_usage_int(
                envelope, _USAGE_OUTPUT_IMAGE_NESTED_PATHS
            )
            cost_amount = self._first_usage_float(
                envelope,
                ("cost", "cost_amount", "total_cost"),
            )
            raw_currency = self._raw_value(envelope, "currency")
            cost_currency = (
                str(raw_currency).strip().upper()
                if isinstance(raw_currency, str) and raw_currency.strip()
                else None
            )
            return ReportedTokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
                cost_amount=cost_amount,
                cost_currency=cost_currency,
                cached_input_tokens=cached_input_tokens,
                input_text_tokens=input_text_tokens,
                input_image_tokens=input_image_tokens,
                output_text_tokens=output_text_tokens,
                output_image_tokens=output_image_tokens,
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
    def _iter_usage_envelopes(cls, response: Any) -> list[Any]:
        """Return the ordered usage envelopes to search for ``response``.

        Shared by ``extract_reported_usage`` and
        ``app.usage.normalizers.normalize_provider_usage`` so the two entry
        points can never search a different envelope set: standardized
        ``usage_metadata``/``usage`` first, then the ``response_metadata``
        fallbacks some LangChain providers use instead.
        """
        envelopes = [
            cls._raw_value(response, "usage_metadata"),
            cls._raw_value(response, "usage"),
        ]
        response_metadata = cls._raw_value(response, "response_metadata")
        if response_metadata is not None:
            envelopes.extend(
                [
                    cls._raw_value(response_metadata, "usage"),
                    cls._raw_value(response_metadata, "token_usage"),
                    response_metadata,
                ]
            )
        return envelopes

    @classmethod
    def _extract_cached_input_tokens(cls, envelope: Any) -> int | None:
        """Canonical cached-input-token extraction for a single envelope.

        Shared by ``extract_reported_usage`` and
        ``app.usage.normalizers.normalize_provider_usage`` so the two can
        never drift on which cache-token shapes they recognize. Tries, in
        order: flat top-level aliases (summed), LangChain's standardized
        ``input_token_details.{cache_read,cache_creation}`` (also summed),
        then first-match provider-raw nested shapes.
        """
        cached_input_tokens = cls._sum_usage_ints(envelope, _USAGE_CACHED_INPUT_FLAT_KEYS)
        if cached_input_tokens is not None:
            return cached_input_tokens
        cached_input_tokens = cls._sum_usage_ints(
            cls._raw_value(envelope, "input_token_details"),
            _USAGE_CACHED_INPUT_DETAIL_KEYS,
        )
        if cached_input_tokens is not None:
            return cached_input_tokens
        return cls._first_nested_usage_int(envelope, _USAGE_CACHED_INPUT_NESTED_PATHS)

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

    @classmethod
    def _sum_usage_ints(cls, data: Any, keys: tuple[str, ...]) -> int | None:
        """Sum every present non-boolean, non-negative int among ``keys``.

        Unlike ``_first_usage_int`` (first match wins), this combines values
        that represent additive sub-quantities of the same concept — e.g.
        Anthropic's ``cache_read_input_tokens`` and
        ``cache_creation_input_tokens`` both count toward "cached input
        tokens". Returns ``None`` only when none of ``keys`` are present.
        """
        total: int | None = None
        for key in keys:
            value = cls._raw_value(data, key)
            if value is None or isinstance(value, bool):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed < 0:
                continue
            total = (total or 0) + parsed
        return total

    @classmethod
    def _first_usage_float(cls, data: Any, keys: tuple[str, ...]) -> float | None:
        for key in keys:
            value = cls._raw_value(data, key)
            if value is None or isinstance(value, bool):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed) and parsed >= 0:
                return parsed
        return None

    @classmethod
    def _first_nested_usage_int(
        cls,
        data: Any,
        paths: tuple[tuple[str, ...], ...],
    ) -> int | None:
        for path in paths:
            value = data
            for key in path:
                value = cls._raw_value(value, key)
                if value is None:
                    break
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
