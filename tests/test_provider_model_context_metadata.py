"""Tests for context-window metadata enrichment in the provider catalog sync.

Covers ``ProviderService._normalize_openai_model``,
``ProviderService._normalize_gemini_model``, and the wiring in
``ProviderService._fetch_gemini_models_sync`` that threads provider-API
``input_token_limit`` / ``output_token_limit`` into normalized catalog entries.

Also confirms ``_apply_recommended_model_flags`` doesn't strip the new
context-window fields when it sorts and tags the recommended entry.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.ai.model_context import resolve_model_context_window
from app.services.provider_service import ProviderService


@pytest.fixture
def service() -> ProviderService:
    """Build a ProviderService with a mocked repository.

    ``settings.model_encryption_key`` is loaded from the workspace ``.env``
    so the constructor's Fernet check succeeds. No DB access happens here.
    """
    return ProviderService(provider_repository=MagicMock())


# ---------------------------------------------------------------------------
# Expected key surface
# ---------------------------------------------------------------------------

_CONTEXT_FIELD_KEYS = {
    "context_window_tokens",
    "max_input_tokens",
    "max_output_tokens",
    "limit_type",
    "context_window_source",
    "context_window_known",
}


# ---------------------------------------------------------------------------
# OpenAI normalization
# ---------------------------------------------------------------------------


def test_normalize_openai_known_model_uses_registry(service: ProviderService) -> None:
    entry = service._normalize_openai_model("gpt-4o")

    assert entry["id"] == "gpt-4o"
    assert entry["provider_type"] == "openai"
    assert entry["context_window_known"] is True
    assert entry["context_window_source"] == "registry"
    assert entry["context_window_tokens"] == 128000
    assert entry["max_input_tokens"] == 128000
    assert entry["max_output_tokens"] == 16384


def test_normalize_openai_family_prefix_match(service: ProviderService) -> None:
    entry = service._normalize_openai_model("gpt-4o-mini-2024-07-18")
    assert entry["context_window_known"] is True
    assert entry["context_window_source"] == "registry"
    assert entry["context_window_tokens"] == 128000


def test_normalize_openai_unknown_model_returns_unknown_window(
    service: ProviderService,
) -> None:
    entry = service._normalize_openai_model("UnknownModelXYZ")

    assert entry["context_window_known"] is False
    assert entry["context_window_source"] == "unknown"
    assert entry["context_window_tokens"] is None
    assert entry["max_input_tokens"] is None
    assert entry["max_output_tokens"] is None


def test_normalize_openai_always_emits_new_keys(service: ProviderService) -> None:
    """Both known and unknown models must emit every new field."""
    for model_id in ("gpt-4o", "totally-unknown-xyz-9999"):
        entry = service._normalize_openai_model(model_id)
        missing = _CONTEXT_FIELD_KEYS - set(entry)
        assert not missing, f"{model_id} missing fields: {missing}"


def test_normalize_openai_preserves_existing_fields(service: ProviderService) -> None:
    entry = service._normalize_openai_model("gpt-4o")
    # Sanity-check we didn't drop any of the pre-existing keys.
    for key in (
        "id",
        "display_name",
        "provider_type",
        "supports_vision",
        "supports_tool_calling",
        "supports_streaming",
        "supports_reasoning",
        "recommended",
    ):
        assert key in entry


# ---------------------------------------------------------------------------
# Gemini normalization
# ---------------------------------------------------------------------------


def test_normalize_gemini_uses_provider_api_when_token_limits_present(
    service: ProviderService,
) -> None:
    entry = service._normalize_gemini_model(
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        supported_actions=["generateContent"],
        thinking_metadata=None,
        input_token_limit=1_000_000,
        output_token_limit=8192,
    )

    assert entry["context_window_known"] is True
    assert entry["context_window_source"] == "provider_api"
    assert entry["context_window_tokens"] == 1_000_000
    assert entry["max_input_tokens"] == 1_000_000
    assert entry["max_output_tokens"] == 8192


def test_normalize_gemini_falls_back_to_registry_when_no_token_limits(
    service: ProviderService,
) -> None:
    entry = service._normalize_gemini_model(
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        supported_actions=["generateContent"],
        thinking_metadata=None,
    )

    assert entry["context_window_known"] is True
    assert entry["context_window_source"] == "registry"
    assert entry["context_window_tokens"] == 1_048_576
    assert entry["max_output_tokens"] == 65536


def test_normalize_gemini_unknown_model_without_limits_is_unknown(
    service: ProviderService,
) -> None:
    entry = service._normalize_gemini_model(
        model_id="gemini-mystery-future",
        display_name=None,
        supported_actions=["generateContent"],
        thinking_metadata=None,
    )

    assert entry["context_window_known"] is False
    assert entry["context_window_source"] == "unknown"
    assert entry["context_window_tokens"] is None
    assert entry["max_input_tokens"] is None
    assert entry["max_output_tokens"] is None


def test_normalize_gemini_unknown_model_with_provider_api_limits(
    service: ProviderService,
) -> None:
    """Provider-API limits should be honored even when the registry has no row."""
    entry = service._normalize_gemini_model(
        model_id="gemini-future-experimental",
        display_name=None,
        supported_actions=["generateContent"],
        thinking_metadata=None,
        input_token_limit=2_000_000,
        output_token_limit=32_768,
    )

    assert entry["context_window_known"] is True
    assert entry["context_window_source"] == "provider_api"
    assert entry["context_window_tokens"] == 2_000_000
    assert entry["max_output_tokens"] == 32_768


def test_normalize_gemini_only_input_token_limit(service: ProviderService) -> None:
    entry = service._normalize_gemini_model(
        model_id="gemini-future-only-input",
        display_name=None,
        supported_actions=["generateContent"],
        thinking_metadata=None,
        input_token_limit=512_000,
    )
    assert entry["context_window_source"] == "provider_api"
    assert entry["context_window_tokens"] == 512_000
    assert entry["max_input_tokens"] == 512_000
    assert entry["max_output_tokens"] is None


def test_normalize_gemini_always_emits_new_keys(service: ProviderService) -> None:
    entry = service._normalize_gemini_model(
        model_id="gemini-totally-unknown",
        display_name=None,
        supported_actions=["generateContent"],
        thinking_metadata=None,
    )
    missing = _CONTEXT_FIELD_KEYS - set(entry)
    assert not missing


# ---------------------------------------------------------------------------
# _fetch_gemini_models_sync wiring
# ---------------------------------------------------------------------------


class _FakeGeminiModel:
    def __init__(
        self,
        name: str,
        display_name: str | None,
        supported_actions: list[str],
        *,
        input_token_limit: int | None = None,
        output_token_limit: int | None = None,
        thinking: Any | None = None,
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.supported_actions = supported_actions
        self.input_token_limit = input_token_limit
        self.output_token_limit = output_token_limit
        self.thinking = thinking


class _FakeGeminiModelsAPI:
    def __init__(self, models: list[_FakeGeminiModel]) -> None:
        self._models = models

    def list(self, config: Any) -> Any:  # noqa: ARG002 - mirrors real signature
        return iter(self._models)


class _FakeGeminiClient:
    def __init__(self, models: list[_FakeGeminiModel]) -> None:
        self.models = _FakeGeminiModelsAPI(models)


def _install_fake_genai(
    monkeypatch: pytest.MonkeyPatch,
    models: list[_FakeGeminiModel],
) -> dict[str, Any]:
    """Install a fake ``google.genai`` module that returns ``models``.

    Returns a dict captured with the api_key argument supplied to
    ``genai.Client(...)``.
    """
    captured: dict[str, Any] = {}

    def _client_factory(*, api_key: str, http_options: Any = None) -> _FakeGeminiClient:
        captured["api_key"] = api_key
        captured["http_options"] = http_options
        return _FakeGeminiClient(models)

    fake_module = ModuleType("google.genai")
    fake_module.Client = _client_factory  # type: ignore[attr-defined]

    # Patch both sys.modules AND the ``genai`` attribute on the ``google``
    # package. ``from google import genai`` first does ``getattr(google,
    # "genai")`` and only falls back to ``sys.modules["google.genai"]`` if the
    # attribute is missing. Since ``google.genai`` was already imported
    # elsewhere in the process, the attribute lookup succeeds and would return
    # the real submodule, bypassing the sys.modules patch. Setting the
    # attribute forces the import to resolve to our fake.
    monkeypatch.setitem(sys.modules, "google.genai", fake_module)
    import google

    monkeypatch.setattr(google, "genai", fake_module, raising=False)
    return captured


def test_fetch_gemini_models_sync_passes_token_limits(
    service: ProviderService, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_models = [
        _FakeGeminiModel(
            name="models/gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            supported_actions=["generateContent"],
            input_token_limit=1_000_000,
            output_token_limit=8192,
        ),
        _FakeGeminiModel(
            name="models/gemini-future-experimental",
            display_name="Gemini Future",
            supported_actions=["generateContent"],
            input_token_limit=2_000_000,
            output_token_limit=16_384,
        ),
        # Filtered out: not a chat-capable gemini model.
        _FakeGeminiModel(
            name="models/text-embedding-005",
            display_name="Embedding",
            supported_actions=["embedContent"],
        ),
    ]
    _install_fake_genai(monkeypatch, fake_models)

    result = service._fetch_gemini_models_sync("fake-api-key")
    by_id = {entry["id"]: entry for entry in result}

    assert "gemini-2.5-flash" in by_id
    flash = by_id["gemini-2.5-flash"]
    assert flash["context_window_source"] == "provider_api"
    assert flash["context_window_tokens"] == 1_000_000
    assert flash["max_output_tokens"] == 8192

    assert "gemini-future-experimental" in by_id
    future = by_id["gemini-future-experimental"]
    assert future["context_window_source"] == "provider_api"
    assert future["context_window_tokens"] == 2_000_000
    assert future["max_output_tokens"] == 16_384

    # Filtered-out embedding entry must not appear.
    assert "text-embedding-005" not in by_id


def test_fetch_gemini_models_sync_falls_back_when_limits_missing(
    service: ProviderService, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_models = [
        _FakeGeminiModel(
            name="models/gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            supported_actions=["generateContent"],
            # No input/output token limits supplied.
        ),
    ]
    _install_fake_genai(monkeypatch, fake_models)

    result = service._fetch_gemini_models_sync("fake-api-key")
    assert len(result) == 1
    entry = result[0]
    assert entry["id"] == "gemini-2.5-flash"
    assert entry["context_window_source"] == "registry"
    assert entry["context_window_known"] is True
    assert entry["context_window_tokens"] == 1_048_576


# ---------------------------------------------------------------------------
# _apply_recommended_model_flags must keep new fields intact
# ---------------------------------------------------------------------------


def test_apply_recommended_flags_preserves_context_window_fields(
    service: ProviderService,
) -> None:
    raw = [
        service._normalize_openai_model("gpt-4o"),
        service._normalize_openai_model("gpt-5-mini"),
        service._normalize_openai_model("UnknownXYZ"),
    ]

    flagged = service._apply_recommended_model_flags("openai", raw)

    by_id = {entry["id"]: entry for entry in flagged}
    # Every entry retains every new key.
    for model_id, entry in by_id.items():
        missing = _CONTEXT_FIELD_KEYS - set(entry)
        assert not missing, f"{model_id} lost fields: {missing}"

    # The known entries still carry their resolved values.
    assert by_id["gpt-4o"]["context_window_tokens"] == 128000
    assert by_id["gpt-5-mini"]["context_window_tokens"] == 400000

    # Unknown stays unknown.
    assert by_id["UnknownXYZ"]["context_window_known"] is False
    assert by_id["UnknownXYZ"]["context_window_source"] == "unknown"

    # And exactly one is flagged as recommended (per preferred order).
    recommended_ids = [entry["id"] for entry in flagged if entry["recommended"]]
    assert recommended_ids == ["gpt-5-mini"]


def test_apply_recommended_flags_preserves_context_window_fields_gemini(
    service: ProviderService,
) -> None:
    raw = [
        service._normalize_gemini_model(
            model_id="gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            supported_actions=["generateContent"],
            thinking_metadata=None,
            input_token_limit=1_000_000,
            output_token_limit=8192,
        ),
        service._normalize_gemini_model(
            model_id="gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            supported_actions=["generateContent"],
            thinking_metadata=None,
        ),
    ]

    flagged = service._apply_recommended_model_flags("gemini", raw)
    by_id = {entry["id"]: entry for entry in flagged}

    assert by_id["gemini-2.5-flash"]["context_window_source"] == "provider_api"
    assert by_id["gemini-2.5-flash"]["context_window_tokens"] == 1_000_000
    assert by_id["gemini-2.5-pro"]["context_window_source"] == "registry"
    assert by_id["gemini-2.5-pro"]["context_window_known"] is True


def test_provider_catalog_round_trip_preserves_separate_io(
    service: ProviderService,
) -> None:
    original = resolve_model_context_window("gemini", "gemini-3-pro-image")

    fields = service._context_window_fields(original)
    resolved = service._resolve_catalog_context_window(
        "gemini",
        "gemini-3-pro-image",
        fields,
    )

    assert resolved == original


# ---------------------------------------------------------------------------
# API schema: ProviderModelOption exposes the new fields via camelCase aliases
# ---------------------------------------------------------------------------


from app.api.model_config import ProviderModelOption as ModelConfigOption  # noqa: E402
from app.api.providers import ProviderModelOption as ProvidersOption  # noqa: E402

_SCHEMA_CLASSES = [
    pytest.param(ProvidersOption, id="app.api.providers.ProviderModelOption"),
    pytest.param(ModelConfigOption, id="app.api.model_config.ProviderModelOption"),
]

_CAMEL_FIELD_KEYS = {
    "contextWindowTokens",
    "maxInputTokens",
    "maxOutputTokens",
    "contextWindowSource",
    "contextWindowKnown",
}


def _full_catalog_entry() -> dict[str, Any]:
    return {
        "id": "gpt-4o",
        "display_name": "GPT-4o",
        "provider_type": "openai",
        "supports_vision": True,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "supports_reasoning": False,
        "recommended": True,
        "context_window_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "context_window_source": "registry",
        "context_window_known": True,
    }


def _legacy_catalog_entry() -> dict[str, Any]:
    """A catalog entry from before the context-window fields existed."""
    return {
        "id": "legacy-model",
        "display_name": "Legacy Model",
        "provider_type": "openai",
        "supports_vision": False,
        "supports_tool_calling": False,
        "supports_streaming": False,
        "supports_reasoning": False,
        "recommended": False,
    }


@pytest.mark.parametrize("option_cls", _SCHEMA_CLASSES)
def test_provider_model_option_accepts_context_window_fields(
    option_cls: type,
) -> None:
    entry = _full_catalog_entry()
    option = option_cls.model_validate(entry)

    assert option.context_window_tokens == 128000
    assert option.max_input_tokens == 128000
    assert option.max_output_tokens == 16384
    assert option.context_window_source == "registry"
    assert option.context_window_known is True


@pytest.mark.parametrize("option_cls", _SCHEMA_CLASSES)
def test_provider_model_option_serializes_camel_case_aliases(
    option_cls: type,
) -> None:
    entry = _full_catalog_entry()
    option = option_cls.model_validate(entry)
    dumped = option.model_dump(by_alias=True)

    for camel_key in _CAMEL_FIELD_KEYS:
        assert camel_key in dumped, f"missing camelCase alias: {camel_key}"

    assert dumped["contextWindowTokens"] == 128000
    assert dumped["maxInputTokens"] == 128000
    assert dumped["maxOutputTokens"] == 16384
    assert dumped["contextWindowSource"] == "registry"
    assert dumped["contextWindowKnown"] is True


@pytest.mark.parametrize("option_cls", _SCHEMA_CLASSES)
def test_provider_model_option_backwards_compatible_without_new_fields(
    option_cls: type,
) -> None:
    """Old cached catalog entries lacking the new fields must still validate."""
    option = option_cls.model_validate(_legacy_catalog_entry())

    assert option.id == "legacy-model"
    assert option.context_window_tokens is None
    assert option.max_input_tokens is None
    assert option.max_output_tokens is None
    assert option.context_window_source is None
    assert option.context_window_known is None


@pytest.mark.parametrize("option_cls", _SCHEMA_CLASSES)
def test_provider_model_option_defaults_to_none_when_omitted(
    option_cls: type,
) -> None:
    option = option_cls.model_validate(
        {
            "id": "x",
            "display_name": "X",
            "provider_type": "openai",
        }
    )

    assert option.context_window_tokens is None
    assert option.max_input_tokens is None
    assert option.max_output_tokens is None
    assert option.context_window_source is None
    assert option.context_window_known is None

    dumped = option.model_dump(by_alias=True)
    for camel_key in _CAMEL_FIELD_KEYS:
        assert dumped[camel_key] is None


@pytest.mark.parametrize("option_cls", _SCHEMA_CLASSES)
def test_provider_model_option_accepts_camel_case_input(option_cls: type) -> None:
    """populate_by_name=True means camelCase input also validates."""
    option = option_cls.model_validate(
        {
            "id": "gpt-4o",
            "displayName": "GPT-4o",
            "providerType": "openai",
            "contextWindowTokens": 128000,
            "maxInputTokens": 128000,
            "maxOutputTokens": 16384,
            "contextWindowSource": "registry",
            "contextWindowKnown": True,
        }
    )

    assert option.display_name == "GPT-4o"
    assert option.provider_type == "openai"
    assert option.context_window_tokens == 128000
    assert option.context_window_source == "registry"
    assert option.context_window_known is True


# ---------------------------------------------------------------------------
# Read-time backfill for legacy cached catalogs
# ---------------------------------------------------------------------------


def test_backfill_context_window_fills_missing_fields(service: ProviderService) -> None:
    """Legacy entries lacking the new fields are populated from the registry."""
    legacy = {
        "id": "gpt-4o",
        "display_name": "GPT-4o",
        "provider_type": "openai",
        "supports_vision": True,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "supports_reasoning": False,
        "recommended": False,
    }

    result = service._backfill_context_window(legacy)

    assert result["context_window_known"] is True
    assert result["context_window_source"] == "registry"
    assert result["context_window_tokens"] == 128000
    assert result["max_input_tokens"] == 128000
    assert result["max_output_tokens"] == 16384


def test_backfill_context_window_preserves_existing_values(
    service: ProviderService,
) -> None:
    """A fresh entry with explicit provider_api values must not be overwritten."""
    fresh = {
        "id": "gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash",
        "provider_type": "gemini",
        "supports_vision": True,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "supports_reasoning": True,
        "recommended": True,
        "context_window_tokens": 999_999,
        "max_input_tokens": 999_999,
        "max_output_tokens": 4242,
        "context_window_source": "provider_api",
        "context_window_known": True,
    }

    result = service._backfill_context_window(fresh)

    assert result["context_window_tokens"] == 999_999
    assert result["max_input_tokens"] == 999_999
    assert result["max_output_tokens"] == 4242
    assert result["context_window_source"] == "provider_api"
    assert result["context_window_known"] is True


def _legacy_cached_provider(provider_type: str, model_id: str) -> MagicMock:
    """Build a provider row whose cached catalog lacks context-window fields."""
    provider = MagicMock()
    provider.id = "00000000-0000-0000-0000-000000000001"
    provider.provider_type = provider_type
    provider.api_key_encrypted = "encrypted"
    provider.provider_metadata = {
        "catalog": {
            "models": [
                {
                    "id": model_id,
                    "display_name": model_id,
                    "provider_type": provider_type,
                    "supports_vision": True,
                    "supports_tool_calling": True,
                    "supports_streaming": True,
                    "supports_reasoning": False,
                    "recommended": True,
                },
            ],
            "sync_status": "ready",
            "last_synced_at": "2025-01-01T00:00:00+00:00",
            "sync_error": None,
            "catalog_version": 1,
        }
    }
    return provider


def _patch_service_for_legacy_catalog(
    service: ProviderService,
    provider_type: str,
    model_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_provider = _legacy_cached_provider(provider_type, model_id)
    service.repository.get_by_user_and_type = MagicMock(return_value=legacy_provider)
    monkeypatch.setattr(service, "_decrypt_key", lambda _enc: "fake-decrypted-key")


def test_get_cached_provider_models_backfills_context_window(
    service: ProviderService, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uuid

    _patch_service_for_legacy_catalog(service, "openai", "gpt-4o", monkeypatch)

    models = service.get_cached_provider_models(uuid.uuid4(), "openai")

    assert len(models) == 1
    entry = models[0]
    assert entry["id"] == "gpt-4o"
    assert entry["context_window_known"] is True
    assert entry["context_window_source"] == "registry"
    assert entry["context_window_tokens"] == 128000
    assert entry["max_input_tokens"] == 128000
    assert entry["max_output_tokens"] == 16384


def test_get_cached_provider_status_backfills_context_window(
    service: ProviderService, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uuid

    _patch_service_for_legacy_catalog(service, "gemini", "gemini-2.5-flash", monkeypatch)

    status = service.get_cached_provider_status(uuid.uuid4(), "gemini")

    assert status["configured"] is True
    assert status["sync_status"] == "ready"
    assert len(status["models"]) == 1
    entry = status["models"][0]
    assert entry["id"] == "gemini-2.5-flash"
    assert entry["context_window_known"] is True
    assert entry["context_window_source"] == "registry"
    assert entry["context_window_tokens"] == 1_048_576
    assert entry["max_output_tokens"] == 65536
