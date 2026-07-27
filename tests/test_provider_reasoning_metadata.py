from __future__ import annotations

from unittest.mock import MagicMock

from app.api.model_config import ProviderModelOption as ConfigModelOption
from app.api.providers import ProviderModelOption
from app.services.provider_service import ProviderService


def _service() -> ProviderService:
    return ProviderService(provider_repository=MagicMock())


def test_catalog_entries_share_reasoning_descriptor() -> None:
    entry = _service()._normalize_openai_model("gpt-5.6-sol")
    assert entry["reasoning_control"]["levels"][-1] == "max"
    assert ConfigModelOption.model_validate(entry).reasoning_control.levels[-1] == "max"
    assert ProviderModelOption.model_validate(entry).reasoning_control.levels[-1] == "max"


def test_gemini_api_thinking_boolean_is_preserved_for_unknown_model() -> None:
    entry = _service()._normalize_gemini_model("gemini-future", "Future", ["generateContent"], True)
    assert entry["supports_reasoning"] is True
    assert entry["reasoning_control"]["supported"] is True
    assert entry["reasoning_control"]["levels"] == []


def test_gemini_latest_alias_descriptor_survives_api_models() -> None:
    entry = _service()._normalize_gemini_model(
        "gemini-pro-latest", "Gemini Pro Latest", ["generateContent"], True
    )
    assert entry["reasoning_control"]["levels"] == ["low", "medium", "high"]
    assert ConfigModelOption.model_validate(entry).reasoning_control.levels == [
        "low",
        "medium",
        "high",
    ]
    assert ProviderModelOption.model_validate(entry).reasoning_control.levels == [
        "low",
        "medium",
        "high",
    ]


def test_cached_legacy_catalog_is_enriched_without_provider_resync() -> None:
    catalog = _service()._normalize_catalog_metadata(
        {
            "catalog": {
                "models": [
                    {
                        "id": "gpt-5.6-sol",
                        "provider_type": "openai",
                        "supports_reasoning": True,
                    }
                ]
            }
        }
    )
    assert catalog["models"][0]["reasoning_control"]["levels"][-1] == "max"
