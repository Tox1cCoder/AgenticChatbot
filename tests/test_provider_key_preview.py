"""``key_preview`` shows the end of the API key, never of its ciphertext.

The preview lets a user recognise which key is stored. It used to be the last
four characters of the Fernet token, which identify nothing, and it changed on
every re-encryption of the same key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.services.provider_service as provider_service_module
from app.core.auth import get_current_user
from app.services.provider_service import ProviderService

API_KEY = "sk-test-0123456789abcdefWXYZ"


@pytest.fixture
def service(monkeypatch) -> ProviderService:
    monkeypatch.setattr(
        provider_service_module.settings, "model_encryption_key", Fernet.generate_key().decode()
    )
    return ProviderService(provider_repository=SimpleNamespace())


def _stored(api_key_encrypted: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        provider_type="openai",
        is_default=True,
        created_at=datetime.now(UTC),
        provider_metadata={},
        api_key_encrypted=api_key_encrypted,
    )


def _listed_preview(service: ProviderService, api_key_encrypted: str) -> str:
    stored = [_stored(api_key_encrypted)]
    service.repository = SimpleNamespace(get_all_by_user=lambda _user_id: stored)
    [listed] = service.get_all_providers(uuid4(), include_encrypted=True)
    return listed["key_preview"]


def test_the_listed_preview_is_the_end_of_the_key(service):
    encrypted = service._encrypt_key(API_KEY)

    preview = _listed_preview(service, encrypted)

    assert preview == "...WXYZ"
    assert encrypted[-4:] not in preview


def test_a_short_key_is_not_mostly_revealed(service):
    assert _listed_preview(service, service._encrypt_key("abcd1234")) == "***"


def test_an_undecryptable_key_lists_a_mask_instead_of_failing(service):
    other_key = Fernet(Fernet.generate_key()).encrypt(API_KEY.encode()).decode()

    assert _listed_preview(service, other_key) == "***"


def test_adding_a_provider_previews_the_submitted_key(service):
    from dependency_injector import providers

    from app.api.providers import router
    from app.core.container import Container

    stored = _stored(service._encrypt_key(API_KEY))
    fake = SimpleNamespace(add_provider=lambda **_kwargs: stored)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid4())

    with Container.provider_service.override(providers.Object(fake)):
        response = TestClient(app).post(
            "/providers", json={"providerType": "openai", "apiKey": API_KEY}
        )

    assert response.status_code == 201
    assert response.json()["data"]["keyPreview"] == "...WXYZ"
