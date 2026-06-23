"""Tests for the widget action resolution endpoint."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

qdrant_client_stub = types.ModuleType("qdrant_client")
qdrant_client_stub.QdrantClient = object
qdrant_models_stub = types.ModuleType("qdrant_client.models")
qdrant_models_stub.FieldCondition = object
qdrant_models_stub.Filter = object
qdrant_models_stub.FilterSelector = object
qdrant_models_stub.MatchValue = object
qdrant_client_stub.models = qdrant_models_stub
sys.modules.setdefault("qdrant_client", qdrant_client_stub)
sys.modules.setdefault("qdrant_client.models", qdrant_models_stub)

sentence_transformers_stub = types.ModuleType("sentence_transformers")
sentence_transformers_stub.CrossEncoder = object
sentence_transformers_stub.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers_stub)

langchain_openai_stub = types.ModuleType("langchain_openai")
langchain_openai_stub.ChatOpenAI = object
sys.modules.setdefault("langchain_openai", langchain_openai_stub)

container_stub = types.ModuleType("app.core.container")
container_stub.get_container = lambda: None
sys.modules.setdefault("app.core.container", container_stub)

widgets_module_spec = importlib.util.spec_from_file_location(
    "widgets_api_actions_under_test",
    Path(__file__).resolve().parents[1] / "app" / "api" / "widgets.py",
)
assert widgets_module_spec is not None and widgets_module_spec.loader is not None
widgets_api = importlib.util.module_from_spec(widgets_module_spec)
widgets_module_spec.loader.exec_module(widgets_api)
router = widgets_api.router

from app.core.auth import get_current_user_id  # noqa: E402
from app.services.widget_runtime import (  # noqa: E402
    InMemoryWidgetStore,
    WidgetConnectionManager,
    WidgetTokenService,
)

TEST_USER_ID = UUID("11111111-1111-1111-1111-111111111111")
TEST_SESSION_ID = "22222222-2222-2222-2222-222222222222"


@pytest.fixture()
def widget_test_client(monkeypatch):
    import app.services.widget_runtime as widget_runtime

    store = InMemoryWidgetStore()
    token_service = WidgetTokenService(secret="test-secret-key-long-enough-32bytes!")
    manager = WidgetConnectionManager()

    monkeypatch.setattr(widget_runtime, "_widget_store", store)
    monkeypatch.setattr(widget_runtime, "_widget_token_service", token_service)
    monkeypatch.setattr(widget_runtime, "_widget_connection_manager", manager)
    monkeypatch.setattr(widgets_api, "_user_can_access_widget_session", lambda *_args: True)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user_id] = lambda: TEST_USER_ID

    with TestClient(app) as client:
        yield client, store


def _seed_widget(store, *, actions=None, state_overrides=None):
    base_state = {
        "html": "<!doctype html><div>Oscillator</div>",
        "height": 540,
        "caption": "demo",
        "actions": actions
        if actions is not None
        else [
            {
                "key": "explain_current_state",
                "label": "Explain",
                "type": "assistant_message",
                "message_template": (
                    "Explain caption={{state.caption}} note={{input_values.note}}"
                ),
            }
        ],
    }
    if state_overrides:
        base_state.update(state_overrides)
    return asyncio.run(
        store.create(
            TEST_SESSION_ID,
            "html",
            base_state,
            title="Oscillator",
        )
    )


def test_widget_action_returns_rendered_message(widget_test_client):
    client, store = widget_test_client
    record = _seed_widget(store)

    response = client.post(
        f"/widgets/{record.widget_id}/actions/explain_current_state",
        json={"input_values": {"note": "demo"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["widget_id"] == record.widget_id
    assert payload["session_id"] == TEST_SESSION_ID
    assert payload["action_key"] == "explain_current_state"
    assert payload["content"] == "Explain caption=demo note=demo"


def test_widget_action_applies_state_patch_before_rendering(widget_test_client):
    client, store = widget_test_client
    record = _seed_widget(store)

    response = client.post(
        f"/widgets/{record.widget_id}/actions/explain_current_state",
        json={
            "input_values": {"note": "ramp"},
            "state_patch": {"caption": "ramped"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["content"] == "Explain caption=ramped note=ramp"

    refreshed = asyncio.run(store.get(record.widget_id))
    assert refreshed is not None
    assert refreshed.state["caption"] == "ramped"
    assert refreshed.state["last_action"]["action_key"] == "explain_current_state"
    assert refreshed.state["last_action"]["content"] == "Explain caption=ramped note=ramp"


def test_widget_action_rejects_contract_breaking_state_patch(widget_test_client):
    """A state_patch that pushes the merged HTML state out of contract (e.g. a
    height outside the iframe range) must be rejected, not silently stored."""
    client, store = widget_test_client
    record = _seed_widget(store)

    response = client.post(
        f"/widgets/{record.widget_id}/actions/explain_current_state",
        json={"state_patch": {"height": 50}},
    )

    assert response.status_code == 400
    assert "height" in response.json()["error"]


def test_widget_action_missing_widget_returns_404(widget_test_client):
    client, _store = widget_test_client

    response = client.post(
        "/widgets/missing-widget/actions/foo",
        json={},
    )

    assert response.status_code == 404


def test_widget_action_unknown_action_returns_404(widget_test_client):
    client, store = widget_test_client
    record = _seed_widget(store)

    response = client.post(
        f"/widgets/{record.widget_id}/actions/unknown_action",
        json={},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["error"]


def test_widget_action_non_assistant_type_returns_400(widget_test_client):
    client, store = widget_test_client
    record = _seed_widget(
        store,
        actions=[
            {
                "key": "ping",
                "type": "noop",
                "message_template": "irrelevant",
            }
        ],
    )

    response = client.post(
        f"/widgets/{record.widget_id}/actions/ping",
        json={},
    )

    assert response.status_code == 400


def test_widget_action_returns_403_when_access_denied(widget_test_client, monkeypatch):
    client, store = widget_test_client
    record = _seed_widget(store)
    monkeypatch.setattr(widgets_api, "_user_can_access_widget_session", lambda *_args: False)

    response = client.post(
        f"/widgets/{record.widget_id}/actions/explain_current_state",
        json={},
    )

    assert response.status_code == 403
    assert response.json()["error"] == "Access denied to this widget"


def test_widget_action_empty_body_allowed(widget_test_client):
    client, store = widget_test_client
    record = _seed_widget(store)

    response = client.post(
        f"/widgets/{record.widget_id}/actions/explain_current_state",
    )

    assert response.status_code == 200
    assert response.json()["content"] == "Explain caption=demo note="
