from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
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
    "widgets_api_under_test",
    Path(__file__).resolve().parents[1] / "app" / "api" / "widgets.py",
)
assert widgets_module_spec is not None and widgets_module_spec.loader is not None
widgets_api = importlib.util.module_from_spec(widgets_module_spec)
widgets_module_spec.loader.exec_module(widgets_api)
router = widgets_api.router

from app.core.auth import get_current_user_id
from app.services.widget_runtime import (
    InMemoryWidgetStore,
    WidgetConnectionManager,
    WidgetTokenService,
)

TEST_USER_ID = UUID("11111111-1111-1111-1111-111111111111")
TEST_SESSION_ID = "22222222-2222-2222-2222-222222222222"
TEST_SESSION_ID_2 = "33333333-3333-3333-3333-333333333333"


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
        yield client, store, token_service


def test_widget_connection_mints_token_and_ws_url(widget_test_client):
    client, store, _token_service = widget_test_client
    created = asyncio.run(
        store.create(
            TEST_SESSION_ID,
            "table",
            {"columns": ["Name", "Price"], "rows": [["Widget", "$10"]]},
            title="Comparison",
        )
    )

    response = client.post(f"/widgets/{created.widget_id}/connection")

    assert response.status_code == 200
    payload = response.json()
    assert payload["widget_id"] == created.widget_id
    assert payload["session_id"] == created.session_id
    assert payload["widget_type"] == "table"
    assert payload["title"] == "Comparison"
    assert payload["status"] == "active"
    assert payload["version"] == 1
    assert payload["token"]
    assert payload["ws_url"].startswith(
        f"/widgets/{created.widget_id}/connect?session_id={created.session_id}&token="
    )


def test_widget_connection_returns_404_for_missing_widget(widget_test_client):
    client, _store, _token_service = widget_test_client

    response = client.post("/widgets/missing-widget/connection")

    assert response.status_code == 404
    assert response.json()["error"] == "Widget missing-widget not found"


def test_widget_connection_returns_403_when_access_denied(widget_test_client, monkeypatch):
    client, store, _token_service = widget_test_client
    created = asyncio.run(
        store.create(
            TEST_SESSION_ID,
            "table",
            {"columns": ["Name"], "rows": [["Widget"]]},
            title="Restricted",
        )
    )
    monkeypatch.setattr(widgets_api, "_user_can_access_widget_session", lambda *_args: False)

    response = client.post(f"/widgets/{created.widget_id}/connection")

    assert response.status_code == 403
    assert response.json()["error"] == "Access denied to this widget"


def test_widget_connection_recovers_invalid_session_id(widget_test_client, monkeypatch):
    client, store, _token_service = widget_test_client
    created = asyncio.run(
        store.create(
            "current_session",
            "table",
            {"columns": ["Name"], "rows": [["Widget"]]},
            title="Recovered",
        )
    )
    monkeypatch.setattr(
        widgets_api,
        "_user_can_access_widget_session",
        lambda _user_id, session_id: session_id == TEST_SESSION_ID,
    )
    monkeypatch.setattr(
        widgets_api,
        "_recover_widget_session_id_from_messages",
        lambda *_args: TEST_SESSION_ID,
    )

    response = client.post(f"/widgets/{created.widget_id}/connection")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == TEST_SESSION_ID
    assert payload["ws_url"].startswith(
        f"/widgets/{created.widget_id}/connect?session_id={TEST_SESSION_ID}&token="
    )


def test_widget_connection_restores_missing_widget_from_message_metadata(
    widget_test_client, monkeypatch
):
    client, store, _token_service = widget_test_client
    widget_id = "restored-widget-id"
    metadata = {
        "live_widgets": [
            {
                "widget_id": widget_id,
                "session_id": "current_session",
                "widget_type": "chart",
                "title": "Recovered Chart",
                "status": "active",
                "version": 2,
                "connection_endpoint": f"/widgets/{widget_id}/connection",
            }
        ],
        "tool_artifacts": [
            {
                "tool": "widget_create",
                "status": "success",
                "error": None,
                "args": {
                    "session_id": "current_session",
                    "widget_type": "chart",
                    "title": "Recovered Chart",
                    "initial_state": (
                        '{"chart_type":"line","labels":["A"],'
                        '"datasets":[{"label":"Series","data":[1]}]}'
                    ),
                },
                "output": (
                    '{"widget_id":"restored-widget-id","session_id":"current_session",'
                    '"widget_type":"chart","title":"Recovered Chart",'
                    '"status":"active","version":2}'
                ),
            }
        ],
    }
    monkeypatch.setattr(
        widgets_api,
        "_iter_widget_messages_for_user",
        lambda *_args: [
            SimpleNamespace(
                conversation_id=UUID(TEST_SESSION_ID),
                message_metadata=metadata,
            )
        ],
    )

    response = client.post(f"/widgets/{widget_id}/connection")

    assert response.status_code == 200
    payload = response.json()
    assert payload["widget_id"] == widget_id
    assert payload["session_id"] == TEST_SESSION_ID
    restored = asyncio.run(store.get(widget_id))
    assert restored is not None
    assert restored.widget_type == "chart"
    assert restored.version == 2
    assert restored.state["chart_type"] == "line"


def test_widget_websocket_sync_and_user_patch(widget_test_client):
    client, store, token_service = widget_test_client
    created = asyncio.run(
        store.create(
            TEST_SESSION_ID_2,
            "list",
            {"items": [{"id": "alpha", "label": "Alpha"}], "selection": None},
            title="Pick One",
        )
    )
    token, _expires_at = token_service.mint(
        widget_id=created.widget_id,
        session_id=created.session_id,
        user_id=str(TEST_USER_ID),
    )

    with client.websocket_connect(
        f"/widgets/{created.widget_id}/connect?session_id={created.session_id}&token={token}"
    ) as websocket:
        initial = websocket.receive_json()
        assert initial["type"] == "widget_state_sync"
        assert initial["widget_id"] == created.widget_id
        assert initial["version"] == 1
        assert initial["state"]["items"][0]["label"] == "Alpha"

        websocket.send_json({"type": "user_state_patch", "patch": {"selection": "alpha"}})
        updated = websocket.receive_json()
        assert updated["type"] == "widget_update"
        assert updated["widget_id"] == created.widget_id
        assert updated["version"] == 2
        assert updated["state"]["selection"] == "alpha"

    persisted = asyncio.run(store.get(created.widget_id))
    assert persisted is not None
    assert persisted.version == 2
    assert persisted.state["selection"] == "alpha"
