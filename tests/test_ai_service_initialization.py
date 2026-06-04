from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from dependency_injector import providers


def test_ai_service_singleton_serializes_concurrent_first_construction(monkeypatch):
    """Concurrent frontend bootstrap calls must not build duplicate AI workflows."""
    from app.core import container as container_module

    container = container_module.container
    container.ai_service.reset()

    monkeypatch.setattr(container_module.settings, "enable_langgraph_checkpoints", False)

    container.qdrant_client.override(providers.Object(object()))
    container.rag_embedding_service.override(providers.Object(object()))
    container.document_repository.override(providers.Factory(lambda: object()))
    container.model_config_service.override(providers.Factory(lambda: object()))
    container.history_provider.override(providers.Factory(lambda: object()))
    container.conversation_repository.override(providers.Factory(lambda: object()))

    active_creations = 0
    max_active_creations = 0
    create_calls = 0
    counter_lock = threading.Lock()
    start_barrier = threading.Barrier(8)

    def fake_create_workflow(**_kwargs):
        nonlocal active_creations, max_active_creations, create_calls
        with counter_lock:
            active_creations += 1
            create_calls += 1
            max_active_creations = max(max_active_creations, active_creations)
        time.sleep(0.05)
        with counter_lock:
            active_creations -= 1
        return SimpleNamespace()

    monkeypatch.setattr(container_module, "create_workflow", fake_create_workflow)

    def resolve_service():
        start_barrier.wait(timeout=5)
        return container.ai_service()

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            services = list(executor.map(lambda _idx: resolve_service(), range(8)))
    finally:
        container.ai_service.reset()
        container.qdrant_client.reset_override()
        container.rag_embedding_service.reset_override()
        container.document_repository.reset_override()
        container.model_config_service.reset_override()
        container.history_provider.reset_override()
        container.conversation_repository.reset_override()

    assert create_calls == 1
    assert max_active_creations == 1
    assert len({id(service) for service in services}) == 1
