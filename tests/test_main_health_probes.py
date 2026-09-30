"""Dependency health probes in ``app.main``.

The Qdrant probe read ``CollectionInfo.vectors_count``, which qdrant-client no
longer has, so a reachable Qdrant reported "Failed to connect" and
``/health/all`` was permanently degraded. The probes also leaked their clients
on the error paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import qdrant_client
from fastapi.testclient import TestClient

import app.main as main
from app.core.config import settings


class _FakeQdrant:
    instances: list[_FakeQdrant] = []

    def __init__(self, url: str) -> None:
        self.closed = False
        _FakeQdrant.instances.append(self)

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=settings.qdrant_collection_name)])

    def get_collection(self, name):
        # The real CollectionInfo: points_count exists, vectors_count does not.
        return SimpleNamespace(points_count=42, indexed_vectors_count=40)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_qdrant(monkeypatch):
    _FakeQdrant.instances = []
    monkeypatch.setattr(qdrant_client, "QdrantClient", _FakeQdrant)
    return _FakeQdrant


def test_a_reachable_qdrant_reports_healthy(fake_qdrant):
    response = TestClient(main.app).get("/health/qdrant")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "collection": settings.qdrant_collection_name,
        "vectors_count": 42,
        "message": "Qdrant connection successful",
    }


def test_the_qdrant_probe_closes_its_client(fake_qdrant):
    TestClient(main.app).get("/health/qdrant")

    (client,) = fake_qdrant.instances
    assert client.closed is True


def test_the_redis_probe_closes_its_client_when_ping_raises(monkeypatch):
    clients = []

    class _FailingRedis:
        def __init__(self) -> None:
            self.closed = False

        def ping(self):
            raise ConnectionError("refused")

        def close(self) -> None:
            self.closed = True

    def _from_url(*_args, **_kwargs):
        clients.append(_FailingRedis())
        return clients[-1]

    monkeypatch.setattr(main.Redis, "from_url", staticmethod(_from_url))

    result = main._probe_redis()

    assert result["status"] == "unhealthy"
    assert clients and clients[0].closed is True


# The probes are unauthenticated, so a failure names the exception class and
# nothing else: connection errors carry hosts, ports and credentialed URLs.
_SECRET_DETAIL = "redis://:hunter2@10.0.0.5:6379 refused"


def test_the_redis_probe_reports_only_the_error_class(monkeypatch):
    class _FailingRedis:
        def ping(self):
            raise ConnectionError(_SECRET_DETAIL)

        def close(self) -> None:
            return None

    monkeypatch.setattr(main.Redis, "from_url", staticmethod(lambda *_a, **_k: _FailingRedis()))

    result = main._probe_redis()

    assert result == {
        "status": "unhealthy",
        "error": "ConnectionError",
        "message": "Failed to connect to Redis",
    }


def test_the_qdrant_probe_reports_only_the_error_class(monkeypatch):
    class _FailingQdrant:
        def __init__(self, url: str) -> None:
            return None

        def get_collections(self):
            raise RuntimeError(_SECRET_DETAIL)

        def close(self) -> None:
            return None

    monkeypatch.setattr(qdrant_client, "QdrantClient", _FailingQdrant)

    result = main._probe_qdrant()

    assert result == {
        "status": "unhealthy",
        "error": "RuntimeError",
        "message": "Failed to connect to Qdrant",
    }


def test_the_celery_probe_reports_only_the_error_class(monkeypatch):
    def _inspect():
        raise OSError(_SECRET_DETAIL)

    monkeypatch.setattr(main.celery_app.control, "inspect", _inspect)

    result = main._probe_celery()

    assert result == {
        "status": "unhealthy",
        "error": "OSError",
        "message": "Failed to connect to Celery",
    }
