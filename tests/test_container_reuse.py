"""Every lazy DI lookup must reuse the process-wide container.

``Container()`` instantiates the declarative container, which rebuilds its
``Database`` singleton — a fresh SQLAlchemy engine and connection pool. The
call sites here run per tool result, per tool call and per HTTP request, so a
per-call container leaks pools until the process dies. ``get_container()``
returns the one built at import time.

Identity on a container ``Singleton`` is the observable difference: the shared
container hands back the same object every time, a fresh one never does.
"""

from __future__ import annotations

from app.core.container import get_container


def test_offload_service_resolution_reuses_the_process_container(monkeypatch):
    from app.ai import tool_execution

    monkeypatch.setattr(
        tool_execution.settings, "tool_result_offload_enabled", True, raising=False
    )

    first = tool_execution._resolve_offload_service()

    assert first is not None
    assert first is tool_execution._resolve_offload_service()
    assert first is get_container().tool_result_blob_service()


def test_blob_download_dependencies_reuse_the_process_container():
    from app.api.tool_result_blobs import _get_repository, _get_service

    assert _get_service() is _get_service()
    assert _get_service() is get_container().tool_result_blob_service()
    assert type(_get_repository()) is type(get_container().tool_result_blob_repository())
