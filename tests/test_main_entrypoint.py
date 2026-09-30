"""``python -m app.main`` binds where the settings say.

The entry point hard-coded ``0.0.0.0:8000``, so ``API_HOST``/``API_PORT`` were
documented and read by nothing.
"""

from __future__ import annotations

import asyncio
import runpy
import warnings

import uvicorn

from app.core.config import settings


def test_the_entrypoint_binds_the_configured_host_and_port(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    # The entry point sets a process-wide loop policy on Windows; keep it out
    # of the test process.
    monkeypatch.setattr(asyncio, "set_event_loop_policy", lambda _policy: None)
    monkeypatch.setattr(settings, "api_host", "127.0.0.2")
    monkeypatch.setattr(settings, "api_port", 8765)

    with warnings.catch_warnings():
        # runpy warns that app.main is already imported; re-running it is the point.
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("app.main", run_name="__main__")

    assert len(calls) == 1
    assert calls[0]["host"] == "127.0.0.2"
    assert calls[0]["port"] == 8765


def test_an_unset_api_host_binds_loopback_only():
    """Listening on every interface is opt-in (``API_HOST=0.0.0.0``), not the default."""
    from app.core.config import Settings

    assert Settings.model_fields["api_host"].default == "127.0.0.1"
