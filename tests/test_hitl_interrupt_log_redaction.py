"""The device runtime WebSocket connects with the session id, so it stays out of logs."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from uuid import uuid4

from app.repositories.hitl_interrupt import HITLInterruptRepository


class _Session:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, _statement):
        return SimpleNamespace(rowcount=2)

    def commit(self):
        return None


def test_expiring_stale_interrupts_does_not_log_the_session_id(caplog):
    session_id = "sidecar-session-7f3a9c"
    repository = HITLInterruptRepository(lambda: _Session())

    with caplog.at_level(logging.INFO, logger="app.repositories.hitl_interrupt"):
        expired = repository.expire_stale_client_tool_interrupts(uuid4(), session_id)

    assert expired == 2
    assert "Expired 2 stale client-tool interrupt(s)" in caplog.text
    assert session_id not in caplog.text
