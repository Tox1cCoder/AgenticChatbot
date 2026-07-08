"""Graph refactor contract guards (Tasks 6 & 8).

Task 6 removed the legacy no-op ``summarize`` node. The plan's original
pre-deletion gate — ``SELECT COUNT(*) FROM checkpoints WHERE checkpoint::text
ILIKE '%summarize%'`` == 0 — is unusable as a permanent guard: that substring
also matches historical ``versions_seen`` / ``channel_versions`` bookkeeping
(retained forever on any thread that ever ran the node) and ordinary user
message content ("please summarize ..."), so it is non-zero on any real DB and
does not indicate danger.

The real, verified invariant is that no live checkpoint *schedules* the removed
node: a pending write whose channel is exactly ``summarize`` would cause a
"node not found" error on resume. That count is asserted here. (The exhaustive
one-time pre-deletion gate loaded every summarize-referencing thread's state
through the compiled graph and confirmed 0 of 247 had ``summarize`` in
``.next``.)
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from app.core.config import settings


def _engine():
    if not settings.database_url.startswith("postgresql"):
        pytest.skip("checkpoint contract requires PostgreSQL")
    return create_engine(settings.database_url)


def test_no_live_checkpoint_write_schedules_removed_summarize_node():
    with _engine().connect() as conn:
        pending = conn.execute(
            text("SELECT COUNT(*) FROM checkpoint_writes WHERE channel = 'summarize'")
        ).scalar_one()
    assert pending == 0
