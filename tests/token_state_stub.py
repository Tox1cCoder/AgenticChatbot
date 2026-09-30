"""Stand in for the users table behind the token revocation check.

The auth dependencies look up every token's user (``app.core.security.token_version``).
Tests that mint tokens for made-up users call :func:`stub_token_states` so the
lookup never reaches a database; every user not listed is live at version 0.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from app.core.security import token_version
from app.core.security.token_version import TokenState, TokenStateCache

LIVE = TokenState(version=0, deleted=False)


def stub_token_states(
    monkeypatch: pytest.MonkeyPatch,
    states: dict[UUID, TokenState | None] | None = None,
    **cache_options,
) -> dict[UUID, TokenState | None]:
    """Install a cache over ``states`` and return the dict, for the test to edit."""
    states = {} if states is None else states
    cache = TokenStateCache(lambda user_id: states.get(user_id, LIVE), **cache_options)
    monkeypatch.setattr(token_version, "token_state_cache", cache)
    return states
