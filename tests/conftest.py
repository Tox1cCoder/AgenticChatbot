"""Shared pytest fixtures and configuration."""

import uuid
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def any_uuid():
    return uuid.uuid4()


@pytest.fixture
def mock_session_factory():
    """Session factory that returns a mock context-manager session."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    factory = MagicMock(return_value=session)
    factory._session = session  # expose for assertions
    return factory
