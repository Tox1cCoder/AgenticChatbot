"""Container-built repositories must be able to use the async transport.

Hand-built repositories prove the twins work; these prove the *application's*
repositories are actually wired for them.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.selector_event_loop


@pytest.fixture(scope="module")
def container():
    from app.core.container import Container

    return Container()


@pytest.mark.parametrize(
    "provider_name",
    [
        "message_repository",
        "conversation_repository",
        "document_repository",
        "conversation_compaction_repository",
    ],
)
def test_repository_receives_an_async_session_factory(container, provider_name):
    repository = getattr(container, provider_name)()
    assert repository.async_session_factory is not None


def test_message_repository_compaction_delegate_is_also_async_capable(container):
    """acreate() delegates into the compaction repository, so it needs one too."""
    repository = container.message_repository()
    assert repository._compaction_repository.async_session_factory is not None


def test_repositories_without_twins_are_left_alone(container):
    """Phase 1 wires only the hot path; the rest must be untouched."""
    repository = container.user_repository()
    assert getattr(repository, "async_session_factory", None) is None


async def test_container_repository_can_query_asynchronously(container, require_async_db):
    repository = container.message_repository()
    assert await repository._arun(lambda s: s.execute(text("select 1")).scalar_one()) == 1
