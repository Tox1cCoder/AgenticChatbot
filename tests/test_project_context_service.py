"""Resolution of a conversation's system instruction, project included."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from app.services.project_context_service import ProjectContextService

OWNER_ID = uuid4()


class _StubRepository:
    """Ownership-checked stand-in for ``ProjectRepository.get_owned``.

    Returns the stored project only when queried with ``owner_id``, mirroring
    the real repository's owner-scoped query — a mismatched owner sees
    nothing, exactly like a foreign project would.
    """

    def __init__(self, project=None, owner_id=OWNER_ID):
        self._project = project
        self._owner_id = owner_id
        self.calls: list = []

    def get_owned(self, owner_id, project_id):
        self.calls.append((owner_id, project_id))
        if owner_id == self._owner_id:
            return self._project
        return None


def test_conversation_without_a_project_returns_its_persona():
    service = ProjectContextService(project_repository=_StubRepository())
    conversation = SimpleNamespace(project_id=None, persona_prompt="Be terse.", owner_id=OWNER_ID)

    assert service.resolve_system_instruction(conversation) == "Be terse."


def test_conversation_without_a_project_never_queries():
    repository = _StubRepository()
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(project_id=None, persona_prompt="Be terse.", owner_id=OWNER_ID)

    service.resolve_system_instruction(conversation)

    assert repository.calls == []


def test_project_instructions_lead_the_persona():
    project_id = uuid4()
    repository = _StubRepository(SimpleNamespace(instructions="Answer in Vietnamese."))
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(
        project_id=project_id, persona_prompt="Be terse.", owner_id=OWNER_ID
    )

    result = service.resolve_system_instruction(conversation)

    assert result == (
        "Project instructions:\nAnswer in Vietnamese.\n\n"
        "Conversation-specific instructions:\nBe terse."
    )
    assert repository.calls == [(OWNER_ID, project_id)]


def test_soft_deleted_project_reads_as_project_less():
    """get_owned filters deleted_at, so a stale pointer must not inherit."""
    repository = _StubRepository(None)
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(
        project_id=uuid4(), persona_prompt="Be terse.", owner_id=OWNER_ID
    )

    assert service.resolve_system_instruction(conversation) == "Be terse."
    assert repository.calls == [(OWNER_ID, conversation.project_id)]


def test_missing_conversation_does_not_raise():
    service = ProjectContextService(project_repository=_StubRepository())

    assert service.resolve_system_instruction(None) is None


def test_long_project_and_persona_are_not_truncated_together():
    """The resume path used to re-sanitize; 16000 characters must survive.

    Filler letters X/Y are chosen deliberately: the brief's original P/Q
    filler collides with the literal "P" in the "Project instructions:"
    header, inflating the count by one and failing for a reason unrelated
    to truncation. X and Y appear in neither header.
    """
    repository = _StubRepository(SimpleNamespace(instructions="X" * 9000))
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(
        project_id=uuid4(), persona_prompt="Y" * 9000, owner_id=OWNER_ID
    )

    result = service.resolve_system_instruction(conversation)

    assert result.count("X") == 8000
    assert result.count("Y") == 8000


def test_conversation_pointing_at_a_foreign_project_resolves_to_persona_only():
    """Closes the fail-open: the resolver uses ``get_owned``, not ``get_live``.

    A conversation whose ``owner_id`` does not match the project's owner must
    never see that project's instructions, even though the project is live
    and the conversation genuinely points at it. Before this fix, the
    resolver used the ownership-agnostic ``get_live`` and would have leaked
    the foreign project's instructions here.
    """
    project_id = uuid4()
    conversation_owner_id = uuid4()
    repository = _StubRepository(
        SimpleNamespace(instructions="Secret roadmap details."),
        owner_id=uuid4(),  # belongs to someone else entirely
    )
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(
        project_id=project_id, persona_prompt="Be terse.", owner_id=conversation_owner_id
    )

    result = service.resolve_system_instruction(conversation)

    assert result == "Be terse."
    assert repository.calls == [(conversation_owner_id, project_id)]
