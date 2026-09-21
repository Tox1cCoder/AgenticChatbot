"""Resolution of a conversation's system instruction, project included."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from app.services.project_context_service import ProjectContextService


class _StubRepository:
    def __init__(self, project=None):
        self._project = project
        self.calls: list = []

    def get_live(self, project_id):
        self.calls.append(project_id)
        return self._project


def test_conversation_without_a_project_returns_its_persona():
    service = ProjectContextService(project_repository=_StubRepository())
    conversation = SimpleNamespace(project_id=None, persona_prompt="Be terse.")

    assert service.resolve_system_instruction(conversation) == "Be terse."


def test_conversation_without_a_project_never_queries():
    repository = _StubRepository()
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(project_id=None, persona_prompt="Be terse.")

    service.resolve_system_instruction(conversation)

    assert repository.calls == []


def test_project_instructions_lead_the_persona():
    project_id = uuid4()
    repository = _StubRepository(SimpleNamespace(instructions="Answer in Vietnamese."))
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(project_id=project_id, persona_prompt="Be terse.")

    result = service.resolve_system_instruction(conversation)

    assert result == (
        "Project instructions:\nAnswer in Vietnamese.\n\n"
        "Conversation-specific instructions:\nBe terse."
    )


def test_soft_deleted_project_reads_as_project_less():
    """get_live filters deleted_at, so a stale pointer must not inherit."""
    repository = _StubRepository(None)
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(project_id=uuid4(), persona_prompt="Be terse.")

    assert service.resolve_system_instruction(conversation) == "Be terse."
    assert repository.calls == [conversation.project_id]


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
    conversation = SimpleNamespace(project_id=uuid4(), persona_prompt="Y" * 9000)

    result = service.resolve_system_instruction(conversation)

    assert result.count("X") == 8000
    assert result.count("Y") == 8000
