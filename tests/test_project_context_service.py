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


# ---------------------------------------------------------------------------
# Long-term memory injection
# ---------------------------------------------------------------------------


class _StubMemoryRepository:
    """Mirrors UserMemoryRepository.list_for_user's scoping contract."""

    def __init__(self, rows=None, raises=False):
        self._rows = rows or []
        self._raises = raises
        self.calls: list = []

    def list_for_user(self, user_id, limit=20, project_id=None):
        self.calls.append((user_id, limit, project_id))
        if self._raises:
            raise RuntimeError("db down")
        return [
            row
            for row in self._rows
            if row["project_id"] is None or row["project_id"] == project_id
        ][:limit]


def _memory_service(rows=None, raises=False, project=None):
    memory = _StubMemoryRepository(rows=rows, raises=raises)
    service = ProjectContextService(
        project_repository=_StubRepository(project=project),
        user_memory_repository=memory,
    )
    return service, memory


def _enable_memory(monkeypatch, limit=20):
    monkeypatch.setattr("app.core.config.settings.enable_user_memory_tools", True)
    monkeypatch.setattr("app.core.config.settings.user_memory_max_prompt_items", limit)


class TestMemoryInjection:
    def test_saved_memory_reaches_the_system_instruction(self, monkeypatch):
        """Recall must not depend on the model deciding to call a tool."""
        _enable_memory(monkeypatch)
        service, _memory = _memory_service(
            rows=[{"content": "prod is eu-west-1", "project_id": None}]
        )
        conversation = SimpleNamespace(
            project_id=None, persona_prompt="Be terse.", owner_id=OWNER_ID
        )

        result = service.resolve_system_instruction(conversation)

        assert "Be terse." in result
        assert "prod is eu-west-1" in result
        assert "BEGIN_UNTRUSTED_USER_MEMORY" in result

    def test_memory_is_scoped_to_the_conversations_project(self, monkeypatch):
        _enable_memory(monkeypatch)
        project_id = uuid4()
        service, memory = _memory_service(
            rows=[
                {"content": "fact in this project", "project_id": str(project_id)},
                {"content": "fact in another project", "project_id": str(uuid4())},
            ],
            project=SimpleNamespace(instructions=None),
        )
        conversation = SimpleNamespace(
            project_id=project_id, persona_prompt=None, owner_id=OWNER_ID
        )

        result = service.resolve_system_instruction(conversation)

        assert memory.calls == [(str(OWNER_ID), 20, str(project_id))]
        assert "fact in this project" in result
        assert "fact in another project" not in result

    def test_memory_comes_after_the_instructions(self, monkeypatch):
        """Memory is reference data; it must never outrank the prompt."""
        _enable_memory(monkeypatch)
        service, _memory = _memory_service(
            rows=[{"content": "remembered", "project_id": None}],
            project=SimpleNamespace(instructions="Project rules."),
        )
        conversation = SimpleNamespace(
            project_id=uuid4(), persona_prompt="Persona rules.", owner_id=OWNER_ID
        )

        result = service.resolve_system_instruction(conversation)

        assert result.index("Project rules.") < result.index("Persona rules.")
        assert result.index("Persona rules.") < result.index("BEGIN_UNTRUSTED_USER_MEMORY")

    def test_the_flag_being_off_suppresses_injection(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.enable_user_memory_tools", False)
        service, memory = _memory_service(rows=[{"content": "hidden", "project_id": None}])
        conversation = SimpleNamespace(
            project_id=None, persona_prompt="Be terse.", owner_id=OWNER_ID
        )

        assert service.resolve_system_instruction(conversation) == "Be terse."
        assert memory.calls == []

    def test_a_zero_limit_suppresses_injection(self, monkeypatch):
        _enable_memory(monkeypatch, limit=0)
        service, memory = _memory_service(rows=[{"content": "hidden", "project_id": None}])
        conversation = SimpleNamespace(
            project_id=None, persona_prompt="Be terse.", owner_id=OWNER_ID
        )

        assert service.resolve_system_instruction(conversation) == "Be terse."
        assert memory.calls == []

    def test_a_failing_memory_lookup_degrades_instead_of_raising(self, monkeypatch):
        """A turn without recall is degraded; a turn that raises is broken."""
        _enable_memory(monkeypatch)
        service, _memory = _memory_service(raises=True)
        conversation = SimpleNamespace(
            project_id=None, persona_prompt="Be terse.", owner_id=OWNER_ID
        )

        assert service.resolve_system_instruction(conversation) == "Be terse."

    def test_no_memories_leaves_the_instruction_byte_identical(self, monkeypatch):
        _enable_memory(monkeypatch)
        service, _memory = _memory_service(rows=[])
        conversation = SimpleNamespace(
            project_id=None, persona_prompt="Be terse.", owner_id=OWNER_ID
        )

        assert service.resolve_system_instruction(conversation) == "Be terse."

    def test_an_unwired_memory_repository_changes_nothing(self, monkeypatch):
        _enable_memory(monkeypatch)
        service = ProjectContextService(project_repository=_StubRepository())
        conversation = SimpleNamespace(
            project_id=None, persona_prompt="Be terse.", owner_id=OWNER_ID
        )

        assert service.resolve_system_instruction(conversation) == "Be terse."

    def test_memory_alone_still_produces_an_instruction(self, monkeypatch):
        """A conversation with no persona and no project still gets recall."""
        _enable_memory(monkeypatch)
        service, _memory = _memory_service(rows=[{"content": "remembered", "project_id": None}])
        conversation = SimpleNamespace(project_id=None, persona_prompt=None, owner_id=OWNER_ID)

        result = service.resolve_system_instruction(conversation)

        assert result is not None
        assert "remembered" in result

    def test_a_conversation_with_no_owner_loads_nothing(self, monkeypatch):
        _enable_memory(monkeypatch)
        service, memory = _memory_service(rows=[{"content": "hidden", "project_id": None}])
        conversation = SimpleNamespace(project_id=None, persona_prompt="Be terse.", owner_id=None)

        assert service.resolve_system_instruction(conversation) == "Be terse."
        assert memory.calls == []
