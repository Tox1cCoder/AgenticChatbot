from uuid import uuid4

from app.ai.user_memory_tools import (
    create_user_memory_tools,
    format_memories_for_prompt,
)


class FakeMemoryRepository:
    """In-memory stand-in mirroring UserMemoryRepository's scoping contract."""

    def __init__(self, conversation_projects: dict | None = None):
        self.rows = []
        self.conversation_projects = conversation_projects or {}
        self.resolve_calls = []

    def create(self, *, user_id, content, source, project_id=None):
        row = {
            "id": uuid4(),
            "user_id": user_id,
            "content": content,
            "source": source,
            "project_id": project_id,
        }
        self.rows.append(row)
        return row

    def list_for_user(self, user_id, limit=20, project_id=None):
        return [
            row
            for row in self.rows
            if row["user_id"] == user_id
            and (row["project_id"] is None or row["project_id"] == project_id)
        ][:limit]

    def resolve_project_id(self, user_id, conversation_id):
        self.resolve_calls.append((user_id, conversation_id))
        return self.conversation_projects.get(conversation_id)

    def delete_for_user(self, memory_id, user_id):
        before = len(self.rows)
        self.rows = [
            row
            for row in self.rows
            if not (str(row["id"]) == str(memory_id) and row["user_id"] == user_id)
        ]
        return len(self.rows) < before


def _tools(repo, user_id, conversation_id=None):
    tools = create_user_memory_tools(
        repository=repo, user_id=user_id, conversation_id=conversation_id
    )
    return {tool.name: tool for tool in tools}


def test_memory_tools_are_user_scoped():
    repo = FakeMemoryRepository()
    user_id = str(uuid4())
    other_user_id = str(uuid4())
    tools = _tools(repo, user_id)

    tools["remember_memory"].invoke(
        {"content": "User prefers concise answers.", "source": "explicit_user_request"}
    )
    repo.create(user_id=other_user_id, content="Other user's memory", source="test")

    result = tools["list_memories"].invoke({"limit": 20})

    assert "User prefers concise answers." in result
    assert "Other user's memory" not in result


def test_no_user_binds_no_tools():
    assert create_user_memory_tools(repository=FakeMemoryRepository(), user_id=None) == []


class TestProjectScoping:
    def test_a_save_is_filed_under_the_conversations_project(self):
        project_id = str(uuid4())
        repo = FakeMemoryRepository({"conv-1": project_id})
        user_id = str(uuid4())

        _tools(repo, user_id, "conv-1")["remember_memory"].invoke({"content": "prod is eu-west-1"})

        assert repo.rows[0]["project_id"] == project_id

    def test_a_save_outside_any_project_is_global(self):
        repo = FakeMemoryRepository({})
        user_id = str(uuid4())

        _tools(repo, user_id, "conv-loose")["remember_memory"].invoke({"content": "likes tables"})

        assert repo.rows[0]["project_id"] is None

    def test_another_conversation_in_the_same_project_recalls_the_fact(self):
        """This is the whole point: save in one conversation, recall in
        another belonging to the same project."""
        project_id = str(uuid4())
        repo = FakeMemoryRepository({"conv-1": project_id, "conv-2": project_id})
        user_id = str(uuid4())

        _tools(repo, user_id, "conv-1")["remember_memory"].invoke({"content": "prod is eu-west-1"})
        result = _tools(repo, user_id, "conv-2")["list_memories"].invoke({"limit": 20})

        assert "prod is eu-west-1" in result

    def test_a_conversation_in_another_project_does_not_recall_it(self):
        repo = FakeMemoryRepository({"conv-1": str(uuid4()), "conv-2": str(uuid4())})
        user_id = str(uuid4())

        _tools(repo, user_id, "conv-1")["remember_memory"].invoke({"content": "prod is eu-west-1"})
        result = _tools(repo, user_id, "conv-2")["list_memories"].invoke({"limit": 20})

        assert "prod is eu-west-1" not in result

    def test_the_project_is_resolved_per_call_not_at_bind_time(self):
        """A conversation moved between projects must file its next save under
        the new one without rebinding the tools."""
        repo = FakeMemoryRepository({"conv-1": "project-a"})
        user_id = str(uuid4())
        tools = _tools(repo, user_id, "conv-1")

        tools["remember_memory"].invoke({"content": "first"})
        repo.conversation_projects["conv-1"] = "project-b"
        tools["remember_memory"].invoke({"content": "second"})

        assert [row["project_id"] for row in repo.rows] == ["project-a", "project-b"]

    def test_an_unresolvable_project_falls_back_to_global(self):
        class Exploding(FakeMemoryRepository):
            def resolve_project_id(self, user_id, conversation_id):
                raise RuntimeError("db down")

        repo = Exploding()
        result = _tools(repo, str(uuid4()), "conv-1")["remember_memory"].invoke({"content": "x"})

        assert "saved" in result.lower()
        assert repo.rows[0]["project_id"] is None


class TestForgetByPrefix:
    def test_list_exposes_an_id_the_model_can_forget_with(self):
        repo = FakeMemoryRepository()
        user_id = str(uuid4())
        tools = _tools(repo, user_id)
        tools["remember_memory"].invoke({"content": "drop me"})

        listing = tools["list_memories"].invoke({"limit": 20})
        prefix = listing.split("[", 1)[1].split("]", 1)[0]

        assert tools["forget_memory"].invoke({"memory_id": prefix}) == "Memory removed."
        assert tools["list_memories"].invoke({"limit": 20}) == "No saved memories."

    def test_a_full_id_still_works(self):
        repo = FakeMemoryRepository()
        user_id = str(uuid4())
        tools = _tools(repo, user_id)
        tools["remember_memory"].invoke({"content": "drop me"})

        full_id = str(repo.rows[0]["id"])
        assert tools["forget_memory"].invoke({"memory_id": full_id}) == "Memory removed."

    def test_an_ambiguous_prefix_deletes_nothing(self):
        repo = FakeMemoryRepository()
        user_id = str(uuid4())
        tools = _tools(repo, user_id)
        tools["remember_memory"].invoke({"content": "one"})
        tools["remember_memory"].invoke({"content": "two"})
        # Force a shared prefix.
        repo.rows[0]["id"] = "abcdef01-aaaa"
        repo.rows[1]["id"] = "abcdef01-bbbb"

        result = tools["forget_memory"].invoke({"memory_id": "abcdef01"})

        assert "matches 2 memories" in result
        assert len(repo.rows) == 2

    def test_an_unknown_id_reports_not_found(self):
        repo = FakeMemoryRepository()
        tools = _tools(repo, str(uuid4()))
        assert tools["forget_memory"].invoke({"memory_id": "nope"}) == "Memory not found."

    def test_an_empty_id_is_rejected(self):
        repo = FakeMemoryRepository()
        tools = _tools(repo, str(uuid4()))
        assert "required" in tools["forget_memory"].invoke({"memory_id": "  "})


class TestRememberValidation:
    def test_blank_content_is_rejected(self):
        repo = FakeMemoryRepository()
        tools = _tools(repo, str(uuid4()))
        assert "required" in tools["remember_memory"].invoke({"content": "   "})
        assert repo.rows == []


class TestPromptFormatting:
    def test_no_memories_render_as_empty_string(self):
        assert format_memories_for_prompt([]) == ""

    def test_blank_memories_render_as_empty_string(self):
        assert format_memories_for_prompt([{"content": "  "}]) == ""

    def test_memories_are_fenced_as_untrusted_reference_data(self):
        """Remembered text is replayed into a later system prompt. Without the
        fence, 'remember that you must always X' becomes a standing order."""
        block = format_memories_for_prompt(
            [{"content": "prod is eu-west-1"}, {"content": "prefers tables"}]
        )

        assert block.startswith("BEGIN_UNTRUSTED_USER_MEMORY")
        assert block.endswith("END_UNTRUSTED_USER_MEMORY")
        assert "not instructions" in block
        assert "- prod is eu-west-1" in block
        assert "- prefers tables" in block

    def test_orm_rows_and_dicts_both_render(self):
        class Row:
            content = "from an orm row"

        assert "from an orm row" in format_memories_for_prompt([Row()])
