from uuid import uuid4

from app.ai.user_memory_tools import create_user_memory_tools


class FakeMemoryRepository:
    def __init__(self):
        self.rows = []

    def create(self, *, user_id, content, source):
        row = {"id": uuid4(), "user_id": user_id, "content": content, "source": source}
        self.rows.append(row)
        return row

    def list_for_user(self, user_id, limit=20):
        return [row for row in self.rows if row["user_id"] == user_id][:limit]

    def delete_for_user(self, memory_id, user_id):
        before = len(self.rows)
        self.rows = [row for row in self.rows if not (str(row["id"]) == str(memory_id) and row["user_id"] == user_id)]
        return len(self.rows) < before


def test_memory_tools_are_user_scoped():
    repo = FakeMemoryRepository()
    user_id = str(uuid4())
    other_user_id = str(uuid4())
    tools = create_user_memory_tools(repository=repo, user_id=user_id)
    remember = next(tool for tool in tools if tool.name == "remember_memory")
    list_memory = next(tool for tool in tools if tool.name == "list_memories")

    remember.invoke({"content": "User prefers concise answers.", "source": "explicit_user_request"})
    repo.create(user_id=other_user_id, content="Other user's memory", source="test")

    result = list_memory.invoke({"limit": 20})

    assert "User prefers concise answers." in result
    assert "Other user's memory" not in result
