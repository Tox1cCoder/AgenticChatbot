from __future__ import annotations

import app.ai.checkpoint as checkpoint_module
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


_CHECKPOINT_TYPES = (
    AgentType,
    MessageRole,
    AgentResponse,
    AgentMessage,
)


def _expected_json_allowlist() -> list[tuple[str, ...]]:
    return [(*symbol.__module__.split("."), symbol.__name__) for symbol in _CHECKPOINT_TYPES]


def _expected_msgpack_allowlist() -> list[tuple[str, str]]:
    return [(symbol.__module__, symbol.__name__) for symbol in _CHECKPOINT_TYPES]


def test_build_checkpoint_serializer_uses_json_allowlist_for_current_langgraph(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSerializer:
        def __init__(self, *, allowed_json_modules=None):
            captured["allowed_json_modules"] = allowed_json_modules

    monkeypatch.setattr(checkpoint_module, "JsonPlusSerializer", FakeSerializer)

    serializer = checkpoint_module._build_checkpoint_serializer()

    assert isinstance(serializer, FakeSerializer)
    assert list(captured["allowed_json_modules"]) == _expected_json_allowlist()


def test_build_checkpoint_serializer_adds_msgpack_allowlist_when_supported(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSerializer:
        def __init__(self, *, allowed_json_modules=None, allowed_msgpack_modules=None):
            captured["allowed_json_modules"] = allowed_json_modules
            captured["allowed_msgpack_modules"] = allowed_msgpack_modules

    monkeypatch.setattr(checkpoint_module, "JsonPlusSerializer", FakeSerializer)

    serializer = checkpoint_module._build_checkpoint_serializer()

    assert isinstance(serializer, FakeSerializer)
    assert list(captured["allowed_json_modules"]) == _expected_json_allowlist()
    assert list(captured["allowed_msgpack_modules"]) == _expected_msgpack_allowlist()
