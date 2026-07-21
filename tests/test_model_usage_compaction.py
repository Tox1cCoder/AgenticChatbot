"""Task 11: conversation-compaction and form-fill usage recording.

Compaction records one ``conversation_compaction`` event per generation
attempt, attributed to the verified conversation owner. The form-filler MCP
tool records a ``form_fill`` event with ``user_id=NULL`` (its stdio transport
carries no verified identity). Credential source (user vs server) is never
exposed in analytics.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

import pytest
from prometheus_client import CollectorRegistry

from app.ai.conversation_compactor import (
    CompactionCredentialResolver,
    ConversationCompactor,
)
from app.ai.conversation_memory import MEMORY_KEYS
from app.ai.token_counter import TokenCounter
from app.observability.model_usage import ModelUsageMetrics
from app.repositories.model_usage import RecordEventCommand, RecordResult
from app.usage import bind_usage_context
from app.usage.recorder import ModelUsageRecorder
from app.usage.types import UsageContext
from app.workers.conversation_compaction import _langchain_generate


class _FakeRepo:
    def __init__(self) -> None:
        self.commands: list[RecordEventCommand] = []

    def record_event(self, command: RecordEventCommand) -> RecordResult:
        self.commands.append(command)
        return RecordResult(inserted=True, event_id=uuid4())


def _recorder(repo: _FakeRepo) -> ModelUsageRecorder:
    return ModelUsageRecorder(
        repository=repo,
        enqueue_failed_write=lambda payload: None,
        metrics=ModelUsageMetrics(registry=CollectorRegistry()),
    )


def _message(sequence: int, role: str, content: str = "content") -> dict:
    return {"sequence": sequence, "role": role, "content": content, "message_metadata": {}}


def _valid_output() -> str:
    payload = {key: [] for key in MEMORY_KEYS}
    payload["facts"] = ["A stable fact."]
    return json.dumps(payload)


def _gemini_generated(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        usage_metadata=SimpleNamespace(
            prompt_token_count=200,
            candidates_token_count=50,
            total_token_count=250,
        ),
    )


def _compactor(recorder, generator):
    return ConversationCompactor(
        token_counter=TokenCounter(),
        generator=generator,
        provider="gemini",
        model="gemini-2.5-flash",
        trigger_messages=1,
        trigger_tokens=0,
        keep_recent_turns=0,
        max_summary_tokens=100_000,
        recorder=recorder,
    )


_TURN = [_message(1, "user"), _message(2, "assistant")]


@pytest.mark.asyncio
async def test_compaction_records_provider_usage_attributed_to_owner():
    repo = _FakeRepo()

    async def _generator(**_kwargs):
        return _gemini_generated(_valid_output())

    compactor = _compactor(_recorder(repo), _generator)
    user_id, conversation_id = uuid4(), uuid4()

    await compactor.compact(_TURN, user_id=user_id, conversation_id=conversation_id, force=True)

    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.status == "success"
    assert command.provider == "gemini"
    assert command.model == "gemini-2.5-flash"
    assert command.context.operation == "conversation_compaction"
    assert command.context.user_id == user_id
    assert command.context.conversation_id == conversation_id
    assert command.usage.input_tokens == 200
    assert command.usage.output_tokens == 50


@pytest.mark.asyncio
async def test_compaction_background_mode_records_without_force():
    repo = _FakeRepo()

    async def _generator(**_kwargs):
        return _gemini_generated(_valid_output())

    compactor = _compactor(_recorder(repo), _generator)
    user_id, conversation_id = uuid4(), uuid4()

    await compactor.compact(
        _TURN,
        user_id=user_id,
        conversation_id=conversation_id,
        force=False,
    )

    assert len(repo.commands) == 1
    assert repo.commands[0].context.operation == "conversation_compaction"
    assert repo.commands[0].context.user_id == user_id
    assert repo.commands[0].context.conversation_id == conversation_id


@pytest.mark.asyncio
async def test_compaction_provider_timeout_records_one_timeout_attempt():
    repo = _FakeRepo()

    async def _generator(**_kwargs):
        await asyncio.sleep(0.05)
        return _gemini_generated(_valid_output())

    compactor = _compactor(_recorder(repo), _generator)

    with pytest.raises(TimeoutError):
        await compactor.compact(
            _TURN,
            user_id=uuid4(),
            conversation_id=uuid4(),
            force=True,
            timeout_seconds=0.001,
        )

    assert len(repo.commands) == 1
    assert repo.commands[0].status == "timeout"
    assert repo.commands[0].error_code == "TimeoutError"


@pytest.mark.asyncio
async def test_compaction_records_error_attempt_and_returns_failure():
    repo = _FakeRepo()

    async def _generator(**_kwargs):
        raise ValueError("provider exploded")

    compactor = _compactor(_recorder(repo), _generator)

    result = await compactor.compact(_TURN, user_id=uuid4(), force=True)

    assert result.success is False
    assert len(repo.commands) == 1
    assert repo.commands[0].status == "error"
    assert repo.commands[0].error_code == "ValueError"


@pytest.mark.asyncio
@pytest.mark.parametrize("credential_source", ["user", "server"])
async def test_compaction_credential_source_not_in_ledger(credential_source):
    repo = _FakeRepo()
    user_id = uuid4()
    observed_api_keys = []

    async def _generator(**kwargs):
        observed_api_keys.append(kwargs["api_key"])
        return _gemini_generated(_valid_output())

    resolver = CompactionCredentialResolver(
        provider="gemini",
        server_credentials={"gemini": "server-secret"},
        user_credential_resolver=(
            lambda resolved_user_id, provider: (
                {
                    "provider_type": provider,
                    "api_key": "user-secret",
                }
                if credential_source == "user" and resolved_user_id == user_id
                else None
            )
        ),
        allow_user_credentials=True,
    )
    compactor = ConversationCompactor(
        token_counter=TokenCounter(),
        generator=_generator,
        provider="gemini",
        model="gemini-2.5-flash",
        trigger_messages=1,
        trigger_tokens=0,
        keep_recent_turns=0,
        max_summary_tokens=100_000,
        credential_resolver=resolver,
        recorder=_recorder(repo),
    )
    await compactor.compact(_TURN, user_id=user_id, force=True)

    expected_key = "user-secret" if credential_source == "user" else "server-secret"
    assert observed_api_keys == [expected_key]

    serialized = json.dumps(asdict(repo.commands[0]), default=str)
    assert expected_key not in serialized
    assert f'"credential_source": "{credential_source}"' not in serialized
    assert f'"key_source": "{credential_source}"' not in serialized
    assert "api_key" not in serialized


@pytest.mark.asyncio
async def test_compaction_without_recorder_records_nothing():
    async def _generator(**_kwargs):
        return _gemini_generated(_valid_output())

    compactor = _compactor(None, _generator)
    result = await compactor.compact(_TURN, user_id=uuid4(), force=True)

    # No recorder -> no events; compaction itself still succeeds.
    assert result.success is True


@pytest.mark.asyncio
async def test_langchain_generate_callsite_records_compaction_operation(monkeypatch):
    repo = _FakeRepo()
    provider_calls = []

    class _LLM:
        async def ainvoke(self, prompt):
            provider_calls.append(prompt)
            return _gemini_generated(_valid_output())

    def _create_model(**kwargs):
        assert kwargs["provider"] == "gemini"
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["api_key"] == "server-secret"
        return _LLM()

    monkeypatch.setattr(
        "app.workers.conversation_compaction.ModelFactory.create_model",
        _create_model,
    )
    compactor = ConversationCompactor(
        token_counter=TokenCounter(),
        generator=_langchain_generate,
        provider="gemini",
        model="gemini-2.5-flash",
        trigger_messages=1,
        trigger_tokens=0,
        keep_recent_turns=0,
        max_summary_tokens=100_000,
        credential_resolver=CompactionCredentialResolver(
            provider="gemini",
            server_credentials={"gemini": "server-secret"},
        ),
        recorder=_recorder(repo),
    )
    user_id, conversation_id = uuid4(), uuid4()

    result = await compactor.compact(
        _TURN,
        user_id=user_id,
        conversation_id=conversation_id,
        force=True,
    )

    assert result.success is True
    assert len(provider_calls) == 1
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.context.operation == "conversation_compaction"
    assert command.context.user_id == user_id
    assert command.context.conversation_id == conversation_id
    assert command.status == "success"
    assert command.usage.input_tokens == 200
    assert command.usage.output_tokens == 50


# ---------------------------------------------------------------------------
# Form-filler MCP tool (unattributed form_fill)
# ---------------------------------------------------------------------------


def test_form_fill_records_unattributed_event():
    from app.ai.mcp_servers.form_filler_server import _generate_form_content

    repo = _FakeRepo()
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_kwargs: _gemini_generated('{"ok": true}'))
    )

    response = _generate_form_content(
        client, "gemini-3-flash-preview", "prompt", recorder=_recorder(repo)
    )

    assert response.content == '{"ok": true}'
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.status == "success"
    assert command.provider == "gemini"
    assert command.model == "gemini-3-flash-preview"
    assert command.context.operation == "form_fill"
    assert command.context.user_id is None


def test_form_fill_uses_configured_model(monkeypatch):
    from app.ai.mcp_servers import form_filler_server

    captured = {}

    class _Models:
        @staticmethod
        def generate_content(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="{}")

    class _Client:
        def __init__(self, **_kwargs):
            self.models = _Models()

    monkeypatch.setattr(form_filler_server.genai, "Client", _Client)
    monkeypatch.setattr(form_filler_server.settings, "gemini_api_key", "key")
    monkeypatch.setattr(
        form_filler_server.settings,
        "form_filler_model",
        "gemini-configured-form-model",
        raising=False,
    )
    monkeypatch.setattr(form_filler_server, "_build_form_fill_recorder", lambda: None)

    assert json.loads(form_filler_server.fill_form("water leak")) == {}
    assert captured["model"] == "gemini-configured-form-model"


def test_form_fill_accepts_no_user_controlled_identity_fields():
    from app.ai.mcp_servers.form_filler_server import fill_form

    assert tuple(inspect.signature(fill_form).parameters) == ("natural_language_input",)


def test_form_fill_recorder_failure_warns_with_exception_class_only(monkeypatch, caplog):
    from app.ai.mcp_servers import form_filler_server

    def _raise():
        raise RuntimeError("database secret must not be logged")

    fake_container_module = SimpleNamespace(get_container=_raise)
    monkeypatch.setitem(sys.modules, "app.core.container", fake_container_module)

    with caplog.at_level(logging.WARNING, logger=form_filler_server.__name__):
        assert form_filler_server._build_form_fill_recorder() is None

    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["Form-fill usage recorder unavailable error=RuntimeError"]
    assert "database secret" not in caplog.text


@pytest.mark.asyncio
async def test_image_user_response_records_one_attributed_attempt():
    from app.ai.agent_config import AGENT_CONFIG
    from app.ai.agents.image_generator_agent import ImageGeneratorAgent

    repo = _FakeRepo()

    class _LLM:
        async def ainvoke(self, _messages):
            return _gemini_generated("Your sunset image is ready!")

    agent = ImageGeneratorAgent.__new__(ImageGeneratorAgent)
    agent.recorder = _recorder(repo)
    agent.langchain_model = _LLM()
    user_id, conversation_id = uuid4(), uuid4()

    with bind_usage_context(
        UsageContext(user_id=user_id, conversation_id=conversation_id, operation="chat")
    ):
        result = await agent._generate_user_facing_response("a sunset")

    assert result == "Your sunset image is ready!"
    assert len(repo.commands) == 1
    command = repo.commands[0]
    assert command.status == "success"
    assert command.provider == "gemini"
    assert command.model == AGENT_CONFIG["image_generator"]["langchain_model"]
    assert command.context.operation == "image_user_response"
    assert command.context.agent_id == "image_generator_agent"
    assert command.context.user_id == user_id
    assert command.context.conversation_id == conversation_id
    assert command.usage.input_tokens == 200
    assert command.usage.output_tokens == 50


@pytest.mark.asyncio
async def test_image_user_response_records_actual_auxiliary_model_dimensions():
    from app.ai.agents.image_generator_agent import ImageGeneratorAgent

    repo = _FakeRepo()

    class _OpenAILLM:
        model_name = "gpt-4.1-mini"
        _llm_type = "openai-chat"

        async def ainvoke(self, _messages):
            return SimpleNamespace(content="Your image is ready!")

    agent = ImageGeneratorAgent.__new__(ImageGeneratorAgent)
    agent.recorder = _recorder(repo)
    agent.langchain_model = _OpenAILLM()

    await agent._generate_user_facing_response("a city skyline")

    assert len(repo.commands) == 1
    assert repo.commands[0].provider == "openai"
    assert repo.commands[0].model == "gpt-4.1-mini"


def test_image_generator_has_no_dangling_legacy_entrypoints():
    from app.ai.agents.image_generator_agent import ImageGeneratorAgent

    assert "invoke_model" not in ImageGeneratorAgent.__dict__
    assert "process_message" not in ImageGeneratorAgent.__dict__
    assert "stream_message" not in ImageGeneratorAgent.__dict__
