"""Request-schema bounds that turned client mistakes into 500s.

An explicit ``null`` for a NOT NULL column reached the database as an
IntegrityError, and a password longer than bcrypt's 72-byte input made bcrypt 5
raise at signup and at login.
"""

from __future__ import annotations

import bcrypt
import pytest
from pydantic import ValidationError

from app.core.security import hash_password, verify_password
from app.schemas.conversation import ConversationUpdate
from app.schemas.feedback import FeedbackUpdate
from app.schemas.task_plan import TaskPlanUpdate
from app.schemas.user import UserCreate


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (ConversationUpdate, {"title": None}),
        (ConversationUpdate, {"planningModeEnabled": None}),
        (FeedbackUpdate, {"rating": None}),
        (TaskPlanUpdate, {"description": None}),
        (TaskPlanUpdate, {"status": None}),
    ],
)
def test_explicit_null_for_a_required_column_is_rejected(schema, payload):
    with pytest.raises(ValidationError, match="cannot be null"):
        schema.model_validate(payload)


@pytest.mark.parametrize(
    ("schema", "payload", "unset"),
    [
        (ConversationUpdate, {"personaPrompt": None}, {"title", "planning_mode_enabled"}),
        (FeedbackUpdate, {"comment": "better"}, {"rating"}),
        (TaskPlanUpdate, {"taskMetadata": {"note": "x"}}, {"description", "status"}),
    ],
)
def test_omitting_a_required_column_leaves_it_unchanged(schema, payload, unset):
    parsed = schema.model_validate(payload)

    assert unset.isdisjoint(parsed.model_dump(exclude_unset=True))


def _signup(password: str) -> dict:
    return {"username": "someone", "email": "someone@example.com", "password": password}


def test_signup_accepts_a_password_of_exactly_72_bytes():
    assert UserCreate.model_validate(_signup("é" * 36)).password == "é" * 36


def test_signup_rejects_a_password_bcrypt_cannot_hash():
    # 37 two-byte characters: under any character limit, over bcrypt's 72 bytes.
    with pytest.raises(ValidationError, match="72 bytes"):
        UserCreate.model_validate(_signup("é" * 37))


def test_login_with_an_overlong_password_is_a_comparison_not_an_error():
    stored = hash_password("a" * 72)

    assert verify_password("a" * 72, stored) is True
    assert verify_password("b" * 100, stored) is False


def test_a_hash_made_before_bcrypt_5_still_verifies_with_the_typed_password():
    # bcrypt < 5 hashed the silently truncated first 72 bytes of a long password.
    typed = "correct horse battery staple " * 4
    legacy_hash = bcrypt.hashpw(typed.encode()[:72], bcrypt.gensalt(4)).decode()

    assert verify_password(typed, legacy_hash) is True
