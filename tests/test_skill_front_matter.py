"""Front-matter parsing rules that reach a person or a credential store.

A ``SKILL.md`` is untrusted content. What it declares is allowed to *suggest*
things -- a credential name to bind, a category to group under -- so each
declared value is bounded and validated here rather than where it is displayed.
"""

from shared.skills.front_matter import (
    is_valid_secret_name,
    parse_declared_secrets,
    parse_skill_front_matter,
)


def _document(front_matter: str) -> str:
    return f"---\n{front_matter}\n---\n\nBody text.\n"


def test_declared_secrets_are_parsed_in_author_order():
    parsed = parse_skill_front_matter(
        _document("name: calendar\ndescription: Demo\nsecrets: CALENDAR_TOKEN, CALENDAR_ID")
    )

    assert parsed.secrets == ["CALENDAR_TOKEN", "CALENDAR_ID"]


def test_secrets_absent_means_no_declaration():
    parsed = parse_skill_front_matter(_document("name: calendar\ndescription: Demo"))

    assert parsed.secrets == []


def test_unbindable_names_are_dropped_rather_than_surfaced():
    """The store would refuse these, so declaring them must not suggest them."""
    parsed = parse_skill_front_matter(
        _document("name: calendar\ndescription: Demo\nsecrets: OK_TOKEN, 1BAD, has-dash, sp ace")
    )

    assert parsed.secrets == ["OK_TOKEN"]


def test_duplicate_declarations_collapse():
    assert parse_declared_secrets("TOKEN, TOKEN , 'TOKEN'") == ["TOKEN"]


def test_quoted_names_are_unwrapped():
    assert parse_declared_secrets('"TOKEN", \'OTHER\'') == ["TOKEN", "OTHER"]


def test_declaration_count_is_capped():
    """A hostile bundle must not be able to flood the credential UI."""
    declared = parse_declared_secrets(",".join(f"TOKEN_{index}" for index in range(50)))

    assert len(declared) == 20
    assert declared[0] == "TOKEN_0"


def test_secret_name_validation_matches_environment_variable_shape():
    assert is_valid_secret_name("ACCESS_TOKEN") is True
    assert is_valid_secret_name("_private") is True
    assert is_valid_secret_name("token2") is True
    assert is_valid_secret_name("2token") is False
    assert is_valid_secret_name("has-dash") is False
    assert is_valid_secret_name("") is False
    assert is_valid_secret_name("A" * 65) is False
