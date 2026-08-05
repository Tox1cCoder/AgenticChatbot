"""Shared YAML front-matter parsing for SKILL.md documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SECRET_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# A declared name only reaches a person as a suggestion, but it comes from an
# untrusted SKILL.md, so the list is bounded and each entry has to be a name the
# secret store would actually accept.
_MAX_DECLARED_SECRETS = 20
_MAX_SECRET_NAME_LENGTH = 64


def is_valid_skill_name(value: str) -> bool:
    """Return whether a name satisfies the portable Agent Skills contract."""
    return 1 <= len(value) <= 64 and _SKILL_NAME_PATTERN.fullmatch(value) is not None


def is_valid_secret_name(value: str) -> bool:
    """Return whether a name can be bound as an environment variable.

    The single definition of a bindable secret name: the secret store validates
    what it stores against this, and front-matter parsing drops anything a skill
    declares that the store would then refuse.
    """
    return (
        1 <= len(value) <= _MAX_SECRET_NAME_LENGTH
        and _SECRET_NAME_PATTERN.fullmatch(value) is not None
    )


@dataclass(frozen=True)
class ParsedSkillFrontMatter:
    """Parsed front-matter plus the stripped markdown body."""

    name: str | None
    description: str
    category: str | None
    tags: list[str]
    secrets: list[str]
    body: str


def split_front_matter(raw: str) -> tuple[str, str] | None:
    """Split raw markdown into YAML block and stripped body."""
    stripped = raw.lstrip()
    if not stripped.startswith("---"):
        return None

    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    end_index = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = i
            break

    if end_index is None:
        return None

    yaml_block = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :]).strip()
    return yaml_block, body


def extract_yaml_value(yaml_block: str, key: str) -> str | None:
    """Extract a simple scalar or folded multi-line YAML value."""
    lines = yaml_block.split("\n")
    for i, line in enumerate(lines):
        if line[:1].isspace():
            continue

        separator_index = line.find(":")
        if separator_index < 0:
            continue

        line_key = line[:separator_index].strip()
        if line_key != key:
            continue

        value = line[separator_index + 1 :].strip()

        if value and value not in (">", "|", ">-", "|-"):
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            return value

        collected: list[str] = []
        for cont_line in lines[i + 1 :]:
            if cont_line and not cont_line[0].isspace():
                break
            collected.append(cont_line.strip())

        return " ".join(part for part in collected if part)

    return None


def parse_skill_front_matter(raw: str) -> ParsedSkillFrontMatter | None:
    """Parse SKILL.md front matter into structured metadata."""
    split_content = split_front_matter(raw)
    if split_content is None:
        return None

    yaml_block, body = split_content
    name = extract_yaml_value(yaml_block, "name")
    description = extract_yaml_value(yaml_block, "description") or ""
    category = extract_yaml_value(yaml_block, "category")
    tags_raw = extract_yaml_value(yaml_block, "tags") or ""
    tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()] if tags_raw else []

    return ParsedSkillFrontMatter(
        name=name,
        description=description,
        category=category,
        tags=tags,
        secrets=parse_declared_secrets(extract_yaml_value(yaml_block, "secrets")),
        body=body,
    )


def parse_declared_secrets(raw: str | None) -> list[str]:
    """Parse the comma-separated environment variable names a skill asks for.

    Declaring them is what lets the UI name the credential a skill needs instead
    of asking a person to remember it. Order is preserved because it is the
    author's order, duplicates collapse, and anything unbindable or beyond the
    cap is dropped rather than surfaced.
    """
    if not raw:
        return []

    declared: list[str] = []
    for candidate in raw.split(","):
        cleaned = candidate.strip().strip("\"'")
        if not cleaned or cleaned in declared or not is_valid_secret_name(cleaned):
            continue
        declared.append(cleaned)
        if len(declared) == _MAX_DECLARED_SECRETS:
            break
    return declared
