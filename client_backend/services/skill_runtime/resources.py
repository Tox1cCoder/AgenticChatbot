"""Reading a skill's own supporting files, confined to its bundle.

Skills in the current convention are written as a short router plus companion
documents: a ``SKILL.md`` that says *when* to read ``root-cause-tracing.md`` or
``references/api.md``, and the agent pulls those in only when they apply. Without
a way to read them, an author's only option is to inline everything into
``SKILL.md``, which spends context on material that is usually irrelevant.

This module is the read side of that. Its whole job is confinement, because the
path arrives from a model:

* resolution stays under the *selected* skill's bundle root, re-checked after the
  OS resolves the path, so ``..`` and absolute paths cannot leave it;
* links are refused rather than followed, since a link inside the bundle can
  point anywhere outside it;
* only regular files are read, with a size cap, so a device node or a huge asset
  cannot be pulled into a prompt; and
* content must decode as UTF-8 text -- binary assets ship and run, they do not
  get read into the conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from client_backend.core.paths import is_under_root
from shared.skills.commands import is_link_like

# A companion document is prose. This bounds one read, not the bundle: a larger
# file still installs and still runs, it just is not readable into a prompt.
MAX_RESOURCE_BYTES = 256 * 1024

# Enough to describe any realistic skill without turning activation into a file
# listing. Truncation is reported rather than hidden.
MAX_LISTED_RESOURCES = 200

# Never listed or readable: provenance the sidecar wrote, caches, and the
# prepared runtime, none of which are authored content.
_EXCLUDED_NAMES = frozenset({"install.json"})
_EXCLUDED_DIRS = frozenset({"__pycache__", ".git", ".venv", "node_modules"})


class SkillResourceError(Exception):
    """A normalized, client-safe failure reading a skill resource."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class SkillResourceListing:
    """The bundle-relative files a skill offers, and whether the list was cut."""

    paths: list[str]
    truncated: bool


@dataclass(frozen=True)
class _ReadableResource:
    relative_path: str
    content: str


def _normalize_resource_path(resource_path: str) -> str:
    candidate = str(resource_path or "").strip().replace("\\", "/")
    if not candidate:
        raise SkillResourceError("resource_path is required.")
    if candidate.startswith("/") or (len(candidate) > 1 and candidate[1] == ":"):
        raise SkillResourceError("resource_path must be relative to the skill's own folder.")
    relative = Path(candidate)
    if ".." in relative.parts:
        raise SkillResourceError("resource_path points outside the skill's own folder.")
    return relative.as_posix()


def _reject_link_components(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if is_link_like(current):
            raise SkillResourceError("Symbolic links inside a skill bundle are not read.")


def _read_policy_resource(bundle_root: Path, resource_path: str) -> _ReadableResource:
    candidate = _normalize_resource_path(resource_path)
    root = bundle_root.resolve()
    relative = Path(candidate)
    if relative.name in _EXCLUDED_NAMES or _EXCLUDED_DIRS.intersection(relative.parts):
        raise SkillResourceError(f"'{candidate}' is not readable content.")

    _reject_link_components(root, relative)
    target = (root / relative).resolve()
    if not is_under_root(target, root):
        raise SkillResourceError("resource_path points outside the skill's own folder.")
    if not target.is_file():
        raise SkillResourceError(f"'{candidate}' is not a file in this skill.")

    try:
        with target.open("rb") as stream:
            raw = stream.read(MAX_RESOURCE_BYTES + 1)
    except OSError as exc:
        raise SkillResourceError(f"'{candidate}' could not be read.") from exc
    if len(raw) > MAX_RESOURCE_BYTES:
        raise SkillResourceError(
            f"'{candidate}' is larger than the {MAX_RESOURCE_BYTES // 1024} KB read limit."
        )
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillResourceError(
            f"'{candidate}' is not a UTF-8 text file. Binary assets ship with the "
            "skill and can be used by its commands, but cannot be read as text."
        ) from exc
    return _ReadableResource(relative.as_posix(), content)


def _scan_skill_resources(bundle_root: Path) -> tuple[str, ...]:
    root = bundle_root.resolve()
    found: list[str] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            readable = _read_policy_resource(root, relative)
        except SkillResourceError:
            continue
        found.append(readable.relative_path)
    found.sort()
    return tuple(found)


@lru_cache(maxsize=256)
def _cached_skill_resources(resolved_bundle_root: str, source_hash: str) -> tuple[str, ...]:
    return _scan_skill_resources(Path(resolved_bundle_root))


def list_skill_resources(
    bundle_root: Path,
    *,
    source_hash: str | None = None,
) -> SkillResourceListing:
    """List readable companion files, relative to the bundle root.

    ``SKILL.md`` is included: a nested bundle keeps it at a path the model may
    legitimately want to re-read, and excluding it would make the listing lie.
    """
    if source_hash:
        found = _cached_skill_resources(str(bundle_root.resolve()), source_hash)
    else:
        found = _scan_skill_resources(bundle_root)
    truncated = len(found) > MAX_LISTED_RESOURCES
    return SkillResourceListing(
        paths=list(found[:MAX_LISTED_RESOURCES]),
        truncated=truncated,
    )


def read_skill_resource(bundle_root: Path, resource_path: str) -> str:
    """Return one text file from inside ``bundle_root``.

    Args:
        bundle_root: The selected skill's installed bundle directory.
        resource_path: Bundle-relative path, as listed at activation.

    Raises:
        SkillResourceError: The path escapes the bundle, is not a readable
            regular text file, or exceeds the size cap.
    """
    return _read_policy_resource(bundle_root, resource_path).content
