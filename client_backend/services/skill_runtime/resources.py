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
from pathlib import Path

from client_backend.core.logging import get_logger
from client_backend.core.paths import is_under_root
from shared.skills.commands import is_link_like

logger = get_logger(__name__)

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


def list_skill_resources(bundle_root: Path) -> SkillResourceListing:
    """List readable companion files, relative to the bundle root.

    ``SKILL.md`` is included: a nested bundle keeps it at a path the model may
    legitimately want to re-read, and excluding it would make the listing lie.
    """
    root = bundle_root.resolve()
    found: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file() or is_link_like(path):
            continue
        relative = path.relative_to(root)
        if relative.name in _EXCLUDED_NAMES:
            continue
        if _EXCLUDED_DIRS.intersection(relative.parts):
            continue
        found.append(relative.as_posix())

    # Sorted as strings, not as Path objects: Path comparison is case-insensitive
    # on Windows, so sorting paths would list a bundle differently depending on
    # the host and make the advertised order untestable.
    found.sort()
    truncated = len(found) > MAX_LISTED_RESOURCES
    return SkillResourceListing(paths=found[:MAX_LISTED_RESOURCES], truncated=truncated)


def read_skill_resource(bundle_root: Path, resource_path: str) -> str:
    """Return one text file from inside ``bundle_root``.

    Args:
        bundle_root: The selected skill's installed bundle directory.
        resource_path: Bundle-relative path, as listed at activation.

    Raises:
        SkillResourceError: The path escapes the bundle, is not a readable
            regular text file, or exceeds the size cap.
    """
    candidate = str(resource_path or "").strip().replace("\\", "/")
    if not candidate:
        raise SkillResourceError("resource_path is required.")
    if candidate.startswith("/") or (len(candidate) > 1 and candidate[1] == ":"):
        raise SkillResourceError(
            "resource_path must be relative to the skill's own folder."
        )

    root = bundle_root.resolve()
    target = (root / candidate).resolve()

    # Checked after resolution, not before: only the resolved path reveals where
    # a chain of `..` segments or an intermediate link actually lands.
    if not is_under_root(target, root):
        raise SkillResourceError(
            "resource_path points outside the skill's own folder."
        )
    if is_link_like(target):
        raise SkillResourceError("Symbolic links inside a skill bundle are not read.")
    if not target.is_file():
        raise SkillResourceError(f"'{candidate}' is not a file in this skill.")
    if _EXCLUDED_DIRS.intersection(target.relative_to(root).parts):
        raise SkillResourceError(f"'{candidate}' is not readable content.")

    size = target.stat().st_size
    if size > MAX_RESOURCE_BYTES:
        raise SkillResourceError(
            f"'{candidate}' is {size // 1024} KB, larger than the "
            f"{MAX_RESOURCE_BYTES // 1024} KB read limit."
        )

    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillResourceError(
            f"'{candidate}' is not a UTF-8 text file. Binary assets ship with the "
            "skill and can be used by its commands, but cannot be read as text."
        ) from exc
    except OSError as exc:
        raise SkillResourceError(f"'{candidate}' could not be read.") from exc
