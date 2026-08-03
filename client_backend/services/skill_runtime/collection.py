"""Discovering the skills inside an uploaded archive.

A skill library is distributed as one repository holding many skills -- obra's
superpowers ships fourteen under ``skills/`` with a plugin manifest at the root.
Requiring one archive per skill means asking a person to unzip a download,
repackage fourteen folders, and upload each one, so this module treats a
collection as the general shape and a single skill as a collection of one.

Two rules keep the result honest. A skill folder is the directory that *directly*
contains a ``SKILL.md``, never an ancestor, because executable assets are
resolved relative to that folder. And the collection's identity comes from a
plugin manifest when the archive ships one, so an update can be matched against
what is installed instead of guessing from a filename.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from client_backend.core.logging import get_logger

logger = get_logger(__name__)

# Manifest locations in use across harnesses. All carry the same fields; the
# first one present wins, so an archive targeting several agents still resolves.
_MANIFEST_PATHS = (
    ".claude-plugin/plugin.json",
    ".codex-plugin/plugin.json",
    ".cursor-plugin/plugin.json",
    "plugin.json",
)

# A library of this size is already unusual; beyond it the archive is more likely
# a mistake (a whole workspace, a monorepo) than a skill collection.
MAX_SKILLS_PER_COLLECTION = 100

_IGNORED_DIRS = frozenset({".git", "__pycache__", ".venv", "node_modules", "dist", "build"})


@dataclass(frozen=True)
class CollectionManifest:
    """Identity of a distributed skill library."""

    name: str
    version: str | None = None
    description: str | None = None


@dataclass
class DiscoveredCollection:
    """The skills an archive contains, and what the archive calls itself."""

    manifest: CollectionManifest
    skill_roots: list[Path] = field(default_factory=list)

    @property
    def is_single_skill(self) -> bool:
        return len(self.skill_roots) == 1


def _load_manifest(root: Path) -> CollectionManifest | None:
    """Read a plugin manifest if the archive ships one."""
    for relative in _MANIFEST_PATHS:
        candidate = root / relative
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("ignoring unreadable plugin manifest %s: %s", relative, exc)
            continue
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or "").strip()
        if not name:
            continue
        return CollectionManifest(
            name=name,
            version=str(payload.get("version") or "").strip() or None,
            description=str(payload.get("description") or "").strip() or None,
        )
    return None


def find_skill_roots(root: Path) -> list[Path]:
    """Return every directory that directly contains a ``SKILL.md``.

    Ancestors are excluded deliberately. A skill's ``bin/`` and ``scripts/`` are
    resolved relative to the folder holding its ``SKILL.md``, so treating a parent
    as the root would publish a skill whose commands are not where it says.
    """
    found: list[Path] = []
    for candidate in root.rglob("SKILL.md"):
        if not candidate.is_file():
            continue
        relative_parts = candidate.relative_to(root).parts[:-1]
        if _IGNORED_DIRS.intersection(relative_parts):
            continue
        found.append(candidate.parent)
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def discover_collection(root: Path, *, fallback_name: str) -> DiscoveredCollection:
    """Describe an extracted archive as a collection of skills.

    A single-skill archive keeps the *archive root* as its bundle root rather
    than the folder holding ``SKILL.md``. That preserves the documented nested
    shape, where a distribution puts its skill at ``skills/my-skill/SKILL.md``
    while its ``bin/`` and ``scripts/`` stay at the top; narrowing to the inner
    folder there would silently drop the skill's commands. Only a genuine
    collection gives each skill its own folder as a root, because with several
    skills there is no shared root that could belong to any one of them.

    Args:
        root: Extraction root, after any single wrapper directory is stripped.
        fallback_name: Used when the archive ships no manifest -- normally the
            uploaded filename, which is what a person recognizes.
    """
    manifest = _load_manifest(root) or CollectionManifest(name=fallback_name)
    skill_roots = find_skill_roots(root)
    if len(skill_roots) == 1:
        skill_roots = [root]
    return DiscoveredCollection(manifest=manifest, skill_roots=skill_roots)
