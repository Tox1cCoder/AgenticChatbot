"""
Local skills registry for client backend.

Manages scanning, loading, and caching of skills from local filesystem paths.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import get_installed_skills_root, get_profile_subdir
from client_backend.services.upstream_auth import get_upstream_auth_service
from shared.skills.front_matter import (
    extract_yaml_value,
    parse_skill_front_matter,
    split_front_matter,
)
from shared.skills.manifest import SkillManifest, load_manifest

logger = get_logger(__name__)

# Fields we're willing to echo back from install_metadata. Task 4 (the
# installer) owns the real shape of this dict; this is a thin, defensive
# allow-list so a scanned SkillMetadata never round-trips something unsafe.
_SAFE_INSTALL_SUMMARY_KEYS = ("installed", "source_hash", "bundle_name", "source")

# Matches POSIX-style ("/..."), Windows drive-letter ("C:\..." / "C:/..."),
# and UNC ("\\server\share") absolute paths, in addition to pathlib's own
# is_absolute() check, so we catch absolute-looking strings even when running
# on a different OS than the one that produced them.
_ABSOLUTE_PATH_PREFIX = re.compile(r"^(?:/|\\\\|[A-Za-z]:[\\/])")


def _looks_like_absolute_path(value: object) -> bool:
    """Return True if ``value`` is a string that contains an absolute path.

    Checks the whole string and each whitespace-delimited token, so a path
    embedded in a larger string (e.g. ``"copied from C:\\Users\\x"``) is caught,
    not only a value that is itself a bare path. This backstops the redaction
    in :meth:`SkillMetadata._install_summary` regardless of how Task 4's
    installer shapes ``install_metadata``.
    """
    if not isinstance(value, str):
        return False
    for candidate in (value, *value.split()):
        if _ABSOLUTE_PATH_PREFIX.match(candidate):
            return True
        try:
            if Path(candidate).is_absolute():
                return True
        except (OSError, ValueError):
            continue
    return False


@dataclass
class SkillMetadata:
    """Metadata for a skill."""

    name: str
    path: Path
    description: str
    content: str
    enabled: bool = True
    category: str | None = None
    tags: list[str] = None
    manifest_path: Path | None = None
    manifest: SkillManifest | None = None
    manifest_error: str | None = None
    # Reserved for the Task 4 installer to populate (source hash, bundle name,
    # etc.). Always None for freshly scanned skills; declared here so
    # to_dict()/to_sync_dict() have a stable, redacted place to surface it.
    install_metadata: dict | None = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def _execution_summary(self) -> dict:
        """JSON-safe summary of manifest presence/validity (never the raw manifest)."""
        manifest = self.manifest
        if manifest is not None:
            status = "manifest_present"
        elif self.manifest_error is not None:
            status = "invalid"
        else:
            status = "instruction_only"

        return {
            "manifest_present": manifest is not None,
            "status": status,
            "capability_count": len(manifest.capabilities) if manifest is not None else 0,
            "permissions": list(manifest.permissions) if manifest is not None else [],
        }

    def _install_summary(self) -> dict:
        """JSON-safe, redacted summary of install_metadata (never an absolute path)."""
        if not self.install_metadata:
            return {"installed": False}

        summary: dict = {}
        for key in _SAFE_INSTALL_SUMMARY_KEYS:
            if key not in self.install_metadata:
                continue
            value = self.install_metadata[key]
            if _looks_like_absolute_path(value):
                continue
            summary[key] = value

        summary.setdefault("installed", bool(self.install_metadata.get("installed", False)))
        return summary

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": str(self.path),
            "description": self.description,
            "enabled": self.enabled,
            "category": self.category,
            "tags": self.tags,
            "content_length": len(self.content) if self.content else 0,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "execution": self._execution_summary(),
            "install": self._install_summary(),
        }

    def to_sync_dict(self) -> dict:
        """
        Convert to a server-sync-safe dictionary.

        Intentionally omits absolute local filesystem paths (including
        ``manifest_path``) to avoid leaking device-specific directory
        structure to the canonical backend.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "category": self.category,
            "tags": self.tags,
            "content_length": len(self.content) if self.content else 0,
            "execution": self._execution_summary(),
            "install": self._install_summary(),
        }

    def to_catalog_entry(self) -> dict:
        """Convert to catalog entry (with content if enabled)."""
        entry = self.to_sync_dict()
        if self.enabled:
            entry["content"] = self.content
        return entry


class LocalSkillsRegistry:
    """
    Registry for local skills on the device.

    Scans configured skill roots and manages skill metadata and content.
    """

    def __init__(self, skill_roots: list[str] | None = None):
        self._explicit_skill_roots = list(skill_roots) if skill_roots is not None else None
        self.skill_roots = (
            list(self._explicit_skill_roots)
            if self._explicit_skill_roots is not None
            else list(client_settings.skills_roots)
        )
        self.skills: dict[str, SkillMetadata] = {}
        self._initialized = False
        self._active_user_id: str | None = None
        self._active_skill_roots: tuple[str, ...] = ()

    async def initialize(self) -> None:
        """
        Initialize the registry by scanning skill roots.
        """
        current_user_id = self._resolve_current_user_id()
        resolved_skill_roots = tuple(self._resolve_skill_roots())
        roots_changed = resolved_skill_roots != self._active_skill_roots

        if self._initialized and not roots_changed:
            self._apply_persisted_skill_state()
            self._active_user_id = current_user_id
            return

        logger.info("Initializing local skills registry...")
        self.skill_roots = list(resolved_skill_roots)

        if not self.skill_roots:
            logger.warning("No skill roots configured")
            self.skills = {}
            self._initialized = True
            self._active_user_id = current_user_id
            self._active_skill_roots = resolved_skill_roots
            return

        await self.scan_skills()

        self._initialized = True
        self._active_user_id = current_user_id
        self._active_skill_roots = resolved_skill_roots
        logger.info(f"Skills registry initialized with {len(self.skills)} skills")

    async def scan_skills(self) -> int:
        """
        Scan all configured skill roots for SKILL.md files.

        Returns:
            Number of skills discovered.
        """
        discovered = 0
        discovered_skills: dict[str, SkillMetadata] = {}

        for root in self.skill_roots:
            try:
                root_path = Path(root).expanduser().resolve()

                if not root_path.exists():
                    logger.warning(f"Skill root does not exist: {root}")
                    continue

                if not root_path.is_dir():
                    logger.warning(f"Skill root is not a directory: {root}")
                    continue

                logger.info(f"Scanning skill root: {root_path}")

                # Find all SKILL.md files
                skill_files = list(root_path.rglob("SKILL.md"))

                for skill_file in skill_files:
                    try:
                        skill = await self._load_skill(skill_file)
                        if skill:
                            # Check if skill already exists (from another root)
                            if skill.name in discovered_skills:
                                logger.warning(
                                    "Skill %s already exists, skipping duplicate from %s",
                                    skill.name,
                                    skill_file,
                                )
                                continue

                            discovered_skills[skill.name] = skill
                            discovered += 1
                            logger.debug(f"Loaded skill: {skill.name}")

                    except Exception as e:
                        logger.error(f"Failed to load skill from {skill_file}: {e}")

            except Exception as e:
                logger.error(f"Failed to scan skill root {root}: {e}")

        self.skills = discovered_skills
        self._apply_persisted_skill_state()
        logger.info(f"Discovered {discovered} skills")
        return discovered

    async def _load_skill(self, skill_file: Path) -> SkillMetadata | None:
        """
        Load a skill from a SKILL.md file.

        Parses YAML front matter identically to the server skills registry so that
        server and client produce the same name, description, and body content for
        the same SKILL.md file.  Front matter is stripped from the activation body
        so the model never sees raw YAML delimiters.

        Falls back to directory-name / first-line heuristics only when no valid
        front matter is present, preserving compatibility with plain markdown skills.

        If a ``skill.json`` file sits alongside SKILL.md, it is parsed and validated
        as a :class:`SkillManifest`. A missing skill.json keeps the skill instruction-
        only (unchanged behavior). A present-but-invalid skill.json (bad JSON or a
        manifest that fails validation) never fails the load — the skill still comes
        back as instruction-only, with ``manifest_error`` set to explain why.

        Args:
            skill_file: Path to the SKILL.md file.

        Returns:
            SkillMetadata if loaded successfully, None otherwise.
        """
        try:
            raw = await asyncio.to_thread(skill_file.read_text, encoding="utf-8")

            parsed = parse_skill_front_matter(raw)

            if parsed is not None:
                if not parsed.name:
                    logger.warning("Missing 'name' in front-matter of %s — skipping", skill_file)
                    return None

                name = parsed.name
                description = parsed.description
                category = parsed.category
                tags = list(parsed.tags)
                content = parsed.body
                used_plain_markdown_fallback = False
            else:
                # No front matter — use directory name and first-line heuristic.
                name = skill_file.parent.name
                content = raw
                used_plain_markdown_fallback = True

                lines = raw.strip().split("\n")
                description = ""
                for line in lines[:10]:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        description = stripped
                        break

                category = None
                tags = []

            if used_plain_markdown_fallback and not description:
                description = f"Skill: {name}"

            manifest_path: Path | None = None
            manifest: SkillManifest | None = None
            manifest_error: str | None = None

            candidate_manifest_path = skill_file.parent / "skill.json"
            # The .exists() probe lives inside the try too: a broken symlink or
            # permission error must never drop an otherwise-valid skill.
            try:
                if candidate_manifest_path.exists():
                    manifest_raw = await asyncio.to_thread(
                        candidate_manifest_path.read_text, encoding="utf-8"
                    )
                    manifest = load_manifest(json.loads(manifest_raw))
                    manifest_path = candidate_manifest_path
            except Exception as exc:
                manifest_error = f"invalid skill.json: {exc}"
                logger.warning(
                    "Failed to parse skill.json for %s: %s", candidate_manifest_path, exc
                )

            install_metadata: dict | None = None
            candidate_install_path = skill_file.parent / "install.json"
            # Same defensive shape as the skill.json block above: a bad or
            # unreadable install.json must never drop an otherwise-valid
            # skill, so parsing failures only leave install_metadata unset.
            try:
                if candidate_install_path.exists():
                    install_raw = await asyncio.to_thread(
                        candidate_install_path.read_text, encoding="utf-8"
                    )
                    parsed_install = json.loads(install_raw)
                    if isinstance(parsed_install, dict):
                        install_metadata = parsed_install
            except Exception as exc:
                logger.warning(
                    "Failed to parse install.json for %s: %s", candidate_install_path, exc
                )

            # A plain-markdown bundle (no front matter) normally takes its
            # name from the containing directory. For an installed bundle that
            # directory is hash-suffixed (e.g. "plain-skill-ab12cd34ef56"),
            # which would diverge from the name the installer recorded and
            # returned, leaving the skill undiscoverable by that name. Prefer
            # the installer's recorded bundle_name so an installed skill is
            # always found and toggled under its recorded name. Front-matter
            # names remain authoritative and are left untouched.
            if used_plain_markdown_fallback and install_metadata:
                recorded_name = install_metadata.get("bundle_name")
                if isinstance(recorded_name, str) and recorded_name.strip():
                    name = recorded_name

            return SkillMetadata(
                name=name,
                path=skill_file,
                description=description,
                content=content,
                enabled=True,  # Default enabled, overridden by persisted state
                category=category,
                tags=tags,
                manifest_path=manifest_path,
                manifest=manifest,
                manifest_error=manifest_error,
                install_metadata=install_metadata,
            )

        except Exception as e:
            logger.error(f"Error loading skill from {skill_file}: {e}")
            return None

    @staticmethod
    def _split_front_matter(raw: str) -> tuple[str, str] | None:
        """
        Split a SKILL.md file into YAML block and body.

        Returns (yaml_block, body) if valid front matter delimiters are found,
        or None if the file does not start with a valid ``---`` block.
        Mirrors the server-side ``SkillsRegistry._split_front_matter`` exactly.
        """
        return split_front_matter(raw)

    @staticmethod
    def _extract_yaml_value(yaml_block: str, key: str) -> str | None:
        """
        Extract a simple scalar or folded multi-line value from a YAML block.

        Handles:
          name: simple-value
          description: >
            multi-line
            folded text

        Mirrors the server-side ``SkillsRegistry._extract_yaml_value`` exactly.
        """
        return extract_yaml_value(yaml_block, key)

    async def reload_skill(self, skill_name: str) -> bool:
        """
        Reload a specific skill from disk.

        Args:
            skill_name: Name of the skill to reload.

        Returns:
            True if reloaded successfully.
        """
        skill = self.skills.get(skill_name)
        if not skill:
            logger.warning(f"Skill not found: {skill_name}")
            return False

        try:
            new_skill = await self._load_skill(skill.path)
            if new_skill:
                # Preserve enabled state
                new_skill.enabled = skill.enabled
                self.skills[skill_name] = new_skill
                logger.info(f"Reloaded skill: {skill_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to reload skill {skill_name}: {e}")

        return False

    def get_skill(self, skill_name: str) -> SkillMetadata | None:
        return self.skills.get(skill_name)

    def get_all_skills(self) -> list[SkillMetadata]:
        return list(self.skills.values())

    def get_enabled_skills(self) -> list[SkillMetadata]:
        return [skill for skill in self.skills.values() if skill.enabled]

    def set_skill_enabled(self, skill_name: str, enabled: bool) -> bool:
        skill = self.skills.get(skill_name)
        if not skill:
            return False

        skill.enabled = enabled
        self._persist_skill_state()
        logger.info(f"Skill {skill_name} {'enabled' if enabled else 'disabled'}")
        return True

    def bulk_set_enabled(self, skill_states: dict[str, bool]) -> int:
        count = 0
        for skill_name, enabled in skill_states.items():
            if self.set_skill_enabled(skill_name, enabled):
                count += 1
        if count:
            self._persist_skill_state()
        return count

    def get_skill_catalog(self, include_content: bool = True) -> dict:
        """
        Generate a skill catalog for syncing to the server.

        Args:
            include_content: Whether to include skill content (for enabled skills).

        Returns:
            Skill catalog dictionary.
        """
        skills_list = []

        for skill in self.skills.values():
            if include_content and skill.enabled:
                skills_list.append(skill.to_catalog_entry())
            else:
                skills_list.append(skill.to_sync_dict())

        return {
            "skills": skills_list,
            "total_count": len(self.skills),
            "enabled_count": len(self.get_enabled_skills()),
            "skill_root_count": len(self.skill_roots),
        }

    def search_skills(
        self,
        query: str,
        enabled_only: bool = False,
    ) -> list[SkillMetadata]:
        """
        Search skills by name, description, or tags.

        Args:
            query: Search query string.
            enabled_only: Only return enabled skills.

        Returns:
            List of matching SkillMetadata objects.
        """
        query_lower = query.lower()
        results = []

        skills_to_search = self.get_enabled_skills() if enabled_only else self.get_all_skills()

        for skill in skills_to_search:
            if (
                query_lower in skill.name.lower()
                or query_lower in skill.description.lower()
                or any(query_lower in tag.lower() for tag in skill.tags)
            ):
                results.append(skill)

        return results

    def get_skills_by_category(self, category: str) -> list[SkillMetadata]:
        return [
            skill
            for skill in self.skills.values()
            if skill.category and skill.category.lower() == category.lower()
        ]

    def get_categories(self) -> set[str]:
        categories = set()
        for skill in self.skills.values():
            if skill.category:
                categories.add(skill.category)
        return categories

    async def refresh(self) -> int:
        """
        Refresh the registry by rescanning all skill roots.

        Returns:
            Number of newly discovered skills.
        """
        logger.info("Refreshing skills registry...")
        self.skill_roots = self._resolve_skill_roots()

        # Keep track of existing skills
        old_count = len(self.skills)

        # Rescan
        await self.scan_skills()

        new_count = len(self.skills)
        discovered = max(new_count - old_count, 0)
        removed = max(old_count - new_count, 0)

        logger.info(
            "Refresh complete: %s total skills (%s new, %s removed)",
            new_count,
            discovered,
            removed,
        )
        self._active_user_id = self._resolve_current_user_id()
        self._active_skill_roots = tuple(self.skill_roots)
        return discovered

    def _resolve_current_user_id(self) -> str | None:
        auth_service = get_upstream_auth_service()
        return auth_service.get_current_user_id()

    def _resolve_skill_roots(self) -> list[str]:
        if self._explicit_skill_roots is not None:
            roots = list(self._explicit_skill_roots)
        else:
            roots = list(client_settings.skills_roots)

        # Installed bundles (Task 4's SkillBundleInstaller) live under the
        # active user's profile, not CLIENT_SKILLS_ROOTS, so they must be
        # scanned unconditionally here. get_profile_subdir() creates the
        # directory as a side effect, so it is only called once a user id is
        # actually available.
        current_user_id = self._resolve_current_user_id()
        if current_user_id:
            roots.append(str(get_installed_skills_root(current_user_id)))

        return roots

    def _get_skill_state_path(self) -> Path | None:
        current_user_id = self._resolve_current_user_id()
        if not current_user_id:
            return None
        return get_profile_subdir(current_user_id, "skills") / "state.json"

    def _load_persisted_skill_state(self) -> dict[str, bool]:
        path = self._get_skill_state_path()
        if path is None or not path.exists():
            return {}

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read persisted skill state: %s", exc)
            return {}

        if not isinstance(payload, dict):
            return {}

        enabled_state = payload.get("enabled")
        if not isinstance(enabled_state, dict):
            return {}

        return {str(name): bool(value) for name, value in enabled_state.items()}

    def _apply_persisted_skill_state(self) -> None:
        for skill in self.skills.values():
            skill.enabled = True

        persisted = self._load_persisted_skill_state()
        if not persisted:
            return

        for skill_name, enabled in persisted.items():
            skill = self.skills.get(skill_name)
            if skill:
                skill.enabled = enabled

    def _persist_skill_state(self) -> None:
        path = self._get_skill_state_path()
        if path is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "enabled": {
                skill_name: skill.enabled for skill_name, skill in sorted(self.skills.items())
            }
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# Global singleton
_skills_registry: LocalSkillsRegistry | None = None


def get_skills_registry() -> LocalSkillsRegistry:
    global _skills_registry
    if _skills_registry is None:
        _skills_registry = LocalSkillsRegistry()
    return _skills_registry


async def initialize_skills_registry() -> None:
    registry = get_skills_registry()
    await registry.initialize()
