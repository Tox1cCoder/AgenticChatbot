"""
Local skills registry for client backend.

Manages scanning, loading, and caching of skills from local filesystem paths.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import get_profile_subdir
from client_backend.services.upstream_auth import get_upstream_auth_service

logger = get_logger(__name__)


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

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

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
        }

    def to_sync_dict(self) -> dict:
        """
        Convert to a server-sync-safe dictionary.

        Intentionally omits absolute local filesystem paths to avoid leaking
        device-specific directory structure to the canonical backend.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "category": self.category,
            "tags": self.tags,
            "content_length": len(self.content) if self.content else 0,
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
                                    f"Skill {skill.name} already exists, skipping duplicate from {skill_file}"
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

        Args:
            skill_file: Path to the SKILL.md file.

        Returns:
            SkillMetadata if loaded successfully, None otherwise.
        """
        try:
            # Read skill content
            content = await asyncio.to_thread(skill_file.read_text, encoding="utf-8")

            # Parse skill name from parent directory
            skill_name = skill_file.parent.name

            # Extract description from first line or first paragraph
            lines = content.strip().split("\n")
            description = ""

            # Look for a description in the first few lines
            for line in lines[:10]:
                line = line.strip()
                if line and not line.startswith("#"):
                    description = line
                    break

            if not description:
                description = f"Skill: {skill_name}"

            # Parse metadata from front matter if present
            category = None
            tags = []

            if content.startswith("---"):
                # Simple front matter parsing
                front_matter_end = content.find("---", 3)
                if front_matter_end > 0:
                    front_matter = content[3:front_matter_end].strip()
                    for line in front_matter.split("\n"):
                        if "category:" in line.lower():
                            category = line.split(":", 1)[1].strip()
                        elif "tags:" in line.lower():
                            tags_str = line.split(":", 1)[1].strip()
                            tags = [t.strip() for t in tags_str.split(",")]

            return SkillMetadata(
                name=skill_name,
                path=skill_file,
                description=description,
                content=content,
                enabled=True,  # Default enabled, can be overridden by user settings
                category=category,
                tags=tags,
            )

        except Exception as e:
            logger.error(f"Error loading skill from {skill_file}: {e}")
            return None

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
            return list(self._explicit_skill_roots)
        return list(client_settings.skills_roots)

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
