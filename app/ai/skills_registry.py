"""
Server-owned skills registry for checked-in repo skills.

This registry loads the server's `./skills` directory, parses Anthropic-style
`SKILL.md` files, persists enabled/disabled state, and provides a generation
counter for cache invalidation. It is an internal production skill source and
is not exposed over a public server HTTP API.
"""

import contextlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from shared.skills.front_matter import extract_yaml_value, parse_skill_front_matter, split_front_matter

logger = logging.getLogger(__name__)


@dataclass
class SkillMeta:
    """Metadata for a server-owned repo skill."""

    name: str
    description: str
    content: str  # full Markdown body (after front-matter)
    enabled: bool
    folder_path: str  # absolute path to the skill folder
    category: str | None = None
    tags: list[str] = field(default_factory=list)


class SkillsRegistry:
    """
    Central registry for skills discovery and state management.

    Mirrors the MCPRegistry pattern: singleton, generation counter, config persistence.
    """

    def __init__(self, skills_dir: str, config_path: str):
        self._skills_dir = skills_dir
        self._config_path = config_path
        self._skills: dict[str, SkillMeta] = {}
        self._generation: int = 0
        self._lock = threading.Lock()

        # Initial load
        self.reload()

    # ------------------------------------------------------------------
    # Discovery & state
    # ------------------------------------------------------------------

    def get_all_skills(self) -> list[SkillMeta]:
        """Returns all discovered skills."""
        return list(self._skills.values())

    def get_active_skills(self) -> list[SkillMeta]:
        """Returns only enabled skills."""
        return [s for s in self._skills.values() if s.enabled]

    def get_skill(self, name: str) -> SkillMeta:
        """
        Returns a single skill by name.

        Raises KeyError if not found (callers should translate to SkillNotFoundError).
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found")
        return self._skills[name]

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------

    def toggle_skill(self, name: str, enabled: bool) -> None:
        """Enable or disable a skill and persist to skills_config.json."""
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found")

        with self._lock:
            self._skills[name].enabled = enabled
            self._save_config()
            self._generation += 1
            logger.info(
                "Skill '%s' %s (generation=%d)",
                name,
                "enabled" if enabled else "disabled",
                self._generation,
            )

    # ------------------------------------------------------------------
    # Generation counter
    # ------------------------------------------------------------------

    def get_skills_generation(self) -> int:
        """
        Get the current skills generation version.

        Agents compare this with their cached version to detect
        when the skills suffix needs to be rebuilt.
        """
        return self._generation

    # ------------------------------------------------------------------
    # Reload from disk
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """
        Rescan the skills/ folder and reload config.

        New skills discovered on disk but absent from config are auto-added
        as enabled=True.
        """
        with self._lock:
            config = self._load_config()
            discovered: dict[str, SkillMeta] = {}

            skills_dir = Path(self._skills_dir)
            if not skills_dir.is_dir():
                logger.warning("Skills directory not found: %s", self._skills_dir)
                self._skills = discovered
                return

            for child in sorted(skills_dir.iterdir()):
                if not child.is_dir():
                    continue
                skill_file = child / "SKILL.md"
                if not skill_file.is_file():
                    continue

                meta = self._parse_skill(skill_file, child)
                if meta is None:
                    continue  # parse error — already logged

                # Determine enabled state from config (default: True for new skills)
                skill_config = config.get("skills", {}).get(meta.name, {})
                meta.enabled = skill_config.get("enabled", True)
                discovered[meta.name] = meta

            self._skills = discovered

            # Merge any newly discovered skills into config
            self._merge_and_save_config(config, discovered)
            self._generation += 1

            logger.info(
                "Skills reload complete: %d skills (%d enabled), generation=%d",
                len(discovered),
                sum(1 for s in discovered.values() if s.enabled),
                self._generation,
            )

    # ------------------------------------------------------------------
    # Front-matter parser (built-in, no external dependency)
    # ------------------------------------------------------------------

    def _parse_skill(self, skill_file: Path, folder: Path) -> SkillMeta | None:
        """Parse a SKILL.md file into a SkillMeta, or None on error."""
        try:
            raw = skill_file.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", skill_file, exc)
            return None

        parsed = parse_skill_front_matter(raw)
        if parsed is None:
            logger.warning("No valid YAML front-matter in %s — skipping", skill_file)
            return None

        if not parsed.name:
            logger.warning("Missing 'name' in front-matter of %s — skipping", skill_file)
            return None

        return SkillMeta(
            name=parsed.name,
            description=parsed.description,
            content=parsed.body,
            enabled=True,  # will be overridden by config
            folder_path=str(folder.resolve()),
            category=parsed.category,
            tags=list(parsed.tags),
        )

    @staticmethod
    def _split_front_matter(raw: str) -> tuple[str, str] | None:
        return split_front_matter(raw)

    @staticmethod
    def _extract_yaml_value(yaml_block: str, key: str) -> str | None:
        return extract_yaml_value(yaml_block, key)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        """Load skills_config.json, returning default if missing/corrupt."""
        config_path = Path(self._config_path)
        if not config_path.is_file():
            return {"skills": {}}
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read skills config (%s): %s", self._config_path, exc)
            return {"skills": {}}

    def _save_config(self) -> None:
        """Write current skill states to skills_config.json atomically."""
        data = {
            "skills": {name: {"enabled": skill.enabled} for name, skill in self._skills.items()}
        }
        self._atomic_write(data)

    def _merge_and_save_config(self, config: dict, discovered: dict[str, SkillMeta]) -> None:
        """Merge discovered skills into config and save."""
        skills_section = config.get("skills", {})
        changed = False

        for name in discovered:
            if name not in skills_section:
                skills_section[name] = {"enabled": True}
                changed = True

        if changed:
            config["skills"] = skills_section
            data = {"skills": {name: {"enabled": discovered[name].enabled} for name in discovered}}
            self._atomic_write(data)

    def _atomic_write(self, data: dict) -> None:
        """Write JSON data atomically using a temp file + os.replace."""
        config_path = Path(self._config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = config_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(data, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(str(tmp_path), str(config_path))
        except Exception as exc:
            logger.error("Failed to write skills config: %s", exc)
            # Clean up temp file if it exists
            with contextlib.suppress(Exception):
                tmp_path.unlink(missing_ok=True)


# ------------------------------------------------------------------
# Module-level singleton helpers (mirrors mcp_registry.py pattern)
# ------------------------------------------------------------------

_registry: SkillsRegistry | None = None


def get_skills_registry() -> SkillsRegistry:
    """Get or create the server-owned skills registry singleton."""
    global _registry
    if _registry is None:
        _registry = SkillsRegistry(
            skills_dir=str(Path(__file__).parent.parent.parent / "skills"),
            config_path=str(Path(__file__).parent / "skills_config.json"),
        )
    return _registry


def get_server_skills_registry() -> SkillsRegistry:
    """Explicit alias for the server-owned repo skills registry."""
    return get_skills_registry()


def get_skills_generation() -> int:
    """Shortcut to get the current skills generation version."""
    return get_skills_registry().get_skills_generation()
