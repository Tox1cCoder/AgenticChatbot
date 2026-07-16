"""Readiness and fixed tool-catalog projection for standard Agent Skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from client_backend.services.local_skills_registry import SkillMetadata

RUN_SKILL_COMMAND = "run_skill_command"
SKILL_SERVER_NAME_PREFIX = "skill_"

COMMAND_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "argv": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "cwd": {
            "type": "string",
            "enum": ["skill", "workspace"],
            "default": "workspace",
        },
    },
    "required": ["argv"],
    "additionalProperties": False,
}

REPAIR_HINT_TYPES = frozenset(
    {
        "setup_skill",
        "rebuild_skill_runtime",
        "inspect_setup_failure",
        "permission_required",
    }
)


@dataclass
class SkillReadiness:
    """Whether a bundle can currently service its fixed command tool."""

    status: str
    setup_status: str
    commands: list[str] = field(default_factory=list)
    runtime_id: str | None = None
    repair_hints: list[dict] = field(default_factory=list)
    detail: str | None = None

    def to_summary(self) -> dict:
        return {
            "status": self.status,
            "setup_status": self.setup_status,
            "commands": list(self.commands),
            "runtime_id": self.runtime_id,
            "repair_hints": [dict(hint) for hint in self.repair_hints],
            "detail": self.detail,
        }


class SkillRuntimeManager:
    """Evaluate bundle readiness and expose one command tool per ready skill."""

    def __init__(self, environment_manager: Any | None = None) -> None:
        if environment_manager is None:
            from client_backend.services.skill_runtime.environment import (
                SkillEnvironmentManager,
            )

            environment_manager = SkillEnvironmentManager()
        self._environment_manager = environment_manager

    def evaluate_readiness(self, skill: SkillMetadata) -> SkillReadiness:
        assets = skill.executable_assets
        bundled_commands = self._bundled_commands(assets)
        has_python_project = bool(assets.get("python_project"))

        if bundled_commands:
            return SkillReadiness(
                status="ready",
                setup_status="not_applicable",
                commands=bundled_commands,
            )

        if not has_python_project:
            return SkillReadiness(
                status="instruction_only",
                setup_status="not_applicable",
            )

        inspection = self._environment_manager.inspect(skill)
        setup_status = str(inspection.get("status") or "setup_required")
        commands = sorted(str(command) for command in inspection.get("commands", []))
        runtime_id = inspection.get("runtime_id")

        if setup_status == "ready" and commands:
            return SkillReadiness(
                status="ready",
                setup_status="ready",
                commands=commands,
                runtime_id=str(runtime_id) if runtime_id else None,
            )

        if setup_status == "stale":
            repair_hints = [{"type": "rebuild_skill_runtime", "skill": skill.name}]
        elif setup_status == "failed":
            repair_hints = [{"type": "inspect_setup_failure", "skill": skill.name}]
        else:
            repair_hints = [{"type": "setup_skill", "skill": skill.name}]

        return SkillReadiness(
            status="not_ready",
            setup_status=setup_status,
            runtime_id=str(runtime_id) if runtime_id else None,
            repair_hints=repair_hints,
            detail=inspection.get("detail"),
        )

    def capability_catalog_entries(
        self,
        skill: SkillMetadata,
        readiness: SkillReadiness,
    ) -> list[dict]:
        if readiness.status != "ready":
            return []

        server_name = SKILL_SERVER_NAME_PREFIX + skill.name.replace("-", "_")
        return [
            {
                "name": RUN_SKILL_COMMAND,
                "description": (
                    f"Run a command bundled with the {skill.name} skill. "
                    "Pass an argv array; this is not a general shell."
                ),
                "origin": "skill",
                "server_name": server_name,
                "qualified_id": f"skill::{skill.name}::{RUN_SKILL_COMMAND}",
                "input_schema": COMMAND_INPUT_SCHEMA,
                "readiness": readiness.to_summary(),
                "mutation": True,
                "source_hash": skill.source_hash,
            }
        ]

    @staticmethod
    def _bundled_commands(assets: dict) -> list[str]:
        commands: list[str] = []
        for name in assets.get("bin", []):
            value = str(name)
            if value.lower().endswith(".py"):
                value = value[:-3]
            commands.append(value)
        commands.extend(str(path) for path in assets.get("scripts", []))
        return sorted(dict.fromkeys(commands))
