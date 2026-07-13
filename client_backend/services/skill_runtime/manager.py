"""Readiness evaluation for executable skills.

Decides, before the model ever tries to call a skill capability, whether the
skill's runtime is actually usable: are its Python dependencies importable,
is its binary on ``PATH``, are its required secrets configured. This module
only *detects and reports* — it never installs a dependency, resolves a
permission, or executes anything. Those live in later slices of this
refactor (Task 4 installer, Task 5 capability catalog, Task 6 permission
enforcement, Task 7 execution engine).
"""

from __future__ import annotations

import importlib.metadata
import shutil
from dataclasses import dataclass, field

from packaging.requirements import InvalidRequirement, Requirement

from client_backend.services.skill_runtime.secrets import SkillSecretStore
from shared.skills.manifest import SUPPORTED_RUNTIME_TYPES, SkillManifest, SkillRuntimeSpec

# Repair-hint "type" values this module can emit. "permission_required" is
# reserved for Task 6 (permission evaluation/enforcement) — the shape is
# defined here so downstream consumers have a stable contract to rely on,
# but this module never evaluates permissions and never emits that hint.
REPAIR_HINT_TYPES = frozenset(
    {
        "unsupported_runtime",
        "invalid_manifest",
        "install_command",
        "install_dependency",
        "configure_secret",
        "permission_required",
    }
)

# Prefix for the synthetic "server_name" a skill's capabilities are grouped
# under in the sidecar tool catalog (e.g. "example-calendar" ->
# "skill_example_calendar"). Mirrors the MCP catalog shape so downstream
# consumers (runtime_bridge, client_runtime_tools, client_tool_catalog) treat
# skill capabilities like any other client-tool "server".
SKILL_SERVER_NAME_PREFIX = "skill_"

def _parse_distribution_name(requirement: str) -> str | None:
    """Extract the distribution name from a PEP 508 requirement string.

    Uses :class:`packaging.requirements.Requirement`, so extras, version
    specifiers, environment markers, and surrounding whitespace are handled
    correctly (e.g. ``" some-pkg[extra]>=1; python_version>='3.10'"`` ->
    ``"some-pkg"``). Only used to presence-check a dependency, not to match a
    version constraint. Returns None for a genuinely unparseable string so
    the caller skips it rather than false-failing.
    """
    try:
        return Requirement(requirement).name
    except InvalidRequirement:
        return None


@dataclass
class SkillReadiness:
    """The result of evaluating whether a skill is ready to execute.

    ``status`` is one of ``"instruction_only"`` (markdown-only skill, nothing
    to execute), ``"ready"``, ``"not_ready"``, or ``"invalid"`` (manifest
    failed to parse).
    """

    status: str
    missing_dependencies: list[str] = field(default_factory=list)
    missing_secrets: list[str] = field(default_factory=list)
    missing_executable: str | None = None
    unsupported_runtime: str | None = None
    repair_hints: list[dict] = field(default_factory=list)
    detail: str | None = None

    def to_summary(self) -> dict:
        """JSON-safe summary for surfacing to clients/logs.

        Deliberately excludes capability_count/permissions — those come
        from the manifest itself at catalog-build time (Task 5), not from
        readiness evaluation.
        """
        return {
            "status": self.status,
            "missing_dependencies": list(self.missing_dependencies),
            "missing_secrets": list(self.missing_secrets),
            "missing_executable": self.missing_executable,
            "unsupported_runtime": self.unsupported_runtime,
            "repair_hints": [dict(hint) for hint in self.repair_hints],
            "detail": self.detail,
        }


class SkillRuntimeManager:
    """Evaluates skill readiness ahead of any execution attempt."""

    def __init__(self, secret_store: SkillSecretStore | None = None) -> None:
        self._secret_store = secret_store if secret_store is not None else SkillSecretStore()

    def evaluate_readiness(
        self,
        manifest: SkillManifest | None,
        manifest_error: str | None = None,
    ) -> SkillReadiness:
        """Evaluate whether ``manifest`` describes a currently-executable skill.

        A manifest that failed to parse (``manifest_error`` set) is
        ``"invalid"``. A skill with no manifest at all and no parse error is
        ``"instruction_only"`` — a markdown-only skill with nothing to
        execute. Otherwise, dependencies, the runtime executable, and
        required secrets are checked and aggregated into ``"ready"`` or
        ``"not_ready"``.
        """
        if manifest is None:
            if manifest_error is not None:
                return SkillReadiness(status="invalid", detail=manifest_error)
            return SkillReadiness(status="instruction_only")

        repair_hints: list[dict] = []

        unsupported_runtime, missing_executable, binary_missing_command, runtime_detail = (
            self._evaluate_runtime(manifest.runtime, repair_hints)
        )
        missing_dependencies = self._missing_python_dependencies(
            manifest.dependencies.python, repair_hints
        )
        missing_secrets = self._missing_secrets(manifest, repair_hints)

        not_ready = bool(
            missing_dependencies
            or missing_secrets
            or missing_executable
            or unsupported_runtime
            or binary_missing_command
        )

        return SkillReadiness(
            status="not_ready" if not_ready else "ready",
            missing_dependencies=missing_dependencies,
            missing_secrets=missing_secrets,
            missing_executable=missing_executable,
            unsupported_runtime=unsupported_runtime,
            repair_hints=repair_hints,
            detail=runtime_detail,
        )

    @staticmethod
    def _evaluate_runtime(
        runtime: SkillRuntimeSpec, repair_hints: list[dict]
    ) -> tuple[str | None, str | None, bool, str | None]:
        """Check the runtime type and, for ``binary``, the executable.

        Returns ``(unsupported_runtime, missing_executable,
        binary_missing_command, detail)``.
        """
        if runtime.type not in SUPPORTED_RUNTIME_TYPES:
            # Defensive only: a manifest that parsed via load_manifest() can
            # never reach this branch, since SkillRuntimeSpec already
            # rejects unknown/reserved types at validation time.
            repair_hints.append({"type": "unsupported_runtime", "runtime_type": runtime.type})
            return runtime.type, None, False, None

        if runtime.type != "binary":
            return None, None, False, None

        command = runtime.command
        if not command:
            repair_hints.append(
                {"type": "invalid_manifest", "reason": "binary runtime missing command"}
            )
            return None, None, True, "binary runtime requires a 'command'"

        if shutil.which(command) is None:
            repair_hints.append({"type": "install_command", "command": command})
            return None, command, False, None

        return None, None, False, None

    @staticmethod
    def _missing_python_dependencies(
        requirements: list[str], repair_hints: list[dict]
    ) -> list[str]:
        """Presence-check each Python requirement via installed distribution metadata.

        This checks PRESENCE only, not whether the installed version
        satisfies the requirement's version specifier — that avoids false
        negatives from a slightly-mismatched-but-workable version. An
        unparseable requirement string is skipped rather than reported
        missing, for the same reason.
        """
        missing: list[str] = []
        for requirement in requirements:
            name = _parse_distribution_name(requirement)
            if name is None:
                continue
            try:
                importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(requirement)
                repair_hints.append(
                    {
                        "type": "install_dependency",
                        "ecosystem": "python",
                        "dependency": requirement,
                    }
                )
        return missing

    def capability_catalog_entries(
        self,
        skill_name: str,
        manifest: SkillManifest,
        readiness: SkillReadiness,
    ) -> list[dict]:
        """Build sidecar tool-catalog entries for a skill's capabilities.

        Only a ``"ready"`` skill exposes anything — an instruction_only,
        not_ready, or invalid skill contributes no entries, since the model
        must never be offered a capability it cannot actually execute. Pure
        and side-effect-free: no IO, no execution, no permission enforcement
        (those live in Tasks 6/7).
        """
        if readiness.status != "ready":
            return []

        server_name = SKILL_SERVER_NAME_PREFIX + skill_name.replace("-", "_")
        return [
            {
                "name": capability.name,
                "description": capability.description,
                "origin": "skill",
                "server_name": server_name,
                "qualified_id": f"skill::{skill_name}::{capability.name}",
                "input_schema": capability.input_schema,
                "readiness": {"status": readiness.status},
                "mutation": capability.is_mutation(),
            }
            for capability in manifest.capabilities
        ]

    def _missing_secrets(self, manifest: SkillManifest, repair_hints: list[dict]) -> list[str]:
        """Check required secrets: manifest-level required secrets plus any
        secret named by a capability, against the injected secret store."""
        required_names = {secret.name for secret in manifest.secrets if secret.required}
        required_names.update(
            name for capability in manifest.capabilities for name in capability.secrets
        )

        missing: list[str] = []
        for name in sorted(required_names):
            if not self._secret_store.has(name):
                missing.append(name)
                repair_hints.append({"type": "configure_secret", "secret": name})
        return missing
