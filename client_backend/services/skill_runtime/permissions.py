"""Static permission evaluation before skill capability execution.

Decides, before any child process or runtime entrypoint is invoked, whether
the current device permission policy allows a capability to run: are its
declared permission tokens (network/filesystem/process/domain families)
granted, and -- if the capability mutates state -- is mutation allowed
outright or does it need a later human approval (Task 9's HITL flow). This
module is pure evaluation: no filesystem access, no network calls, no
subprocess, no secret resolution. Task 7 (the execution engine) is the only
caller that actually invokes a runtime, and it must consult
:class:`SkillPermissionEvaluator` first and refuse to execute on any
non-``allowed`` decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.skills.errors import PERMISSION_DENIED, PERMISSION_REQUIRED, error_payload
from shared.skills.manifest import SkillCapabilitySpec, SkillRuntimeSpec

# The one reserved runtime type this evaluator hard-blocks: `shell`. The other
# reserved types (node_package, mcp_server) can never reach here because
# SkillRuntimeSpec's validator already rejects them at manifest-load time; only
# `shell` is singled out because it is the explicit escape hatch the plan
# forbids until permission + HITL enforcement exist.
_RESERVED_SHELL_RUNTIME_TYPE = "shell"

# The blanket grant that covers every permission token.
_WILDCARD = "*"

# The structural "this capability mutates state" marker. It is handled via
# SkillCapabilitySpec.mutation / SkillPermissionPolicy.allow_mutation rather
# than as an ordinary permission token, but a manifest may also declare it
# literally inside `capability.permissions` -- if so it is treated the same
# way as the `mutation` flag, never checked as an arbitrary token.
_MUTATION_TOKEN = "mutation"

# Permission families that support a "<family>*" wildcard grant in addition
# to an exact-token or blanket "*" grant. Any token outside these families
# (process:spawn, domain labels like calendar:read/desktop:automation) only
# matches exactly or via the blanket wildcard.
_PREFIXED_WILDCARD_FAMILIES = ("network:", "filesystem:read:", "filesystem:write:")

# Placeholder written in place of every argument value when redacting. Never
# a value that could itself leak information about the original argument.
_REDACTION_PLACEHOLDER = "<redacted>"


def redact_arguments(arguments: dict | None) -> dict:
    """Replace every value in ``arguments`` with a redaction placeholder.

    Keeps the same keys so a repair hint can still name which arguments were
    involved, while guaranteeing no argument value -- which may carry a
    secret or other sensitive user data -- ever appears in a permission
    error, an approval prompt, or a log line built from this dict.
    """
    if not arguments:
        return {}
    return {key: _REDACTION_PLACEHOLDER for key in arguments}


@dataclass(frozen=True)
class SkillPermissionPolicy:
    """What the current device permission policy grants.

    ``granted`` holds permission tokens exactly as a capability would declare
    them (e.g. ``"network:api.example.com"``, ``"filesystem:read:/data"``),
    plus optionally their wildcard forms (``"network:*"``,
    ``"filesystem:read:*"``) or the blanket ``"*"``. ``allow_mutation`` is
    independent of the token list: it governs whether a ``mutation: true``
    capability may run without a later HITL approval (Task 9), since
    mutation is a structural property of the capability rather than a named
    resource.
    """

    granted: frozenset[str] = frozenset()
    allow_mutation: bool = False

    @classmethod
    def deny_all(cls) -> SkillPermissionPolicy:
        """The safe default: nothing granted, mutation always needs approval."""
        return cls()

    @classmethod
    def allow_all(cls) -> SkillPermissionPolicy:
        """Grants every permission token and allows mutation outright."""
        return cls(granted=frozenset({_WILDCARD}), allow_mutation=True)

    def grants(self, token: str) -> bool:
        """Return True iff ``token`` is covered by this policy's grants."""
        if _WILDCARD in self.granted or token in self.granted:
            return True
        return self._covered_by_family_wildcard(token)

    def _covered_by_family_wildcard(self, token: str) -> bool:
        for family in _PREFIXED_WILDCARD_FAMILIES:
            if not token.startswith(family):
                continue
            if f"{family}{_WILDCARD}" in self.granted:
                return True
            if family.startswith("filesystem:"):
                return self._covered_by_path_prefix(family, token)
            return False
        return False

    def _covered_by_path_prefix(self, family: str, token: str) -> bool:
        """Check filesystem path-prefix grants within one read/write family.

        A granted ``"<family><prefix>"`` covers ``"<family><path>"`` iff
        ``path`` equals ``prefix`` or is nested under it with a ``/``
        boundary -- a granted ``filesystem:read:/data`` covers ``/data`` and
        ``/data/sub`` but NOT ``/database``.

        An empty or root-only prefix (``"filesystem:read:"`` or
        ``"filesystem:read:/"``) is treated as granting NOTHING here, so a
        blank/mis-serialized path can never silently open the whole
        filesystem; use the explicit ``"filesystem:read:*"`` wildcard to
        grant everything.
        """
        path = token[len(family) :].rstrip("/")
        for granted_token in self.granted:
            if not granted_token.startswith(family):
                continue
            prefix = granted_token[len(family) :].rstrip("/")
            if not prefix:
                continue
            if path == prefix or path.startswith(prefix + "/"):
                return True
        return False


@dataclass
class PermissionDecision:
    """The outcome of evaluating one capability against one policy.

    ``code`` is ``PERMISSION_DENIED`` for a hard, ungrantable block (the
    reserved ``shell`` runtime), ``PERMISSION_REQUIRED`` for a soft block a
    permission grant or HITL approval could unblock, or ``None`` when
    ``allowed`` is True.
    """

    allowed: bool
    code: str | None
    message: str | None
    missing: list[str] = field(default_factory=list)
    requires_approval: bool = False

    def to_error_payload(self, arguments: dict | None = None) -> dict:
        """Build the plan's failure-envelope ``error`` sub-object for this decision.

        Never includes a raw argument value -- only the (redacted) argument
        keys -- since a blocked capability's error may be surfaced to a
        human approver or written to a log, and arguments can carry secrets
        or other sensitive user data.
        """
        repair = {
            "missing": list(self.missing),
            "requires_approval": self.requires_approval,
            "arguments_redacted": redact_arguments(arguments),
        }
        return error_payload(self.code, self.message, repair=repair)


class SkillPermissionEvaluator:
    """Evaluates a capability against a device permission policy before execution."""

    def __init__(self, policy: SkillPermissionPolicy | None = None) -> None:
        self._policy = policy if policy is not None else SkillPermissionPolicy.deny_all()

    def evaluate(
        self, *, capability: SkillCapabilitySpec, runtime: SkillRuntimeSpec
    ) -> PermissionDecision:
        """Decide whether ``capability`` may run under ``runtime`` right now.

        A reserved ``shell`` runtime is a hard, ungrantable block regardless
        of permissions (``PERMISSION_DENIED``). Otherwise every declared
        permission token must be granted, and a mutating capability must
        either have mutation allowed outright by the policy or be flagged
        as needing HITL approval (``PERMISSION_REQUIRED`` either way).
        """
        if runtime.type == _RESERVED_SHELL_RUNTIME_TYPE:
            return PermissionDecision(
                allowed=False,
                code=PERMISSION_DENIED,
                message="shell runtime is reserved and cannot be executed",
                requires_approval=False,
            )

        missing = [
            token
            for token in capability.permissions
            if token != _MUTATION_TOKEN and not self._policy.grants(token)
        ]

        mutation_declared = capability.mutation or _MUTATION_TOKEN in capability.permissions
        # Mutation is a structural property, gated SOLELY by allow_mutation --
        # never satisfiable through the resource-grant token set. Otherwise a
        # blanket "*" resource grant (or a literal "mutation" token) would
        # silently unlock state changes without the explicit mutation opt-in
        # (or Task 9's HITL approval).
        mutation_needs_approval = mutation_declared and not self._policy.allow_mutation
        if mutation_needs_approval:
            missing.append(_MUTATION_TOKEN)

        if missing:
            return PermissionDecision(
                allowed=False,
                code=PERMISSION_REQUIRED,
                message=self._missing_permission_message(missing, mutation_needs_approval),
                missing=missing,
                requires_approval=mutation_needs_approval,
            )

        return PermissionDecision(allowed=True, code=None, message=None)

    @staticmethod
    def _missing_permission_message(missing: list[str], requires_approval: bool) -> str:
        tokens = ", ".join(missing)
        if requires_approval:
            return f"capability needs a granted permission or approval: {tokens}"
        return f"capability requires permissions that are not granted: {tokens}"
