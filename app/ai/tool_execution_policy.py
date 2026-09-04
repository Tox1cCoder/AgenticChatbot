"""Origin-aware tool identity and deployment policy matching.

This module defines the canonical identity every tool is normalized to before
any execution policy (timeouts, retries, metadata trust) is resolved; the pure
matching helpers that turn deployment-configured `ToolExecutionPolicyOverride`
rules (see `app.core.config`) into an ordered, unambiguous list of rules for
one tool identity; and the layered resolver that combines trusted metadata,
those rules, and safety caps into one final `ToolExecutionPolicy`.

The scoped execution context (`tool_policy_context` / `get_current_tool_policy`)
lets a running tool call read the policy that governed its own invocation
without threading it through every call site.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.core.config import (
    InternalToolExecutionPolicy,
    ToolExecutionPolicyMatch,
    ToolExecutionPolicyOverride,
    settings,
)

# Allowed values for ToolIdentity.tool_origin. Kept in sync with the
# Literal declared on ToolExecutionPolicyMatch.tool_origin in app.core.config.
_KNOWN_TOOL_ORIGINS = frozenset({"internal", "server_mcp", "client_mcp", "client_skill"})
_DEFAULT_TOOL_ORIGIN = "internal"

# Specificity levels, least to most specific, per the "Deployment
# Configuration" contract: origin default < origin+exposed name <
# origin+server+source name < origin+qualified id.
_SPECIFICITY_ORIGIN_DEFAULT = 1
_SPECIFICITY_EXPOSED_NAME = 2
_SPECIFICITY_SERVER_SOURCE = 3
_SPECIFICITY_QUALIFIED_ID = 4


class AmbiguousToolExecutionPolicyError(ValueError):
    """Raised when two or more deployment rules match a tool identity at the
    same specificity level.

    Deployment configuration must resolve a tool's policy deterministically;
    it must never depend on dict iteration order.
    """


class ToolExecutionPolicyValidationError(ValueError):
    """Raised when a resolved tool execution policy violates a safety invariant.

    Covers every fail-closed case that is not the dict-ordering ambiguity
    above: an unsafe automatic-retry budget, an outer-timeout disable request
    outside the code-owned allowlist, an accumulated cap that leaves no room
    for the cancellation grace, or a client tool whose soft timeout is too
    short to preserve the strict client-execution < bridge-response < soft
    deadline ordering.
    """


# Code-owned allowlist for `disable_outer_timeout`, deliberately empty.
#
# It held exactly one identity, `internal::dispatch_subagents`, because that
# tool ran a whole fan-out inside a single interactive call and its inner
# worker/provider operations carried the real budgets. In routing-v2 there is
# no such call: `dispatch_subagents` is bound to the Planning model as a schema
# only, fan-out is parent-graph topology, and `TOOL_STAGE_NODES` is empty, so
# nothing routes it through this resolver at all.
#
# The check below stays rather than the entry, so any future attempt to grant
# the exemption — from deployment config or from application metadata — fails
# closed instead of inheriting an allowance nothing reviews any more.
_DISABLE_OUTER_TIMEOUT_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()

# `resolve_tool_execution_policy`'s `invocation_kind` values. A generic
# background execution mode is explicitly out of scope for this plan (see
# "Explicitly Deferred"), so only these three interactive invocation shapes
# are accepted.
_VALID_INVOCATION_KINDS = frozenset({"native_async", "sync_thread", "client_runtime"})

# Origins whose tools run through a client-side bridge and therefore need a
# client-execution and bridge-response deadline in addition to the server
# soft/hard/total timeouts.
_CLIENT_TOOL_ORIGINS = frozenset({"client_mcp", "client_skill"})

# Application-owned fields a trusted `metadata["application_execution_policy"]`
# dict (internal tools only) may set. Mirrors `ToolExecutionPolicyOverride`
# minus `match`/`trust_mcp_metadata`, which are deployment-config-only
# concepts.
_INTERNAL_TRUSTED_POLICY_FIELDS = (
    "timeout_seconds",
    "hard_timeout_seconds",
    "total_timeout_seconds",
    "max_timeout_seconds",
    "max_attempts",
    "retry_safe",
    "idempotent",
    "disable_outer_timeout",
    "timeout_hint",
)

# Remote MCP metadata is diagnostic-only by default. When an exact rule opts
# in with `trust_mcp_metadata=True`, only these four fields normalize — never
# disable_outer_timeout, hard/total timeouts, max_attempts, or cancellation
# (see "Metadata Trust").
_REMOTE_TIMEOUT_HINT_MAX_LENGTH = 240


@dataclass(frozen=True)
class ToolIdentity:
    """Canonical, application-owned identity used for all policy matching.

    The exact key for policy lookups is the tuple (tool_origin,
    qualified_tool_id) — never qualified_tool_id alone, because the same
    qualified id can exist under different origins (for example, a
    server-side and a client-side copy of the same MCP tool).
    """

    tool_origin: str
    qualified_tool_id: str
    exposed_tool_name: str
    source_tool_name: str
    server_name: str | None


def resolve_tool_identity(tool: Any, *, exposed_tool_name: str) -> ToolIdentity:
    """Derive the canonical identity of a bound tool for policy matching.

    Reads only the application-owned metadata keys written by
    `clone_mcp_tool()` (server MCP tools) or `ClientRuntimeToolSpec` (client
    tools). A tool that declares none of these keys — the common case for
    first-party, internal tools — is treated as `internal` and falls back to
    `internal::{source_tool_name}` for its qualified id, since internal tools
    never declare an application-owned qualified id of their own.

    Aliasing (binding the same underlying tool under a different name) only
    changes `exposed_tool_name`; `source_tool_name` and the qualified id are
    read from metadata (or the tool's own stable name) and stay unaffected.
    """

    raw_metadata = getattr(tool, "metadata", None)
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

    tool_origin = str(metadata.get("tool_origin") or "").strip() or _DEFAULT_TOOL_ORIGIN
    if tool_origin not in _KNOWN_TOOL_ORIGINS:
        raise ToolExecutionPolicyValidationError(
            f"Unknown tool_origin {tool_origin!r}; expected one of {sorted(_KNOWN_TOOL_ORIGINS)}"
        )

    exposed_name = str(exposed_tool_name or "").strip()
    fallback_source_name = str(getattr(tool, "name", "") or exposed_name).strip()
    source_tool_name = str(metadata.get("source_tool_name") or fallback_source_name).strip()

    qualified_tool_id = str(metadata.get("qualified_tool_id") or "").strip()
    if not qualified_tool_id:
        qualified_tool_id = f"internal::{source_tool_name}"

    raw_server_name = metadata.get("server_name")
    server_name = str(raw_server_name).strip() or None if raw_server_name else None

    return ToolIdentity(
        tool_origin=tool_origin,
        qualified_tool_id=qualified_tool_id,
        exposed_tool_name=exposed_name,
        source_tool_name=source_tool_name,
        server_name=server_name,
    )


def policy_match_specificity(match: ToolExecutionPolicyMatch) -> int:
    """Classify a validated policy match into one of four specificity levels.

    Relies on `ToolExecutionPolicyMatch`'s own validation to guarantee exactly
    one of the four shapes is present, so this only classifies — it does not
    re-check shape validity.
    """

    if match.qualified_tool_id is not None:
        return _SPECIFICITY_QUALIFIED_ID
    if match.server_name is not None and match.source_tool_name is not None:
        return _SPECIFICITY_SERVER_SOURCE
    if match.exposed_tool_name is not None:
        return _SPECIFICITY_EXPOSED_NAME
    return _SPECIFICITY_ORIGIN_DEFAULT


def _rule_matches_identity(
    match: ToolExecutionPolicyMatch,
    identity: ToolIdentity,
    specificity: int,
) -> bool:
    if match.tool_origin != identity.tool_origin:
        return False
    if specificity == _SPECIFICITY_EXPOSED_NAME:
        return match.exposed_tool_name == identity.exposed_tool_name
    if specificity == _SPECIFICITY_SERVER_SOURCE:
        return (
            match.server_name == identity.server_name
            and match.source_tool_name == identity.source_tool_name
        )
    if specificity == _SPECIFICITY_QUALIFIED_ID:
        return match.qualified_tool_id == identity.qualified_tool_id
    return True


def matching_policy_rules(
    identity: ToolIdentity,
    rules: dict[str, ToolExecutionPolicyOverride],
) -> list[tuple[str, ToolExecutionPolicyOverride]]:
    """Return every deployment rule matching one tool identity, least to most specific.

    Applies all matching rules — a broad origin-level cap and a narrow exact
    override can both apply to the same tool at once, and it is the caller's
    job (the policy resolver) to combine them. What this function guarantees
    is that at most one rule matches at each specificity level: two or more
    matching rules at the same level raise `AmbiguousToolExecutionPolicyError`
    naming every conflicting diagnostic key, since deployment config must not
    leave a tool's resolved policy dependent on dict ordering.
    """

    buckets: dict[int, list[tuple[str, ToolExecutionPolicyOverride]]] = {}

    for config_key, override in rules.items():
        specificity = policy_match_specificity(override.match)
        if not _rule_matches_identity(override.match, identity, specificity):
            continue
        buckets.setdefault(specificity, []).append((config_key, override))

    for specificity, matches in buckets.items():
        if len(matches) > 1:
            conflicting_keys = ", ".join(sorted(key for key, _ in matches))
            raise AmbiguousToolExecutionPolicyError(
                "Ambiguous tool execution policy rules for identity "
                f"(tool_origin={identity.tool_origin!r}, "
                f"qualified_tool_id={identity.qualified_tool_id!r}) "
                f"at specificity {specificity}: {conflicting_keys}"
            )

    ordered_matches: list[tuple[str, ToolExecutionPolicyOverride]] = []
    for specificity in sorted(buckets):
        ordered_matches.extend(buckets[specificity])
    return ordered_matches


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """The fully resolved execution policy for one tool call.

    Produced by `resolve_tool_execution_policy` from layered defaults,
    deployment configuration, and (narrowly) trusted metadata. Every timeout
    field is a concrete number — `outer_timeout_disabled` is the flag a
    runner checks before deciding whether to enforce them at all.
    """

    identity: ToolIdentity
    timeout_seconds: float
    hard_timeout_seconds: float
    total_timeout_seconds: float
    max_attempts: int
    retry_safe: bool
    idempotent: bool
    metadata_trusted: bool
    outer_timeout_disabled: bool
    cancellation: str
    client_execution_timeout_seconds: float | None
    client_response_timeout_seconds: float | None
    policy_source: str
    policy_config_keys: tuple[str, ...]
    timeout_hint: str


def _cancellation_for_invocation_kind(invocation_kind: str) -> str:
    return "cooperative" if invocation_kind == "native_async" else "abandon_only"


def _trusted_internal_policy_fields(raw_metadata: dict[str, Any]) -> dict[str, Any]:
    """Read the trusted `application_execution_policy` bag for an internal tool.

    Trusted because this namespace is only ever written by this application's
    own tool-construction code (see "Metadata Trust") — never by a remote MCP
    server or client sidecar. Unknown keys are ignored rather than rejected;
    this is a first-party, code-owned contract, not user input.
    """

    app_policy = raw_metadata.get("application_execution_policy")
    if not isinstance(app_policy, dict):
        return {}
    try:
        validated = InternalToolExecutionPolicy.model_validate(app_policy)
    except ValidationError as exc:
        raise ToolExecutionPolicyValidationError(
            f"Invalid application_execution_policy metadata: {exc}"
        ) from exc
    return {
        key: value
        for key, value in validated.model_dump(exclude_none=True).items()
        if key in _INTERNAL_TRUSTED_POLICY_FIELDS
    }


def _remote_mcp_trusted_fields(raw_metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize only the four allowlisted remote MCP fields, nothing else.

    Reads exactly: the `idempotentHint` annotation and, from `_meta`,
    `execution_timeout_seconds`, `retry_safe`, and `timeout_hint` (trimmed to
    240 characters). Every other remote field — including any request to
    disable the outer timeout, change hard/total timeouts, raise
    `max_attempts`, or alter cancellation — is never read here, so it can
    never reach the resolved policy regardless of trust.
    """

    fields: dict[str, Any] = {}

    idempotent_hint = raw_metadata.get("idempotentHint")
    if isinstance(idempotent_hint, bool):
        fields["idempotent"] = idempotent_hint

    remote_meta = raw_metadata.get("_meta")
    if isinstance(remote_meta, dict):
        timeout_value = remote_meta.get("execution_timeout_seconds")
        if (
            isinstance(timeout_value, (int, float))
            and not isinstance(timeout_value, bool)
            and timeout_value > 0
        ):
            fields["timeout_seconds"] = float(timeout_value)

        retry_safe_value = remote_meta.get("retry_safe")
        if isinstance(retry_safe_value, bool):
            fields["retry_safe"] = retry_safe_value

        timeout_hint_value = remote_meta.get("timeout_hint")
        if isinstance(timeout_hint_value, str) and timeout_hint_value:
            fields["timeout_hint"] = timeout_hint_value[:_REMOTE_TIMEOUT_HINT_MAX_LENGTH]

    return fields


def resolve_tool_execution_policy(
    tool: Any,
    *,
    exposed_tool_name: str,
    invocation_kind: str,
) -> ToolExecutionPolicy:
    """Resolve the final execution policy for one bound tool call.

    Layering order: (1) an internal tool's own trusted
    `application_execution_policy` metadata; (2) deployment-configured rules,
    applied least to most specific, each overriding only the fields it sets —
    except `max_timeout_seconds`, which never gets overwritten: every matching
    rule's value (a broad origin-level cap alongside a narrow exact override)
    accumulates as a running minimum, so a broad operational cap can never be
    bypassed by a more-specific rule that sets a larger one; (3) — only when
    the single matching exact rule set `trust_mcp_metadata=True` — the four
    allowlisted remote MCP fields, filling in only fields that exact rule left
    unset. The accumulated `max_timeout_seconds` minimum (plus the global
    interactive cap) is then enforced as a wall-clock ceiling, followed by
    invariant validation (`soft <= hard <= total <= cap`, with
    `hard - soft >= cancellation grace`, shortening soft where needed) and the
    two fail-closed safety checks: `disable_outer_timeout` against the
    code-owned allowlist, and `max_attempts > 1` against
    `retry_safe`/`idempotent`.

    The soft/hard invariant is strict (`soft < hard`) whenever the configured
    cancellation grace is greater than zero. `soft == hard` is permitted only
    in the degenerate `tool_execution_cancellation_grace_seconds == 0`
    configuration (legal because that field is `ge=0`), where the soft-cancel
    and hard-abandon phases collapse into the same instant by construction —
    there is no grace window left to reserve.

    Raises `AmbiguousToolExecutionPolicyError` if deployment config matches
    the same identity twice at one specificity level, or
    `ToolExecutionPolicyValidationError` if the resolved policy would violate
    a safety invariant.
    """

    if invocation_kind not in _VALID_INVOCATION_KINDS:
        raise ValueError(
            f"Unknown tool invocation_kind {invocation_kind!r}; expected one of "
            f"{sorted(_VALID_INVOCATION_KINDS)}"
        )

    identity = resolve_tool_identity(tool, exposed_tool_name=exposed_tool_name)

    raw_metadata_attr = getattr(tool, "metadata", None)
    raw_metadata: dict[str, Any] = raw_metadata_attr if isinstance(raw_metadata_attr, dict) else {}

    effective: dict[str, Any] = {
        "timeout_seconds": None,
        "hard_timeout_seconds": None,
        "total_timeout_seconds": None,
        "max_timeout_seconds": None,
        "max_attempts": 1,
        "retry_safe": False,
        "idempotent": False,
        "disable_outer_timeout": False,
        "timeout_hint": "",
    }

    internal_trusted = False
    if identity.tool_origin == "internal":
        internal_fields = _trusted_internal_policy_fields(raw_metadata)
        if internal_fields:
            internal_trusted = True
            effective.update(internal_fields)

    rules = matching_policy_rules(identity, settings.tool_execution_policies)

    exact_rule: ToolExecutionPolicyOverride | None = None
    for _config_key, override in rules:
        if override.timeout_seconds is not None:
            effective["timeout_seconds"] = override.timeout_seconds
        if override.hard_timeout_seconds is not None:
            effective["hard_timeout_seconds"] = override.hard_timeout_seconds
        if override.total_timeout_seconds is not None:
            effective["total_timeout_seconds"] = override.total_timeout_seconds
        if override.max_timeout_seconds is not None:
            current_cap = effective["max_timeout_seconds"]
            effective["max_timeout_seconds"] = (
                override.max_timeout_seconds
                if current_cap is None
                else min(current_cap, override.max_timeout_seconds)
            )
        if override.max_attempts is not None:
            effective["max_attempts"] = override.max_attempts
        if override.retry_safe is not None:
            effective["retry_safe"] = override.retry_safe
        if override.idempotent is not None:
            effective["idempotent"] = override.idempotent
        if override.disable_outer_timeout:
            effective["disable_outer_timeout"] = True
        if override.timeout_hint is not None:
            effective["timeout_hint"] = override.timeout_hint
        if policy_match_specificity(override.match) == _SPECIFICITY_QUALIFIED_ID:
            exact_rule = override

    remote_trusted = False
    if exact_rule is not None and exact_rule.trust_mcp_metadata:
        remote_fields = _remote_mcp_trusted_fields(raw_metadata)
        if remote_fields:
            remote_trusted = True
            if "timeout_seconds" in remote_fields and exact_rule.timeout_seconds is None:
                effective["timeout_seconds"] = remote_fields["timeout_seconds"]
            if "retry_safe" in remote_fields and exact_rule.retry_safe is None:
                effective["retry_safe"] = remote_fields["retry_safe"]
            if "idempotent" in remote_fields and exact_rule.idempotent is None:
                effective["idempotent"] = remote_fields["idempotent"]
            if "timeout_hint" in remote_fields and exact_rule.timeout_hint is None:
                effective["timeout_hint"] = remote_fields["timeout_hint"]

    metadata_trusted = internal_trusted or remote_trusted
    config_keys = tuple(config_key for config_key, _ in rules)

    identity_key = (identity.tool_origin, identity.qualified_tool_id)
    if effective["disable_outer_timeout"]:
        if not internal_trusted:
            raise ToolExecutionPolicyValidationError(
                "disable_outer_timeout requires trusted application metadata"
            )
        if identity_key not in _DISABLE_OUTER_TIMEOUT_ALLOWLIST:
            permitted = sorted(_DISABLE_OUTER_TIMEOUT_ALLOWLIST)
            raise ToolExecutionPolicyValidationError(
                f"disable_outer_timeout is permitted for {permitted or 'no identity'}; "
                f"got {identity_key!r}"
            )

    if effective["max_attempts"] > 1 and not (effective["retry_safe"] or effective["idempotent"]):
        raise ToolExecutionPolicyValidationError(
            f"max_attempts={effective['max_attempts']} for identity {identity_key!r} "
            "requires retry_safe or idempotent to be true after trust and override "
            "resolution"
        )

    default_timeout = float(getattr(settings, "tool_execution_timeout", 30) or 30)
    grace = float(settings.tool_execution_cancellation_grace_seconds)
    global_cap = float(settings.tool_execution_max_interactive_timeout_seconds)

    configured_soft = effective["timeout_seconds"]
    soft = float(configured_soft if configured_soft is not None else default_timeout)

    configured_hard = effective["hard_timeout_seconds"]
    hard = float(configured_hard if configured_hard is not None else soft + grace)

    configured_total = effective["total_timeout_seconds"]
    total = float(configured_total if configured_total is not None else hard)

    cap = global_cap
    if effective["max_timeout_seconds"] is not None:
        cap = min(cap, float(effective["max_timeout_seconds"]))

    if cap <= grace:
        raise ToolExecutionPolicyValidationError(
            f"accumulated max timeout cap {cap} for identity {identity_key!r} does not "
            f"leave room for the configured cancellation grace {grace}"
        )

    total = min(total, cap)
    hard = min(hard, total)
    if hard <= grace:
        raise ToolExecutionPolicyValidationError(
            f"resolved hard timeout {hard} for identity {identity_key!r} does not leave "
            f"room for the configured cancellation grace {grace}"
        )

    # Invariant enforced from here on: `hard - soft >= grace` (equivalently
    # `soft <= hard`). When `grace > 0` this forces strict `soft < hard`. When
    # `grace == 0` (legal — the field is `ge=0`) the floor collapses to
    # `soft == hard`: the soft-cancel and hard-abandon phases coincide by
    # construction because there is no grace window left to reserve. Deployment
    # config may legitimately land in that degenerate configuration; it must
    # not raise here.
    if soft >= hard or (hard - soft) < grace:
        soft = hard - grace

    client_execution_timeout_seconds: float | None = None
    client_response_timeout_seconds: float | None = None
    if identity.tool_origin in _CLIENT_TOOL_ORIGINS:
        execution_grace = float(settings.tool_execution_client_execution_grace_seconds)
        response_grace = float(settings.tool_execution_client_response_grace_seconds)
        client_execution_timeout_seconds = soft - execution_grace
        client_response_timeout_seconds = soft - response_grace
        if not (0 < client_execution_timeout_seconds < client_response_timeout_seconds < soft):
            raise ToolExecutionPolicyValidationError(
                f"resolved soft timeout {soft} for identity {identity_key!r} is too short "
                "to preserve strict client deadline ordering (client execution < bridge "
                "response < server soft)"
            )

    if config_keys and metadata_trusted:
        policy_source = "config+metadata"
    elif config_keys:
        policy_source = "config"
    elif metadata_trusted:
        policy_source = "metadata"
    else:
        policy_source = "default"

    return ToolExecutionPolicy(
        identity=identity,
        timeout_seconds=soft,
        hard_timeout_seconds=hard,
        total_timeout_seconds=total,
        max_attempts=int(effective["max_attempts"]),
        retry_safe=bool(effective["retry_safe"]),
        idempotent=bool(effective["idempotent"]),
        metadata_trusted=metadata_trusted,
        outer_timeout_disabled=bool(effective["disable_outer_timeout"]),
        cancellation=_cancellation_for_invocation_kind(invocation_kind),
        client_execution_timeout_seconds=client_execution_timeout_seconds,
        client_response_timeout_seconds=client_response_timeout_seconds,
        policy_source=policy_source,
        policy_config_keys=config_keys,
        timeout_hint=str(effective["timeout_hint"] or ""),
    )


_current_tool_policy: ContextVar[ToolExecutionPolicy | None] = ContextVar(
    "_current_tool_policy", default=None
)


@contextmanager
def tool_policy_context(policy: ToolExecutionPolicy) -> Iterator[ToolExecutionPolicy]:
    """Scope `get_current_tool_policy()` to `policy` for the duration of the block.

    Uses a `ContextVar` token so nested calls (a tool that itself invokes
    other tools) and concurrent `asyncio` tasks each see their own policy and
    restore the previous one on exit — never a bare `None` reset that would
    clobber an outer scope's policy.
    """

    token = _current_tool_policy.set(policy)
    try:
        yield policy
    finally:
        _current_tool_policy.reset(token)


def get_current_tool_policy() -> ToolExecutionPolicy | None:
    """Return the policy governing the tool call currently executing, if any."""

    return _current_tool_policy.get()
