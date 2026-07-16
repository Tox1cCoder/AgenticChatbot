"""Origin-aware tool identity and deployment policy matching.

This module defines the canonical identity every tool is normalized to before
any execution policy (timeouts, retries, metadata trust) is resolved, plus the
pure matching helpers that turn deployment-configured
`ToolExecutionPolicyOverride` rules (see `app.core.config`) into an ordered,
unambiguous list of rules for one tool identity.

Layered resolution of the final `ToolExecutionPolicy` — metadata trust
normalization, accumulated caps, and the scoped execution context — is a
later addition. This module covers only identity extraction and
configuration matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import ToolExecutionPolicyMatch, ToolExecutionPolicyOverride

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
