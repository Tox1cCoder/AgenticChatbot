"""Wire schemas for skill archive uploads and installation operations.

These models are the single boundary between two naming conventions that both
have to hold: the service layer speaks snake_case Python dicts
(``SkillBundleInstaller.preview()`` and ``.install()`` predate this feature and
are consumed by path-based callers), while the browser contract in
``plans/SKILL_INSTALLATION_FE_CONTRACT.md`` is camelCase.

Crossing that boundary needs typed fields, not a passthrough ``dict``. A
Pydantic ``alias_generator`` renames declared model fields; it does not touch the
*contents* of a ``dict[str, Any]`` field, so carrying the installer's payload
through untyped would serialize ``source_hash`` and ``confirmation_required``
straight onto the wire. Every value that reaches a client is therefore declared
here, and :meth:`SkillArchivePreview.from_installer_preview` is the only place
that reads the installer's key names.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

UploadState = Literal["staged", "installing", "installed"]
OperationState = Literal["pending", "running", "succeeded", "failed", "cancelled"]
OperationPhase = Literal[
    "validating",
    "waitingForLock",
    "copying",
    "preparingRuntime",
    "committing",
    "refreshingCatalog",
    "syncingCatalog",
]
CatalogSyncStatus = Literal["synced", "pending", "disconnected"]
InstallAction = Literal["installed", "updated"]
InstallSource = Literal["profile", "configured_root"]


def to_camel(value: str) -> str:
    """Convert a snake_case field name to its camelCase wire alias."""
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class CamelModel(BaseModel):
    """Strict base for every new skill installation schema.

    ``populate_by_name`` keeps the documented snake_case request aliases working
    during the frontend transition, and ``extra="forbid"`` means a client typo or
    an unexpected field is a 422 rather than a silently ignored instruction.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    def to_api(self) -> dict[str, Any]:
        """Serialize for an HTTP response: camelCase keys, JSON-safe values."""
        return self.model_dump(mode="json", by_alias=True)

    def to_record(self) -> dict[str, Any]:
        """Serialize for on-disk persistence: stable snake_case field names."""
        return self.model_dump(mode="json")


class SkillExecutableAssets(CamelModel):
    """Executable content a bundle carries, as discovered by the registry."""

    bin: list[str] = Field(default_factory=list)
    scripts: list[str] = Field(default_factory=list)
    python_project: bool = False


class SkillArchiveSetupPreview(CamelModel):
    """What preparing this bundle's runtime would do, without doing any of it."""

    python_project: bool = False
    dependencies: list[str] = Field(default_factory=list)
    build_requirements: list[str] = Field(default_factory=list)
    declared_commands: list[str] = Field(default_factory=list)
    dependency_lock: str | None = None
    confirmation_required: bool = False


class SkillExistingSkill(CamelModel):
    """The installed skill an upload would collide with, if any.

    ``replaceable`` is what the UI branches on. It is ``False`` for a skill
    discovered from a configured root, which the sidecar does not own and must
    never overwrite -- the user's own directory is not ours to rewrite.
    """

    name: str
    source_hash: str
    install_source: InstallSource
    enabled: bool
    replaceable: bool


class SkillArchivePreview(CamelModel):
    """Everything a client needs to decide whether to install a staged archive."""

    name: str
    source_hash: str
    bundle_shape: Literal["direct", "nested"]
    executable_assets: SkillExecutableAssets
    setup: SkillArchiveSetupPreview
    existing_skill: SkillExistingSkill | None = None

    @classmethod
    def from_installer_preview(
        cls,
        payload: dict[str, Any],
        *,
        existing_skill: SkillExistingSkill | None = None,
    ) -> SkillArchivePreview:
        """Build a preview from ``SkillBundleInstaller.preview()`` output.

        The only reader of the installer's snake_case keys. Keeping that
        knowledge in one classmethod is what stops the two conventions from
        leaking into each other across the upload, operation, and API layers.
        """
        return cls(
            name=str(payload["name"]),
            source_hash=str(payload["source_hash"]),
            bundle_shape=payload.get("bundle_shape") or "direct",
            executable_assets=SkillExecutableAssets(**(payload.get("executable_assets") or {})),
            setup=SkillArchiveSetupPreview(**(payload.get("setup") or {})),
            existing_skill=existing_skill,
        )


class SkillCollectionInfo(CamelModel):
    """Identity of the library an archive contains.

    Present for every upload, including a single skill, so a client renders one
    shape. ``version`` and ``description`` come from a plugin manifest when the
    archive ships one, and are ``None`` for a plain folder.
    """

    name: str
    version: str | None = None
    description: str | None = None
    skill_count: int = 1


class SkillArchiveSummary(CamelModel):
    """Non-sensitive measurements of a staged archive.

    ``filename`` is the sanitized base name of what the user selected, echoed so
    the confirmation UI can name the file. It is never used to build a path.

    ``skipped_link_count`` reports symbolic links dropped during extraction, which
    source downloads of real repositories often carry. It is surfaced rather than
    hidden so a bundle that depended on one is diagnosable.
    """

    filename: str
    compressed_bytes: int
    expanded_bytes: int
    file_count: int
    skipped_link_count: int = 0


class SkillUploadRecord(CamelModel):
    """A staged, validated, not-yet-installed archive.

    Persisted under the owner's profile and reloaded after a restart. ``owner``
    and ``request_fingerprint`` are internal: :meth:`to_api` drops them so a
    response can neither confirm another user's id nor expose the fingerprint
    used for idempotency.
    """

    version: int = 1
    upload_id: str
    owner: str
    state: UploadState = "staged"
    created_at: datetime
    expires_at: datetime
    archive: SkillArchiveSummary
    # The archive's first skill. Retained as the primary preview so a client
    # written against the single-skill contract keeps working; `skills` is the
    # complete list and is what a collection-aware client should render.
    preview: SkillArchivePreview
    collection: SkillCollectionInfo | None = None
    skills: list[SkillArchivePreview] = Field(default_factory=list)
    request_fingerprint: str | None = None
    operation_id: str | None = None

    def to_api(self) -> dict[str, Any]:
        payload = super().to_api()
        for internal in ("owner", "version", "requestFingerprint"):
            payload.pop(internal, None)
        return payload


class SkillInstallationRequest(CamelModel):
    """Client confirmation that starts one installation.

    Every field is a deliberate act by the user: the hash binds the request to
    the preview they saw, ``approve_setup`` authorizes running project-controlled
    build code, and ``replace_source_hash`` authorizes overwriting a specific
    installed bundle. None of them may be inferred server-side.
    """

    expected_source_hash: str
    approve_setup: bool = False
    replace_source_hash: str | None = None


class SkillInstallationFailure(CamelModel):
    """A normalized, client-safe failure for a terminal operation."""

    code: str
    message: str
    retryable: bool = False


class SkillInstallationResult(CamelModel):
    """What a succeeded installation produced, including the fresh catalog."""

    action: InstallAction
    name: str
    source_hash: str
    runtime_status: str
    catalog: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_installer_result(
        cls,
        payload: dict[str, Any],
        *,
        catalog: dict[str, Any],
    ) -> SkillInstallationResult:
        """Build a result from ``SkillBundleInstaller.install()`` output.

        ``install_id`` is deliberately dropped: it names a directory inside the
        profile, which upload and operation responses never expose. The catalog
        is already camelCase because the catalog service projects it that way.
        """
        return cls(
            action=payload.get("action") or "installed",
            name=str(payload["name"]),
            source_hash=str(payload["source_hash"]),
            runtime_status=str(payload.get("runtime_status") or "unknown"),
            catalog=catalog,
        )


class SkillInstallationOperationModel(CamelModel):
    """A persisted installation operation, as stored and as served.

    ``owner``, ``upload_source_hash``, and ``commit_started_at`` drive recovery
    after a crash and are dropped from :meth:`to_api`; a client cannot act on
    them, and ``commit_started_at`` in particular describes an internal boundary.
    """

    version: int = 1
    operation_id: str
    upload_id: str
    owner: str
    state: OperationState = "pending"
    phase: OperationPhase = "validating"
    created_at: datetime
    expires_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    commit_started_at: datetime | None = None
    cancel_requested: bool = False
    upload_source_hash: str | None = None
    request_fingerprint: str | None = None
    result: SkillInstallationResult | None = None
    failure: SkillInstallationFailure | None = None

    def to_api(self) -> dict[str, Any]:
        payload = super().to_api()
        for internal in (
            "owner",
            "version",
            "commitStartedAt",
            "cancelRequested",
            "uploadSourceHash",
            "requestFingerprint",
        ):
            payload.pop(internal, None)
        return payload

    @property
    def is_terminal(self) -> bool:
        return self.state in {"succeeded", "failed", "cancelled"}
