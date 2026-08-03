"""An uploaded skill reaches chat, and only on the device that installed it.

The unit suites verify each stage in isolation against doubles. This one runs the
real archive validator, upload store, installer, and catalog projection over a
tracked fixture bundle, then feeds the resulting catalog into the canonical
resolver -- the two halves that ordinarily meet only across a network.

Device scoping is the reason it exists: a skill installs into one machine's
profile, and the canonical server must offer it to that device's chat turns and
to nothing else.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.skill_resolver import list_resolved_skills
from client_backend.schemas.skill_installation import SkillInstallationRequest
from client_backend.services.local_skills_registry import LocalSkillsRegistry
from client_backend.services.skill_catalog import SkillCatalogService
from client_backend.services.skill_runtime.operations import SkillInstallationService
from client_backend.services.skill_runtime.uploads import SkillUploadService

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "skills" / "google_calendar"
# The bundle's front matter, not its folder or archive name, decides this.
FIXTURE_SKILL_NAME = "cli-anything-google-calendar"
USER_ID = "user-a"


def _fixture_zip(bundle: Path) -> bytes:
    """Zip a tracked bundle the way a user would, minus build leftovers.

    ``__pycache__`` and ``.pyc`` are skipped because the installer's copy ignores
    them and the bundle hash excludes them; including them would put bytes in the
    archive that never reach the installed copy.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob("*")):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            if "__pycache__" in path.parts:
                continue
            archive.write(path, f"{bundle.name}/{path.relative_to(bundle).as_posix()}")
    return buffer.getvalue()


class _AsyncReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk = self._payload[self._offset :]
            self._offset = len(self._payload)
            return chunk
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class _EnvironmentManager:
    """Real preview shape; preparation is stubbed so no build ever runs."""

    def __init__(self) -> None:
        self.prepared: list[str] = []

    def preview(self, skill) -> dict:
        return {
            "python_project": bool(skill.executable_assets["python_project"]),
            "dependencies": [],
            "build_requirements": [],
            "declared_commands": [],
            "dependency_lock": None,
            "confirmation_required": bool(skill.executable_assets["python_project"]),
        }

    def prepare(self, skill, *, approve_setup, force=False) -> dict:
        self.prepared.append(skill.name)
        return {"status": "ready", "commands": []}

    def inspect(self, skill) -> dict:
        return {"status": "ready", "commands": []}

    def remove_skill(self, name) -> None:
        return None


class _SecretStore:
    def remove_skill(self, name) -> bool:
        return True


class _ReadinessManager:
    def evaluate_readiness(self, _skill):
        return SimpleNamespace(status="ready", setup_status="not_required")


class _BridgeStub:
    """Captures what the sidecar would publish to the canonical server."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.published: list[dict] = []
        self.registry: LocalSkillsRegistry | None = None

    def is_connected(self) -> bool:
        return True

    def get_registered_device_id(self) -> str:
        return self.device_id

    async def refresh_catalogs(self) -> None:
        assert self.registry is not None
        self.published.append(self.registry.get_skill_catalog(include_content=False))


@pytest.fixture()
def sidecar_profile(tmp_path, monkeypatch):
    """A real sidecar skill stack rooted in a temporary profile."""
    from client_backend.core.config import client_settings
    from client_backend.services import local_skills_registry as registry_module
    from client_backend.services.skill_runtime import install as install_module

    original_profile_root = client_settings.profile_root
    client_settings.profile_root = str(tmp_path / "profile")
    monkeypatch.setattr(
        registry_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: USER_ID),
    )
    monkeypatch.setitem(
        SkillCatalogService.__init__.__globals__,
        "SkillRuntimeManager",
        _ReadinessManager,
    )

    registry = LocalSkillsRegistry(skill_roots=[])
    environment = _EnvironmentManager()
    device_id = str(uuid4())
    bridge = _BridgeStub(device_id)
    bridge.registry = registry

    installer = install_module.SkillBundleInstaller(
        registry=registry,
        environment_manager=environment,
        secret_store=_SecretStore(),
    )
    uploads = SkillUploadService(
        environment_manager=environment,
        registry=registry,
        audit=None,
    )
    catalog = SkillCatalogService(registry=registry, bridge_factory=lambda: bridge)
    operations = SkillInstallationService(
        installer_factory=lambda: installer,
        upload_service=uploads,
        catalog_service=catalog,
        audit=None,
    )
    try:
        yield SimpleNamespace(
            uploads=uploads,
            operations=operations,
            catalog=catalog,
            registry=registry,
            environment=environment,
            bridge=bridge,
            device_id=device_id,
        )
    finally:
        client_settings.profile_root = original_profile_root


async def _wait_terminal(service: SkillInstallationService, operation_id: str):
    import asyncio

    for _ in range(400):
        operation = service.get_owned(USER_ID, operation_id)
        if operation.is_terminal:
            return operation
        await asyncio.sleep(0.005)
    raise AssertionError("installation never reached a terminal state")


def _resolver_sessions(monkeypatch, *, user_id, device_id, catalog):
    """Point the canonical resolver at the catalog the sidecar published."""
    session = SimpleNamespace(
        user_id=user_id,
        session_id="session-a",
        skill_catalog=catalog,
    )
    monkeypatch.setattr(
        "app.services.client_device_service.ClientDeviceService.lookup_active_session",
        lambda requested: session if str(requested) == str(device_id) else None,
    )


@pytest.mark.asyncio
async def test_uploaded_skill_reaches_only_originating_device_chat(
    sidecar_profile, monkeypatch
):
    archive = _fixture_zip(FIXTURE_BUNDLE)

    upload = await sidecar_profile.uploads.stage(
        user_id=USER_ID,
        filename="google-calendar.zip",
        stream=_AsyncReader(archive),
    )
    assert upload.preview.name == FIXTURE_SKILL_NAME

    operation = await sidecar_profile.operations.start(
        USER_ID,
        upload.upload_id,
        SkillInstallationRequest(
            expected_source_hash=upload.preview.source_hash,
            approve_setup=False,
        ),
    )
    operation = await _wait_terminal(sidecar_profile.operations, operation.operation_id)

    assert operation.state == "succeeded", operation.failure
    assert operation.result.name == FIXTURE_SKILL_NAME
    assert sidecar_profile.environment.prepared == []

    published = sidecar_profile.bridge.published[-1]
    user_id = uuid4()
    _resolver_sessions(
        monkeypatch,
        user_id=user_id,
        device_id=sidecar_profile.device_id,
        catalog=published,
    )

    origin = list_resolved_skills(user_id=str(user_id), device_id=sidecar_profile.device_id)
    other = list_resolved_skills(user_id=str(user_id), device_id=str(uuid4()))
    unbound = list_resolved_skills(user_id=str(user_id), device_id=None)

    assert {skill.name for skill in origin} == {FIXTURE_SKILL_NAME}
    assert other == []
    assert unbound == []


@pytest.mark.asyncio
async def test_installed_bundle_keeps_its_command_and_hides_local_paths(sidecar_profile):
    upload = await sidecar_profile.uploads.stage(
        user_id=USER_ID,
        filename="google-calendar.zip",
        stream=_AsyncReader(_fixture_zip(FIXTURE_BUNDLE)),
    )
    operation = await sidecar_profile.operations.start(
        USER_ID,
        upload.upload_id,
        SkillInstallationRequest(expected_source_hash=upload.preview.source_hash),
    )
    await _wait_terminal(sidecar_profile.operations, operation.operation_id)

    installed = sidecar_profile.registry.get_skill(FIXTURE_SKILL_NAME)
    assert installed is not None
    assert installed.executable_assets["bin"] == ["cli-anything-google-calendar.py"]

    # Provenance is recorded as an upload, never as the staging directory it
    # came from -- that path is deleted moments later.
    metadata = installed.install_metadata or {}
    assert metadata["source"] == "upload"
    assert "source_path" not in metadata


@pytest.mark.asyncio
async def test_second_upload_of_the_same_skill_needs_an_explicit_replacement(sidecar_profile):
    archive = _fixture_zip(FIXTURE_BUNDLE)
    first_upload = await sidecar_profile.uploads.stage(
        user_id=USER_ID,
        filename="google-calendar.zip",
        stream=_AsyncReader(archive),
    )
    first = await sidecar_profile.operations.start(
        USER_ID,
        first_upload.upload_id,
        SkillInstallationRequest(expected_source_hash=first_upload.preview.source_hash),
    )
    await _wait_terminal(sidecar_profile.operations, first.operation_id)

    second_upload = await sidecar_profile.uploads.stage(
        user_id=USER_ID,
        filename="google-calendar.zip",
        stream=_AsyncReader(archive),
    )
    assert second_upload.preview.existing_skill is not None
    assert second_upload.preview.existing_skill.replaceable is True

    conflicted = await sidecar_profile.operations.start(
        USER_ID,
        second_upload.upload_id,
        SkillInstallationRequest(expected_source_hash=second_upload.preview.source_hash),
    )
    conflicted = await _wait_terminal(sidecar_profile.operations, conflicted.operation_id)

    assert conflicted.state == "failed"
    assert conflicted.failure.code == "SKILL_INSTALL_CONFLICT"


@pytest.mark.asyncio
async def test_uploaded_skill_is_absent_from_a_catalog_published_before_it(sidecar_profile):
    """The generation must advance, or a client could keep serving the old list."""
    before = await sidecar_profile.catalog.snapshot(force=True)
    assert before["totalCount"] == 0

    upload = await sidecar_profile.uploads.stage(
        user_id=USER_ID,
        filename="google-calendar.zip",
        stream=_AsyncReader(_fixture_zip(FIXTURE_BUNDLE)),
    )
    operation = await sidecar_profile.operations.start(
        USER_ID,
        upload.upload_id,
        SkillInstallationRequest(expected_source_hash=upload.preview.source_hash),
    )
    operation = await _wait_terminal(sidecar_profile.operations, operation.operation_id)

    after = operation.result.catalog
    assert after["totalCount"] == 1
    assert after["catalogGeneration"] > before["catalogGeneration"]
    assert after["skills"][0]["name"] == FIXTURE_SKILL_NAME
