"""Every skill read and mutation answers with one catalog shape.

Before this, a toggle returned ``{"message": ...}`` and a reload returned
``{"message": ...}``, so a client had to issue a follow-up ``GET /skills`` and
hope it did not race the mutation it just made. Each route now returns the
catalog it produced, tagged with a generation the client can order responses by.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import skills as skills_api
from client_backend.services import skill_catalog as catalog_module


class _Skill:
    def __init__(self, name: str, *, enabled: bool = True, command_capable: bool = True):
        self.name = name
        self.description = f"Description for {name}"
        self.enabled = enabled
        self.path = Path(f"/tmp/{name}/SKILL.md")
        self.content = f"Content for {name}"
        self.source_hash = "a" * 64
        self.executable_assets = {
            "bin": [f"{name}.py"] if command_capable else [],
            "scripts": [],
            "python_project": False,
        }


class _RegistryStub:
    def __init__(self):
        self.skills = {"demo": _Skill("demo")}
        self.initialize_calls = 0
        self.refresh_calls = 0
        self.toggle_calls: list[tuple[str, bool]] = []

    async def initialize(self) -> None:
        self.initialize_calls += 1

    def get_all_skills(self):
        return list(self.skills.values())

    def get_skill(self, name: str):
        return self.skills.get(name)

    def set_skill_enabled(self, name: str, enabled: bool) -> bool:
        self.toggle_calls.append((name, enabled))
        skill = self.skills.get(name)
        if skill is None:
            return False
        skill.enabled = enabled
        return True

    async def refresh(self) -> int:
        self.refresh_calls += 1
        return 0

    def _resolve_current_user_id(self) -> str:
        return "user-a"


class _BridgeStub:
    def __init__(self, *, connected: bool = True, device_id: str | None = "device-123"):
        self.connected = connected
        self.device_id = device_id
        self.refresh_calls = 0
        self.raise_on_refresh: Exception | None = None

    def is_connected(self) -> bool:
        return self.connected

    def get_registered_device_id(self) -> str | None:
        return self.device_id

    async def refresh_catalogs(self) -> None:
        self.refresh_calls += 1
        if self.raise_on_refresh is not None:
            raise self.raise_on_refresh


class _ReadinessManager:
    def __init__(self, status: str, setup_status: str):
        self.status = status
        self.setup_status = setup_status

    def evaluate_readiness(self, _skill):
        return self


class _InstallerStub:
    def __init__(self) -> None:
        self.uninstall_calls: list[str] = []

    async def uninstall(self, name: str) -> dict:
        self.uninstall_calls.append(name)
        return {"name": name, "removed": True, "cleanup_status": "complete"}

    async def setup(self, name: str, *, expected_source_hash, approve_setup) -> dict:
        return {"name": name, "source_hash": expected_source_hash, "status": "ready"}


def _assert_catalog(data: dict) -> None:
    """Every catalog payload a client may cache must carry these fields."""
    assert data["deviceId"] == "device-123"
    assert isinstance(data["catalogGeneration"], int)
    assert data["catalogSyncStatus"] in {"synced", "pending", "disconnected"}
    assert data["totalCount"] == len(data["skills"])
    assert data["enabledCount"] == sum(1 for skill in data["skills"] if skill["enabled"])


@pytest.fixture()
def skills_env(tmp_path, monkeypatch):
    """A skills app whose catalog service is backed by stubs on a temp profile."""
    registry = _RegistryStub()
    bridge = _BridgeStub()
    installer = _InstallerStub()

    catalog_globals = catalog_module.SkillCatalogService.__init__.__globals__
    locks_globals = catalog_globals["profile_lock"].__wrapped__.__globals__
    monkeypatch.setitem(
        catalog_globals,
        "get_skill_catalog_state_path",
        lambda _user: tmp_path / "catalog_state.json",
    )
    monkeypatch.setitem(
        catalog_globals,
        "SkillRuntimeManager",
        lambda: _ReadinessManager("ready", "not_required"),
    )
    monkeypatch.setitem(locks_globals, "get_skill_locks_root", lambda _user: tmp_path / "locks")

    service = catalog_module.SkillCatalogService(registry=registry, bridge_factory=lambda: bridge)
    api_globals = skills_api.list_skills.__globals__
    monkeypatch.setitem(api_globals, "get_skill_catalog_service", lambda: service)
    monkeypatch.setitem(api_globals, "get_skills_registry", lambda: registry)
    monkeypatch.setitem(api_globals, "get_skill_installer", lambda: installer)

    app = FastAPI()
    app.include_router(skills_api.router)
    app.dependency_overrides[skills_api.require_local_session] = lambda: object()
    with TestClient(app) as client:
        yield type(
            "_SkillsEnv",
            (),
            {
                "client": client,
                "registry": registry,
                "bridge": bridge,
                "installer": installer,
                "service": service,
            },
        )


def test_list_returns_the_catalog(skills_env):
    response = skills_env.client.get("/skills")

    assert response.status_code == 200
    data = response.json()["data"]
    _assert_catalog(data)
    assert data["skills"][0]["name"] == "demo"
    assert data["skills"][0]["commandCapable"] is True


def test_detail_still_returns_content_for_one_skill(skills_env):
    response = skills_env.client.get("/skills/demo")

    assert response.status_code == 200
    assert response.json()["data"]["content"] == "Content for demo"
    assert response.json()["data"]["runtimeStatus"] == "ready"


def test_unknown_skill_detail_is_a_404(skills_env):
    response = skills_env.client.get("/skills/missing")

    assert response.status_code == 404


def test_reload_returns_refreshed_catalog(skills_env):
    response = skills_env.client.post("/skills/reload")

    assert response.status_code == 200
    _assert_catalog(response.json()["data"])
    assert skills_env.registry.refresh_calls >= 1
    assert skills_env.bridge.refresh_calls == 1


def test_toggle_returns_refreshed_catalog(skills_env):
    response = skills_env.client.patch("/skills/demo/toggle?enabled=false")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["message"] == "Skill 'demo' disabled"
    _assert_catalog(data["catalog"])
    assert data["catalog"]["enabledCount"] == 0
    assert skills_env.registry.toggle_calls == [("demo", False)]
    assert skills_env.bridge.refresh_calls == 1


def test_toggling_an_unknown_skill_is_a_404(skills_env):
    response = skills_env.client.patch("/skills/missing/toggle?enabled=false")

    assert response.status_code == 404


def test_uninstall_returns_refreshed_catalog(skills_env):
    response = skills_env.client.post("/skills/uninstall", json={"name": "demo"})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["removed"] is True
    assert data["cleanup_status"] == "complete"
    _assert_catalog(data["catalog"])
    assert skills_env.installer.uninstall_calls == ["demo"]


def test_setup_returns_refreshed_catalog(skills_env):
    response = skills_env.client.post(
        "/skills/demo/setup",
        json={"expectedSourceHash": "a" * 64, "approveSetup": True},
    )

    assert response.status_code == 200
    _assert_catalog(response.json()["data"]["catalog"])


def test_mutation_requests_accept_documented_snake_case(skills_env):
    response = skills_env.client.post(
        "/skills/demo/setup",
        json={"expected_source_hash": "a" * 64, "approve_setup": True},
    )

    assert response.status_code == 200


def test_mutation_requests_reject_unknown_fields(skills_env):
    """A misspelled approval field must fail loudly, not default to False."""
    response = skills_env.client.post(
        "/skills/demo/setup",
        json={"expectedSourceHash": "a" * 64, "approveSetupNow": True},
    )

    assert response.status_code == 422


def test_bridge_sync_failure_still_succeeds_as_pending(skills_env):
    skills_env.bridge.raise_on_refresh = RuntimeError("offline")

    response = skills_env.client.post("/skills/reload")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["catalogSyncStatus"] == "pending"


def test_disconnected_bridge_reports_disconnected(skills_env):
    skills_env.bridge.connected = False
    skills_env.bridge.device_id = None

    response = skills_env.client.post("/skills/reload")

    assert response.json()["data"]["catalogSyncStatus"] == "disconnected"
    assert response.json()["data"]["deviceId"] is None


def test_catalog_generation_advances_across_a_mutation(skills_env):
    before = skills_env.client.get("/skills").json()["data"]["catalogGeneration"]

    toggled = skills_env.client.patch("/skills/demo/toggle?enabled=false").json()

    assert toggled["data"]["catalog"]["catalogGeneration"] == before + 1


def test_skill_summary_marks_instruction_only_skill_as_not_command_capable(monkeypatch):
    monkeypatch.setitem(
        catalog_module.skill_summary.__globals__,
        "SkillRuntimeManager",
        lambda: _ReadinessManager("instruction_only", "not_required"),
    )

    summary = catalog_module.skill_summary(_Skill("instructions", command_capable=False))

    assert summary["commandCapable"] is False
    assert summary["runtimeStatus"] == "instruction_only"


def test_skill_summary_exposes_safe_setup_status(monkeypatch):
    monkeypatch.setitem(
        catalog_module.skill_summary.__globals__,
        "SkillRuntimeManager",
        lambda: _ReadinessManager("not_ready", "setup_required"),
    )

    summary = catalog_module.skill_summary(_Skill("python-skill", command_capable=False))

    assert summary["commandCapable"] is False
    assert summary["runtimeStatus"] == "not_ready"
    assert summary["setupStatus"] == "setup_required"
    assert "runtimeRoot" not in summary
    assert "commandPath" not in summary
    assert "setupLogs" not in summary
    assert "secretBindings" not in summary
