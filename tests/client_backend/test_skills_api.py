from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import skills as skills_api


class _Skill:
    def __init__(self, name: str, *, enabled: bool = True, command_capable: bool = True):
        self.name = name
        self.description = f"Description for {name}"
        self.enabled = enabled
        self.path = Path(f"/tmp/{name}/SKILL.md")
        self.content = f"Content for {name}"
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


class _BridgeStub:
    def __init__(self, *, connected: bool = True, device_id: str | None = "device-123"):
        self.connected = connected
        self.device_id = device_id
        self.refresh_calls = 0

    def is_connected(self) -> bool:
        return self.connected

    def get_registered_device_id(self) -> str | None:
        return self.device_id

    async def refresh_catalogs(self) -> None:
        self.refresh_calls += 1


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(skills_api.router)
    app.dependency_overrides[skills_api.require_local_session] = lambda: object()
    return app


def test_skills_routes_return_server_style_payloads(monkeypatch):
    registry = _RegistryStub()
    monkeypatch.setattr(skills_api, "get_skills_registry", lambda: registry)

    with TestClient(_build_app()) as client:
        list_response = client.get("/skills")
        detail_response = client.get("/skills/demo")
        toggle_response = client.patch("/skills/demo/toggle?enabled=false")
        reload_response = client.post("/skills/reload")
        missing_response = client.get("/skills/missing")

    assert list_response.status_code == 200
    assert list_response.json()["data"]["totalCount"] == 1
    assert list_response.json()["data"]["enabledCount"] == 1
    skill_summary = list_response.json()["data"]["skills"][0]
    assert skill_summary["commandCapable"] is True
    assert skill_summary["runtimeStatus"] == "ready"
    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["content"] == "Content for demo"
    assert toggle_response.status_code == 200
    assert toggle_response.json()["data"]["message"] == "Skill 'demo' disabled"
    assert reload_response.status_code == 200
    assert reload_response.json()["message"] == "Skills reloaded"
    assert missing_response.status_code == 404
    assert registry.initialize_calls == 4
    assert registry.refresh_calls == 1
    assert registry.toggle_calls == [("demo", False)]


def test_skill_summary_marks_instruction_only_skill_as_not_command_capable():
    summary = skills_api._skill_summary(
        _Skill("instructions", command_capable=False)
    )

    assert summary["commandCapable"] is False
    assert summary["runtimeStatus"] == "instruction_only"


def test_reload_skills_refreshes_runtime_catalogs_when_bridge_active(monkeypatch):
    registry = _RegistryStub()
    bridge = _BridgeStub()
    monkeypatch.setattr(skills_api, "get_skills_registry", lambda: registry)
    monkeypatch.setattr(skills_api, "get_runtime_bridge", lambda: bridge, raising=False)

    with TestClient(_build_app()) as client:
        response = client.post("/skills/reload")

    assert response.status_code == 200
    assert registry.refresh_calls == 1
    assert bridge.refresh_calls == 1


def test_toggle_skill_refreshes_runtime_catalogs_when_bridge_active(monkeypatch):
    registry = _RegistryStub()
    bridge = _BridgeStub()
    monkeypatch.setattr(skills_api, "get_skills_registry", lambda: registry)
    monkeypatch.setattr(skills_api, "get_runtime_bridge", lambda: bridge, raising=False)

    with TestClient(_build_app()) as client:
        response = client.patch("/skills/demo/toggle?enabled=false")

    assert response.status_code == 200
    assert registry.toggle_calls == [("demo", False)]
    assert bridge.refresh_calls == 1
