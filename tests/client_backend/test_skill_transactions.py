from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from client_backend.core.config import client_settings
from client_backend.core.paths import get_installed_skills_root, get_skill_operations_root
from client_backend.services import local_skills_registry as registry_module
from client_backend.services.local_skills_registry import LocalSkillsRegistry
from client_backend.services.skill_runtime.collection import DiscoveredSkill
from client_backend.services.skill_runtime.environment import SkillEnvironmentManager
from client_backend.services.skill_runtime.install import SkillBundleInstaller, SkillInstallSpec

USER_ID = "user-a"


class _SecretStore:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, str]] = {}

    def remove_skill(self, name: str) -> bool:
        return self.values.pop(name, None) is not None


def _write_skill(root: Path, name: str, body: str) -> DiscoveredSkill:
    root.mkdir(parents=True)
    skill_file = root / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: {name}\n---\n{body}\n",
        encoding="utf-8",
    )
    return DiscoveredSkill(bundle_root=root, skill_file=skill_file)


@pytest.fixture()
def transaction_env(tmp_path, monkeypatch):
    original_profile_root = client_settings.profile_root
    client_settings.profile_root = str(tmp_path / "profiles")
    monkeypatch.setattr(
        registry_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: USER_ID),
    )
    registry = LocalSkillsRegistry(skill_roots=[])
    environment = SkillEnvironmentManager(runtime_base=tmp_path / "runtimes")
    secrets = _SecretStore()
    installer = SkillBundleInstaller(
        registry=registry,
        environment_manager=environment,
        secret_store=secrets,
    )
    try:
        yield SimpleNamespace(
            root=tmp_path,
            registry=registry,
            environment=environment,
            secrets=secrets,
            installer=installer,
        )
    finally:
        client_settings.profile_root = original_profile_root


async def _spec(installer: SkillBundleInstaller, discovered: DiscoveredSkill, **kwargs):
    preview = await installer.preview(discovered)
    return SkillInstallSpec(
        discovered=discovered,
        expected_source_hash=preview["source_hash"],
        approve_setup=False,
        replace_source_hash=kwargs.get("replace_source_hash"),
        source_kind="upload",
    )


def _installed_payloads() -> list[dict]:
    root = get_installed_skills_root(USER_ID)
    if not root.is_dir():
        return []
    return [
        json.loads((bundle / "install.json").read_text(encoding="utf-8"))
        for bundle in root.iterdir()
        if bundle.is_dir() and (bundle / "install.json").is_file()
    ]


@pytest.mark.asyncio
async def test_later_promotion_failure_removes_every_fresh_member(transaction_env, monkeypatch):
    one = _write_skill(transaction_env.root / "sources" / "one", "one", "new one")
    two = _write_skill(transaction_env.root / "sources" / "two", "two", "new two")
    specs = [
        await _spec(transaction_env.installer, one),
        await _spec(transaction_env.installer, two),
    ]

    assert hasattr(transaction_env.installer, "install_many"), (
        "collection installs require one transaction coordinator"
    )
    from client_backend.services.skill_runtime import transactions

    real_replace = transactions._replace_path

    def fail_second_promotion(source: Path, destination: Path) -> None:
        if ".stage-" in source.name and destination.name.startswith("two-"):
            raise OSError("promote two")
        real_replace(source, destination)

    monkeypatch.setattr(transactions, "_replace_path", fail_second_promotion)

    with pytest.raises(OSError, match="promote two"):
        await transaction_env.installer.install_many(specs, transaction_id="tx-fresh")

    assert _installed_payloads() == []
    journals = list(get_skill_operations_root(USER_ID).glob("transaction-*.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text(encoding="utf-8"))["state"] == "rolled_back"


@pytest.mark.asyncio
async def test_later_failure_restores_old_bundles_runtimes_and_secrets(
    transaction_env, monkeypatch
):
    old_one = _write_skill(transaction_env.root / "old" / "one", "one", "old one")
    old_two = _write_skill(transaction_env.root / "old" / "two", "two", "old two")
    first = await transaction_env.installer.install(old_one)
    second = await transaction_env.installer.install(old_two)
    old_hashes = {"one": first["source_hash"], "two": second["source_hash"]}
    for name, source_hash in old_hashes.items():
        (transaction_env.root / "runtimes" / name / source_hash).mkdir(parents=True)
    transaction_env.secrets.values["one"] = {"TOKEN": "encrypted"}

    new_one = _write_skill(transaction_env.root / "new" / "one", "one", "new one")
    new_two = _write_skill(transaction_env.root / "new" / "two", "two", "new two")
    specs = [
        await _spec(
            transaction_env.installer,
            new_one,
            replace_source_hash=old_hashes["one"],
        ),
        await _spec(
            transaction_env.installer,
            new_two,
            replace_source_hash=old_hashes["two"],
        ),
    ]
    from client_backend.services.skill_runtime import transactions

    real_replace = transactions._replace_path

    def fail_second_promotion(source: Path, destination: Path) -> None:
        if ".stage-" in source.name and destination.name.startswith("two-"):
            raise OSError("promote two")
        real_replace(source, destination)

    monkeypatch.setattr(transactions, "_replace_path", fail_second_promotion)

    with pytest.raises(OSError, match="promote two"):
        await transaction_env.installer.install_many(specs, transaction_id="tx-update")

    restored = {payload["bundle_name"]: payload["source_hash"] for payload in _installed_payloads()}
    assert restored == old_hashes
    for name, source_hash in old_hashes.items():
        assert (transaction_env.root / "runtimes" / name / source_hash).is_dir()
    assert transaction_env.secrets.values["one"] == {"TOKEN": "encrypted"}
