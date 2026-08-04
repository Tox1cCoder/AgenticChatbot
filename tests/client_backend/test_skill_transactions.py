from __future__ import annotations

import asyncio
import json
import threading
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
from client_backend.services.skill_runtime.locks import SKILLS_MUTATION_SCOPE, profile_lock
from shared.skills.errors import SKILL_INSTALL_CONFLICT, SkillRuntimeError

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


@pytest.mark.asyncio
async def test_recovery_resumes_an_incomplete_precommit_rollback(transaction_env, monkeypatch):
    one = _write_skill(transaction_env.root / "sources" / "one", "one", "new one")
    two = _write_skill(transaction_env.root / "sources" / "two", "two", "new two")
    specs = [
        await _spec(transaction_env.installer, one),
        await _spec(transaction_env.installer, two),
    ]
    from client_backend.services.skill_runtime import transactions

    real_replace = transactions._replace_path
    real_rmtree = transactions.shutil.rmtree

    def fail_second_promotion(source: Path, destination: Path) -> None:
        if ".stage-" in source.name and destination.name.startswith("two-"):
            raise OSError("promote two")
        real_replace(source, destination)

    def fail_first_rollback(path: Path) -> None:
        if path.name.startswith("one-") and ".stage-" not in path.name:
            raise OSError("rollback one")
        real_rmtree(path)

    monkeypatch.setattr(transactions, "_replace_path", fail_second_promotion)
    monkeypatch.setattr(transactions.shutil, "rmtree", fail_first_rollback)
    with pytest.raises(OSError, match="promote two"):
        await transaction_env.installer.install_many(specs, transaction_id="tx-recover")

    journal = next(get_skill_operations_root(USER_ID).glob("transaction-*.json"))
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "rolling_back"

    monkeypatch.setattr(transactions.shutil, "rmtree", real_rmtree)
    outcomes = await transactions.recover_install_transactions(
        USER_ID,
        transaction_env.installer,
    )

    assert outcomes == {"tx-recover": "rolled_back"}
    assert _installed_payloads() == []
    transaction_env.installer.finalize_transaction("tx-recover")
    assert not journal.exists()


@pytest.mark.asyncio
async def test_failed_update_preserves_a_preexisting_same_hash_runtime(
    transaction_env, monkeypatch
):
    one = _write_skill(transaction_env.root / "sources" / "one", "one", "same one")
    installed = await transaction_env.installer.install(one)
    runtime = transaction_env.root / "runtimes" / "one" / installed["source_hash"]
    runtime.mkdir(parents=True)
    two = _write_skill(transaction_env.root / "sources" / "two", "two", "new two")
    specs = [
        await _spec(
            transaction_env.installer,
            one,
            replace_source_hash=installed["source_hash"],
        ),
        await _spec(transaction_env.installer, two),
    ]
    from client_backend.services.skill_runtime import transactions

    real_replace = transactions._replace_path

    def fail_second_promotion(source: Path, destination: Path) -> None:
        if ".stage-" in source.name and destination.name.startswith("two-"):
            raise OSError("promote two")
        real_replace(source, destination)

    monkeypatch.setattr(transactions, "_replace_path", fail_second_promotion)

    with pytest.raises(OSError, match="promote two"):
        await transaction_env.installer.install_many(specs, transaction_id="tx-same-hash")

    assert runtime.is_dir()


@pytest.mark.asyncio
async def test_transaction_refreshes_stale_registry_before_authorizing_install(transaction_env):
    await transaction_env.registry.initialize()
    existing = _write_skill(transaction_env.root / "existing" / "one", "one", "existing")
    other_registry = LocalSkillsRegistry(skill_roots=[])
    other_installer = SkillBundleInstaller(
        registry=other_registry,
        environment_manager=transaction_env.environment,
        secret_store=transaction_env.secrets,
    )
    await other_installer.install(existing)
    replacement = _write_skill(transaction_env.root / "replacement" / "one", "one", "replacement")
    spec = await _spec(transaction_env.installer, replacement)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await transaction_env.installer.install_many([spec], transaction_id="tx-stale")

    assert exc_info.value.code == SKILL_INSTALL_CONFLICT


@pytest.mark.asyncio
async def test_committed_recovery_rejects_tampered_bundle_content(transaction_env):
    one = _write_skill(transaction_env.root / "sources" / "one", "one", "original")
    specs = [await _spec(transaction_env.installer, one)]
    installed = await transaction_env.installer.install_many(specs, transaction_id="tx-tampered")
    target = get_installed_skills_root(USER_ID) / installed[0]["install_id"]
    (target / "SKILL.md").write_text(
        "---\nname: one\ndescription: one\n---\ntampered\n", encoding="utf-8"
    )
    from client_backend.services.skill_runtime import transactions

    outcomes = await transactions.recover_install_transactions(USER_ID, transaction_env.installer)

    assert outcomes == {"tx-tampered": "failed"}


@pytest.mark.asyncio
async def test_recovery_holds_the_global_mutation_lock(transaction_env, monkeypatch):
    one = _write_skill(transaction_env.root / "sources" / "one", "one", "original")
    await transaction_env.installer.install_many(
        [await _spec(transaction_env.installer, one)], transaction_id="tx-lock"
    )
    from client_backend.services.skill_runtime import transactions

    entered = threading.Event()
    release = threading.Event()
    real_hash = transactions.compute_skill_bundle_hash

    def blocking_hash(path: Path):
        entered.set()
        assert release.wait(timeout=5)
        return real_hash(path)

    monkeypatch.setattr(transactions, "compute_skill_bundle_hash", blocking_hash)
    recovery = asyncio.create_task(
        transactions.recover_install_transactions(USER_ID, transaction_env.installer)
    )
    assert await asyncio.to_thread(entered.wait, 5)
    acquired = asyncio.Event()

    async def competing_mutation() -> None:
        async with profile_lock(USER_ID, SKILLS_MUTATION_SCOPE):
            acquired.set()

    competitor = asyncio.create_task(competing_mutation())
    await asyncio.sleep(0.05)
    assert not acquired.is_set()
    release.set()
    await recovery
    await competitor
    assert acquired.is_set()


@pytest.mark.asyncio
async def test_recovery_removes_an_unjournaled_cancelled_install_stage(transaction_env):
    orphan = get_installed_skills_root(USER_ID) / "one-deadbeef.stage-cancelled"
    orphan.mkdir(parents=True)
    (orphan / "SKILL.md").write_text("partial", encoding="utf-8")
    from client_backend.services.skill_runtime import transactions

    outcomes = await transactions.recover_install_transactions(USER_ID, transaction_env.installer)

    assert outcomes == {}
    assert not orphan.exists()


@pytest.mark.asyncio
async def test_cancellation_waits_for_inflight_promotion_then_commits(transaction_env, monkeypatch):
    one = _write_skill(transaction_env.root / "sources" / "one", "one", "new one")
    spec = await _spec(transaction_env.installer, one)
    from client_backend.services.skill_runtime import transactions

    entered = threading.Event()
    release = threading.Event()
    real_replace = transactions._replace_path

    def blocking_promotion(source: Path, destination: Path) -> None:
        if ".stage-" in source.name:
            entered.set()
            assert release.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr(transactions, "_replace_path", blocking_promotion)
    task = asyncio.create_task(
        transaction_env.installer.install_many([spec], transaction_id="tx-cancel")
    )
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    release.set()
    result = await task

    assert result[0]["name"] == "one"
    assert {payload["bundle_name"] for payload in _installed_payloads()} == {"one"}
