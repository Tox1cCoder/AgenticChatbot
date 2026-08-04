"""Durable all-or-nothing promotion of prepared skill bundles."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from client_backend.core.logging import get_logger
from client_backend.core.paths import get_installed_skills_root, get_skill_operations_root
from client_backend.services.skill_runtime.locks import (
    SKILLS_MUTATION_SCOPE,
    profile_lock,
)
from client_backend.services.skill_runtime.state import atomic_write_json, read_json_object

if TYPE_CHECKING:
    from client_backend.services.skill_runtime.install import (
        PreparedSkillInstall,
        SkillBundleInstaller,
        SkillInstallObserver,
        SkillInstallSpec,
    )

logger = get_logger(__name__)

_JOURNAL_PREFIX = "transaction-"


def _replace_path(source: Path, destination: Path) -> None:
    os.replace(source, destination)


class TransactionMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source_hash: str
    previous_source_hash: str | None = None
    stage_name: str
    target_name: str
    previous_name: str | None = None
    backup_name: str
    runtime_status: str
    action: Literal["installed", "updated"]
    promoted: bool = False
    previous_backed_up: bool = False


class InstallTransactionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    transaction_id: str
    owner: str
    state: Literal[
        "prepared",
        "committing",
        "committed",
        "rolling_back",
        "rolled_back",
    ]
    members: list[TransactionMember]
    cleanup_complete: bool = False
    cleanup_error: str | None = None


class SkillInstallTransaction:
    """Prepare every member, then promote or restore the complete set."""

    def __init__(self, installer: SkillBundleInstaller, user_id: str) -> None:
        self._installer = installer
        self._user_id = user_id

    async def execute(
        self,
        specs: list[SkillInstallSpec],
        *,
        transaction_id: str,
        observer: SkillInstallObserver | None = None,
    ) -> list[dict]:
        if not specs:
            raise ValueError("a skill install transaction requires at least one member")
        if self._journal_path(transaction_id).exists():
            raise RuntimeError("a transaction journal already exists for this operation")
        async with profile_lock(self._user_id, SKILLS_MUTATION_SCOPE):
            prepared = await self._prepare_all(specs, observer)
            record = self._record(transaction_id, prepared)
            self._persist(record)
            try:
                if observer is not None:
                    await observer.phase("committing")
                    await observer.before_commit()
                record.state = "committing"
                self._persist(record)
                for item, member in zip(prepared, record.members, strict=True):
                    if item.previous is not None and item.previous.exists():
                        await asyncio.to_thread(
                            _replace_path,
                            item.previous,
                            item.backup,
                        )
                        member.previous_backed_up = True
                        self._persist(record)
                    await asyncio.to_thread(_replace_path, item.stage, item.target)
                    member.promoted = True
                    self._persist(record)
                record.state = "committed"
                self._persist(record)
            except BaseException:
                await self._rollback(record)
                raise

            await self._finish_committed_cleanup(record)
            try:
                await self._installer._registry.refresh()
            except Exception:  # noqa: BLE001 - commit is already durable
                logger.warning(
                    "registry refresh failed after committed skill transaction", exc_info=True
                )
            return [self._result(item) for item in prepared]

    async def _prepare_all(
        self,
        specs: list[SkillInstallSpec],
        observer: SkillInstallObserver | None,
    ) -> list[PreparedSkillInstall]:
        prepared: list[PreparedSkillInstall] = []
        try:
            for spec in specs:
                item = await self._installer.prepare_install(
                    spec,
                    observer=observer,
                    user_id=self._user_id,
                )
                duplicate = any(previous.name == item.name for previous in prepared)
                prepared.append(item)
                if duplicate:
                    raise ValueError(f"duplicate skill '{item.name}' in one transaction")
        except BaseException:
            await self._discard_prepared(prepared)
            raise
        return prepared

    async def _discard_prepared(self, prepared: list[PreparedSkillInstall]) -> None:
        for item in reversed(prepared):
            if item.stage.exists():
                with contextlib.suppress(OSError):
                    await asyncio.to_thread(shutil.rmtree, item.stage)
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    self._installer._environment.remove_runtime,
                    item.name,
                    item.source_hash,
                )

    def _record(
        self,
        transaction_id: str,
        prepared: list[PreparedSkillInstall],
    ) -> InstallTransactionRecord:
        return InstallTransactionRecord(
            transaction_id=transaction_id,
            owner=self._user_id,
            state="prepared",
            members=[
                TransactionMember(
                    name=item.name,
                    source_hash=item.source_hash,
                    previous_source_hash=item.previous_source_hash,
                    stage_name=item.stage.name,
                    target_name=item.target.name,
                    previous_name=item.previous.name if item.previous is not None else None,
                    backup_name=item.backup.name,
                    runtime_status=item.runtime_status,
                    action=item.action,
                )
                for item in prepared
            ],
        )

    async def _rollback(self, record: InstallTransactionRecord) -> None:
        record.state = "rolling_back"
        self._persist(record)
        failures: list[str] = []
        root = get_installed_skills_root(self._user_id)
        for member in reversed(record.members):
            try:
                target = self._member_path(root, member.target_name)
                stage = self._member_path(root, member.stage_name)
                backup = self._member_path(root, member.backup_name)
                previous = (
                    self._member_path(root, member.previous_name)
                    if member.previous_name is not None
                    else None
                )
                if member.promoted and target.exists():
                    await asyncio.to_thread(shutil.rmtree, target)
                    member.promoted = False
                    self._persist(record)
                if member.previous_backed_up and previous is not None and backup.exists():
                    await asyncio.to_thread(_replace_path, backup, previous)
                    member.previous_backed_up = False
                    self._persist(record)
                if stage.exists():
                    await asyncio.to_thread(shutil.rmtree, stage)
                await asyncio.to_thread(
                    self._installer._environment.remove_runtime,
                    member.name,
                    member.source_hash,
                )
            except Exception as exc:  # noqa: BLE001 - durable retry state
                failures.append(f"{member.name}: {exc}")
        if failures:
            record.cleanup_error = "; ".join(failures)
            self._persist(record)
            return
        record.state = "rolled_back"
        record.cleanup_complete = True
        record.cleanup_error = None
        self._persist(record)

    async def _finish_committed_cleanup(self, record: InstallTransactionRecord) -> None:
        root = get_installed_skills_root(self._user_id)
        failures: list[str] = []
        for member in record.members:
            try:
                backup = self._member_path(root, member.backup_name)
                if backup.exists():
                    await asyncio.to_thread(shutil.rmtree, backup)
                if (
                    member.previous_source_hash
                    and member.previous_source_hash != member.source_hash
                ):
                    await asyncio.to_thread(
                        self._installer._environment.remove_runtime,
                        member.name,
                        member.previous_source_hash,
                    )
            except Exception as exc:  # noqa: BLE001 - committed cleanup is retryable
                failures.append(f"{member.name}: {exc}")
        record.cleanup_complete = not failures
        record.cleanup_error = "; ".join(failures) or None
        self._persist(record)

    async def recover(self) -> Literal["committed", "rolled_back", "failed"]:
        record = self._read()
        if record is None:
            return "failed"
        if record.state == "committed":
            if not self._committed_members_match(record):
                return "failed"
            await self._finish_committed_cleanup(record)
            return "committed" if record.cleanup_complete else "failed"
        if record.state == "rolled_back" and record.cleanup_complete:
            return "rolled_back"
        await self._rollback(record)
        return "rolled_back" if record.state == "rolled_back" else "failed"

    def finalize(self) -> None:
        record = self._read()
        if record is None:
            return
        if record.state not in {"committed", "rolled_back"} or not record.cleanup_complete:
            raise RuntimeError("cannot finalize an incomplete skill transaction")
        self._journal_path().unlink(missing_ok=True)

    def _committed_members_match(self, record: InstallTransactionRecord) -> bool:
        root = get_installed_skills_root(self._user_id)
        for member in record.members:
            target = self._member_path(root, member.target_name)
            payload = read_json_object(target / "install.json")
            if payload is None or payload.get("source_hash") != member.source_hash:
                return False
        return True

    def _persist(self, record: InstallTransactionRecord) -> None:
        atomic_write_json(self._journal_path(record.transaction_id), record.model_dump(mode="json"))

    def _read(self) -> InstallTransactionRecord | None:
        payload = read_json_object(self._journal_path())
        if payload is None:
            return None
        record = InstallTransactionRecord.model_validate(payload)
        if record.owner != self._user_id:
            raise RuntimeError("skill transaction owner does not match the active profile")
        return record

    def _journal_path(self, transaction_id: str | None = None) -> Path:
        identifier = transaction_id or getattr(self, "_transaction_id", None)
        if not identifier:
            raise RuntimeError("transaction id is required")
        self._transaction_id = str(identifier)
        digest = hashlib.sha256(str(identifier).encode("utf-8")).hexdigest()[:32]
        return get_skill_operations_root(self._user_id) / f"{_JOURNAL_PREFIX}{digest}.json"

    @staticmethod
    def _member_path(root: Path, name: str) -> Path:
        if not name or name in {".", ".."} or Path(name).name != name:
            raise RuntimeError("transaction member path is not one confined component")
        return root / name

    @staticmethod
    def _result(item: PreparedSkillInstall) -> dict:
        return {
            "name": item.name,
            "install_id": item.target.name,
            "source_hash": item.source_hash,
            "runtime_status": item.runtime_status,
            "action": item.action,
        }


async def recover_install_transactions(
    user_id: str,
    installer: SkillBundleInstaller,
) -> dict[str, Literal["committed", "rolled_back", "failed"]]:
    root = get_skill_operations_root(user_id)
    if not root.is_dir():
        return {}
    outcomes: dict[str, Literal["committed", "rolled_back", "failed"]] = {}
    for path in sorted(root.glob(f"{_JOURNAL_PREFIX}*.json")):
        payload = read_json_object(path)
        if payload is None:
            continue
        record = InstallTransactionRecord.model_validate(payload)
        transaction = SkillInstallTransaction(installer, user_id)
        transaction._transaction_id = record.transaction_id
        outcomes[record.transaction_id] = await transaction.recover()
    return outcomes
