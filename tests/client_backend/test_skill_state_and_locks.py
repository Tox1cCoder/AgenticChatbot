"""Durability and mutual-exclusion primitives for skill installation.

An installation is a multi-step mutation of shared on-disk state that can be
interrupted by a crash or raced by a second request. These two modules are what
make the rest of the workflow recoverable: state writes are all-or-nothing, and
one skill can only be mutated by one worker at a time within and across
processes.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os

import pytest

from client_backend.services.skill_runtime.locks import (
    SkillLockTimeoutError,
    profile_lock,
)
from client_backend.services.skill_runtime.state import (
    SkillStateError,
    atomic_write_json,
    read_json_object,
)


def _live_locks() -> dict:
    """The lock module's globals, which is what ``profile_lock`` resolves names in.

    Not ``sys.modules[...]``: another test in this directory evicts every
    ``client_backend*`` module, after which a fresh import is a different object
    from the one this file closed over. See the same note in test_skill_uploads.py.
    """
    return inspect.unwrap(profile_lock).__globals__


def _live_settings():
    """The settings proxy the lock module resolves against."""
    return _live_locks()["client_settings"]


def test_atomic_write_json_never_leaves_partial_target(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"version": 1, "state": "ready"})
    real_replace = os.replace

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write_json(target, {"version": 1, "state": "changed"})

    monkeypatch.setattr(os, "replace", real_replace)
    assert read_json_object(target)["state"] == "ready"
    assert not list(tmp_path.glob("*.tmp-*"))


def test_atomic_write_json_creates_parents_and_round_trips(tmp_path):
    target = tmp_path / "nested" / "deeper" / "state.json"

    atomic_write_json(target, {"version": 1, "generation": 7})

    assert read_json_object(target) == {"version": 1, "generation": 7}


def test_read_json_object_returns_none_for_missing_file(tmp_path):
    assert read_json_object(tmp_path / "absent.json") is None


@pytest.mark.parametrize("payload", ["[1, 2]", '"text"', "not json at all", ""])
def test_read_json_object_rejects_non_object_payloads(tmp_path, payload):
    target = tmp_path / "state.json"
    target.write_text(payload, encoding="utf-8")

    with pytest.raises(SkillStateError):
        read_json_object(target)


def test_atomic_write_json_leaves_no_temporary_files_on_success(tmp_path):
    target = tmp_path / "state.json"

    for generation in range(3):
        atomic_write_json(target, {"version": 1, "generation": generation})

    assert [path.name for path in tmp_path.iterdir()] == ["state.json"]


@pytest.mark.asyncio
async def test_profile_lock_times_out_for_same_scope(monkeypatch):
    monkeypatch.setattr(_live_settings(), "skill_install_lock_timeout_seconds", 0.05)
    async with profile_lock("user-a", "skill:demo"):
        with pytest.raises(SkillLockTimeoutError):
            async with profile_lock("user-a", "skill:demo"):
                pass


@pytest.mark.asyncio
async def test_profile_locks_allow_different_scopes():
    async with profile_lock("user-a", "skill:a"), profile_lock("user-a", "skill:b"):
        pass


@pytest.mark.asyncio
async def test_profile_locks_are_scoped_per_user(monkeypatch):
    monkeypatch.setattr(_live_settings(), "skill_install_lock_timeout_seconds", 0.05)
    async with profile_lock("user-a", "skill:demo"), profile_lock("user-b", "skill:demo"):
        pass


@pytest.mark.asyncio
async def test_profile_lock_is_released_after_an_exception(monkeypatch):
    monkeypatch.setattr(_live_settings(), "skill_install_lock_timeout_seconds", 0.05)

    with pytest.raises(RuntimeError):
        async with profile_lock("user-a", "skill:demo"):
            raise RuntimeError("body failed")

    async with profile_lock("user-a", "skill:demo"):
        pass


@pytest.mark.asyncio
async def test_profile_lock_serializes_concurrent_waiters():
    """A waiter runs only after the holder releases, never interleaved."""
    order: list[str] = []

    async def worker(name: str) -> None:
        async with profile_lock("user-a", "skill:demo", timeout_seconds=5):
            order.append(f"enter-{name}")
            await asyncio.sleep(0.01)
            order.append(f"exit-{name}")

    await asyncio.gather(worker("a"), worker("b"))

    assert order in (
        ["enter-a", "exit-a", "enter-b", "exit-b"],
        ["enter-b", "exit-b", "enter-a", "exit-a"],
    )


@pytest.mark.asyncio
async def test_lock_filenames_never_embed_raw_scope_text(tmp_path, monkeypatch):
    """Skill names and user ids are hashed, never pasted into a filename."""
    monkeypatch.setitem(_live_locks(), "get_skill_locks_root", lambda user_id: tmp_path)
    async with profile_lock("user-a", "skill:../escape"):
        names = [path.name for path in tmp_path.iterdir()]

    assert names
    for name in names:
        assert "escape" not in name
        assert ".." not in name
        assert "user-a" not in name


def test_lifecycle_audit_never_serializes_paths_or_uploaded_names(tmp_path):
    from client_backend.services.skill_runtime.audit import (
        LIFECYCLE_AUDIT_FIELDS,
        SkillLifecycleAuditWriter,
    )

    writer = SkillLifecycleAuditWriter(path=tmp_path / "lifecycle.jsonl")
    writer.write(
        event="upload_staged",
        user_id="user-a",
        upload_id="upload-a",
        skill="demo",
        source_hash="a" * 64,
        status="succeeded",
        metrics={"compressed_bytes": 10, "file_count": 2},
    )
    payload = json.loads((tmp_path / "lifecycle.jsonl").read_text(encoding="utf-8"))

    assert set(payload) <= LIFECYCLE_AUDIT_FIELDS
    assert "path" not in json.dumps(payload).lower()
    assert "filename" not in json.dumps(payload).lower()


def test_lifecycle_audit_drops_unknown_and_sensitive_keys(tmp_path):
    from client_backend.services.skill_runtime.audit import SkillLifecycleAuditWriter

    writer = SkillLifecycleAuditWriter(path=tmp_path / "lifecycle.jsonl")
    writer.write(
        event="install_committed",
        user_id="user-a",
        staging_path="C:/Users/dev/AppData/staging",
        filename="my-secret-bundle.zip",
        access_token="super-secret",
        setup_output="pip install ...",
    )
    line = (tmp_path / "lifecycle.jsonl").read_text(encoding="utf-8")

    assert "AppData" not in line
    assert "my-secret-bundle" not in line
    assert "super-secret" not in line
    assert "pip install" not in line
    assert json.loads(line)["event"] == "install_committed"


def test_lifecycle_audit_rejects_non_numeric_metrics(tmp_path):
    from client_backend.services.skill_runtime.audit import SkillLifecycleAuditWriter

    writer = SkillLifecycleAuditWriter(path=tmp_path / "lifecycle.jsonl")
    writer.write(
        event="upload_rejected",
        user_id="user-a",
        metrics={"file_count": 3, "leaked": "C:/Users/dev/archive.zip"},
    )
    payload = json.loads((tmp_path / "lifecycle.jsonl").read_text(encoding="utf-8"))

    assert payload["metrics"] == {"file_count": 3}


def test_lifecycle_audit_failure_never_propagates(tmp_path, monkeypatch):
    """Audit is a side channel: a write failure must not fail the operation."""
    from client_backend.services.skill_runtime.audit import SkillLifecycleAuditWriter

    writer = SkillLifecycleAuditWriter(path=tmp_path / "lifecycle.jsonl")
    monkeypatch.setattr(
        SkillLifecycleAuditWriter,
        "_append_line",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    writer.write(event="upload_staged", user_id="user-a")


def test_lifecycle_audit_appends_one_line_per_event(tmp_path):
    from client_backend.services.skill_runtime.audit import SkillLifecycleAuditWriter

    writer = SkillLifecycleAuditWriter(path=tmp_path / "lifecycle.jsonl")
    writer.write(event="upload_staged", user_id="user-a")
    writer.write(event="install_started", user_id="user-a", operation_id="operation-a")

    lines = (tmp_path / "lifecycle.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert [json.loads(line)["event"] for line in lines] == [
        "upload_staged",
        "install_started",
    ]
    assert all(json.loads(line)["timestamp"].endswith("+00:00") for line in lines)
