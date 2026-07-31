# Sidecar Skill ZIP Installation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a secure, retry-safe ZIP upload and installation workflow to the device sidecar, with fresh catalog responses and equivalent Streamlit and AI SDK frontend behavior.

**Architecture:** A confined archive validator feeds a user-scoped staged-upload store. A persisted asynchronous operation service invokes the existing atomic directory installer under per-skill locks, then a catalog service commits a generation-tagged local snapshot and independently synchronizes the runtime bridge. Streamlit and AI SDK frontends consume the same documented sidecar contract.

**Tech Stack:** Python 3.10+, FastAPI 0.139.2, Starlette 1.3.1, Pydantic v2, `zipfile`, `filelock`, asyncio, Requests/Streamlit, pytest, Ruff.

## Global Constraints

- Accept ZIP only; do not accept TAR, GZIP, 7z, RAR, URLs, or multiple skill bundles.
- Preserve authenticated path-based `POST /skills/install/preview` and `POST /skills/install`.
- A staged upload never executes setup code.
- Default limits: 25 MiB uploaded, 100 MiB expanded, 50 MiB per file, 2,000 entries, 200:1 compression ratio, 20 path components, and 240 portable path characters.
- Default upload TTL: 30 minutes. Default terminal operation receipt TTL: 60 minutes.
- New installs return conflict on any existing name. Updates require `replaceSourceHash` and may replace only profile-installed skills.
- A failed update must preserve the previous bundle and runtime.
- A catalog-sync failure after local commit must not change the committed operation into failure.
- Uploads, operations, bundles, runtimes, locks, and receipts are scoped by server profile and authenticated user.
- New JSON schemas serialize camelCase, accept documented snake_case aliases, and reject unknown fields.
- Never expose staging/runtime paths, archive contents, raw setup output, dependency commands, or secret values through upload/operation responses or logs.
- Keep `/skills/*` and `/api/skills/*` behavior equivalent.
- Keep the AI SDK UI Message Stream unchanged; skill management remains ordinary sidecar HTTP.
- Use TDD for every behavior change and commit after each independently testable task.

## File Structure

### New files

- `client_backend/services/skill_runtime/state.py` — versioned atomic JSON reads/writes.
- `client_backend/services/skill_runtime/locks.py` — bounded in-process and cross-process profile locks.
- `client_backend/services/skill_runtime/archive.py` — ZIP preflight and confined streaming extraction.
- `client_backend/services/skill_runtime/uploads.py` — user-scoped upload records, quotas, expiry, and cleanup.
- `client_backend/services/skill_runtime/operations.py` — persisted asynchronous installation state machine and recovery.
- `client_backend/services/skill_catalog.py` — fresh catalog projection, persisted generation, and best-effort bridge sync.
- `client_backend/schemas/skill_installation.py` — strict camelCase upload/operation request and response models.
- `client_backend/api/skill_errors.py` — stable skill error mapping and route-scoped envelope handlers.
- `tests/client_backend/test_skill_state_and_locks.py` — persistence and lock behavior.
- `tests/client_backend/test_skill_archive.py` — hostile ZIP and resource-limit coverage.
- `tests/client_backend/test_skill_uploads.py` — upload lifecycle, isolation, quota, and cleanup.
- `tests/client_backend/test_skill_catalog.py` — generation, freshness, and degraded sync behavior.
- `tests/client_backend/test_skill_operations.py` — idempotency, transitions, cancellation, recovery, and replacement.
- `tests/client_backend/test_skill_upload_api.py` — multipart and operation wire contract.
- `tests/test_demo_skill_installation.py` — Streamlit transport, session state, polling, and render behavior.
- `tests/test_skill_installation_chat_integration.py` — installed skill publication and device-bound AI SDK chat resolution.

### Modified files

- `pyproject.toml` — declare `filelock` directly.
- `client_backend/core/config.py` — upload, quota, TTL, lock, CORS, and network settings.
- `client_backend/core/paths.py` — validated upload, operation, lock, and catalog paths.
- `client_backend/core/auth.py` — exact restored upstream bearer verification.
- `client_backend/main.py` — configured CORS, skill error handlers, recovery startup, and operation shutdown.
- `client_backend/services/local_skills_registry.py` — freshness metadata and deterministic catalog inputs.
- `client_backend/services/skill_runtime/install.py` — guarded replacement, upload provenance, and sync decoupling.
- `client_backend/services/skill_runtime/audit.py` — non-sensitive upload/install lifecycle audit events.
- `client_backend/schemas/skills.py` — strict alias-compatible existing mutation schemas.
- `client_backend/api/skills.py` — upload/operation routes and catalog-service delegation.
- `client_backend/api/common.py` — allow stable `code` in local response envelopes.
- `client_backend/services/runtime_bridge.py` — serialize bridge catalog refreshes and expose sync state safely.
- `demo.py` — multipart helpers, upload/install operation UI, cleanup, and catalog application.
- `tests/client_backend/test_skill_installation.py` — replacement and post-commit behavior.
- `tests/client_backend/test_skills_api.py` — new catalog shape and mutation responses.
- `tests/client_backend/test_runtime_bridge.py` — concurrent refresh serialization.
- `tests/test_demo_sidecar_auth.py` — restored bearer regression.
- `tests/test_skills_architecture.py` — sidecar-only ownership and Streamlit route guardrails.
- `README.md` — browser ZIP workflow and sidecar endpoints.
- `docs/skill-runtime.md` — archive/install lifecycle and error table.
- `plans/SKILLS_MCP_HITL_FE_CONTRACT.md` — link to the dedicated upload contract.
- `plans/SKILL_INSTALLATION_FE_CONTRACT.md` — keep examples synchronized with implemented schemas.

---

### Task 1: Secure Configuration, Profile Paths, Authentication, and CORS

**Files:**
- Modify: `pyproject.toml`
- Modify: `client_backend/core/config.py:37-149`
- Modify: `client_backend/core/paths.py:50-150`
- Modify: `client_backend/core/auth.py:45-105`
- Modify: `client_backend/main.py:65-87`
- Test: `tests/test_demo_sidecar_auth.py`
- Create: `tests/client_backend/test_skill_upload_security_config.py`

**Interfaces:**
- Produces: validated `ClientSettings.skill_upload_*`, `skill_install_*`, and `allowed_origins` settings.
- Produces: `get_skill_uploads_root(user_id)`, `get_skill_operations_root(user_id)`, `get_skill_locks_root(user_id)`, and `get_skill_catalog_state_path(user_id)`.
- Produces: `require_local_session()` that authorizes a restored upstream bearer only after exact token equality.
- Consumes: existing `profile_subdir_path()` and `initialize_client_environment()`.

- [ ] **Step 1: Write failing authentication, path, settings, and CORS tests**

```python
def test_forged_bearer_subject_cannot_authorize_restored_session(monkeypatch):
    forged = jwt.encode({"sub": "user-a"}, "attacker", algorithm="HS256")
    auth = _restorable_auth_service(user_id="user-a", access_token="real-token")
    monkeypatch.setattr(auth_module, "get_upstream_auth_service", lambda: auth)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=forged)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth_module.require_local_session(credentials))

    assert exc_info.value.status_code == 401


def test_exact_restored_access_token_remains_compatible(monkeypatch):
    token = jwt.encode({"sub": "user-a"}, "upstream", algorithm="HS256")
    auth = _restorable_auth_service(user_id="user-a", access_token=token)
    monkeypatch.setattr(auth_module, "get_upstream_auth_service", lambda: auth)

    payload = asyncio.run(
        auth_module.require_local_session(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        )
    )

    assert payload.user_id == "user-a"


def test_skill_profile_paths_reject_traversal():
    with pytest.raises(ValueError):
        get_skill_uploads_root("../other-user")


def test_skill_upload_defaults_are_production_bounded():
    settings = ClientSettings(_env_file=None)
    assert settings.skill_upload_max_bytes == 25 * 1024 * 1024
    assert settings.skill_upload_max_expanded_bytes == 100 * 1024 * 1024
    assert settings.skill_upload_max_file_bytes == 50 * 1024 * 1024
    assert settings.skill_upload_max_entries == 2000
    assert settings.skill_upload_ttl_seconds == 1800
    assert settings.skill_operation_receipt_ttl_seconds == 3600
```

Add a `create_app()` assertion that production CORS receives configured origins
and never `["*"]`. Add a startup assertion that a non-loopback
`backend_host` fails unless `allow_non_loopback_backend=true`.

- [ ] **Step 2: Run the focused tests and verify the failures**

Run:

```powershell
.\.conda\python.exe -m pytest tests/test_demo_sidecar_auth.py tests/client_backend/test_skill_upload_security_config.py -q
```

Expected: failures for forged-token acceptance and missing settings/path helpers.

- [ ] **Step 3: Add explicit configuration and safe path helpers**

Add direct dependency:

```toml
"filelock>=3.20.0,<4.0.0",
```

Add settings with integer validators rejecting non-positive limits:

```python
skill_upload_max_bytes: int = 25 * 1024 * 1024
skill_upload_max_expanded_bytes: int = 100 * 1024 * 1024
skill_upload_max_file_bytes: int = 50 * 1024 * 1024
skill_upload_max_entries: int = 2_000
skill_upload_max_compression_ratio: int = 200
skill_upload_max_path_depth: int = 20
skill_upload_max_path_chars: int = 240
skill_upload_ttl_seconds: int = 1_800
skill_operation_receipt_ttl_seconds: int = 3_600
skill_upload_max_outstanding: int = 5
skill_upload_quota_bytes: int = 250 * 1024 * 1024
skill_upload_rate_limit_count: int = 10
skill_upload_rate_limit_window_seconds: int = 60
skill_install_lock_timeout_seconds: int = 10
skill_catalog_freshness_seconds: float = 1.0
allowed_origins: list[str] = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8501",
    "http://localhost:8501",
]
allow_non_loopback_backend: bool = False
```

Add a `mode="before"` validator that parses `allowed_origins` from a
comma-separated environment string, strips whitespace, rejects `"*"`, and
requires each value to be a syntactically valid explicit origin.

Validate profile components before joining:

```python
def get_skill_uploads_root(user_id: str) -> Path:
    safe_user = _validate_profile_component(user_id, "user_id")
    return profile_subdir_path(safe_user, "skills") / "uploads"
```

Implement the other three helpers with the same validated user component.

- [ ] **Step 4: Harden restored bearer verification and CORS**

Replace subject-only authorization with:

```python
current_access_token = auth_service.get_current_access_token()
if (
    restored_user_id
    and current_user_id
    and str(restored_user_id) == str(current_user_id)
    and current_access_token
    and secrets.compare_digest(raw_token, current_access_token)
):
    return _build_compat_session_payload(str(current_user_id))
```

Configure `CORSMiddleware` from `client_settings.allowed_origins`. Reject a
non-loopback host during environment initialization unless the explicit opt-in
is set.

- [ ] **Step 5: Run focused tests, lint changed files, and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/test_demo_sidecar_auth.py tests/client_backend/test_skill_upload_security_config.py -q
.\.conda\python.exe -m ruff check client_backend/core/config.py client_backend/core/paths.py client_backend/core/auth.py client_backend/main.py
```

Expected: all pass.

Commit:

```powershell
git add pyproject.toml client_backend/core tests/test_demo_sidecar_auth.py tests/client_backend/test_skill_upload_security_config.py client_backend/main.py
git commit -m "security: harden sidecar skill upload foundations"
```

---

### Task 2: Atomic State, Bounded Locks, and Lifecycle Audit

**Files:**
- Create: `client_backend/services/skill_runtime/state.py`
- Create: `client_backend/services/skill_runtime/locks.py`
- Modify: `client_backend/services/skill_runtime/audit.py`
- Create: `tests/client_backend/test_skill_state_and_locks.py`
- Modify: `tests/client_backend/test_skill_audit.py`

**Interfaces:**
- Produces: `atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None`.
- Produces: `read_json_object(path: Path) -> dict[str, Any] | None`.
- Produces: `profile_lock(user_id: str, scope: str, timeout_seconds: float | None = None) -> AsyncContextManager[None]`.
- Produces: `SkillLockTimeoutError(scope: str)`.
- Produces: `SkillLifecycleAuditWriter.write(*, event: str, user_id: str, device_id: str | None = None, upload_id: str | None = None, operation_id: str | None = None, skill: str | None = None, source_hash: str | None = None, status: str | None = None, phase: str | None = None, error_code: str | None = None, duration_ms: int | None = None, sync_status: str | None = None, metrics: Mapping[str, int] | None = None) -> None` with a fixed non-sensitive field allowlist.
- Consumes: `get_skill_locks_root()` and `client_settings.skill_install_lock_timeout_seconds`.

- [ ] **Step 1: Write failing atomicity and lock tests**

```python
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


@pytest.mark.asyncio
async def test_profile_lock_times_out_for_same_scope(monkeypatch):
    monkeypatch.setattr(client_settings, "skill_install_lock_timeout_seconds", 0.05)
    async with profile_lock("user-a", "skill:demo"):
        with pytest.raises(SkillLockTimeoutError):
            async with profile_lock("user-a", "skill:demo"):
                pass


@pytest.mark.asyncio
async def test_profile_locks_allow_different_scopes():
    async with profile_lock("user-a", "skill:a"):
        async with profile_lock("user-a", "skill:b"):
            pass


def test_lifecycle_audit_never_serializes_paths_or_uploaded_names(tmp_path):
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
    payload = json.loads((tmp_path / "lifecycle.jsonl").read_text())
    assert set(payload) <= LIFECYCLE_AUDIT_FIELDS
    assert "path" not in json.dumps(payload).lower()
    assert "filename" not in json.dumps(payload).lower()
```

- [ ] **Step 2: Run the tests and verify missing-module failures**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_state_and_locks.py tests/client_backend/test_skill_audit.py -q
```

Expected: collection fails because the new modules do not exist.

- [ ] **Step 3: Implement version-safe atomic JSON**

Use a random sibling temporary file, flush and `os.fsync()` before
`os.replace()`, best-effort parent-directory sync on POSIX, restrictive file
mode, and `finally` cleanup:

```python
def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        with contextlib.suppress(OSError):
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
```

`read_json_object` returns `None` for a missing file and raises a typed
`SkillStateError` for malformed or non-object JSON.

- [ ] **Step 4: Implement two-level async/profile locking**

Use a process-local `asyncio.Lock` keyed by the resolved lock path, then acquire
`filelock.FileLock` in `asyncio.to_thread`. Derive filenames from SHA-256 of
the scope rather than placing raw skill/user text into a filename. Release both
locks in `finally`.

- [ ] **Step 5: Implement allowlisted lifecycle auditing**

Add `SkillLifecycleAuditWriter` alongside the execution audit writer. Accept
only `event`, UTC timestamp, user/device-safe IDs, upload/operation IDs, skill
name, source hash, status, phase, error code, duration, sync status, and numeric
byte/file counters. Ignore all other caller keys. Audit failure remains
best-effort and never changes the primary operation result.

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_state_and_locks.py tests/client_backend/test_skill_audit.py -q
.\.conda\python.exe -m ruff check client_backend/services/skill_runtime/state.py client_backend/services/skill_runtime/locks.py client_backend/services/skill_runtime/audit.py tests/client_backend/test_skill_state_and_locks.py tests/client_backend/test_skill_audit.py
```

Expected: all pass.

Commit:

```powershell
git add client_backend/services/skill_runtime/state.py client_backend/services/skill_runtime/locks.py client_backend/services/skill_runtime/audit.py tests/client_backend/test_skill_state_and_locks.py tests/client_backend/test_skill_audit.py
git commit -m "feat: add durable skill coordination and lifecycle audit"
```

---

### Task 3: Hostile ZIP Preflight and Confined Extraction

**Files:**
- Create: `client_backend/services/skill_runtime/archive.py`
- Create: `tests/client_backend/test_skill_archive.py`

**Interfaces:**
- Produces: immutable `SkillArchiveLimits.from_settings()`.
- Produces: immutable `SkillArchiveSummary(compressed_bytes, expanded_bytes, file_count)`.
- Produces: `SkillArchiveValidator.extract(archive_path: Path, destination: Path) -> SkillArchiveSummary`.
- Produces: `SkillArchiveError(code: str, message: str, status_code: int)`.
- Consumes: global upload limit settings and `is_under_root()`.

- [ ] **Step 1: Write the valid archive and resource-limit tests**

```python
def test_extracts_valid_skill_zip_member_by_member(tmp_path):
    archive = _zip(tmp_path / "skill.zip", {
        "demo/SKILL.md": "---\nname: demo\ndescription: Demo\n---\nBody",
        "demo/bin/demo.py": "print('ok')\n",
    })
    destination = tmp_path / "out"

    summary = _validator().extract(archive, destination)

    assert summary.file_count == 2
    assert (destination / "demo" / "SKILL.md").is_file()


@pytest.mark.parametrize(
    ("limit_name", "value", "code"),
    [
        ("max_entries", 1, "SKILL_ARCHIVE_TOO_MANY_FILES"),
        ("max_expanded_bytes", 4, "SKILL_ARCHIVE_TOO_LARGE"),
        ("max_file_bytes", 4, "SKILL_ARCHIVE_TOO_LARGE"),
    ],
)
def test_rejects_resource_limits_before_leaving_output(
    tmp_path, limit_name, value, code
):
    archive = _zip(tmp_path / "skill.zip", {"SKILL.md": "12345"})
    limits = replace(_limits(), **{limit_name: value})

    with pytest.raises(SkillArchiveError) as exc_info:
        SkillArchiveValidator(limits).extract(archive, tmp_path / "out")

    assert exc_info.value.code == code
    assert not (tmp_path / "out").exists()
```

- [ ] **Step 2: Write parameterized path and entry-type attack tests**

Cover these exact names:

```python
UNSAFE_MEMBER_NAMES = [
    "../escape",
    "nested/../../escape",
    "/absolute/SKILL.md",
    r"C:\escape\SKILL.md",
    r"\\server\share\SKILL.md",
    r"nested\..\escape",
    "NUL.txt",
    "demo/trailing. /SKILL.md",
]
```

Use parameterized rejection tests:

```python
@pytest.mark.parametrize(
    ("members", "code"),
    [
        ([("a.txt", b"1"), ("a.txt", b"2")], "SKILL_ARCHIVE_PATH_UNSAFE"),
        ([("A.txt", b"1"), ("a.txt", b"2")], "SKILL_ARCHIVE_PATH_UNSAFE"),
        ([("e\u0301.txt", b"1"), ("\u00e9.txt", b"2")], "SKILL_ARCHIVE_PATH_UNSAFE"),
    ],
)
def test_rejects_duplicate_portable_paths(tmp_path, members, code):
    archive = _zip_entries(tmp_path / "bad.zip", members)
    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(archive, tmp_path / "out")
    assert exc_info.value.code == code
```

Create `ZipInfo` fixtures with encrypted flags, symlink/FIFO/device Unix modes,
unsupported compression IDs, depth 21, and paths over 240 characters. Corrupt
one stored member byte for the CRC test and truncate the final 16 bytes for the
central-directory test. Generate a repeated-byte archive whose declared ratio
is over 200 for the ratio assertion.

- [ ] **Step 3: Run the tests and confirm module-not-found failure**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_archive.py -q
```

Expected: collection fails because `archive.py` is absent.

- [ ] **Step 4: Implement complete preflight**

Normalize each ZIP name to forward-slash components, reject backslashes rather
than reinterpret them, Unicode-normalize and casefold collision keys, inspect
Unix mode from `ZipInfo.external_attr`, and check every declared limit before
creating the destination.

Never trust extension or MIME type as validation. `zipfile.is_zipfile()` and
successful central-directory parsing are required.

- [ ] **Step 5: Implement bounded member streaming**

Create a temporary extraction sibling, open each destination with exclusive
creation, copy in 1 MiB chunks while incrementing per-file and total counters,
and verify the destination remains under the temporary root before opening.
On any exception, delete the temporary extraction root. Promote it to the
requested destination with `os.replace()` only after every member succeeds.

- [ ] **Step 6: Run all archive tests and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_archive.py -q
.\.conda\python.exe -m ruff check client_backend/services/skill_runtime/archive.py tests/client_backend/test_skill_archive.py
```

Expected: all archive cases pass on Windows; POSIX-only mode cases skip only
when the test platform cannot encode the relevant entry type.

Commit:

```powershell
git add client_backend/services/skill_runtime/archive.py tests/client_backend/test_skill_archive.py
git commit -m "feat: validate and confine skill zip extraction"
```

---

### Task 4: User-Scoped Staged Upload Store

**Files:**
- Create: `client_backend/schemas/skill_installation.py`
- Create: `client_backend/services/skill_runtime/uploads.py`
- Create: `tests/client_backend/test_skill_uploads.py`

**Interfaces:**
- Produces: strict `CamelModel` with `alias_generator`, `populate_by_name=True`, and `extra="forbid"`.
- Produces: `SkillUploadRecord` with version, owner, state, timestamps, archive summary, preview, request fingerprint, and operation ID.
- Produces: `SkillInstallationRequest(expected_source_hash, approve_setup, replace_source_hash)`.
- Produces: `SkillUploadService.stage(*, user_id: str, filename: str, stream: AsyncReadable) -> SkillUploadRecord`.
- Produces: `get_owned(user_id, upload_id)`, `delete(user_id, upload_id)`, `claim_for_operation(user_id, upload_id, request_fingerprint, operation_id)`, `mark_succeeded(user_id, upload_id)`, `cleanup_expired(user_id)`, `recover(user_id)`, and `ensure_recovered(user_id)`.
- Consumes: archive validator, installer preview, atomic state, locks, paths, and quotas.
- Consumes: `SkillLifecycleAuditWriter` for accepted/rejected/expired upload events.

- [ ] **Step 1: Write failing upload lifecycle and isolation tests**

```python
@pytest.mark.asyncio
async def test_stage_streams_once_previews_without_setup_and_persists_record(upload_env):
    stream = _AsyncReader(_valid_skill_zip_bytes())

    record = await upload_env.service.stage(
        user_id="user-a",
        filename="../../calendar.zip",
        stream=stream,
    )

    assert record.state == "staged"
    assert record.preview["name"] == "demo"
    assert record.archive.filename == "calendar.zip"
    assert stream.max_requested_chunk <= 1024 * 1024
    assert upload_env.environment.prepared == []
    assert upload_env.service.get_owned("user-a", record.upload_id).upload_id == record.upload_id


def test_foreign_and_expired_uploads_are_indistinguishable(upload_env, frozen_clock):
    record = upload_env.persist_upload(owner="user-a")
    with pytest.raises(SkillUploadNotFoundError):
        upload_env.service.get_owned("user-b", record.upload_id)
    frozen_clock.advance(seconds=1801)
    with pytest.raises(SkillUploadNotFoundError):
        upload_env.service.get_owned("user-a", record.upload_id)
```

Use explicit validation/quota cases:

```python
@pytest.mark.parametrize(
    ("filename", "payload", "code"),
    [
        ("", b"PK", "SKILL_ARCHIVE_TYPE_UNSUPPORTED"),
        ("demo.tar", b"PK", "SKILL_ARCHIVE_TYPE_UNSUPPORTED"),
        ("demo.zip", b"x" * (25 * 1024 * 1024 + 1), "SKILL_ARCHIVE_TOO_LARGE"),
    ],
)
@pytest.mark.asyncio
async def test_rejects_invalid_upload_input_and_removes_partial_stage(
    upload_env, filename, payload, code
):
    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id="user-a",
            filename=filename,
            stream=_AsyncReader(payload),
        )
    assert exc_info.value.code == code
    assert upload_env.stage_directories() == []


@pytest.mark.asyncio
async def test_rate_and_storage_quotas_are_profile_scoped(upload_env):
    upload_env.persist_uploads("user-a", count=5, bytes_each=1024)
    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id="user-a",
            filename="sixth.zip",
            stream=_AsyncReader(_valid_skill_zip_bytes()),
        )
    assert exc_info.value.code == "SKILL_UPLOAD_QUOTA_EXCEEDED"
```

Add concrete state assertions that delete is blocked during `installing`,
expired stages disappear during `ensure_recovered("user-a")`, an `ENOSPC`
write maps to `SKILL_STORAGE_INSUFFICIENT`, and ten accepted/rejected attempts
within 60 seconds make the eleventh return a retryable rate-limit error.

- [ ] **Step 2: Run the tests and verify missing implementation failures**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_uploads.py -q
```

Expected: collection fails for missing upload service/schema.

- [ ] **Step 3: Implement strict shared schemas and persisted upload records**

Use:

```python
def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )
```

Use UTC ISO timestamps in persisted JSON and Pydantic `datetime` fields in
memory. Keep ownership fields out of `model_dump_for_api()`.

- [ ] **Step 4: Implement streaming, quota reservation, preview, and cleanup**

Read at most 1 MiB per call, enforce the running upload limit, flush/fsync the
archive, invoke archive extraction in `asyncio.to_thread`, then call
`SkillBundleInstaller.preview(extracted_root)`. Persist the receipt only after
both extraction and preview succeed. Delete the entire random staging
directory on failure.

Quota checks, the persisted 60-second attempt window, and record transitions
run inside `profile_lock(user_id, "uploads")`. Before extraction, require
available disk space to exceed the configured remaining expansion allowance.
Write normalized lifecycle audit events after accepted, rejected, cancelled,
and expired transitions; audit failure never changes the upload response.

- [ ] **Step 5: Run lifecycle tests and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_uploads.py tests/client_backend/test_skill_archive.py -q
.\.conda\python.exe -m ruff check client_backend/schemas/skill_installation.py client_backend/services/skill_runtime/uploads.py tests/client_backend/test_skill_uploads.py
```

Expected: all pass.

Commit:

```powershell
git add client_backend/schemas/skill_installation.py client_backend/services/skill_runtime/uploads.py tests/client_backend/test_skill_uploads.py
git commit -m "feat: stage user-scoped skill uploads"
```

---

### Task 5: Guarded Installer Replacement and Post-Commit Semantics

**Files:**
- Modify: `client_backend/services/skill_runtime/install.py:70-374`
- Modify: `shared/skills/errors.py:13-27`
- Modify: `tests/client_backend/test_skill_installation.py`

**Interfaces:**
- Produces: `SkillBundleInstaller.install(self, source: str | Path, *, expected_source_hash: str | None = None, approve_setup: bool = False, replace_source_hash: str | None = None, source_kind: Literal["path", "upload"] = "path", observer: SkillInstallObserver | None = None) -> dict`.
- Produces: async `SkillInstallObserver.phase(name: str) -> None` and `SkillInstallObserver.before_commit() -> None` cooperative lifecycle hooks.
- Produces: `SKILL_SOURCE_CHANGED` and `SKILL_CONFIGURED_ROOT_CONFLICT`.
- Produces: idempotent uninstall result with `cleanup_status: "complete" | "pending"`.
- Removes: runtime-bridge synchronization from installer methods.
- Consumes: existing atomic stage/backup promotion and registry refresh.

- [ ] **Step 1: Write failing replacement and provenance tests**

```python
@pytest.mark.asyncio
async def test_existing_profile_skill_requires_matching_replace_hash(install_env):
    first = _write_skill(install_env.sources / "v1", body="v1")
    second = _write_skill(install_env.sources / "v2", body="v2")
    registry, installer = _installer(install_env)
    installed = await installer.install(first)
    old_hash = installed["source_hash"]

    with pytest.raises(SkillRuntimeError) as conflict:
        await installer.install(second)
    assert conflict.value.code == SKILL_INSTALL_CONFLICT

    updated = await installer.install(second, replace_source_hash=old_hash)
    assert updated["action"] == "updated"
    assert registry.get_skill("demo-skill").content == "v2"


@pytest.mark.asyncio
async def test_update_rejects_stale_hash_and_preserves_previous_bundle(install_env):
    first = _write_skill(install_env.sources / "v1", body="v1")
    second = _write_skill(install_env.sources / "v2", body="v2")
    registry, installer = _installer(install_env)
    await installer.install(first)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(second, replace_source_hash="0" * 64)

    assert exc_info.value.code == SKILL_SOURCE_CHANGED
    assert registry.get_skill("demo-skill").content == "v1"


@pytest.mark.asyncio
async def test_upload_provenance_never_persists_staging_path(install_env):
    source = _write_skill(install_env.sources / "upload")
    _, installer = _installer(install_env)
    result = await installer.install(source, source_kind="upload")
    metadata = _read_install_metadata(
        get_installed_skills_root(USER_ID) / result["install_id"]
    )
    assert metadata["source"] == "upload"
    assert "source_path" not in metadata
```

Add these exact cases:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_configured_root_skill_cannot_be_replaced(install_env, enabled):
    configured = _write_skill(install_env.configured / "demo", body="configured")
    registry, installer = _installer(install_env)
    await registry.initialize()
    registry.set_skill_enabled("demo-skill", enabled)
    replacement = _write_skill(install_env.sources / "replacement", body="new")
    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(
            replacement,
            replace_source_hash=registry.get_skill("demo-skill").source_hash,
        )
    assert exc_info.value.code == SKILL_CONFIGURED_ROOT_CONFLICT


@pytest.mark.asyncio
async def test_repeated_uninstall_is_idempotent_and_cleanup_can_be_retried(install_env):
    source = _write_skill(install_env.sources / "demo")
    _, installer = _installer(install_env)
    await installer.install(source)
    first = await installer.uninstall("demo-skill")
    second = await installer.uninstall("demo-skill")
    assert first["removed"] is True
    assert second["removed"] is False
```

Inject environment/secret cleanup failure and assert the bundle is removed,
the result reports `cleanup_status == "pending"`, and a later uninstall/recovery
finishes cleanup. Assert the runtime bridge is never called by
install/setup/uninstall.

- [ ] **Step 2: Run installation tests and confirm failures**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_installation.py -q
```

Expected: replacement signatures/codes are absent and installer still invokes
bridge synchronization.

- [ ] **Step 3: Implement explicit replacement checks under the mutation lock**

After initial discovery, enter `profile_lock(user_id,
f"skill:{source_skill.name}")`, rediscover and rehash the source, initialize
the registry, then apply:

```python
if existing is not None:
    installed = bool((existing.install_metadata or {}).get("installed"))
    if not installed:
        raise SkillRuntimeError(
            SKILL_CONFIGURED_ROOT_CONFLICT,
            f"skill '{source_skill.name}' comes from a configured root and cannot be replaced",
        )
    if replace_source_hash is None:
        raise SkillRuntimeError(
            SKILL_INSTALL_CONFLICT,
            f"a skill named '{source_skill.name}' already exists; confirm an explicit update",
        )
    if replace_source_hash != existing.source_hash:
        raise SkillRuntimeError(
            SKILL_SOURCE_CHANGED,
            "installed skill changed after preview; reload and confirm its current source hash",
        )
elif replace_source_hash is not None:
    raise SkillRuntimeError(
        SKILL_SOURCE_CHANGED,
        "no installed skill matches the requested replacement; reload the catalog",
    )
```

Preserve the old installed bundle until the new copied bundle and runtime are
ready. Return `action: "installed"` or `"updated"`.

Emit lifecycle phases around the existing boundaries. Immediately before the
first atomic promotion, call `await observer.before_commit()`; an operation
observer uses this hook to either raise its normalized cancellation exception
or durably record `commitStartedAt`. Path-based callers pass no observer.

Before removing an installed bundle, persist a cleanup receipt under the
user's skill operations root. Mark bundle, runtime, and secret cleanup
independently and remove the receipt only when all are complete. A repeated
uninstall resumes that receipt; when neither a bundle nor receipt exists it
returns `removed: false, cleanup_status: "complete"`.

- [ ] **Step 4: Remove bridge synchronization from installer commit paths**

Delete `_refresh_runtime_bridge_catalogs_if_connected()` and its calls. Keep
the registry refresh because existing installer callers require immediate
local visibility. Catalog synchronization moves to `SkillCatalogService`.

- [ ] **Step 5: Run installation and registry tests and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_installation.py tests/client_backend/test_skills_registry.py -q
.\.conda\python.exe -m ruff check client_backend/services/skill_runtime/install.py shared/skills/errors.py tests/client_backend/test_skill_installation.py
```

Expected: all pass.

Commit:

```powershell
git add client_backend/services/skill_runtime/install.py shared/skills/errors.py tests/client_backend/test_skill_installation.py
git commit -m "feat: guard skill bundle replacements by source hash"
```

---

### Task 6: Generation-Tagged Fresh Catalog Service

**Files:**
- Create: `client_backend/services/skill_catalog.py`
- Modify: `client_backend/services/local_skills_registry.py:176-220,617-670`
- Modify: `client_backend/services/runtime_bridge.py:100-120,223-251`
- Create: `tests/client_backend/test_skill_catalog.py`
- Modify: `tests/client_backend/test_runtime_bridge.py`

**Interfaces:**
- Produces: `SkillCatalogService.snapshot(force: bool = False, sync: bool = False) -> dict[str, Any]`.
- Produces: `SkillCatalogService.after_mutation(sync: bool = True) -> dict[str, Any]`.
- Produces: `SkillCatalogService.retry_pending_sync() -> dict[str, Any]`.
- Produces: `get_skill_catalog_service()` and `close_skill_catalog_service()`.
- Consumes: registry, readiness manager, catalog state path, atomic JSON, and runtime bridge.

- [ ] **Step 1: Write failing freshness, generation, and sync-degradation tests**

```python
@pytest.mark.asyncio
async def test_catalog_generation_changes_only_when_projection_changes(catalog_env):
    first = await catalog_env.service.snapshot(force=True)
    second = await catalog_env.service.snapshot(force=True)
    catalog_env.add_skill("new-skill")
    third = await catalog_env.service.snapshot(force=True)

    assert second["catalogGeneration"] == first["catalogGeneration"]
    assert third["catalogGeneration"] == first["catalogGeneration"] + 1


@pytest.mark.asyncio
async def test_concurrent_freshness_checks_share_one_scan(catalog_env):
    await asyncio.gather(*(catalog_env.service.snapshot(force=False) for _ in range(10)))
    assert catalog_env.registry.refresh_calls == 1


@pytest.mark.asyncio
async def test_sync_failure_returns_committed_catalog_as_pending(catalog_env):
    catalog_env.bridge.raise_on_refresh = RuntimeError("offline")
    snapshot = await catalog_env.service.after_mutation(sync=True)

    assert snapshot["catalogSyncStatus"] == "pending"
    assert snapshot["skills"]
```

Add restart persistence, TTL bypass, force reload, disconnected bridge, and
lower-level bridge concurrent-refresh serialization tests.

- [ ] **Step 2: Run tests and verify missing service failures**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_catalog.py tests/client_backend/test_runtime_bridge.py -q
```

Expected: collection/service failures.

- [ ] **Step 3: Implement deterministic projection and persistent generation**

Move `_skill_summary` and list payload construction from the API into the
catalog service. Serialize the projection with sorted keys and sorted skills,
hash it with SHA-256, compare it to persisted `projectionHash`, and increment
the persisted integer generation only on change.

The persisted state shape is:

```json
{
  "version": 1,
  "generation": 42,
  "projectionHash": "sha256",
  "syncStatus": "pending"
}
```

Use a single async refresh lock and a monotonic freshness deadline. Persist
state while holding the profile catalog lock.

- [ ] **Step 4: Decouple and serialize runtime synchronization**

Add `_catalog_refresh_lock = asyncio.Lock()` to the bridge and wrap
`refresh_catalogs()`. The catalog service catches bridge/network exceptions,
persists `pending`, and returns the local snapshot. On success it persists
`synced`; when disconnected it returns `disconnected`.

- [ ] **Step 5: Run catalog/runtime tests and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_catalog.py tests/client_backend/test_runtime_bridge.py tests/client_backend/test_skills_registry.py -q
.\.conda\python.exe -m ruff check client_backend/services/skill_catalog.py client_backend/services/local_skills_registry.py client_backend/services/runtime_bridge.py tests/client_backend/test_skill_catalog.py
```

Expected: all pass.

Commit:

```powershell
git add client_backend/services/skill_catalog.py client_backend/services/local_skills_registry.py client_backend/services/runtime_bridge.py tests/client_backend/test_skill_catalog.py tests/client_backend/test_runtime_bridge.py
git commit -m "feat: add fresh generation-tagged skill catalogs"
```

---

### Task 7: Persisted Asynchronous Installation Operations

**Files:**
- Create: `client_backend/services/skill_runtime/operations.py`
- Create: `tests/client_backend/test_skill_operations.py`

**Interfaces:**
- Produces: `SkillInstallationOperation` with state, phase, timestamps, result, and normalized failure.
- Produces: `SkillInstallationService.start(user_id, upload_id, request) -> SkillInstallationOperation`.
- Produces: `get_owned(user_id, operation_id)`, `cancel(user_id, operation_id)`, `recover(user_id)`, `ensure_recovered(user_id)`, `cleanup_expired(user_id)`, and `shutdown()`.
- Consumes: `SkillInstallationRequest` from Task 4, upload service, installer lifecycle observer, catalog service, atomic state, and upload/operation locks.
- Consumes: `SkillLifecycleAuditWriter` from Task 2 for non-sensitive state-transition audit records.

- [ ] **Step 1: Write failing transition and idempotency tests**

```python
@pytest.mark.asyncio
async def test_start_is_idempotent_for_identical_request(operation_env):
    upload = operation_env.staged_upload()
    request = _request(expected_source_hash=upload.preview["source_hash"])

    first = await operation_env.service.start("user-a", upload.upload_id, request)
    second = await operation_env.service.start("user-a", upload.upload_id, request)

    assert second.operation_id == first.operation_id
    assert operation_env.installer.install_calls == 1


@pytest.mark.asyncio
async def test_start_rejects_different_request_for_claimed_upload(operation_env):
    upload = operation_env.staged_upload()
    await operation_env.service.start("user-a", upload.upload_id, _request())

    with pytest.raises(SkillOperationConflictError):
        await operation_env.service.start(
            "user-a",
            upload.upload_id,
            _request(approve_setup=True),
        )


@pytest.mark.asyncio
async def test_sync_failure_keeps_operation_succeeded(operation_env):
    operation_env.catalog.sync_status = "pending"
    operation = await operation_env.start_and_wait()

    assert operation.state == "succeeded"
    assert operation.result["catalog"]["catalogSyncStatus"] == "pending"
```

- [ ] **Step 2: Write cancellation, recovery, and failure-preservation tests**

Implement the cancellation matrix as a parameterized test:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "commit_started", "expected"),
    [
        ("pending", False, "cancelled"),
        ("running", False, "cancelled"),
        ("running", True, "SKILL_OPERATION_COMMITTED"),
    ],
)
async def test_cancel_respects_atomic_commit_boundary(
    operation_env, state, commit_started, expected
):
    operation = operation_env.persist_operation(
        state=state,
        commit_started=commit_started,
    )
    if expected == "SKILL_OPERATION_COMMITTED":
        with pytest.raises(SkillOperationError) as exc_info:
            await operation_env.service.cancel("user-a", operation.operation_id)
        assert exc_info.value.code == expected
    else:
        cancelled = await operation_env.service.cancel(
            "user-a", operation.operation_id
        )
        assert cancelled.state == expected
```

Exercise both recovery branches:

```python
@pytest.mark.asyncio
async def test_recovery_reconciles_commit_and_fails_precommit(operation_env):
    committed = operation_env.persist_operation(
        state="running",
        source_hash="a" * 64,
        commit_started=True,
    )
    operation_env.persist_installed_bundle(source_hash="a" * 64)
    precommit = operation_env.persist_operation(
        state="running",
        source_hash="b" * 64,
        commit_started=False,
    )

    await operation_env.service.recover("user-a")

    assert operation_env.get(committed).state == "succeeded"
    assert operation_env.get(precommit).state == "failed"
    assert operation_env.get(precommit).failure["retryable"] is True
```

Also assert successful install deletes ZIP/extracted bytes but retains the
receipt, terminal receipts expire after 60 minutes, and foreign lookup is
not-found.

- [ ] **Step 3: Run tests and confirm missing implementation**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_operations.py -q
```

Expected: collection fails because `operations.py` is absent.

- [ ] **Step 4: Implement the persisted state machine**

Persist the operation before creating its task. Use exact states
`pending/running/succeeded/failed/cancelled` and exact phases
`validating/waitingForLock/copying/preparingRuntime/committing/refreshingCatalog/syncingCatalog`.

Fingerprint normalized request JSON:

```python
fingerprint = hashlib.sha256(
    json.dumps(
        request.model_dump(mode="json", by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
```

The worker rechecks upload owner, expiry, source hash, and active user before
calling the installer. The installer owns the per-skill mutation lock; the
operation service must not acquire that same scope recursively. Store
background tasks by operation ID and consume their exceptions in a done
callback.

Write lifecycle events for queued, running phase changes, cancellation,
terminal result, recovery, cleanup, and catalog-sync status. Pass only
normalized IDs, hashes, byte/file counts, phase, duration, sync state, and
error code to the Task 2 writer.

- [ ] **Step 5: Implement cancellation and recovery boundaries**

Pass an observer whose `phase()` persists each phase and whose
`before_commit()` checks the cancellation flag before persisting
`commit_started_at`. Cancellation after that field is non-cancellable.
Recovery compares operation source hash to installed metadata to distinguish a
committed result from a pre-commit interruption.

- [ ] **Step 6: Run operation/upload/installer tests and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_operations.py tests/client_backend/test_skill_uploads.py tests/client_backend/test_skill_installation.py tests/client_backend/test_skill_audit.py -q
.\.conda\python.exe -m ruff check client_backend/services/skill_runtime/operations.py tests/client_backend/test_skill_operations.py
```

Expected: all pass.

Commit:

```powershell
git add client_backend/services/skill_runtime/operations.py tests/client_backend/test_skill_operations.py
git commit -m "feat: persist asynchronous skill installations"
```

---

### Task 8: Stable Error Envelopes and ZIP/Operation API

**Files:**
- Create: `client_backend/api/skill_errors.py`
- Modify: `client_backend/api/common.py:83-100`
- Modify: `client_backend/api/skills.py:1-330`
- Modify: `client_backend/main.py:39-90`
- Create: `tests/client_backend/test_skill_upload_api.py`
- Modify: `tests/client_backend/test_skills_api.py`

**Interfaces:**
- Produces: `register_skill_exception_handlers(app: FastAPI) -> None`.
- Produces: multipart `POST /skills/uploads`.
- Produces: `DELETE /skills/uploads/{upload_id}`.
- Produces: `POST /skills/uploads/{upload_id}/install`.
- Produces: `GET/DELETE /skills/installations/{operation_id}`.
- Consumes: upload/operation/catalog singletons and strict schemas.

- [ ] **Step 1: Write failing multipart and operation contract tests**

```python
def test_stage_upload_returns_201_camel_case_preview(api_env):
    response = api_env.client.post(
        "/skills/uploads",
        files={"file": ("demo.zip", _valid_skill_zip_bytes(), "application/zip")},
        headers=api_env.auth_headers,
    )

    assert response.status_code == 201
    data = response.json()["data"]
    assert data["uploadId"]
    assert data["preview"]["sourceHash"]
    assert "source_hash" not in data["preview"]


def test_start_install_returns_202_and_status_url(api_env):
    upload = api_env.stage()
    response = api_env.client.post(
        f"/skills/uploads/{upload['uploadId']}/install",
        json={
            "expectedSourceHash": upload["preview"]["sourceHash"],
            "approveSetup": False,
        },
        headers=api_env.auth_headers,
    )

    assert response.status_code == 202
    assert response.json()["data"]["statusUrl"].startswith("/skills/installations/")
```

Add snake_case compatibility, unknown-field 422, non-ZIP 415, limit 413,
foreign/expired 404, conflict 409, lock 423, storage 507, cancellation, failed
operation polling with HTTP 200, missing auth envelope, and `/api/skills`
parity tests.

- [ ] **Step 2: Run API tests and verify route failures**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_upload_api.py tests/client_backend/test_skills_api.py -q
```

Expected: new routes return 404 and legacy payload assertions lack new fields.

- [ ] **Step 3: Implement stable skill route error mapping**

Extend `make_api_response` with optional `code`. Register global
`HTTPException` and `RequestValidationError` handlers that delegate to
FastAPI's default handlers unless `request.url.path` starts with `/skills/`,
equals `/skills`, or starts with `/api/skills`. Map only safe normalized fields
into the skill envelope.

- [ ] **Step 4: Add static upload routes before `/{name}`**

Declare `/uploads`, `/uploads/{upload_id}`, and `/installations/{operation_id}`
routes before the dynamic skill-name route. Use `UploadFile = File(...)` and
pass the authenticated session user into the upload service. Always close the
`UploadFile` in `finally`.

- [ ] **Step 5: Wire recovery and shutdown into lifespan**

Add `ensure_recovered(user_id)` to the upload and operation services and call
it once, under a user-scoped recovery lock, on the first authenticated skill
request for that profile. This avoids assuming that a user session exists
during process lifespan startup. On shutdown call operation-service
`shutdown()` before stopping the runtime bridge.

- [ ] **Step 6: Run API suites and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_upload_api.py tests/client_backend/test_skills_api.py tests/client_backend/test_skill_operations.py -q
.\.conda\python.exe -m ruff check client_backend/api/skill_errors.py client_backend/api/skills.py client_backend/api/common.py client_backend/main.py tests/client_backend/test_skill_upload_api.py
```

Expected: all pass.

Commit:

```powershell
git add client_backend/api/skill_errors.py client_backend/api/common.py client_backend/api/skills.py client_backend/main.py tests/client_backend/test_skill_upload_api.py tests/client_backend/test_skills_api.py
git commit -m "feat: expose sidecar skill upload operations"
```

---

### Task 9: Refactor Existing Skill Mutations onto the Catalog Contract

**Files:**
- Modify: `client_backend/schemas/skills.py:1-102`
- Modify: `client_backend/api/skills.py:80-330`
- Modify: `tests/client_backend/test_skills_api.py`
- Modify: `tests/client_backend/test_skill_installation.py`

**Interfaces:**
- Produces: catalog payload from list, reload, toggle, setup, uninstall, and path install.
- Produces: alias-compatible `SkillInstallRequest`, `SkillInstallPreviewRequest`, `SkillSetupRequest`, and `SkillUninstallRequest`.
- Consumes: `SkillCatalogService.snapshot()` and `.after_mutation()`.

- [ ] **Step 1: Rewrite route tests to assert one catalog shape**

```python
def _assert_catalog(data):
    assert data["deviceId"] == "device-123"
    assert isinstance(data["catalogGeneration"], int)
    assert data["catalogSyncStatus"] in {"synced", "pending", "disconnected"}
    assert data["totalCount"] == len(data["skills"])


def test_reload_returns_refreshed_catalog(skills_client):
    response = skills_client.post("/skills/reload")
    assert response.status_code == 200
    _assert_catalog(response.json()["data"])


def test_toggle_returns_refreshed_catalog(skills_client):
    response = skills_client.patch("/skills/demo/toggle?enabled=false")
    assert response.status_code == 200
    _assert_catalog(response.json()["data"]["catalog"])
```

Add equivalent assertions for path install, setup, and uninstall. Assert a
bridge sync failure produces success plus `pending`.

- [ ] **Step 2: Run focused route tests and verify old payload failures**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skills_api.py tests/client_backend/test_skill_installation.py -q
```

Expected: old message-only mutation responses fail catalog assertions.

- [ ] **Step 3: Apply strict aliases to existing request models**

Make existing models inherit `CamelModel`. Retain Python snake_case attributes
while accepting and serializing documented camelCase fields.

- [ ] **Step 4: Route every read/mutation through SkillCatalogService**

Use:

```python
catalog = await get_skill_catalog_service().snapshot(force=True, sync=True)
```

for reload, and:

```python
catalog = await get_skill_catalog_service().after_mutation(sync=True)
```

after install/setup/uninstall/toggle. `GET /skills` uses bounded freshness;
detail lookup calls snapshot before registry lookup.

- [ ] **Step 5: Run all sidecar skill tests and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skills_api.py tests/client_backend/test_skill_installation.py tests/client_backend/test_skill_catalog.py tests/client_backend/test_skills_registry.py -q
.\.conda\python.exe -m ruff check client_backend/schemas/skills.py client_backend/api/skills.py tests/client_backend/test_skills_api.py
```

Expected: all pass.

Commit:

```powershell
git add client_backend/schemas/skills.py client_backend/api/skills.py tests/client_backend/test_skills_api.py tests/client_backend/test_skill_installation.py
git commit -m "refactor: unify sidecar skill catalog responses"
```

---

### Task 10: Streamlit Multipart Transport and Safe Session State

**Files:**
- Modify: `demo.py:3089-3175,4464-4485,4884-4963`
- Create: `tests/test_demo_skill_installation.py`
- Modify: `tests/test_demo_sidecar_auth.py`

**Interfaces:**
- Produces: `make_api_multipart_request(endpoint, *, files, form=None) -> dict`.
- Produces: `stage_skill_zip(filename, content, content_type)`, `start_skill_install(upload_id, expected_source_hash, approve_setup, replace_source_hash=None)`, `get_skill_installation(operation_id)`, `cancel_skill_upload(upload_id)`, and `cancel_skill_installation(operation_id)`.
- Produces: `_clear_skill_installation_session_state(cleanup_remote: bool)`.
- Consumes: existing authenticated Requests session and response-error extraction.

- [ ] **Step 1: Write failing multipart transport tests**

```python
def test_stage_skill_zip_uses_authenticated_multipart_without_json(demo_module, monkeypatch):
    session = _RecordingSession(_success_upload_response())
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)
    demo_module.st.session_state.auth_token = "local-token"

    result = demo_module.stage_skill_zip(
        "demo.zip",
        _valid_skill_zip_bytes(),
        "application/zip",
    )

    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/skills/uploads")
    assert call["files"]["file"][0] == "demo.zip"
    assert "json" not in call or call["json"] is None
    assert call["headers"]["Authorization"] == "Bearer local-token"
    assert result["uploadId"]
```

Use explicit transport/session assertions:

```python
@pytest.mark.parametrize("status_code", [413, 415])
def test_upload_errors_use_shared_safe_message(demo_module, monkeypatch, status_code):
    session = _RecordingSession(_error_response(status_code, "safe message"))
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)
    assert demo_module.stage_skill_zip("demo.zip", b"PK", "application/zip") is None
    assert demo_module.st.session_state["_last_api_error_message"] == "safe message"


def test_logout_discards_ids_and_bytes_before_remote_cleanup(demo_module, monkeypatch):
    demo_module.st.session_state.skill_upload_id = "upload-a"
    demo_module.st.session_state.skill_upload_bytes = b"must-not-persist"
    deleted = []
    monkeypatch.setattr(
        demo_module, "cancel_skill_upload", lambda upload_id: deleted.append(upload_id)
    )
    demo_module._clear_skill_installation_session_state(cleanup_remote=True)
    assert "skill_upload_id" not in demo_module.st.session_state
    assert "skill_upload_bytes" not in demo_module.st.session_state
    assert deleted == ["upload-a"]
```

Assert operation helpers URL-encode IDs and successful multipart/mutation
requests increment `api_cache_version`.

- [ ] **Step 2: Run tests and verify helper failures**

Run:

```powershell
.\.conda\python.exe -m pytest tests/test_demo_skill_installation.py tests/test_demo_sidecar_auth.py -q
```

Expected: missing helper failures.

- [ ] **Step 3: Implement dedicated multipart transport**

Do not overload `make_api_request` with ambiguous simultaneous JSON/files.
Reuse auth/error/cache invalidation behavior in a focused helper:

```python
def make_api_multipart_request(
    endpoint: str,
    *,
    files: dict[str, tuple[str, bytes, str]],
    form: dict[str, str] | None = None,
) -> dict:
    response = get_http_session().post(
        f"{API_BASE_URL}{endpoint}",
        files=files,
        data=form or {},
        headers=_auth_headers(),
        timeout=STREAM_REQUEST_TIMEOUT,
    )
    return _handle_api_response(response, mutation=True)
```

Factor shared response handling instead of copying authentication transitions
and toast behavior.

- [ ] **Step 4: Implement upload/operation helpers and safe cleanup**

URL-encode upload/operation/skill identifiers. Store only safe preview and
opaque IDs. Clear keys on logout and user transition. Remote cleanup is
best-effort and must not block clearing local state.

- [ ] **Step 5: Run transport/session tests and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/test_demo_skill_installation.py tests/test_demo_sidecar_auth.py tests/test_hitl_demo_panel.py -q
.\.conda\python.exe -m ruff check demo.py tests/test_demo_skill_installation.py
```

Expected: all pass.

Commit:

```powershell
git add demo.py tests/test_demo_skill_installation.py tests/test_demo_sidecar_auth.py
git commit -m "feat: add Streamlit skill upload transport"
```

---

### Task 11: Streamlit Preview, Approval, Polling, and Catalog UI

**Files:**
- Modify: `demo.py:8691-8740`
- Modify: `tests/test_demo_skill_installation.py`
- Modify: `tests/test_skills_architecture.py`

**Interfaces:**
- Produces: `render_skill_install_panel()`.
- Produces: non-blocking operation polling driven by a Streamlit fragment and rerun timestamps.
- Consumes: Task 10 helpers and terminal operation catalog.

- [ ] **Step 1: Write failing render-contract tests**

Use the repository's established source/AST helper style to assert:

```python
def test_streamlit_skill_panel_has_zip_preview_and_explicit_approvals():
    source = Path("demo.py").read_text(encoding="utf-8")
    assert 'st.file_uploader(' in source
    assert 'type=["zip"]' in source
    assert "Install only skills you trust" in source
    assert "approve_setup" in source
    assert "replace_source_hash" in source


def test_streamlit_polling_does_not_block_in_a_while_loop():
    tree = ast.parse(Path("demo.py").read_text(encoding="utf-8"))
    polling = _function_node(tree, "_poll_skill_installation")
    assert not any(isinstance(node, ast.While) for node in ast.walk(polling))
```

Add mocked state-transition tests for staged preview, explicit setup approval,
configured-root non-replaceable conflict, guarded update, running operation,
terminal success, terminal failure, pending catalog sync, cancellation after
commit, and selecting a new ZIP.

- [ ] **Step 2: Run UI tests and confirm missing panel/state failures**

Run:

```powershell
.\.conda\python.exe -m pytest tests/test_demo_skill_installation.py tests/test_skills_architecture.py -q
```

Expected: missing uploader/polling contract failures.

- [ ] **Step 3: Implement preview and confirmation UI**

Render filename, expanded size, file count, commands/scripts, dependencies,
build requirements, and existing skill state. Require separate unchecked
checkboxes for setup and replacement. Never render raw HTML from archive
metadata.

- [ ] **Step 4: Implement rerun-based bounded polling**

Wrap the status widget in `@st.fragment(run_every=0.5)`. Store `next_poll_at`
and `poll_delay_seconds`; use delays `0.5, 1.0, 2.0, 3.0`, capped at 3 seconds.
Each fragment rerun performs at most one status request and skips the request
until `next_poll_at`. Stop the fragment's polling behavior on terminal state
or expiry.

On success, apply `result.catalog`, increment the API cache version, and show
distinct copy for `synced`, `pending`, and `disconnected`.

- [ ] **Step 5: Run Streamlit/architecture tests and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/test_demo_skill_installation.py tests/test_skills_architecture.py tests/test_demo_refactor_contract.py tests/test_hitl_demo_panel.py -q
.\.conda\python.exe -m ruff check demo.py tests/test_demo_skill_installation.py tests/test_skills_architecture.py
```

Expected: all pass.

Commit:

```powershell
git add demo.py tests/test_demo_skill_installation.py tests/test_skills_architecture.py
git commit -m "feat: add Streamlit skill installation workflow"
```

---

### Task 12: Device-Bound AI SDK Chat Integration

**Files:**
- Create: `tests/test_skill_installation_chat_integration.py`
- Modify: `tests/client_backend/test_runtime_bridge.py`
- Modify: `tests/test_skills_tool.py`

**Interfaces:**
- Verifies: uploaded skill projection reaches the connected device catalog.
- Verifies: canonical resolution is user/device scoped.
- Verifies: AI SDK chat request device context remains unchanged.
- Consumes: real upload/archive/installer/catalog code with stubbed server transport.

- [ ] **Step 1: Write an end-to-end fixture ZIP integration test**

```python
@pytest.mark.asyncio
async def test_uploaded_skill_reaches_only_originating_device_chat(
    tmp_path, sidecar_profile, monkeypatch
):
    archive = _fixture_zip("tests/fixtures/skills/google_calendar")
    upload = await upload_service.stage(
        user_id="user-a",
        filename="google-calendar.zip",
        stream=_AsyncReader(archive),
    )
    operation = await installation_service.start(
        "user-a",
        upload.upload_id,
        SkillInstallationRequest(
            expected_source_hash=upload.preview["source_hash"],
            approve_setup=False,
        ),
    )
    operation = await _wait_terminal(operation.operation_id)
    assert operation.state == "succeeded"

    origin = list_resolved_skills(user_id="user-a", device_id="device-a")
    other = list_resolved_skills(user_id="user-a", device_id="device-b")
    unbound = list_resolved_skills(user_id="user-a", device_id=None)

    assert {skill.name for skill in origin} == {"google-calendar"}
    assert other == []
    assert unbound == []
```

Use the real catalog projection and capture
`update_device_skill_catalog(device_id="device-a", catalog=captured_catalog)`. Feed that
catalog into the canonical `ClientDeviceService` stub used by
`list_resolved_skills`.

- [ ] **Step 2: Add AI SDK request device-context regression**

Use the sidecar proxy test client and capture the upstream JSON:

```python
def test_ai_sdk_chat_keeps_stream_contract_and_stamps_local_device(
    sidecar_client, upstream_capture, connected_bridge
):
    response = sidecar_client.post(
        "/api/chat/conversation-a",
        json={
            "messages": [{"role": "user", "content": "Use calendar"}],
            "deviceId": "foreign-device",
            "inlineRichResponseV1": True,
        },
    )

    assert upstream_capture.json["device_id"] == connected_bridge.device_id
    assert "deviceId" not in upstream_capture.json
    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
    assert response.text.rstrip().endswith("data: [DONE]")
```

- [ ] **Step 3: Run integration and existing device-isolation tests**

Run:

```powershell
.\.conda\python.exe -m pytest tests/test_skill_installation_chat_integration.py tests/test_skill_device_isolation.py tests/test_skills_tool.py tests/client_backend/test_runtime_bridge.py tests/test_image_stream_http_contract.py -q
```

Expected: all pass because Tasks 4-9 already completed the production wiring.

- [ ] **Step 4: Run focused AI SDK regressions and commit**

Run:

```powershell
.\.conda\python.exe -m pytest tests/test_skill_installation_chat_integration.py tests/test_skill_device_isolation.py tests/test_skills_tool.py tests/test_image_stream_http_contract.py tests/client_backend/test_image_stream_proxy.py -q
```

Expected: all pass.

Commit:

```powershell
git add tests/test_skill_installation_chat_integration.py tests/client_backend/test_runtime_bridge.py tests/test_skills_tool.py
git commit -m "test: verify installed skills in device-bound AI SDK chat"
```

---

### Task 13: Documentation, Contract Synchronization, and Complete Verification

**Files:**
- Modify: `README.md:798-838,1177-1178`
- Modify: `docs/skill-runtime.md`
- Modify: `plans/SKILLS_MCP_HITL_FE_CONTRACT.md`
- Modify: `plans/SKILL_INSTALLATION_FE_CONTRACT.md`
- Modify: `tests/test_production_readiness_contract.py`
- Modify: `tests/test_skills_architecture.py`

**Interfaces:**
- Produces: documentation that matches the implemented OpenAPI schema and state machine.
- Verifies: no server-owned skill upload route and no Streamlit/path-install regression.
- Consumes: all prior implementation tasks.

- [ ] **Step 1: Add failing documentation/architecture assertions**

```python
def test_readme_documents_zip_upload_operation_flow():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "POST /skills/uploads" in readme
    assert "GET /skills/installations/{operationId}" in readme
    assert "replaceSourceHash" in readme


def test_canonical_server_still_exposes_no_skill_upload_router():
    source = Path("app/main.py").read_text(encoding="utf-8")
    assert "skill_upload" not in source
    assert "skills_router" not in source
```

Add a contract test that every documented error code appears in the
implementation error registry and every operation phase appears in the
Pydantic enum.

- [ ] **Step 2: Run documentation contract tests and confirm failures**

Run:

```powershell
.\.conda\python.exe -m pytest tests/test_production_readiness_contract.py tests/test_skills_architecture.py -q
```

Expected: missing new endpoint/error documentation assertions fail.

- [ ] **Step 3: Update all documentation from implemented schemas**

Document:

- ZIP-only upload and configured limits;
- upload/operation lifecycle;
- guarded updates;
- catalog generation/sync states;
- Streamlit and AI SDK cache behavior;
- sidecar-only ownership;
- executable-code trust boundary;
- normalized errors and recovery.

Compare every JSON example in `plans/SKILL_INSTALLATION_FE_CONTRACT.md` against
`model_dump(mode="json", by_alias=True)` output from the final models. Remove or
correct any field that implementation does not emit.

- [ ] **Step 4: Run focused functional suites**

Run:

```powershell
.\.conda\python.exe -m pytest tests/client_backend/test_skill_archive.py tests/client_backend/test_skill_uploads.py tests/client_backend/test_skill_operations.py tests/client_backend/test_skill_upload_api.py tests/client_backend/test_skill_catalog.py tests/client_backend/test_skill_installation.py tests/client_backend/test_skills_api.py tests/test_demo_skill_installation.py tests/test_skill_installation_chat_integration.py -q
```

Expected: all pass.

- [ ] **Step 5: Run security, Streamlit, and AI SDK regression suites**

Run:

```powershell
.\.conda\python.exe -m pytest tests/test_demo_sidecar_auth.py tests/test_skills_architecture.py tests/test_skill_device_isolation.py tests/test_skills_tool.py tests/test_hitl_demo_panel.py tests/test_image_stream_http_contract.py tests/client_backend/test_image_stream_proxy.py tests/client_backend/test_runtime_bridge.py -q
```

Expected: all pass.

- [ ] **Step 6: Run lint, diff checks, and the full test suite**

Run:

```powershell
.\.conda\python.exe -m ruff check client_backend demo.py tests/client_backend tests/test_demo_skill_installation.py tests/test_skill_installation_chat_integration.py
git diff --check
.\.conda\python.exe -m pytest -q
```

Expected: Ruff clean, no whitespace errors, and the full suite passes.

- [ ] **Step 7: Commit final documentation and contract alignment**

```powershell
git add README.md docs/skill-runtime.md plans/SKILLS_MCP_HITL_FE_CONTRACT.md plans/SKILL_INSTALLATION_FE_CONTRACT.md tests/test_production_readiness_contract.py tests/test_skills_architecture.py
git commit -m "docs: publish sidecar skill installation contract"
```

## Final Review Checklist

- [ ] A valid ZIP is uploaded once, previewed without setup, and installed
      through a persisted operation.
- [ ] Every archive/path/quota attack in Task 3 fails before data escapes or
      limits are exceeded.
- [ ] A forged restored bearer cannot authorize skill management.
- [ ] New install and explicit hash-guarded update semantics match the FE
      contract.
- [ ] Configured-root skills cannot be overwritten.
- [ ] Replacement failure preserves the old bundle and runtime.
- [ ] Operation retry, cancellation, expiry, and restart recovery are
      deterministic.
- [ ] Local commit remains success when bridge synchronization is pending.
- [ ] Catalog generation survives restart and prevents stale cache replacement.
- [ ] Streamlit uses multipart and non-blocking polling without retaining ZIP
      bytes.
- [ ] AI SDK chat remains wire-compatible and resolves the skill only for the
      originating device.
- [ ] `/skills` and `/api/skills` remain equivalent.
- [ ] Documentation examples match actual camelCase serialized models.
- [ ] Focused tests, security regressions, Ruff, diff checks, and the full test
      suite pass.
