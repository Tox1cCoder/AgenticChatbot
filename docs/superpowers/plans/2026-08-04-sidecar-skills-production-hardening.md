# Sidecar and Skills Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Windows sidecar bundle self-repairing and make modern skill-resource and collection installation behavior match its documented production guarantees.

**Architecture:** The bundle builders will consume one canonical set of handoff templates, whose launcher validates Python, repairs pip, fingerprints the environment, and propagates failures. Skill discovery will carry an exact document path separately from its bundle root, resource access will use one cached confinement policy, and all installs will run through a journaled collection transaction that stages first, retains replacement backups, serializes catalog reads, and deterministically rolls back or finalizes after a crash.

**Tech Stack:** Python 3.10+, PowerShell 5.1/7, FastAPI, Pydantic v2, asyncio, filelock, pytest, Ruff.

## Global Constraints

- Preserve existing HTTP response shapes, device binding, per-skill enablement, hash-bound approval, encrypted secrets, and upload-only installation.
- Support Windows PowerShell 5.1 and current PowerShell 7 without optional PowerShell modules.
- Accept any working Python 3.10 or newer; do not require exactly Python 3.10.
- Never overwrite or delete the user's sidecar configuration while repairing the bundle-owned `.venv`.
- Treat `allowed-tools` as experimental pre-approval metadata, not a restrictive allowlist.
- Do not implement Phase 2 namespacing, remote fetching, or catalog search in this change.
- Do not remove a documented compatibility API without proof that supported consumers no longer use it.
- Follow red-green-refactor for every behavior change and preserve the user's untracked `superpowers-main.zip`.

---

## File Structure

**Create:**

- `scripts/client-backend-bundle/requirements-client.txt` — canonical sidecar dependencies.
- `scripts/client-backend-bundle/start-client-backend.ps1` — canonical Windows bootstrap.
- `scripts/client-backend-bundle/start-client-backend.bat` — canonical batch entrypoint.
- `scripts/client-backend-bundle/README.client_backend.md` — canonical handoff instructions.
- `client_backend/services/skill_runtime/transactions.py` — install transaction models, journal, commit, rollback, and recovery.
- `tests/client_backend/test_skill_transactions.py` — real transaction and crash-recovery coverage.

**Modify:**

- `scripts/build_client_backend_bundle.py` — copy canonical handoff files.
- `scripts/build-client-backend-bundle.ps1` — copy the same handoff files.
- `tests/test_client_backend_bundle.py` — replace source-grep assertions with launcher integration tests.
- `client_backend/services/skill_runtime/resources.py` — one listing/reading policy and manifest cache.
- `tests/client_backend/test_skill_resources.py` — metadata, links, size, and text regressions.
- `client_backend/services/skill_runtime/collection.py` — explicit discovered-skill model.
- `client_backend/schemas/skill_installation.py` — safe persisted member descriptors and full transaction recovery data.
- `client_backend/services/skill_runtime/uploads.py` — persist and resolve exactly previewed members.
- `tests/client_backend/test_skill_collection.py` and `tests/client_backend/test_skill_uploads.py` — nested and persisted-member coverage.
- `client_backend/services/skill_runtime/environment.py` — remove one hash-scoped runtime without deleting the whole skill.
- `client_backend/services/skill_runtime/install.py` — split prepare from mutation and delegate commits to transactions.
- `client_backend/services/skill_runtime/operations.py` — install all previewed members as one transaction and recover from the complete expected set.
- `client_backend/services/skill_runtime/locks.py` — shared skills-mutation scope.
- `client_backend/services/skill_catalog.py` — prevent scans during a transaction.
- `client_backend/services/local_skills_registry.py` — remove dead frontmatter wrappers and stale comments.
- `tests/client_backend/test_skill_installation.py`, `test_skill_operations.py`, `test_skill_catalog.py`, and integration tests — updated contracts.
- `docs/superpowers/plans/2026-08-03-modern-skills-architecture-proposal.md`, `README.md`, and `docs/skill-runtime.md` — verified architecture and bootstrap guidance.

---

### Task 1: Canonical Templates and a Self-Repairing Windows Launcher

**Files:**

- Create: `scripts/client-backend-bundle/requirements-client.txt`
- Create: `scripts/client-backend-bundle/start-client-backend.ps1`
- Create: `scripts/client-backend-bundle/start-client-backend.bat`
- Create: `scripts/client-backend-bundle/README.client_backend.md`
- Modify: `scripts/build_client_backend_bundle.py`
- Modify: `scripts/build-client-backend-bundle.ps1`
- Test: `tests/test_client_backend_bundle.py`

**Interfaces:**

- Consumes: repository source tree and optional builder `OutputRoot`.
- Produces: byte-equivalent bundle handoff files from both builders; launcher exit code equals bootstrap/sidecar result.
- Produces: marker value `2|<python-major>.<python-minor>|<requirements-sha256>`.

- [ ] **Step 1: Add a real temporary-bundle launcher harness**

Add Windows-only helpers that create an empty-requirements bundle, a minimal
`client_backend/__main__.py`, and optionally a pip-less venv:

```python
def _launcher_fixture(tmp_path: Path, *, without_pip: bool = False) -> Path:
    bundle = _build_bundle(tmp_path / "output")
    (bundle / "requirements-client.txt").write_text("", encoding="utf-8")
    main = bundle / "client_backend" / "__main__.py"
    main.write_text("raise SystemExit(7)\n", encoding="utf-8")
    if without_pip:
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(bundle / ".venv")],
            check=True,
        )
    return bundle


def _run_launcher(bundle: Path, *, disable_module_autoload: bool = False):
    command = f"& '{bundle / 'start-client-backend.ps1'}'"
    if disable_module_autoload:
        command = (
            "Remove-Item Function:\\Get-FileHash -ErrorAction SilentlyContinue; "
            "$PSModuleAutoLoadingPreference='None'; " + command
        )
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=bundle,
        capture_output=True,
        text=True,
    )
```

- [ ] **Step 2: Write failing behavioral regressions**

```python
@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_launcher_needs_no_get_file_hash_command(tmp_path):
    bundle = _launcher_fixture(tmp_path, without_pip=True)
    result = _run_launcher(bundle, disable_module_autoload=True)
    assert result.returncode == 7, result.stderr or result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_launcher_repairs_a_pipless_venv_without_network(tmp_path):
    bundle = _launcher_fixture(tmp_path, without_pip=True)
    result = _run_launcher(bundle)
    assert result.returncode == 7, result.stderr or result.stdout
    pip_probe = subprocess.run(
        [bundle / ".venv" / "Scripts" / "python.exe", "-m", "pip", "--version"],
        capture_output=True,
        text=True,
    )
    assert pip_probe.returncode == 0, pip_probe.stderr
    assert (bundle / ".venv" / ".client_requirements_installed").read_text().startswith("2|")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_launcher_propagates_the_sidecar_exit_code(tmp_path):
    bundle = _launcher_fixture(tmp_path)
    assert _run_launcher(bundle).returncode == 7


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_failed_requirements_install_never_writes_the_marker(tmp_path):
    bundle = _launcher_fixture(tmp_path)
    (bundle / "requirements-client.txt").write_text(
        "--no-index\npackage-that-does-not-exist==0\n",
        encoding="utf-8",
    )
    result = _run_launcher(bundle)
    assert result.returncode != 0
    assert not (bundle / ".venv" / ".client_requirements_installed").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_requirements_content_not_mtime_controls_reinstall(tmp_path):
    bundle = _launcher_fixture(tmp_path)
    first = _run_launcher(bundle)
    marker = bundle / ".venv" / ".client_requirements_installed"
    first_fingerprint = marker.read_text(encoding="utf-8")
    requirements = bundle / "requirements-client.txt"
    requirements.write_text("# changed without a newer timestamp\n", encoding="utf-8")
    old_time = requirements.stat().st_mtime - 3600
    os.utime(requirements, (old_time, old_time))

    second = _run_launcher(bundle)

    assert first.returncode == second.returncode == 7
    assert marker.read_text(encoding="utf-8") != first_fingerprint
```

- [ ] **Step 3: Run the launcher tests and verify the expected failures**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\test_client_backend_bundle.py -k launcher -v
```

Expected: failures show the current `Get-FileHash`, pip-less venv, exact-3.10, or
exit-propagation behavior. The test harness itself must not fail from quoting or
network access.

- [ ] **Step 4: Move the four embedded artifacts into canonical files**

Copy the current requirements, batch file, and README contents verbatim into the
new template directory, then replace builder constants with:

```python
BUNDLE_TEMPLATE_ROOT = REPO_ROOT / "scripts" / "client-backend-bundle"
BUNDLE_TEMPLATE_NAMES = (
    "requirements-client.txt",
    "start-client-backend.ps1",
    "start-client-backend.bat",
    "README.client_backend.md",
)


def _copy_bundle_templates(bundle_root: Path) -> None:
    for name in BUNDLE_TEMPLATE_NAMES:
        shutil.copy2(BUNDLE_TEMPLATE_ROOT / name, bundle_root / name)
```

The PowerShell builder uses the same directory:

```powershell
$templateRoot = Join-Path $scriptRoot "client-backend-bundle"
foreach ($name in @(
    "requirements-client.txt",
    "start-client-backend.ps1",
    "start-client-backend.bat",
    "README.client_backend.md"
)) {
    Copy-Item -LiteralPath (Join-Path $templateRoot $name) `
        -Destination (Join-Path $bundleRoot $name) -Force
}
```

- [ ] **Step 5: Implement the launcher bootstrap functions**

The canonical PowerShell file defines these exact boundaries:

```powershell
function Invoke-NativeChecked {
    param([string]$FilePath, [string[]]$ArgumentList, [string]$Phase)
    & $FilePath @ArgumentList
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "$Phase failed with exit code $code while running '$FilePath'."
    }
}

function Get-Sha256 {
    param([string]$Path)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try { $bytes = $sha.ComputeHash($stream) } finally { $stream.Dispose() }
    } finally { $sha.Dispose() }
    return ([System.BitConverter]::ToString($bytes)).Replace("-", "")
}

function Test-PythonCandidate {
    param([string]$Command, [string[]]$PrefixArguments)
    $probe = @($PrefixArguments) + @(
        "-c",
        "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
    )
    & $Command @probe 2>$null
    return $LASTEXITCODE -eq 0
}
```

Represent the chosen interpreter as a small object containing `Command` and
`PrefixArguments`. Probe `py -3`, `python`, then `python3`; create or clear the
bundle-owned venv; probe/repair pip using `ensurepip`; compute the version/hash
fingerprint; install requirements without upgrading pip; write the marker via a
temporary sibling plus `Move-Item`; then capture and return the sidecar exit code.

- [ ] **Step 6: Run launcher and full bundle tests**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\test_client_backend_bundle.py -v
```

Expected: all tests pass; the launcher tests perform no network access.

- [ ] **Step 7: Commit the launcher fix**

```powershell
git add scripts/client-backend-bundle scripts/build_client_backend_bundle.py scripts/build-client-backend-bundle.ps1 tests/test_client_backend_bundle.py
git commit -m "fix: make sidecar launcher self-repairing"
```

---

### Task 2: Enforce One Resource Listing and Read Policy

**Files:**

- Modify: `client_backend/services/skill_runtime/resources.py`
- Modify: `client_backend/services/runtime_bridge.py`
- Test: `tests/client_backend/test_skill_resources.py`
- Test: `tests/client_backend/test_runtime_bridge.py`

**Interfaces:**

- Consumes: `bundle_root: Path`, `resource_path: str`, optional `source_hash: str`.
- Produces: `SkillResourceListing(paths: list[str], truncated: bool)` containing only policy-readable files.
- Produces: `read_skill_resource(...) -> str`; `install.json` and link-like components always raise `SkillResourceError`.

- [ ] **Step 1: Write failing exclusion and intermediate-link tests**

```python
def test_provenance_metadata_cannot_be_read_directly(bundle):
    with pytest.raises(SkillResourceError, match="not readable"):
        read_skill_resource(bundle, "install.json")


def test_oversized_and_binary_files_are_not_advertised(bundle):
    (bundle / "huge.md").write_text("x" * (MAX_RESOURCE_BYTES + 1), encoding="utf-8")
    (bundle / "binary.dat").write_bytes(b"\xff\xfe")
    paths = list_skill_resources(bundle).paths
    assert "huge.md" not in paths
    assert "binary.dat" not in paths


def test_refuses_a_linked_intermediate_directory(bundle, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret", encoding="utf-8")
    link = bundle / "linked"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlink creation is unavailable")
    with pytest.raises(SkillResourceError, match="link"):
        read_skill_resource(bundle, "linked/secret.md")
```

- [ ] **Step 2: Run the resource tests and verify they fail for policy drift**

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_resources.py -v
```

Expected: `install.json` is currently readable and unreadable files are currently
listed.

- [ ] **Step 3: Implement a single resource loader**

Add an internal result and loader used by both public functions:

```python
@dataclass(frozen=True)
class _ReadableResource:
    relative_path: str
    content: str


def _read_policy_resource(bundle_root: Path, resource_path: str) -> _ReadableResource:
    candidate = _normalize_resource_path(resource_path)
    root = bundle_root.resolve()
    relative = Path(candidate)
    if relative.name in _EXCLUDED_NAMES or _EXCLUDED_DIRS.intersection(relative.parts):
        raise SkillResourceError(f"'{candidate}' is not readable content.")
    _reject_link_components(root, relative)
    target = (root / relative).resolve()
    if not is_under_root(target, root) or not target.is_file():
        raise SkillResourceError(f"'{candidate}' is not a file in this skill.")
    with target.open("rb") as stream:
        raw = stream.read(MAX_RESOURCE_BYTES + 1)
    if len(raw) > MAX_RESOURCE_BYTES:
        raise SkillResourceError(f"'{candidate}' exceeds the read limit.")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillResourceError(f"'{candidate}' is not a UTF-8 text file.") from exc
    return _ReadableResource(relative.as_posix(), content)
```

`_reject_link_components()` walks `root / part` without resolving each original
component and calls `is_link_like()` before continuing. `list_skill_resources()`
uses the same loader, skips rejected files, sorts strings, and caps the result.

Cache manifests with `functools.lru_cache(maxsize=256)` on
`(resolved_bundle_root, source_hash)` and pass `skill.source_hash` from the runtime
bridge. Do not cache file contents returned by model-selected reads.

- [ ] **Step 4: Run resource and bridge tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_resources.py tests\client_backend\test_runtime_bridge.py -v
```

Expected: all pass, including direct `install.json` denial.

- [ ] **Step 5: Commit resource hardening**

```powershell
git add client_backend/services/skill_runtime/resources.py client_backend/services/runtime_bridge.py tests/client_backend/test_skill_resources.py tests/client_backend/test_runtime_bridge.py
git commit -m "security: enforce skill resource read policy"
```

---

### Task 3: Preserve Exact Skill Documents Through Preview and Install

**Files:**

- Modify: `client_backend/services/skill_runtime/collection.py`
- Modify: `client_backend/schemas/skill_installation.py`
- Modify: `client_backend/services/skill_runtime/uploads.py`
- Modify: `client_backend/services/skill_runtime/install.py`
- Test: `tests/client_backend/test_skill_collection.py`
- Test: `tests/client_backend/test_skill_uploads.py`
- Test: `tests/test_skill_installation_chat_integration.py`

**Interfaces:**

- Produces: `DiscoveredSkill(bundle_root: Path, skill_file: Path)`.
- Produces: `DiscoveredCollection(manifest, skills: list[DiscoveredSkill])`.
- Persists: internal `SkillArchiveMember(name, bundle_path, skill_path)` entries.
- Produces: `SkillUploadService.discovered_skills(user_id, upload_id) -> list[tuple[str, DiscoveredSkill]]`.

- [ ] **Step 1: Add a failing nested single-skill upload/install test**

Build and stage an in-memory ZIP whose document is nested but whose command is at
the archive root:

```python
def _nested_single_skill_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("skills/demo/SKILL.md", SKILL_MD)
        archive.writestr("bin/demo.py", "print('ok')\n")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_nested_single_skill_preserves_document_and_bundle_root(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="nested-demo.zip",
        stream=_AsyncReader(_nested_single_skill_zip_bytes()),
    )

    members = upload_env.service.discovered_skills(USER_A, record.upload_id)
    assert len(members) == 1
    name, discovered = members[0]
    assert name == "demo"
    assert discovered.bundle_root == upload_env.service.extracted_root(
        USER_A, record.upload_id
    )
    assert (
        discovered.skill_file.relative_to(discovered.bundle_root).as_posix()
        == "skills/demo/SKILL.md"
    )
    assert (discovered.bundle_root / "bin/demo.py").is_file()
```

In the installer integration test, pass that persisted descriptor to `install()`
and assert the promoted bundle retains both paths:

```python
members = upload_service.discovered_skills(USER_ID, record.upload_id)
installed = await installer.install(members[0][1], approve_setup=True)
target = installed_root / installed["name"]
assert (target / "skills/demo/SKILL.md").is_file()
assert (target / "bin/demo.py").is_file()
```

- [ ] **Step 2: Run focused tests and verify the install-path failure**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_collection.py tests\client_backend\test_skill_uploads.py tests\test_skill_installation_chat_integration.py -k "nested or collection" -v
```

Expected: the new test fails where `skill_roots()` reads
`archive_root / "SKILL.md"`.

- [ ] **Step 3: Introduce the explicit discovery model**

```python
@dataclass(frozen=True)
class DiscoveredSkill:
    bundle_root: Path
    skill_file: Path

    def __post_init__(self) -> None:
        root = self.bundle_root.resolve()
        document = self.skill_file.resolve()
        if document.name != "SKILL.md" or root != document.parent and root not in document.parents:
            raise ValueError("skill document must be inside its bundle root")

    @property
    def relative_skill_file(self) -> str:
        return self.skill_file.resolve().relative_to(self.bundle_root.resolve()).as_posix()
```

Replace `skill_roots` on `DiscoveredCollection` with `skills`. One found document
becomes `DiscoveredSkill(root, found_skill_file)`; multiple documents become
`DiscoveredSkill(found.parent, found)`.

- [ ] **Step 4: Persist safe member paths in the upload receipt**

```python
class SkillArchiveMember(CamelModel):
    name: str
    bundle_path: str
    skill_path: str


class SkillUploadRecord(CamelModel):
    # existing fields...
    members: list[SkillArchiveMember] = Field(default_factory=list)

    def to_api(self) -> dict[str, Any]:
        payload = super().to_api()
        for internal in ("owner", "version", "requestFingerprint", "members"):
            payload.pop(internal, None)
        return payload
```

Validate both relative paths on rehydration with `is_under_root`, verify the
document is named `SKILL.md`, and verify its parsed name equals the receipt member
name. Remove `SkillUploadService.skill_roots()` after all callers use
`discovered_skills()`.

- [ ] **Step 5: Make installer preview/install accept `DiscoveredSkill`**

Add `_load_discovered_skill(discovered)` and retain `_discover_source(path)` only
as the trusted-local-path adapter. Hash `bundle_root`, read the exact `skill_file`,
and calculate `bundle_shape` from their relationship. Upload operations pass the
descriptor directly and never rescan for a different document.

- [ ] **Step 6: Run discovery, upload, and integration tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_collection.py tests\client_backend\test_skill_uploads.py tests\client_backend\test_skill_installation.py tests\test_skill_installation_chat_integration.py -v
```

- [ ] **Step 7: Commit exact-member discovery**

```powershell
git add client_backend/services/skill_runtime/collection.py client_backend/schemas/skill_installation.py client_backend/services/skill_runtime/uploads.py client_backend/services/skill_runtime/install.py tests/client_backend/test_skill_collection.py tests/client_backend/test_skill_uploads.py tests/client_backend/test_skill_installation.py tests/test_skill_installation_chat_integration.py
git commit -m "fix: preserve previewed skill members through install"
```

---

### Task 4: Split Install Preparation From Mutation and Add Targeted Runtime Cleanup

**Files:**

- Modify: `client_backend/services/skill_runtime/environment.py`
- Modify: `client_backend/services/skill_runtime/install.py`
- Test: `tests/client_backend/test_skill_environment.py`
- Test: `tests/client_backend/test_skill_installation.py`

**Interfaces:**

- Produces: `SkillEnvironmentManager.remove_runtime(skill_name, source_hash) -> None`.
- Produces: `PreparedSkillInstall` with stage/target/previous/backup paths, hashes, action, and runtime status.
- Produces: `SkillBundleInstaller.prepare_install(spec) -> PreparedSkillInstall` without changing installed bundles.

- [ ] **Step 1: Write failing targeted-runtime cleanup tests**

```python
def test_remove_runtime_removes_only_the_requested_hash(runtime_manager, tmp_path):
    first = runtime_manager._skill_runtime_root("demo") / ("a" * 64)
    second = runtime_manager._skill_runtime_root("demo") / ("b" * 64)
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    runtime_manager.remove_runtime("demo", "b" * 64)
    assert first.is_dir()
    assert not second.exists()
```

Add an installer test that calls `prepare_install()` for a replacement and asserts
the old installed bundle still exists while the returned stage exists.

- [ ] **Step 2: Run the tests and verify the new interfaces are absent**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_environment.py tests\client_backend\test_skill_installation.py -k "remove_runtime or prepare_install" -v
```

- [ ] **Step 3: Add the transaction preparation types**

In `install.py`:

```python
@dataclass(frozen=True)
class SkillInstallSpec:
    discovered: DiscoveredSkill
    expected_source_hash: str | None
    approve_setup: bool
    replace_source_hash: str | None
    source_kind: Literal["path", "upload"]


@dataclass
class PreparedSkillInstall:
    name: str
    source_hash: str
    action: Literal["installed", "updated"]
    stage: Path
    target: Path
    previous: Path | None
    backup: Path
    runtime_status: str
    previous_source_hash: str | None
```

Move validation, copy, copied-hash verification, metadata writing, and runtime
preparation into `prepare_install()`. Do not call `os.replace`, delete a stale
bundle, refresh the registry, or remove secrets in this method.

- [ ] **Step 4: Add hash-scoped runtime removal**

```python
def remove_runtime(self, skill_name: str, source_hash: str) -> None:
    target = self._skill_runtime_root(skill_name) / source_hash
    if target.exists() and is_under_root(target, self._base()):
        shutil.rmtree(target)
    parent = target.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
```

Keep `remove_skill()` for explicit uninstall only.

- [ ] **Step 5: Run installer and environment tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_environment.py tests\client_backend\test_skill_installation.py -v
```

- [ ] **Step 6: Commit preparation refactor**

```powershell
git add client_backend/services/skill_runtime/environment.py client_backend/services/skill_runtime/install.py tests/client_backend/test_skill_environment.py tests/client_backend/test_skill_installation.py
git commit -m "refactor: stage skill installs before mutation"
```

---

### Task 5: Add Journaled Collection Commit and Rollback

**Files:**

- Create: `client_backend/services/skill_runtime/transactions.py`
- Create: `tests/client_backend/test_skill_transactions.py`
- Modify: `client_backend/services/skill_runtime/install.py`
- Modify: `client_backend/services/skill_runtime/locks.py`
- Modify: `client_backend/core/paths.py`

**Interfaces:**

- Produces: `SKILLS_MUTATION_SCOPE = "skills-mutation"`.
- Produces: `SkillInstallTransaction.execute(specs, transaction_id, observer) -> list[dict]`;
  observer events include `member_promoted` after journal persistence.
- Persists: `transaction-<id>.json` with `state`, complete member descriptors, and promotion progress.
- Produces: `recover_install_transactions(user_id, installer) -> dict[str, Literal["committed", "rolled_back", "failed"]]`.

- [ ] **Step 1: Write a failing fresh-install rollback test**

Use real temporary bundle directories and inject a promotion failure for the
second member:

```python
@pytest.mark.asyncio
async def test_later_promotion_failure_removes_every_fresh_member(transaction_env):
    specs = transaction_env.specs("one", "two")
    transaction_env.fail_promote_for = "two"
    with pytest.raises(OSError, match="promote two"):
        await transaction_env.transaction.execute(specs, transaction_id="tx-fresh")
    assert transaction_env.installed_names() == []
    assert transaction_env.journal("tx-fresh")["state"] == "rolled_back"
```

- [ ] **Step 2: Write a failing replacement restoration test**

```python
@pytest.mark.asyncio
async def test_later_failure_restores_old_bundle_runtime_and_secrets(transaction_env):
    old_hashes = transaction_env.install_old_versions("one", "two")
    transaction_env.secret_store.set_for_test("one", "TOKEN", "encrypted")
    transaction_env.fail_promote_for = "two"
    with pytest.raises(OSError):
        await transaction_env.transaction.execute(
            transaction_env.update_specs("one", "two"), transaction_id="tx-update"
        )
    assert transaction_env.installed_hashes() == old_hashes
    assert transaction_env.runtime_exists("one", old_hashes["one"])
    assert transaction_env.secret_store.list_for_skill("one") == ["TOKEN"]
```

- [ ] **Step 3: Run transaction tests and confirm there is no coordinator**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_transactions.py -v
```

- [ ] **Step 4: Implement journal models and confined relative paths**

Define Pydantic models with `extra="forbid"`:

```python
class TransactionMember(BaseModel):
    name: str
    source_hash: str
    previous_source_hash: str | None = None
    stage_name: str
    target_name: str
    previous_name: str | None = None
    backup_name: str
    promoted: bool = False
    previous_backed_up: bool = False


class InstallTransactionRecord(BaseModel):
    version: int = 1
    transaction_id: str
    owner: str
    state: Literal["prepared", "committing", "committed", "rolling_back", "rolled_back"]
    members: list[TransactionMember]
    cleanup_complete: bool = False
    cleanup_error: str | None = None
```

Every stored name is one path component. Resolve it beneath the installed root
and reject `.`, `..`, separators, or escapes before any mutation.

- [ ] **Step 5: Implement commit and rollback order**

Under `profile_lock(user_id, SKILLS_MUTATION_SCOPE)`:

1. Persist `prepared` with every member.
2. Persist `committing`.
3. Move each prior bundle to its backup and persist `previous_backed_up=True`.
4. Move its stage to target and persist `promoted=True`.
5. After all members, persist `committed`.
6. Delete backups and old hash-scoped runtimes.
7. Persist `cleanup_complete=True`; retain the terminal journal until its owning
   operation receipt is durably finalized.

On any exception before `committed`, persist `rolling_back`; iterate members in
reverse; remove promoted targets; restore backups; remove only new hash-scoped
runtimes; persist `rolled_back` and `cleanup_complete=True`. If cleanup fails,
persist `cleanup_error`, retain the incomplete state, and report a failed recovery
outcome. Add `finalize(transaction_id)`, which deletes only a terminal,
cleanup-complete journal and rejects any other state.

- [ ] **Step 6: Make one-skill `install()` use the same transaction**

`SkillBundleInstaller.install()` adapts a path or `DiscoveredSkill` to one
`SkillInstallSpec`, calls `SkillInstallTransaction.execute()`, refreshes the
registry once, finalizes its terminal journal after no operation receipt remains
to be written, and returns the sole result. Add `install_many()` for operations to
pass a complete list and refresh once; it retains the journal for the operation
service to finalize after persisting success or completed rollback.

- [ ] **Step 7: Run transaction and installer tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_transactions.py tests\client_backend\test_skill_installation.py -v
```

- [ ] **Step 8: Commit transaction support**

```powershell
git add client_backend/core/paths.py client_backend/services/skill_runtime/locks.py client_backend/services/skill_runtime/transactions.py client_backend/services/skill_runtime/install.py tests/client_backend/test_skill_transactions.py tests/client_backend/test_skill_installation.py
git commit -m "fix: make skill collection installs transactional"
```

---

### Task 6: Recover Operations From the Complete Transaction State

**Files:**

- Modify: `client_backend/schemas/skill_installation.py`
- Modify: `client_backend/services/skill_runtime/operations.py`
- Modify: `client_backend/services/skill_runtime/transactions.py`
- Test: `tests/client_backend/test_skill_operations.py`
- Test: `tests/client_backend/test_skill_transactions.py`

**Interfaces:**

- Persists: `transaction_id: str | None` and `expected_source_hashes: dict[str, str]` on operation receipts.
- Consumes: `install_many(specs, transaction_id=operation_id, observer=...)`.
- Recovery succeeds only when the transaction is committed and every expected member hash is installed.

- [ ] **Step 1: Replace the first-hash recovery test with complete-set cases**

```python
@pytest.mark.asyncio
async def test_recovery_does_not_accept_only_the_first_collection_member(operation_env):
    operation = operation_env.persist_operation(
        state="running",
        commit_started=True,
        expected_source_hashes={"one": "a" * 64, "two": "b" * 64},
        transaction_id="tx-partial",
    )
    operation_env.persist_installed_bundle(name="one", source_hash="a" * 64)
    operation_env.persist_transaction("tx-partial", state="committing")
    await operation_env.service.recover(USER_A)
    recovered = operation_env.get(operation)
    assert recovered.state == "failed"
    assert recovered.failure.code == "SKILL_INTERRUPTED"


@pytest.mark.asyncio
async def test_recovery_accepts_committed_complete_collection(operation_env):
    expected = {"one": "a" * 64, "two": "b" * 64}
    operation = operation_env.persist_operation(
        state="running", commit_started=True,
        expected_source_hashes=expected, transaction_id="tx-complete"
    )
    for name, source_hash in expected.items():
        operation_env.persist_installed_bundle(name=name, source_hash=source_hash)
    operation_env.persist_transaction("tx-complete", state="committed")
    await operation_env.service.recover(USER_A)
    assert operation_env.get(operation).state == "succeeded"
```

- [ ] **Step 2: Run operation recovery tests and verify the first-hash logic fails them**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_operations.py -k recovery -v
```

- [ ] **Step 3: Extend persisted operation state compatibly**

Add defaulted fields so version-1 receipts still parse:

```python
transaction_id: str | None = None
expected_source_hashes: dict[str, str] = Field(default_factory=dict)
```

Drop both from `to_api()`. At `start()`, derive the full name/hash mapping from
`upload.skills or [upload.preview]`, assign `transaction_id=operation_id`, and
retain `upload_source_hash` only for old-receipt compatibility.

- [ ] **Step 4: Install the collection in one coordinator call**

Replace the per-entry install loop and `_rollback_installed()` with construction
of all `SkillInstallSpec` values followed by:

```python
installed = await installer.install_many(
    specs,
    transaction_id=operation_id,
    observer=observer,
)
```

Remove the old rollback helper and its inaccurate comments.

- [ ] **Step 5: Recover transaction journals before classifying receipts**

`recover(user_id)` first invokes transaction recovery. A receipt succeeds only if
its journal outcome is `committed` and `_installed_skill_hashes()` contains every
`expected_source_hashes` item. Any pre-commit recovery that rolled back becomes
retryable `SKILL_INTERRUPTED`. Persist the terminal operation receipt first, then
call `finalize(transaction_id)`. If the process stops between those writes, the
next recovery sees an already-terminal receipt and safely finalizes its terminal,
cleanup-complete journal. Missing journals are accepted only for already-terminal
operation receipts; a non-terminal receipt with no journal never infers success
from installed bundles alone.

- [ ] **Step 6: Run operation, transaction, and chat integration tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_operations.py tests\client_backend\test_skill_transactions.py tests\test_skill_installation_chat_integration.py -v
```

- [ ] **Step 7: Commit complete-set recovery**

```powershell
git add client_backend/schemas/skill_installation.py client_backend/services/skill_runtime/operations.py client_backend/services/skill_runtime/transactions.py tests/client_backend/test_skill_operations.py tests/client_backend/test_skill_transactions.py tests/test_skill_installation_chat_integration.py
git commit -m "fix: recover complete skill collection transactions"
```

---

### Task 7: Prevent Catalogs From Observing Intermediate Transactions

**Files:**

- Modify: `client_backend/services/skill_runtime/locks.py`
- Modify: `client_backend/services/skill_catalog.py`
- Modify: `client_backend/services/skill_runtime/install.py`
- Test: `tests/client_backend/test_skill_catalog.py`
- Test: `tests/client_backend/test_skill_state_and_locks.py`
- Test: `tests/client_backend/test_skill_transactions.py`

**Interfaces:**

- Produces: all installed-bundle mutations and registry rescans share `SKILLS_MUTATION_SCOPE`.
- Catalog generation advances once after a successful collection transaction and never for an intermediate set.

- [ ] **Step 1: Write a failing concurrency test**

Instrument the catalog's `profile_lock` wrapper so the test knows exactly when
the scan has attempted to acquire the mutation lock. Pause the real transaction
from its persisted `member_promoted` observer event:

```python
@pytest.mark.asyncio
async def test_catalog_waits_for_the_complete_collection_transaction(
    transaction_catalog_env, monkeypatch
):
    first_promoted = asyncio.Event()
    release_transaction = asyncio.Event()
    catalog_lock_attempted = asyncio.Event()

    async def observer(event: str, payload: dict) -> None:
        if event == "member_promoted" and payload["index"] == 0:
            first_promoted.set()
            await release_transaction.wait()

    original_lock = transaction_catalog_env.profile_lock

    @asynccontextmanager
    async def observed_profile_lock(user_id: str, scope: str):
        catalog_lock_attempted.set()
        async with original_lock(user_id, scope):
            yield

    monkeypatch.setitem(
        transaction_catalog_env.catalog_globals,
        "profile_lock",
        observed_profile_lock,
    )
    transaction_task = asyncio.create_task(
        transaction_catalog_env.transaction.execute(
            transaction_catalog_env.specs("one", "two"),
            transaction_id="tx-catalog",
            observer=observer,
        )
    )
    await asyncio.wait_for(first_promoted.wait(), timeout=1)
    snapshot_task = asyncio.create_task(
        transaction_catalog_env.catalog.snapshot(force=True)
    )
    await asyncio.wait_for(catalog_lock_attempted.wait(), timeout=1)
    assert not snapshot_task.done()

    release_transaction.set()
    await transaction_task
    snapshot = await snapshot_task

    assert {skill["name"] for skill in snapshot["skills"]} >= {"one", "two"}
```

The fixture uses the real file lock, real installed directories, and real
registry scan; events control ordering, so no timing sleep is permitted.

- [ ] **Step 2: Verify the current catalog can scan between promotions**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_catalog.py -k intermediate -v
```

- [ ] **Step 3: Share the mutation lock with registry refresh**

Wrap the initialize/refresh block in `SkillCatalogService._refresh_if_stale()`:

```python
user_id = self._resolve_user_id()
if user_id is None:
    await registry.initialize()
    await registry.refresh()
else:
    async with profile_lock(user_id, SKILLS_MUTATION_SCOPE):
        await registry.initialize()
        await registry.refresh()
```

Ensure install/uninstall do not call catalog snapshot while holding the same
non-reentrant process lock. Registry refresh needed for the mutation happens
inside the transaction; publication happens after lock release.

- [ ] **Step 4: Run lock, catalog, transaction, and API tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\client_backend\test_skill_state_and_locks.py tests\client_backend\test_skill_catalog.py tests\client_backend\test_skill_transactions.py tests\client_backend\test_skills_api.py -v
```

- [ ] **Step 5: Commit catalog isolation**

```powershell
git add client_backend/services/skill_runtime/locks.py client_backend/services/skill_catalog.py client_backend/services/skill_runtime/install.py tests/client_backend/test_skill_state_and_locks.py tests/client_backend/test_skill_catalog.py tests/client_backend/test_skill_transactions.py tests/client_backend/test_skills_api.py
git commit -m "fix: isolate skill catalogs from partial transactions"
```

---

### Task 8: Correct the Architecture Proposal and Remove Dead or Redundant Paths

**Files:**

- Modify: `docs/superpowers/plans/2026-08-03-modern-skills-architecture-proposal.md`
- Modify: `README.md`
- Modify: `docs/skill-runtime.md`
- Modify: `client_backend/services/local_skills_registry.py`
- Modify: affected tests after reference audit

**Interfaces:**

- Produces: documentation matching verified code and the current Agent Skills specification.
- Removes: duplicated builder content, obsolete source-grep tests, unused frontmatter wrappers, stale task-number comments, old collection rollback path.
- Retains: documented path-based install routes and `/api` compatibility alias, because repository contracts prove they remain supported.

- [ ] **Step 1: Update the proposal's gap table and Phase 1 status**

Make these exact corrections:

- G1 and G2: closed and hardened by the commits from Tasks 2–7.
- G3: “No persisted collection provenance/version or update discovery”; note that
  guarded upload replacement already updates installed skills.
- G4: unchanged flat-name collision problem.
- G5: list `license`, `compatibility`, `metadata`, and experimental
  `allowed-tools`; record missing non-empty/bounds/directory-name validation.
- G6: replace `~50` as a decision with a measurement requirement.
- Phase 1: remove the nonexistent persistent shared `collection_id`; describe the
  operational transaction journal separately.
- `allowed-tools`: describe pre-approval semantics, not restriction.

- [ ] **Step 2: Remove proven dead registry wrappers and stale comments**

Run:

```powershell
rg -n '_split_front_matter|_extract_yaml_value' app client_backend shared tests
```

If the only production hits remain the two definitions, delete them and remove
their now-unused imports. Replace “Task 4” comments with responsibility-based
language such as “installer metadata”.

- [ ] **Step 3: Remove obsolete rollback and duplicated-builder tests**

Delete `_rollback_installed()` and any stub fields/tests that only assert it calls
`uninstall()`. Delete tests that regex-extract embedded PowerShell blocks, because
canonical-template equality and executable launcher tests supersede them. Keep
requirements coverage, first-party import coverage, and generated-tree equality.

- [ ] **Step 4: Audit compatibility routes and record retention**

Run:

```powershell
rg -n 'skills/install/preview|skills/install|source_path|/api/skills' README.md docs plans demo.py app client_backend tests
```

The current evidence identifies path-based install as supported trusted local
tooling and `/api` as a tested compatibility alias. Retain them; update comments
to state the consumer and removal gate. Do not call them dead or deprecated.

- [ ] **Step 5: Run documentation and architecture contract tests**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\test_skills_architecture.py tests\test_skills_tool.py tests\test_production_readiness_contract.py tests\client_backend\test_skill_installation.py tests\client_backend\test_skill_upload_api.py -v
```

- [ ] **Step 6: Run dead/deprecated reference scans**

```powershell
rg -n -i 'Task 4|collection_id|all-or-nothing|Get-FileHash|LastWriteTimeUtc|START_PS1_CONTENT|REQUIREMENTS_CONTENT' scripts client_backend tests docs README.md
rg -n '_rollback_installed|_split_front_matter|_extract_yaml_value|skill_roots\(' client_backend tests
```

Expected: remaining matches are historical discussion explicitly labeled as such,
or none. Remove any production match whose symbol has no caller.

- [ ] **Step 7: Commit documentation and cleanup**

```powershell
git add docs/superpowers/plans/2026-08-03-modern-skills-architecture-proposal.md README.md docs/skill-runtime.md client_backend/services/local_skills_registry.py client_backend/services/skill_runtime/operations.py tests
git commit -m "docs: align modern skills architecture with hardened runtime"
```

---

### Task 9: Full Verification and Rebuilt Handoff Artifact

**Files:**

- Regenerate: `dist/client-backend-bundle/`
- Regenerate: `dist/client-backend-bundle.zip`
- Verify only; commit generated artifacts only if repository tracking policy changes.

**Interfaces:**

- Consumes: all completed tasks.
- Produces: tested clean-extraction bundle for frontend handoff.

- [ ] **Step 1: Run formatting and lint checks**

```powershell
& '.\.venv\Scripts\python.exe' -m ruff format --check scripts client_backend tests
& '.\.venv\Scripts\python.exe' -m ruff check scripts client_backend tests
```

Expected: zero formatting or lint errors.

- [ ] **Step 2: Run the complete skills and bundle regression set**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest `
  tests\test_client_backend_bundle.py `
  tests\client_backend\test_skill_resources.py `
  tests\client_backend\test_skill_collection.py `
  tests\client_backend\test_skill_uploads.py `
  tests\client_backend\test_skill_installation.py `
  tests\client_backend\test_skill_transactions.py `
  tests\client_backend\test_skill_operations.py `
  tests\client_backend\test_skill_catalog.py `
  tests\client_backend\test_skill_state_and_locks.py `
  tests\client_backend\test_runtime_bridge.py `
  tests\client_backend\test_skill_upload_api.py `
  tests\test_skill_installation_chat_integration.py `
  tests\test_skill_device_isolation.py `
  tests\test_skills_tool.py `
  tests\test_skills_architecture.py -v
```

Expected: all pass; platform-incapable link tests may skip with explicit reasons.

- [ ] **Step 3: Build with both builders and compare generated trees**

Build into two separate temporary output roots:

```powershell
$pythonOutput = Join-Path $env:TEMP "sidecar-python-builder"
$powershellOutput = Join-Path $env:TEMP "sidecar-powershell-builder"
& '.\.venv\Scripts\python.exe' scripts\build_client_backend_bundle.py --output-root $pythonOutput
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build-client-backend-bundle.ps1 -OutputRoot $powershellOutput
& '.\.venv\Scripts\python.exe' -c "from pathlib import Path; import hashlib, sys; a,b=map(Path,sys.argv[1:]); fa={p.relative_to(a).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in a.rglob('*') if p.is_file()}; fb={p.relative_to(b).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in b.rglob('*') if p.is_file()}; assert fa==fb,(fa.keys()^fb.keys(), {k for k in fa.keys()&fb.keys() if fa[k]!=fb[k]})" "$pythonOutput\client-backend-bundle" "$powershellOutput\client-backend-bundle"
```

Expected: comparison exits zero.

- [ ] **Step 4: Rebuild the workspace handoff artifact**

```powershell
& '.\.venv\Scripts\python.exe' scripts\build_client_backend_bundle.py
```

Confirm the generated ZIP contains canonical launchers and no `.venv`, `.env.client`,
cache, staging, backup, transaction, or untracked repository files.

- [ ] **Step 5: Smoke-test a clean extraction**

Extract the ZIP into a new temporary directory, replace its requirements with an
empty local fixture, run the launcher with module autoload disabled, and verify
the minimal sidecar module's nonzero exit code is propagated. Then run the real
bundle's `python -m client_backend doctor` using its repaired venv and test config.

- [ ] **Step 6: Inspect repository state and final commit range**

```powershell
git status --short
git log --oneline c15c77e..HEAD
git diff --check c15c77e..HEAD
```

Expected: only intentional tracked changes and the pre-existing untracked
`superpowers-main.zip`; no accidental `.env.client`, `.venv`, cache, or temporary
transaction files.

- [ ] **Step 7: Request code review and address findings**

Use `superpowers:requesting-code-review` with base `c15c77e`, current `HEAD`, this
plan, and the approved design. Fix all Critical and Important findings, rerun the
affected red-green tests, then repeat Steps 1–6.

- [ ] **Step 8: Commit any review fixes**

```powershell
git add --update
git commit -m "fix: close sidecar hardening review findings"
```

Skip this commit only when review required no tracked changes.
