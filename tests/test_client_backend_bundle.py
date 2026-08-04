import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

FIRST_PARTY_ROOTS = ("app", "shared", "client_backend")

# Modules the bundle imports but deliberately does not ship. Anything missing
# and not listed here is builder drift and must fail. Empty is the healthy
# state: the bundle should ship every first-party module it imports.
INTENTIONALLY_ABSENT_FROM_BUNDLE: frozenset[str] = frozenset()


def _first_party_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(
                alias.name for alias in node.names if alias.name.split(".")[0] in FIRST_PARTY_ROOTS
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] in FIRST_PARTY_ROOTS
        ):
            found.add(node.module)
    return found


def _resolve_module(module: str, root: Path) -> Path | None:
    base = root / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _missing_first_party_modules(bundle_root: Path) -> dict[str, set[str]]:
    """Return {missing module: importers} for the bundle's whole import graph."""
    queue = [
        path
        for name in FIRST_PARTY_ROOTS
        if (bundle_root / name).is_dir()
        for path in (bundle_root / name).rglob("*.py")
    ]
    seen: set[Path] = set()
    missing: dict[str, set[str]] = {}

    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)

        for module in _first_party_imports(path):
            resolved = _resolve_module(module, bundle_root)
            if resolved is not None:
                queue.append(resolved)
            else:
                importer = path.relative_to(bundle_root).as_posix()
                missing.setdefault(module, set()).add(importer)

    return missing


def test_runtime_protocol_source_compiles_from_source():
    runtime_protocol = REPO_ROOT / "app" / "schemas" / "runtime_protocol.py"

    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(runtime_protocol)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="bundle script is PowerShell/Windows-specific")
def test_client_backend_bundle_can_import_local_skills_registry(tmp_path):
    output_root = tmp_path / "bundle-output"
    build_script = REPO_ROOT / "scripts" / "build-client-backend-bundle.ps1"

    build = subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(build_script),
            "-OutputRoot",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert build.returncode == 0, build.stderr or build.stdout

    bundle_root = output_root / "client-backend-bundle"
    assert (bundle_root / "app" / "ai" / "mcp_config.json").is_file()
    assert (bundle_root / "app" / "ai" / "mcp_servers" / "time_server.py").is_file()
    assert (bundle_root / "app" / "services" / "widget_runtime.py").is_file()
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, r'{bundle_root}'); "
                "import client_backend.services.local_skills_registry; "
                "print('OK')"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=bundle_root,
    )

    assert probe.returncode == 0, probe.stderr or probe.stdout


def _build_bundle(output_root: Path) -> Path:
    from build_client_backend_bundle import build

    bundle_root, _ = build(output_root)
    return bundle_root


def _launcher_fixture(tmp_path: Path, *, without_pip: bool = False) -> Path:
    """Build a real handoff bundle whose sidecar has a deterministic exit."""
    bundle = _build_bundle(tmp_path / "output")
    (bundle / "requirements-client.txt").write_text("", encoding="utf-8")
    (bundle / "client_backend" / "__main__.py").write_text(
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    if without_pip:
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(bundle / ".venv")],
            check=True,
        )
    return bundle


def _run_launcher(bundle: Path, *, block_get_file_hash: bool = False):
    launcher = bundle / "start-client-backend.ps1"
    command = f"& '{launcher}'"
    if block_get_file_hash:
        command = "function Get-FileHash { throw 'Get-FileHash is unavailable' }; " + command
    command += "; exit $LASTEXITCODE"
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=bundle,
        capture_output=True,
        text=True,
    )


def _run_batch_launcher(bundle: Path):
    return subprocess.run(
        [str(bundle / "start-client-backend.bat")],
        cwd=bundle,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_launcher_needs_no_get_file_hash_command(tmp_path):
    bundle = _launcher_fixture(tmp_path, without_pip=True)

    result = _run_launcher(bundle, block_get_file_hash=True)

    assert result.returncode == 7, result.stderr or result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_launcher_repairs_a_pipless_venv_without_network(tmp_path):
    bundle = _launcher_fixture(tmp_path, without_pip=True)

    result = _run_launcher(bundle)

    assert result.returncode == 7, result.stderr or result.stdout
    pip_probe = subprocess.run(
        [
            bundle / ".venv" / "Scripts" / "python.exe",
            "-m",
            "pip",
            "--version",
        ],
        capture_output=True,
        text=True,
    )
    assert pip_probe.returncode == 0, pip_probe.stderr
    marker = bundle / ".venv" / ".client_requirements_installed"
    assert marker.read_text(encoding="utf-8-sig").startswith("2|")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_launcher_propagates_the_sidecar_exit_code(tmp_path):
    bundle = _launcher_fixture(tmp_path)

    result = _run_launcher(bundle)

    assert result.returncode == 7, result.stderr or result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_batch_launcher_propagates_the_sidecar_exit_code(tmp_path):
    bundle = _launcher_fixture(tmp_path)

    result = _run_batch_launcher(bundle)

    assert result.returncode == 7, result.stderr or result.stdout


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
    first_fingerprint = marker.read_text(encoding="utf-8-sig")
    requirements = bundle / "requirements-client.txt"
    requirements.write_text("# changed without a newer timestamp\n", encoding="utf-8")
    old_time = requirements.stat().st_mtime - 3600
    os.utime(requirements, (old_time, old_time))

    second = _run_launcher(bundle)

    assert first.returncode == second.returncode == 7
    assert marker.read_text(encoding="utf-8-sig") != first_fingerprint


def test_bundle_ships_every_first_party_module_it_imports(tmp_path):
    """The bundle's app/ allow-list is hand-maintained and silently drifts.

    A module reachable from the startup path but absent from the bundle only
    surfaces as a ModuleNotFoundError on the recipient's machine, so assert the
    whole import graph resolves inside the bundle.
    """
    bundle_root = _build_bundle(tmp_path / "bundle-output")

    missing = _missing_first_party_modules(bundle_root)
    unexpected = {
        module: sorted(importers)
        for module, importers in missing.items()
        if module not in INTENTIONALLY_ABSENT_FROM_BUNDLE
    }

    assert not unexpected, (
        "bundle imports first-party modules it does not ship; add them to the "
        f"builders or to INTENTIONALLY_ABSENT_FROM_BUNDLE: {unexpected}"
    )


def test_bundle_can_import_the_real_server_startup_path(tmp_path):
    """``client_backend.main`` is what start-client-backend.bat actually loads.

    Importing a leaf module proves far less: the crash the recipient sees comes
    from the FastAPI app assembling its routers.
    """
    bundle_root = _build_bundle(tmp_path / "bundle-output")

    probe = subprocess.run(
        [sys.executable, "-c", "import client_backend.main; print('OK')"],
        capture_output=True,
        text=True,
        cwd=bundle_root,
    )

    assert probe.returncode == 0, probe.stderr or probe.stdout
    assert "OK" in probe.stdout


# Third-party packages the sidecar reaches only through an optional path, plus
# the ones whose import name differs from their distribution name. Anything else
# the bundle imports must appear in its requirements file, or a fresh install
# starts and immediately dies on ModuleNotFoundError.
THIRD_PARTY_IMPORT_TO_DISTRIBUTION = {
    "dotenv": "python-dotenv",
    "jwt": "PyJWT",
    "multipart": "python-multipart",
    "yaml": "PyYAML",
}

# Imported behind a feature the bundle does not ship, or provided transitively by
# a declared package. Each entry is a deliberate exclusion, not an oversight.
BUNDLE_REQUIREMENTS_EXEMPT = frozenset(
    {
        "app",
        "client_backend",
        "shared",
        # Pulled in by fastapi/uvicorn rather than declared directly.
        "starlette",
        "anyio",
        "sniffio",
        "click",
        "h11",
        "certifi",
        "idna",
        "typing_extensions",
        "annotated_types",
        "pydantic_core",
        "charset_normalizer",
        "urllib3",
        "requests",
        # Server-side only; the sidecar's copied app/ modules import them behind
        # guards that never run in the bundle.
        "sqlalchemy",
        "alembic",
        "celery",
        "langchain",
        "langchain_core",
        "langgraph",
        "openai",
        "psycopg2",
        "qdrant_client",
        "sentence_transformers",
        "torch",
        "transformers",
        "tiktoken",
        "nltk",
        "numpy",
        "PIL",
        "pypdf",
        "docx",
        "openpyxl",
        "pdfplumber",
        "tabulate",
        "dependency_injector",
        "prometheus_client",
        "truststore",
        "httpcore",
        "httpx2",
        "streamlit",
        "markdown",
        "dateutil",
        "pytest",
    }
)


def _bundle_requirement_names() -> set[str]:
    names = set()
    requirements = (
        REPO_ROOT / "scripts" / "client-backend-bundle" / "requirements-client.txt"
    ).read_text(encoding="utf-8")
    for line in requirements.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        name = re.split(r"[<>=!\[;]", entry, maxsplit=1)[0].strip()
        names.add(_normalize_distribution(name))
    return names


def _normalize_distribution(name: str) -> str:
    """PEP 503 normalization, so pydantic_settings matches pydantic-settings."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _third_party_imports(root: Path) -> dict[str, set[str]]:
    """Map every third-party top-level import in the tree to its importers."""
    found: dict[str, set[str]] = {}
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top in sys.stdlib_module_names or top in BUNDLE_REQUIREMENTS_EXEMPT:
                    continue
                found.setdefault(top, set()).add(path.relative_to(REPO_ROOT).as_posix())
    return found


def test_bundle_requirements_cover_every_third_party_import_of_the_sidecar():
    """A missing entry here is a bundle that dies on startup, not a test nit.

    The builder embeds its own requirements list rather than reusing the
    repository manifests, so adding a dependency to pyproject.toml does not reach
    the shipped sidecar. filelock was exactly that: imported by the skill lock
    module, absent from the bundle, and undetectable until first run.
    """
    declared = _bundle_requirement_names()
    imported = _third_party_imports(REPO_ROOT / "client_backend")

    missing = {
        module: sorted(importers)
        for module, importers in sorted(imported.items())
        if _normalize_distribution(THIRD_PARTY_IMPORT_TO_DISTRIBUTION.get(module, module))
        not in declared
    }

    assert not missing, f"bundle requirements omit imported packages: {missing}"


def _tracked_files(prefix: str) -> set[str]:
    listing = subprocess.run(
        ["git", "ls-files", "--", prefix],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in listing.stdout.splitlines() if line.strip()}


def test_bundle_ships_every_tracked_mcp_server_and_nothing_untracked(tmp_path):
    """The builder copies named scripts, so drift has to be caught here.

    A recursive copy of app/ai/mcp_servers used to sweep in whatever a developer
    had installed beside the tracked scripts -- OCR binaries and model weights in
    one working tree added ~112 MiB -- which both bloated the distributable and
    made the artifact depend on one machine's untracked files. Copying explicitly
    fixes that but can silently miss a newly tracked server, so both directions
    are asserted.
    """
    from build_client_backend_bundle import build

    bundle_root, _ = build(tmp_path)
    shipped_dir = bundle_root / "app" / "ai" / "mcp_servers"
    shipped = {path.name for path in shipped_dir.iterdir() if path.is_file()}
    tracked = {
        name.rsplit("/", 1)[-1]
        for name in _tracked_files("app/ai/mcp_servers")
        if name.endswith(".py")
    }

    assert tracked, "no tracked MCP servers found; the guard would pass vacuously"
    assert shipped == tracked, (
        f"bundle MCP servers drifted from the tracked set: {shipped ^ tracked}"
    )
    assert not [path for path in shipped_dir.rglob("*") if path.is_dir()], (
        "the bundle must not ship MCP server subdirectories, which are unversioned "
        "local installations"
    )


def test_bundle_contains_no_untracked_repository_content(tmp_path):
    """Everything shipped must be either tracked or generated by the builder."""
    from build_client_backend_bundle import build

    bundle_root, _ = build(tmp_path)
    generated = {
        "requirements-client.txt",
        "start-client-backend.ps1",
        "start-client-backend.bat",
        "README.client_backend.md",
        "__init__.py",
    }
    tracked_names = {name.rsplit("/", 1)[-1] for name in _tracked_files(".")}

    unexplained = sorted(
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name not in generated and path.name not in tracked_names
    )

    assert not unexplained, (
        f"bundle ships files that are neither tracked nor generated: {unexplained}"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell builder is Windows-specific")
def test_both_builders_copy_the_canonical_handoff_files(tmp_path):
    python_bundle = _build_bundle(tmp_path / "python-output")
    powershell_output = tmp_path / "powershell-output"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "scripts" / "build-client-backend-bundle.ps1"),
            "-OutputRoot",
            str(powershell_output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    canonical_root = REPO_ROOT / "scripts" / "client-backend-bundle"
    powershell_bundle = powershell_output / "client-backend-bundle"
    for name in (
        "requirements-client.txt",
        "start-client-backend.ps1",
        "start-client-backend.bat",
        "README.client_backend.md",
    ):
        expected = (canonical_root / name).read_bytes()
        assert (python_bundle / name).read_bytes() == expected
        assert (powershell_bundle / name).read_bytes() == expected
