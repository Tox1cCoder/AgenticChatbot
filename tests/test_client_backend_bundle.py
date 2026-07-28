import ast
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
                alias.name
                for alias in node.names
                if alias.name.split(".")[0] in FIRST_PARTY_ROOTS
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
    assert (
        bundle_root / "app" / "ai" / "mcp_servers" / "time_server.py"
    ).is_file()
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
