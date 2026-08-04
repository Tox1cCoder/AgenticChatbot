"""
Build the client backend source bundle into dist/.

Python port of scripts/build-client-backend-bundle.ps1 — produces the same
artifacts (dist/client-backend-bundle/ and dist/client-backend-bundle.zip)
on any platform:

    python scripts/build_client_backend_bundle.py [--output-root PATH]
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TEMPLATE_ROOT = REPO_ROOT / "scripts" / "client-backend-bundle"
BUNDLE_TEMPLATE_NAMES = (
    "requirements-client.txt",
    "start-client-backend.ps1",
    "start-client-backend.bat",
    "README.client_backend.md",
)

CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
CACHE_FILE_SUFFIXES = {".pyc", ".pyo"}


def _copy_mcp_servers(destination: Path) -> None:
    """Copy only the MCP server scripts this repository actually tracks.

    Not a ``copytree`` of the directory: developers keep unversioned server
    installations beside the tracked scripts -- OCR binaries, model weights --
    and a recursive copy sweeps all of it into a distributable. That both bloats
    the artifact by orders of magnitude and makes the build non-reproducible,
    since the result depends on what happens to sit in one machine's working
    tree. Only the tracked ``*.py`` servers are part of the product.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for script in sorted((REPO_ROOT / "app" / "ai" / "mcp_servers").glob("*.py")):
        shutil.copy2(script, destination / script.name)


def _ignore_caches(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in CACHE_DIR_NAMES}


def _remove_cache_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in CACHE_FILE_SUFFIXES:
            path.unlink()


def _copy_bundle_templates(bundle_root: Path) -> None:
    for name in BUNDLE_TEMPLATE_NAMES:
        shutil.copy2(BUNDLE_TEMPLATE_ROOT / name, bundle_root / name)


def build(output_root: Path) -> tuple[Path, Path]:
    bundle_root = output_root / "client-backend-bundle"
    bundle_zip = output_root / "client-backend-bundle.zip"

    # Empty the directory rather than removing it: on Windows the directory
    # node itself is often held open (shell cwd, IDE watcher) and rmdir fails.
    if bundle_root.exists():
        for child in bundle_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        bundle_root.mkdir(parents=True)

    shutil.copytree(
        REPO_ROOT / "client_backend",
        bundle_root / "client_backend",
        ignore=_ignore_caches,
    )
    shutil.copytree(
        REPO_ROOT / "shared",
        bundle_root / "shared",
        ignore=_ignore_caches,
    )

    app_dir = bundle_root / "app"
    (app_dir / "ai").mkdir(parents=True)
    (app_dir / "core").mkdir(parents=True)
    (app_dir / "schemas").mkdir(parents=True)
    (app_dir / "services").mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "app" / "ai" / "mcp_config.json",
        app_dir / "ai" / "mcp_config.json",
    )
    _copy_mcp_servers(app_dir / "ai" / "mcp_servers")
    shutil.copy2(
        REPO_ROOT / "app" / "core" / "config.py",
        app_dir / "core" / "config.py",
    )
    shutil.copy2(
        REPO_ROOT / "app" / "core" / "build_info.py",
        app_dir / "core" / "build_info.py",
    )
    shutil.copy2(
        REPO_ROOT / "app" / "core" / "mcp_adapter_utils.py",
        app_dir / "core" / "mcp_adapter_utils.py",
    )
    shutil.copy2(
        REPO_ROOT / "app" / "schemas" / "runtime_protocol.py",
        app_dir / "schemas" / "runtime_protocol.py",
    )
    shutil.copy2(
        REPO_ROOT / "app" / "services" / "widget_contract.py",
        app_dir / "services" / "widget_contract.py",
    )
    shutil.copy2(
        REPO_ROOT / "app" / "services" / "widget_runtime.py",
        app_dir / "services" / "widget_runtime.py",
    )
    shutil.copy2(
        REPO_ROOT / ".env.client.example",
        bundle_root / ".env.client.example",
    )

    _remove_cache_files(bundle_root)

    _copy_bundle_templates(bundle_root)
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "ai" / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "core" / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "schemas" / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "services" / "__init__.py").write_text("", encoding="utf-8")
    if bundle_zip.exists():
        bundle_zip.unlink()
    with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_root))

    return bundle_root, bundle_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the client backend bundle.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "dist"),
        help="Directory that receives client-backend-bundle/ and the zip.",
    )
    args = parser.parse_args()

    bundle_root, bundle_zip = build(Path(args.output_root))
    print("Built client backend bundle:")
    print(f"  Folder: {bundle_root}")
    print(f"  Zip:    {bundle_zip}")


if __name__ == "__main__":
    main()
