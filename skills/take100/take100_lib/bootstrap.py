"""Process setup shared by the skill's commands, run before any client module is imported.

The commands must work wherever the bundle is unpacked: under a skill runtime that injects
the credentials, or under Claude Code or another agent that runs them from a shell with
the credentials in the environment.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

SKILL_NAME = "take100"
BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PACKAGES = ("requests", "truststore")

# Direct CLI runs inside the Sample Chatbot repo only: the encrypted bindings live in this
# profile's secret store, which the skill runtime would otherwise inject. Anywhere else the
# repo is not found and the credentials must come from the environment.
LOCAL_PROFILE_USER_ID = "c3f04191-895a-4fb4-95c1-9f3fe66ac8f1"
LOCAL_SECRET_STORE = Path("client_backend") / "services" / "skill_runtime" / "secrets.py"


def prepare(required_bindings: tuple[str, ...]) -> None:
    """Check dependencies, trust the OS certificate store, and load credentials."""
    _require_packages()

    import truststore

    # take100dot.com sends its certificate without the AlphaSSL intermediate. Windows and
    # macOS complete the chain from their own trust store; certifi's bundle cannot.
    # Chain and hostname verification stay enabled.
    truststore.inject_into_ssl()

    # Work content and employee names are Vietnamese; a cp1252 console would fail on them.
    sys.stdout.reconfigure(encoding="utf-8")
    _load_local_bindings(required_bindings)


def _require_packages() -> None:
    missing = [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
    if not missing:
        return
    requirements = BUNDLE_ROOT / "requirements.txt"
    error = (
        f"Missing Python packages: {', '.join(missing)}. "
        f'Install them with: python -m pip install -r "{requirements}"'
    )
    print(json.dumps({"success": False, "error": error}, ensure_ascii=False, indent=2))
    sys.exit(1)


def _find_local_repo() -> Path | None:
    for parent in BUNDLE_ROOT.parents:
        if (parent / LOCAL_SECRET_STORE).is_file():
            return parent
    return None


def _load_local_bindings(required_bindings: tuple[str, ...]) -> None:
    """Fill missing credentials from the local profile store when this bundle sits in the repo."""
    if all(os.getenv(name) for name in required_bindings):
        return
    repo_root = _find_local_repo()
    if repo_root is None:
        return
    sys.path.insert(0, str(repo_root))
    from client_backend.services.skill_runtime.secrets import SkillSecretStore

    class LocalProfileStore(SkillSecretStore):
        @staticmethod
        def _resolve_user_id():
            return LOCAL_PROFILE_USER_ID

    for name, value in LocalProfileStore().get_for_skill(SKILL_NAME).items():
        os.environ.setdefault(name, value)
