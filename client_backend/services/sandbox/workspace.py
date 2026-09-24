"""The workspace the sandbox account works in when none is configured.

The account may never open the user's repositories: a workspace grant is
inherited, so opening one would also hand the account the repo's ``.git`` and
any environment folder, both of which run code as the user (see
``runtime.protected_paths``). So when no workspace is configured, the sidecar
gives the account a clean workspace it owns instead -- under the profile,
holding no ``.git`` and no environment -- rather than the read-only runtime
folder, where it can create nothing.

To work on a project inside the sandbox, copy its source into this folder
(without ``.git`` or a virtual environment), let the model work, then review
the changes and apply them back to the real repository yourself. The account
never touches the repository, so it can never plant a git hook or a ``.pth``
file that would later run as you.
"""

from __future__ import annotations

from pathlib import Path

from client_backend.core.config import client_settings

__all__ = ["MANAGED_WORKSPACE_DIRNAME", "ensure_managed_workspace", "managed_workspace_root"]

MANAGED_WORKSPACE_DIRNAME = "workspace"


def managed_workspace_root() -> Path:
    """Where the sandbox account works when ``workspace_roots`` is empty."""

    return Path(client_settings.profile_root) / MANAGED_WORKSPACE_DIRNAME


def ensure_managed_workspace() -> Path:
    """Create the managed workspace if it is missing, and return it.

    Created by the signed-in user, who owns the profile, so the folder is the
    user's: the account is granted access to it at launch, it does not own it.
    """

    root = managed_workspace_root()
    root.mkdir(parents=True, exist_ok=True)
    return root
