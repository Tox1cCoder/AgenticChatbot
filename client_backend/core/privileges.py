"""Detect whether the sidecar holds administrator rights.

Every tool the server dispatches here runs with this process's token, so an
elevated sidecar hands a remote model administrator rights over the machine.
"""

from __future__ import annotations

import ctypes
import os
import sys

from client_backend.core.logging import get_logger

logger = get_logger(__name__)


def is_process_elevated() -> bool:
    """True for an elevated (UAC) Windows token or a POSIX root process.

    Under UAC an administrator's ordinary process runs with a filtered token,
    so this is true only when the sidecar was started "as administrator".
    """

    if sys.platform == "win32":
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except (AttributeError, OSError) as exc:
            # This guards against an accidental elevated launch, not an
            # adversary; refusing every start on a broken probe would lock out
            # ordinary users, so the failure is reported and treated as not
            # elevated.
            logger.warning("Could not determine whether the sidecar is elevated: %s", exc)
            return False
    return os.geteuid() == 0
