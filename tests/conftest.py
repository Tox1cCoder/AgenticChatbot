"""Shared pytest configuration and fixtures.

This module is auto-loaded by pytest and provides shared utilities for all tests.
"""

from __future__ import annotations

import atexit
import json
import os
import tempfile
from pathlib import Path

# Set this before pytest imports application modules. Client settings otherwise
# fall through to the developer's live LOCALAPPDATA profile, where enabled MCP
# processes must never be started by the test suite.
_PYTEST_RUNTIME = tempfile.TemporaryDirectory(prefix="sample-chatbot-pytest-")
_PYTEST_MCP_CONFIG = Path(_PYTEST_RUNTIME.name) / "mcp_config.json"
_PYTEST_MCP_CONFIG.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
os.environ["CLIENT_MCP_CONFIG_PATH"] = str(_PYTEST_MCP_CONFIG)
atexit.register(_PYTEST_RUNTIME.cleanup)


class FakePopen:
    """Minimal subprocess.Popen stub that records the command and supports wait()."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.pid = 99999

    def wait(self):
        return 0

    def terminate(self):
        pass

    def poll(self):
        return 0


# Export FakePopen for use in test modules
__all__ = ["FakePopen"]
