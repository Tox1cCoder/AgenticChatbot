"""Shared pytest configuration and fixtures.

This module is auto-loaded by pytest and provides shared utilities for all tests.
"""


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
