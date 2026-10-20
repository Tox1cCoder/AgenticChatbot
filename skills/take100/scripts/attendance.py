"""Skill command: real check-in/check-out punches from the VT Check Attendance panel.

Run from the skill folder: ``python scripts/attendance.py``.
Credentials: ATTENDANCE_USERNAME and ATTENDANCE_PASSWORD for the panel; TAKE100_EMAIL and
TAKE100_PASSWORD to read the employee code and compare with the filed applications.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from take100_lib import bootstrap  # noqa: E402


def main() -> None:
    bootstrap.prepare(
        ("ATTENDANCE_USERNAME", "ATTENDANCE_PASSWORD", "TAKE100_EMAIL", "TAKE100_PASSWORD")
    )
    from take100_lib import attendance_api

    attendance_api.main()


if __name__ == "__main__":
    main()
