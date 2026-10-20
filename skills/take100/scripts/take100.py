"""Skill command: take100dot.com timesheets and leave/overtime applications.

Run from the skill folder: ``python scripts/take100.py --action list``.
Credentials: TAKE100_EMAIL and TAKE100_PASSWORD.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from take100_lib import bootstrap  # noqa: E402


def main() -> None:
    bootstrap.prepare(("TAKE100_EMAIL", "TAKE100_PASSWORD"))
    from take100_lib import take100_api

    take100_api.main()


if __name__ == "__main__":
    main()
