"""Credential-free Calendar dry-run fixture."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    subcommands = parser.add_subparsers(dest="resource", required=True)
    schedule = subcommands.add_parser("schedule")
    schedule_subcommands = schedule.add_subparsers(dest="action", required=True)
    create = schedule_subcommands.add_parser("create")
    create.add_argument("--calendar-id", required=True)
    create.add_argument("--title", required=True)
    create.add_argument("--start", required=True)
    create.add_argument("--end", required=True)
    create.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "calendar_id": args.calendar_id,
                "body": {
                    "summary": args.title,
                    "start": {"dateTime": args.start},
                    "end": {"dateTime": args.end},
                },
            }
        )
    )


if __name__ == "__main__":
    main()
