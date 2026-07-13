"""Stdlib-only runner for the echo-python fixture skill.

Reads a ``--message`` argument and prints a JSON object proving the
python_script runtime invoked this script directly. Used only by
tests/fixtures/skills/echo_python; requires no secrets, no network, and no
third-party dependencies.
"""

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    print(json.dumps({"echoed": args.message, "runtime": "python_script"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
