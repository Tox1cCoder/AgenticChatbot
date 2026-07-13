import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: echo-python MESSAGE")
    print(json.dumps({"message": sys.argv[1]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
