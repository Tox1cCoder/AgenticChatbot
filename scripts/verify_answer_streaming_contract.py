"""Verify AI SDK answer streaming against a designated test server.

Opt-in and live: sends one real chat turn, so it calls the configured model
provider. The deterministic gates are the pytest suites and the Node SDK
harness in ``tests/ai_sdk_wire``; run this only against a test environment
with a test account.

Configuration (environment only; the token is never printed):

    ANSWER_STREAM_VERIFY_URL              origin, e.g. http://127.0.0.1:8000 or the sidecar
    ANSWER_STREAM_VERIFY_TOKEN            bearer token for that origin
    ANSWER_STREAM_VERIFY_CONVERSATION_ID  an existing conversation owned by the test account
    ANSWER_STREAM_VERIFY_PROMPT           optional; defaults to a question that needs the web

Usage::

    .venv/Scripts/python.exe scripts/verify_answer_streaming_contract.py --origin direct
    .venv/Scripts/python.exe scripts/verify_answer_streaming_contract.py --origin sidecar \\
        --save-wire <dir>

``--save-wire`` writes the raw stream plus a ``manifest.json`` entry, so
``AI_SDK_FIXTURE_DIR=<dir> npm --prefix tests/ai_sdk_wire test`` parses the live
stream with the real SDK. Exit status: 0 all checks passed, 1 a check failed,
2 configuration error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_V1_HEADER = "x-vercel-ai-ui-message-stream"
_DEFAULT_PROMPT = "Search the web: what is the latest stable Python release? Cite sources."

#: Keys the pinned SDK accepts on the standard chunks this backend sends that
#: carry backend data; anything else fails the SDK's strict schema.
_STRICT_KEYS = {
    "error": {"type", "errorText"},
    "tool-output-available": {
        "type",
        "toolCallId",
        "output",
        "providerExecuted",
        "dynamic",
        "preliminary",
    },
    "finish": {"type", "finishReason", "messageMetadata"},
}


@dataclass
class WireReport:
    problems: list[str] = field(default_factory=list)
    message_id: str | None = None
    visible_answer: str = ""
    reconciles: int = 0
    finish_reason: str | None = None
    error_code: str | None = None
    first_answer_index: int | None = None
    frames: int = 0


def _data_lines(wire: str) -> list[str]:
    return [line[len("data: ") :] for line in wire.split("\n") if line.startswith("data: ")]


class _AnswerReducer:
    """The frontend reducer: authoritative ?? repair overlay ?? SDK text."""

    def __init__(self) -> None:
        self.sdk_text = ""
        self.repair: str | None = None
        self.authoritative: str | None = None

    def apply(self, kind: str, frame: dict[str, Any], data: dict[str, Any]) -> None:
        if kind == "text-delta":
            self.sdk_text += frame.get("delta", "")
        elif kind == "data-answer-reconcile":
            self.authoritative = data.get("content", "")
            self.repair = None
        elif kind == "data-answer-preview" and self.authoritative is None:
            self._preview(data)

    def _preview(self, data: dict[str, Any]) -> None:
        operation = data.get("operation")
        if operation == "replace":
            self.repair = data.get("text", "")
        elif operation == "append":
            self.repair = (self.repair or "") + data.get("text", "")
        elif operation == "clear":
            self.repair = None

    def visible(self) -> str:
        if self.authoritative is not None:
            return self.authoritative
        return self.repair if self.repair is not None else self.sdk_text


def _frame_problems(index: int, frame: dict[str, Any]) -> list[str]:
    kind = frame.get("type")
    if kind == "heartbeat":
        return [f"frame {index}: raw heartbeat frame (use a comment)"]
    allowed = _STRICT_KEYS.get(kind)
    if allowed is not None and not set(frame) <= allowed:
        return [f"frame {index}: {kind} carries non-SDK keys {sorted(set(frame) - allowed)}"]
    return []


def _record(report: WireReport, index: int, kind: str, frame: dict, data: dict) -> None:
    if kind == "start":
        report.message_id = frame.get("messageId")
    elif kind == "data-answer-reconcile":
        report.reconciles += 1
    elif kind == "data-stream-error":
        report.error_code = data.get("errorCode")
    elif kind == "finish":
        report.finish_reason = frame.get("finishReason")
    is_answer = kind == "text-delta" or (
        kind == "data-answer-preview" and data.get("operation") in {"replace", "append"}
    )
    if is_answer and report.first_answer_index is None:
        report.first_answer_index = index


def analyze_wire(wire: str) -> WireReport:
    """Check one complete AI SDK stream; compute what a conforming client shows."""
    report = WireReport()
    lines = _data_lines(wire)
    done = [index for index, raw in enumerate(lines) if raw == "[DONE]"]
    if len(done) != 1 or done[0] != len(lines) - 1:
        report.problems.append(f"expected exactly one trailing [DONE], found {len(done)}")

    reducer = _AnswerReducer()
    for index, raw in enumerate(lines):
        if raw == "[DONE]":
            continue
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            report.problems.append(f"frame {index} is not JSON")
            continue
        report.frames += 1
        report.problems += _frame_problems(index, frame)
        kind = frame.get("type")
        data = frame.get("data") if isinstance(frame.get("data"), dict) else {}
        _record(report, index, kind, frame, data)
        reducer.apply(kind, frame, data)

    if report.reconciles > 1:
        report.problems.append(f"expected at most one reconcile, found {report.reconciles}")
    report.visible_answer = reducer.visible()
    return report


def compare_with_history(message_id: str | None, visible: str, history: dict) -> list[str]:
    """The streamed answer must equal what history returns for the same message."""
    messages = ((history or {}).get("data") or {}).get("messages") or []
    match = next((m for m in messages if m.get("id") == message_id), None)
    if match is None:
        return [f"streamed message {message_id} is not in conversation history"]
    if (match.get("content") or "") != visible:
        return [
            "displayed answer differs from history "
            f"(shown {len(visible)} chars, saved {len(match.get('content') or '')} chars)"
        ]
    return []


def _save_wire(directory: Path, wire: str, *, origin: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"live-{origin}-{int(time.time())}.sse"
    (directory / name).write_text(wire, encoding="utf-8", newline="")
    manifest = directory / "manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    entries.append({"file": name, "expect": "accept", "requires": []})
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return directory / name


def _config() -> dict[str, str]:
    names = {
        "url": "ANSWER_STREAM_VERIFY_URL",
        "token": "ANSWER_STREAM_VERIFY_TOKEN",
        "conversation_id": "ANSWER_STREAM_VERIFY_CONVERSATION_ID",
    }
    missing = [env for env in names.values() if not os.environ.get(env)]
    if missing:
        raise SystemExit(f"missing configuration: {', '.join(missing)}")
    config = {key: os.environ[env] for key, env in names.items()}
    config["prompt"] = os.environ.get("ANSWER_STREAM_VERIFY_PROMPT") or _DEFAULT_PROMPT
    return config


def _run(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    import httpx

    headers = {"Authorization": f"Bearer {config['token']}"}
    body = {
        "messages": [
            {"role": "user", "parts": [{"type": "text", "text": config["prompt"]}]},
        ],
        "streamWebAnswerV1": not args.no_web_capability,
    }
    chat_url = f"{config['url'].rstrip('/')}/api/chat/{config['conversation_id']}"
    timeout = httpx.Timeout(connect=30, read=None, write=30, pool=30)
    started = time.monotonic()
    chunks: list[str] = []
    first_answer_at: float | None = None
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", chat_url, json=body, headers=headers) as response:
            if response.status_code != 200:
                return [f"chat returned HTTP {response.status_code}"]
            problems = []
            if response.headers.get(_V1_HEADER) != "v1":
                problems.append(f"missing {_V1_HEADER}: v1 header")
            for text in response.iter_text():
                chunks.append(text)
                if first_answer_at is None and '"type":"text-delta"' in text:
                    first_answer_at = time.monotonic()
        wire = "".join(chunks)
        report = analyze_wire(wire)
        problems += report.problems
        history_url = (
            f"{config['url'].rstrip('/')}/ai/conversations/{config['conversation_id']}"
            "/messages?limit=20&orderDirection=desc"
        )
        if report.finish_reason != "error":
            history = client.get(history_url, headers=headers).json()
            problems += compare_with_history(report.message_id, report.visible_answer, history)

    if args.save_wire:
        path = _save_wire(Path(args.save_wire), wire, origin=args.origin)
        print(f"saved stream: {path}")
    first = f"{first_answer_at - started:.2f}s" if first_answer_at else "none"
    print(
        f"origin={args.origin} frames={report.frames} reconciles={report.reconciles} "
        f"finish_reason={report.finish_reason or 'none'} error_code={report.error_code or 'none'} "
        f"first_text_delta={first} total={time.monotonic() - started:.2f}s "
        f"answer_chars={len(report.visible_answer)}"
    )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--origin", choices=("direct", "sidecar"), required=True)
    parser.add_argument("--save-wire", metavar="DIR")
    parser.add_argument(
        "--no-web-capability",
        action="store_true",
        help="omit streamWebAnswerV1 to check the validated fallback",
    )
    args = parser.parse_args(argv)
    try:
        config = _config()
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    problems = _run(args, config)
    for problem in problems:
        print(f"FAIL: {problem}")
    print("PASS" if not problems else f"{len(problems)} check(s) failed")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
