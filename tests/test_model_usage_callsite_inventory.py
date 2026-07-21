"""Reviewed inventory of terminal provider calls and provider-client constructors."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "model_usage_callsite_manifest.json"
ALLOWED_DISPOSITIONS = {
    "instrumented",
    "local_model",
    "non_provider_tool_dispatch",
    "workflow_stream",
    "startup_validation",
}
SIMPLE_TERMINALS = {
    "generate_content",
    "generate_content_stream",
    "embed_content",
    "ainvoke",
    "invoke",
    "astream",
    "stream",
    "agenerate",
    "generate",
}
COMPOUND_TERMINALS = {
    "responses.create",
    "chat.completions.create",
    "images.generate",
    "images.edit",
}
CONSTRUCTORS = {
    "ChatGoogleGenerativeAI",
    "ChatOpenAI",
    "genai.Client",
    "OpenAI",
    "AsyncOpenAI",
}


def _call_chain(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_chain(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _call_chain(node.func)
    if isinstance(node, ast.Subscript):
        return _call_chain(node.value)
    return None


def _is_terminal_call(chain: str) -> bool:
    leaf = chain.rsplit(".", 1)[-1]
    return (
        leaf in SIMPLE_TERMINALS
        or any(chain.endswith(f".{suffix}") or chain == suffix for suffix in COMPOUND_TERMINALS)
        or chain in CONSTRUCTORS
        or (leaf in {"ChatGoogleGenerativeAI", "ChatOpenAI", "OpenAI", "AsyncOpenAI"})
    )


def _thread_dispatch_callable(node: ast.Call, chain: str) -> str | None:
    """Return a provider callable handed to a standard thread dispatcher."""
    if chain in {"asyncio.to_thread", "to_thread"} and node.args:
        return _call_chain(node.args[0])
    if chain.endswith(".run_in_executor") and len(node.args) >= 2:
        return _call_chain(node.args[1])
    return None


class _CallsiteVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.scope: list[str] = []
        self.callsites: list[dict[str, str]] = []

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        chain = _call_chain(node.func)
        if chain and _is_terminal_call(chain):
            self.callsites.append(
                {
                    "file": self.relative_path,
                    "function": ".".join(self.scope) or "<module>",
                    "call_chain": chain,
                }
            )
        elif chain:
            dispatched = _thread_dispatch_callable(node, chain)
            if dispatched and _is_terminal_call(dispatched):
                self.callsites.append(
                    {
                        "file": self.relative_path,
                        "function": ".".join(self.scope) or "<module>",
                        "call_chain": dispatched,
                    }
                )
        self.generic_visit(node)


def _tracked_python_sources() -> list[Path]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "app", "client_backend"],
        cwd=ROOT,
    ).decode("utf-8")
    return [
        ROOT / relative
        for relative in tracked.split("\0")
        if relative.endswith(".py") and (ROOT / relative).is_file()
    ]


def _discover_callsites() -> list[dict[str, str]]:
    discovered = []
    for path in _tracked_python_sources():
        relative = path.relative_to(ROOT).as_posix()
        visitor = _CallsiteVisitor(relative)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=relative))
        discovered.extend(visitor.callsites)
    return sorted(discovered, key=lambda entry: tuple(entry.values()))


def _test_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _test_function_symbols(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    symbols = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    symbols.update(node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute))
    return symbols


def test_terminal_model_calls_match_reviewed_manifest_exactly():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_identities = [
        {key: entry[key] for key in ("file", "function", "call_chain")} for entry in manifest
    ]

    assert _discover_callsites() == sorted(
        expected_identities, key=lambda entry: tuple(entry.values())
    )
    assert all(entry["disposition"] in ALLOWED_DISPOSITIONS for entry in manifest)


def test_inventory_contains_only_git_tracked_source_files():
    tracked = {
        line.strip()
        for line in subprocess.check_output(
            ["git", "ls-files", "--", "app", "client_backend"],
            cwd=ROOT,
            text=True,
        ).splitlines()
        if line.strip().endswith(".py")
    }

    assert tracked
    assert all(entry["file"] in tracked for entry in _discover_callsites())


def test_inventory_detects_provider_callable_passed_to_asyncio_to_thread():
    assert {
        "file": "app/ai/agents/router.py",
        "function": "Router._call_llm._generate",
        "call_chain": "self.gemini_client.models.generate_content",
    } in _discover_callsites()


def test_startup_validation_disposition_is_never_used_for_generation_calls():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    constructor_leaves = {
        "ChatGoogleGenerativeAI",
        "ChatOpenAI",
        "Client",
        "OpenAI",
        "AsyncOpenAI",
    }

    for entry in manifest:
        if entry["disposition"] == "startup_validation":
            assert entry["call_chain"].rsplit(".", 1)[-1] in constructor_leaves, entry


def test_each_instrumented_callsite_names_an_operation_and_exercising_test():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest:
        if entry["disposition"] != "instrumented":
            continue
        operation = entry.get("operation")
        exercising_test = entry.get("exercising_test")
        assert isinstance(operation, str) and operation and len(operation) <= 64, entry
        assert isinstance(exercising_test, str) and "::" in exercising_test, entry
        relative_path, function = exercising_test.split("::", 1)
        test_path = ROOT / relative_path
        assert test_path.is_file(), entry
        assert function in _test_functions(test_path), entry
        exercise_symbol = entry.get("exercise_symbol")
        if exercise_symbol is not None:
            assert exercise_symbol in _test_function_symbols(test_path, function), entry
