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
    # Thin ChatGoogleGenerativeAI subclass that normalizes Gemini thought
    # blocks into standard reasoning blocks; still a provider client
    # constructor and so still inventoried.
    "ReasoningNormalizedChatGoogleGenerativeAI",
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


class _CallsiteVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.scope: list[str] = []
        self.callsites: list[dict[str, str]] = []
        self.aliases: dict[str, str] = {}

    @staticmethod
    def _canonical_chain(chain: str) -> str:
        aliases = {
            "google.genai.Client": "genai.Client",
            "langchain_google_genai.ChatGoogleGenerativeAI": "ChatGoogleGenerativeAI",
            "langchain_openai.ChatOpenAI": "ChatOpenAI",
            "gemini_content.ReasoningNormalizedChatGoogleGenerativeAI": (
                "ReasoningNormalizedChatGoogleGenerativeAI"
            ),
        }
        return aliases.get(chain, chain)

    def _resolve_chain(self, node: ast.AST) -> str | None:
        chain = _call_chain(node)
        if not chain:
            return None
        root, separator, remainder = chain.partition(".")
        resolved_root = self.aliases.get(root, root)
        resolved = f"{resolved_root}.{remainder}" if separator else resolved_root
        return self._canonical_chain(resolved)

    def _thread_dispatch_callable(self, node: ast.Call, chain: str) -> str | None:
        """Return a provider callable handed to a standard thread dispatcher."""
        callable_node: ast.AST | None = None
        if chain in {"asyncio.to_thread", "to_thread"} and node.args:
            callable_node = node.args[0]
        elif chain.endswith(".run_in_executor") and len(node.args) >= 2:
            callable_node = node.args[1]
        if callable_node is None:
            return None

        if isinstance(callable_node, ast.Call):
            wrapper = self._resolve_chain(callable_node.func)
            if wrapper == "functools.partial" and callable_node.args:
                callable_node = callable_node.args[0]
        return self._resolve_chain(callable_node)

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

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            local_name = imported.asname or imported.name.split(".", 1)[0]
            self.aliases[local_name] = imported.name if imported.asname else local_name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        for imported in node.names:
            local_name = imported.asname or imported.name
            self.aliases[local_name] = self._canonical_chain(f"{node.module}.{imported.name}")

    def visit_Assign(self, node: ast.Assign) -> None:
        resolved: str | None = None
        if isinstance(node.value, ast.Call) and self._resolve_chain(node.value.func) == "getattr":
            if (
                len(node.value.args) >= 2
                and isinstance(node.value.args[1], ast.Constant)
                and isinstance(node.value.args[1].value, str)
            ):
                owner = self._resolve_chain(node.value.args[0])
                if owner:
                    candidate = self._canonical_chain(f"{owner}.{node.value.args[1].value}")
                    if _is_terminal_call(candidate):
                        resolved = candidate
        elif isinstance(node.value, (ast.Name, ast.Attribute)):
            candidate = self._resolve_chain(node.value)
            if candidate and _is_terminal_call(candidate):
                resolved = candidate

        if resolved:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = resolved
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        chain = self._resolve_chain(node.func)
        if chain and _is_terminal_call(chain):
            self.callsites.append(
                {
                    "file": self.relative_path,
                    "function": ".".join(self.scope) or "<module>",
                    "call_chain": chain,
                }
            )
        elif chain:
            dispatched = self._thread_dispatch_callable(node, chain)
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


def _discover_source(source: str) -> list[dict[str, str]]:
    visitor = _CallsiteVisitor("sample.py")
    visitor.visit(ast.parse(source))
    return visitor.callsites


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
    symbols.update(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    return symbols


def _test_function_assertion_symbols(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    symbols: set[str] = set()
    for assertion in (node for node in ast.walk(function) if isinstance(node, ast.Assert)):
        symbols.update(node.id for node in ast.walk(assertion.test) if isinstance(node, ast.Name))
        symbols.update(
            node.attr for node in ast.walk(assertion.test) if isinstance(node, ast.Attribute)
        )
        symbols.update(
            node.value
            for node in ast.walk(assertion.test)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
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


def test_inventory_resolves_imported_constructor_aliases():
    discovered = _discover_source(
        """
from google.genai import Client as GeminiClient

def build():
    return GeminiClient(api_key="secret")
"""
    )

    assert discovered == [
        {
            "file": "sample.py",
            "function": "build",
            "call_chain": "genai.Client",
        }
    ]


def test_inventory_resolves_safe_getattr_constructor_indirection():
    discovered = _discover_source(
        """
import openai

def build():
    cls = getattr(openai, "AsyncOpenAI")
    return cls(api_key="secret")
"""
    )

    assert discovered == [
        {
            "file": "sample.py",
            "function": "build",
            "call_chain": "openai.AsyncOpenAI",
        }
    ]


def test_inventory_unwraps_partial_passed_to_executor():
    discovered = _discover_source(
        """
from functools import partial as bind

async def call(loop, client):
    await loop.run_in_executor(
        None,
        bind(client.models.generate_content, model="gemini-test"),
    )
"""
    )

    assert discovered == [
        {
            "file": "sample.py",
            "function": "call",
            "call_chain": "client.models.generate_content",
        }
    ]


def test_inventory_contains_dynamic_async_openai_constructor():
    assert {
        "file": "app/services/provider_service.py",
        "function": "ProviderService._fetch_openai_models",
        "call_chain": "openai.AsyncOpenAI",
    } in _discover_callsites()


def test_startup_validation_disposition_is_never_used_for_generation_calls():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    constructor_leaves = {
        "ChatGoogleGenerativeAI",
        "ReasoningNormalizedChatGoogleGenerativeAI",
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
        proof_symbols = _test_function_symbols(test_path, function)
        assertion_symbols = _test_function_assertion_symbols(test_path, function)
        exercise_symbol = entry.get("exercise_symbol")
        assert isinstance(exercise_symbol, str) and exercise_symbol, entry
        assert exercise_symbol in proof_symbols, entry
        assert entry["operation"] in assertion_symbols, entry
