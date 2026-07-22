"""Dependency floors required by the model-usage analytics implementation."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def project_requirement(name: str) -> Requirement:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(value) for value in document["project"]["dependencies"]]
    return next(requirement for requirement in requirements if requirement.name.lower() == name)


def test_sqlalchemy_floor_supports_values_cte() -> None:
    requirement = project_requirement("sqlalchemy")

    assert Version("2.0.42") in requirement.specifier
    assert Version("2.0.41") not in requirement.specifier
    assert Version("3.0.0") not in requirement.specifier


def test_fastapi_floor_supports_pydantic_query_parameter_models() -> None:
    requirement = project_requirement("fastapi")

    assert Version("0.139.2") in requirement.specifier
    assert Version("0.139.1") not in requirement.specifier
    assert Version("0.140.0") not in requirement.specifier
