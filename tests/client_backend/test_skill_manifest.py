"""Tests for shared.skills.manifest: skill.json model + validation."""

import pytest
from pydantic import ValidationError

from shared.skills.manifest import SkillManifest, load_manifest


def _valid_manifest_dict() -> dict:
    return {
        "schema_version": "1.0",
        "name": "example-calendar",
        "display_name": "Example Calendar",
        "description": "Inspect and manage calendar events.",
        "runtime": {
            "type": "python_module",
            "module": "skills.example_calendar.cli",
            "entrypoint": "cli",
        },
        "dependencies": {"python": ["click>=8"], "node": [], "system": []},
        "secrets": [
            {
                "name": "EXAMPLE_CALENDAR_ACCESS_TOKEN",
                "required": True,
                "description": "OAuth access token with calendar scopes.",
            }
        ],
        "permissions": ["network:api.example.com", "calendar:read"],
        "capabilities": [
            {
                "name": "event_list",
                "description": "List events from a calendar.",
                "input_schema": {
                    "type": "object",
                    "properties": {"time_min": {"type": "string"}},
                    "required": ["time_min"],
                },
                "execution": {
                    "argv": ["--json", "event", "list", "--time-min", "{time_min}"]
                },
                "permissions": ["calendar:read"],
                "secrets": ["EXAMPLE_CALENDAR_ACCESS_TOKEN"],
                "mutation": False,
            }
        ],
    }


class TestValidManifest:
    def test_parses_full_manifest_and_exposes_expected_fields(self):
        manifest = load_manifest(_valid_manifest_dict())

        assert isinstance(manifest, SkillManifest)
        assert manifest.schema_version == "1.0"
        assert manifest.name == "example-calendar"
        assert manifest.display_name == "Example Calendar"
        assert manifest.runtime.type == "python_module"
        assert manifest.runtime.module == "skills.example_calendar.cli"
        assert manifest.runtime.entrypoint == "cli"
        assert manifest.dependencies.python == ["click>=8"]
        assert manifest.dependencies.node == []
        assert manifest.secrets[0].name == "EXAMPLE_CALENDAR_ACCESS_TOKEN"
        assert manifest.secrets[0].required is True
        assert manifest.permissions == ["network:api.example.com", "calendar:read"]

        assert len(manifest.capabilities) == 1
        capability = manifest.capabilities[0]
        assert capability.name == "event_list"
        assert capability.description == "List events from a calendar."
        assert capability.input_schema["type"] == "object"
        assert capability.execution.argv[0] == "--json"
        assert capability.execution.json_output is False
        assert capability.permissions == ["calendar:read"]
        assert capability.secrets == ["EXAMPLE_CALENDAR_ACCESS_TOKEN"]
        assert capability.mutation is False

    def test_optional_sections_default_when_omitted(self):
        data = _valid_manifest_dict()
        del data["dependencies"]
        del data["secrets"]
        del data["permissions"]
        del data["display_name"]

        manifest = load_manifest(data)

        assert manifest.dependencies.python == []
        assert manifest.dependencies.node == []
        assert manifest.dependencies.system == []
        assert manifest.secrets == []
        assert manifest.permissions == []
        assert manifest.display_name is None

    def test_provider_neutral_manifest_has_no_google_specific_fields(self):
        manifest = load_manifest(_valid_manifest_dict())

        dumped = str(manifest.model_dump()).lower()
        for banned_token in ("google", "gcal", "oauth2client"):
            assert banned_token not in dumped

        assert manifest.runtime.type in {"python_module", "python_script", "binary"}


class TestMissingRequiredFields:
    def test_capability_missing_description_raises(self):
        data = _valid_manifest_dict()
        del data["capabilities"][0]["description"]

        with pytest.raises(ValidationError):
            load_manifest(data)

    def test_capability_empty_description_raises(self):
        data = _valid_manifest_dict()
        data["capabilities"][0]["description"] = "   "

        with pytest.raises(ValidationError):
            load_manifest(data)

    def test_capability_missing_name_raises(self):
        data = _valid_manifest_dict()
        del data["capabilities"][0]["name"]

        with pytest.raises(ValidationError):
            load_manifest(data)

    def test_capability_missing_input_schema_raises(self):
        data = _valid_manifest_dict()
        del data["capabilities"][0]["input_schema"]

        with pytest.raises(ValidationError):
            load_manifest(data)

    def test_capability_missing_execution_raises(self):
        data = _valid_manifest_dict()
        del data["capabilities"][0]["execution"]

        with pytest.raises(ValidationError):
            load_manifest(data)

    def test_manifest_missing_capabilities_raises(self):
        data = _valid_manifest_dict()
        del data["capabilities"]

        with pytest.raises(ValidationError):
            load_manifest(data)

    def test_runtime_missing_type_raises(self):
        data = _valid_manifest_dict()
        del data["runtime"]["type"]

        with pytest.raises(ValidationError):
            load_manifest(data)

    @pytest.mark.parametrize("field", ["schema_version", "name", "description", "runtime"])
    def test_manifest_missing_top_level_required_field_raises(self, field):
        data = _valid_manifest_dict()
        del data[field]

        with pytest.raises(ValidationError):
            load_manifest(data)

    def test_manifest_empty_description_raises(self):
        data = _valid_manifest_dict()
        data["description"] = "   "

        with pytest.raises(ValidationError):
            load_manifest(data)


class TestRuntimeTypeValidation:
    @pytest.mark.parametrize("supported_type", ["python_module", "python_script", "binary"])
    def test_supported_runtime_types_are_accepted(self, supported_type):
        data = _valid_manifest_dict()
        data["runtime"] = {"type": supported_type}

        manifest = load_manifest(data)

        assert manifest.runtime.type == supported_type

    @pytest.mark.parametrize("bad_type", ["totally_unknown", "http_api"])
    def test_unknown_runtime_type_raises_and_names_it(self, bad_type):
        data = _valid_manifest_dict()
        data["runtime"]["type"] = bad_type

        with pytest.raises(ValidationError, match=bad_type):
            load_manifest(data)

    @pytest.mark.parametrize("reserved_type", ["shell", "node_package", "mcp_server"])
    def test_reserved_runtime_type_raises_and_says_reserved(self, reserved_type):
        data = _valid_manifest_dict()
        data["runtime"]["type"] = reserved_type

        with pytest.raises(ValidationError) as exc_info:
            load_manifest(data)

        message = str(exc_info.value)
        assert reserved_type in message
        assert "reserved" in message.lower()


class TestDuplicateCapabilityNames:
    def test_duplicate_capability_names_raise_and_name_the_duplicate(self):
        data = _valid_manifest_dict()
        data["capabilities"].append(dict(data["capabilities"][0]))

        with pytest.raises(ValidationError) as exc_info:
            load_manifest(data)

        assert "event_list" in str(exc_info.value)


class TestCapabilityNameSafety:
    @pytest.mark.parametrize(
        "unsafe_name",
        ["event list", "event.list", "1event", "event::list", "event-list", ""],
    )
    def test_unsafe_capability_name_raises(self, unsafe_name):
        data = _valid_manifest_dict()
        data["capabilities"][0]["name"] = unsafe_name

        with pytest.raises(ValidationError):
            load_manifest(data)

    @pytest.mark.parametrize("safe_name", ["event_list", "EventList", "a", "a1_b2"])
    def test_safe_capability_name_is_accepted(self, safe_name):
        data = _valid_manifest_dict()
        data["capabilities"][0]["name"] = safe_name

        manifest = load_manifest(data)

        assert manifest.capabilities[0].name == safe_name


class TestInputSchemaMustBeDict:
    @pytest.mark.parametrize("bad_schema", [["not", "a", "dict"], "a string", 123, None])
    def test_non_dict_input_schema_raises(self, bad_schema):
        data = _valid_manifest_dict()
        data["capabilities"][0]["input_schema"] = bad_schema

        with pytest.raises(ValidationError):
            load_manifest(data)

    def test_empty_input_schema_raises(self):
        data = _valid_manifest_dict()
        data["capabilities"][0]["input_schema"] = {}

        with pytest.raises(ValidationError):
            load_manifest(data)


class TestExecutionSpec:
    def test_empty_execution_object_raises_because_argv_is_required(self):
        data = _valid_manifest_dict()
        data["capabilities"][0]["execution"] = {}

        with pytest.raises(ValidationError):
            load_manifest(data)

    def test_empty_argv_is_accepted_runtimes_may_take_no_arguments(self):
        # Intentional: a python_module/binary capability may invoke its
        # entrypoint with no positional argv. Only input_schema is required
        # to be non-empty, not argv.
        data = _valid_manifest_dict()
        data["capabilities"][0]["execution"] = {"argv": []}

        manifest = load_manifest(data)

        assert manifest.capabilities[0].execution.argv == []


class TestSchemaVersion:
    def test_unsupported_schema_version_raises(self):
        data = _valid_manifest_dict()
        data["schema_version"] = "2.0"

        with pytest.raises(ValidationError, match="2.0"):
            load_manifest(data)

    def test_supported_schema_version_is_accepted(self):
        data = _valid_manifest_dict()
        data["schema_version"] = "1.0"

        manifest = load_manifest(data)

        assert manifest.schema_version == "1.0"


class TestSkillName:
    @pytest.mark.parametrize("bad_name", ["", "-leading-hyphen", "has space", "has.dot"])
    def test_invalid_skill_name_raises(self, bad_name):
        data = _valid_manifest_dict()
        data["name"] = bad_name

        with pytest.raises(ValidationError):
            load_manifest(data)

    def test_skill_name_allows_hyphen(self):
        data = _valid_manifest_dict()
        data["name"] = "example-calendar"

        manifest = load_manifest(data)

        assert manifest.name == "example-calendar"
