from collections.abc import Mapping
from typing import Any

import pytest

from testpilot.registry import ToolRegistry
from testpilot.types import ToolCall, ToolResult


class EchoTool:
    name = "echo"
    description = "Return supplied arguments."
    parameters: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "count": {"type": "integer", "minimum": 1, "maximum": 3},
            "enabled": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "mode": {"type": "string", "enum": ["brief", "full"]},
            "options": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
                "additionalProperties": False,
            },
        },
        "required": ["message"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        return ToolResult.success(dict(arguments))


class FailingTool(EchoTool):
    name = "failing"

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        raise RuntimeError("boom")


def test_schemas_export_chat_completion_shape_and_are_defensive() -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)

    schemas = registry.schemas()

    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Return supplied arguments.",
                "parameters": tool.parameters,
            },
        }
    ]
    assert "strict" not in schemas[0]["function"]
    schemas[0]["function"]["parameters"]["properties"]["message"]["type"] = "integer"
    assert (
        registry.schemas()[0]["function"]["parameters"]["properties"]["message"]["type"] == "string"
    )


def test_names_preserve_registration_order_and_duplicate_fails() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(FailingTool())

    assert registry.names() == ("echo", "failing")
    with pytest.raises(ValueError, match="echo"):
        registry.register(EchoTool())


def test_schemas_preserve_registration_order() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(FailingTool())

    assert [schema["function"]["name"] for schema in registry.schemas()] == ["echo", "failing"]


def test_unknown_tool_returns_failure() -> None:
    result = ToolRegistry().execute("missing", {})
    assert result.error_code == "unknown_tool"


@pytest.mark.parametrize(
    ("arguments", "path"),
    [
        ({}, "message"),
        ({"message": "ok", "extra": 1}, "extra"),
        ({"message": 1}, "message"),
        ({"message": "ok", "count": True}, "count"),
        ({"message": "ok", "enabled": "yes"}, "enabled"),
        ({"message": "ok", "tags": ["x", 2]}, "tags[1]"),
        ({"message": "ok", "mode": "other"}, "mode"),
        ({"message": "ok", "count": 0}, "count"),
        ({"message": "ok", "count": 4}, "count"),
        ({"message": "ok", "options": {}}, "options.label"),
        ({"message": "ok", "options": {"label": "x", "other": 1}}, "options.other"),
    ],
)
def test_invalid_arguments_report_field_path(arguments: dict[str, Any], path: str) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    result = registry.execute("echo", arguments)

    assert result.error_code == "invalid_arguments"
    assert result.error is not None
    assert path in result.error


def test_non_mapping_arguments_return_invalid_arguments() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    result = registry.execute("echo", ["not", "an", "object"])  # type: ignore[arg-type]
    assert result.error_code == "invalid_arguments"
    assert result.error is not None
    assert "$" in result.error


def test_invalid_schema_is_rejected_at_registration() -> None:
    tool = EchoTool()
    tool.parameters = {"type": "number"}
    with pytest.raises(ValueError, match="invalid schema"):
        ToolRegistry().register(tool)


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": "object", "properties": {"bad": {"type": "string", "enum": [object()]}}},
        {"type": "object", "properties": {1: {"type": "string"}}},
        {"type": "object", "properties": {"bad": {"type": "string", "example": object()}}},
        {"type": "object", "properties": {"bad": {"type": "number"}}},
        {"type": "object", "properties": {"bad": {}}},
        {"type": "object", "items": {"type": "string"}},
        {"type": "string", "properties": {}},
        {"type": "boolean", "required": []},
        {"type": "integer", "additionalProperties": False},
    ],
)
def test_non_json_native_or_structurally_invalid_schema_is_rejected(
    parameters: dict[str, Any],
) -> None:
    tool = EchoTool()
    tool.parameters = parameters

    with pytest.raises(ValueError, match="invalid schema"):
        ToolRegistry().register(tool)


@pytest.mark.parametrize(
    ("parameters", "arguments"),
    [
        (
            {"type": "object", "enum": [{"value": 1}]},
            {"value": True},
        ),
    ],
)
def test_enum_comparison_is_type_sensitive(
    parameters: dict[str, Any], arguments: dict[str, Any]
) -> None:
    tool = EchoTool()
    tool.parameters = parameters
    registry = ToolRegistry()
    registry.register(tool)

    result = registry.execute("echo", arguments)

    assert result.error_code == "invalid_arguments"


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": "object", "properties": {"flag": {"type": "boolean", "enum": [1]}}},
        {"type": "object", "properties": {"count": {"type": "integer", "enum": [True]}}},
    ],
)
def test_enum_members_must_match_the_node_type(parameters: dict[str, Any]) -> None:
    tool = EchoTool()
    tool.parameters = parameters

    with pytest.raises(ValueError, match="invalid schema"):
        ToolRegistry().register(tool)


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": "object", "x-note": float("nan")},
        {"type": "object", "x-note": float("inf")},
        {"type": "object", "x-note": float("-inf")},
        {"type": "string", "minLength": 1},
        {"type": "string", "minimum": 1},
        {"type": "boolean", "x-note": "unsupported"},
        {"type": "object", "items": {"type": "string"}},
        {"type": "array", "properties": {}},
        {"type": "string", "description": 1},
    ],
)
def test_non_finite_or_unsupported_schema_keywords_are_rejected(parameters: dict[str, Any]) -> None:
    tool = EchoTool()
    tool.parameters = parameters

    with pytest.raises(ValueError, match="invalid schema"):
        ToolRegistry().register(tool)


def test_tool_exception_and_bad_return_become_failures() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool())
    assert registry.execute("failing", {"message": "x"}).error_code == "tool_exception"

    class BadTool(EchoTool):
        name = "bad"

        def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
            return "not a result"  # type: ignore[return-value]

    registry.register(BadTool())
    assert registry.execute("bad", {"message": "x"}).error_code == "tool_exception"


def test_execute_passes_independent_plain_json_copy_and_success_result() -> None:
    class MutatingTool(EchoTool):
        name = "mutating"

        def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
            assert type(arguments) is dict
            assert type(arguments["tags"]) is list
            arguments["tags"].append("changed")
            return ToolResult.success("passed-through")

    registry = ToolRegistry()
    registry.register(MutatingTool())
    supplied = {"message": "x", "tags": ["original"]}

    result = registry.execute("mutating", supplied)

    assert result == ToolResult.success("passed-through")
    assert supplied == {"message": "x", "tags": ["original"]}


def test_execute_accepts_frozen_tool_call_arguments_with_nested_arrays() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    call = ToolCall(id="call_1", name="echo", arguments={"message": "x", "tags": ["one", "two"]})

    result = registry.execute(call.name, call.arguments)

    assert result == ToolResult.success({"message": "x", "tags": ["one", "two"]})


def test_execute_converts_copy_or_mapping_errors_to_invalid_arguments() -> None:
    class ExplodingMapping(Mapping[str, Any]):
        def __iter__(self) -> Any:
            raise RuntimeError("no iteration")

        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("no lookup")

    registry = ToolRegistry()
    registry.register(EchoTool())

    result = registry.execute("echo", ExplodingMapping())

    assert result.error_code == "invalid_arguments"
    assert result.error is not None
    assert "no iteration" in result.error


def test_registration_caches_tool_metadata_and_schema() -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)
    tool.name = "changed"
    tool.description = "Changed."
    tool.parameters["properties"]["message"]["type"] = "integer"

    assert registry.names() == ("echo",)
    function = registry.schemas()[0]["function"]
    assert function["name"] == "echo"
    assert function["description"] == "Return supplied arguments."
    assert function["parameters"]["properties"]["message"]["type"] == "string"


@pytest.mark.parametrize(
    ("name", "description"),
    [
        ("", "valid"),
        ("has space", "valid"),
        ("x" * 65, "valid"),
        ("valid", ""),
        ("valid", 1),
    ],
)
def test_registration_rejects_invalid_name_or_description(name: str, description: Any) -> None:
    tool = EchoTool()
    tool.name = name
    tool.description = description

    with pytest.raises(ValueError):
        ToolRegistry().register(tool)


@pytest.mark.parametrize(
    "arguments",
    [
        {1: "not a string key"},
        {"message": object()},
        {"message": "x", "tags": [object()]},
    ],
)
def test_execute_rejects_non_json_native_runtime_arguments(arguments: dict[Any, Any]) -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    tool.parameters = {"type": "object"}
    registry.register(tool)

    result = registry.execute("echo", arguments)

    assert result.error_code == "invalid_arguments"


@pytest.mark.parametrize("name", [123, ["echo"]])
def test_execute_rejects_non_string_or_unhashable_tool_name(name: Any) -> None:
    result = ToolRegistry().execute(name, {})  # type: ignore[arg-type]
    assert result.error_code == "unknown_tool"
