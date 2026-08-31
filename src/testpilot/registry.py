"""Static tool registration and local validation for tool calls."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Protocol

from .types import ToolResult


class Tool(Protocol):
    """A locally executable tool exposed to the language model."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, Any]: ...

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult: ...


class ToolRegistry:
    """An ordered, immutable-at-registration collection of tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, tool: Tool) -> None:
        """Register *tool*, rejecting duplicate names and unsupported schemas."""
        name = tool.name
        description = tool.description
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None:
            raise ValueError("tool name must match [A-Za-z0-9_-]{1,64}")
        if not isinstance(description, str) or not description:
            raise ValueError("tool description must be a non-empty string")
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        try:
            schema = _normalise_schema(tool.parameters, "$", root=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid schema: {exc}") from exc
        self._tools[name] = tool
        self._schemas[name] = schema
        self._descriptions[name] = description

    def names(self) -> tuple[str, ...]:
        """Return registered tool names in registration order."""
        return tuple(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Return defensive Chat Completions function-tool declarations."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": self._descriptions[name],
                    "parameters": _json_copy(self._schemas[name]),
                },
            }
            for name, tool in self._tools.items()
        ]

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Validate and execute a registered tool without leaking exceptions."""
        if not isinstance(name, str):
            return ToolResult.failure(f"unknown tool: {name!r}", "unknown_tool")
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(f"unknown tool: {name}", "unknown_tool")
        if not isinstance(arguments, Mapping):
            return ToolResult.failure("$: expected object", "invalid_arguments")

        try:
            normalized_arguments = _json_copy(arguments)
            error = _validate(self._schemas[name], normalized_arguments, "$")
        except Exception as exc:  # noqa: BLE001 - untrusted mappings may raise while inspected.
            return ToolResult.failure(
                f"invalid arguments: {type(exc).__name__}: {exc}", "invalid_arguments"
            )
        if error is not None:
            return ToolResult.failure(error, "invalid_arguments")
        try:
            result = tool.execute(normalized_arguments)
        except Exception as exc:  # noqa: BLE001 - registered tools are an exception boundary.
            return ToolResult.failure(
                f"tool {name} raised {type(exc).__name__}: {exc}", "tool_exception"
            )
        if not isinstance(result, ToolResult):
            return ToolResult.failure(f"tool {name} returned non-ToolResult", "tool_exception")
        return result


def _normalise_schema(schema: Any, path: str, *, root: bool = False) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        raise TypeError(f"{path}: schema must be an object")
    schema = _schema_copy(schema)
    schema_type = schema.get("type")
    if schema_type not in {"object", "string", "integer", "boolean", "array"}:
        raise ValueError(f"{path}.type: unsupported type {schema_type!r}")
    if root and schema_type != "object":
        raise ValueError(f"{path}.type: root schema must be object")
    allowed = {"type", "description", "enum"}
    allowed.update(
        {
            "object": {"properties", "required", "additionalProperties"},
            "array": {"items"},
            "integer": {"minimum", "maximum"},
        }.get(schema_type, set())
    )
    unsupported = set(schema).difference(allowed)
    if unsupported:
        raise ValueError(f"{path}.{min(unsupported)}: unsupported keyword")
    if "description" in schema and not isinstance(schema["description"], str):
        raise ValueError(f"{path}.description: must be a string")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"{path}.enum: must be an array")
    object_keywords = {"properties", "required", "additionalProperties"}
    if schema_type != "object" and object_keywords.intersection(schema):
        raise ValueError(f"{path}: object keywords require object type")
    if schema_type != "array" and "items" in schema:
        raise ValueError(f"{path}.items: only allowed for array type")
    if schema_type == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict) or not all(isinstance(key, str) for key in properties):
            raise ValueError(f"{path}.properties: must be an object with string keys")
        schema["properties"] = {
            key: _normalise_schema(value, f"{path}.{key}") for key, value in properties.items()
        }
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
            raise ValueError(f"{path}.required: must be an array of strings")
        if any(key not in properties for key in required):
            raise ValueError(f"{path}.required: names must be properties")
        schema["required"] = required
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, bool):
            raise ValueError(f"{path}.additionalProperties: must be boolean")
        schema["additionalProperties"] = additional
    elif schema_type == "array":
        if "items" not in schema:
            raise ValueError(f"{path}.items: required for array")
        schema["items"] = _normalise_schema(schema["items"], f"{path}.items")
    elif schema_type == "integer":
        for keyword in ("minimum", "maximum"):
            if keyword in schema and (
                not isinstance(schema[keyword], int) or isinstance(schema[keyword], bool)
            ):
                raise ValueError(f"{path}.{keyword}: must be an integer")
    if "enum" in schema:
        for index, member in enumerate(schema["enum"]):
            if _validate(schema, member, path, check_enum=False) is not None:
                raise ValueError(f"{path}.enum[{index}]: value does not match schema type")
    return schema


def _validate(
    schema: Mapping[str, Any], value: Any, path: str, *, check_enum: bool = True
) -> str | None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, Mapping):
            return f"{path}: expected object"
        properties = schema["properties"]
        for required in schema["required"]:
            if required not in value:
                return f"{path}.{required}: required property is missing"
        if not schema["additionalProperties"]:
            for key in value:
                if key not in properties:
                    return f"{path}.{key}: additional property is not allowed"
        for key, item in value.items():
            child = properties.get(key)
            if child is not None:
                error = _validate(child, item, f"{path}.{key}")
                if error:
                    return error
    elif schema_type == "string" and not isinstance(value, str):
        return f"{path}: expected string"
    elif schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"{path}: expected integer"
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: must be at least {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: must be at most {schema['maximum']}"
    elif schema_type == "boolean" and not isinstance(value, bool):
        return f"{path}: expected boolean"
    elif schema_type == "array":
        if not isinstance(value, list):
            return f"{path}: expected array"
        for index, item in enumerate(value):
            error = _validate(schema["items"], item, f"{path}[{index}]")
            if error:
                return error
    if (
        check_enum
        and "enum" in schema
        and not any(_json_equal(value, item) for item in schema["enum"])
    ):
        return f"{path}: value is not in enum"
    return None


def _json_copy(value: Any) -> Any:
    """Copy mappings and sequences into ordinary JSON-native containers."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"value is not JSON-native: {type(value).__name__}")


def _schema_copy(value: Any) -> Any:
    """Copy schema data while rejecting values outside the JSON data model."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("schema object keys must be strings")
        return {key: _schema_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_schema_copy(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("schema contains non-finite float")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ValueError(f"schema contains non-JSON value {type(value).__name__}")


def _json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values recursively without Python's bool/int coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return left == right
