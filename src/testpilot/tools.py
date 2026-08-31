"""Local file tools backed by a confined :class:`Workspace`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from .types import ToolResult
from .workspace import Workspace, WorkspaceError


class _WorkspaceTool:
    """Common error conversion for workspace-backed tools."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    @staticmethod
    def _failure(error: WorkspaceError) -> ToolResult:
        return ToolResult.failure(error.message, error.code)

    @staticmethod
    def _arguments(
        arguments: object,
        parameters: Mapping[str, Any],
    ) -> dict[str, Any] | ToolResult:
        """Validate direct calls just as the registry validates model calls."""
        if not isinstance(arguments, Mapping):
            return ToolResult.failure("arguments must be an object", "invalid_arguments")
        try:
            copied = dict(arguments)
            if not all(isinstance(key, str) for key in copied):
                raise ValueError("argument names must be strings")
            properties = parameters["properties"]
            required = parameters["required"]
            unknown = set(copied).difference(properties)
            if unknown:
                raise ValueError("unexpected argument")
            if any(name not in copied for name in required):
                raise ValueError("required argument is missing")
            for name, value in copied.items():
                schema = properties[name]
                expected = schema["type"]
                if expected == "string" and not isinstance(value, str):
                    raise ValueError(f"{name} must be a string")
                if expected == "integer" and (
                    not isinstance(value, int) or isinstance(value, bool)
                ):
                    raise ValueError(f"{name} must be an integer")
                if expected == "integer" and "minimum" in schema and value < schema["minimum"]:
                    raise ValueError(f"{name} is below its minimum")
        except Exception as exc:  # noqa: BLE001 - untrusted mappings may fail during access.
            return ToolResult.failure(f"invalid arguments: {exc}", "invalid_arguments")
        return copied


class ListFilesTool(_WorkspaceTool):
    """List files in a workspace directory."""

    name = "list_files"
    description = "List files under a workspace-relative directory, with an optional glob."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative directory; defaults to '.'."},
            "glob": {"type": "string", "description": "Optional relative glob pattern."},
        },
        "required": [],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        validated = self._arguments(arguments, self.parameters)
        if isinstance(validated, ToolResult):
            return validated
        try:
            data = self.workspace.list_files(
                validated.get("path", "."),
                glob=validated.get("glob"),
            )
        except WorkspaceError as exc:
            return self._failure(exc)
        return ToolResult.success(data, truncated=data["truncated"])


class ReadFileTool(_WorkspaceTool):
    """Read bounded UTF-8 text from a workspace file."""

    name = "read_file"
    description = "Read a bounded UTF-8 file or inclusive 1-based line range."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        validated = self._arguments(arguments, self.parameters)
        if isinstance(validated, ToolResult):
            return validated
        try:
            data = self.workspace.read_file(
                validated["path"],
                start_line=validated.get("start_line"),
                end_line=validated.get("end_line"),
            )
        except WorkspaceError as exc:
            return self._failure(exc)
        return ToolResult.success(data, truncated=data["truncated"])


class SearchTextTool(_WorkspaceTool):
    """Search workspace text using literal substring matching."""

    name = "search_text"
    description = "Search UTF-8 workspace files for a literal substring (not a regular expression)."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Non-empty literal text to find."},
            "path": {"type": "string", "description": "Relative path; defaults to '.'."},
            "glob": {"type": "string", "description": "Optional relative glob pattern."},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        validated = self._arguments(arguments, self.parameters)
        if isinstance(validated, ToolResult):
            return validated
        try:
            data = self.workspace.search_text(
                validated["query"],
                validated.get("path", "."),
                glob=validated.get("glob"),
            )
        except WorkspaceError as exc:
            return self._failure(exc)
        return ToolResult.success(data, truncated=data["truncated"])


class EditFileTool(_WorkspaceTool):
    """Apply one exact text replacement to a workspace file."""

    name = "edit_file"
    description = "Replace old_text only when it occurs exactly once in a UTF-8 workspace file."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "old_text": {"type": "string", "description": "Exact unique text to replace."},
            "new_text": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        validated = self._arguments(arguments, self.parameters)
        if isinstance(validated, ToolResult):
            return validated
        try:
            data = self.workspace.edit_file(
                validated["path"],
                validated["old_text"],
                validated["new_text"],
            )
        except WorkspaceError as exc:
            return self._failure(exc)
        return ToolResult.success(data)


class WriteFileTool(_WorkspaceTool):
    """Atomically write a UTF-8 workspace file."""

    name = "write_file"
    description = "Atomically create or replace a UTF-8 file inside the workspace."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "content": {"type": "string", "description": "Complete UTF-8 text content."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        validated = self._arguments(arguments, self.parameters)
        if isinstance(validated, ToolResult):
            return validated
        try:
            data = self.workspace.write_file(
                validated["path"],
                validated["content"],
            )
        except WorkspaceError as exc:
            return self._failure(exc)
        return ToolResult.success(data)
