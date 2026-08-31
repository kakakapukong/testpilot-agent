"""Bounded, transaction-aware chat history for the agent loop."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class BoundedContext:
    """Keep stable instruction/task anchors and a bounded tail of exchanges.

    An exchange is stored as one assistant message plus every tool result that
    answers its calls.  Pruning works only at exchange boundaries, preventing
    orphaned ``tool`` messages in the model-facing history.
    """

    def __init__(
        self,
        developer_message: Mapping[str, Any],
        user_message: Mapping[str, Any],
        *,
        max_recent_groups: int = 8,
        max_tool_content_chars: int = 8_000,
    ) -> None:
        if max_recent_groups < 0:
            raise ValueError("max_recent_groups must not be negative")
        if max_tool_content_chars < 1:
            raise ValueError("max_tool_content_chars must be at least 1")

        self._developer = _copy_message(developer_message)
        self._user = _copy_message(user_message)
        if self._developer.get("role") not in {"developer", "system"}:
            raise ValueError("developer_message must have a developer or system role")
        if self._user.get("role") != "user":
            raise ValueError("user_message must have a user role")
        self._max_recent_groups = max_recent_groups
        self._max_tool_content_chars = max_tool_content_chars
        self._groups: list[list[dict[str, Any]]] = []

    def append_transaction(
        self,
        assistant_message: Mapping[str, Any],
        tool_messages: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Append an assistant response and its complete tool-result set."""
        assistant = _copy_message(assistant_message)
        if assistant.get("role") != "assistant":
            raise ValueError("assistant_message must have an assistant role")

        call_ids = _tool_call_ids(assistant)
        copied_tools = [_copy_message(message) for message in tool_messages]
        tool_ids: set[str] = set()
        for message in copied_tools:
            if message.get("role") != "tool":
                raise ValueError("tool_messages must have a tool role")
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or tool_call_id not in call_ids:
                raise ValueError("every tool message must match an assistant tool call")
            if tool_call_id in tool_ids:
                raise ValueError("only one tool message is allowed per tool call")
            tool_ids.add(tool_call_id)
            message["content"] = _truncate_tool_content(
                message.get("content", ""), self._max_tool_content_chars
            )
        if tool_ids != call_ids:
            raise ValueError("every assistant tool call requires one tool message")

        self._groups.append([assistant, *copied_tools])
        if len(self._groups) > self._max_recent_groups:
            del self._groups[: len(self._groups) - self._max_recent_groups]

    def messages(self) -> list[dict[str, Any]]:
        """Return a defensive JSON-native copy suitable for an API request."""
        result = [_json_copy(self._developer), _json_copy(self._user)]
        for group in self._groups:
            result.extend(_json_copy(message) for message in group)
        return result


def _tool_call_ids(assistant: Mapping[str, Any]) -> set[str]:
    raw_calls = assistant.get("tool_calls", [])
    if raw_calls is None:
        return set()
    if not isinstance(raw_calls, list):
        raise TypeError("assistant tool_calls must be a list")
    call_ids: set[str] = set()
    for call in raw_calls:
        if not isinstance(call, Mapping) or not isinstance(call.get("id"), str):
            raise TypeError("assistant tool calls require string ids")
        call_id = call["id"]
        if call_id in call_ids:
            raise ValueError("assistant tool call ids must be unique")
        call_ids.add(call_id)
    return call_ids


def _truncate_tool_content(content: Any, limit: int) -> str:
    if not isinstance(content, str):
        raise TypeError("tool message content must be a string")
    if len(content) <= limit:
        return content
    marker = "…[cut]"
    if limit <= len(marker):
        return "…"
    return f"{content[: limit - len(marker)]}{marker}"


def _copy_message(message: Mapping[str, Any]) -> dict[str, Any]:
    return _json_copy(message)


def _json_copy(value: Any) -> Any:
    """Copy only JSON-compatible values, rejecting silently lossy objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("messages must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("message object keys must be strings")
            copied[key] = _json_copy(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]
    raise ValueError(f"messages must be JSON-native, got {type(value).__name__}")
