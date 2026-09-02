"""Model adapters that normalize tool-calling responses for TestPilot."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from testpilot.context import _json_copy
from testpilot.types import AssistantTurn, ToolCall


class ModelError(RuntimeError):
    """A normalized model-adapter failure safe to surface to the agent loop."""

    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ModelClient(Protocol):
    """The small model boundary used by :class:`testpilot.agent.AgentRunner`."""

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: Any = "auto",
    ) -> AssistantTurn:
        """Return one normalized assistant turn for the given chat state."""


class FakeModel:
    """Deterministic scripted model used for offline loop tests and demos."""

    def __init__(self, scripted_turns: Sequence[AssistantTurn]) -> None:
        self._turns = list(scripted_turns)
        self.received_inputs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        self.received_tool_choices: list[Any] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: Any = "auto",
    ) -> AssistantTurn:
        self.received_inputs.append((_json_copy(messages), _json_copy(tools)))
        self.received_tool_choices.append(tool_choice)
        if not self._turns:
            raise ModelError("scripted model is exhausted", code="model_exhausted", retryable=False)
        return self._turns.pop(0)


class OpenAIChatModel:
    """OpenAI-compatible Chat Completions adapter with bounded retries.

    A supplied ``client`` is used as-is, so its own retry policy remains the
    caller's responsibility.  Clients created here explicitly disable SDK
    retries so this adapter's three-attempt maximum stays meaningful.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        max_retries: int = 2,
        sleep_fn: Any = time.sleep,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-blank string")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._model = model.strip()
        self._api_key = api_key
        self._base_url = base_url
        self._client = client
        self._max_retries = max_retries
        self._sleep_fn = sleep_fn

    def __repr__(self) -> str:
        return (
            "OpenAIChatModel("
            f"model={self._model!r}, api_key_configured={self._api_key is not None}, "
            f"base_url_configured={self._base_url is not None})"
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: Any = "auto",
    ) -> AssistantTurn:
        request_messages = _compatible_chat_messages(messages)
        request_tools = _json_copy(tools)
        client = self._get_client()
        for attempt in range(self._max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=_json_copy(request_messages),
                    tools=_json_copy(request_tools),
                    tool_choice=tool_choice,
                )
            except Exception as error:  # noqa: BLE001 - SDK exceptions vary by client version.
                if _is_authentication_error(error):
                    raise ModelError(
                        "model authentication or permission failed",
                        code="model_authentication_failed",
                        retryable=False,
                    ) from None
                transient = _is_transient_error(error)
                if transient and attempt < self._max_retries:
                    self._sleep_fn(0.05 * (2**attempt))
                    continue
                raise ModelError(
                    "transient model request failed" if transient else "model request failed",
                    code="model_transient_failure" if transient else "model_request_failed",
                    retryable=transient,
                ) from None
            return _normalize_response(response)
        raise AssertionError("unreachable retry loop")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError:
            raise ModelError(
                "OpenAI SDK is not installed; install the api extra to make requests",
                code="sdk_unavailable",
                retryable=False,
            ) from None
        options: dict[str, Any] = {"max_retries": 0}
        if self._api_key is not None:
            options["api_key"] = self._api_key
        if self._base_url is not None:
            options["base_url"] = self._base_url
        self._client = OpenAI(**options)
        return self._client


def _compatible_chat_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy chat messages and map OpenAI-only roles for compatible providers."""
    compatible = _json_copy(messages)
    for message in compatible:
        if isinstance(message, dict) and message.get("role") == "developer":
            message["role"] = "system"
    return compatible


def _normalize_response(response: Any) -> AssistantTurn:
    choices = _field(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise ModelError(
            "model response did not contain a choice",
            code="invalid_model_response",
            retryable=False,
        )
    message = _field(choices[0], "message")
    if message is None:
        raise ModelError(
            "model response choice did not contain a message",
            code="invalid_model_response",
            retryable=False,
        )
    raw_content = _field(message, "content")
    content = "" if raw_content is None else raw_content
    if not isinstance(content, str):
        raise ModelError(
            "model message content was not text", code="invalid_model_response", retryable=False
        )
    raw_calls = _field(message, "tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        raise ModelError(
            "model tool calls were malformed", code="invalid_model_response", retryable=False
        )
    calls = tuple(_normalize_tool_call(raw_call) for raw_call in raw_calls)
    return AssistantTurn(content=content, tool_calls=calls)


def _normalize_tool_call(raw_call: Any) -> ToolCall:
    call_id = _field(raw_call, "id")
    function = _field(raw_call, "function")
    name = _field(function, "name") if function is not None else None
    raw_arguments = _field(function, "arguments") if function is not None else None
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        raise ModelError(
            "model tool call lacked an id or function name",
            code="invalid_model_response",
            retryable=False,
        )
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ToolCall(
            id=call_id,
            name=name,
            arguments={},
            argument_error="tool arguments are not valid JSON",
        )
    if not isinstance(arguments, dict):
        return ToolCall(
            id=call_id,
            name=name,
            arguments={},
            argument_error="tool arguments must be a JSON object",
        )
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_authentication_error(error: Exception) -> bool:
    name = type(error).__name__.lower()
    return _status_code(error) in {401, 403} or "authentication" in name or "permission" in name


def _is_transient_error(error: Exception) -> bool:
    status = _status_code(error)
    if status == 429 or (status is not None and status >= 500):
        return True
    name = type(error).__name__.lower()
    return "timeout" in name or "connection" in name
