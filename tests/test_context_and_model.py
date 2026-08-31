from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from testpilot.types import AssistantTurn, ToolCall


def _assistant_call(call_id: str = "call-1") -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "I will inspect the file.",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
            }
        ],
    }


def _tool_message(call_id: str = "call-1", content: str = "contents") -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_context_keeps_both_anchors_and_only_newest_transactions() -> None:
    from testpilot.context import BoundedContext

    context = BoundedContext(
        {"role": "developer", "content": "be careful"},
        {"role": "user", "content": "fix the bug"},
        max_recent_groups=1,
    )
    context.append_transaction(_assistant_call("old"), [_tool_message("old")])
    context.append_transaction(_assistant_call("new"), [_tool_message("new")])

    messages = context.messages()

    assert [message["role"] for message in messages] == ["developer", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["id"] == "new"
    assert messages[3]["tool_call_id"] == "new"


def test_context_never_leaves_a_tool_message_without_its_assistant_call() -> None:
    from testpilot.context import BoundedContext

    context = BoundedContext(
        {"role": "system", "content": "be careful"},
        {"role": "user", "content": "fix the bug"},
        max_recent_groups=1,
    )
    context.append_transaction(_assistant_call("first"), [_tool_message("first")])
    context.append_transaction({"role": "assistant", "content": "done"})

    messages = context.messages()

    assert messages == [
        {"role": "system", "content": "be careful"},
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "done"},
    ]


def test_context_truncates_large_tool_content_with_visible_marker() -> None:
    from testpilot.context import BoundedContext

    context = BoundedContext(
        {"role": "developer", "content": "be careful"},
        {"role": "user", "content": "fix the bug"},
        max_tool_content_chars=8,
    )
    context.append_transaction(_assistant_call(), [_tool_message(content="0123456789abcdef")])

    content = context.messages()[-1]["content"]

    assert len(content) <= 8
    assert content.startswith("01")
    assert content.endswith("…[cut]")


def test_context_defensively_copies_anchors_transactions_and_output() -> None:
    from testpilot.context import BoundedContext

    developer = {"role": "developer", "content": ["safe"]}
    task = {"role": "user", "content": "fix"}
    assistant = _assistant_call()
    tool = _tool_message()
    context = BoundedContext(developer, task)
    context.append_transaction(assistant, [tool])
    developer["content"].append("changed")
    assistant["tool_calls"][0]["function"]["name"] = "changed"
    tool["content"] = "changed"

    copied = context.messages()
    copied[0]["content"].append("output-change")
    copied[-1]["content"] = "output-change"

    stable = context.messages()
    assert stable[0]["content"] == ["safe"]
    assert stable[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert stable[-1]["content"] == "contents"


def test_fake_model_returns_scripted_turns_records_copied_inputs_and_errors_when_empty() -> None:
    from testpilot.model import FakeModel, ModelError

    turn = AssistantTurn(
        content="inspect", tool_calls=(ToolCall("id", "read_file", {"path": "a.py"}),)
    )
    model = FakeModel([turn])
    messages = [{"role": "user", "content": "fix"}]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    assert model.complete(messages, tools) == turn
    messages[0]["content"] = "mutated"
    tools[0]["function"]["name"] = "mutated"
    assert model.received_inputs[0][0][0]["content"] == "fix"
    assert model.received_inputs[0][1][0]["function"]["name"] == "read_file"
    with pytest.raises(ModelError, match="exhausted") as caught:
        model.complete(messages, tools)
    assert caught.value.code == "model_exhausted"
    assert not caught.value.retryable


class _FakeCreate:
    def __init__(self, response: object | list[object] | BaseException) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.response, list):
            current = self.response.pop(0)
        else:
            current = self.response
        if isinstance(current, BaseException):
            raise current
        return current


def _client_with(response: object | list[object] | BaseException) -> tuple[object, _FakeCreate]:
    create = _FakeCreate(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=create))
    return client, create


def _response(*, content: object = "", tool_calls: list[object] | None = None) -> object:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _sdk_call(call_id: str, name: str, arguments: str) -> object:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_openai_model_normalizes_content_and_multiple_tool_calls() -> None:
    from testpilot.model import OpenAIChatModel

    client, create = _client_with(
        _response(
            content=None,
            tool_calls=[
                _sdk_call("first", "read_file", '{"path":"a.py"}'),
                _sdk_call("second", "search_text", '{"query":"TODO"}'),
            ],
        )
    )
    model = OpenAIChatModel(model="test-model", client=client)
    messages = [{"role": "user", "content": "fix"}]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    turn = model.complete(messages, tools)

    assert turn.content == ""
    assert [(call.id, call.name, call.arguments_dict()) for call in turn.tool_calls] == [
        ("first", "read_file", {"path": "a.py"}),
        ("second", "search_text", {"query": "TODO"}),
    ]
    assert create.calls[0]["model"] == "test-model"
    assert create.calls[0]["tool_choice"] == "auto"
    messages[0]["content"] = "changed"
    tools[0]["function"]["name"] = "changed"
    assert create.calls[0]["messages"][0]["content"] == "fix"
    assert create.calls[0]["tools"][0]["function"]["name"] == "read_file"


@pytest.mark.parametrize("arguments", ["{broken", "[]"])
def test_openai_model_keeps_bad_arguments_as_an_invalid_callable_turn(arguments: str) -> None:
    from testpilot.model import OpenAIChatModel

    client, _ = _client_with(_response(tool_calls=[_sdk_call("bad", "read_file", arguments)]))

    call = OpenAIChatModel(model="test-model", client=client).complete([], []).tool_calls[0]

    assert call.id == "bad"
    assert call.name == "read_file"
    assert call.arguments_dict() == {}
    assert call.argument_error


def test_openai_model_retries_transient_failure_at_most_twice() -> None:
    from testpilot.model import OpenAIChatModel

    class RateLimitError(Exception):
        status_code = 429

    client, create = _client_with(
        [RateLimitError("busy"), RateLimitError("busy"), _response(content="ok")]
    )
    sleeps: list[float] = []

    turn = OpenAIChatModel(model="test-model", client=client, sleep_fn=sleeps.append).complete(
        [], []
    )

    assert turn.content == "ok"
    assert len(create.calls) == 3
    assert sleeps == [0.05, 0.1]


def test_openai_model_reports_retry_exhaustion_without_original_error_text() -> None:
    from testpilot.model import ModelError, OpenAIChatModel

    class RateLimitError(Exception):
        status_code = 429

    secret = "upstream-error-detail-that-must-not-leak"
    client, create = _client_with(
        [RateLimitError(secret), RateLimitError(secret), RateLimitError(secret)]
    )
    sleeps: list[float] = []

    with pytest.raises(ModelError) as caught:
        OpenAIChatModel(model="test-model", client=client, sleep_fn=sleeps.append).complete([], [])

    assert caught.value.code == "model_transient_failure"
    assert caught.value.retryable
    assert secret not in str(caught.value)
    assert len(create.calls) == 3
    assert sleeps == [0.05, 0.1]


def test_openai_model_uses_no_sdk_retries_when_it_constructs_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testpilot.model import OpenAIChatModel

    constructed_with: list[dict[str, object]] = []
    response = _response(content="ok")

    def make_client(**kwargs: object) -> object:
        constructed_with.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCreate(response)))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=make_client))

    assert OpenAIChatModel(model="test-model").complete([], []).content == "ok"
    assert constructed_with == [{"max_retries": 0}]


@pytest.mark.parametrize("model", ["", " \t\n", 123, None])
def test_openai_model_rejects_blank_or_non_string_model_names(model: object) -> None:
    from testpilot.model import OpenAIChatModel

    with pytest.raises(ValueError, match="model"):
        OpenAIChatModel(model=model)  # type: ignore[arg-type]


def test_openai_model_does_not_retry_authentication_or_leak_api_key() -> None:
    from testpilot.model import ModelError, OpenAIChatModel

    class AuthenticationError(Exception):
        status_code = 401

    secret = "super-secret-key"
    client, create = _client_with(AuthenticationError(secret))
    model = OpenAIChatModel(model="test-model", api_key=secret, client=client)

    with pytest.raises(ModelError) as caught:
        model.complete([], [])

    assert caught.value.code == "model_authentication_failed"
    assert not caught.value.retryable
    assert len(create.calls) == 1
    assert secret not in str(caught.value)
    assert secret not in repr(model)


def test_openai_model_rejects_responses_without_a_choice() -> None:
    from testpilot.model import ModelError, OpenAIChatModel

    client, _ = _client_with(SimpleNamespace(choices=[]))

    with pytest.raises(ModelError) as caught:
        OpenAIChatModel(model="test-model", client=client).complete([], [])

    assert caught.value.code == "invalid_model_response"
