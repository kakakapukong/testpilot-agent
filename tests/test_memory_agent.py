from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from testpilot.command import FinishTool
from testpilot.memory import MemoryDraft
from testpilot.memory_agent import (
    MAX_MEMORY_ASSISTANT_CONTENT_CHARS,
    MAX_MEMORY_EVIDENCE_PATH_CHARS,
    MAX_MEMORY_TOOL_ARGUMENT_CHARS,
    MemoryAgent,
    MemoryAgentError,
    build_memory_registry,
)
from testpilot.model import FakeModel, ModelError
from testpilot.types import AssistantTurn, ToolCall


def _call(
    call_id: str,
    arguments: dict[str, Any],
    *,
    name: str = "submit_memory",
    bad: str | None = None,
) -> ToolCall:
    return ToolCall(call_id, name, arguments, argument_error=bad)


def _valid_turn(call_id: str = "memory") -> AssistantTurn:
    return AssistantTurn(
        "summary",
        (
            _call(
                call_id,
                {
                    "problem": "path bug",
                    "root_cause": "separator was not normalized",
                    "solution": "normalize at the boundary",
                    "verification": "fixed pytest passed",
                    "keywords": ["path", "pytest", "windows"],
                },
            ),
        ),
    )


def _summarize(agent: MemoryAgent) -> MemoryDraft:
    return agent.summarize(
        task="Fix app.py path handling.",
        final_text="Normalized the path once.",
        changed_files=("src/app.py",),
        verification_exit_code=0,
        review_feedback="No blocking issue.",
    )


def test_memory_registry_contains_only_structured_submission() -> None:
    registry = build_memory_registry()

    assert registry.names() == ("submit_memory",)
    schema = registry.schemas()[0]["function"]
    assert schema["name"] == "submit_memory"
    assert schema["parameters"]["additionalProperties"] is False
    assert set(schema["parameters"]["required"]) == {
        "problem",
        "root_cause",
        "solution",
        "verification",
        "keywords",
    }


def test_memory_agent_returns_a_valid_submitted_draft() -> None:
    model = FakeModel([_valid_turn()])
    agent = MemoryAgent(model, build_memory_registry())

    result = _summarize(agent)

    assert result == MemoryDraft(
        "path bug",
        "separator was not normalized",
        "normalize at the boundary",
        "fixed pytest passed",
        ("path", "pytest", "windows"),
    )
    messages, tools = model.received_inputs[0]
    assert [tool["function"]["name"] for tool in tools] == ["submit_memory"]
    assert messages[0]["role"] == "developer"
    assert "evidence" in messages[0]["content"]
    evidence = json.loads(messages[1]["content"])
    assert evidence == {
        "changed_files": ["src/app.py"],
        "final_text": "Normalized the path once.",
        "review_feedback": "No blocking issue.",
        "task": "Fix app.py path handling.",
        "verification_exit_code": 0,
    }


def test_memory_agent_retries_an_invalid_submission_then_accepts_valid_one() -> None:
    invalid = AssistantTurn(
        "bad",
        (
            _call(
                "bad",
                {
                    "problem": "",
                    "root_cause": "cause",
                    "solution": "solution",
                    "verification": "pytest",
                    "keywords": ["path", "pytest", "windows"],
                },
            ),
        ),
    )
    model = FakeModel([invalid, _valid_turn("corrected")])

    result = _summarize(MemoryAgent(model, build_memory_registry()))

    assert result.solution == "normalize at the boundary"
    failure = json.loads(model.received_inputs[1][0][-1]["content"])
    assert failure["error_code"] == "invalid_memory_draft"
    assert failure["error"] == "memory draft is invalid"


def test_memory_agent_rejects_mixed_submissions_then_recovers() -> None:
    mixed = AssistantTurn(
        "two terminal calls",
        (
            _valid_turn("one").tool_calls[0],
            _valid_turn("two").tool_calls[0],
        ),
    )
    model = FakeModel([mixed, _valid_turn("later")])

    result = _summarize(MemoryAgent(model, build_memory_registry()))

    assert result.problem == "path bug"
    failures = [
        json.loads(message["content"])
        for message in model.received_inputs[1][0]
        if message["role"] == "tool"
    ]
    assert {failure["error_code"] for failure in failures} == {
        "memory_submission_must_be_separate"
    }


def test_memory_agent_can_recover_from_unknown_and_unparsed_tools() -> None:
    model = FakeModel(
        [
            AssistantTurn("unknown", (_call("unknown", {}, name="read_file"),)),
            AssistantTurn(
                "unparsed",
                (_call("bad-json", {}, bad="private parser detail"),),
            ),
            _valid_turn("valid"),
        ]
    )

    result = _summarize(MemoryAgent(model, build_memory_registry()))

    assert result.verification == "fixed pytest passed"
    unknown = json.loads(model.received_inputs[1][0][-1]["content"])
    unparsed = json.loads(model.received_inputs[2][0][-1]["content"])
    assert unknown["error_code"] == "unknown_tool"
    assert unparsed["error_code"] == "invalid_arguments"
    assert "private parser detail" not in json.dumps(unparsed)


def test_memory_agent_bounds_all_supplied_evidence() -> None:
    model = FakeModel([_valid_turn()])
    files = tuple(f"src/{index:03d}.py" for index in reversed(range(70)))

    MemoryAgent(model, build_memory_registry()).summarize(
        task="t" * 5_000,
        final_text="f" * 3_000,
        changed_files=files,
        verification_exit_code=0,
        review_feedback="r" * 3_000,
    )

    evidence = json.loads(model.received_inputs[0][0][1]["content"])
    assert len(evidence["task"]) == 4_000
    assert len(evidence["final_text"]) == 2_000
    assert len(evidence["review_feedback"]) == 2_000
    assert evidence["changed_files"] == sorted(files)[:50]


def test_memory_agent_redacts_host_evidence_before_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "memory-agent-secret-12345"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    model = FakeModel([_valid_turn()])

    MemoryAgent(model, build_memory_registry()).summarize(
        task=f"Fix token={secret}",
        final_text=f"Removed {secret}",
        changed_files=(f"src/{secret}.py",),
        verification_exit_code=0,
        review_feedback=f"No issue; password={secret}",
    )

    serialized = json.dumps(model.received_inputs[0][0], ensure_ascii=True)
    assert secret not in serialized
    assert "[REDACTED]" in serialized


def test_memory_agent_rejects_an_unbounded_evidence_path() -> None:
    path = f"src/{'x' * MAX_MEMORY_EVIDENCE_PATH_CHARS}.py"

    with pytest.raises(ValueError, match="changed file path"):
        MemoryAgent(FakeModel([_valid_turn()]), build_memory_registry()).summarize(
            task="task",
            final_text="done",
            changed_files=(path,),
            verification_exit_code=0,
            review_feedback="pass",
        )


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("task", " ", ValueError),
        ("task", 3, TypeError),
        ("final_text", 3, TypeError),
        ("changed_files", "src/app.py", TypeError),
        ("changed_files", ("src/app.py", ""), ValueError),
        ("changed_files", (), ValueError),
        ("verification_exit_code", False, ValueError),
        ("verification_exit_code", 1, ValueError),
        ("review_feedback", " ", ValueError),
    ],
)
def test_memory_agent_validates_host_evidence(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    arguments: dict[str, object] = {
        "task": "task",
        "final_text": "done",
        "changed_files": ("src/app.py",),
        "verification_exit_code": 0,
        "review_feedback": "pass",
    }
    arguments[field] = value

    with pytest.raises(error_type):
        MemoryAgent(FakeModel([_valid_turn()]), build_memory_registry()).summarize(
            **arguments  # type: ignore[arg-type]
        )


def test_memory_agent_stops_when_model_returns_no_tools() -> None:
    with pytest.raises(MemoryAgentError) as caught:
        _summarize(MemoryAgent(FakeModel([AssistantTurn("done")]), build_memory_registry()))

    assert caught.value.code == "memory_stopped_without_submission"


def test_memory_agent_stops_at_its_iteration_budget() -> None:
    invalid = AssistantTurn("unknown", (_call("unknown", {}, name="read_file"),))

    with pytest.raises(MemoryAgentError) as caught:
        _summarize(MemoryAgent(FakeModel([invalid]), build_memory_registry(), max_iterations=1))

    assert caught.value.code == "memory_max_iterations"


@dataclass
class RaisingModel:
    error: BaseException

    def complete(self, messages: object, tools: object) -> AssistantTurn:
        del messages, tools
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        ModelError("model-secret", code="model_authentication_failed", retryable=False),
        RuntimeError("runtime-secret"),
        KeyboardInterrupt("interrupt-secret"),
    ],
)
def test_memory_agent_model_failures_are_sanitized(error: BaseException) -> None:
    agent = MemoryAgent(RaisingModel(error), build_memory_registry())

    with pytest.raises(MemoryAgentError) as caught:
        _summarize(agent)

    assert caught.value.code == "memory_model_failed"
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("call_id", ["", "same"])
def test_memory_agent_rejects_invalid_or_duplicate_tool_call_ids(call_id: str) -> None:
    calls = (
        (_call("", _valid_turn().tool_calls[0].arguments_dict()),)
        if not call_id
        else (
            _valid_turn("same").tool_calls[0],
            _valid_turn("same").tool_calls[0],
        )
    )
    agent = MemoryAgent(FakeModel([AssistantTurn("invalid", calls)]), build_memory_registry())

    with pytest.raises(MemoryAgentError) as caught:
        _summarize(agent)

    assert caught.value.code == "memory_invalid_response"


@pytest.mark.parametrize(
    "turn",
    [
        AssistantTurn("x" * (MAX_MEMORY_ASSISTANT_CONTENT_CHARS + 1), (_valid_turn().tool_calls[0],)),
        AssistantTurn(
            "oversized arguments",
            (
                _call(
                    "oversized",
                    {"payload": "x" * (MAX_MEMORY_TOOL_ARGUMENT_CHARS + 1)},
                    name="unknown",
                ),
            ),
        ),
        AssistantTurn(object(), (_valid_turn().tool_calls[0],)),  # type: ignore[arg-type]
    ],
)
def test_memory_agent_rejects_unbounded_or_non_json_turns(turn: AssistantTurn) -> None:
    model = FakeModel([turn])

    with pytest.raises(MemoryAgentError) as caught:
        _summarize(MemoryAgent(model, build_memory_registry()))

    assert caught.value.code == "memory_invalid_response"
    assert len(model.received_inputs) == 1


def test_memory_agent_rejects_a_registry_with_other_tools() -> None:
    registry = build_memory_registry()
    registry.register(FinishTool())

    with pytest.raises(ValueError, match="memory registry"):
        MemoryAgent(FakeModel([]), registry)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 4])
def test_memory_agent_requires_a_positive_iteration_limit(value: Any) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        MemoryAgent(FakeModel([]), build_memory_registry(), max_iterations=value)


def test_submit_memory_rejects_unknown_fields_without_echoing_content() -> None:
    secret = "private-memory-draft"
    result = build_memory_registry().execute(
        "submit_memory",
        {
            "problem": secret,
            "root_cause": "cause",
            "solution": "solution",
            "verification": "pytest",
            "keywords": ["path", "pytest", "windows"],
            "extra": True,
        },
    )

    assert not result.ok
    assert result.error_code == "invalid_arguments"
    assert secret not in (result.error or "")
