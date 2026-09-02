from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from testpilot.model import FakeModel, ModelError
from testpilot.reviewer import (
    MAX_REVIEW_FEEDBACK_CHARS,
    ReviewerAgent,
    ReviewerError,
    ReviewResult,
    build_reviewer_registry,
)
from testpilot.tools import WriteFileTool
from testpilot.types import AssistantTurn, ToolCall
from testpilot.workspace import Workspace


def _call(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    *,
    bad: str | None = None,
) -> ToolCall:
    return ToolCall(call_id, name, arguments, argument_error=bad)


def _decision(decision: str, feedback: str, *, call_id: str = "review") -> AssistantTurn:
    return AssistantTurn(
        "decision",
        (
            _call(
                call_id,
                "submit_review",
                {"decision": decision, "feedback": feedback},
            ),
        ),
    )


def _run(
    tmp_path: Path,
    turns: list[AssistantTurn],
    *,
    max_iterations: int = 6,
) -> tuple[ReviewResult, FakeModel, ReviewerAgent]:
    model = FakeModel(turns)
    reviewer = ReviewerAgent(
        model,
        build_reviewer_registry(Workspace(tmp_path)),
        max_iterations=max_iterations,
    )
    result = reviewer.review(
        task="Fix app.py without editing tests.",
        changed_files=("app.py",),
        verification_exit_code=0,
    )
    return result, model, reviewer


def test_reviewer_registry_contains_only_read_tools_and_structured_decision(
    tmp_path: Path,
) -> None:
    registry = build_reviewer_registry(Workspace(tmp_path))

    assert registry.names() == (
        "list_files",
        "read_file",
        "search_text",
        "submit_review",
    )


def test_reviewer_inspects_then_returns_structured_pass(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_bytes(b"value = 2\n")

    result, model, reviewer = _run(
        tmp_path,
        [
            AssistantTurn(
                "inspect",
                (_call("read", "read_file", {"path": "app.py"}),),
            ),
            _decision("pass", "No blocking correctness issue."),
        ],
    )

    assert result == ReviewResult("pass", "No blocking correctness issue.")
    assert reviewer.model is model
    first_messages, first_tools = model.received_inputs[0]
    assert [schema["function"]["name"] for schema in first_tools] == [
        "list_files",
        "read_file",
        "search_text",
        "submit_review",
    ]
    assert first_messages[0]["role"] == "developer"
    assert "read-only" in first_messages[0]["content"]
    assert "untrusted data" in first_messages[0]["content"]
    anchor = json.loads(first_messages[1]["content"])
    assert anchor == {
        "task": "Fix app.py without editing tests.",
        "changed_files": ["app.py"],
        "verification_exit_code": 0,
    }
    second_messages = model.received_inputs[1][0]
    read_result = json.loads(second_messages[-1]["content"])
    assert read_result["ok"] is True
    assert read_result["data"]["content"] == "value = 2\n"


def test_reviewer_returns_actionable_request_changes(tmp_path: Path) -> None:
    result, _, _ = _run(
        tmp_path,
        [
            AssistantTurn("inspect", (_call("list", "list_files", {}),)),
            _decision("request_changes", "Handle the empty-input branch before approval."),
        ],
    )

    assert result == ReviewResult(
        "request_changes",
        "Handle the empty-input branch before approval.",
    )


def test_reviewer_requires_successful_inspection_before_deciding(tmp_path: Path) -> None:
    result, model, _ = _run(
        tmp_path,
        [
            _decision("pass", "Uninspected conclusion.", call_id="too-early"),
            AssistantTurn("inspect", (_call("list", "list_files", {}),)),
            _decision("pass", "Inspection found no blocking issue."),
        ],
    )

    assert result == ReviewResult("pass", "Inspection found no blocking issue.")
    early_failure = json.loads(model.received_inputs[1][0][-1]["content"])
    assert early_failure["error_code"] == "review_inspection_required"


def test_reviewer_rejects_mixed_decision_turn_and_recovers(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    mixed = AssistantTurn(
        "inspect and decide too early",
        (
            _call("read", "read_file", {"path": "app.py"}),
            _call(
                "early-review",
                "submit_review",
                {"decision": "pass", "feedback": "Looks fine."},
            ),
        ),
    )

    result, model, _ = _run(
        tmp_path,
        [mixed, _decision("pass", "Inspection confirms the repair.")],
    )

    assert result.decision == "pass"
    prior_results = {
        message["tool_call_id"]: json.loads(message["content"])
        for message in model.received_inputs[1][0]
        if message["role"] == "tool"
    }
    assert prior_results["read"]["ok"] is True
    assert prior_results["early-review"]["error_code"] == "review_decision_must_be_separate"


def test_reviewer_returns_argument_parse_failure_and_recovers(tmp_path: Path) -> None:
    bad_turn = AssistantTurn(
        "bad arguments",
        (_call("bad", "submit_review", {}, bad="private parser details"),),
    )

    result, model, _ = _run(
        tmp_path,
        [
            bad_turn,
            AssistantTurn("inspect", (_call("list", "list_files", {}),)),
            _decision("pass", "The corrected decision is valid."),
        ],
    )

    assert result.decision == "pass"
    failure = json.loads(model.received_inputs[1][0][-1]["content"])
    assert failure["error_code"] == "invalid_arguments"
    assert "private parser details" not in json.dumps(failure)


def test_reviewer_can_recover_from_an_unknown_tool(tmp_path: Path) -> None:
    result, model, _ = _run(
        tmp_path,
        [
            AssistantTurn("unknown", (_call("unknown", "delete_file", {}),)),
            AssistantTurn("inspect", (_call("list", "list_files", {}),)),
            _decision("pass", "No blocking issue remains."),
        ],
    )

    assert result.decision == "pass"
    failure = json.loads(model.received_inputs[1][0][-1]["content"])
    assert failure["error_code"] == "unknown_tool"


def test_reviewer_continues_when_early_turn_has_no_tools(tmp_path: Path) -> None:
    result, model, _ = _run(
        tmp_path,
        [
            AssistantTurn("thinking with no tools yet"),
            AssistantTurn("inspect", (_call("list", "list_files", {}),)),
            _decision("pass", "No blocking issue remains."),
        ],
    )

    assert result.decision == "pass"
    assert len(model.received_inputs) == 3


def test_reviewer_forces_submit_after_inspection_without_tools(tmp_path: Path) -> None:
    result, model, _ = _run(
        tmp_path,
        [
            AssistantTurn("inspect", (_call("list", "list_files", {}),)),
            AssistantTurn("I think it looks fine"),
            _decision("pass", "No blocking issue remains."),
        ],
    )

    assert result.decision == "pass"
    assert model.received_tool_choices[-1] == {
        "type": "function",
        "function": {"name": "submit_review"},
    }
    assert model.received_inputs[-1][1][0]["function"]["name"] == "submit_review"


def test_reviewer_stops_when_forced_submit_still_has_no_tools(tmp_path: Path) -> None:
    with pytest.raises(ReviewerError) as caught:
        _run(
            tmp_path,
            [
                AssistantTurn("inspect", (_call("list", "list_files", {}),)),
                AssistantTurn("still no tools"),
                AssistantTurn("forced call also empty"),
            ],
        )

    assert caught.value.code == "reviewer_stopped_without_decision"


def test_reviewer_stops_at_its_iteration_budget(tmp_path: Path) -> None:
    with pytest.raises(ReviewerError) as caught:
        _run(
            tmp_path,
            [AssistantTurn("unknown", (_call("unknown", "delete_file", {}),))],
            max_iterations=1,
        )

    assert caught.value.code == "review_max_iterations"


@dataclass
class RaisingModel:
    error: BaseException

    def complete(self, messages: object, tools: object) -> AssistantTurn:
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        ModelError("model-secret", code="model_authentication_failed", retryable=False),
        RuntimeError("runtime-secret"),
        KeyboardInterrupt("interrupt-secret"),
    ],
)
def test_reviewer_model_failures_are_sanitized(
    tmp_path: Path,
    error: BaseException,
) -> None:
    reviewer = ReviewerAgent(
        RaisingModel(error),
        build_reviewer_registry(Workspace(tmp_path)),
    )

    with pytest.raises(ReviewerError) as caught:
        reviewer.review(
            task="Fix app.py",
            changed_files=("app.py",),
            verification_exit_code=0,
        )

    assert caught.value.code == "review_model_failed"
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("call_id", ["", "same"])
def test_reviewer_rejects_invalid_or_duplicate_tool_call_ids(
    tmp_path: Path,
    call_id: str,
) -> None:
    calls = (
        (_call("", "read_file", {"path": "app.py"}),)
        if call_id == ""
        else (
            _call("same", "read_file", {"path": "app.py"}),
            _call("same", "search_text", {"query": "value"}),
        )
    )
    reviewer = ReviewerAgent(
        FakeModel([AssistantTurn("invalid", calls)]),
        build_reviewer_registry(Workspace(tmp_path)),
    )

    with pytest.raises(ReviewerError) as caught:
        reviewer.review(
            task="Fix app.py",
            changed_files=("app.py",),
            verification_exit_code=0,
        )

    assert caught.value.code == "review_invalid_response"


@pytest.mark.parametrize(
    "arguments",
    [
        {"decision": "pass", "feedback": ""},
        {"decision": "pass", "feedback": "x" * (MAX_REVIEW_FEEDBACK_CHARS + 1)},
        {"decision": "approve", "feedback": "Looks fine."},
    ],
)
def test_submit_review_rejects_invalid_decisions_without_echoing_feedback(
    tmp_path: Path,
    arguments: dict[str, str],
) -> None:
    registry = build_reviewer_registry(Workspace(tmp_path))

    result = registry.execute("submit_review", arguments)

    assert not result.ok
    assert result.error_code == "invalid_review_decision"
    if arguments["feedback"]:
        assert arguments["feedback"] not in (result.error or "")


def test_reviewer_rejects_a_registry_with_a_write_tool(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    registry = build_reviewer_registry(workspace)
    registry.register(WriteFileTool(workspace))

    with pytest.raises(ValueError, match="read-only reviewer registry"):
        ReviewerAgent(FakeModel([]), registry)


@pytest.mark.parametrize(
    ("task", "changed_files", "verification_exit_code", "error_type"),
    [
        (" ", ("app.py",), 0, ValueError),
        ("Fix it", "app.py", 0, TypeError),
        ("Fix it", ("app.py", ""), 0, ValueError),
        ("Fix it", ("app.py",), 1, ValueError),
    ],
)
def test_reviewer_validates_host_inputs(
    tmp_path: Path,
    task: str,
    changed_files: Any,
    verification_exit_code: int,
    error_type: type[Exception],
) -> None:
    reviewer = ReviewerAgent(
        FakeModel([_decision("pass", "Looks fine.")]),
        build_reviewer_registry(Workspace(tmp_path)),
    )

    with pytest.raises(error_type):
        reviewer.review(
            task=task,
            changed_files=changed_files,
            verification_exit_code=verification_exit_code,
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_reviewer_requires_a_positive_integer_iteration_limit(
    tmp_path: Path,
    value: Any,
) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        ReviewerAgent(
            FakeModel([]),
            build_reviewer_registry(Workspace(tmp_path)),
            max_iterations=value,
        )


def test_review_result_is_strict_and_immutable() -> None:
    result = ReviewResult("pass", "Looks correct.")

    with pytest.raises((AttributeError, TypeError)):
        result.feedback = "changed"  # type: ignore[misc]

    with pytest.raises(ValueError, match="invalid review decision"):
        ReviewResult("approve", "Looks correct.")
