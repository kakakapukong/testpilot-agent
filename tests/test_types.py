import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from testpilot.types import AgentRunResult, AssistantTurn, RunPhase, RunState, ToolCall, ToolResult


def test_tool_result_failure_serializes_all_fields() -> None:
    result = ToolResult.failure("outside workspace", code="path_outside_workspace")

    assert result.to_dict() == {
        "ok": False,
        "data": None,
        "error": "outside workspace",
        "error_code": "path_outside_workspace",
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "truncated": False,
    }


def test_tool_result_success_serializes_data() -> None:
    result = ToolResult.success({"path": "a.py"})

    serialized = result.to_dict()
    assert serialized["ok"] is True
    assert serialized["data"] == {"path": "a.py"}


def test_tool_result_rejects_contradictory_direct_construction() -> None:
    with pytest.raises(ValueError, match="successful"):
        ToolResult(ok=True, error="unexpected", error_code="unexpected_error")

    with pytest.raises(ValueError, match="failed"):
        ToolResult(ok=False)


def test_tool_result_factories_do_not_accept_contradictory_statuses() -> None:
    with pytest.raises(TypeError):
        ToolResult.success(error="unexpected")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        ToolResult.failure("failed", code="failure", ok=True)  # type: ignore[call-arg]


def test_tool_result_prevents_field_reassignment() -> None:
    result = ToolResult.success()

    with pytest.raises(FrozenInstanceError):
        result.stdout = "changed"


def test_successful_tool_result_rejects_a_nonzero_exit_code() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        ToolResult.success(exit_code=1)


def test_successful_tool_result_rejects_timeout_and_allows_zero_exit_code() -> None:
    with pytest.raises(ValueError, match="timed out"):
        ToolResult.success(timed_out=True)

    assert ToolResult.success(exit_code=0).exit_code == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ToolResult.success(exit_code=False),
        lambda: ToolResult(ok=1),
        lambda: ToolResult.success(stdout=1),
        lambda: ToolResult.success(stderr=2),
        lambda: ToolResult.failure(3, "failure"),
        lambda: ToolResult.failure("failure", 4),
    ],
)
def test_tool_result_rejects_noncanonical_control_field_types(factory: Any) -> None:
    with pytest.raises(TypeError):
        factory()


def test_record_edit_tracks_an_unverified_change() -> None:
    state = RunState()

    state.record_edit("src/app.py")

    assert state.phase is RunPhase.EDIT
    assert state.edit_count == 1
    assert state.source_edit_count == 1
    assert state.changed_files == {"src/app.py"}
    assert state.verified_after_last_edit is False


def test_verification_applies_only_until_the_next_edit() -> None:
    state = RunState()
    state.record_edit("src/app.py")

    state.record_verification(0)

    assert state.phase is RunPhase.VERIFY
    assert state.last_verify_exit_code == 0
    assert state.verified_after_last_edit is True

    state.record_edit("src/next.py")

    assert state.verified_after_last_edit is False


def test_failed_verification_with_exit_zero_is_not_marked_passed() -> None:
    state = RunState()
    state.record_edit("src/app.py")

    state.record_verification(0, passed=False)

    assert state.last_verify_exit_code == 0
    assert state.verified_after_last_edit is False


def test_record_edit_counts_python_source_separately_from_other_files() -> None:
    state = RunState()

    state.record_edit("README.md")
    state.record_edit("src/types.pyi")

    assert state.edit_count == 2
    assert state.source_edit_count == 1


def test_record_review_tracks_a_passing_review_without_invalidating_verification() -> None:
    state = RunState()
    state.record_edit("src/app.py")
    state.record_verification(0)

    state.record_review("passed")

    assert state.phase is RunPhase.REVIEW
    assert state.review_status == "passed"
    assert state.review_rounds == 1
    assert state.review_rework_count == 0
    assert state.reviewed_edit_count == 1
    assert state.reviewed_source_edit_count == 1
    assert state.verified_after_last_edit is True


@pytest.mark.parametrize("status", ["changes_requested", "unavailable"])
def test_non_passing_review_invalidates_verification(status: str) -> None:
    state = RunState()
    state.record_edit("src/app.py")
    state.record_verification(0)

    state.record_review(status)

    assert state.phase is RunPhase.REVIEW
    assert state.review_status == status
    assert state.review_rounds == 1
    assert state.reviewed_edit_count == 1
    assert state.verified_after_last_edit is False


@pytest.mark.parametrize("status", [None, True, "", "approved", 0, []])
def test_record_review_rejects_every_other_value(status: Any) -> None:
    state = RunState()

    with pytest.raises(ValueError, match="invalid review status"):
        state.record_review(status)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["approved", "rejected", "unavailable"])
def test_record_approval_accepts_only_stable_decisions(status: str) -> None:
    state = RunState()

    state.record_approval(status)

    assert state.approval_status == status


@pytest.mark.parametrize("status", [None, True, "", "pending", 0, []])
def test_record_approval_rejects_every_other_value(status: Any) -> None:
    state = RunState()

    with pytest.raises(ValueError, match="invalid approval status"):
        state.record_approval(status)  # type: ignore[arg-type]


def test_frozen_turn_value_types_support_basic_construction() -> None:
    tool_call = ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})
    turn = AssistantTurn(content="Reading file.", tool_calls=(tool_call,))

    assert tool_call.argument_error is None
    assert turn.tool_calls == (tool_call,)


def test_tool_call_defensively_deep_freezes_arguments() -> None:
    arguments: dict[str, Any] = {"path": "a.py", "options": {"lines": [1, 2]}}
    tool_call = ToolCall(id="call_1", name="read_file", arguments=arguments)

    arguments["path"] = "changed.py"
    arguments["options"]["lines"].append(3)

    assert tool_call.arguments["path"] == "a.py"
    assert tool_call.arguments["options"]["lines"] == (1, 2)
    with pytest.raises(TypeError):
        tool_call.arguments["new"] = "value"  # type: ignore[index]


def test_tool_call_exports_a_defensive_json_native_arguments_copy() -> None:
    tool_call = ToolCall(
        id="call_1",
        name="read_file",
        arguments={"options": {"lines": [1, 2]}},
    )

    exported = tool_call.arguments_dict()

    assert json.dumps(exported)
    assert exported["options"]["lines"] == [1, 2]
    exported["options"]["lines"].append(3)
    assert tool_call.arguments["options"]["lines"] == (1, 2)


def test_frozen_agent_run_result_supports_basic_construction() -> None:
    state = RunState()
    messages = ("assistant message",)
    trace_path = Path("trace.jsonl")

    result = AgentRunResult(
        success=True,
        final_text="Completed.",
        stop_reason="verified",
        state=state,
        messages=messages,
        trace_path=trace_path,
    )

    assert result.success is True
    assert result.final_text == "Completed."
    assert result.stop_reason == "verified"
    assert result.state is state
    assert result.messages == messages
    assert result.trace_path == trace_path


def test_agent_run_result_prevents_top_level_field_reassignment() -> None:
    result = AgentRunResult(
        success=True,
        final_text="Completed.",
        stop_reason="verified",
        state=RunState(),
        messages=(),
    )

    with pytest.raises(AttributeError):
        result.success = False
