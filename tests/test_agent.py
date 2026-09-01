from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from testpilot.model import FakeModel, ModelError
from testpilot.registry import ToolRegistry
from testpilot.tools import EditFileTool, WriteFileTool
from testpilot.types import AssistantTurn, ToolCall, ToolResult
from testpilot.workspace import Workspace


@dataclass
class MemoryTool:
    name: str = "read_file"
    description: str = "A test tool."
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }
    )
    result: ToolResult = field(default_factory=lambda: ToolResult.success({"content": "source"}))
    seen: list[dict[str, Any]] = field(default_factory=list)

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        self.seen.append(dict(arguments))
        return self.result


class EditTool(MemoryTool):
    name = "edit_file"

    def __init__(self, result: ToolResult | None = None) -> None:
        super().__init__(
            name="edit_file",
            description="Edit a source file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
            result=result or ToolResult.success({"path": "app.py", "changed": True}),
        )


class EchoPathEditTool(EditTool):
    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        self.seen.append(dict(arguments))
        return ToolResult.success({"path": arguments["path"], "changed": True})


class FinishRequestTool(MemoryTool):
    name = "finish"

    def __init__(self) -> None:
        super().__init__(
            name="finish",
            description="Request verification.",
            parameters={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
            result=ToolResult.success({"finish_requested": True}),
        )


class ScriptedVerifier:
    def __init__(self, results: list[ToolResult]) -> None:
        self.results = results
        self.calls = 0

    def verify(self) -> ToolResult:
        self.calls += 1
        return self.results.pop(0)


@dataclass
class FakeApproval:
    decision: Any = True
    request_error: BaseException | None = None
    rollback_error: BaseException | None = None
    requests: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    rollback_calls: int = 0

    def request(
        self,
        *,
        changed_files: tuple[str, ...],
        verification_exit_code: int,
    ) -> bool:
        self.requests.append((tuple(changed_files), verification_exit_code))
        if self.request_error is not None:
            raise self.request_error
        return self.decision

    def rollback(self) -> None:
        self.rollback_calls += 1
        if self.rollback_error is not None:
            raise self.rollback_error


@dataclass
class CapturingTrace:
    path: Path
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def record(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self.events.append((event, dict(payload or {})))


def _registry(*tools: Any) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _call(
    call_id: str, name: str, arguments: dict[str, Any], *, bad: str | None = None
) -> ToolCall:
    return ToolCall(call_id, name, arguments, argument_error=bad)


def _runner(
    turns: list[AssistantTurn],
    registry: ToolRegistry,
    verifier: ScriptedVerifier,
    **kwargs: Any,
) -> Any:
    from testpilot.agent import AgentRunner

    return AgentRunner(FakeModel(turns), registry, verifier, **kwargs)


def _success_verifier() -> ScriptedVerifier:
    return ScriptedVerifier([ToolResult.success({"verified": True}, exit_code=0)])


def _verified_runner(*, approval: FakeApproval, trace: CapturingTrace | None = None) -> Any:
    edit = EchoPathEditTool()
    finish = FinishRequestTool()
    return _runner(
        [
            AssistantTurn(
                "fix",
                (
                    _call(
                        "edit-z",
                        "edit_file",
                        {"path": "z.py", "old_text": "old", "new_text": "new"},
                    ),
                    _call(
                        "edit-a",
                        "edit_file",
                        {"path": "a.py", "old_text": "old", "new_text": "new"},
                    ),
                ),
            ),
            AssistantTurn("finish", (_call("finish", "finish", {"summary": "done"}),)),
        ],
        _registry(edit, finish),
        _success_verifier(),
        approval=approval,
        trace=trace,
    )


def _bypassed_tool_result(**fields: Any) -> ToolResult:
    """Construct a malformed frozen value to test the Agent's trust boundary."""
    result = object.__new__(ToolResult)
    values = {
        "ok": True,
        "data": None,
        "error": None,
        "error_code": None,
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "truncated": False,
    }
    values.update(fields)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def test_agent_runs_read_edit_finish_and_only_succeeds_after_verification() -> None:
    read = MemoryTool()
    edit = EditTool()
    finish = FinishRequestTool()
    runner = _runner(
        [
            AssistantTurn("inspect", (_call("read", "read_file", {"path": "app.py"}),)),
            AssistantTurn(
                "fix",
                (_call("edit", "edit_file", {"path": "app.py", "old_text": "x", "new_text": "y"}),),
            ),
            AssistantTurn("done", (_call("finish", "finish", {"summary": "fixed"}),)),
        ],
        _registry(read, edit, finish),
        _success_verifier(),
    )

    result = runner.run("Fix app.py")

    assert result.success
    assert result.stop_reason == "verified"
    assert result.state.edit_count == 1
    assert result.state.verified_after_last_edit
    assert result.state.approval_status is None
    assert [message["role"] for message in result.messages] == [
        "developer",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert json.loads(result.messages[-1]["content"])["ok"]
    assert result.messages[-1]["tool_call_id"] == "finish"


def test_verified_repair_requires_and_records_approval() -> None:
    approval = FakeApproval(decision=True)

    result = _verified_runner(approval=approval).run("Fix app.py")

    assert result.success
    assert result.stop_reason == "verified"
    assert result.state.approval_status == "approved"
    assert approval.requests == [(("a.py", "z.py"), 0)]
    assert approval.rollback_calls == 0


def test_rejected_repair_rolls_back_once_and_fails() -> None:
    approval = FakeApproval(decision=False)

    result = _verified_runner(approval=approval).run("Fix app.py")

    assert not result.success
    assert result.stop_reason == "approval_rejected"
    assert result.state.approval_status == "rejected"
    assert approval.requests == [(("a.py", "z.py"), 0)]
    assert approval.rollback_calls == 1


def test_approval_request_exception_fails_closed_without_leaking_details() -> None:
    secret = "private/path.py and private diff contents"
    approval = FakeApproval(request_error=RuntimeError(secret))
    trace = CapturingTrace(Path("trace.jsonl"))

    result = _verified_runner(approval=approval, trace=trace).run("Fix app.py")

    assert not result.success
    assert result.stop_reason == "approval_unavailable"
    assert result.state.approval_status == "unavailable"
    assert approval.requests == [(("a.py", "z.py"), 0)]
    assert approval.rollback_calls == 1
    assert secret not in result.final_text
    assert secret not in json.dumps(trace.events, ensure_ascii=False)


def test_approval_request_keyboard_interrupt_fails_closed_and_rolls_back() -> None:
    approval = FakeApproval(request_error=KeyboardInterrupt("private interrupt details"))

    try:
        result = _verified_runner(approval=approval).run("Fix app.py")
    except KeyboardInterrupt:
        pytest.fail("approval request interruption escaped the fail-closed boundary")

    assert not result.success
    assert result.stop_reason == "approval_unavailable"
    assert result.state.approval_status == "unavailable"
    assert approval.rollback_calls == 1


def test_approval_request_does_not_capture_system_exit() -> None:
    approval = FakeApproval(request_error=SystemExit(7))

    with pytest.raises(SystemExit) as raised:
        _verified_runner(approval=approval).run("Fix app.py")

    assert raised.value.code == 7
    assert approval.rollback_calls == 0


@pytest.mark.parametrize("decision", [None, 0, 1, "yes", object()])
def test_non_boolean_approval_response_fails_closed(decision: Any) -> None:
    approval = FakeApproval(decision=decision)

    result = _verified_runner(approval=approval).run("Fix app.py")

    assert not result.success
    assert result.stop_reason == "approval_unavailable"
    assert result.state.approval_status == "unavailable"
    assert approval.rollback_calls == 1


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [(False, "rejected"), (None, "unavailable")],
)
def test_rollback_exception_overrides_approval_stop_reason(
    decision: Any,
    expected_status: str,
) -> None:
    approval = FakeApproval(
        decision=decision,
        rollback_error=RuntimeError("private rollback details"),
    )

    result = _verified_runner(approval=approval).run("Fix app.py")

    assert not result.success
    assert result.stop_reason == "rollback_failed"
    assert result.state.approval_status == expected_status
    assert approval.rollback_calls == 1


def test_rollback_keyboard_interrupt_is_reported_as_rollback_failed() -> None:
    approval = FakeApproval(
        decision=False,
        rollback_error=KeyboardInterrupt("private rollback interrupt"),
    )

    try:
        result = _verified_runner(approval=approval).run("Fix app.py")
    except KeyboardInterrupt:
        pytest.fail("rollback interruption escaped the fail-closed boundary")

    assert not result.success
    assert result.stop_reason == "rollback_failed"
    assert result.state.approval_status == "rejected"
    assert approval.rollback_calls == 1


def test_rollback_does_not_capture_system_exit() -> None:
    approval = FakeApproval(decision=False, rollback_error=SystemExit(9))

    with pytest.raises(SystemExit) as raised:
        _verified_runner(approval=approval).run("Fix app.py")

    assert raised.value.code == 9
    assert approval.rollback_calls == 1


def test_duplicate_finish_in_one_turn_verifies_and_requests_approval_once() -> None:
    edit = EditTool()
    finish = FinishRequestTool()
    verifier = _success_verifier()
    approval = FakeApproval()
    runner = _runner(
        [
            AssistantTurn(
                "edit",
                (_call("edit", "edit_file", {"path": "app.py", "old_text": "a", "new_text": "b"}),),
            ),
            AssistantTurn(
                "finish twice",
                (
                    _call("finish-1", "finish", {"summary": "done"}),
                    _call("finish-2", "finish", {"summary": "again"}),
                ),
            ),
        ],
        _registry(edit, finish),
        verifier,
        approval=approval,
    )

    result = runner.run("Fix")

    assert result.success
    assert verifier.calls == 1
    assert len(finish.seen) == 1
    assert approval.requests == [(("app.py",), 0)]
    assert approval.rollback_calls == 0
    duplicate = next(
        message for message in result.messages if message.get("tool_call_id") == "finish-2"
    )
    assert json.loads(duplicate["content"])["error_code"] == "duplicate_finish"


def test_later_edit_blocks_approval_and_next_turn_can_finish() -> None:
    edit = EditTool()
    finish = FinishRequestTool()
    verifier = ScriptedVerifier(
        [
            ToolResult.success({"verified": True}, exit_code=0),
            ToolResult.success({"verified": True}, exit_code=0),
        ]
    )
    approval = FakeApproval()
    runner = _runner(
        [
            AssistantTurn(
                "first edit",
                (_call("edit-1", "edit_file", {"path": "app.py", "old_text": "a", "new_text": "b"}),),
            ),
            AssistantTurn(
                "finish edit finish",
                (
                    _call("finish-1", "finish", {"summary": "first"}),
                    _call("edit-2", "edit_file", {"path": "app.py", "old_text": "b", "new_text": "c"}),
                    _call("finish-2", "finish", {"summary": "duplicate"}),
                ),
            ),
            AssistantTurn("finish next turn", (_call("finish-3", "finish", {"summary": "done"}),)),
        ],
        _registry(edit, finish),
        verifier,
        approval=approval,
    )

    result = runner.run("Fix")

    assert result.success
    assert verifier.calls == 2
    assert len(finish.seen) == 2
    assert approval.requests == [(("app.py",), 0)]
    duplicate = next(
        message for message in result.messages if message.get("tool_call_id") == "finish-2"
    )
    assert json.loads(duplicate["content"])["error_code"] == "duplicate_finish"


def test_approval_trace_has_safe_ordered_metadata_without_paths() -> None:
    private_path = "private/directory/secret.py"
    edit = EditTool(ToolResult.success({"path": private_path, "changed": True}))
    finish = FinishRequestTool()
    approval = FakeApproval(decision=False)
    trace = CapturingTrace(Path("trace.jsonl"))
    runner = _runner(
        [
            AssistantTurn(
                "fix",
                (
                    _call(
                        "edit",
                        "edit_file",
                        {"path": private_path, "old_text": "private old", "new_text": "private new"},
                    ),
                ),
            ),
            AssistantTurn("finish", (_call("finish", "finish", {"summary": "private"}),)),
        ],
        _registry(edit, finish),
        _success_verifier(),
        approval=approval,
        trace=trace,
    )

    result = runner.run("Fix")

    assert result.stop_reason == "approval_rejected"
    stages = [
        (index, payload)
        for index, (event, payload) in enumerate(trace.events)
        if event == "approval"
    ]
    assert [payload["stage"] for _, payload in stages] == ["start", "complete", "rollback"]
    assert stages[0][1] == {
        "stage": "start",
        "changed_file_count": 1,
        "verification_exit": 0,
    }
    assert stages[1][1] == {
        "stage": "complete",
        "decision": "rejected",
        "ok": True,
        "error_code": None,
    }
    assert stages[2][1] == {
        "stage": "rollback",
        "decision": "rejected",
        "ok": True,
        "error_code": None,
    }
    verification_complete_index = next(
        index
        for index, (event, payload) in enumerate(trace.events)
        if event == "verification" and payload["stage"] == "complete"
    )
    stop_index = next(index for index, (event, _) in enumerate(trace.events) if event == "stop")
    assert verification_complete_index < stages[0][0] < stages[1][0] < stages[2][0] < stop_index
    trace_text = json.dumps(trace.events, ensure_ascii=False)
    assert private_path not in trace_text
    assert "private old" not in trace_text
    assert "private new" not in trace_text


def test_agent_returns_bad_arguments_to_model_without_executing_tool() -> None:
    read = MemoryTool()
    runner = _runner(
        [
            AssistantTurn("bad", (_call("bad", "read_file", {}, bad="broken JSON"),)),
            AssistantTurn("good", (_call("good", "read_file", {"path": "app.py"}),)),
        ],
        _registry(read),
        _success_verifier(),
    )

    result = runner.run("Inspect")

    assert not result.success
    assert result.stop_reason == "model_exhausted"
    assert read.seen == [{"path": "app.py"}]
    assert json.loads(result.messages[3]["content"])["error_code"] == "invalid_arguments"


def test_agent_treats_an_empty_argument_error_as_invalid_arguments() -> None:
    read = MemoryTool()
    runner = _runner(
        [AssistantTurn("bad", (_call("bad", "read_file", {"path": "app.py"}, bad=""),))],
        _registry(read),
        _success_verifier(),
    )

    result = runner.run("Inspect")

    assert read.seen == []
    assert json.loads(result.messages[-1]["content"])["error_code"] == "invalid_arguments"


def test_agent_allows_model_to_recover_from_unknown_tool() -> None:
    read = MemoryTool()
    runner = _runner(
        [
            AssistantTurn("wrong", (_call("missing", "not_registered", {}),)),
            AssistantTurn("right", (_call("read", "read_file", {"path": "app.py"}),)),
        ],
        _registry(read),
        _success_verifier(),
    )

    result = runner.run("Inspect")

    assert result.stop_reason == "model_exhausted"
    assert read.seen == [{"path": "app.py"}]
    assert json.loads(result.messages[3]["content"])["error_code"] == "unknown_tool"


def test_agent_failed_finish_returns_verifier_result_and_can_succeed_after_a_second_edit() -> None:
    edit = EditTool()
    finish = FinishRequestTool()
    verifier = ScriptedVerifier(
        [
            ToolResult.failure("tests fail", "command_failed", exit_code=1, stderr="hidden"),
            ToolResult.success({"verified": True}, exit_code=0),
        ]
    )
    runner = _runner(
        [
            AssistantTurn(
                "edit one",
                (
                    _call(
                        "edit1", "edit_file", {"path": "app.py", "old_text": "a", "new_text": "b"}
                    ),
                ),
            ),
            AssistantTurn("finish one", (_call("finish1", "finish", {"summary": "try"}),)),
            AssistantTurn(
                "edit two",
                (
                    _call(
                        "edit2", "edit_file", {"path": "app.py", "old_text": "b", "new_text": "c"}
                    ),
                ),
            ),
            AssistantTurn("finish two", (_call("finish2", "finish", {"summary": "done"}),)),
        ],
        _registry(edit, finish),
        verifier,
    )

    result = runner.run("Fix")

    assert result.success
    assert verifier.calls == 2
    assert result.state.edit_count == 2
    assert result.state.last_verify_exit_code == 0
    first_finish = next(
        message for message in result.messages if message.get("tool_call_id") == "finish1"
    )
    assert json.loads(first_finish["content"])["error_code"] == "command_failed"


def test_agent_rejects_finish_when_no_edit_has_succeeded() -> None:
    finish = FinishRequestTool()
    verifier = _success_verifier()
    runner = _runner(
        [AssistantTurn("done", (_call("finish", "finish", {"summary": "nothing"}),))],
        _registry(finish),
        verifier,
    )

    result = runner.run("Fix")

    assert not result.success
    assert result.stop_reason == "model_exhausted"
    assert verifier.calls == 0
    assert json.loads(result.messages[-1]["content"])["error_code"] == "no_edits"


def test_agent_rejects_finish_when_only_a_non_source_file_changed() -> None:
    edit = EditTool(ToolResult.success({"path": "README.md", "changed": True}))
    finish = FinishRequestTool()
    verifier = _success_verifier()
    runner = _runner(
        [
            AssistantTurn(
                "edit docs",
                (
                    _call(
                        "edit",
                        "edit_file",
                        {"path": "README.md", "old_text": "a", "new_text": "b"},
                    ),
                ),
            ),
            AssistantTurn("done", (_call("finish", "finish", {"summary": "done"}),)),
        ],
        _registry(edit, finish),
        verifier,
    )

    result = runner.run("Fix the Python code")

    assert not result.success
    assert result.state.edit_count == 1
    assert verifier.calls == 0
    assert json.loads(result.messages[-1]["content"])["error_code"] == "no_source_edits"


def test_agent_requires_verifier_exit_code_zero_before_succeeding() -> None:
    edit = EditTool()
    finish = FinishRequestTool()
    verifier = ScriptedVerifier([ToolResult.success({"verified": "but missing status"})])
    runner = _runner(
        [
            AssistantTurn(
                "edit",
                (_call("edit", "edit_file", {"path": "app.py", "old_text": "a", "new_text": "b"}),),
            ),
            AssistantTurn("finish", (_call("finish", "finish", {"summary": "done"}),)),
        ],
        _registry(edit, finish),
        verifier,
    )

    result = runner.run("Fix")

    assert not result.success
    assert result.stop_reason == "model_exhausted"
    assert result.state.last_verify_exit_code == -1
    assert json.loads(result.messages[-1]["content"])["error_code"] == "verification_invalid_result"


@pytest.mark.parametrize(
    "malformed",
    [
        _bypassed_tool_result(exit_code=False),
        _bypassed_tool_result(stdout=123),
        _bypassed_tool_result(error=123),
    ],
)
def test_agent_never_succeeds_from_bypassed_malformed_verifier_result(
    malformed: ToolResult,
) -> None:
    edit = EditTool()
    finish = FinishRequestTool()
    runner = _runner(
        [
            AssistantTurn(
                "edit",
                (_call("edit", "edit_file", {"path": "app.py", "old_text": "a", "new_text": "b"}),),
            ),
            AssistantTurn("finish", (_call("finish", "finish", {"summary": "done"}),)),
        ],
        _registry(edit, finish),
        ScriptedVerifier([malformed]),
    )

    result = runner.run("Fix")

    assert not result.success
    assert json.loads(result.messages[-1]["content"])["error_code"] == "verification_invalid_result"


def test_agent_stops_on_a_model_response_without_tools() -> None:
    runner = _runner([AssistantTurn("I am done")], _registry(), _success_verifier())

    result = runner.run("Fix")

    assert not result.success
    assert result.stop_reason == "model_stopped_without_finish"
    assert result.final_text == "I am done"


def test_agent_reports_model_error_without_original_error_details() -> None:
    class BrokenModel:
        def complete(self, messages: Any, tools: Any) -> AssistantTurn:
            raise ModelError("secret upstream issue", code="model_request_failed", retryable=False)

    from testpilot.agent import AgentRunner

    result = AgentRunner(BrokenModel(), _registry(), _success_verifier()).run("Fix")

    assert not result.success
    assert result.stop_reason == "model_request_failed"
    assert "secret" not in result.final_text


def test_agent_stops_after_maximum_iterations() -> None:
    read = MemoryTool()
    runner = _runner(
        [
            AssistantTurn("one", (_call("one", "read_file", {"path": "a.py"}),)),
            AssistantTurn("two", (_call("two", "read_file", {"path": "b.py"}),)),
        ],
        _registry(read),
        _success_verifier(),
        max_iterations=1,
    )

    result = runner.run("Fix")

    assert result.stop_reason == "max_iterations"
    assert read.seen == [{"path": "a.py"}]


def test_agent_stops_after_repeated_call_batches_without_progress() -> None:
    read = MemoryTool()
    turns = [
        AssistantTurn("again", (_call(f"read-{index}", "read_file", {"path": "a.py"}),))
        for index in range(3)
    ]
    runner = _runner(turns, _registry(read), _success_verifier(), max_repeated_calls=3)

    result = runner.run("Fix")

    assert result.stop_reason == "repeated_no_progress"
    assert len(read.seen) == 3


def test_agent_executes_multiple_calls_serially_and_protects_test_assets_by_default() -> None:
    read = MemoryTool()
    edit = EditTool()
    runner = _runner(
        [
            AssistantTurn(
                "two calls",
                (
                    _call("read", "read_file", {"path": "app.py"}),
                    _call(
                        "edit",
                        "edit_file",
                        {"path": "tests/test_app.py", "old_text": "x", "new_text": "y"},
                    ),
                ),
            )
        ],
        _registry(read, edit),
        _success_verifier(),
    )

    result = runner.run("Fix")

    assert read.seen == [{"path": "app.py"}]
    assert edit.seen == []
    tool_messages = [message for message in result.messages if message.get("role") == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["read", "edit"]
    assert json.loads(tool_messages[1]["content"])["error_code"] == "protected_path"


def test_agent_rejects_duplicate_tool_call_ids_before_any_side_effect() -> None:
    edit = EditTool()
    finish = FinishRequestTool()
    verifier = _success_verifier()
    runner = _runner(
        [
            AssistantTurn(
                "malformed ids",
                (
                    _call(
                        "duplicate",
                        "edit_file",
                        {"path": "app.py", "old_text": "a", "new_text": "b"},
                    ),
                    _call("duplicate", "finish", {"summary": "should not run"}),
                ),
            )
        ],
        _registry(edit, finish),
        verifier,
    )

    result = runner.run("Fix")

    assert not result.success
    assert result.stop_reason == "invalid_model_response"
    assert edit.seen == []
    assert verifier.calls == 0
    assert result.messages == (
        {
            "role": "developer",
            "content": result.messages[0]["content"],
        },
        {"role": "user", "content": "Fix"},
    )


def test_agent_does_not_succeed_when_later_call_edits_after_finish() -> None:
    edit = EditTool()
    finish = FinishRequestTool()
    verifier = _success_verifier()
    approval = FakeApproval()
    runner = _runner(
        [
            AssistantTurn(
                "first edit",
                (
                    _call(
                        "edit1", "edit_file", {"path": "app.py", "old_text": "a", "new_text": "b"}
                    ),
                ),
            ),
            AssistantTurn(
                "finish then edit",
                (
                    _call("finish", "finish", {"summary": "done"}),
                    _call(
                        "edit2", "edit_file", {"path": "app.py", "old_text": "b", "new_text": "c"}
                    ),
                ),
            ),
        ],
        _registry(edit, finish),
        verifier,
        approval=approval,
    )

    result = runner.run("Fix")

    assert not result.success
    assert result.stop_reason == "model_exhausted"
    assert verifier.calls == 1
    assert result.state.edit_count == 2
    assert not result.state.verified_after_last_edit
    assert approval.requests == []
    assert approval.rollback_calls == 0


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_a.py",
        "tests/unit/fixtures/expected.txt",
        "test/unit/fixtures/expected.txt",
        "test_root.py",
        "src/test_a.py",
        "calculator_test.py",
        "src/calculator_test.py",
        "conftest.py",
        "pytest.ini",
        ".pytest.ini",
        "pytest.toml",
        "pyproject.toml",
    ],
)
def test_agent_blocks_default_protected_verification_assets(path: str) -> None:
    edit = EditTool()
    runner = _runner(
        [
            AssistantTurn(
                "edit",
                (_call("edit", "edit_file", {"path": path, "old_text": "a", "new_text": "b"}),),
            )
        ],
        _registry(edit),
        _success_verifier(),
    )

    result = runner.run("Fix")

    assert edit.seen == []
    assert json.loads(result.messages[-1]["content"])["error_code"] == "protected_path"


def test_agent_can_explicitly_disable_protected_path_rules() -> None:
    edit = EditTool()
    runner = _runner(
        [
            AssistantTurn(
                "edit",
                (
                    _call(
                        "edit",
                        "edit_file",
                        {"path": "tests/test_a.py", "old_text": "a", "new_text": "b"},
                    ),
                ),
            )
        ],
        _registry(edit),
        _success_verifier(),
        protected_patterns=(),
    )

    runner.run("Fix")

    assert edit.seen == [{"path": "tests/test_a.py", "old_text": "a", "new_text": "b"}]


def test_agent_rejects_string_protected_patterns() -> None:
    from testpilot.agent import AgentRunner

    with pytest.raises(TypeError, match="protected_patterns"):
        AgentRunner(FakeModel([]), _registry(), _success_verifier(), protected_patterns="tests/**")  # type: ignore[arg-type]


@pytest.mark.skipif(os.name != "nt", reason="Windows aliases are case-insensitive")
def test_agent_early_protection_handles_windows_case_alias_with_real_workspace(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "tests"
    protected.mkdir()
    target = protected / "test_asset.py"
    target.write_text("old", encoding="utf-8")
    registry = _registry(EditFileTool(Workspace(tmp_path)))
    runner = _runner(
        [
            AssistantTurn(
                "attempt test edit",
                (
                    _call(
                        "edit",
                        "edit_file",
                        {"path": "TESTS\\test_asset.py", "old_text": "old", "new_text": "new"},
                    ),
                ),
            )
        ],
        registry,
        _success_verifier(),
    )

    result = runner.run("Fix")

    assert target.read_text(encoding="utf-8") == "old"
    assert json.loads(result.messages[-1]["content"])["error_code"] == "protected_path"


def test_workspace_canonical_protection_blocks_symlinked_edit_and_write(tmp_path: Path) -> None:
    protected = tmp_path / "tests"
    protected.mkdir()
    target = protected / "test_asset.py"
    target.write_text("old", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(protected, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    workspace = Workspace(tmp_path)
    registry = _registry(EditFileTool(workspace), WriteFileTool(workspace))
    runner = _runner(
        [
            AssistantTurn(
                "attempt aliases",
                (
                    _call(
                        "edit",
                        "edit_file",
                        {"path": "alias/test_asset.py", "old_text": "old", "new_text": "new"},
                    ),
                    _call(
                        "write", "write_file", {"path": "alias/new_source.py", "content": "unsafe"}
                    ),
                ),
            )
        ],
        registry,
        _success_verifier(),
    )

    result = runner.run("Fix")

    assert target.read_text(encoding="utf-8") == "old"
    assert not (protected / "new_source.py").exists()
    assert [json.loads(message["content"])["error_code"] for message in result.messages[-2:]] == [
        "protected_path",
        "protected_path",
    ]


@pytest.mark.parametrize(
    "turn",
    [
        AssistantTurn(
            123, (_call("edit", "edit_file", {"path": "app.py", "old_text": "a", "new_text": "b"}),)
        ),  # type: ignore[arg-type]
        AssistantTurn("bad", (object(),)),  # type: ignore[arg-type]
        AssistantTurn("bad", (ToolCall("edit", 123, {"path": "app.py"}),)),  # type: ignore[arg-type]
        AssistantTurn("bad", (ToolCall("edit", "edit_file", {"path": Path("app.py")}),)),
        AssistantTurn("bad", (ToolCall("edit", "edit_file", {"value": {"not-json"}}),)),
        AssistantTurn("bad", (ToolCall("edit", "edit_file", {"value": math.nan}),)),
        AssistantTurn("bad", (ToolCall("edit", "edit_file", {}, argument_error=123),)),  # type: ignore[arg-type]
    ],
)
def test_agent_rejects_malformed_turn_before_tool_side_effects(turn: AssistantTurn) -> None:
    edit = EditTool()
    runner = _runner([turn], _registry(edit), _success_verifier())

    result = runner.run("Fix")

    assert result.stop_reason == "invalid_model_response"
    assert edit.seen == []
    assert len(result.messages) == 2


@pytest.mark.parametrize("bad_data", [Path("not-json"), {"set"}, math.nan])
def test_agent_converts_non_json_tool_results_to_structured_failures(bad_data: Any) -> None:
    read = MemoryTool(result=ToolResult.success({"value": bad_data}))
    runner = _runner(
        [AssistantTurn("read", (_call("read", "read_file", {"path": "app.py"}),))],
        _registry(read),
        _success_verifier(),
    )

    result = runner.run("Fix")

    assert result.stop_reason == "model_exhausted"
    assert json.loads(result.messages[-1]["content"])["error_code"] == "invalid_tool_result"


@pytest.mark.parametrize("bad_data", [Path("not-json"), {"set"}, math.nan])
def test_agent_converts_non_json_verifier_results_to_structured_failures(bad_data: Any) -> None:
    edit = EditTool()
    finish = FinishRequestTool()
    verifier = ScriptedVerifier([ToolResult.success({"value": bad_data}, exit_code=0)])
    runner = _runner(
        [
            AssistantTurn(
                "edit",
                (_call("edit", "edit_file", {"path": "app.py", "old_text": "a", "new_text": "b"}),),
            ),
            AssistantTurn("finish", (_call("finish", "finish", {"summary": "done"}),)),
        ],
        _registry(edit, finish),
        verifier,
    )

    result = runner.run("Fix")

    assert not result.success
    assert result.stop_reason == "model_exhausted"
    assert json.loads(result.messages[-1]["content"])["error_code"] == "verification_invalid_result"


def test_agent_creates_fresh_anchored_context_for_each_run() -> None:
    model = FakeModel([AssistantTurn("first"), AssistantTurn("second")])
    from testpilot.agent import AgentRunner

    runner = AgentRunner(model, _registry(), _success_verifier())

    first = runner.run("FIRST")
    second = runner.run("SECOND")

    assert first.stop_reason == second.stop_reason == "model_stopped_without_finish"
    assert [message["role"] for message in model.received_inputs[1][0]] == ["developer", "user"]
    assert model.received_inputs[1][0][1]["content"] == "SECOND"


def test_failed_verification_is_not_marked_verified_after_last_edit() -> None:
    edit = EditTool()
    finish = FinishRequestTool()
    approval = FakeApproval()
    runner = _runner(
        [
            AssistantTurn(
                "edit",
                (_call("edit", "edit_file", {"path": "app.py", "old_text": "a", "new_text": "b"}),),
            ),
            AssistantTurn("finish", (_call("finish", "finish", {"summary": "done"}),)),
        ],
        _registry(edit, finish),
        ScriptedVerifier([ToolResult.failure("fails", "command_failed", exit_code=1)]),
        approval=approval,
    )

    result = runner.run("Fix")

    assert result.state.last_verify_exit_code == 1
    assert not result.state.verified_after_last_edit
    assert approval.requests == []
    assert approval.rollback_calls == 0


def test_agent_trace_is_best_effort_and_does_not_fail_run(tmp_path: Path) -> None:
    class BrokenTrace:
        path = tmp_path / "trace.jsonl"

        def record(self, event: str, payload: Mapping[str, Any]) -> None:
            raise OSError("disk unavailable")

    runner = _runner([AssistantTurn("done")], _registry(), _success_verifier(), trace=BrokenTrace())

    result = runner.run("Fix")

    assert result.stop_reason == "model_stopped_without_finish"
    assert result.trace_path == tmp_path / "trace.jsonl"


def test_agent_trace_records_safe_argument_summaries_and_elapsed_time(tmp_path: Path) -> None:
    trace = CapturingTrace(tmp_path / "trace.jsonl")
    edit = EditTool()
    finish = FinishRequestTool()
    private_source = "private source text that must not enter the trace"
    runner = _runner(
        [
            AssistantTurn(
                "fix",
                (
                    _call(
                        "edit",
                        "edit_file",
                        {
                            "path": "app.py",
                            "old_text": "old",
                            "new_text": private_source,
                        },
                    ),
                ),
            ),
            AssistantTurn("done", (_call("finish", "finish", {"summary": "fixed"}),)),
        ],
        _registry(edit, finish),
        _success_verifier(),
        trace=trace,
    )

    result = runner.run("Fix")

    assert result.success
    edit_call = next(
        payload
        for event, payload in trace.events
        if event == "tool_call" and payload["tool"] == "edit_file"
    )
    assert edit_call["argument_summary"] == {
        "count": 3,
        "json_chars": 97,
        "parse_error": False,
        "types": {"string": 3},
    }
    tool_results = [payload for event, payload in trace.events if event == "tool_result"]
    assert tool_results
    assert all(
        isinstance(payload["duration_ms"], (int, float)) and payload["duration_ms"] >= 0
        for payload in tool_results
    )
    verification_complete = next(
        payload
        for event, payload in trace.events
        if event == "verification" and payload["stage"] == "complete"
    )
    assert verification_complete["ok"] is True
    assert verification_complete["exit_code"] == 0
    assert verification_complete["duration_ms"] >= 0
    model_completions = [
        payload
        for event, payload in trace.events
        if event == "model_turn" and payload["stage"] == "complete"
    ]
    assert len(model_completions) == 2
    assert all(payload["ok"] is True for payload in model_completions)
    assert all(payload["duration_ms"] >= 0 for payload in model_completions)
    assert private_source not in json.dumps(trace.events, ensure_ascii=False)
