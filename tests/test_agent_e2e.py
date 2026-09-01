from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from testpilot.approval import ChangeJournal
from testpilot.command import CommandRunner, FinishTool, Verifier
from testpilot.model import FakeModel
from testpilot.registry import ToolRegistry
from testpilot.reviewer import ReviewResult
from testpilot.tools import EditFileTool, ReadFileTool
from testpilot.types import AssistantTurn, ToolCall
from testpilot.workspace import Workspace


@dataclass
class JournalApproval:
    journal: ChangeJournal
    approved: bool
    requests: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    commit_calls: int = 0
    rollback_calls: int = 0

    def request(
        self,
        *,
        changed_files: tuple[str, ...],
        verification_exit_code: int,
    ) -> bool:
        self.requests.append((tuple(changed_files), verification_exit_code))
        return self.approved

    def commit(self) -> None:
        self.commit_calls += 1
        self.journal.commit()

    def rollback(self) -> None:
        self.rollback_calls += 1
        self.journal.rollback()


@dataclass
class RecordingReviewer:
    result: ReviewResult
    requests: list[tuple[str, tuple[str, ...], int]] = field(default_factory=list)

    def review(
        self,
        *,
        task: str,
        changed_files: tuple[str, ...],
        verification_exit_code: int,
    ) -> ReviewResult:
        self.requests.append((task, tuple(changed_files), verification_exit_code))
        return self.result


def _link_directory(alias: Path, target: Path) -> None:
    try:
        alias.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        pass
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        pytest.skip(f"directory link unavailable: {error}")


def _real_approval_runner(
    tmp_path: Path,
    *,
    approved: bool,
) -> tuple[object, JournalApproval, CommandRunner, Path, bytes]:
    from testpilot.agent import AgentRunner

    source = tmp_path / "calculator.py"
    original = b"def add(left: int, right: int) -> int:\n    return left - right\n"
    source.write_bytes(original)
    (tmp_path / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    command_runner = CommandRunner(tmp_path)
    journal = ChangeJournal(tmp_path)
    approval = JournalApproval(journal, approved=approved)
    workspace = Workspace(tmp_path, change_recorder=journal)
    registry = ToolRegistry()
    registry.register(EditFileTool(workspace))
    registry.register(FinishTool())
    model = FakeModel(
        [
            AssistantTurn(
                "fix",
                (
                    ToolCall(
                        "edit",
                        "edit_file",
                        {
                            "path": "calculator.py",
                            "old_text": "left - right",
                            "new_text": "left + right",
                        },
                    ),
                ),
            ),
            AssistantTurn("finish", (ToolCall("finish", "finish", {"summary": "fixed"}),)),
        ]
    )
    agent = AgentRunner(
        model,
        registry,
        Verifier(command_runner, [sys.executable, "-m", "pytest", "-q"]),
        approval=approval,
    )
    return agent, approval, command_runner, source, original


def test_agent_repairs_buggy_calculator_and_only_succeeds_when_real_pytest_passes(
    tmp_path: Path,
) -> None:
    from testpilot.agent import AgentRunner

    (tmp_path / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n", encoding="utf-8"
    )
    (tmp_path / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    runner = CommandRunner(tmp_path)
    assert not runner.run([sys.executable, "-m", "pytest", "-q"]).ok

    workspace = Workspace(tmp_path)
    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace))
    registry.register(EditFileTool(workspace))
    registry.register(FinishTool())
    model = FakeModel(
        [
            AssistantTurn("inspect", (ToolCall("read", "read_file", {"path": "calculator.py"}),)),
            AssistantTurn(
                "fix",
                (
                    ToolCall(
                        "edit",
                        "edit_file",
                        {
                            "path": "calculator.py",
                            "old_text": "left - right",
                            "new_text": "left + right + 0",
                        },
                    ),
                ),
            ),
            AssistantTurn("finish", (ToolCall("finish", "finish", {"summary": "fixed addition"}),)),
        ]
    )

    result = AgentRunner(
        model,
        registry,
        Verifier(runner, [sys.executable, "-m", "pytest", "-q"]),
    ).run("Fix the calculator add function.")

    assert result.success
    assert result.stop_reason == "verified"
    assert result.state.edit_count == 1
    assert "left + right + 0" in (tmp_path / "calculator.py").read_text(encoding="utf-8")
    assert runner.run([sys.executable, "-m", "pytest", "-q"]).ok


def test_real_workspace_keeps_verified_bytes_after_approval(tmp_path: Path) -> None:
    agent, approval, command_runner, source, original = _real_approval_runner(
        tmp_path,
        approved=True,
    )

    result = agent.run("Fix the calculator.")

    assert result.success
    assert result.stop_reason == "verified"
    assert result.state.approval_status == "approved"
    assert approval.requests == [(("calculator.py",), 0)]
    assert approval.commit_calls == 1
    assert approval.rollback_calls == 0
    assert source.read_bytes() != original
    assert b"left + right" in source.read_bytes()
    assert command_runner.run([sys.executable, "-m", "pytest", "-q"]).ok


def test_real_workspace_runs_review_after_pytest_before_approval(tmp_path: Path) -> None:
    agent, approval, command_runner, source, original = _real_approval_runner(
        tmp_path,
        approved=True,
    )
    reviewer = RecordingReviewer(ReviewResult("pass", "The repair is correct."))
    agent.reviewer = reviewer

    result = agent.run("Fix the calculator.")

    assert result.success
    assert result.state.review_status == "passed"
    assert reviewer.requests == [("Fix the calculator.", ("calculator.py",), 0)]
    assert approval.requests == [(('calculator.py',), 0)]
    assert source.read_bytes() != original
    assert command_runner.run([sys.executable, "-m", "pytest", "-q"]).ok


def test_real_workspace_restores_exact_original_bytes_after_rejection(tmp_path: Path) -> None:
    agent, approval, command_runner, source, original = _real_approval_runner(
        tmp_path,
        approved=False,
    )

    result = agent.run("Fix the calculator.")

    assert not result.success
    assert result.stop_reason == "approval_rejected"
    assert result.state.approval_status == "rejected"
    assert approval.requests == [(("calculator.py",), 0)]
    assert approval.commit_calls == 0
    assert approval.rollback_calls == 1
    assert source.read_bytes() == original
    assert not command_runner.run([sys.executable, "-m", "pytest", "-q"]).ok


def test_agent_cannot_rewrite_an_explicit_pytest_target_to_fake_success(tmp_path: Path) -> None:
    from testpilot.agent import AgentRunner

    checks = tmp_path / "checks"
    checks.mkdir()
    verification_test = checks / "case.py"
    verification_test.write_text("def test_gate():\n    assert False\n", encoding="utf-8")
    command_runner = CommandRunner(tmp_path)
    verifier = Verifier(
        command_runner,
        [sys.executable, "-m", "pytest", "checks/case.py", "-q"],
    )
    workspace = Workspace(tmp_path)
    registry = ToolRegistry()
    registry.register(EditFileTool(workspace))
    registry.register(FinishTool())
    model = FakeModel(
        [
            AssistantTurn(
                "rewrite verification target",
                (
                    ToolCall(
                        "edit-verifier",
                        "edit_file",
                        {
                            "path": "checks/case.py",
                            "old_text": "assert False",
                            "new_text": "assert True",
                        },
                    ),
                ),
            ),
            AssistantTurn(
                "finish",
                (ToolCall("finish", "finish", {"summary": "verification now passes"}),),
            ),
        ]
    )

    result = AgentRunner(model, registry, verifier).run("Fix the failing project.")

    assert not result.success
    assert result.state.edit_count == 0
    assert verification_test.read_text(encoding="utf-8") == "def test_gate():\n    assert False\n"
    assert not verifier.verify().ok


def test_agent_cannot_rewrite_a_verifier_through_a_directory_alias(tmp_path: Path) -> None:
    from testpilot.agent import AgentRunner

    verification_directory = tmp_path / "verification"
    verification_directory.mkdir()
    verification_test = verification_directory / "case.py"
    verification_test.write_text("def test_gate():\n    assert False\n", encoding="utf-8")
    _link_directory(tmp_path / "alias", verification_directory)
    command_runner = CommandRunner(tmp_path)
    verifier = Verifier(
        command_runner,
        [sys.executable, "-m", "pytest", "verification/case.py", "-q"],
    )
    workspace = Workspace(tmp_path)
    registry = ToolRegistry()
    registry.register(EditFileTool(workspace))
    registry.register(FinishTool())
    model = FakeModel(
        [
            AssistantTurn(
                "rewrite verifier through alias",
                (
                    ToolCall(
                        "edit-verifier",
                        "edit_file",
                        {
                            "path": "alias/case.py",
                            "old_text": "assert False",
                            "new_text": "assert True",
                        },
                    ),
                ),
            ),
            AssistantTurn(
                "finish",
                (ToolCall("finish", "finish", {"summary": "verification now passes"}),),
            ),
        ]
    )

    result = AgentRunner(model, registry, verifier).run("Fix the failing project.")

    assert not result.success
    assert result.state.edit_count == 0
    assert verification_test.read_text(encoding="utf-8") == "def test_gate():\n    assert False\n"
