from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from testpilot.command import CommandRunner, FinishTool, Verifier
from testpilot.model import FakeModel
from testpilot.registry import ToolRegistry
from testpilot.tools import EditFileTool, ReadFileTool
from testpilot.types import AssistantTurn, ToolCall
from testpilot.workspace import Workspace


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
