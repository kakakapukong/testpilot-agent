"""A completely local, keyless TestPilot repair demonstration."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from .agent import AgentRunner
from .command import CommandRunner, FinishTool, RunCommandTool, Verifier
from .model import FakeModel
from .registry import ToolRegistry
from .tools import EditFileTool, ListFilesTool, ReadFileTool, SearchTextTool, WriteFileTool
from .trace import JsonlTrace
from .types import AssistantTurn, ToolCall
from .workspace import Workspace


def _registry(workspace: Workspace, runner: CommandRunner) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        ListFilesTool(workspace),
        ReadFileTool(workspace),
        SearchTextTool(workspace),
        EditFileTool(workspace),
        WriteFileTool(workspace),
        RunCommandTool(runner),
        FinishTool(),
    ):
        registry.register(tool)
    return registry


def _script() -> list[AssistantTurn]:
    return [
        AssistantTurn(
            "Inspect the source", (ToolCall("read", "read_file", {"path": "calculator.py"}),)
        ),
        AssistantTurn(
            "Repair subtraction",
            (
                ToolCall(
                    "edit",
                    "edit_file",
                    {
                        "path": "calculator.py",
                        "old_text": "return left + right",
                        "new_text": "return left - right",
                    },
                ),
            ),
        ),
        AssistantTurn(
            "Ask the host to verify", (ToolCall("finish", "finish", {"summary": "Done"}),)
        ),
    ]


def _prepare_workspace(root: Path) -> None:
    (root / "calculator.py").write_text(
        "def subtract(left, right):\n    return left + right\n", encoding="utf-8"
    )
    (root / "test_calculator.py").write_text(
        "from calculator import subtract\n\n\ndef test_subtract():\n    assert subtract(5, 3) == 2\n",
        encoding="utf-8",
    )


def run_demo(root: Path) -> bool:
    """Run the failure -> repair -> independent verification sequence at *root*."""
    _prepare_workspace(root)
    command_runner = CommandRunner(root)
    verifier = Verifier(command_runner, (sys.executable, "-m", "pytest", "-q"))
    before = verifier.verify()
    if before.ok:
        print("BEFORE=UNEXPECTED_PASS")
        return False
    print("BEFORE=FAIL")
    workspace = Workspace(root)
    agent = AgentRunner(
        FakeModel(_script()),
        _registry(workspace, command_runner),
        verifier,
        trace=JsonlTrace(root / ".testpilot" / "traces" / "offline-demo.jsonl"),
        max_iterations=6,
    )
    result = agent.run("Fix calculator.subtract without editing tests.")
    print(f"AGENT={'SUCCESS' if result.success else 'FAILED'}")
    after = verifier.verify()
    if result.success and after.ok:
        print("AFTER=PASS")
        return True
    print("AFTER=FAIL")
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TestPilot's offline repair demo.")
    parser.add_argument(
        "--keep", metavar="PATH", help="Keep the demo workspace at this new or empty path."
    )
    args = parser.parse_args(argv)
    if args.keep:
        root = Path(args.keep).expanduser().resolve(strict=False)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            print("Demo --keep path must be new or empty.")
            return 1
        root.mkdir(parents=True, exist_ok=True)
        return 0 if run_demo(root) else 1
    with TemporaryDirectory(prefix="testpilot-offline-demo-") as temporary:
        return 0 if run_demo(Path(temporary)) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
