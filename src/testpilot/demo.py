"""A completely local, keyless TestPilot repair demonstration."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from .agent import AgentRunner
from .approval import ChangeJournal
from .checkpoint import CheckpointRequest, CheckpointSession, CheckpointStore
from .command import CommandRunner, FinishTool, RunCommandTool, Verifier
from .model import FakeModel
from .registry import ToolRegistry
from .reviewer import ReviewerAgent, build_reviewer_registry
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


def _interrupted_script() -> list[AssistantTurn]:
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
    ]


def _resume_script() -> list[AssistantTurn]:
    return [
        AssistantTurn(
            "Ask the host to verify after restart",
            (ToolCall("finish", "finish", {"summary": "Done"}),),
        )
    ]


def _review_script() -> list[AssistantTurn]:
    return [
        AssistantTurn(
            "Inspect the repaired source",
            (ToolCall("review-read", "read_file", {"path": "calculator.py"}),),
        ),
        AssistantTurn(
            "Submit the independent review",
            (
                ToolCall(
                    "review-decision",
                    "submit_review",
                    {
                        "decision": "pass",
                        "feedback": "Subtraction now uses the correct operator.",
                    },
                ),
            ),
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


class _DemoApproval:
    """Deterministic approval used only to make the keyless demo repeatable."""

    def __init__(self, journal: ChangeJournal) -> None:
        self.journal = journal
        self.approved = False

    def request(
        self,
        *,
        changed_files: Sequence[str],
        verification_exit_code: int,
    ) -> bool:
        self.approved = bool(changed_files) and verification_exit_code == 0
        return self.approved

    def commit(self) -> None:
        self.journal.commit()

    def rollback(self) -> None:
        self.journal.rollback()


def run_demo(root: Path) -> bool:
    """Interrupt and resume one fully local verified repair at *root*."""
    _prepare_workspace(root)
    command_runner = CommandRunner(root)
    verify_command = (sys.executable, "-m", "pytest", "-q")
    verifier = Verifier(command_runner, verify_command)
    before = verifier.verify()
    if before.ok:
        print("BEFORE=UNEXPECTED_PASS")
        return False
    print("BEFORE=FAIL")

    task = "Fix calculator.subtract without editing tests."
    trace_path = root / ".testpilot" / "traces" / "offline-demo.jsonl"
    trace = JsonlTrace(trace_path)
    journal = ChangeJournal(root)
    workspace = Workspace(root, change_recorder=journal)
    store = CheckpointStore(root)
    first_session = CheckpointSession.create(
        store=store,
        journal=journal,
        request=CheckpointRequest(
            task=task,
            verifier=verify_command,
            max_iterations=3,
            trace_path=".testpilot/traces/offline-demo.jsonl",
        ),
    )
    first = AgentRunner(
        FakeModel(_interrupted_script()),
        _registry(workspace, command_runner),
        verifier,
        trace=trace,
        checkpoint=first_session,
        max_iterations=3,
    ).run(task)
    if first.success or not first.resume_available or not first_session.path.exists():
        print("INTERRUPTED=FAILED")
        return False
    print("INTERRUPTED=CHECKPOINTED")

    restored_journal = ChangeJournal(root)
    restored_session, resume = CheckpointSession.restore(
        store=store,
        journal=restored_journal,
        run_id=first_session.run_id,
    )
    restored_workspace = Workspace(root, change_recorder=restored_journal)
    reviewer = ReviewerAgent(
        FakeModel(_review_script()),
        build_reviewer_registry(restored_workspace),
    )
    approval = _DemoApproval(restored_journal)
    second = AgentRunner(
        FakeModel(_resume_script()),
        _registry(restored_workspace, command_runner),
        verifier,
        trace=trace,
        approval=approval,
        reviewer=reviewer,
        checkpoint=restored_session,
        max_iterations=3,
    ).run(task, resume=resume)
    print(f"RESUMED={'SUCCESS' if second.success else 'FAILED'}")
    print(
        "VERIFIED=PASS"
        if second.state.last_verify_exit_code == 0
        and second.state.verified_after_last_edit
        else "VERIFIED=FAILED"
    )
    print(f"REVIEWED={'PASS' if second.state.review_status == 'passed' else 'FAILED'}")
    print(f"APPROVED={'SIMULATED' if approval.approved else 'FAILED'}")
    after = verifier.verify()
    if second.success and approval.approved and after.ok and not restored_session.path.exists():
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
