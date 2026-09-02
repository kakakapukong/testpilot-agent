"""A completely local, keyless TestPilot repair demonstration."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .agent import AgentRunner
from .approval import ChangeJournal
from .checkpoint import CheckpointRequest, CheckpointSession, CheckpointStore
from .command import CommandRunner, FinishTool, RunCommandTool, Verifier
from .memory import MemoryStore
from .memory_agent import MemoryAgent, build_memory_registry
from .model import FakeModel
from .registry import ToolRegistry
from .reviewer import ReviewerAgent, build_reviewer_registry
from .tools import EditFileTool, ListFilesTool, ReadFileTool, SearchTextTool, WriteFileTool
from .trace import JsonlTrace
from .types import AgentRunResult, AssistantTurn, ToolCall
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


def _memory_script() -> list[AssistantTurn]:
    return [
        AssistantTurn(
            "Submit a reusable summary",
            (
                ToolCall(
                    "memory-submit",
                    "submit_memory",
                    {
                        "problem": "A Python subtraction helper returned an addition result.",
                        "root_cause": "The implementation used the addition operator by mistake.",
                        "solution": "Inspect the arithmetic return expression and use subtraction.",
                        "verification": "Run the host-owned pytest suite and independent review.",
                        "keywords": ["python", "subtraction", "operator", "calculator"],
                    },
                ),
            ),
        )
    ]


def _second_repair_script() -> list[AssistantTurn]:
    return [
        AssistantTurn(
            "Inspect the similar helper",
            (ToolCall("read-second", "read_file", {"path": "calculator.py"}),),
        ),
        AssistantTurn(
            "Apply the remembered subtraction pattern",
            (
                ToolCall(
                    "edit-second",
                    "edit_file",
                    {
                        "path": "calculator.py",
                        "old_text": "def difference(left, right):\n    return left + right",
                        "new_text": "def difference(left, right):\n    return left - right",
                    },
                ),
            ),
        ),
        AssistantTurn(
            "Ask the host to verify the similar repair",
            (ToolCall("finish-second", "finish", {"summary": "Done"}),),
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


def _prepare_second_task(root: Path) -> None:
    with (root / "calculator.py").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n\ndef difference(left, right):\n    return left + right\n")
    with (root / "test_calculator.py").open(
        "a", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(
            "\n\nfrom calculator import difference\n\n\n"
            "def test_difference():\n    assert difference(9, 4) == 5\n"
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


@dataclass(frozen=True)
class MemoryDemoSummary:
    """The two successful logical tasks in the keyless memory demonstration."""

    first: AgentRunResult
    second: AgentRunResult


class _DemoError(RuntimeError):
    """One deterministic demo invariant failed without exposing private content."""


def run_memory_demo(root: Path) -> MemoryDemoSummary:
    """Run a checkpointed repair, save its memory, then retrieve it for a new task."""
    _prepare_workspace(root)
    command_runner = CommandRunner(root)
    verify_command = (sys.executable, "-m", "pytest", "-q")
    verifier = Verifier(command_runner, verify_command)
    if verifier.verify().ok:
        raise _DemoError("BEFORE=UNEXPECTED_PASS")

    task = "Fix the subtraction operator bug in calculator.subtract without editing tests."
    trace_path = root / ".testpilot" / "traces" / "offline-demo.jsonl"
    trace = JsonlTrace(trace_path)
    journal = ChangeJournal(root)
    workspace = Workspace(root, change_recorder=journal)
    checkpoint_store = CheckpointStore(root)
    first_session = CheckpointSession.create(
        store=checkpoint_store,
        journal=journal,
        request=CheckpointRequest(
            task=task,
            verifier=verify_command,
            max_iterations=3,
            trace_path=".testpilot/traces/offline-demo.jsonl",
        ),
    )
    interrupted = AgentRunner(
        FakeModel(_interrupted_script()),
        _registry(workspace, command_runner),
        verifier,
        trace=trace,
        checkpoint=first_session,
        max_iterations=3,
    ).run(task)
    if interrupted.success or not interrupted.resume_available or not first_session.path.exists():
        raise _DemoError("INTERRUPTED=FAILED")

    restored_journal = ChangeJournal(root)
    restored_session, resume = CheckpointSession.restore(
        store=checkpoint_store,
        journal=restored_journal,
        run_id=first_session.run_id,
    )
    restored_workspace = Workspace(root, change_recorder=restored_journal)
    reviewer = ReviewerAgent(
        FakeModel(_review_script()),
        build_reviewer_registry(restored_workspace),
    )
    first_approval = _DemoApproval(restored_journal)
    memory_store = MemoryStore(root)
    first_result = AgentRunner(
        FakeModel(_resume_script()),
        _registry(restored_workspace, command_runner),
        verifier,
        trace=trace,
        approval=first_approval,
        reviewer=reviewer,
        checkpoint=restored_session,
        memory_store=memory_store,
        memory_agent=MemoryAgent(FakeModel(_memory_script()), build_memory_registry()),
        max_iterations=3,
    ).run(task, resume=resume)
    after_first = verifier.verify()
    if (
        not first_result.success
        or not first_approval.approved
        or not after_first.ok
        or restored_session.path.exists()
        or first_result.memory_saved != "yes"
    ):
        raise _DemoError("RESUMED=FAILED")

    _prepare_second_task(root)
    if verifier.verify().ok:
        raise _DemoError("MEMORY_SECOND_SETUP=UNEXPECTED_PASS")
    second_task = "Fix the subtraction operator bug in calculator.difference without editing tests."
    second_trace_path = root / ".testpilot" / "traces" / "offline-memory-second.jsonl"
    second_trace = JsonlTrace(second_trace_path)
    second_journal = ChangeJournal(root)
    second_workspace = Workspace(root, change_recorder=second_journal)
    second_session = CheckpointSession.create(
        store=checkpoint_store,
        journal=second_journal,
        request=CheckpointRequest(
            task=second_task,
            verifier=verify_command,
            max_iterations=4,
            trace_path=".testpilot/traces/offline-memory-second.jsonl",
        ),
    )
    second_approval = _DemoApproval(second_journal)
    second_result = AgentRunner(
        FakeModel(_second_repair_script()),
        _registry(second_workspace, command_runner),
        verifier,
        trace=second_trace,
        approval=second_approval,
        reviewer=ReviewerAgent(
            FakeModel(_review_script()),
            build_reviewer_registry(second_workspace),
        ),
        checkpoint=second_session,
        memory_store=memory_store,
        memory_agent=MemoryAgent(FakeModel(_memory_script()), build_memory_registry()),
        max_iterations=4,
    ).run(second_task)
    if (
        not second_result.success
        or not second_approval.approved
        or not verifier.verify().ok
        or second_result.memories_retrieved != 1
    ):
        raise _DemoError("MEMORY_SECOND=FAILED")
    return MemoryDemoSummary(first_result, second_result)


def run_demo(root: Path) -> bool:
    """Render the stable public output of the keyless checkpoint-and-memory demo."""
    try:
        summary = run_memory_demo(root)
    except _DemoError as error:
        print(str(error))
        return False
    print("BEFORE=FAIL")
    print("INTERRUPTED=CHECKPOINTED")
    print("RESUMED=SUCCESS")
    print(
        "VERIFIED=PASS"
        if summary.first.state.last_verify_exit_code == 0
        and summary.first.state.verified_after_last_edit
        else "VERIFIED=FAILED"
    )
    print(f"REVIEWED={'PASS' if summary.first.state.review_status == 'passed' else 'FAILED'}")
    print(
        f"APPROVED={'SIMULATED' if summary.first.state.approval_status == 'approved' else 'FAILED'}"
    )
    print("AFTER=PASS")
    print(f"MEMORY_FIRST_SAVED={summary.first.memory_saved}")
    print(f"MEMORY_SECOND_RETRIEVED={summary.second.memories_retrieved}")
    memory_reused = summary.second.success and summary.second.memories_retrieved == 1
    print(f"MEMORY_REUSED={'yes' if memory_reused else 'no'}")
    return memory_reused


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
