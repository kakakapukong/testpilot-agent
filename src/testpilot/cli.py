"""Small, deliberately explicit command-line entry point for TestPilot."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import monotonic_ns
from typing import Any

from .agent import AgentRunner
from .approval import ChangeJournal, ConsoleApprovalWorkflow
from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointRequest,
    CheckpointSession,
    CheckpointStore,
    ResumeData,
)
from .command import CommandRunner, FinishTool, RunCommandTool, Verifier
from .memory import MemoryStore
from .memory_agent import MemoryAgent, build_memory_registry
from .model import OpenAIChatModel
from .registry import ToolRegistry
from .reviewer import ReviewerAgent, build_reviewer_registry
from .tools import EditFileTool, ListFilesTool, ReadFileTool, SearchTextTool, WriteFileTool
from .trace import JsonlTrace
from .workspace import DEFAULT_PRIVATE_PATTERNS, DEFAULT_PROTECTED_PATTERNS, Workspace

_CHECKPOINT_ERROR_CODES = frozenset(
    {
        "checkpoint_cleanup_failed",
        "checkpoint_finalize_failed",
        "checkpoint_invalid",
        "checkpoint_load_failed",
        "checkpoint_save_failed",
        "checkpoint_too_large",
        "checkpoint_workspace_changed",
        "checkpoint_workspace_mismatch",
    }
)
_MEMORY_WARNING_CODES = frozenset(
    {
        "memory_invalid",
        "memory_invalid_response",
        "memory_load_failed",
        "memory_max_iterations",
        "memory_model_failed",
        "memory_save_failed",
        "memory_stopped_without_submission",
        "memory_too_large",
    }
)


@dataclass(frozen=True)
class CliConfig:
    """Already-validated runtime configuration; values are never logged."""

    workspace: Path
    verifier: tuple[str, ...]
    task: str
    api_key: str
    model: str
    base_url: str | None
    trace_path: Path
    max_iterations: int


@dataclass(frozen=True)
class RunSetup:
    """Fully validated host runtime shared by fresh and resumed modes."""

    config: CliConfig
    journal: ChangeJournal
    checkpoint: CheckpointSession
    resume: ResumeData | None


class _ConfigError(ValueError):
    """A safe-to-display configuration failure without embedded input values."""


def positive_integer(value: str) -> int:
    """Argparse validator for a positive iteration budget."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="testpilot",
        description=(
            "A verification-gated coding Agent with independent review and repository memory."
        ),
    )
    parser.add_argument(
        "--workspace", required=True, metavar="PATH", help="Existing repository directory."
    )
    parser.add_argument(
        "--verify",
        metavar="COMMAND",
        help="Fixed host-owned restricted pytest command, e.g. 'python -m pytest -q'.",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="Resume one active checkpoint in this workspace.",
    )
    parser.add_argument(
        "--trace",
        metavar="PATH",
        help="Optional JSONL trace path, relative to the workspace unless absolute.",
    )
    parser.add_argument(
        "--max-iterations",
        type=positive_integer,
        default=None,
        help="Model/tool rounds for this invocation (fresh default: 12).",
    )
    parser.add_argument(
        "task",
        nargs="?",
        metavar="TASK",
        help="Natural-language repair task for a fresh run.",
    )
    return parser


def _split_windows_command(command: str) -> tuple[str, ...]:
    """Parse one Windows command line while preserving literal path separators."""
    parts: list[str] = []
    index = 0
    while index < len(command):
        while index < len(command) and command[index] in " \t":
            index += 1
        if index == len(command):
            break

        part: list[str] = []
        quoted = False
        while index < len(command):
            character = command[index]
            if character in " \t" and not quoted:
                break
            if character == "\\":
                start = index
                while index < len(command) and command[index] == "\\":
                    index += 1
                backslashes = index - start
                if index < len(command) and command[index] == '"':
                    part.extend("\\" * (backslashes // 2))
                    if backslashes % 2:
                        part.append('"')
                        index += 1
                    else:
                        quoted = not quoted
                        index += 1
                else:
                    part.extend("\\" * backslashes)
                continue
            if character == '"':
                if quoted and index + 1 < len(command) and command[index + 1] == '"':
                    part.append('"')
                    index += 2
                else:
                    quoted = not quoted
                    index += 1
                continue
            part.append(character)
            index += 1
        if quoted:
            raise ValueError("unclosed quotation mark")
        parts.append("".join(part))
    return tuple(parts)


def parse_verify_command(command: str, *, windows: bool | None = None) -> tuple[str, ...]:
    """Split a shell-like command string without ever invoking a shell."""
    if not isinstance(command, str) or not command.strip():
        raise _ConfigError("verification command is required")
    try:
        use_windows_rules = os.name == "nt" if windows is None else windows
        parts = (
            _split_windows_command(command) if use_windows_rules else tuple(shlex.split(command))
        )
    except ValueError as exc:
        raise _ConfigError("verification command has invalid quoting") from exc
    if not parts:
        raise _ConfigError("verification command is required")
    return parts


def build_agent(
    config: CliConfig,
    *,
    journal: ChangeJournal | None = None,
    checkpoint: CheckpointSession | None = None,
    input_fn: Callable[[str], object] = input,
    output_fn: Callable[[str], object] = print,
) -> AgentRunner:
    """Assemble independent repair, read-only review, and memory Agents."""
    command_runner = CommandRunner(config.workspace)
    verifier = Verifier(command_runner, config.verifier)
    try:
        trace_pattern = (
            config.trace_path.resolve(strict=False)
            .relative_to(command_runner.workspace_root)
            .as_posix()
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("trace path must stay inside the workspace") from exc
    protected_patterns = tuple(
        dict.fromkeys((*DEFAULT_PROTECTED_PATTERNS, *verifier.protected_patterns, trace_pattern))
    )
    if journal is None:
        journal = checkpoint.journal if checkpoint is not None else ChangeJournal(config.workspace)
    if checkpoint is not None and checkpoint.journal is not journal:
        raise ValueError("checkpoint and runtime must share one change journal")
    workspace = Workspace(
        config.workspace,
        protected_patterns=protected_patterns,
        private_patterns=DEFAULT_PRIVATE_PATTERNS,
        change_recorder=journal,
    )
    registry = ToolRegistry()
    for tool in (
        ListFilesTool(workspace),
        ReadFileTool(workspace),
        SearchTextTool(workspace),
        EditFileTool(workspace),
        WriteFileTool(workspace),
        RunCommandTool(command_runner),
        FinishTool(),
    ):
        registry.register(tool)
    repair_model = OpenAIChatModel(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    review_model = OpenAIChatModel(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    memory_model = OpenAIChatModel(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    reviewer = ReviewerAgent(review_model, build_reviewer_registry(workspace))
    memory_agent = MemoryAgent(memory_model, build_memory_registry())
    return AgentRunner(
        repair_model,
        registry,
        verifier,
        trace=JsonlTrace(config.trace_path),
        approval=ConsoleApprovalWorkflow(journal, input_fn=input_fn, output_fn=output_fn),
        reviewer=reviewer,
        checkpoint=checkpoint,
        memory_store=MemoryStore(config.workspace),
        memory_agent=memory_agent,
        max_iterations=config.max_iterations,
        protected_patterns=protected_patterns,
    )


def _workspace_path(value: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise _ConfigError("workspace must be an existing directory") from None
    if not path.is_dir():
        raise _ConfigError("workspace must be an existing directory")
    return path


def _trace_path(workspace: Path, supplied: str | None) -> Path:
    attempts = 8 if supplied is None else 1
    for _ in range(attempts):
        try:
            if supplied is None:
                timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                candidate = (
                    workspace
                    / ".testpilot"
                    / "traces"
                    / f"run-{timestamp}-{secrets.token_hex(4)}.jsonl"
                )
            else:
                raw = Path(supplied).expanduser()
                candidate = raw if raw.is_absolute() else workspace / raw
            if candidate.suffix.lower() != ".jsonl":
                raise _ConfigError("trace path must be a new .jsonl file inside the workspace")
            if candidate.exists() or candidate.is_symlink():
                raise FileExistsError

            resolved = candidate.resolve(strict=False)
            resolved.relative_to(workspace)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            canonical_parent = resolved.parent.resolve(strict=True)
            canonical_parent.relative_to(workspace)
            reserved = canonical_parent / resolved.name
            with reserved.open("x", encoding="utf-8", newline="\n"):
                pass
            return reserved
        except FileExistsError:
            if supplied is None:
                continue
            raise _ConfigError(
                "trace path must be a new .jsonl file inside the workspace"
            ) from None
        except _ConfigError:
            raise
        except (OSError, RuntimeError, ValueError):
            raise _ConfigError(
                "trace path must be a new .jsonl file inside the workspace"
            ) from None
    raise _ConfigError("could not reserve a unique trace path inside the workspace")


def _environment_configuration(
    workspace: Path, verify: str, task: str, trace: str | None, max_iterations: int
) -> CliConfig:
    normalized_task = task.strip()
    if not normalized_task:
        raise _ConfigError("task must contain non-whitespace text")
    key, model, base_url = _environment_values()
    runner = CommandRunner(workspace)
    parsed_verify = parse_verify_command(verify)
    canonical_verify = runner.canonical_model_command(parsed_verify)
    if canonical_verify is None:
        raise _ConfigError(
            "verification command must be a supported workspace-confined pytest command"
        )
    return CliConfig(
        workspace=workspace,
        verifier=canonical_verify,
        task=normalized_task,
        api_key=key,
        model=model,
        base_url=base_url,
        trace_path=_trace_path(workspace, trace),
        max_iterations=max_iterations,
    )


def _fresh_setup(
    workspace: Path,
    *,
    verify: str | None,
    task: str | None,
    trace: str | None,
    max_iterations: int | None,
    output_fn: Callable[[str], object] = print,
) -> RunSetup:
    """Validate and reserve every host-owned object for a new run."""
    if task is None or verify is None:
        raise _ConfigError("fresh mode requires both task and verification command")
    budget = _invocation_budget(max_iterations, default=12)
    config = _environment_configuration(workspace, verify, task, trace, budget)
    journal = ChangeJournal(workspace)
    checkpoint = CheckpointSession.create(
        store=CheckpointStore(workspace),
        journal=journal,
        request=CheckpointRequest(
            task=config.task,
            verifier=config.verifier,
            max_iterations=config.max_iterations,
            trace_path=config.trace_path.relative_to(workspace).as_posix(),
        ),
        on_ready=_checkpoint_ready_callback(output_fn),
    )
    return RunSetup(config, journal, checkpoint, None)


def _resume_setup(
    workspace: Path,
    *,
    run_id: str,
    verify: str | None = None,
    task: str | None = None,
    trace: str | None = None,
    max_iterations: int | None = None,
    output_fn: Callable[[str], object] = print,
) -> RunSetup:
    """Restore and revalidate a run completely before model construction."""
    if verify is not None or task is not None or trace is not None:
        raise _ConfigError("resume mode does not accept task, verification, or trace")
    _invocation_budget(max_iterations, default=1)
    started_ns = monotonic_ns()
    journal = ChangeJournal(workspace)
    checkpoint, resume = CheckpointSession.restore(
        store=CheckpointStore(workspace),
        journal=journal,
        run_id=run_id,
        on_ready=_checkpoint_ready_callback(output_fn),
    )

    command_runner = CommandRunner(workspace)
    canonical_verify = command_runner.canonical_model_command(
        checkpoint.request.verifier
    )
    if canonical_verify is None:
        raise CheckpointError("checkpoint_invalid")
    trace_path = _existing_trace_path(workspace, checkpoint.request.trace_path)
    key, model, base_url = _environment_values()
    budget = (
        checkpoint.request.max_iterations
        if max_iterations is None
        else max_iterations
    )
    config = CliConfig(
        workspace=workspace,
        verifier=canonical_verify,
        task=checkpoint.request.task,
        api_key=key,
        model=model,
        base_url=base_url,
        trace_path=trace_path,
        max_iterations=budget,
    )
    try:
        JsonlTrace(trace_path).record(
            "checkpoint_restore",
            {
                "run_id": checkpoint.run_id,
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "safe_point": checkpoint.safe_point,
                "ok": True,
                "error_code": None,
                "duration_ms": _elapsed_ms(started_ns),
            },
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    return RunSetup(config, journal, checkpoint, resume)


def _environment_values() -> tuple[str, str, str | None]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not key or not model:
        raise _ConfigError("OPENAI_API_KEY and OPENAI_MODEL must both be non-empty")
    return key, model, os.environ.get("OPENAI_BASE_URL", "").strip() or None


def _invocation_budget(value: int | None, *, default: int) -> int:
    budget = default if value is None else value
    if type(budget) is not int or budget < 1:
        raise _ConfigError("max iterations must be a positive integer")
    return budget


def _checkpoint_ready_callback(
    output_fn: Callable[[str], object],
) -> Callable[[str, Path], None]:
    def announce(run_id: str, path: Path) -> None:
        del path
        output_fn(f"run_id={run_id}")
        output_fn(f"checkpoint=.testpilot/checkpoints/{run_id}.json")

    return announce


def _existing_trace_path(workspace: Path, relative: str) -> Path:
    try:
        candidate = workspace.joinpath(*PurePosixPath(relative).parts)
        if candidate.suffix.lower() != ".jsonl" or candidate.is_symlink():
            raise OSError
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(workspace)
        if resolved != candidate:
            raise OSError
        status = candidate.stat(follow_symlinks=False)
        if not stat.S_ISREG(status.st_mode):
            raise OSError
        return candidate
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint_invalid") from exc


def _elapsed_ms(started_ns: int) -> float:
    return round((monotonic_ns() - started_ns) / 1_000_000, 3)


def _print_result(result: object, trace_path: Path) -> int:
    """Print the only user-facing run summary; never include model or tool content."""
    success = bool(getattr(result, "success", False))
    state = getattr(result, "state", None)
    changed_files: Any = getattr(state, "changed_files", set())
    if not isinstance(changed_files, set):
        changed_files = set()
    stop_reason = getattr(result, "stop_reason", None)
    exit_code = getattr(state, "last_verify_exit_code", None)
    review_status = getattr(state, "review_status", None)
    review = (
        review_status
        if isinstance(review_status, str)
        and review_status in {"passed", "changes_requested", "unavailable"}
        else "-"
    )
    raw_review_rounds = getattr(state, "review_rounds", 0)
    review_rounds = (
        raw_review_rounds
        if type(raw_review_rounds) is int and raw_review_rounds >= 0
        else 0
    )
    raw_review_reworks = getattr(state, "review_rework_count", 0)
    review_reworks = (
        raw_review_reworks
        if type(raw_review_reworks) is int and raw_review_reworks in {0, 1}
        else 0
    )
    approval_status = getattr(state, "approval_status", None)
    approval = (
        approval_status
        if isinstance(approval_status, str)
        and approval_status in {"approved", "rejected", "unavailable"}
        else "-"
    )
    run_id = getattr(result, "run_id", None)
    resume_available = getattr(result, "resume_available", False) is True
    warning = getattr(result, "checkpoint_warning", None)
    raw_memories_retrieved = getattr(result, "memories_retrieved", 0)
    memories_retrieved = (
        raw_memories_retrieved
        if type(raw_memories_retrieved) is int and 0 <= raw_memories_retrieved <= 3
        else 0
    )
    raw_memory_saved = getattr(result, "memory_saved", "no")
    memory_saved = (
        raw_memory_saved
        if isinstance(raw_memory_saved, str)
        and raw_memory_saved in {"yes", "no", "duplicate"}
        else "no"
    )
    raw_memory_warning = getattr(result, "memory_warning", None)
    memory_warning = (
        raw_memory_warning
        if isinstance(raw_memory_warning, str)
        and raw_memory_warning in _MEMORY_WARNING_CODES
        else "-"
    )
    print(f"STATUS={'SUCCESS' if success else 'FAILED'}")
    print(f"stop_reason={stop_reason or 'unknown'}")
    rendered_changes = json.dumps(
        sorted(item for item in changed_files if isinstance(item, str)),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    print(f"changed_files={rendered_changes}")
    print(f"verification_exit={exit_code if type(exit_code) is int else '-'}")
    print(f"review={review}")
    print(f"review_rounds={review_rounds}")
    print(f"review_reworks={review_reworks}")
    print(f"approval={approval}")
    print(f"run_id={run_id if isinstance(run_id, str) else '-'}")
    print(f"resume_available={'yes' if resume_available else 'no'}")
    print(
        "checkpoint_warning="
        + (warning if warning == "checkpoint_cleanup_failed" else "-")
    )
    print(f"memories_retrieved={memories_retrieved}")
    print(f"memory_saved={memory_saved}")
    print(f"memory_warning={memory_warning}")
    print(f"trace={trace_path}")
    return 0 if success else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning a process status instead of leaking exceptions."""
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        workspace = _workspace_path(arguments.workspace)
        if arguments.resume is None:
            setup = _fresh_setup(
                workspace,
                verify=arguments.verify,
                task=arguments.task,
                trace=arguments.trace,
                max_iterations=arguments.max_iterations,
            )
        else:
            setup = _resume_setup(
                workspace,
                run_id=arguments.resume,
                verify=arguments.verify,
                task=arguments.task,
                trace=arguments.trace,
                max_iterations=arguments.max_iterations,
            )
        agent = build_agent(
            config=setup.config,
            journal=setup.journal,
            checkpoint=setup.checkpoint,
        )
        if setup.resume is None:
            result = agent.run(setup.config.task)
        else:
            result = agent.run(setup.config.task, resume=setup.resume)
    except _ConfigError:
        print("STATUS=CONFIG_ERROR")
        return 2
    except CheckpointError as error:
        code = (
            error.code
            if isinstance(error.code, str) and error.code in _CHECKPOINT_ERROR_CODES
            else "checkpoint_invalid"
        )
        print("STATUS=FAILED")
        print(f"stop_reason={code}")
        return 1
    except Exception:  # noqa: BLE001 - public boundary must never disclose SDK credential details.
        # Do not echo exception details: SDKs and URLs can include credentials.
        print("STATUS=FAILED")
        print("stop_reason=runtime_setup_failed")
        return 1
    return _print_result(result, setup.config.trace_path)


if __name__ == "__main__":  # pragma: no cover - exercised through __main__ module
    raise SystemExit(main())
