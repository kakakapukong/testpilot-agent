"""Small, deliberately explicit command-line entry point for TestPilot."""

from __future__ import annotations

import argparse
import os
import secrets
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent import AgentRunner
from .command import CommandRunner, FinishTool, RunCommandTool, Verifier
from .model import OpenAIChatModel
from .registry import ToolRegistry
from .tools import EditFileTool, ListFilesTool, ReadFileTool, SearchTextTool, WriteFileTool
from .trace import JsonlTrace
from .workspace import DEFAULT_PROTECTED_PATTERNS, Workspace


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
        description="A verification-gated single coding agent for a small Python workspace.",
    )
    parser.add_argument(
        "--workspace", required=True, metavar="PATH", help="Existing repository directory."
    )
    parser.add_argument(
        "--verify",
        required=True,
        metavar="COMMAND",
        help="Fixed host-owned restricted pytest command, e.g. 'python -m pytest -q'.",
    )
    parser.add_argument(
        "--trace",
        metavar="PATH",
        help="Optional JSONL trace path, relative to the workspace unless absolute.",
    )
    parser.add_argument(
        "--max-iterations",
        type=positive_integer,
        default=12,
        help="Maximum model/tool rounds (default: 12).",
    )
    parser.add_argument("task", metavar="TASK", help="Natural-language repair task.")
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


def build_agent(config: CliConfig) -> AgentRunner:
    """Assemble the fixed seven-tool local runtime after validation succeeds."""
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
    workspace = Workspace(config.workspace, protected_patterns=protected_patterns)
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
    return AgentRunner(
        OpenAIChatModel(model=config.model, api_key=config.api_key, base_url=config.base_url),
        registry,
        verifier,
        trace=JsonlTrace(config.trace_path),
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
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not key or not model:
        raise _ConfigError("OPENAI_API_KEY and OPENAI_MODEL must both be non-empty")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
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


def _print_result(result: object, trace_path: Path) -> int:
    """Print the only user-facing run summary; never include model or tool content."""
    success = bool(getattr(result, "success", False))
    state = getattr(result, "state", None)
    changed_files: Any = getattr(state, "changed_files", set())
    if not isinstance(changed_files, set):
        changed_files = set()
    stop_reason = getattr(result, "stop_reason", None)
    exit_code = getattr(state, "last_verify_exit_code", None)
    print(f"STATUS={'SUCCESS' if success else 'FAILED'}")
    print(f"stop_reason={stop_reason or 'unknown'}")
    print(f"changed_files={','.join(sorted(str(item) for item in changed_files)) or '-'}")
    print(f"verification_exit={exit_code if type(exit_code) is int else '-'}")
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
        config = _environment_configuration(
            workspace, arguments.verify, arguments.task, arguments.trace, arguments.max_iterations
        )
        agent = build_agent(config=config)
        result = agent.run(config.task)
    except _ConfigError:
        print("STATUS=CONFIG_ERROR")
        return 2
    except Exception:  # noqa: BLE001 - public boundary must never disclose SDK credential details.
        # Do not echo exception details: SDKs and URLs can include credentials.
        print("STATUS=FAILED")
        print("stop_reason=runtime_setup_failed")
        return 1
    return _print_result(result, config.trace_path)


if __name__ == "__main__":  # pragma: no cover - exercised through __main__ module
    raise SystemExit(main())
