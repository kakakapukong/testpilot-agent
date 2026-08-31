"""Bounded local command execution and independent verification."""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import Any

from .types import ToolResult

_PYTEST_LAUNCHER_NAMES = frozenset({"pytest", "pytest.exe"})
_BATCH_WRAPPER_SUFFIXES = frozenset({".cmd", ".bat"})
_SENSITIVE_ENV_PARTS = ("API_KEY", "_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_BLOCKED_PROCESS_ENV_NAMES = frozenset(
    {"PYTHONHOME", "PYTHONPATH", "PYTHONSAFEPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
)
_MODEL_PYTEST_FLAGS = frozenset(
    {
        "--cache-clear",
        "--co",
        "--collect-only",
        "--disable-warnings",
        "--exitfirst",
        "--fixtures",
        "--fixtures-per-test",
        "--help",
        "--lf",
        "--nf",
        "--quiet",
        "--setup-only",
        "--setup-plan",
        "--setup-show",
        "--strict-config",
        "--strict-markers",
        "--verbose",
        "--version",
        "-x",
    }
)
_MODEL_PYTEST_VALUE_OPTIONS = frozenset(
    {
        "--capture",
        "--code-highlight",
        "--color",
        "--durations",
        "--durations-min",
        "--maxfail",
        "--show-capture",
        "--tb",
        "-k",
        "-m",
    }
)


class CommandRunner:
    """Run a small, explicit set of safe test commands in one workspace."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        default_timeout: float = 30.0,
        max_timeout: float = 120.0,
        output_limit: int = 20_000,
    ) -> None:
        if (
            not math.isfinite(default_timeout)
            or not math.isfinite(max_timeout)
            or default_timeout <= 0
            or max_timeout <= 0
            or default_timeout > max_timeout
        ):
            raise ValueError("timeouts must be positive and default_timeout <= max_timeout")
        if output_limit < 1:
            raise ValueError("output_limit must be positive")
        self.workspace_root = Path(workspace_root).expanduser().resolve(strict=False)
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.output_limit = output_limit
        self._python_executable = Path(sys.executable).resolve()
        self._pytest_launchers = self._find_pytest_launchers()

    def run(self, argv: Sequence[str], *, timeout_seconds: float | None = None) -> ToolResult:
        """Execute an allowlisted argv command without a shell or caller-set cwd."""
        error = self._validate_workspace()
        if error is not None:
            return error
        normalized = self._normalise_argv(argv)
        if normalized is None:
            return ToolResult.failure(
                "command argv must be a non-empty array of strings", "invalid_arguments"
            )
        execution_argv = self.canonical_command(normalized)
        if execution_argv is None:
            return ToolResult.failure(
                f"command is not allowed: {normalized[0]!r}", "command_not_allowed"
            )
        timeout = self._normalise_timeout(timeout_seconds)
        if timeout is None:
            return ToolResult.failure(
                "timeout_seconds must be a positive number within the configured limit",
                "invalid_arguments",
            )

        try:
            return_code, stdout, stderr, timed_out, truncated = self._run_bounded_process(
                execution_argv, timeout
            )
        except (OSError, TypeError, ValueError) as exc:
            return ToolResult.failure(f"could not start command: {exc}", "command_start_failed")
        if timed_out:
            return ToolResult.failure(
                "command timed out",
                "timeout",
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                truncated=truncated,
            )
        if return_code != 0:
            return ToolResult.failure(
                f"command exited with status {return_code}",
                "command_failed",
                exit_code=return_code,
                stdout=stdout,
                stderr=stderr,
                truncated=truncated,
            )
        return ToolResult.success(
            {"argv": list(execution_argv)},
            exit_code=0,
            stdout=stdout,
            stderr=stderr,
            truncated=truncated,
        )

    def _validate_workspace(self) -> ToolResult | None:
        if not self.workspace_root.exists() or not self.workspace_root.is_dir():
            return ToolResult.failure(
                f"workspace is not an existing directory: {self.workspace_root}",
                "invalid_workspace",
            )
        return None

    @staticmethod
    def _normalise_argv(argv: Sequence[str]) -> tuple[str, ...] | None:
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
            return None
        if not all(isinstance(part, str) and part and "\x00" not in part for part in argv):
            return None
        return tuple(argv)

    def _normalise_timeout(self, timeout_seconds: float | None) -> float | None:
        timeout = self.default_timeout if timeout_seconds is None else timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            return None
        timeout_float = float(timeout)
        if not 0 < timeout_float <= self.max_timeout:
            return None
        return timeout_float

    def _find_pytest_launchers(self) -> frozenset[Path]:
        launchers: set[Path] = set()
        for name in _PYTEST_LAUNCHER_NAMES:
            resolved = _resolve_from_path(name)
            if resolved is not None and _is_safe_pytest_launcher_path(resolved):
                launchers.add(resolved)
        return frozenset(launchers)

    def canonical_command(self, argv: Sequence[str]) -> tuple[str, ...] | None:
        """Return an executable-pinned command or ``None`` without spawning a process."""
        normalized = self._normalise_argv(argv)
        if normalized is None:
            return None
        executable = self._trusted_executable(normalized[0])
        if executable is None:
            return None
        return (str(executable), *normalized[1:])

    def canonical_model_command(self, argv: Sequence[str]) -> tuple[str, ...] | None:
        """Return a canonical model-safe pytest command, never a general Python command."""
        canonical = self.canonical_command(argv)
        if canonical is None:
            return None
        executable = Path(canonical[0])
        if executable == self._python_executable:
            if len(canonical) >= 3 and canonical[1:3] == ("-m", "pytest"):
                return (
                    canonical
                    if _model_pytest_arguments_are_safe(self.workspace_root, canonical[3:])
                    else None
                )
            return None
        if executable in self._pytest_launchers:
            return (
                canonical
                if _model_pytest_arguments_are_safe(self.workspace_root, canonical[1:])
                else None
            )
        return None

    def _trusted_executable(self, program: str) -> Path | None:
        """Resolve an allowed program once, then use that exact path for execution."""
        if Path(program).suffix.lower() in _BATCH_WRAPPER_SUFFIXES:
            return None
        candidate = _resolved_program(program)
        if candidate is None:
            return None
        if candidate == self._python_executable:
            return candidate
        if (
            _is_safe_pytest_launcher_path(candidate)
            and candidate in self._pytest_launchers
            and not _is_within(candidate, self.workspace_root)
        ):
            return candidate
        return None

    def _run_bounded_process(
        self, execution_argv: tuple[str, ...], timeout: float
    ) -> tuple[int, str, str, bool, bool]:
        """Run one command with a fresh pyc cache, cleaning it on every exit path."""
        with TemporaryDirectory(prefix="testpilot-pyc-") as pyc_prefix:
            environment = _safe_environment()
            environment["PYTHONPYCACHEPREFIX"] = pyc_prefix
            return self._run_bounded_process_with_env(execution_argv, timeout, environment)

    def _run_bounded_process_with_env(
        self, execution_argv: tuple[str, ...], timeout: float, environment: Mapping[str, str]
    ) -> tuple[int, str, str, bool, bool]:
        """Run a process while draining both pipes without retaining unbounded output."""
        popen_kwargs: dict[str, Any] = {
            "cwd": self.workspace_root,
            "env": environment,
            "shell": False,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(execution_argv, **popen_kwargs)
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_capture = _StreamCapture(self.output_limit)
        stderr_capture = _StreamCapture(self.output_limit)
        stdout_thread = Thread(
            target=_drain_stream, args=(process.stdout, stdout_capture), daemon=True
        )
        stderr_thread = Thread(
            target=_drain_stream, args=(process.stderr, stderr_capture), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            _join_drainers((stdout_thread, stderr_thread), (process.stdout, process.stderr))
        return (
            int(process.returncode if process.returncode is not None else -1),
            stdout_capture.text(),
            stderr_capture.text(),
            timed_out,
            stdout_capture.truncated or stderr_capture.truncated,
        )


class RunCommandTool:
    """Expose the bounded runner as a model-callable tool."""

    name = "run_command"
    description = "Run a restricted pytest command against workspace-only targets."
    parameters: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "argv": {"type": "array", "items": {"type": "string"}},
            "timeout_seconds": {"type": "integer", "minimum": 1},
        },
        "required": ["argv"],
        "additionalProperties": False,
    }

    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        argv = arguments.get("argv")
        if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
            return ToolResult.failure("argv must be an array of strings", "invalid_arguments")
        timeout = arguments.get("timeout_seconds")
        if timeout is not None and (not isinstance(timeout, int) or isinstance(timeout, bool)):
            return ToolResult.failure("timeout_seconds must be an integer", "invalid_arguments")
        canonical = self.runner.canonical_model_command(argv)
        if canonical is None:
            return ToolResult.failure("only pytest commands are allowed", "command_not_allowed")
        return self.runner.run(canonical, timeout_seconds=timeout)


class FinishTool:
    """Record a model request to finish; the agent performs verification itself."""

    name = "finish"
    description = "Request independent verification after summarising the completed work."
    parameters: Mapping[str, Any] = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        summary = arguments.get("summary")
        if not isinstance(summary, str):
            return ToolResult.failure("summary must be a string", "invalid_arguments")
        return ToolResult.success({"finish_requested": True, "summary": summary})


@dataclass(frozen=True, init=False)
class Verifier:
    """Immutable, workspace-confined pytest command owned by the host."""

    runner: CommandRunner
    command: tuple[str, ...]
    protected_patterns: tuple[str, ...]

    def __init__(self, runner: CommandRunner, command: Sequence[str]) -> None:
        normalized = CommandRunner._normalise_argv(command)
        if normalized is None:
            raise ValueError("verification command must be a non-empty array of strings")
        canonical = runner.canonical_model_command(normalized)
        if canonical is None:
            raise ValueError("verification command must be a supported pytest command")
        object.__setattr__(self, "runner", runner)
        object.__setattr__(self, "command", canonical)
        object.__setattr__(
            self,
            "protected_patterns",
            _verifier_protected_patterns(runner, canonical),
        )

    def verify(self) -> ToolResult:
        """Run exactly the command captured at construction time."""
        return self.runner.run(self.command)


def _verifier_protected_patterns(runner: CommandRunner, command: Sequence[str]) -> tuple[str, ...]:
    """Derive direct workspace assets selected by an immutable verifier command."""
    canonical = runner.canonical_model_command(command)
    if canonical is None:
        return ()
    executable = Path(canonical[0])
    if executable in runner._pytest_launchers:
        return _pytest_asset_patterns(runner, canonical[1:])
    if executable == runner._python_executable and canonical[1:3] == ("-m", "pytest"):
        return _pytest_asset_patterns(runner, canonical[3:])
    return ()


def _pytest_asset_patterns(runner: CommandRunner, arguments: Sequence[str]) -> tuple[str, ...]:
    patterns: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in _MODEL_PYTEST_VALUE_OPTIONS:
            index += 2
            continue
        if argument.startswith("-"):
            index += 1
            continue
        _append_asset_pattern(patterns, runner, argument.split("::", 1)[0])
        index += 1
    return tuple(dict.fromkeys(patterns))


def _append_asset_pattern(patterns: list[str], runner: CommandRunner, raw: str) -> None:
    pattern = _workspace_asset_pattern(runner, raw)
    if pattern is not None:
        patterns.append(pattern)


def _workspace_asset_pattern(runner: CommandRunner, raw: str) -> str | None:
    if not raw or "\x00" in raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = runner.workspace_root / candidate
    try:
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(runner.workspace_root).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None
    if not relative or relative == ".":
        return None
    return f"{relative}/**" if resolved.is_dir() else relative


def _resolve_from_path(program: str) -> Path | None:
    found = shutil.which(program)
    if found is None:
        return None
    try:
        return Path(found).resolve(strict=True)
    except OSError:
        return None


def _resolved_program(program: str) -> Path | None:
    path = Path(program)
    try:
        if path.is_absolute() or path.parent != Path("."):
            return path.resolve(strict=True)
    except OSError:
        return None
    return _resolve_from_path(program)


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _is_safe_pytest_launcher_path(candidate: Path) -> bool:
    """Reject Windows batch wrappers even when a bare PATH lookup found one."""
    return candidate.suffix.lower() in {"", ".exe"}


def _model_pytest_arguments_are_safe(root: Path, arguments: Sequence[str]) -> bool:
    """Allow read-only pytest selection/reporting options and workspace targets only."""
    index = 0
    positional_only = False
    while index < len(arguments):
        argument = arguments[index]
        if not positional_only and argument == "--":
            positional_only = True
            index += 1
            continue
        if not positional_only and argument in _MODEL_PYTEST_FLAGS:
            index += 1
            continue
        if not positional_only and _is_safe_short_pytest_cluster(argument):
            index += 1
            continue
        if not positional_only and argument in _MODEL_PYTEST_VALUE_OPTIONS:
            if index + 1 >= len(arguments):
                return False
            index += 2
            continue
        if not positional_only and any(
            argument.startswith(f"{option}=") for option in _MODEL_PYTEST_VALUE_OPTIONS
        ):
            index += 1
            continue
        if not positional_only and argument.startswith("-"):
            return False
        if not _is_workspace_pytest_target(root, argument):
            return False
        index += 1
    return True


def _is_safe_short_pytest_cluster(argument: str) -> bool:
    return len(argument) > 1 and argument.startswith("-") and set(argument[1:]) <= set("qvsx")


def _is_workspace_pytest_target(root: Path, raw: str) -> bool:
    target = raw.split("::", 1)[0]
    if not target or target.startswith("@"):
        return False
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return _is_within(resolved, root)


def _safe_environment() -> dict[str, str]:
    """Retain ordinary process setup but never pass likely credentials to commands."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _BLOCKED_PROCESS_ENV_NAMES
        and not any(marker in key.upper() for marker in _SENSITIVE_ENV_PARTS)
    }
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


@dataclass
class _StreamCapture:
    limit: int
    parts: list[str] = field(default_factory=list)
    size: int = 0
    truncated: bool = False

    def append(self, chunk: str) -> None:
        remaining = self.limit - self.size
        if remaining > 0:
            kept = chunk[:remaining]
            self.parts.append(kept)
            self.size += len(kept)
        if len(chunk) > max(remaining, 0):
            self.truncated = True

    def text(self) -> str:
        if not self.truncated:
            return "".join(self.parts)
        marker = "...[truncated]"
        if self.limit <= len(marker):
            return marker[: self.limit]
        raw = "".join(self.parts)
        return raw[: self.limit - len(marker)] + marker


def _drain_stream(stream: Any, capture: _StreamCapture) -> None:
    try:
        while chunk := stream.read(4096):
            capture.append(chunk)
    except (OSError, ValueError):
        return
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _join_drainers(threads: tuple[Thread, Thread], streams: tuple[Any, Any]) -> None:
    for thread in threads:
        thread.join(timeout=5)
    for thread, stream in zip(threads, streams):
        if thread.is_alive():
            try:
                stream.close()
            except OSError:
                pass
            thread.join(timeout=1)


def _terminate_process_tree(process: Any) -> None:
    if os.name == "nt":
        try:
            taskkill = (
                Path(os.environ.get("SystemRoot", r"C:\\Windows")) / "System32" / "taskkill.exe"
            )
            subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            # These APIs are intentionally discovered at runtime because they are
            # absent from both Windows and its platform-specific type stubs.
            getpgid = getattr(os, "getpgid")  # noqa: B009
            killpg = getattr(os, "killpg")  # noqa: B009
            sigkill = getattr(signal, "SIGKILL")  # noqa: B009
            killpg(getpgid(process.pid), sigkill)
        except (OSError, ProcessLookupError):
            pass
    try:
        process.kill()
    except OSError:
        pass
