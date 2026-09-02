"""Core value types for TestPilot agent runs."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any


class RunPhase(str, Enum):
    """The current phase of an agent run."""

    DISCOVER = "discover"
    EDIT = "edit"
    VERIFY = "verify"
    REVIEW = "review"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolCall:
    """A tool request made by an assistant turn."""

    id: str
    name: str
    arguments: Mapping[str, Any]
    argument_error: str | None = None

    def __post_init__(self) -> None:
        """Defensively freeze a copy of the supplied arguments."""
        object.__setattr__(self, "arguments", _freeze_argument(self.arguments))

    def arguments_dict(self) -> dict[str, Any]:
        """Return a defensive, JSON-native copy of the tool arguments."""
        return {key: _thaw_argument(value) for key, value in self.arguments.items()}


@dataclass(frozen=True)
class AssistantTurn:
    """An assistant response and its requested tool calls."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolResult:
    """A normalized result returned by a tool execution."""

    ok: bool
    data: Any = None
    error: str | None = None
    error_code: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        """Ensure a result has a state consistent with its outcome."""
        if type(self.ok) is not bool:
            raise TypeError("ok must be a boolean")
        if type(self.timed_out) is not bool or type(self.truncated) is not bool:
            raise TypeError("timed_out and truncated must be booleans")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise TypeError("exit_code must be an integer or None")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("stdout and stderr must be strings")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if self.error_code is not None and not isinstance(self.error_code, str):
            raise TypeError("error_code must be a string or None")
        if self.ok and (self.error is not None or self.error_code is not None):
            raise ValueError("successful tool results cannot contain error details")
        if self.ok and self.exit_code not in (None, 0):
            raise ValueError("successful tool results cannot have a non-zero exit code")
        if self.ok and self.timed_out:
            raise ValueError("successful tool results cannot be timed out")
        if not self.ok and (not self.error or not self.error_code):
            raise ValueError("failed tool results require non-empty error and error_code")

    @classmethod
    def success(
        cls,
        data: Any = None,
        *,
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
        truncated: bool = False,
    ) -> "ToolResult":
        """Create a successful tool result."""
        return cls(
            ok=True,
            data=data,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            truncated=truncated,
        )

    @classmethod
    def failure(
        cls,
        error: str,
        code: str,
        *,
        data: Any = None,
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
        truncated: bool = False,
    ) -> "ToolResult":
        """Create a failed tool result."""
        return cls(
            ok=False,
            data=data,
            error=error,
            error_code=code,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            truncated=truncated,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result using the stable tool-result shape."""
        return {
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
        }


@dataclass
class RunState:
    """Mutable accounting state for a single agent run."""

    phase: RunPhase = RunPhase.DISCOVER
    iteration: int = 0
    edit_count: int = 0
    source_edit_count: int = 0
    changed_files: set[str] = field(default_factory=set)
    last_verify_exit_code: int | None = None
    verified_after_last_edit: bool = False
    consecutive_no_progress: int = 0
    stop_reason: str | None = None
    approval_status: str | None = None
    review_status: str | None = None
    review_rounds: int = 0
    review_rework_count: int = 0
    reviewed_edit_count: int | None = None
    reviewed_source_edit_count: int | None = None

    def record_edit(self, path: str) -> None:
        """Record an edit and invalidate any prior verification."""
        self.phase = RunPhase.EDIT
        self.edit_count += 1
        if Path(path).suffix.lower() in {".py", ".pyi"}:
            self.source_edit_count += 1
        self.changed_files.add(path)
        self.verified_after_last_edit = False

    def record_verification(self, exit_code: int, *, passed: bool | None = None) -> None:
        """Record verification performed after the currently tracked edits."""
        if type(exit_code) is not int:
            raise ValueError("exit_code must be an integer")
        if passed is not None and type(passed) is not bool:
            raise ValueError("passed must be a boolean or None")
        self.phase = RunPhase.VERIFY
        self.last_verify_exit_code = exit_code
        self.verified_after_last_edit = exit_code == 0 if passed is None else passed

    def record_review(self, status: str) -> None:
        """Record a stable review outcome for the currently verified edits."""
        if not isinstance(status, str) or status not in {
            "passed",
            "changes_requested",
            "unavailable",
        }:
            raise ValueError("invalid review status")
        self.phase = RunPhase.REVIEW
        self.review_status = status
        self.review_rounds += 1
        self.reviewed_edit_count = self.edit_count
        self.reviewed_source_edit_count = self.source_edit_count
        if status != "passed":
            self.verified_after_last_edit = False

    def record_approval(self, status: str) -> None:
        """Record one of the stable human-approval outcomes."""
        if not isinstance(status, str) or status not in {
            "approved",
            "rejected",
            "unavailable",
        }:
            raise ValueError("invalid approval status")
        self.approval_status = status

    def invalidate_for_resume(self) -> None:
        """Discard stale success evidence while preserving cumulative history."""
        self.verified_after_last_edit = False
        self.approval_status = None
        self.stop_reason = None
        if self.review_status != "changes_requested":
            self.review_status = None
            self.reviewed_edit_count = None
            self.reviewed_source_edit_count = None
        self.phase = RunPhase.EDIT if self.changed_files else RunPhase.DISCOVER


@dataclass(frozen=True)
class AgentRunResult:
    """The final outcome of an agent run with immutable top-level fields."""

    success: bool
    final_text: str
    stop_reason: str | None
    state: RunState
    messages: tuple[Any, ...]
    trace_path: Path | None = None
    run_id: str | None = None
    checkpoint_path: Path | None = None
    resume_available: bool = False
    checkpoint_warning: str | None = None
    memories_retrieved: int = 0
    memory_saved: str = "no"
    memory_warning: str | None = None


def _freeze_argument(value: Any) -> Any:
    """Recursively copy and freeze supported tool-argument containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_argument(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_argument(item) for item in value)
    return value


def _thaw_argument(value: Any) -> Any:
    """Recursively copy frozen tool arguments into JSON-native containers."""
    if isinstance(value, Mapping):
        return {key: _thaw_argument(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_argument(item) for item in value]
    return value
