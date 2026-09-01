"""The verification-gated, single-agent tool loop."""

from __future__ import annotations

import json
import os
import posixpath
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic_ns
from typing import Any, Protocol

from .command import Verifier
from .context import BoundedContext
from .model import ModelClient, ModelError
from .registry import ToolRegistry
from .trace import JsonlTrace
from .types import AgentRunResult, AssistantTurn, RunPhase, RunState, ToolCall, ToolResult
from .workspace import DEFAULT_PROTECTED_PATTERNS, is_protected_relative_path


class _Verifier(Protocol):
    def verify(self) -> ToolResult: ...


class _Trace(Protocol):
    path: Path

    def record(self, event: str, payload: Mapping[str, Any] | None = None) -> None: ...


class _ApprovalWorkflow(Protocol):
    def request(
        self,
        *,
        changed_files: Sequence[str],
        verification_exit_code: int,
    ) -> bool: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class AgentRunner:
    """Run one model-driven repair loop and gate success on host verification.

    ``finish`` is deliberately not a success command.  It only asks the host
    to run the immutable :class:`Verifier` command captured outside the model.
    """

    def __init__(
        self,
        model: ModelClient,
        registry: ToolRegistry,
        verifier: Verifier | _Verifier,
        *,
        trace: JsonlTrace | _Trace | None = None,
        approval: _ApprovalWorkflow | None = None,
        max_iterations: int = 12,
        max_repeated_calls: int = 3,
        context_max_recent_groups: int = 8,
        context_max_tool_content_chars: int = 8_000,
        protected_patterns: Sequence[str] = DEFAULT_PROTECTED_PATTERNS,
    ) -> None:
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if (
            not isinstance(max_repeated_calls, int)
            or isinstance(max_repeated_calls, bool)
            or max_repeated_calls < 1
        ):
            raise ValueError("max_repeated_calls must be a positive integer")
        if isinstance(protected_patterns, str) or not isinstance(protected_patterns, Sequence):
            raise TypeError("protected_patterns must be a sequence of strings")
        if not all(isinstance(pattern, str) and pattern for pattern in protected_patterns):
            raise ValueError("protected_patterns must contain non-empty strings")
        if (
            not isinstance(context_max_recent_groups, int)
            or isinstance(context_max_recent_groups, bool)
            or context_max_recent_groups < 0
        ):
            raise ValueError("context_max_recent_groups must be a non-negative integer")
        if (
            not isinstance(context_max_tool_content_chars, int)
            or isinstance(context_max_tool_content_chars, bool)
            or context_max_tool_content_chars < 1
        ):
            raise ValueError("context_max_tool_content_chars must be a positive integer")
        self.model = model
        self.registry = registry
        self.verifier = verifier
        self.trace = trace
        self.approval = approval
        self.max_iterations = max_iterations
        self.max_repeated_calls = max_repeated_calls
        self.context_max_recent_groups = context_max_recent_groups
        self.context_max_tool_content_chars = context_max_tool_content_chars
        verifier_patterns = getattr(verifier, "protected_patterns", ())
        if isinstance(verifier_patterns, str) or not isinstance(verifier_patterns, Sequence):
            raise TypeError("verifier protected_patterns must be a sequence of strings")
        if not all(isinstance(pattern, str) and pattern for pattern in verifier_patterns):
            raise ValueError("verifier protected_patterns must contain non-empty strings")
        self.protected_patterns = tuple(dict.fromkeys((*protected_patterns, *verifier_patterns)))
        verifier_runner = getattr(verifier, "runner", None)
        verifier_root = getattr(verifier_runner, "workspace_root", None)
        self.protected_root = Path(verifier_root) if verifier_root is not None else None

    def run(self, task: str) -> AgentRunResult:
        """Attempt *task*, returning a structured result for every exit path."""
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-blank string")
        # Each call receives the current task as a fresh, immutable user anchor.
        context = BoundedContext(
            {"role": "developer", "content": _developer_prompt()},
            {"role": "user", "content": task},
            max_recent_groups=self.context_max_recent_groups,
            max_tool_content_chars=self.context_max_tool_content_chars,
        )
        state = RunState()
        final_text = ""
        last_signature: str | None = None
        self._trace(
            "run_start", {"task_chars": len(task), "tool_count": len(self.registry.names())}
        )

        for iteration in range(1, self.max_iterations + 1):
            state.iteration = iteration
            self._trace("model_turn", {"iteration": iteration, "stage": "start"})
            model_started_ns = monotonic_ns()
            try:
                turn = self.model.complete(context.messages(), self.registry.schemas())
            except ModelError as error:
                self._trace(
                    "model_turn",
                    {
                        "iteration": iteration,
                        "stage": "complete",
                        "ok": False,
                        "error_code": error.code,
                        "duration_ms": _elapsed_ms(model_started_ns),
                    },
                )
                return self._stop(
                    False,
                    f"Model request stopped ({error.code}).",
                    error.code,
                    state,
                    context,
                )
            except Exception:  # noqa: BLE001 - model clients are an external exception boundary.
                self._trace(
                    "model_turn",
                    {
                        "iteration": iteration,
                        "stage": "complete",
                        "ok": False,
                        "error_code": "model_request_failed",
                        "duration_ms": _elapsed_ms(model_started_ns),
                    },
                )
                return self._stop(
                    False,
                    "Model request stopped unexpectedly.",
                    "model_request_failed",
                    state,
                    context,
                )
            self._trace(
                "model_turn",
                {
                    "iteration": iteration,
                    "stage": "complete",
                    "ok": True,
                    "error_code": None,
                    "duration_ms": _elapsed_ms(model_started_ns),
                },
            )

            if not isinstance(turn, AssistantTurn):
                return self._stop(
                    False,
                    "Model returned an invalid turn.",
                    "invalid_model_response",
                    state,
                    context,
                )
            final_text = turn.content
            if not _valid_turn(turn):
                return self._stop(
                    False,
                    "Model returned invalid tool-call identifiers.",
                    "invalid_model_response",
                    state,
                    context,
                )
            assistant_message = _assistant_message(turn)
            if not turn.tool_calls:
                context.append_transaction(assistant_message)
                return self._stop(False, final_text, "model_stopped_without_finish", state, context)

            tool_messages: list[dict[str, Any]] = []
            made_progress = False
            successful_finish = False
            finish_seen = False
            signature = _call_batch_signature(turn.tool_calls)
            for call in turn.tool_calls:
                self._trace(
                    "tool_call",
                    {
                        "iteration": iteration,
                        "tool": call.name,
                        "argument_summary": _argument_summary(call),
                    },
                )
                started_ns = monotonic_ns()
                if call.name == "finish" and finish_seen:
                    result = ToolResult.failure(
                        "finish may be requested only once per assistant turn",
                        "duplicate_finish",
                    )
                    progressed = False
                    finished = False
                else:
                    if call.name == "finish":
                        finish_seen = True
                    result, progressed, finished = self._execute_call(call, state)
                duration_ms = _elapsed_ms(started_ns)
                made_progress = made_progress or progressed
                successful_finish = successful_finish or finished
                tool_messages.append(_tool_message(call, result))
                self._trace(
                    "tool_result",
                    {
                        "iteration": iteration,
                        "tool": call.name,
                        "ok": result.ok,
                        "error_code": result.error_code,
                        "exit_code": result.exit_code,
                        "changed": _is_changed_result(result),
                        "duration_ms": duration_ms,
                    },
                )

            context.append_transaction(assistant_message, tool_messages)
            if made_progress:
                state.consecutive_no_progress = 0
                last_signature = None
            elif signature == last_signature:
                state.consecutive_no_progress += 1
            else:
                last_signature = signature
                state.consecutive_no_progress = 1

            # The full assistant turn must be represented before declaring success.
            # A later edit in the same turn invalidates the earlier verification.
            if successful_finish and state.verified_after_last_edit:
                approval_failure = self._request_approval(state)
                if approval_failure is not None:
                    return self._stop(False, final_text, approval_failure, state, context)
                return self._stop(True, final_text, "verified", state, context)
            if state.consecutive_no_progress >= self.max_repeated_calls:
                return self._stop(False, final_text, "repeated_no_progress", state, context)

        return self._stop(False, final_text, "max_iterations", state, context)

    def _execute_call(self, call: ToolCall, state: RunState) -> tuple[ToolResult, bool, bool]:
        if call.argument_error is not None:
            return (
                ToolResult.failure("tool arguments could not be parsed", "invalid_arguments"),
                False,
                False,
            )
        if call.name in {"edit_file", "write_file"} and _is_protected_path(
            call.arguments.get("path"), self.protected_patterns, self.protected_root
        ):
            # This is an early feedback layer only; Workspace's canonical path
            # check remains the security boundary for aliases and symlinks.
            return (
                ToolResult.failure(
                    "editing verification assets is not allowed by this runner",
                    "protected_path",
                ),
                False,
                False,
            )
        if call.name != "finish":
            result = _normalise_tool_result(
                self.registry.execute(call.name, call.arguments), "invalid_tool_result"
            )
            if _is_successful_edit(call.name, result):
                path = result.data.get("path")
                assert isinstance(path, str)
                state.record_edit(path)
                return result, True, False
            return result, False, False

        requested = _normalise_tool_result(
            self.registry.execute(call.name, call.arguments), "invalid_tool_result"
        )
        if not requested.ok:
            return requested, False, False
        if state.edit_count < 1:
            return (
                ToolResult.failure("at least one successful edit is required", "no_edits"),
                False,
                False,
            )
        if not any(Path(path).suffix.lower() in {".py", ".pyi"} for path in state.changed_files):
            return (
                ToolResult.failure(
                    "at least one Python source edit is required", "no_source_edits"
                ),
                False,
                False,
            )
        verification = self._verify(state)
        verified = _verification_succeeded(verification)
        return verification, verified, verified

    def _verify(self, state: RunState) -> ToolResult:
        self._trace("verification", {"stage": "start"})
        started_ns = monotonic_ns()
        try:
            result = self.verifier.verify()
        except Exception:  # noqa: BLE001 - verifier implementations are an external boundary.
            result = ToolResult.failure(
                "independent verification could not run", "verification_exception"
            )
        result = _normalise_tool_result(result, "verification_invalid_result")
        if result.ok and result.exit_code != 0:
            result = ToolResult.failure(
                "independent verifier must return exit code 0",
                "verification_invalid_result",
                data=result.data,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=result.timed_out,
                truncated=result.truncated,
            )
        result = _normalise_tool_result(result, "verification_invalid_result")
        exit_code = result.exit_code
        state.record_verification(
            exit_code if type(exit_code) is int else -1,
            passed=_verification_succeeded(result),
        )
        self._trace(
            "verification",
            {
                "stage": "complete",
                "ok": result.ok,
                "exit_code": state.last_verify_exit_code,
                "duration_ms": _elapsed_ms(started_ns),
            },
        )
        return result

    def _request_approval(self, state: RunState) -> str | None:
        approval = self.approval
        if approval is None:
            return None

        changed_files = tuple(sorted(state.changed_files))
        verification_exit = state.last_verify_exit_code
        if verification_exit != 0:
            return None
        self._trace(
            "approval",
            {
                "stage": "start",
                "changed_file_count": len(changed_files),
                "verification_exit": verification_exit,
            },
        )

        try:
            response = approval.request(
                changed_files=changed_files,
                verification_exit_code=verification_exit,
            )
        except (Exception, KeyboardInterrupt):  # noqa: BLE001 - approval fails closed.
            decision = "unavailable"
            request_ok = False
            request_error_code = "approval_request_failed"
        else:
            if response is True:
                try:
                    approval.commit()
                except (Exception, KeyboardInterrupt):  # noqa: BLE001 - commit fails closed.
                    decision = "unavailable"
                    request_ok = False
                    request_error_code = "approval_commit_failed"
                else:
                    decision = "approved"
                    request_ok = True
                    request_error_code = None
            elif response is False:
                decision = "rejected"
                request_ok = True
                request_error_code = None
            else:
                decision = "unavailable"
                request_ok = False
                request_error_code = "approval_invalid_response"

        state.record_approval(decision)
        self._trace(
            "approval",
            {
                "stage": "complete",
                "decision": decision,
                "ok": request_ok,
                "error_code": request_error_code,
            },
        )
        if decision == "approved":
            return None

        try:
            approval.rollback()
        except (Exception, KeyboardInterrupt):  # noqa: BLE001 - rollback fails closed.
            self._trace(
                "approval",
                {
                    "stage": "rollback",
                    "decision": decision,
                    "ok": False,
                    "error_code": "rollback_failed",
                },
            )
            return "rollback_failed"
        self._trace(
            "approval",
            {
                "stage": "rollback",
                "decision": decision,
                "ok": True,
                "error_code": None,
            },
        )
        return "approval_rejected" if decision == "rejected" else "approval_unavailable"

    def _stop(
        self,
        success: bool,
        final_text: str,
        reason: str,
        state: RunState,
        context: BoundedContext,
    ) -> AgentRunResult:
        state.phase = RunPhase.SUCCESS if success else RunPhase.FAILED
        state.stop_reason = reason
        self._trace("stop", {"success": success, "reason": reason, "iteration": state.iteration})
        raw_path = getattr(self.trace, "path", None) if self.trace is not None else None
        trace_path = Path(raw_path) if isinstance(raw_path, (str, Path)) else None
        return AgentRunResult(
            success=success,
            final_text=final_text,
            stop_reason=reason,
            state=state,
            messages=tuple(context.messages()),
            trace_path=trace_path,
        )

    def _trace(self, event: str, payload: Mapping[str, Any]) -> None:
        if self.trace is None:
            return
        try:
            self.trace.record(event, payload)
        except Exception:  # noqa: BLE001 - tracing must never discard an agent result.
            # Tracing is audit support, never a new reason to lose a repair result.
            return


def _developer_prompt() -> str:
    return (
        "You are a careful coding agent. Inspect before editing, use only the supplied tools, "
        "and make small source changes. You may request finish only after a successful source "
        "edit; the host independently runs the fixed verifier and is the only authority that can "
        "report success. Do not modify test or verification configuration files."
    )


def _assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": turn.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                },
            }
            for call in turn.tool_calls
        ],
    }


def _valid_turn(turn: AssistantTurn) -> bool:
    """Validate all model output before serializing or executing one call."""
    if not isinstance(turn.content, str) or not isinstance(turn.tool_calls, tuple):
        return False
    identifiers: set[str] = set()
    for call in turn.tool_calls:
        if not isinstance(call, ToolCall):
            return False
        if not isinstance(call.id, str) or not call.id or call.id in identifiers:
            return False
        if not isinstance(call.name, str) or not call.name:
            return False
        if call.argument_error is not None and not isinstance(call.argument_error, str):
            return False
        if not isinstance(call.arguments, Mapping) or not all(
            isinstance(key, str) for key in call.arguments
        ):
            return False
        try:
            arguments = call.arguments_dict()
            json.dumps(arguments, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError, AttributeError):
            return False
        identifiers.add(call.id)
    return True


def _tool_message(call: ToolCall, result: ToolResult) -> dict[str, Any]:
    safe_result = _normalise_tool_result(result, "invalid_tool_result")
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": json.dumps(
            safe_result.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False
        ),
    }


def _is_changed_result(result: ToolResult) -> bool:
    return bool(
        result.ok
        and isinstance(result.data, Mapping)
        and result.data.get("changed") is True
        and isinstance(result.data.get("path"), str)
    )


def _is_successful_edit(name: str, result: ToolResult) -> bool:
    return name in {"edit_file", "write_file"} and _is_changed_result(result)


def _verification_succeeded(result: ToolResult) -> bool:
    return result.ok and type(result.exit_code) is int and result.exit_code == 0


def _call_batch_signature(calls: Sequence[ToolCall]) -> str:
    return json.dumps(
        [
            {
                "name": call.name,
                "arguments": call.arguments_dict(),
                "argument_error": bool(call.argument_error),
            }
            for call in calls
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _argument_summary(call: ToolCall) -> dict[str, Any]:
    """Describe argument shape and size without retaining model-provided values."""
    arguments = call.arguments_dict()
    type_counts: dict[str, int] = {}
    for value in arguments.values():
        kind = _json_value_kind(value)
        type_counts[kind] = type_counts.get(kind, 0) + 1
    encoded = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "count": len(arguments),
        "json_chars": len(encoded),
        "parse_error": call.argument_error is not None,
        "types": dict(sorted(type_counts.items())),
    }


def _json_value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    return "array"


def _elapsed_ms(started_ns: int) -> float:
    return round(max(0, monotonic_ns() - started_ns) / 1_000_000, 3)


def _is_protected_path(
    path: Any, patterns: Sequence[str], workspace_root: Path | None = None
) -> bool:
    if not isinstance(path, str):
        return False
    normalised = posixpath.normpath(path.replace("\\", "/"))
    if normalised in {"", ".", ".."} or normalised.startswith(("../", "/")):
        return False
    if is_protected_relative_path(
        normalised,
        patterns,
        case_insensitive=os.name == "nt",
    ):
        return True
    if workspace_root is None:
        return False
    try:
        canonical = (workspace_root / Path(path)).resolve(strict=False)
        canonical_relative = canonical.relative_to(workspace_root).as_posix()
    except (OSError, RuntimeError, ValueError):
        return False
    return is_protected_relative_path(
        canonical_relative,
        patterns,
        case_insensitive=os.name == "nt",
    )


def _normalise_tool_result(result: Any, error_code: str) -> ToolResult:
    """Keep malformed third-party ToolResult payloads out of chat history."""
    if not isinstance(result, ToolResult):
        return ToolResult.failure("tool returned an invalid result", error_code)
    try:
        canonical = ToolResult(
            ok=result.ok,
            data=result.data,
            error=result.error,
            error_code=result.error_code,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            truncated=result.truncated,
        )
        json.dumps(canonical.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (AttributeError, TypeError, ValueError):
        return ToolResult.failure("tool result is not JSON serializable", error_code)
    return canonical
