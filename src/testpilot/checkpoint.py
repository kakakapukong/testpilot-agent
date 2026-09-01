"""Strict, atomic local checkpoints for resumable TestPilot runs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

from .approval import ApprovalError, ChangeJournal, JournalSnapshot
from .context import BoundedContext
from .types import RunPhase, RunState

CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_MAX_CHECKPOINT_BYTES = 16_000_000
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{16}\Z")

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "workspace",
        "request",
        "runtime",
        "journal",
        "fingerprints",
        "lifecycle",
        "created_at",
        "updated_at",
    }
)
_REQUEST_KEYS = frozenset({"task", "verifier", "max_iterations", "trace_path"})
_RUNTIME_KEYS = frozenset({"context", "state", "last_call_signature"})
_STATE_KEYS = frozenset(
    {
        "phase",
        "iteration",
        "edit_count",
        "source_edit_count",
        "changed_files",
        "last_verify_exit_code",
        "verified_after_last_edit",
        "consecutive_no_progress",
        "stop_reason",
        "approval_status",
        "review_status",
        "review_rounds",
        "review_rework_count",
        "reviewed_edit_count",
        "reviewed_source_edit_count",
    }
)
_JOURNAL_RECORD_KEYS = frozenset({"path", "original", "mode", "missing_parents"})
_FINGERPRINT_KEYS = frozenset({"path", "kind", "mode", "sha256"})
_APPROVAL_STATUSES = frozenset({"approved", "rejected", "unavailable"})
_REVIEW_STATUSES = frozenset({"passed", "changes_requested", "unavailable"})


class CheckpointError(RuntimeError):
    """A content-free checkpoint failure with a stable public code."""

    def __init__(self, code: str) -> None:
        super().__init__("checkpoint operation failed")
        self.code = code


@dataclass(frozen=True)
class CheckpointRequest:
    """Host-owned run configuration persisted without model credentials."""

    task: str
    verifier: tuple[str, ...]
    max_iterations: int
    trace_path: str

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.task, str) or not self.task.strip():
                raise ValueError
            if (
                not isinstance(self.verifier, tuple)
                or not self.verifier
                or not all(isinstance(item, str) and item for item in self.verifier)
            ):
                raise ValueError
            _non_negative_integer(self.max_iterations, positive=True)
            _relative_path(self.trace_path)
            if Path(self.trace_path).suffix.lower() != ".jsonl":
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise CheckpointError("checkpoint_invalid") from exc


@dataclass(frozen=True)
class FileFingerprint:
    """Bounded identity of one journal-tracked current file."""

    path: str
    kind: str
    mode: int | None
    sha256: str | None

    def __post_init__(self) -> None:
        try:
            _relative_path(self.path)
            if self.kind == "missing":
                if self.mode is not None or self.sha256 is not None:
                    raise ValueError
            elif self.kind == "file":
                _file_mode(self.mode)
                if (
                    not isinstance(self.sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
                ):
                    raise ValueError
            else:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise CheckpointError("checkpoint_invalid") from exc


@dataclass(frozen=True)
class RunCheckpoint:
    """Closed, validated representation of one safe run boundary."""

    schema_version: int
    run_id: str
    workspace_identity: str
    request: CheckpointRequest
    context: Mapping[str, Any]
    state: Mapping[str, Any]
    last_call_signature: str | None
    journal: tuple[JournalSnapshot, ...]
    fingerprints: tuple[FileFingerprint, ...]
    lifecycle_status: str
    safe_point: int
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        try:
            if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError
            _run_id(self.run_id)
            if not isinstance(self.workspace_identity, str) or not self.workspace_identity:
                raise ValueError
            if not isinstance(self.request, CheckpointRequest):
                raise TypeError
            normalized_context = BoundedContext.from_snapshot(
                _thaw_json(self.context)
            ).snapshot()
            normalized_state = encode_run_state(decode_run_state(_thaw_json(self.state)))
            if self.last_call_signature is not None and not isinstance(
                self.last_call_signature, str
            ):
                raise TypeError
            if not isinstance(self.journal, tuple) or not all(
                isinstance(item, JournalSnapshot) for item in self.journal
            ):
                raise TypeError
            if not isinstance(self.fingerprints, tuple) or not all(
                isinstance(item, FileFingerprint) for item in self.fingerprints
            ):
                raise TypeError
            journal_paths = [_validate_journal_snapshot(item).path for item in self.journal]
            fingerprint_paths = [item.path for item in self.fingerprints]
            if journal_paths != sorted(set(journal_paths)):
                raise ValueError
            if fingerprint_paths != sorted(set(fingerprint_paths)):
                raise ValueError
            if journal_paths != fingerprint_paths:
                raise ValueError
            if self.lifecycle_status not in {"active", "terminal"}:
                raise ValueError
            _non_negative_integer(self.safe_point)
            created = _timestamp(self.created_at)
            updated = _timestamp(self.updated_at)
            if updated < created:
                raise ValueError
        except CheckpointError:
            raise
        except (TypeError, ValueError) as exc:
            raise CheckpointError("checkpoint_invalid") from exc

        object.__setattr__(self, "context", _freeze_json(normalized_context))
        object.__setattr__(self, "state", _freeze_json(normalized_state))


@dataclass(frozen=True)
class ResumeData:
    """Validated mutable runtime objects reconstructed from one checkpoint."""

    context: BoundedContext
    state: RunState
    last_call_signature: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.context, BoundedContext) or not isinstance(
            self.state, RunState
        ):
            raise CheckpointError("checkpoint_invalid")
        if self.last_call_signature is not None and not isinstance(
            self.last_call_signature, str
        ):
            raise CheckpointError("checkpoint_invalid")


@dataclass(frozen=True)
class FinalizeResult:
    """Checkpoint cleanup result that cannot alter the business outcome."""

    cleanup_warning: str | None

    def __post_init__(self) -> None:
        if self.cleanup_warning not in {None, "checkpoint_cleanup_failed"}:
            raise ValueError("invalid checkpoint cleanup warning")


def workspace_identity(root: Path) -> str:
    """Return the canonical, platform-normalized identity for a workspace."""
    try:
        resolved = Path(root).resolve(strict=True)
        if not resolved.is_dir():
            raise OSError
    except (OSError, RuntimeError) as exc:
        raise CheckpointError("checkpoint_invalid") from exc
    return os.path.normcase(str(resolved))


def fingerprint_path(
    root: Path,
    relative: str,
    *,
    max_bytes: int,
) -> FileFingerprint:
    """Hash one bounded regular file without following workspace aliases."""
    try:
        if type(max_bytes) is not int or max_bytes < 1:
            raise TypeError
        canonical_root = Path(root).resolve(strict=True)
        if not canonical_root.is_dir():
            raise ValueError
        normalized = _relative_path(relative)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint_invalid") from exc

    candidate = canonical_root.joinpath(*PurePosixPath(normalized).parts)
    try:
        if candidate.is_symlink():
            raise CheckpointError("checkpoint_workspace_changed")
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(canonical_root)
        if resolved != candidate:
            raise CheckpointError("checkpoint_workspace_changed")
        if not candidate.exists():
            return FileFingerprint(normalized, "missing", None, None)

        before = candidate.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise CheckpointError("checkpoint_workspace_changed")

        digest = hashlib.sha256()
        total = 0
        with candidate.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
                raise CheckpointError("checkpoint_workspace_changed")
            while chunk := stream.read(64 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise CheckpointError("checkpoint_workspace_changed")
                digest.update(chunk)
            after = os.fstat(stream.fileno())

        current = candidate.stat(follow_symlinks=False)
        if (
            not _same_file(before, after)
            or not _same_file(after, current)
            or before.st_size != total
            or after.st_size != total
            or current.st_size != total
            or getattr(before, "st_mtime_ns", None)
            != getattr(current, "st_mtime_ns", None)
            or candidate.is_symlink()
            or candidate.resolve(strict=True) != candidate
        ):
            raise CheckpointError("checkpoint_workspace_changed")
        return FileFingerprint(
            normalized,
            "file",
            stat.S_IMODE(current.st_mode),
            digest.hexdigest(),
        )
    except CheckpointError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointError("checkpoint_workspace_changed") from exc


def encode_run_state(state: RunState) -> dict[str, Any]:
    """Encode all stable RunState fields after validating their relationships."""
    if not isinstance(state, RunState):
        raise CheckpointError("checkpoint_invalid")
    payload: dict[str, Any] = {
        "phase": state.phase.value if isinstance(state.phase, RunPhase) else state.phase,
        "iteration": state.iteration,
        "edit_count": state.edit_count,
        "source_edit_count": state.source_edit_count,
        "changed_files": sorted(state.changed_files)
        if isinstance(state.changed_files, set)
        else state.changed_files,
        "last_verify_exit_code": state.last_verify_exit_code,
        "verified_after_last_edit": state.verified_after_last_edit,
        "consecutive_no_progress": state.consecutive_no_progress,
        "stop_reason": state.stop_reason,
        "approval_status": state.approval_status,
        "review_status": state.review_status,
        "review_rounds": state.review_rounds,
        "review_rework_count": state.review_rework_count,
        "reviewed_edit_count": state.reviewed_edit_count,
        "reviewed_source_edit_count": state.reviewed_source_edit_count,
    }
    return _state_payload(decode_run_state(payload))


def decode_run_state(payload: Mapping[str, Any]) -> RunState:
    """Decode a strictly shaped state mapping into a fresh mutable RunState."""
    try:
        data = _exact_object(payload, _STATE_KEYS)
        phase_value = _required_string(data["phase"])
        phase = RunPhase(phase_value)
        iteration = _non_negative_integer(data["iteration"])
        edit_count = _non_negative_integer(data["edit_count"])
        source_edit_count = _non_negative_integer(data["source_edit_count"])
        if source_edit_count > edit_count:
            raise ValueError

        changed_raw = data["changed_files"]
        if not isinstance(changed_raw, list):
            raise TypeError
        changed_files = [_relative_path(item) for item in changed_raw]
        if changed_files != sorted(set(changed_files)):
            raise ValueError

        last_verify = _optional_integer(data["last_verify_exit_code"])
        verified = _required_boolean(data["verified_after_last_edit"])
        no_progress = _non_negative_integer(data["consecutive_no_progress"])
        stop_reason = _optional_non_blank_string(data["stop_reason"])
        approval_status = _optional_choice(data["approval_status"], _APPROVAL_STATUSES)
        review_status = _optional_choice(data["review_status"], _REVIEW_STATUSES)
        review_rounds = _non_negative_integer(data["review_rounds"])
        review_rework_count = _non_negative_integer(data["review_rework_count"])
        if review_rework_count not in {0, 1} or review_rework_count > review_rounds:
            raise ValueError
        reviewed_edit_count = _optional_non_negative_integer(data["reviewed_edit_count"])
        reviewed_source_edit_count = _optional_non_negative_integer(
            data["reviewed_source_edit_count"]
        )
        if (reviewed_edit_count is None) != (reviewed_source_edit_count is None):
            raise ValueError
        if reviewed_edit_count is not None and reviewed_edit_count > edit_count:
            raise ValueError
        if (
            reviewed_source_edit_count is not None
            and reviewed_source_edit_count > source_edit_count
        ):
            raise ValueError
        if review_status is not None and review_rounds < 1:
            raise ValueError
        if review_status == "changes_requested" and (
            review_rework_count != 1 or reviewed_edit_count is None
        ):
            raise ValueError
    except CheckpointError:
        raise
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint_invalid") from exc

    return RunState(
        phase=phase,
        iteration=iteration,
        edit_count=edit_count,
        source_edit_count=source_edit_count,
        changed_files=set(changed_files),
        last_verify_exit_code=last_verify,
        verified_after_last_edit=verified,
        consecutive_no_progress=no_progress,
        stop_reason=stop_reason,
        approval_status=approval_status,
        review_status=review_status,
        review_rounds=review_rounds,
        review_rework_count=review_rework_count,
        reviewed_edit_count=reviewed_edit_count,
        reviewed_source_edit_count=reviewed_source_edit_count,
    )


class CheckpointStore:
    """Read and atomically replace bounded checkpoint JSON files."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    ) -> None:
        try:
            if (
                type(max_checkpoint_bytes) is not int
                or max_checkpoint_bytes < 1
            ):
                raise ValueError
            resolved = Path(root).resolve(strict=True)
            if not resolved.is_dir():
                raise ValueError
            self.root = resolved
            self.max_checkpoint_bytes = max_checkpoint_bytes
            self.directory = self._ensure_directory()
        except CheckpointError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise CheckpointError("checkpoint_invalid") from exc

    def new_run_id(self) -> str:
        """Return a fresh random ID whose target is currently unused."""
        directory = self._stable_directory()
        for _ in range(16):
            run_id = secrets.token_hex(8)
            target = directory / f"{run_id}.json"
            if not target.exists() and not target.is_symlink():
                return run_id
        raise CheckpointError("checkpoint_save_failed")

    def path_for(self, run_id: str) -> Path:
        """Derive a target only from a strictly validated run ID."""
        return self.directory / f"{_run_id(run_id)}.json"

    def save(self, checkpoint: RunCheckpoint) -> None:
        """Atomically write a complete checkpoint without harming the old one."""
        if not isinstance(checkpoint, RunCheckpoint):
            raise CheckpointError("checkpoint_save_failed")
        if checkpoint.workspace_identity != workspace_identity(self.root):
            raise CheckpointError("checkpoint_workspace_mismatch")
        encoded = _encode_checkpoint(checkpoint)
        if len(encoded) > self.max_checkpoint_bytes:
            raise CheckpointError("checkpoint_too_large")

        temporary_path: Path | None = None
        try:
            directory = self._stable_directory()
            target = directory / f"{_run_id(checkpoint.run_id)}.json"
            if target.is_symlink():
                raise CheckpointError("checkpoint_invalid")
            if target.exists() and not stat.S_ISREG(target.stat(follow_symlinks=False).st_mode):
                raise CheckpointError("checkpoint_invalid")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=directory,
                prefix=f".{checkpoint.run_id}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                os.chmod(temporary_path, 0o600)
            except OSError:
                pass
            os.replace(temporary_path, target)
            temporary_path = None
        except CheckpointError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise CheckpointError("checkpoint_save_failed") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def load(self, run_id: str) -> RunCheckpoint:
        """Load one bounded, strictly validated checkpoint."""
        validated_id = _run_id(run_id)
        try:
            directory = self._stable_directory()
            target = directory / f"{validated_id}.json"
            if target.is_symlink():
                raise CheckpointError("checkpoint_invalid")
            if not target.exists():
                raise CheckpointError("checkpoint_load_failed")
            if not stat.S_ISREG(target.stat(follow_symlinks=False).st_mode):
                raise CheckpointError("checkpoint_invalid")
            with target.open("rb") as stream:
                encoded = stream.read(self.max_checkpoint_bytes + 1)
        except CheckpointError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise CheckpointError("checkpoint_load_failed") from exc
        if len(encoded) > self.max_checkpoint_bytes:
            raise CheckpointError("checkpoint_too_large")
        checkpoint = _decode_checkpoint(encoded)
        if checkpoint.run_id != validated_id:
            raise CheckpointError("checkpoint_invalid")
        return checkpoint

    def delete(self, run_id: str) -> None:
        """Delete one validated checkpoint; a missing target is already clean."""
        validated_id = _run_id(run_id)
        try:
            directory = self._stable_directory()
            target = directory / f"{validated_id}.json"
            if target.is_symlink():
                raise CheckpointError("checkpoint_invalid")
            if target.exists() and not stat.S_ISREG(target.stat(follow_symlinks=False).st_mode):
                raise CheckpointError("checkpoint_invalid")
            target.unlink(missing_ok=True)
        except CheckpointError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise CheckpointError("checkpoint_cleanup_failed") from exc

    def _ensure_directory(self) -> Path:
        testpilot = self.root / ".testpilot"
        checkpoints = testpilot / "checkpoints"
        for candidate in (testpilot, checkpoints):
            if candidate.is_symlink():
                raise CheckpointError("checkpoint_invalid")
            candidate.mkdir(exist_ok=True)
            if candidate.is_symlink() or not candidate.is_dir():
                raise CheckpointError("checkpoint_invalid")
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(self.root)
            except ValueError as exc:
                raise CheckpointError("checkpoint_invalid") from exc
            if resolved != candidate:
                raise CheckpointError("checkpoint_invalid")
        return checkpoints

    def _stable_directory(self) -> Path:
        testpilot = self.root / ".testpilot"
        checkpoints = testpilot / "checkpoints"
        try:
            if (
                testpilot.is_symlink()
                or checkpoints.is_symlink()
                or not testpilot.is_dir()
                or not checkpoints.is_dir()
                or checkpoints.resolve(strict=True) != self.directory
            ):
                raise CheckpointError("checkpoint_invalid")
        except (OSError, RuntimeError) as exc:
            raise CheckpointError("checkpoint_invalid") from exc
        return self.directory


class CheckpointSession:
    """Own one run's safe points, restoration, and terminal cleanup."""

    def __init__(
        self,
        *,
        store: CheckpointStore,
        journal: ChangeJournal,
        request: CheckpointRequest,
        run_id: str,
        created_at: str,
        safe_point: int = 0,
        latest: RunCheckpoint | None = None,
        on_ready: Callable[[str, Path], None] | None = None,
    ) -> None:
        self.store = store
        self.journal = journal
        self.request = request
        self.run_id = run_id
        self.created_at = created_at
        self.safe_point = safe_point
        self.active = True
        self._latest = latest
        self._on_ready = on_ready
        self._announced = False

    @classmethod
    def create(
        cls,
        *,
        store: CheckpointStore,
        journal: ChangeJournal,
        request: CheckpointRequest,
        run_id: str | None = None,
        on_ready: Callable[[str, Path], None] | None = None,
    ) -> CheckpointSession:
        """Bind a new run to one canonical workspace without writing yet."""
        _validate_session_inputs(store, journal, request, on_ready)
        chosen_id = store.new_run_id() if run_id is None else _run_id(run_id)
        target = store.path_for(chosen_id)
        if target.exists() or target.is_symlink():
            raise CheckpointError("checkpoint_invalid")
        return cls(
            store=store,
            journal=journal,
            request=request,
            run_id=chosen_id,
            created_at=_utc_now(),
            on_ready=on_ready,
        )

    @classmethod
    def restore(
        cls,
        *,
        store: CheckpointStore,
        journal: ChangeJournal,
        run_id: str,
        on_ready: Callable[[str, Path], None] | None = None,
    ) -> tuple[CheckpointSession, ResumeData]:
        """Validate an active safe point before rebuilding mutable run state."""
        _validate_session_inputs(store, journal, None, on_ready)
        checkpoint = store.load(run_id)
        if checkpoint.lifecycle_status != "active":
            raise CheckpointError("checkpoint_invalid")
        if checkpoint.workspace_identity != workspace_identity(store.root):
            raise CheckpointError("checkpoint_workspace_mismatch")

        current = _capture_fingerprints(
            store.root,
            checkpoint.journal,
            max_bytes=journal.max_snapshot_bytes,
        )
        if not _fingerprints_match(checkpoint.fingerprints, current):
            raise CheckpointError("checkpoint_workspace_changed")

        try:
            context = BoundedContext.from_snapshot(_thaw_json(checkpoint.context))
            _validate_context_task(context, checkpoint.request.task)
            state = decode_run_state(_thaw_json(checkpoint.state))
            state.invalidate_for_resume()
        except CheckpointError:
            raise
        except (TypeError, ValueError) as exc:
            raise CheckpointError("checkpoint_invalid") from exc

        try:
            journal.restore_snapshots(checkpoint.journal)
        except ApprovalError as exc:
            raise CheckpointError("checkpoint_invalid") from exc

        session = cls(
            store=store,
            journal=journal,
            request=checkpoint.request,
            run_id=checkpoint.run_id,
            created_at=checkpoint.created_at,
            safe_point=checkpoint.safe_point,
            latest=checkpoint,
            on_ready=on_ready,
        )
        return session, ResumeData(
            context=context,
            state=state,
            last_call_signature=checkpoint.last_call_signature,
        )

    @property
    def path(self) -> Path:
        """Return this run's validated checkpoint path."""
        return self.store.path_for(self.run_id)

    def save(
        self,
        *,
        context: BoundedContext,
        state: RunState,
        last_call_signature: str | None,
    ) -> None:
        """Atomically replace the run's most recent complete safe point."""
        if not self.active:
            raise CheckpointError("checkpoint_save_failed")
        if not isinstance(context, BoundedContext) or not isinstance(state, RunState):
            raise CheckpointError("checkpoint_invalid")
        if last_call_signature is not None and not isinstance(
            last_call_signature, str
        ):
            raise CheckpointError("checkpoint_invalid")
        _validate_context_task(context, self.request.task)

        try:
            journal = self.journal.export_snapshots()
        except ApprovalError as exc:
            raise CheckpointError("checkpoint_save_failed") from exc
        fingerprints = _capture_fingerprints(
            self.store.root,
            journal,
            max_bytes=self.journal.max_snapshot_bytes,
        )
        next_safe_point = self.safe_point + 1
        checkpoint = RunCheckpoint(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            run_id=self.run_id,
            workspace_identity=workspace_identity(self.store.root),
            request=self.request,
            context=context.snapshot(),
            state=encode_run_state(state),
            last_call_signature=last_call_signature,
            journal=journal,
            fingerprints=fingerprints,
            lifecycle_status="active",
            safe_point=next_safe_point,
            created_at=self.created_at,
            updated_at=_utc_now(),
        )
        self.store.save(checkpoint)
        self._latest = checkpoint
        self.safe_point = next_safe_point
        self._announce_ready()

    def finalize(self, outcome: str) -> FinalizeResult:
        """Mark a persisted run terminal, then remove its sensitive state."""
        if outcome not in {"completed", "approved", "rolled_back"}:
            raise ValueError("invalid checkpoint outcome")
        if not self.active or self._latest is None:
            raise CheckpointError("checkpoint_finalize_failed")

        terminal = replace(
            self._latest,
            lifecycle_status="terminal",
            updated_at=_utc_now(),
        )
        try:
            self.store.save(terminal)
        except CheckpointError:
            try:
                self.store.delete(self.run_id)
            except CheckpointError as delete_error:
                raise CheckpointError("checkpoint_finalize_failed") from delete_error
            self._latest = terminal
            self.active = False
            return FinalizeResult(None)

        self._latest = terminal
        self.active = False
        try:
            self.store.delete(self.run_id)
        except CheckpointError:
            return FinalizeResult("checkpoint_cleanup_failed")
        return FinalizeResult(None)

    def _announce_ready(self) -> None:
        if self._announced or self._on_ready is None:
            return
        self._announced = True
        self._on_ready(self.run_id, self.path)


def _validate_session_inputs(
    store: object,
    journal: object,
    request: object | None,
    on_ready: object,
) -> None:
    if not isinstance(store, CheckpointStore) or not isinstance(journal, ChangeJournal):
        raise CheckpointError("checkpoint_invalid")
    if request is not None and not isinstance(request, CheckpointRequest):
        raise CheckpointError("checkpoint_invalid")
    if on_ready is not None and not callable(on_ready):
        raise CheckpointError("checkpoint_invalid")
    if os.path.normcase(str(journal.root)) != workspace_identity(store.root):
        raise CheckpointError("checkpoint_workspace_mismatch")


def _validate_context_task(context: BoundedContext, task: str) -> None:
    try:
        messages = context.messages()
        user = messages[1]
        if user.get("role") != "user" or user.get("content") != task:
            raise ValueError
    except (IndexError, TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint_invalid") from exc


def _capture_fingerprints(
    root: Path,
    snapshots: Sequence[JournalSnapshot],
    *,
    max_bytes: int,
) -> tuple[FileFingerprint, ...]:
    try:
        if isinstance(snapshots, (str, bytes)) or not isinstance(snapshots, Sequence):
            raise TypeError
        normalized = tuple(_validate_journal_snapshot(item) for item in snapshots)
        paths = [item.path for item in normalized]
        if paths != sorted(set(paths)):
            raise ValueError
    except CheckpointError:
        raise
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint_invalid") from exc
    return tuple(
        fingerprint_path(root, snapshot.path, max_bytes=max_bytes)
        for snapshot in normalized
    )


def _fingerprints_match(
    expected: Sequence[FileFingerprint],
    current: Sequence[FileFingerprint],
) -> bool:
    if len(expected) != len(current):
        return False
    for left, right in zip(expected, current, strict=True):
        if (
            left.path != right.path
            or left.kind != right.kind
            or left.mode != right.mode
        ):
            return False
        if left.sha256 is None or right.sha256 is None:
            if left.sha256 is not right.sha256:
                return False
        elif not hmac.compare_digest(left.sha256, right.sha256):
            return False
    return True


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare stable file identity where the platform exposes it."""
    left_inode = getattr(left, "st_ino", 0)
    right_inode = getattr(right, "st_ino", 0)
    left_device = getattr(left, "st_dev", 0)
    right_device = getattr(right, "st_dev", 0)
    return (
        (not left_inode or not right_inode or left_inode == right_inode)
        and (not left_device or not right_device or left_device == right_device)
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _state_payload(state: RunState) -> dict[str, Any]:
    return {
        "phase": state.phase.value,
        "iteration": state.iteration,
        "edit_count": state.edit_count,
        "source_edit_count": state.source_edit_count,
        "changed_files": sorted(state.changed_files),
        "last_verify_exit_code": state.last_verify_exit_code,
        "verified_after_last_edit": state.verified_after_last_edit,
        "consecutive_no_progress": state.consecutive_no_progress,
        "stop_reason": state.stop_reason,
        "approval_status": state.approval_status,
        "review_status": state.review_status,
        "review_rounds": state.review_rounds,
        "review_rework_count": state.review_rework_count,
        "reviewed_edit_count": state.reviewed_edit_count,
        "reviewed_source_edit_count": state.reviewed_source_edit_count,
    }


def _encode_checkpoint(checkpoint: RunCheckpoint) -> bytes:
    payload = {
        "schema_version": checkpoint.schema_version,
        "run_id": checkpoint.run_id,
        "workspace": {"identity": checkpoint.workspace_identity},
        "request": {
            "task": checkpoint.request.task,
            "verifier": list(checkpoint.request.verifier),
            "max_iterations": checkpoint.request.max_iterations,
            "trace_path": checkpoint.request.trace_path,
        },
        "runtime": {
            "context": _thaw_json(checkpoint.context),
            "state": _thaw_json(checkpoint.state),
            "last_call_signature": checkpoint.last_call_signature,
        },
        "journal": {
            "snapshots": [
                {
                    "path": snapshot.path,
                    "original": _encode_original(snapshot.original),
                    "mode": snapshot.mode,
                    "missing_parents": list(snapshot.missing_parents),
                }
                for snapshot in checkpoint.journal
            ]
        },
        "fingerprints": [
            {
                "path": fingerprint.path,
                "kind": fingerprint.kind,
                "mode": fingerprint.mode,
                "sha256": fingerprint.sha256,
            }
            for fingerprint in checkpoint.fingerprints
        ],
        "lifecycle": {
            "status": checkpoint.lifecycle_status,
            "safe_point": checkpoint.safe_point,
        },
        "created_at": checkpoint.created_at,
        "updated_at": checkpoint.updated_at,
    }
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CheckpointError("checkpoint_invalid") from exc


def _decode_checkpoint(encoded: bytes) -> RunCheckpoint:
    try:
        text = encoded.decode("utf-8")
        raw = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        root = _exact_object(raw, _ROOT_KEYS)
        workspace = _exact_object(root["workspace"], frozenset({"identity"}))
        request_raw = _exact_object(root["request"], _REQUEST_KEYS)
        verifier_raw = request_raw["verifier"]
        if not isinstance(verifier_raw, list):
            raise TypeError
        request = CheckpointRequest(
            task=_required_string(request_raw["task"]),
            verifier=tuple(_required_string(item) for item in verifier_raw),
            max_iterations=_non_negative_integer(
                request_raw["max_iterations"],
                positive=True,
            ),
            trace_path=_required_string(request_raw["trace_path"]),
        )
        runtime = _exact_object(root["runtime"], _RUNTIME_KEYS)
        context = BoundedContext.from_snapshot(
            _exact_object(runtime["context"], None)
        ).snapshot()
        state = encode_run_state(
            decode_run_state(_exact_object(runtime["state"], _STATE_KEYS))
        )
        signature = runtime["last_call_signature"]
        if signature is not None and not isinstance(signature, str):
            raise TypeError

        journal_outer = _exact_object(root["journal"], frozenset({"snapshots"}))
        journal_raw = journal_outer["snapshots"]
        if not isinstance(journal_raw, list):
            raise TypeError
        journal = tuple(_decode_journal_snapshot(item) for item in journal_raw)
        journal_paths = [item.path for item in journal]
        if journal_paths != sorted(set(journal_paths)):
            raise ValueError

        fingerprints_raw = root["fingerprints"]
        if not isinstance(fingerprints_raw, list):
            raise TypeError
        fingerprints = tuple(_decode_fingerprint(item) for item in fingerprints_raw)
        fingerprint_paths = [item.path for item in fingerprints]
        if fingerprint_paths != sorted(set(fingerprint_paths)):
            raise ValueError

        lifecycle = _exact_object(root["lifecycle"], frozenset({"status", "safe_point"}))
        return RunCheckpoint(
            schema_version=_non_negative_integer(root["schema_version"]),
            run_id=_required_string(root["run_id"]),
            workspace_identity=_required_string(workspace["identity"]),
            request=request,
            context=context,
            state=state,
            last_call_signature=signature,
            journal=journal,
            fingerprints=fingerprints,
            lifecycle_status=_required_string(lifecycle["status"]),
            safe_point=_non_negative_integer(lifecycle["safe_point"]),
            created_at=_required_string(root["created_at"]),
            updated_at=_required_string(root["updated_at"]),
        )
    except CheckpointError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint_invalid") from exc


def _decode_journal_snapshot(value: Any) -> JournalSnapshot:
    record = _exact_object(value, _JOURNAL_RECORD_KEYS)
    path = _relative_path(record["path"])
    original = _decode_original(record["original"])
    mode = record["mode"]
    if original is None:
        if mode is not None:
            raise ValueError
    else:
        _file_mode(mode)
    parents_raw = record["missing_parents"]
    if not isinstance(parents_raw, list):
        raise TypeError
    parents = tuple(_relative_path(item) for item in parents_raw)
    if len(parents) != len(set(parents)):
        raise ValueError
    target_parts = PurePosixPath(path).parts
    for parent in parents:
        parent_parts = PurePosixPath(parent).parts
        if len(parent_parts) >= len(target_parts) or target_parts[: len(parent_parts)] != parent_parts:
            raise ValueError
    return JournalSnapshot(path, original, mode, parents)


def _decode_fingerprint(value: Any) -> FileFingerprint:
    record = _exact_object(value, _FINGERPRINT_KEYS)
    return FileFingerprint(
        path=_required_string(record["path"]),
        kind=_required_string(record["kind"]),
        mode=record["mode"],
        sha256=record["sha256"],
    )


def _validate_journal_snapshot(snapshot: JournalSnapshot) -> JournalSnapshot:
    path = _relative_path(snapshot.path)
    if snapshot.original is None:
        if snapshot.mode is not None:
            raise ValueError
    elif not isinstance(snapshot.original, bytes):
        raise TypeError
    else:
        _file_mode(snapshot.mode)
    if not isinstance(snapshot.missing_parents, tuple):
        raise TypeError
    parents = tuple(_relative_path(item) for item in snapshot.missing_parents)
    target_parts = PurePosixPath(path).parts
    if len(parents) != len(set(parents)):
        raise ValueError
    for parent in parents:
        parent_parts = PurePosixPath(parent).parts
        if len(parent_parts) >= len(target_parts) or target_parts[: len(parent_parts)] != parent_parts:
            raise ValueError
    return JournalSnapshot(
        path,
        None if snapshot.original is None else bytes(snapshot.original),
        snapshot.mode,
        parents,
    )


def _encode_original(value: bytes | None) -> str | None:
    if value is None:
        return None
    return base64.b64encode(value).decode("ascii")


def _decode_original(value: object) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise ValueError from exc


def _reject_constant(value: str) -> Any:
    raise ValueError("non-finite JSON number")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _exact_object(
    value: Any,
    keys: frozenset[str] | None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError
    if keys is not None and set(value) != keys:
        raise ValueError
    if not all(isinstance(key, str) for key in value):
        raise TypeError
    return value


def _required_string(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError
    return value


def _optional_non_blank_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError
    return value


def _required_boolean(value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError
    return value


def _non_negative_integer(value: Any, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise TypeError
    return value


def _optional_integer(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError
    return value


def _optional_non_negative_integer(value: Any) -> int | None:
    if value is None:
        return None
    return _non_negative_integer(value)


def _optional_choice(value: Any, choices: frozenset[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in choices:
        raise ValueError
    return value


def _file_mode(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 0o7777:
        raise TypeError
    return value


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise TypeError
    pure = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        pure.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in pure.parts
        or pure == PurePosixPath(".")
        or pure.as_posix() != value
    ):
        raise ValueError
    return value


def _run_id(value: Any) -> str:
    if not isinstance(value, str) or RUN_ID_PATTERN.fullmatch(value) is None:
        raise CheckpointError("checkpoint_invalid")
    return value


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TypeError
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError
    return parsed


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError("checkpoint_invalid")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointError("checkpoint_invalid")
            copied[key] = _freeze_json(item)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise CheckpointError("checkpoint_invalid")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
