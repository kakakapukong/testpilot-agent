"""Reversible snapshots for workspace file changes."""

from __future__ import annotations

import errno
import json
import os
import stat
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_SNAPSHOT_BYTES = 1_000_000


@dataclass(frozen=True)
class ChangeSummary:
    """Content-free line statistics for one captured workspace path."""

    path: str
    status: str
    additions: int
    deletions: int


class ApprovalError(RuntimeError):
    """A safe approval failure whose message contains no file contents."""


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    original: bytes | None
    mode: int | None
    missing_parents: tuple[Path, ...]


class ChangeJournal:
    """Capture the first state of changed files and restore it on demand."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_snapshot_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
    ) -> None:
        if (
            not isinstance(max_snapshot_bytes, int)
            or isinstance(max_snapshot_bytes, bool)
            or max_snapshot_bytes < 1
        ):
            raise ValueError("max_snapshot_bytes must be a positive integer")
        try:
            self.root = Path(root).resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ApprovalError("could not initialize workspace change journal") from exc
        self.max_snapshot_bytes = max_snapshot_bytes
        self._snapshots: dict[str, _Snapshot] = {}

    def capture(self, path: Path) -> None:
        """Save a path's bytes and mode once, before its first write."""
        target, relative = self._normalize(path)
        key = relative.as_posix()
        if key in self._snapshots:
            return

        try:
            missing_parents = self._missing_parents(target)
            if target.exists():
                with target.open("rb") as stream:
                    original = self._read_bounded(stream)
                    mode = stat.S_IMODE(os.fstat(stream.fileno()).st_mode)
            else:
                original = None
                mode = None
        except OSError as exc:
            raise ApprovalError("could not capture workspace change") from exc

        self._snapshots[key] = _Snapshot(
            path=target,
            original=original,
            mode=mode,
            missing_parents=missing_parents,
        )

    def summaries(self) -> tuple[ChangeSummary, ...]:
        """Return deterministic line counts without exposing file contents."""
        summaries: list[ChangeSummary] = []
        try:
            for relative, snapshot in sorted(self._snapshots.items()):
                current = self._read_current(snapshot.path)
                additions, deletions = self._line_changes(snapshot.original or b"", current)
                summaries.append(
                    ChangeSummary(
                        path=relative,
                        status="modified" if snapshot.original is not None else "created",
                        additions=additions,
                        deletions=deletions,
                    )
                )
        except ApprovalError:
            raise
        except OSError as exc:
            raise ApprovalError("could not summarize workspace changes") from exc
        return tuple(summaries)

    def rollback(self) -> None:
        """Restore old files atomically and remove files created during the run."""
        failed = False
        snapshots = tuple(self._snapshots.values())

        for snapshot in snapshots:
            if snapshot.original is None:
                continue
            try:
                self._restore(snapshot)
            except (ApprovalError, OSError):
                failed = True

        for snapshot in snapshots:
            if snapshot.original is not None:
                continue
            try:
                self._remove_created_file(snapshot.path)
            except (ApprovalError, OSError):
                failed = True

        missing_parents = {
            parent
            for snapshot in snapshots
            if snapshot.original is None
            for parent in snapshot.missing_parents
        }
        for parent in sorted(
            missing_parents,
            key=lambda candidate: (-len(candidate.parts), str(candidate)),
        ):
            try:
                self._remove_empty_created_directory(parent)
            except (ApprovalError, OSError):
                failed = True

        if failed:
            raise ApprovalError("could not roll back workspace changes")
        self.commit()

    def commit(self) -> None:
        """Forget completed-run snapshots so the journal can start a fresh run."""
        self._snapshots.clear()

    def _normalize(self, path: Path) -> tuple[Path, Path]:
        try:
            supplied = Path(path)
            target = (
                supplied.resolve(strict=False)
                if supplied.is_absolute()
                else (self.root / supplied).resolve(strict=False)
            )
            relative = target.relative_to(self.root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ApprovalError("path must be inside the workspace") from exc
        if relative == Path("."):
            raise ApprovalError("path must name a file inside the workspace")
        return target, relative

    def _missing_parents(self, target: Path) -> tuple[Path, ...]:
        missing: list[Path] = []
        parent = target.parent
        while parent != self.root and not parent.exists():
            missing.append(parent)
            parent = parent.parent
        return tuple(missing)

    def _read_current(self, target: Path) -> bytes:
        self._ensure_stable_parent(target)
        if target.is_symlink():
            try:
                resolved = target.resolve(strict=False)
            except RuntimeError as exc:
                raise ApprovalError("workspace path changed after capture") from exc
            if resolved != target:
                raise ApprovalError("workspace path changed after capture")
        if not target.exists():
            return b""
        if not target.is_file():
            raise ApprovalError("workspace path is no longer a file")
        with target.open("rb") as stream:
            return self._read_bounded(stream)

    def _read_bounded(self, stream: object) -> bytes:
        read = getattr(stream, "read", None)
        if not callable(read):
            raise ApprovalError("could not read workspace change")
        content = read(self.max_snapshot_bytes + 1)
        if not isinstance(content, bytes):
            raise ApprovalError("could not read workspace change")
        if len(content) > self.max_snapshot_bytes:
            raise ApprovalError("workspace file is too large to journal safely")
        return content

    @staticmethod
    def _line_changes(before: bytes, after: bytes) -> tuple[int, int]:
        """Count the changed middle span in linear time.

        Common leading and trailing lines are excluded.  Unchanged islands
        between separate edits remain in the counted span, so the result is a
        conservative summary rather than an expensive minimal diff.
        """
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        prefix = 0
        prefix_limit = min(len(before_lines), len(after_lines))
        while prefix < prefix_limit and before_lines[prefix] == after_lines[prefix]:
            prefix += 1

        suffix = 0
        before_remaining = len(before_lines) - prefix
        after_remaining = len(after_lines) - prefix
        suffix_limit = min(before_remaining, after_remaining)
        while (
            suffix < suffix_limit
            and before_lines[len(before_lines) - suffix - 1]
            == after_lines[len(after_lines) - suffix - 1]
        ):
            suffix += 1

        return (
            len(after_lines) - prefix - suffix,
            len(before_lines) - prefix - suffix,
        )

    def _restore(self, snapshot: _Snapshot) -> None:
        assert snapshot.original is not None
        assert snapshot.mode is not None
        target = snapshot.path
        self._ensure_stable_parent(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_stable_parent(target)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(snapshot.original)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_path, snapshot.mode)
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _remove_created_file(self, target: Path) -> None:
        self._ensure_stable_parent(target)
        if target.is_symlink() or target.exists():
            if target.is_dir() and not target.is_symlink():
                raise ApprovalError("created workspace path is no longer a file")
            target.unlink()

    def _remove_empty_created_directory(self, directory: Path) -> None:
        if directory == self.root or directory.is_symlink() or not directory.exists():
            return
        self._ensure_stable_parent(directory)
        if not directory.is_dir():
            raise ApprovalError("created workspace parent is no longer a directory")
        try:
            directory.rmdir()
        except OSError as exc:
            if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                raise

    def _ensure_stable_parent(self, target: Path) -> None:
        try:
            resolved_parent = target.parent.resolve(strict=False)
            resolved_parent.relative_to(self.root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ApprovalError("workspace path changed after capture") from exc
        if resolved_parent != target.parent:
            raise ApprovalError("workspace path changed after capture")


class ConsoleApprovalWorkflow:
    """Ask once for approval using content-free change summaries."""

    def __init__(
        self,
        journal: ChangeJournal,
        *,
        input_fn: Callable[[str], object],
        output_fn: Callable[[str], object],
    ) -> None:
        self.journal = journal
        self.input_fn = input_fn
        self.output_fn = output_fn
        self._journal_complete = True

    def request(
        self,
        *,
        changed_files: Sequence[str],
        verification_exit_code: int,
    ) -> bool:
        """Display only safe metadata and accept an explicit ``y`` or ``yes``."""
        self._journal_complete = True
        if isinstance(changed_files, (str, bytes)) or not all(
            isinstance(path, str) and path for path in changed_files
        ):
            raise ApprovalError("successful change list is invalid")
        requested_paths = tuple(sorted(set(changed_files)))
        summaries = self.journal.summaries()
        by_path: dict[str, ChangeSummary] = {}
        for summary in summaries:
            if (
                not isinstance(summary, ChangeSummary)
                or not isinstance(summary.path, str)
                or not summary.path
                or summary.status not in {"modified", "created"}
                or type(summary.additions) is not int
                or summary.additions < 0
                or type(summary.deletions) is not int
                or summary.deletions < 0
                or summary.path in by_path
            ):
                raise ApprovalError("change journal summary is invalid")
            by_path[summary.path] = summary
        if any(path not in by_path for path in requested_paths):
            self._journal_complete = False
            raise ApprovalError("change journal is incomplete")

        self.output_fn("APPROVAL_REQUIRED")
        self.output_fn(f"verification_exit={verification_exit_code}")
        for path in requested_paths:
            summary = by_path[path]
            status = "M" if summary.status == "modified" else "A"
            rendered_path = json.dumps(summary.path, ensure_ascii=True)
            self.output_fn(
                f"{status} {rendered_path} (+{summary.additions}/-{summary.deletions})"
            )

        try:
            response = self.input_fn("Accept verified changes? [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            return False
        return isinstance(response, str) and response.strip().lower() in {"y", "yes"}

    def rollback(self) -> None:
        """Restore the journaled workspace state."""
        self.journal.rollback()
        if not self._journal_complete:
            raise ApprovalError("change journal is incomplete; rollback cannot be guaranteed")

    def commit(self) -> None:
        """Accept the current baseline and prepare the journal for another run."""
        self.journal.commit()
