"""Reversible snapshots for workspace file changes."""

from __future__ import annotations

import errno
import os
import stat
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


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

    def __init__(self, root: str | Path) -> None:
        try:
            self.root = Path(root).resolve(strict=False)
        except OSError as exc:
            raise ApprovalError("could not initialize workspace change journal") from exc
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
                    original = stream.read()
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

    def _normalize(self, path: Path) -> tuple[Path, Path]:
        try:
            supplied = Path(path)
            target = (
                supplied.resolve(strict=False)
                if supplied.is_absolute()
                else (self.root / supplied).resolve(strict=False)
            )
            relative = target.relative_to(self.root)
        except (OSError, ValueError) as exc:
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
            resolved = target.resolve(strict=False)
            if resolved != target:
                raise ApprovalError("workspace path changed after capture")
        if not target.exists():
            return b""
        if not target.is_file():
            raise ApprovalError("workspace path is no longer a file")
        return target.read_bytes()

    @staticmethod
    def _line_changes(before: bytes, after: bytes) -> tuple[int, int]:
        matcher = SequenceMatcher(
            None,
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            autojunk=False,
        )
        additions = 0
        deletions = 0
        for operation, before_start, before_end, after_start, after_end in matcher.get_opcodes():
            if operation in {"replace", "insert"}:
                additions += after_end - after_start
            if operation in {"replace", "delete"}:
                deletions += before_end - before_start
        return additions, deletions

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
                temporary.write(snapshot.original)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
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
        except (OSError, ValueError) as exc:
            raise ApprovalError("workspace path changed after capture") from exc
        if resolved_parent != target.parent:
            raise ApprovalError("workspace path changed after capture")
