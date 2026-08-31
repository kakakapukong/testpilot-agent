"""Workspace-confined UTF-8 file operations."""

from __future__ import annotations

import os
import stat
import tempfile
from bisect import insort
from collections.abc import Sequence
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

DEFAULT_PROTECTED_PATTERNS = (
    "tests/**",
    "test/**",
    "test_*.py",
    "**/test_*.py",
    "*_test.py",
    "**/*_test.py",
    "conftest",
    "conftest.py",
    "pytest.py",
    "pytest.ini",
    ".pytest.ini",
    "pytest.toml",
    ".pytest.toml",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
    ".testpilot/traces/**",
)


class WorkspaceError(Exception):
    """A predictable workspace operation failure with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _ChangeRecorder(Protocol):
    def capture(self, path: Path) -> None: ...


class Workspace:
    """Operate on a workspace after resolving user paths and symlinks.

    This prevents model-supplied paths from escaping the workspace or modifying
    protected project files. It is not an OS sandbox against a malicious local
    process: a path can still change between resolution and replacement.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_read_chars: int = 50_000,
        max_write_chars: int | None = None,
        max_results: int = 200,
        max_scanned_entries: int = 5_000,
        max_search_chars: int = 1_000_000,
        protected_patterns: Sequence[str] = DEFAULT_PROTECTED_PATTERNS,
        change_recorder: _ChangeRecorder | None = None,
    ) -> None:
        if (
            not isinstance(max_read_chars, int)
            or isinstance(max_read_chars, bool)
            or max_read_chars < 1
        ):
            raise ValueError("max_read_chars must be a positive integer")
        if max_write_chars is None:
            max_write_chars = max_read_chars
        if (
            not isinstance(max_write_chars, int)
            or isinstance(max_write_chars, bool)
            or max_write_chars < 1
        ):
            raise ValueError("max_write_chars must be a positive integer")
        if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        if (
            not isinstance(max_scanned_entries, int)
            or isinstance(max_scanned_entries, bool)
            or max_scanned_entries < 1
        ):
            raise ValueError("max_scanned_entries must be a positive integer")
        if (
            not isinstance(max_search_chars, int)
            or isinstance(max_search_chars, bool)
            or max_search_chars < 1
        ):
            raise ValueError("max_search_chars must be a positive integer")
        if isinstance(protected_patterns, str) or not isinstance(protected_patterns, Sequence):
            raise TypeError("protected_patterns must be a sequence of strings")
        if not all(isinstance(pattern, str) and pattern for pattern in protected_patterns):
            raise ValueError("protected_patterns must contain non-empty strings")
        self.root = Path(root).resolve(strict=False)
        self.max_read_chars = max_read_chars
        self.max_write_chars = max_write_chars
        self.max_results = max_results
        self.max_scanned_entries = max_scanned_entries
        self.max_search_chars = max_search_chars
        self.protected_patterns = tuple(protected_patterns)
        self.change_recorder = change_recorder

    def read_file(
        self,
        path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read a UTF-8 file under the workspace root."""
        requested_start = start_line is not None
        if start_line is None:
            start_line = 1
        if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1:
            raise WorkspaceError(
                "invalid_line_range",
                "start_line must be an integer of at least 1",
            )
        if end_line is not None and (
            not isinstance(end_line, int) or isinstance(end_line, bool) or end_line < start_line
        ):
            raise WorkspaceError(
                "invalid_line_range",
                "end_line must be an integer greater than or equal to start_line",
            )
        resolved = self._resolve(path)
        if not resolved.exists():
            raise WorkspaceError("file_not_found", f"file does not exist: {path}")
        if not resolved.is_file():
            raise WorkspaceError("not_a_file", f"path is not a file: {path}")

        selected: list[str] = []
        current_line = 1
        newline_count = 0
        saw_content = False
        last_was_newline = False
        previous_was_carriage_return = False
        examined_chars = 0
        reached_eof = False
        range_complete = False
        last_selected_line: int | None = None
        try:
            with resolved.open("r", encoding="utf-8", errors="strict", newline="") as stream:
                while examined_chars < self.max_read_chars:
                    request_size = min(8_192, self.max_read_chars - examined_chars)
                    chunk = stream.read(request_size)
                    if not chunk:
                        reached_eof = True
                        break
                    examined_chars += len(chunk)
                    chunk_ends_at_eof = len(chunk) < request_size
                    if "\x00" in chunk:
                        raise WorkspaceError("binary_file", f"file contains NUL bytes: {path}")
                    saw_content = True
                    for offset, character in enumerate(chunk):
                        is_crlf_continuation = character == "\n" and previous_was_carriage_return
                        line_number = current_line - 1 if is_crlf_continuation else current_line
                        if end_line is not None and line_number > end_line:
                            range_complete = True
                            break
                        in_range = line_number >= start_line and (
                            end_line is None or line_number <= end_line
                        )
                        if in_range:
                            selected.append(character)
                            last_selected_line = line_number
                        if character == "\r":
                            newline_count += 1
                            current_line += 1
                            last_was_newline = True
                            previous_was_carriage_return = True
                        elif character == "\n":
                            if not previous_was_carriage_return:
                                newline_count += 1
                                current_line += 1
                            last_was_newline = True
                            previous_was_carriage_return = False
                        else:
                            last_was_newline = False
                            previous_was_carriage_return = False
                        if end_line is not None and line_number == end_line and character == "\n":
                            range_complete = True
                            if chunk_ends_at_eof and offset == len(chunk) - 1:
                                reached_eof = True
                            break
                    if range_complete:
                        break
                    if chunk_ends_at_eof:
                        reached_eof = True
                        break
        except WorkspaceError:
            raise
        except UnicodeDecodeError as exc:
            raise WorkspaceError("binary_file", f"file is not valid UTF-8 text: {path}") from exc
        except OSError as exc:
            raise WorkspaceError("read_failed", f"could not read {path}: {exc}") from exc

        total_lines = (
            (0 if not saw_content else newline_count + (0 if last_was_newline else 1))
            if reached_eof
            else None
        )
        if total_lines is not None and (
            (requested_start and start_line > total_lines)
            or (end_line is not None and end_line > total_lines)
        ):
            raise WorkspaceError(
                "line_range_out_of_bounds",
                f"requested lines exceed the file's {total_lines} lines",
            )
        if range_complete and end_line is not None:
            actual_end = end_line
        elif total_lines is not None:
            actual_end = end_line if end_line is not None else total_lines
        else:
            actual_end = last_selected_line if last_selected_line is not None else start_line
        return {
            "path": self._relative(resolved),
            "content": "".join(selected),
            "start_line": start_line,
            "end_line": actual_end,
            "total_lines": total_lines,
            "total_lines_exact": total_lines is not None,
            "truncated": not reached_eof and not range_complete,
        }

    def list_files(self, path: str = ".", *, glob: str | None = None) -> dict[str, Any]:
        """List a deterministic, bounded set of files beneath *path*."""
        base = self._resolve(path, allow_root=True)
        if not base.exists():
            raise WorkspaceError("path_not_found", f"path does not exist: {path}")
        if not base.is_dir():
            raise WorkspaceError("not_a_directory", f"path is not a directory: {path}")
        pattern = self._validate_glob(glob)
        smallest: list[str] = []
        total = 0
        candidates, scanned_entries, scan_truncated = self._iter_files(base, pattern)
        for _, relative in candidates:
            total += 1
            insort(smallest, relative)
            if len(smallest) > self.max_results:
                smallest.pop()
        return {
            "path": "." if base == self.root else self._relative(base),
            "files": smallest,
            "scanned_entries": scanned_entries,
            "scan_truncated": scan_truncated,
            "truncated": total > self.max_results or scan_truncated,
        }

    def search_text(
        self,
        query: str,
        path: str = ".",
        *,
        glob: str | None = None,
    ) -> dict[str, Any]:
        """Search UTF-8 files for a literal substring with bounded output."""
        if not isinstance(query, str) or not query:
            raise WorkspaceError("invalid_query", "query must be a non-empty string")
        base = self._resolve(path, allow_root=True)
        if not base.exists():
            raise WorkspaceError("path_not_found", f"path does not exist: {path}")
        pattern = self._validate_glob(glob)

        scanned_entries: int
        scan_truncated: bool
        if base.is_file():
            relative = self._relative(base)
            candidates = (
                [(base, relative)] if glob is None or PurePosixPath(relative).match(pattern) else []
            )
            scanned_entries = 1
            scan_truncated = False
        elif base.is_dir():
            candidates, scanned_entries, scan_truncated = self._iter_files(base, pattern)
        else:
            raise WorkspaceError("invalid_path", f"path cannot be searched: {path}")

        matches: list[tuple[str, int, str]] = []
        skipped_paths: list[str] = []
        truncated_files: list[str] = []
        match_count = 0
        skipped_count = 0
        truncated_file_count = 0
        search_chars = 0
        search_chars_truncated = False
        for candidate, relative in candidates:
            remaining_chars = self.max_search_chars - search_chars
            if remaining_chars == 0:
                search_chars_truncated = True
                break
            try:
                checked_content, file_truncated = self._read_search_content(
                    candidate,
                    relative,
                    min(self.max_read_chars, remaining_chars),
                )
            except WorkspaceError as exc:
                if exc.code == "binary_file":
                    skipped_count += 1
                    self._insert_bounded(skipped_paths, relative)
                    continue
                raise
            if file_truncated:
                truncated_file_count += 1
                self._insert_bounded(truncated_files, relative)
            search_chars += len(checked_content)
            for line_number, text in enumerate(checked_content.splitlines(), start=1):
                if query not in text:
                    continue
                match_count += 1
                insort(matches, (relative, line_number, text))
                if len(matches) > self.max_results:
                    matches.pop()
            if search_chars == self.max_search_chars:
                search_chars_truncated = True
                break

        matches_truncated = match_count > self.max_results
        skipped_truncated = skipped_count > self.max_results
        truncated_files_truncated = truncated_file_count > self.max_results

        return {
            "query": query,
            "path": "." if base == self.root else self._relative(base),
            "scanned_entries": scanned_entries,
            "scan_truncated": scan_truncated,
            "search_chars": search_chars,
            "search_chars_truncated": search_chars_truncated,
            "matches": [
                {"path": relative, "line": line_number, "text": text}
                for relative, line_number, text in matches
            ],
            "matches_truncated": matches_truncated,
            "skipped": [
                {"path": relative, "error_code": "binary_file"} for relative in skipped_paths
            ],
            "skipped_truncated": skipped_truncated,
            "truncated_files": truncated_files,
            "truncated_files_truncated": truncated_files_truncated,
            "truncated": (
                matches_truncated
                or skipped_truncated
                or bool(truncated_files)
                or truncated_files_truncated
                or scan_truncated
                or search_chars_truncated
            ),
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Atomically write UTF-8 text beneath the workspace root."""
        if not isinstance(content, str):
            raise WorkspaceError("invalid_content", "content must be a string")
        if "\x00" in content:
            raise WorkspaceError("invalid_content", "content must not contain NUL characters")
        if len(content) > self.max_write_chars:
            raise WorkspaceError(
                "file_too_large",
                f"content exceeds the {self.max_write_chars}-character write limit",
            )
        resolved = self._resolve(path)
        self._assert_not_protected(resolved)
        encoded = content.encode("utf-8")
        existing_mode: int | None = None
        if resolved.is_file():
            try:
                with resolved.open("rb") as existing:
                    if existing.read(len(encoded) + 1) == encoded:
                        return {"path": self._relative(resolved), "changed": False}
                existing_mode = stat.S_IMODE(resolved.stat().st_mode)
            except OSError as exc:
                raise WorkspaceError("write_failed", f"could not inspect {path}: {exc}") from exc
        if self.change_recorder is not None:
            try:
                self.change_recorder.capture(resolved)
            except Exception as exc:
                raise WorkspaceError(
                    "snapshot_failed",
                    "could not snapshot file before writing",
                ) from exc
        temporary_path: Path | None = None
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_inside(resolved.parent.resolve(strict=False))
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=resolved.parent,
                prefix=f".{resolved.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            if existing_mode is not None:
                try:
                    os.chmod(temporary_path, existing_mode)
                except OSError:
                    if os.name != "nt":
                        raise
            # Resolve again immediately before replacing in case a parent path
            # was swapped for a symlink while the temporary file was prepared.
            resolved = self._resolve(path)
            self._assert_not_protected(resolved)
            os.replace(temporary_path, resolved)
            temporary_path = None
        except OSError as exc:
            raise WorkspaceError("write_failed", f"could not write {path}: {exc}") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return {"path": self._relative(resolved), "changed": True}

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
    ) -> dict[str, Any]:
        """Atomically replace exactly one occurrence of *old_text*."""
        if not isinstance(old_text, str) or not isinstance(new_text, str) or not old_text:
            raise WorkspaceError(
                "invalid_edit",
                "old_text must be non-empty and both texts strings",
            )
        if old_text == new_text:
            raise WorkspaceError("no_change", "old_text and new_text are identical")
        self._assert_not_protected(self._resolve(path))
        read = self.read_file(path)
        if read["truncated"]:
            raise WorkspaceError("file_too_large", "file is too large for a safe exact edit")
        content = read["content"]
        occurrences = content.count(old_text)
        if occurrences == 0:
            raise WorkspaceError("missing_match", "old_text was not found")
        if occurrences > 1:
            raise WorkspaceError("ambiguous_edit", "old_text occurs more than once")
        return self.write_file(path, content.replace(old_text, new_text, 1))

    def _resolve(self, path: str, *, allow_root: bool = False) -> Path:
        if not isinstance(path, str) or not path:
            raise WorkspaceError("invalid_path", "path must be a non-empty relative path")
        if os.name == "nt" and any(
            component not in {"", ".", ".."} and component.endswith((" ", "."))
            for component in path.replace("\\", "/").split("/")
        ):
            raise WorkspaceError(
                "invalid_path",
                "Windows path components must not end with a space or dot",
            )
        supplied = Path(path)
        if supplied.is_absolute() or bool(supplied.anchor) or bool(supplied.drive):
            raise WorkspaceError("absolute_path_not_allowed", "absolute paths are not allowed")
        try:
            candidate = (self.root / supplied).resolve(strict=False)
        except (OSError, ValueError) as exc:
            raise WorkspaceError("invalid_path", f"invalid workspace path: {path!r}") from exc
        self._ensure_inside(candidate)
        if not allow_root and candidate == self.root:
            raise WorkspaceError("invalid_path", "path must name a file")
        return candidate

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _insert_bounded(self, values: list[str], value: str) -> None:
        """Keep only the lexicographically smallest configured result paths."""
        insort(values, value)
        if len(values) > self.max_results:
            values.pop()

    def _read_search_content(self, resolved: Path, path: str, limit: int) -> tuple[str, bool]:
        """Read at most *limit* characters for search, without probing past it."""
        try:
            with resolved.open("r", encoding="utf-8", errors="strict", newline="") as stream:
                content = stream.read(limit)
        except UnicodeDecodeError as exc:
            raise WorkspaceError("binary_file", f"file is not valid UTF-8 text: {path}") from exc
        except OSError as exc:
            raise WorkspaceError("read_failed", f"could not read {path}: {exc}") from exc
        if "\x00" in content:
            raise WorkspaceError("binary_file", f"file contains NUL bytes: {path}")
        return content, len(content) == limit

    def _ensure_inside(self, candidate: Path) -> None:
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError("path_outside_workspace", "path escapes workspace root") from exc

    def _assert_not_protected(self, resolved: Path) -> None:
        """Reject canonical workspace paths reserved for tests and configuration."""
        relative = self._relative(resolved)
        if is_protected_relative_path(
            relative,
            self.protected_patterns,
            case_insensitive=os.name == "nt",
        ):
            raise WorkspaceError("protected_path", f"path is protected: {relative}")
        # Atomic replacement changes only this directory entry, not hard-link peers.

    def _validate_glob(self, pattern: str | None) -> str:
        if pattern is None:
            return "**/*"
        if not isinstance(pattern, str) or not pattern:
            raise WorkspaceError("invalid_glob", "glob must be a non-empty relative pattern")
        parsed = Path(pattern)
        if parsed.is_absolute() or parsed.anchor or parsed.drive or ".." in parsed.parts:
            raise WorkspaceError("invalid_glob", "glob must not escape the selected path")
        return pattern

    def _iter_files(self, base: Path, pattern: str) -> tuple[list[tuple[Path, str]], int, bool]:
        """Walk at most the configured number of directory entries."""
        files: list[tuple[Path, str]] = []
        scanned_entries = 0
        pending = [base]
        visited = {base.resolve(strict=False)}
        try:
            while pending:
                directory = pending.pop()
                with os.scandir(directory) as entries:
                    for entry in entries:
                        scanned_entries += 1
                        candidate = Path(entry.path)
                        resolved = candidate.resolve(strict=False)
                        inside_workspace = True
                        try:
                            resolved.relative_to(self.root)
                        except ValueError:
                            inside_workspace = False

                        if inside_workspace:
                            relative_to_base = candidate.relative_to(base).as_posix()
                            if resolved.is_file() and _glob_matches(relative_to_base, pattern):
                                files.append((resolved, self._relative(resolved)))
                            if _is_traversable_directory(entry):
                                canonical_directory = resolved
                                if canonical_directory not in visited:
                                    visited.add(canonical_directory)
                                    pending.append(candidate)

                        if scanned_entries == self.max_scanned_entries:
                            return files, scanned_entries, True
        except (OSError, ValueError) as exc:
            raise WorkspaceError(
                "invalid_glob",
                f"could not evaluate glob {pattern!r}: {exc}",
            ) from exc
        return files, scanned_entries, False


def _matches_protected_pattern(path: str, pattern: str) -> bool:
    """Match trailing ``/**`` as an actual recursive directory subtree."""
    if pattern.endswith("/**") and len(pattern) > 3:
        directory_pattern = pattern[:-3]
        parts = path.split("/")
        for length in range(1, len(parts)):
            if PurePosixPath("/".join(parts[:length])).match(directory_pattern):
                return True
    return PurePosixPath(path).match(pattern)


def is_protected_relative_path(
    path: str,
    patterns: Sequence[str],
    *,
    case_insensitive: bool = False,
) -> bool:
    """Apply all verification-asset patterns with recursive subtree semantics."""
    candidate = path.lower() if case_insensitive else path
    return any(
        _matches_protected_pattern(
            candidate,
            pattern.lower() if case_insensitive else pattern,
        )
        for pattern in patterns
    )


def _is_traversable_directory(entry: os.DirEntry[str]) -> bool:
    if not entry.is_dir(follow_symlinks=False):
        return False
    if os.name != "nt":
        return True
    attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    return not bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _glob_matches(path: str, pattern: str) -> bool:
    """Match glob segments while treating ``**`` as the only recursive wildcard."""
    path_parts = PurePosixPath(path).parts
    pattern_parts = PurePosixPath(pattern.replace("\\", "/")).parts

    @cache
    def matches(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        token = pattern_parts[pattern_index]
        if token == "**":
            return matches(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and matches(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatchcase(path_parts[path_index], token)
            and matches(path_index + 1, pattern_index + 1)
        )

    return matches(0, 0)
