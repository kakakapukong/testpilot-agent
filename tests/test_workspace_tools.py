import os
import stat
from pathlib import Path
from typing import Any, Self

import pytest

import testpilot.workspace as workspace_module
from testpilot.registry import ToolRegistry
from testpilot.tools import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchTextTool,
    WriteFileTool,
)
from testpilot.workspace import Workspace, WorkspaceError


@pytest.mark.parametrize("path", ["../outside.txt", "nested/../../outside.txt"])
def test_workspace_rejects_parent_traversal(tmp_path: Path, path: str) -> None:
    workspace = Workspace(tmp_path / "repo")

    with pytest.raises(WorkspaceError) as raised:
        workspace.read_file(path)

    assert raised.value.code == "path_outside_workspace"


def test_workspace_rejects_absolute_outside_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside.txt"
    root.mkdir()
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(WorkspaceError) as raised:
        Workspace(root).read_file(str(outside.resolve()))

    assert raised.value.code == "absolute_path_not_allowed"


@pytest.mark.skipif(os.name != "nt", reason="drive-relative paths are Windows-specific")
def test_workspace_rejects_drive_relative_path(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError) as raised:
        Workspace(tmp_path).read_file("C:outside.txt")

    assert raised.value.code == "absolute_path_not_allowed"


def test_workspace_rejects_symlink_escape_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable on this system")

    with pytest.raises(WorkspaceError) as raised:
        Workspace(root).read_file("link/secret.txt")

    assert raised.value.code == "path_outside_workspace"


def test_read_file_returns_utf8_inclusive_line_range(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "notes.txt").write_bytes("第一行\n第二行\n第三行\n".encode())

    result = Workspace(root).read_file("notes.txt", start_line=2, end_line=3)

    assert result == {
        "path": "notes.txt",
        "content": "第二行\n第三行\n",
        "start_line": 2,
        "end_line": 3,
        "total_lines": 3,
        "total_lines_exact": True,
        "truncated": False,
    }


@pytest.mark.parametrize(
    ("start_line", "end_line", "code"),
    [
        (0, None, "invalid_line_range"),
        (True, None, "invalid_line_range"),
        (3, 2, "invalid_line_range"),
        (4, None, "line_range_out_of_bounds"),
        (1, 4, "line_range_out_of_bounds"),
    ],
)
def test_read_file_validates_line_range(
    tmp_path: Path,
    start_line: Any,
    end_line: Any,
    code: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    with pytest.raises(WorkspaceError) as raised:
        Workspace(root).read_file(
            "notes.txt",
            start_line=start_line,
            end_line=end_line,
        )

    assert raised.value.code == code


def test_read_file_is_bounded_and_marks_truncation(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "large.txt").write_text("abcdefgh", encoding="utf-8")

    result = Workspace(root, max_read_chars=5).read_file("large.txt")

    assert result["content"] == "abcde"
    assert result["truncated"] is True


def test_read_file_never_requests_characters_past_its_hard_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "large.txt"
    target.write_text("abcdefgh", encoding="utf-8")

    class BudgetStream:
        def __init__(self) -> None:
            self.position = 0
            self.requested = 0

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, size: int) -> str:
            self.requested += size
            if self.requested > 5:
                raise AssertionError("read requested characters past the configured budget")
            content = "abcdefgh"
            chunk = content[self.position : self.position + size]
            self.position += len(chunk)
            return chunk

    stream = BudgetStream()
    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: stream)

    result = Workspace(tmp_path, max_read_chars=5).read_file("large.txt")

    assert stream.requested == 5
    assert result["content"] == "abcde"
    assert result["total_lines"] is None
    assert result["total_lines_exact"] is False
    assert result["truncated"] is True


@pytest.mark.parametrize("contents", [b"before\x00after", b"valid\n\xffinvalid"])
def test_read_file_rejects_binary_or_invalid_utf8(tmp_path: Path, contents: bytes) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "binary.dat").write_bytes(contents)

    with pytest.raises(WorkspaceError) as raised:
        Workspace(root).read_file("binary.dat")

    assert raised.value.code == "binary_file"


def test_list_files_is_sorted_globbed_and_limited(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "nested").mkdir(parents=True)
    for relative in ("z.py", "a.py", "nested/b.py", "nested/skip.txt"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")

    result = Workspace(root, max_results=2).list_files(".", glob="**/*.py")

    assert result == {
        "path": ".",
        "files": ["a.py", "nested/b.py"],
        "scanned_entries": 5,
        "scan_truncated": False,
        "truncated": True,
    }


def test_list_files_only_returns_files_under_selected_directory(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "nested" / "deeper").mkdir(parents=True)
    (root / "top.txt").write_text("top", encoding="utf-8")
    (root / "nested" / "deeper" / "item.txt").write_text("item", encoding="utf-8")

    result = Workspace(root).list_files("nested")

    assert result["files"] == ["nested/deeper/item.txt"]


@pytest.mark.parametrize("pattern", ["../*.py", "../../**/*"])
def test_list_files_rejects_escaping_glob(tmp_path: Path, pattern: str) -> None:
    with pytest.raises(WorkspaceError) as raised:
        Workspace(tmp_path).list_files(".", glob=pattern)

    assert raised.value.code == "invalid_glob"


@pytest.mark.skipif(os.name != "nt", reason="drive-relative globs are Windows-specific")
def test_list_files_rejects_drive_relative_glob(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError) as raised:
        Workspace(tmp_path).list_files(".", glob="C:*.py")

    assert raised.value.code == "invalid_glob"


def test_list_files_counts_directories_against_scan_budget(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "folder").mkdir(parents=True)
    (root / "folder" / "item.txt").write_text("item", encoding="utf-8")

    result = Workspace(root, max_scanned_entries=1).list_files(".")

    assert result["files"] == []
    assert result["scanned_entries"] == 1
    assert result["scan_truncated"] is True
    assert result["truncated"] is True


def test_glob_scan_budget_counts_nonmatching_directory_entries(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "folder").mkdir(parents=True)
    (root / "folder" / "ordinary.txt").write_text("ordinary", encoding="utf-8")

    result = Workspace(root, max_scanned_entries=1).list_files(".", glob="**/*.rare")

    assert result["files"] == []
    assert result["scanned_entries"] == 1
    assert result["scan_truncated"] is True
    assert result["truncated"] is True


def test_list_files_does_not_request_a_candidate_past_scan_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    candidate = root / "a.txt"
    candidate.write_text("content", encoding="utf-8")

    real_scandir = os.scandir

    class OneCandidate:
        def __init__(self, path: str | os.PathLike[str]) -> None:
            self.inner = real_scandir(path)
            self.calls = 0

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args
            self.inner.close()

        def __iter__(self) -> "OneCandidate":
            return self

        def __next__(self) -> os.DirEntry[str]:
            self.calls += 1
            if self.calls == 1:
                return next(self.inner)
            raise AssertionError("scan requested an entry past its budget")

    iterators: list[OneCandidate] = []

    def limited_scandir(path: str | os.PathLike[str]) -> OneCandidate:
        iterator = OneCandidate(path)
        iterators.append(iterator)
        return iterator

    monkeypatch.setattr(workspace_module.os, "scandir", limited_scandir)

    result = Workspace(root, max_scanned_entries=1).list_files(".")

    assert result["files"] == ["a.txt"]
    assert result["scanned_entries"] == 1
    assert result["scan_truncated"] is True
    assert len(iterators) == 1
    assert iterators[0].calls == 1


def test_search_text_is_literal_reports_lines_and_respects_limit(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("needle.*\nneedle.* twice\n", encoding="utf-8")
    (root / "b.py").write_text("needleZZ\nneedle.*\n", encoding="utf-8")

    result = Workspace(root, max_results=2).search_text("needle.*", ".", glob="*.py")

    assert result["matches"] == [
        {"path": "a.py", "line": 1, "text": "needle.*"},
        {"path": "a.py", "line": 2, "text": "needle.* twice"},
    ]
    assert result["matches_truncated"] is True
    assert result["truncated"] is True
    assert result["skipped"] == []


def test_search_text_is_deterministic_by_posix_path_then_line(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    files = {
        "z.py": "needle z\n",
        "nested/m.py": "before\nneedle m first\nneedle m second\n",
        "a.py": "before\nneedle a\n",
    }
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    result = Workspace(root).search_text("needle", ".", glob="**/*.py")

    assert result["matches"] == [
        {"path": "a.py", "line": 2, "text": "needle a"},
        {"path": "nested/m.py", "line": 2, "text": "needle m first"},
        {"path": "nested/m.py", "line": 3, "text": "needle m second"},
        {"path": "z.py", "line": 1, "text": "needle z"},
    ]


def test_search_text_skips_binary_files_with_structured_reason(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("find me", encoding="utf-8")
    (root / "b.dat").write_bytes(b"find\x00me")

    result = Workspace(root).search_text("find", ".")

    assert result["matches"] == [{"path": "a.txt", "line": 1, "text": "find me"}]
    assert result["skipped"] == [{"path": "b.dat", "error_code": "binary_file"}]


def test_search_marks_skipped_metadata_overflow_and_keeps_smallest_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("z.dat", "m.dat", "a.dat"):
        (root / name).write_bytes(b"binary\x00data")

    result = Workspace(root, max_results=2).search_text("needle", ".")

    assert result["skipped"] == [
        {"path": "a.dat", "error_code": "binary_file"},
        {"path": "m.dat", "error_code": "binary_file"},
    ]
    assert result["skipped_truncated"] is True
    assert result["truncated"] is True


def test_search_marks_result_truncated_when_a_file_read_is_bounded(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "large.txt").write_text("prefix\nneedle after bound\n", encoding="utf-8")

    result = Workspace(root, max_read_chars=6).search_text("needle", ".")

    assert result["matches"] == []
    assert result["truncated_files"] == ["large.txt"]
    assert result["truncated"] is True


def test_search_marks_truncated_file_metadata_overflow_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("z.txt", "m.txt", "a.txt"):
        (root / name).write_text("long content", encoding="utf-8")

    result = Workspace(root, max_read_chars=2, max_results=2).search_text("needle", ".")

    assert result["truncated_files"] == ["a.txt", "m.txt"]
    assert result["truncated_files_truncated"] is True
    assert result["truncated"] is True


def test_search_stops_at_entry_scan_budget_and_reports_it(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (root / name).write_text("needle", encoding="utf-8")

    result = Workspace(root, max_scanned_entries=1).search_text("needle", ".")

    assert result["scanned_entries"] == 1
    assert result["scan_truncated"] is True
    assert len(result["matches"]) <= 1
    assert result["truncated"] is True


def test_search_stops_at_total_character_budget(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "large.txt").write_text("abcdeneedle", encoding="utf-8")

    result = Workspace(root, max_search_chars=5).search_text("needle", ".")

    assert result["matches"] == []
    assert result["search_chars"] == 5
    assert result["search_chars_truncated"] is True
    assert result["truncated_files"] == ["large.txt"]
    assert result["truncated"] is True


def test_search_reads_no_more_than_its_character_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "large.txt").write_text("abcde", encoding="utf-8")
    read_sizes: list[int] = []

    class SpyStream:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int = -1) -> str:
            assert 0 <= size <= 5
            read_sizes.append(size)
            return "abcde"

    def spy_open(self: Path, *args: object, **kwargs: object) -> SpyStream:
        assert self == root / "large.txt"
        return SpyStream()

    monkeypatch.setattr(Path, "open", spy_open)

    result = Workspace(root, max_search_chars=5).search_text("needle", ".")

    assert read_sizes == [5]
    assert result["search_chars"] == 5
    assert result["search_chars_truncated"] is True


def test_searching_one_file_uses_the_default_glob(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "note.txt").write_text("needle", encoding="utf-8")

    result = Workspace(root).search_text("needle", "note.txt")

    assert result["matches"] == [{"path": "note.txt", "line": 1, "text": "needle"}]
    assert result["scanned_entries"] == 1
    assert result["truncated"] is False


def test_searching_one_file_honors_a_matching_glob(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "note.txt").write_text("needle", encoding="utf-8")

    result = Workspace(root).search_text("needle", "note.txt", glob="*.txt")

    assert result["matches"] == [{"path": "note.txt", "line": 1, "text": "needle"}]
    assert result["scanned_entries"] == 1
    assert result["truncated"] is False


def test_searching_one_file_honors_a_non_matching_glob(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "note.txt").write_text("needle", encoding="utf-8")

    result = Workspace(root).search_text("needle", "note.txt", glob="*.py")

    assert result["matches"] == []
    assert result["scanned_entries"] == 1
    assert result["truncated"] is False


def test_write_file_creates_nested_directories_and_uses_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    workspace = Workspace(root)
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        target_path = Path(target)
        assert source_path.exists()
        assert source_path.parent == target_path.parent
        replace_calls.append((source_path, target_path))
        real_replace(source, target)

    monkeypatch.setattr("testpilot.workspace.os.replace", recording_replace)

    result = workspace.write_file("src/new.py", "answer = 42\n")

    assert result == {"path": "src/new.py", "changed": True}
    assert (root / "src" / "new.py").read_text(encoding="utf-8") == "answer = 42\n"
    assert len(replace_calls) == 1
    assert list((root / "src").iterdir()) == [root / "src" / "new.py"]


def test_failed_atomic_replace_preserves_existing_file_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_text("old\n", encoding="utf-8")

    def failing_replace(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("testpilot.workspace.os.replace", failing_replace)

    with pytest.raises(WorkspaceError) as raised:
        Workspace(root).write_file("app.py", "new\n")

    assert raised.value.code == "write_failed"
    assert target.read_text(encoding="utf-8") == "old\n"
    assert list(root.iterdir()) == [target]


def test_write_file_identical_content_is_noop_without_touching_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"value = 1\n")
    before_mtime = target.stat().st_mtime_ns

    def unexpected_replace(source: object, destination: object) -> None:
        pytest.fail("identical content must not reach os.replace")

    class UnexpectedRecorder:
        def capture(self, path: Path) -> None:
            pytest.fail("identical content must not be captured")

    monkeypatch.setattr("testpilot.workspace.os.replace", unexpected_replace)

    result = Workspace(root, change_recorder=UnexpectedRecorder()).write_file(
        "app.py",
        "value = 1\n",
    )

    assert result == {"path": "app.py", "changed": False}
    assert target.read_bytes() == b"value = 1\n"
    assert target.stat().st_mtime_ns == before_mtime


def test_write_file_captures_resolved_path_before_creating_parent(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    captured: list[Path] = []

    class Recorder:
        def capture(self, path: Path) -> None:
            assert not path.parent.exists()
            captured.append(path)

    result = Workspace(root, change_recorder=Recorder()).write_file(
        "nested/app.py",
        "value = 1\n",
    )

    assert result == {"path": "nested/app.py", "changed": True}
    assert captured == [(root / "nested/app.py").resolve()]


def test_snapshot_failure_does_not_write_the_target(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"old\n")

    class FailingRecorder:
        def capture(self, path: Path) -> None:
            assert path == target.resolve()
            raise RuntimeError("sensitive recorder detail")

    with pytest.raises(WorkspaceError) as raised:
        Workspace(root, change_recorder=FailingRecorder()).write_file("app.py", "new\n")

    assert raised.value.code == "snapshot_failed"
    assert raised.value.message == "could not snapshot file before writing"
    assert "sensitive recorder detail" not in str(raised.value)
    assert target.read_bytes() == b"old\n"
    assert list(root.iterdir()) == [target]


def test_failed_fsync_cleans_hidden_temp_file_and_new_parent_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    def failing_fsync(file_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr("testpilot.workspace.os.fsync", failing_fsync)

    with pytest.raises(WorkspaceError) as raised:
        Workspace(root).write_file("nested/new.py", "new\n")

    assert raised.value.code == "write_failed"
    assert not (root / "nested").exists()


def test_write_file_rejects_a_target_changed_after_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    original_parent = root / "original"
    switched_parent = root / "switched"
    original_parent.mkdir(parents=True)
    switched_parent.mkdir()
    original_target = original_parent / "app.py"
    switched_target = switched_parent / "app.py"
    original_target.write_bytes(b"original\n")
    switched_target.write_bytes(b"switched\n")

    class SwitchingRecorder:
        switched = False

        def capture(self, path: Path) -> None:
            assert path == original_target.resolve()
            self.switched = True

    recorder = SwitchingRecorder()
    workspace = Workspace(root, change_recorder=recorder)
    real_resolve = workspace._resolve

    def switching_resolve(path: str, *, allow_root: bool = False) -> Path:
        if path == "alias/app.py":
            # Models a workspace-internal parent symlink switched during capture.
            return (switched_target if recorder.switched else original_target).resolve()
        return real_resolve(path, allow_root=allow_root)

    monkeypatch.setattr(workspace, "_resolve", switching_resolve)

    with pytest.raises(WorkspaceError) as raised:
        workspace.write_file("alias/app.py", "replacement\n")

    assert raised.value.code == "path_changed_after_snapshot"
    assert raised.value.message == "workspace path changed after snapshot"
    assert original_target.read_bytes() == b"original\n"
    assert switched_target.read_bytes() == b"switched\n"
    assert list(original_parent.glob(".app.py.*.tmp")) == []


def test_write_file_rejects_content_over_default_write_limit(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, max_read_chars=3)

    with pytest.raises(WorkspaceError) as raised:
        workspace.write_file("large.txt", "four")

    assert raised.value.code == "file_too_large"
    assert not (tmp_path / "large.txt").exists()


def test_write_file_has_independently_configurable_write_limit(tmp_path: Path) -> None:
    result = Workspace(tmp_path, max_read_chars=3, max_write_chars=5).write_file(
        "allowed.txt",
        "12345",
    )

    assert result["changed"] is True


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_new.py",
        "test_local.py",
        "nested/test_nested.py",
        "conftest",
        "conftest.py",
        "pytest.py",
        "pytest.ini",
        "pyproject.toml",
        "tox.ini",
        "setup.cfg",
        ".testpilot/traces/audit.jsonl",
    ],
)
def test_write_file_protects_test_and_project_configuration_paths(
    tmp_path: Path,
    path: str,
) -> None:
    with pytest.raises(WorkspaceError) as raised:
        Workspace(tmp_path).write_file(path, "changed\n")

    assert raised.value.code == "protected_path"


def test_write_file_protects_the_entire_deep_tests_subtree(tmp_path: Path) -> None:
    target = tmp_path / "tests" / "unit" / "fixtures" / "sample.txt"

    with pytest.raises(WorkspaceError) as raised:
        Workspace(tmp_path).write_file("tests/unit/fixtures/sample.txt", "changed\n")

    assert raised.value.code == "protected_path"
    assert not target.exists()


@pytest.mark.parametrize(
    "path",
    [
        "test/unit/fixtures/sample.txt",
        "package/test/unit/fixtures/sample.txt",
        "calculator_test.py",
        "package/unit/calculator_test.py",
    ],
)
def test_write_file_protects_other_common_pytest_locations(
    tmp_path: Path,
    path: str,
) -> None:
    with pytest.raises(WorkspaceError) as raised:
        Workspace(tmp_path).write_file(path, "changed\n")

    assert raised.value.code == "protected_path"
    assert not (tmp_path / path).exists()


@pytest.mark.parametrize("path", [".pytest.ini", "pytest.toml", ".pytest.toml"])
def test_write_file_protects_pytest_configuration_variants(
    tmp_path: Path,
    path: str,
) -> None:
    with pytest.raises(WorkspaceError) as raised:
        Workspace(tmp_path).write_file(path, "[pytest]\n")

    assert raised.value.code == "protected_path"
    assert not (tmp_path / path).exists()


def test_edit_file_protects_test_and_project_configuration_paths(tmp_path: Path) -> None:
    target = tmp_path / "pyproject.toml"
    target.write_text("old\n", encoding="utf-8")

    with pytest.raises(WorkspaceError) as raised:
        Workspace(tmp_path).edit_file("pyproject.toml", "old", "new")

    assert raised.value.code == "protected_path"
    assert target.read_text(encoding="utf-8") == "old\n"


def test_protected_patterns_can_be_explicitly_disabled(tmp_path: Path) -> None:
    target = tmp_path / "pyproject.toml"
    workspace = Workspace(tmp_path, protected_patterns=())

    workspace.write_file("pyproject.toml", "old\n")
    result = workspace.edit_file("pyproject.toml", "old", "new")

    assert result == {"path": "pyproject.toml", "changed": True}
    assert target.read_text(encoding="utf-8") == "new\n"


@pytest.mark.parametrize("operation", ["read", "search", "write", "edit", "list"])
def test_checkpoint_tree_is_host_private_for_every_workspace_operation(
    tmp_path: Path,
    operation: str,
) -> None:
    private = tmp_path / ".testpilot" / "checkpoints"
    private.mkdir(parents=True)
    (private / "0123456789abcdef.json").write_text(
        '{"task":"private"}', encoding="utf-8"
    )
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceError) as caught:
        if operation == "read":
            workspace.read_file(".testpilot/checkpoints/0123456789abcdef.json")
        elif operation == "search":
            workspace.search_text("private", ".testpilot/checkpoints")
        elif operation == "write":
            workspace.write_file(".testpilot/checkpoints/new.json", "{}")
        elif operation == "edit":
            workspace.edit_file(
                ".testpilot/checkpoints/0123456789abcdef.json",
                "private",
                "changed",
            )
        else:
            workspace.list_files(".testpilot/checkpoints")

    assert caught.value.code == "private_path"


@pytest.mark.parametrize("operation", ["read", "search", "write", "edit", "list"])
def test_memory_tree_is_host_private_for_every_workspace_operation(
    tmp_path: Path,
    operation: str,
) -> None:
    private = tmp_path / ".testpilot" / "memories"
    private.mkdir(parents=True)
    (private / "entries.jsonl").write_text('{"problem":"private"}\n', encoding="utf-8")
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceError) as caught:
        if operation == "read":
            workspace.read_file(".testpilot/memories/entries.jsonl")
        elif operation == "search":
            workspace.search_text("private", ".testpilot/memories")
        elif operation == "write":
            workspace.write_file(".testpilot/memories/new.jsonl", "{}\n")
        elif operation == "edit":
            workspace.edit_file(
                ".testpilot/memories/entries.jsonl",
                "private",
                "changed",
            )
        else:
            workspace.list_files(".testpilot/memories")

    assert caught.value.code == "private_path"


def test_root_listing_and_search_prune_checkpoint_tree(tmp_path: Path) -> None:
    private = tmp_path / ".testpilot" / "checkpoints"
    private.mkdir(parents=True)
    (private / "0123456789abcdef.json").write_text("sentinel", encoding="utf-8")
    (tmp_path / "app.py").write_text("sentinel\n", encoding="utf-8")
    workspace = Workspace(tmp_path)

    assert workspace.list_files(".")["files"] == ["app.py"]
    assert workspace.search_text("sentinel")["matches"] == [
        {"path": "app.py", "line": 1, "text": "sentinel"}
    ]


def test_root_listing_and_search_prune_memory_tree(tmp_path: Path) -> None:
    private = tmp_path / ".testpilot" / "memories"
    private.mkdir(parents=True)
    (private / "entries.jsonl").write_text("memory-sentinel\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("memory-sentinel\n", encoding="utf-8")
    workspace = Workspace(tmp_path)

    assert workspace.list_files(".")["files"] == ["app.py"]
    assert workspace.search_text("memory-sentinel")["matches"] == [
        {"path": "app.py", "line": 1, "text": "memory-sentinel"}
    ]


def test_checkpoint_tree_is_private_through_a_symlink_alias(tmp_path: Path) -> None:
    private = tmp_path / ".testpilot" / "checkpoints"
    private.mkdir(parents=True)
    target = private / "0123456789abcdef.json"
    target.write_text("sentinel", encoding="utf-8")
    alias = tmp_path / "checkpoint-alias"
    try:
        alias.symlink_to(private, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this system")

    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceError) as caught:
        workspace.read_file("checkpoint-alias/0123456789abcdef.json")
    assert caught.value.code == "private_path"
    assert workspace.list_files(".")["files"] == []


def test_private_patterns_can_be_explicitly_disabled(tmp_path: Path) -> None:
    private = tmp_path / ".testpilot" / "checkpoints"
    private.mkdir(parents=True)
    target = private / "state.json"
    target.write_bytes(b"visible\n")

    workspace = Workspace(tmp_path, private_patterns=())

    assert workspace.read_file(".testpilot/checkpoints/state.json")["content"] == "visible\n"


@pytest.mark.parametrize("patterns", [".testpilot/checkpoints/**", ("",)])
def test_workspace_rejects_invalid_private_patterns(
    tmp_path: Path,
    patterns: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        Workspace(tmp_path, private_patterns=patterns)  # type: ignore[arg-type]


def test_write_file_protects_symlink_aliases_to_protected_paths(tmp_path: Path) -> None:
    protected_directory = tmp_path / "tests"
    protected_directory.mkdir()
    target = protected_directory / "test_target.py"
    target.write_text("old\n", encoding="utf-8")
    file_alias = tmp_path / "alias.py"
    directory_alias = tmp_path / "alias_tests"
    try:
        file_alias.symlink_to(target)
        directory_alias.symlink_to(protected_directory, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this system")

    workspace = Workspace(tmp_path)
    for path in ("alias.py", "alias_tests/new.py"):
        with pytest.raises(WorkspaceError) as raised:
            workspace.write_file(path, "changed\n")
        assert raised.value.code == "protected_path"


def test_write_file_uses_windows_case_insensitive_protection(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    if os.name == "nt":
        with pytest.raises(WorkspaceError) as raised:
            workspace.write_file("PYPROJECT.TOML", "changed\n")
        assert raised.value.code == "protected_path"
    else:
        assert workspace.write_file("PYPROJECT.TOML", "changed\n")["changed"] is True


@pytest.mark.skipif(os.name != "nt", reason="Win32 strips trailing spaces and dots")
@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml.",
        "test_example.py.",
        "src. /app.py",
        "src/app.py ",
    ],
)
def test_workspace_rejects_windows_alias_path_components(
    tmp_path: Path,
    path: str,
) -> None:
    with pytest.raises(WorkspaceError) as raised:
        Workspace(tmp_path).write_file(path, "changed\n")

    assert raised.value.code == "invalid_path"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable on Windows")
def test_write_file_preserves_existing_posix_mode(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "script.sh"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o751)

    Workspace(root).write_file("script.sh", "new\n")

    assert stat.S_IMODE(target.stat().st_mode) == 0o751


def test_edit_file_replaces_one_exact_match_atomically(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "a.py"
    target.write_text("before\nx = 1\nafter\n", encoding="utf-8")

    result = Workspace(root).edit_file("a.py", "x = 1", "x = 2")

    assert result == {"path": "a.py", "changed": True}
    assert target.read_text(encoding="utf-8") == "before\nx = 2\nafter\n"


@pytest.mark.parametrize(
    ("contents", "old_text", "new_text", "code"),
    [
        ("x = 1\n", "missing", "replacement", "missing_match"),
        ("x = 1\nx = 1\n", "x = 1", "x = 2", "ambiguous_edit"),
        ("x = 1\n", "x = 1", "x = 1", "no_change"),
    ],
)
def test_failed_edit_does_not_modify_file(
    tmp_path: Path,
    contents: str,
    old_text: str,
    new_text: str,
    code: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "a.py"
    target.write_text(contents, encoding="utf-8")

    with pytest.raises(WorkspaceError) as raised:
        Workspace(root).edit_file("a.py", old_text, new_text)

    assert raised.value.code == code
    assert target.read_text(encoding="utf-8") == contents


@pytest.fixture
def toolset(tmp_path: Path) -> dict[str, Any]:
    workspace = Workspace(tmp_path)
    tools = [
        ListFilesTool(workspace),
        ReadFileTool(workspace),
        SearchTextTool(workspace),
        EditFileTool(workspace),
        WriteFileTool(workspace),
    ]
    return {tool.name: tool for tool in tools}


@pytest.mark.parametrize(
    "tool_name",
    ["list_files", "read_file", "search_text", "edit_file", "write_file"],
)
def test_direct_tool_calls_reject_non_mapping_arguments(
    toolset: dict[str, Any],
    tool_name: str,
) -> None:
    result = toolset[tool_name].execute(["not", "an", "object"])

    assert not result.ok
    assert result.error_code == "invalid_arguments"


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("list_files", {"path": 1}),
        ("read_file", {"path": "a.py", "start_line": True}),
        ("read_file", {"path": "a.py", "start_line": 0}),
        ("search_text", {"query": 1}),
        ("edit_file", {"path": "a.py", "old_text": "x", "new_text": 1}),
        ("write_file", {"path": "a.py", "content": object()}),
        ("list_files", {"unknown": "value"}),
    ],
)
def test_direct_tool_calls_reject_schema_invalid_values(
    toolset: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    result = toolset[tool_name].execute(arguments)

    assert not result.ok
    assert result.error_code == "invalid_arguments"


def test_tools_convert_workspace_errors_to_structured_failures(toolset: dict[str, Any]) -> None:
    result = toolset["read_file"].execute({"path": "../outside.txt"})

    assert not result.ok
    assert result.error_code == "path_outside_workspace"


def test_tools_convert_malformed_paths_to_structured_failures(toolset: dict[str, Any]) -> None:
    result = toolset["read_file"].execute({"path": "bad\x00name.txt"})

    assert not result.ok
    assert result.error_code == "invalid_path"


def test_read_tool_propagates_truncation_to_tool_result(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("abcdef", encoding="utf-8")

    result = ReadFileTool(Workspace(tmp_path, max_read_chars=3)).execute({"path": "large.txt"})

    assert result.ok
    assert result.truncated is True
    assert result.data["content"] == "abc"


def test_edit_and_write_tools_mark_successful_changes(
    toolset: dict[str, Any],
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")

    edited = toolset["edit_file"].execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"}
    )
    written = toolset["write_file"].execute({"path": "nested/b.py", "content": "value = 3\n"})

    assert edited.ok and edited.data == {"path": "a.py", "changed": True}
    assert written.ok and written.data == {"path": "nested/b.py", "changed": True}


def test_write_tool_reports_successful_noop(toolset: dict[str, Any], tmp_path: Path) -> None:
    (tmp_path / "same.py").write_bytes(b"same\n")

    result = toolset["write_file"].execute({"path": "same.py", "content": "same\n"})

    assert result.ok
    assert result.data == {"path": "same.py", "changed": False}


def test_all_file_tool_schemas_register_and_dispatch(
    toolset: dict[str, Any],
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
    registry = ToolRegistry()
    for tool in toolset.values():
        registry.register(tool)

    assert registry.names() == (
        "list_files",
        "read_file",
        "search_text",
        "edit_file",
        "write_file",
    )
    assert registry.execute("list_files", {}).ok
    assert registry.execute("read_file", {"path": "a.py"}).ok
    assert registry.execute("search_text", {"query": "needle"}).ok
