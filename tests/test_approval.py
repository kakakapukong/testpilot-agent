import os
import stat
from pathlib import Path

import pytest

from testpilot.approval import (
    ApprovalError,
    ChangeJournal,
    ChangeSummary,
    ConsoleApprovalWorkflow,
    JournalSnapshot,
)
from testpilot.workspace import Workspace, WorkspaceError


def test_change_journal_summarizes_and_restores_existing_and_new_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    original = root / "app.py"
    original.write_bytes(b"value = 1\n")
    journal = ChangeJournal(root)
    workspace = Workspace(root, change_recorder=journal)

    workspace.write_file("app.py", "value = 2\nextra = True\n")
    workspace.write_file("nested/new.py", "created = True\n")

    summaries = journal.summaries()
    assert summaries == (
        ChangeSummary("app.py", "modified", additions=2, deletions=1),
        ChangeSummary("nested/new.py", "created", additions=1, deletions=0),
    )
    assert "value = 1" not in repr(summaries)
    assert "value = 2" not in repr(summaries)

    replace_calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.exists()
        assert source_path.parent == destination_path.parent
        replace_calls.append((source_path, destination_path))
        real_replace(source, destination)

    monkeypatch.setattr("testpilot.approval.os.replace", recording_replace)

    journal.rollback()

    assert original.read_bytes() == b"value = 1\n"
    assert not (root / "nested/new.py").exists()
    assert not (root / "nested").exists()
    assert len(replace_calls) == 1
    assert replace_calls[0][1] == original


def test_change_journal_captures_only_the_first_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"original\n")
    journal = ChangeJournal(root)
    workspace = Workspace(root, change_recorder=journal)

    workspace.write_file("app.py", "intermediate\n")
    workspace.write_file("app.py", "final\nextra\n")

    assert journal.summaries() == (
        ChangeSummary("app.py", "modified", additions=2, deletions=1),
    )

    journal.rollback()

    assert target.read_bytes() == b"original\n"


def test_commit_uses_the_approved_bytes_as_the_next_run_baseline(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"original\n")
    journal = ChangeJournal(root)
    workspace = Workspace(root, change_recorder=journal)

    workspace.write_file("app.py", "approved\n")
    journal.commit()
    workspace.write_file("app.py", "second-run\n")
    journal.rollback()

    assert target.read_bytes() == b"approved\n"


def test_successful_rollback_forgets_snapshots_before_the_next_run(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"original\n")
    journal = ChangeJournal(root)
    workspace = Workspace(root, change_recorder=journal)

    workspace.write_file("app.py", "rejected\n")
    journal.rollback()
    target.write_bytes(b"new-baseline\n")
    workspace.write_file("app.py", "second-run\n")
    journal.rollback()

    assert target.read_bytes() == b"new-baseline\n"


def test_change_journal_snapshot_round_trip_rolls_back_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"original\x00bytes\n")
    original_mode = stat.S_IMODE(target.stat().st_mode)
    first = ChangeJournal(root)
    workspace = Workspace(root, change_recorder=first)
    workspace.write_file("app.py", "replacement\n")
    workspace.write_file("new/nested.py", "created = True\n")

    exported = first.export_snapshots()
    restored = ChangeJournal(root)
    restored.restore_snapshots(exported)
    restored.rollback()

    assert target.read_bytes() == b"original\x00bytes\n"
    assert stat.S_IMODE(target.stat().st_mode) == original_mode
    assert not (root / "new").exists()
    assert tuple(snapshot.path for snapshot in exported) == ("app.py", "new/nested.py")
    assert exported[1].missing_parents == ("new",)


def test_change_journal_restore_is_all_or_nothing_for_duplicate_or_escaping_paths(
    tmp_path: Path,
) -> None:
    valid = JournalSnapshot("app.py", b"old\n", 0o644, ())
    journal = ChangeJournal(tmp_path)

    with pytest.raises(ApprovalError, match="restore"):
        journal.restore_snapshots((valid, valid))
    assert journal.export_snapshots() == ()

    with pytest.raises(ApprovalError, match="restore"):
        journal.restore_snapshots(
            (JournalSnapshot("../outside.py", b"old\n", 0o644, ()),)
        )
    assert journal.export_snapshots() == ()


@pytest.mark.parametrize(
    "snapshot",
    [
        JournalSnapshot("app.py", b"old\n", None, ()),
        JournalSnapshot("app.py", None, 0o644, ()),
        JournalSnapshot("app.py", "old", 0o644, ()),  # type: ignore[arg-type]
        JournalSnapshot("nested/app.py", b"old\n", 0o644, ("other",)),
        JournalSnapshot(".", b"old\n", 0o644, ()),
    ],
)
def test_change_journal_restore_rejects_invalid_snapshot_records(
    tmp_path: Path,
    snapshot: JournalSnapshot,
) -> None:
    journal = ChangeJournal(tmp_path)

    with pytest.raises(ApprovalError, match="restore"):
        journal.restore_snapshots((snapshot,))

    assert journal.export_snapshots() == ()


@pytest.mark.parametrize("path_kind", ["native", "forward_slash", "drive_relative"])
def test_change_journal_restore_rejects_an_absolute_or_drive_snapshot_path(
    tmp_path: Path,
    path_kind: str,
) -> None:
    journal = ChangeJournal(tmp_path)
    if path_kind == "native":
        path = str(tmp_path / "app.py")
    elif path_kind == "forward_slash":
        path = (tmp_path / "app.py").as_posix()
    else:
        path = "C:app.py"
    snapshot = JournalSnapshot(path, b"old\n", 0o644, ())

    with pytest.raises(ApprovalError, match="restore"):
        journal.restore_snapshots((snapshot,))


def test_change_journal_restore_rejects_a_populated_journal(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("old\n", encoding="utf-8")
    journal = ChangeJournal(tmp_path)
    Workspace(tmp_path, change_recorder=journal).write_file("app.py", "new\n")

    with pytest.raises(ApprovalError, match="restore"):
        journal.restore_snapshots(journal.export_snapshots())


def test_change_journal_restore_does_not_touch_tracked_files_until_rollback(
    tmp_path: Path,
) -> None:
    target = tmp_path / "app.py"
    target.write_text("current\n", encoding="utf-8")
    journal = ChangeJournal(tmp_path)

    journal.restore_snapshots((JournalSnapshot("app.py", b"original\n", 0o644, ()),))

    assert target.read_text(encoding="utf-8") == "current\n"
    journal.rollback()
    assert target.read_bytes() == b"original\n"


def test_change_journal_rejects_an_oversized_original_before_snapshotting(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "large.py"
    target.write_bytes(b"123456789")
    journal = ChangeJournal(root, max_snapshot_bytes=8)

    with pytest.raises(ApprovalError, match="too large"):
        journal.capture(target)


def test_change_journal_rejects_an_oversized_current_file_before_summarizing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"old\n")
    journal = ChangeJournal(root, max_snapshot_bytes=8)
    journal.capture(target)
    target.write_bytes(b"123456789")

    with pytest.raises(ApprovalError, match="too large"):
        journal.summaries()


def test_line_change_summary_uses_a_linear_conservative_middle_span() -> None:
    before = b"same-start\nold-a\nshared-middle\nold-b\nsame-end\n"
    after = b"same-start\nnew-a\nshared-middle\nnew-b\nsame-end\n"

    assert ChangeJournal._line_changes(before, after) == (3, 3)


def test_rollback_preserves_a_preexisting_empty_parent_directory(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    parent = root / "existing"
    parent.mkdir(parents=True)
    journal = ChangeJournal(root)

    Workspace(root, change_recorder=journal).write_file("existing/new.py", "new\n")
    journal.rollback()

    assert parent.is_dir()
    assert list(parent.iterdir()) == []


def test_summary_treats_a_missing_original_file_as_fully_deleted(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"first\nsecond\n")
    journal = ChangeJournal(root)
    journal.capture(target)

    target.unlink()

    assert journal.summaries() == (
        ChangeSummary("app.py", "modified", additions=0, deletions=2),
    )


def test_capture_rejects_paths_outside_the_workspace(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")

    with pytest.raises(ApprovalError, match="workspace"):
        ChangeJournal(root).capture(outside)


def test_change_journal_init_converts_resolve_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail = "SYMLINK_LOOP_DETAIL"

    def looping_resolve(self: Path, *, strict: bool = False) -> Path:
        raise RuntimeError(detail)

    monkeypatch.setattr(Path, "resolve", looping_resolve)

    with pytest.raises(ApprovalError) as raised:
        ChangeJournal(tmp_path)

    assert str(raised.value) == "could not initialize workspace change journal"
    assert detail not in str(raised.value)


def test_capture_converts_resolve_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = ChangeJournal(tmp_path)
    detail = "SYMLINK_LOOP_DETAIL"

    def looping_resolve(self: Path, *, strict: bool = False) -> Path:
        raise RuntimeError(detail)

    monkeypatch.setattr(Path, "resolve", looping_resolve)

    with pytest.raises(ApprovalError) as raised:
        journal.capture(Path("app.py"))

    assert str(raised.value) == "path must be inside the workspace"
    assert detail not in str(raised.value)


def test_summary_converts_parent_resolve_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "app.py"
    target.write_bytes(b"original\n")
    journal = ChangeJournal(tmp_path)
    journal.capture(target)
    detail = "SYMLINK_LOOP_DETAIL"

    def looping_resolve(self: Path, *, strict: bool = False) -> Path:
        raise RuntimeError(detail)

    monkeypatch.setattr(Path, "resolve", looping_resolve)

    with pytest.raises(ApprovalError) as raised:
        journal.summaries()

    assert str(raised.value) == "workspace path changed after capture"
    assert detail not in str(raised.value)


def test_summary_converts_target_symlink_resolve_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "app.py"
    target.write_bytes(b"original\n")
    journal = ChangeJournal(tmp_path)
    journal.capture(target)
    detail = "TARGET_SYMLINK_LOOP_DETAIL"
    real_resolve = Path.resolve

    def selective_resolve(self: Path, *, strict: bool = False) -> Path:
        if self == target:
            raise RuntimeError(detail)
        return real_resolve(self, strict=strict)

    def selective_is_symlink(self: Path) -> bool:
        return self == target

    monkeypatch.setattr(Path, "resolve", selective_resolve)
    monkeypatch.setattr(Path, "is_symlink", selective_is_symlink)

    with pytest.raises(ApprovalError) as raised:
        journal.summaries()

    assert str(raised.value) == "workspace path changed after capture"
    assert detail not in str(raised.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable on Windows")
def test_rollback_restores_the_original_posix_mode(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "script.sh"
    target.write_bytes(b"old\n")
    target.chmod(0o751)
    journal = ChangeJournal(root)

    Workspace(root, change_recorder=journal).write_file("script.sh", "new\n")
    target.chmod(0o600)
    journal.rollback()

    assert target.read_bytes() == b"old\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o751


def test_rollback_failure_raises_a_safe_error_and_cleans_its_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    secret = "SOURCE_CONTENT_MUST_NOT_LEAK"
    target.write_text(f"{secret}\n", encoding="utf-8")
    journal = ChangeJournal(root)
    Workspace(root, change_recorder=journal).write_file("app.py", "replacement\n")

    def failing_replace(source: object, destination: object) -> None:
        raise OSError(f"replace failed near {secret}")

    monkeypatch.setattr("testpilot.approval.os.replace", failing_replace)

    with pytest.raises(ApprovalError) as raised:
        journal.rollback()

    assert str(raised.value) == "could not roll back workspace changes"
    assert secret not in str(raised.value)
    assert list(root.glob(".app.py.*.tmp")) == []


def test_rollback_fsync_failure_cleans_its_restore_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "app.py"
    target.write_bytes(b"old\n")
    journal = ChangeJournal(root)
    Workspace(root, change_recorder=journal).write_file("app.py", "new\n")

    def failing_fsync(file_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr("testpilot.approval.os.fsync", failing_fsync)

    with pytest.raises(ApprovalError, match="roll back"):
        journal.rollback()

    assert list(root.glob(".app.py.*.tmp")) == []


@pytest.mark.parametrize("response", ["yes", "y", " YES ", "Y"])
def test_console_approval_accepts_only_yes_responses(
    tmp_path: Path,
    response: str,
) -> None:
    journal = ChangeJournal(tmp_path)
    workflow = ConsoleApprovalWorkflow(
        journal,
        input_fn=lambda prompt: response,
        output_fn=lambda line: None,
    )

    assert workflow.request(changed_files=(), verification_exit_code=0)


@pytest.mark.parametrize("response", ["", "   ", "no", "approve", None, 1, object()])
def test_console_approval_rejects_blank_other_and_non_string_responses(
    tmp_path: Path,
    response: object,
) -> None:
    journal = ChangeJournal(tmp_path)
    workflow = ConsoleApprovalWorkflow(
        journal,
        input_fn=lambda prompt: response,
        output_fn=lambda line: None,
    )

    assert not workflow.request(changed_files=(), verification_exit_code=0)


@pytest.mark.parametrize("exception", [EOFError(), KeyboardInterrupt()])
def test_console_approval_rejects_when_input_is_unavailable(
    tmp_path: Path,
    exception: BaseException,
) -> None:
    journal = ChangeJournal(tmp_path)

    def unavailable_input(prompt: str) -> str:
        raise exception

    workflow = ConsoleApprovalWorkflow(
        journal,
        input_fn=unavailable_input,
        output_fn=lambda line: None,
    )

    assert not workflow.request(changed_files=(), verification_exit_code=0)


def test_console_approval_prints_exact_sorted_content_free_summary(tmp_path: Path) -> None:
    secret_before = "SOURCE_BEFORE_MUST_NOT_LEAK"
    secret_after = "SOURCE_AFTER_MUST_NOT_LEAK"
    modified = tmp_path / "zeta.py"
    modified.write_text(f"{secret_before}\n", encoding="utf-8")
    journal = ChangeJournal(tmp_path)
    workspace = Workspace(tmp_path, change_recorder=journal)
    workspace.write_file("zeta.py", f"{secret_after}\nextra = True\n")
    workspace.write_file("alpha.py", "created = True\n")
    lines: list[str] = []
    prompts: list[str] = []

    def accept(prompt: str) -> str:
        prompts.append(prompt)
        return "yes"

    workflow = ConsoleApprovalWorkflow(journal, input_fn=accept, output_fn=lines.append)

    assert workflow.request(
        changed_files=("zeta.py", "alpha.py"),
        verification_exit_code=0,
    )
    assert lines == [
        "APPROVAL_REQUIRED",
        "verification_exit=0",
        'A "alpha.py" (+1/-0)',
        'M "zeta.py" (+2/-1)',
    ]
    assert prompts == ["Accept verified changes? [y/N]: "]
    rendered = "\n".join((*lines, *prompts))
    assert secret_before not in rendered
    assert secret_after not in rendered


def test_console_approval_escapes_control_and_bidi_characters_in_paths() -> None:
    hostile_path = "safe.py\nAPPROVED\x1b[2J\u202e.py"

    class SummaryJournal:
        def summaries(self) -> tuple[ChangeSummary, ...]:
            return (ChangeSummary(hostile_path, "modified", additions=1, deletions=1),)

        def rollback(self) -> None:
            return None

    lines: list[str] = []
    workflow = ConsoleApprovalWorkflow(
        SummaryJournal(),
        input_fn=lambda prompt: "yes",
        output_fn=lines.append,
    )

    assert workflow.request(changed_files=(hostile_path,), verification_exit_code=0)
    assert lines == [
        "APPROVAL_REQUIRED",
        "verification_exit=0",
        'M "safe.py\\nAPPROVED\\u001b[2J\\u202e.py" (+1/-1)',
    ]
    assert all("\n" not in line and "\x1b" not in line and "\u202e" not in line for line in lines)


def test_console_approval_filters_snapshots_without_a_successful_edit() -> None:
    class SummaryJournal:
        def summaries(self) -> tuple[ChangeSummary, ...]:
            return (
                ChangeSummary("changed.py", "modified", additions=1, deletions=1),
                ChangeSummary("failed.py", "created", additions=0, deletions=0),
            )

        def rollback(self) -> None:
            return None

    lines: list[str] = []
    workflow = ConsoleApprovalWorkflow(
        SummaryJournal(),
        input_fn=lambda prompt: "yes",
        output_fn=lines.append,
    )

    assert workflow.request(changed_files=("changed.py",), verification_exit_code=0)
    assert lines == [
        "APPROVAL_REQUIRED",
        "verification_exit=0",
        'M "changed.py" (+1/-1)',
    ]


def test_console_approval_omits_a_snapshot_left_by_a_failed_workspace_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = ChangeJournal(tmp_path)
    workspace = Workspace(tmp_path, change_recorder=journal)
    real_replace = os.replace
    replace_calls = 0

    def fail_first_replace(source: object, destination: object) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise OSError("simulated write failure")
        real_replace(source, destination)

    monkeypatch.setattr("testpilot.workspace.os.replace", fail_first_replace)

    with pytest.raises(WorkspaceError, match="could not write"):
        workspace.write_file("leftover/failed.py", "not written\n")
    workspace.write_file("changed.py", "written\n")
    lines: list[str] = []
    workflow = ConsoleApprovalWorkflow(
        journal,
        input_fn=lambda prompt: "yes",
        output_fn=lines.append,
    )

    assert workflow.request(changed_files=("changed.py",), verification_exit_code=0)
    assert lines == [
        "APPROVAL_REQUIRED",
        "verification_exit=0",
        'A "changed.py" (+1/-0)',
    ]
    assert not (tmp_path / "leftover").exists()


def test_console_approval_fails_closed_when_a_successful_edit_has_no_snapshot() -> None:
    class EmptyJournal:
        def __init__(self) -> None:
            self.rollback_calls = 0

        def summaries(self) -> tuple[ChangeSummary, ...]:
            return ()

        def rollback(self) -> None:
            self.rollback_calls += 1

    prompted = False

    def forbidden_input(prompt: str) -> str:
        nonlocal prompted
        prompted = True
        return "yes"

    journal = EmptyJournal()
    workflow = ConsoleApprovalWorkflow(
        journal,
        input_fn=forbidden_input,
        output_fn=lambda line: None,
    )

    with pytest.raises(ApprovalError, match="incomplete"):
        workflow.request(changed_files=("missing.py",), verification_exit_code=0)
    assert not prompted
    with pytest.raises(ApprovalError, match="incomplete"):
        workflow.rollback()
    assert journal.rollback_calls == 1


def test_console_approval_rollback_delegates_to_journal() -> None:
    class RecordingJournal:
        def __init__(self) -> None:
            self.rollback_calls = 0

        def rollback(self) -> None:
            self.rollback_calls += 1

    journal = RecordingJournal()
    workflow = ConsoleApprovalWorkflow(
        journal,
        input_fn=lambda prompt: "no",
        output_fn=lambda line: None,
    )

    workflow.rollback()

    assert journal.rollback_calls == 1
