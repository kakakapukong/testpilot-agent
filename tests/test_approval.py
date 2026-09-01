import os
import stat
from pathlib import Path

import pytest

from testpilot.approval import (
    ApprovalError,
    ChangeJournal,
    ChangeSummary,
    ConsoleApprovalWorkflow,
)
from testpilot.workspace import Workspace


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
        "A alpha.py (+1/-0)",
        "M zeta.py (+2/-1)",
    ]
    assert prompts == ["Accept verified changes? [y/N]: "]
    rendered = "\n".join((*lines, *prompts))
    assert secret_before not in rendered
    assert secret_after not in rendered


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
