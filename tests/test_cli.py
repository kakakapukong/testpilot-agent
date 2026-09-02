"""Behaviour tests for the public TestPilot command-line interface."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from testpilot.approval import ChangeJournal
from testpilot.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointRequest,
    CheckpointSession,
    CheckpointStore,
)
from testpilot.context import BoundedContext
from testpilot.types import AgentRunResult, RunState
from testpilot.workspace import Workspace


def _result(
    *,
    success: bool,
    reason: str = "verified",
    approval_status: str | None = None,
    review_status: str | None = None,
    review_rounds: int = 0,
    review_reworks: int = 0,
    run_id: str | None = None,
    resume_available: bool = False,
    checkpoint_warning: str | None = None,
    memories_retrieved: int = 0,
    memory_saved: str = "no",
    memory_warning: str | None = None,
) -> AgentRunResult:
    state = RunState(stop_reason=reason)
    state.changed_files.add("calculator.py")
    state.last_verify_exit_code = 0 if success else 1
    if approval_status is not None:
        state.record_approval(approval_status)
    state.review_status = review_status
    state.review_rounds = review_rounds
    state.review_rework_count = review_reworks
    return AgentRunResult(
        success,
        "private model text",
        reason,
        state,
        (),
        None,
        run_id=run_id,
        checkpoint_path=None,
        resume_available=resume_available,
        checkpoint_warning=checkpoint_warning,
        memories_retrieved=memories_retrieved,
        memory_saved=memory_saved,
        memory_warning=memory_warning,
    )


class _Runner:
    def __init__(self, result: AgentRunResult) -> None:
        self.result = result
        self.task: str | None = None
        self.resume: object | None = None

    def run(self, task: str, *, resume: object | None = None) -> AgentRunResult:
        self.task = task
        self.resume = resume
        return self.result


def _saved_cli_checkpoint(
    root: Path,
    *,
    task: str = "Fix app.py",
    verifier: tuple[str, ...] | None = None,
) -> tuple[CheckpointStore, CheckpointSession, Path]:
    trace = root / ".testpilot" / "traces" / "original.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text('{"event":"before"}\n', encoding="utf-8")
    target = root / "app.py"
    target.write_bytes(b"old\n")
    journal = ChangeJournal(root)
    Workspace(root, change_recorder=journal).write_file("app.py", "new\n")
    store = CheckpointStore(root)
    session = CheckpointSession.create(
        store=store,
        journal=journal,
        request=CheckpointRequest(
            task=task,
            verifier=verifier or (sys.executable, "-m", "pytest", "-q"),
            max_iterations=12,
            trace_path=".testpilot/traces/original.jsonl",
        ),
    )
    context = BoundedContext(
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": task},
    )
    session.save(
        context=context,
        state=RunState(
            iteration=2,
            edit_count=1,
            source_edit_count=1,
            changed_files={"app.py"},
        ),
        last_call_signature="stored-signature",
    )
    return store, session, trace


def _link_directory(alias: Path, target: Path) -> None:
    try:
        alias.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        pass
    try:
        alias.parent.mkdir(parents=True, exist_ok=True)
        import subprocess

        subprocess.run(
            [
                "cmd",
                "/c",
                "mklink",
                "/J",
                str(alias),
                str(target),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        pytest.skip(f"directory link unavailable: {error}")


def test_help_does_not_require_api_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    from testpilot.cli import main

    assert main(["--help"]) == 0
    assert "--workspace" in capsys.readouterr().out


def test_parser_accepts_separate_fresh_and_resume_forms() -> None:
    from testpilot.cli import build_parser

    fresh = build_parser().parse_args(
        ["--workspace", ".", "--verify", "python -m pytest -q", "Fix app.py"]
    )
    resumed = build_parser().parse_args(
        ["--workspace", ".", "--resume", "0123456789abcdef"]
    )

    assert fresh.resume is None
    assert fresh.max_iterations is None
    assert resumed.resume == "0123456789abcdef"
    assert resumed.task is None
    assert resumed.verify is None


@pytest.mark.parametrize(
    "extra",
    [
        ["Task is forbidden"],
        ["--verify", "python -m pytest -q"],
        ["--trace", ".testpilot/traces/new.jsonl"],
    ],
)
def test_resume_rejects_fresh_only_arguments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
) -> None:
    from testpilot.cli import main

    arguments = [
        "--workspace",
        str(tmp_path),
        "--resume",
        "0123456789abcdef",
        *extra,
    ]

    assert main(arguments) == 2
    assert capsys.readouterr().out.splitlines() == ["STATUS=CONFIG_ERROR"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--verify", "python -m pytest -q"],
        ["Fix app.py"],
    ],
)
def test_fresh_mode_requires_both_task_and_verifier(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    from testpilot.cli import main

    assert main(["--workspace", str(tmp_path), *arguments]) == 2
    assert capsys.readouterr().out.splitlines() == ["STATUS=CONFIG_ERROR"]


def test_resume_rejects_an_invalid_run_id_without_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from testpilot import cli

    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    monkeypatch.setenv("OPENAI_MODEL", "private-model")
    constructed = False

    def forbidden_model(**kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(cli, "OpenAIChatModel", forbidden_model)

    assert cli.main(["--workspace", str(tmp_path), "--resume", "../escape"]) == 1
    output = capsys.readouterr().out
    assert "stop_reason=checkpoint_invalid" in output
    assert "private-key" not in output
    assert not constructed


def test_cli_rejects_missing_or_non_directory_workspace(capsys: pytest.CaptureFixture[str]) -> None:
    from testpilot.cli import main

    assert main(["--workspace", "missing", "--verify", "python -m pytest -q", "Fix it"]) == 2
    assert "STATUS=CONFIG_ERROR" in capsys.readouterr().out


def test_cli_rejects_trace_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from testpilot.cli import main

    monkeypatch.setenv("OPENAI_API_KEY", "sentinel-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    assert (
        main(
            [
                "--workspace",
                str(tmp_path),
                "--trace",
                str(tmp_path.parent / "outside.jsonl"),
                "--verify",
                "python -m pytest -q",
                "Fix it",
            ]
        )
        == 2
    )
    assert "sentinel-key" not in capsys.readouterr().out


def test_default_trace_rejects_a_testpilot_symlink_that_escapes_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from testpilot.cli import _trace_path, main

    outside = tmp_path.parent / "outside-traces"
    outside.mkdir()
    alias = tmp_path / ".testpilot"
    _link_directory(alias, outside)
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    with pytest.raises(ValueError, match="trace path"):
        _trace_path(tmp_path.resolve(), None)
    assert main(["--workspace", str(tmp_path), "--verify", "python -m pytest -q", "Fix it"]) == 2
    assert not list(outside.iterdir())
    assert "sentinel-key" not in capsys.readouterr().out


def test_default_trace_path_stays_inside_workspace(tmp_path: Path) -> None:
    from testpilot.cli import _trace_path

    trace = _trace_path(tmp_path.resolve(), None)

    assert trace.resolve(strict=False).relative_to(tmp_path.resolve()).parts[:2] == (
        ".testpilot",
        "traces",
    )
    assert trace.is_file()
    assert trace.read_bytes() == b""


def test_default_trace_paths_are_distinct_reserved_files(tmp_path: Path) -> None:
    from testpilot.cli import _trace_path

    first = _trace_path(tmp_path.resolve(), None)
    second = _trace_path(tmp_path.resolve(), None)

    assert first != second
    assert first.is_file()
    assert second.is_file()


@pytest.mark.parametrize("existing_kind", ["file", "directory"])
def test_trace_path_rejects_an_existing_target(tmp_path: Path, existing_kind: str) -> None:
    from testpilot.cli import _trace_path

    target = tmp_path / "existing.jsonl"
    if existing_kind == "file":
        target.touch()
    else:
        target.mkdir()

    with pytest.raises(ValueError, match=r"new \.jsonl file"):
        _trace_path(tmp_path.resolve(), str(target))


def test_trace_path_rejects_non_jsonl_target(tmp_path: Path) -> None:
    from testpilot.cli import _trace_path

    with pytest.raises(ValueError, match=r"new \.jsonl file"):
        _trace_path(tmp_path.resolve(), "calculator.py")


def test_trace_path_atomically_reserves_a_new_supplied_target(tmp_path: Path) -> None:
    from testpilot.cli import _trace_path

    trace = _trace_path(tmp_path.resolve(), ".testpilot/traces/my-run.jsonl")

    assert trace.is_file()
    assert trace.read_bytes() == b""


@pytest.mark.parametrize("missing", ["OPENAI_API_KEY", "OPENAI_MODEL"])
def test_cli_requires_non_empty_api_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    from testpilot.cli import main

    monkeypatch.setenv("OPENAI_API_KEY", "sentinel-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "")
    monkeypatch.delenv(missing, raising=False)
    assert main(["--workspace", str(tmp_path), "--verify", "python -m pytest -q", "Fix it"]) == 2
    output = capsys.readouterr().out
    assert "STATUS=CONFIG_ERROR" in output
    assert "sentinel-api-key" not in output


def test_cli_parses_and_prechecks_verifier_before_building_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from testpilot import cli

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "model")
    called = False

    def forbidden_factory(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(cli, "build_agent", forbidden_factory)
    assert (
        cli.main(["--workspace", str(tmp_path), "--verify", "powershell -Command nope", "Fix it"])
        == 2
    )
    assert not called
    assert "STATUS=CONFIG_ERROR" in capsys.readouterr().out


def test_cli_rejects_a_custom_python_verifier_before_building_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from testpilot import cli

    (tmp_path / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "model")
    called = False

    def forbidden_factory(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(cli, "build_agent", forbidden_factory)

    assert (
        cli.main(
            [
                "--workspace",
                str(tmp_path),
                "--verify",
                f'"{sys.executable}" verify.py',
                "Fix it",
            ]
        )
        == 2
    )
    assert not called
    assert "STATUS=CONFIG_ERROR" in capsys.readouterr().out


def test_cli_rejects_a_whitespace_only_task_before_building_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from testpilot import cli

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "model")
    called = False

    def forbidden_factory(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(cli, "build_agent", forbidden_factory)

    assert cli.main(["--workspace", str(tmp_path), "--verify", "python -m pytest", " \t "]) == 2
    assert not called
    assert "STATUS=CONFIG_ERROR" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("corrupt", "checkpoint_invalid"),
        ("workspace", "checkpoint_workspace_mismatch"),
        ("fingerprint", "checkpoint_workspace_changed"),
        ("verifier", "checkpoint_invalid"),
        ("terminal", "checkpoint_invalid"),
    ],
)
def test_resume_validation_stops_before_any_model_is_constructed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
    expected_reason: str,
) -> None:
    from testpilot import cli

    original = tmp_path / "original"
    original.mkdir()
    workspace = original
    if case == "corrupt":
        store = CheckpointStore(original)
        run_id = store.new_run_id()
        store.path_for(run_id).write_text(
            '{"private":"checkpoint source must stay hidden"}',
            encoding="utf-8",
        )
    elif case == "verifier":
        store, session, _ = _saved_cli_checkpoint(
            original,
            task="Private repair task",
            verifier=(sys.executable, "private_verify.py"),
        )
        run_id = session.run_id
    else:
        store, session, _ = _saved_cli_checkpoint(
            original,
            task="Private repair task",
        )
        run_id = session.run_id
        if case == "workspace":
            workspace = tmp_path / "other"
            workspace.mkdir()
            copied_store = CheckpointStore(workspace)
            shutil.copyfile(
                store.path_for(run_id),
                copied_store.path_for(run_id),
            )
        elif case == "fingerprint":
            (original / "app.py").write_text("external private source\n", encoding="utf-8")
        elif case == "terminal":
            checkpoint = store.load(run_id)
            store.save(
                replace(
                    checkpoint,
                    lifecycle_status="terminal",
                    updated_at=datetime.now(UTC).isoformat(),
                )
            )

    monkeypatch.setenv("OPENAI_API_KEY", "private-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "private-model")
    constructed = False

    def forbidden_model(**kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(cli, "OpenAIChatModel", forbidden_model)

    assert cli.main(["--workspace", str(workspace), "--resume", run_id]) == 1
    output = capsys.readouterr().out
    assert f"stop_reason={expected_reason}" in output
    assert "Private repair task" not in output
    assert "private source" not in output
    assert "private-api-key" not in output
    assert not constructed


def test_cli_strips_task_edges_before_running_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from testpilot import cli

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "model")
    runner = _Runner(_result(success=True))
    monkeypatch.setattr(cli, "build_agent", lambda **kwargs: runner)

    assert (
        cli.main(["--workspace", str(tmp_path), "--verify", "python -m pytest", "  Repair it  "])
        == 0
    )
    assert runner.task == "Repair it"
    assert "STATUS=SUCCESS" in capsys.readouterr().out


def test_fresh_main_assembles_one_shared_journal_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testpilot import cli

    monkeypatch.setenv("OPENAI_API_KEY", "fresh-key")
    monkeypatch.setenv("OPENAI_MODEL", "fresh-model")
    runner = _Runner(_result(success=True))
    captured: dict[str, Any] = {}

    def recording_factory(**kwargs: object) -> _Runner:
        captured.update(kwargs)
        return runner

    monkeypatch.setattr(cli, "build_agent", recording_factory)

    assert (
        cli.main(
            [
                "--workspace",
                str(tmp_path),
                "--verify",
                f'"{sys.executable}" -m pytest -q',
                "  Fix app.py  ",
            ]
        )
        == 0
    )

    config = captured["config"]
    journal = captured["journal"]
    checkpoint = captured["checkpoint"]
    assert isinstance(config, cli.CliConfig)
    assert isinstance(journal, ChangeJournal)
    assert isinstance(checkpoint, CheckpointSession)
    assert checkpoint.journal is journal
    assert checkpoint.request.task == "Fix app.py"
    assert checkpoint.request.verifier == config.verifier
    assert checkpoint.request.trace_path == config.trace_path.relative_to(tmp_path).as_posix()
    assert config.max_iterations == 12
    assert config.api_key == "fresh-key"
    assert runner.task == "Fix app.py"
    assert runner.resume is None


def test_resume_main_reuses_stored_request_trace_and_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testpilot import cli

    _, saved_session, trace = _saved_cli_checkpoint(tmp_path, task="Stored task")
    monkeypatch.setenv("OPENAI_API_KEY", "current-key")
    monkeypatch.setenv("OPENAI_MODEL", "current-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://current.invalid/v1")
    runner = _Runner(_result(success=True, run_id=saved_session.run_id))
    captured: dict[str, Any] = {}

    def recording_factory(**kwargs: object) -> _Runner:
        captured.update(kwargs)
        return runner

    monkeypatch.setattr(cli, "build_agent", recording_factory)

    assert (
        cli.main(
            [
                "--workspace",
                str(tmp_path),
                "--resume",
                saved_session.run_id,
                "--max-iterations",
                "3",
            ]
        )
        == 0
    )

    config = captured["config"]
    journal = captured["journal"]
    checkpoint = captured["checkpoint"]
    assert isinstance(config, cli.CliConfig)
    assert isinstance(journal, ChangeJournal)
    assert isinstance(checkpoint, CheckpointSession)
    assert checkpoint.journal is journal
    assert config.task == "Stored task"
    assert config.verifier == checkpoint.request.verifier
    assert config.trace_path == trace
    assert config.max_iterations == 3
    assert checkpoint.request.max_iterations == 12
    assert (config.api_key, config.model, config.base_url) == (
        "current-key",
        "current-model",
        "https://current.invalid/v1",
    )
    assert runner.task == "Stored task"
    assert runner.resume is not None
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    restored = records[-1]
    assert restored["event"] == "checkpoint_restore"
    assert set(restored["payload"]) == {
        "run_id",
        "schema_version",
        "safe_point",
        "ok",
        "error_code",
        "duration_ms",
    }
    assert restored["payload"]["run_id"] == saved_session.run_id
    assert restored["payload"]["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert restored["payload"]["ok"] is True


def test_checkpoint_ready_callback_prints_only_relative_safe_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from testpilot.cli import _fresh_setup

    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    monkeypatch.setenv("OPENAI_MODEL", "private-model")
    setup = _fresh_setup(
        tmp_path.resolve(),
        verify=f'"{sys.executable}" -m pytest -q',
        task="Private task",
        trace=None,
        max_iterations=None,
    )
    context = BoundedContext(
        {"role": "developer", "content": "private rules"},
        {"role": "user", "content": "Private task"},
    )

    setup.checkpoint.save(
        context=context,
        state=RunState(),
        last_call_signature=None,
    )

    assert capsys.readouterr().out.splitlines() == [
        f"run_id={setup.checkpoint.run_id}",
        f"checkpoint=.testpilot/checkpoints/{setup.checkpoint.run_id}.json",
    ]


def test_windows_verify_parser_preserves_unquoted_backslashes() -> None:
    from testpilot.cli import parse_verify_command

    assert parse_verify_command(r"C:\Python311\python.exe -m pytest tests\unit", windows=True) == (
        r"C:\Python311\python.exe",
        "-m",
        "pytest",
        r"tests\unit",
    )


def test_windows_verify_parser_unquotes_paths_with_spaces() -> None:
    from testpilot.cli import parse_verify_command

    assert parse_verify_command(
        r'"C:\Program Files\Python\python.exe" -m pytest "tests\unit test.py"',
        windows=True,
    ) == (
        r"C:\Program Files\Python\python.exe",
        "-m",
        "pytest",
        r"tests\unit test.py",
    )


@pytest.mark.parametrize("success, expected_exit", [(True, 0), (False, 1)])
def test_cli_prints_compact_result_without_secrets_or_model_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    success: bool,
    expected_exit: int,
) -> None:
    from testpilot import cli

    monkeypatch.setenv("OPENAI_API_KEY", "cli-key-sentinel")
    monkeypatch.setenv("OPENAI_MODEL", "model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://token:base-secret@example.invalid")
    runner = _Runner(
        _result(
            success=success,
            reason="verified" if success else "max_iterations",
            approval_status="approved" if success else None,
            review_status="passed" if success else None,
            review_rounds=1 if success else 0,
        )
    )
    monkeypatch.setattr(cli, "build_agent", lambda **kwargs: runner)

    assert (
        cli.main(
            [
                "--workspace",
                str(tmp_path),
                "--verify",
                f'"{sys.executable}" -m pytest -q',
                "Repair it",
            ]
        )
        == expected_exit
    )
    output = capsys.readouterr().out
    assert f"STATUS={'SUCCESS' if success else 'FAILED'}" in output
    assert 'changed_files=["calculator.py"]' in output
    assert f"review={'passed' if success else '-'}" in output
    assert f"review_rounds={1 if success else 0}" in output
    assert "review_reworks=0" in output
    assert f"approval={'approved' if success else '-'}" in output
    assert "private model text" not in output
    assert "cli-key-sentinel" not in output
    assert "base-secret" not in output


def test_cli_sanitizes_memory_store_setup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from testpilot import cli

    monkeypatch.setenv("OPENAI_API_KEY", "memory-setup-key-sentinel")
    monkeypatch.setenv("OPENAI_MODEL", "private-model")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://person:memory-base-secret@example.invalid/v1",
    )
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda **kwargs: object())

    def failing_store(workspace: Path) -> object:
        del workspace
        raise RuntimeError("memory-setup-key-sentinel memory-private-detail")

    monkeypatch.setattr(cli, "MemoryStore", failing_store)

    assert (
        cli.main(
            [
                "--workspace",
                str(tmp_path),
                "--verify",
                f'"{sys.executable}" -m pytest -q',
                "Private memory task",
            ]
        )
        == 1
    )

    output = capsys.readouterr().out
    assert "STATUS=FAILED" in output
    assert "stop_reason=runtime_setup_failed" in output
    assert "memory-setup-key-sentinel" not in output
    assert "memory-base-secret" not in output
    assert "memory-private-detail" not in output
    assert "Private memory task" not in output


def test_print_result_includes_stable_checkpoint_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from testpilot.cli import _print_result

    result = _result(
        success=False,
        reason="model_transient_failure",
        run_id="0123456789abcdef",
        resume_available=True,
        checkpoint_warning="checkpoint_cleanup_failed",
    )

    _print_result(result, tmp_path / "trace.jsonl")

    lines = capsys.readouterr().out.splitlines()
    assert "run_id=0123456789abcdef" in lines
    assert "resume_available=yes" in lines
    assert "checkpoint_warning=checkpoint_cleanup_failed" in lines


def test_build_agent_registers_all_seven_tools(tmp_path: Path) -> None:
    from testpilot.cli import CliConfig, build_agent

    agent = build_agent(
        CliConfig(
            workspace=tmp_path,
            verifier=(sys.executable, "-m", "pytest", "-q"),
            task="Fix it",
            api_key="key",
            model="model",
            base_url=None,
            trace_path=tmp_path / "trace.jsonl",
            max_iterations=2,
        )
    )

    assert agent.registry.names() == (
        "list_files",
        "read_file",
        "search_text",
        "edit_file",
        "write_file",
        "run_command",
        "finish",
    )
    assert agent.reviewer is not None
    assert agent.reviewer.registry.names() == (
        "list_files",
        "read_file",
        "search_text",
        "submit_review",
    )
    assert agent.reviewer.model is not agent.model
    assert agent.max_iterations == 2


def test_build_agent_creates_distinct_repair_reviewer_and_memory_model_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testpilot import cli

    created: list[object] = []
    configurations: list[dict[str, object]] = []

    def recording_model(**kwargs: object) -> object:
        model = object()
        created.append(model)
        configurations.append(dict(kwargs))
        return model

    monkeypatch.setattr(cli, "OpenAIChatModel", recording_model)

    agent = cli.build_agent(
        cli.CliConfig(
            workspace=tmp_path,
            verifier=(sys.executable, "-m", "pytest", "-q"),
            task="Fix it",
            api_key="key",
            model="model",
            base_url="https://example.invalid/v1",
            trace_path=tmp_path / "trace.jsonl",
            max_iterations=2,
        )
    )

    assert len(created) == 3
    assert configurations == [
        {
            "model": "model",
            "api_key": "key",
            "base_url": "https://example.invalid/v1",
        },
        {
            "model": "model",
            "api_key": "key",
            "base_url": "https://example.invalid/v1",
        },
        {
            "model": "model",
            "api_key": "key",
            "base_url": "https://example.invalid/v1",
        },
    ]
    assert agent.model is created[0]
    assert agent.reviewer.model is created[1]
    assert agent.memory_agent.model is created[2]
    assert agent.memory_agent.registry.names() == ("submit_memory",)
    assert agent.memory_store.workspace == tmp_path.resolve()
    assert agent.memory_store.path == (
        tmp_path.resolve() / ".testpilot" / "memories" / "entries.jsonl"
    )


def test_print_result_includes_stable_memory_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from testpilot.cli import _print_result

    result = _result(
        success=True,
        memories_retrieved=2,
        memory_saved="duplicate",
        memory_warning="memory_load_failed",
    )

    assert _print_result(result, tmp_path / "trace.jsonl") == 0

    lines = capsys.readouterr().out.splitlines()
    assert "memories_retrieved=2" in lines
    assert "memory_saved=duplicate" in lines
    assert "memory_warning=memory_load_failed" in lines
    assert "private model text" not in lines


def test_print_result_rejects_malformed_or_untrusted_memory_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from testpilot.cli import _print_result

    hostile = "memory_load_failed\nSTATUS=SUCCESS\x1b[2Jsecret"
    result = _result(success=False)
    object.__setattr__(result, "memories_retrieved", True)
    object.__setattr__(result, "memory_saved", "yes\nSTATUS=SUCCESS")
    object.__setattr__(result, "memory_warning", hostile)

    assert _print_result(result, tmp_path / "trace.jsonl") == 1

    lines = capsys.readouterr().out.splitlines()
    assert "memories_retrieved=0" in lines
    assert "memory_saved=no" in lines
    assert "memory_warning=-" in lines
    assert lines.count("STATUS=SUCCESS") == 0
    assert "secret" not in "\n".join(lines)


def test_build_agent_shares_one_journal_with_workspace_and_console_approval(
    tmp_path: Path,
) -> None:
    from testpilot.approval import ChangeSummary, ConsoleApprovalWorkflow
    from testpilot.cli import CliConfig, build_agent

    output: list[str] = []
    agent = build_agent(
        CliConfig(
            workspace=tmp_path,
            verifier=(sys.executable, "-m", "pytest", "-q"),
            task="Fix it",
            api_key="key",
            model="model",
            base_url=None,
            trace_path=tmp_path / "trace.jsonl",
            max_iterations=2,
        ),
        input_fn=lambda prompt: "yes",
        output_fn=output.append,
    )

    assert isinstance(agent.approval, ConsoleApprovalWorkflow)
    result = agent.registry.execute("write_file", {"path": "app.py", "content": "value = 1\n"})

    assert result.ok
    review_read = agent.reviewer.registry.execute("read_file", {"path": "app.py"})
    assert review_read.ok
    assert "value = 1" in review_read.data["content"]
    assert agent.approval.journal.summaries() == (
        ChangeSummary("app.py", "created", additions=1, deletions=0),
    )
    assert agent.approval.request(changed_files=("app.py",), verification_exit_code=0)
    assert output == [
        "APPROVAL_REQUIRED",
        "verification_exit=0",
        'A "app.py" (+1/-0)',
    ]


def test_build_agent_reuses_the_supplied_checkpoint_journal_everywhere(
    tmp_path: Path,
) -> None:
    from testpilot.approval import ConsoleApprovalWorkflow
    from testpilot.cli import CliConfig, build_agent

    config = CliConfig(
        workspace=tmp_path,
        verifier=(sys.executable, "-m", "pytest", "-q"),
        task="Fix it",
        api_key="key",
        model="model",
        base_url=None,
        trace_path=tmp_path / ".testpilot" / "traces" / "run.jsonl",
        max_iterations=2,
    )
    journal = ChangeJournal(tmp_path)
    checkpoint = CheckpointSession.create(
        store=CheckpointStore(tmp_path),
        journal=journal,
        request=CheckpointRequest(
            task=config.task,
            verifier=config.verifier,
            max_iterations=config.max_iterations,
            trace_path=".testpilot/traces/run.jsonl",
        ),
    )

    agent = build_agent(
        config,
        journal=journal,
        checkpoint=checkpoint,
        input_fn=lambda prompt: "yes",
        output_fn=lambda line: None,
    )

    assert agent.checkpoint is checkpoint
    assert isinstance(agent.approval, ConsoleApprovalWorkflow)
    assert agent.approval.journal is journal
    written = agent.registry.execute(
        "write_file",
        {"path": "app.py", "content": "value = 1\n"},
    )
    private_read = agent.registry.execute(
        "read_file",
        {"path": f".testpilot/checkpoints/{checkpoint.run_id}.json"},
    )
    assert written.ok
    assert journal.export_snapshots()[0].path == "app.py"
    assert private_read.error_code == "private_path"


@pytest.mark.parametrize(
    ("approval_status", "expected"),
    [
        ("approved", "approved"),
        ("rejected", "rejected"),
        ("unavailable", "unavailable"),
        (None, "-"),
    ],
)
def test_print_result_includes_stable_approval_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    approval_status: str | None,
    expected: str,
) -> None:
    from testpilot.cli import _print_result

    result = _result(success=approval_status == "approved", approval_status=approval_status)

    _print_result(result, tmp_path / "trace.jsonl")

    assert f"approval={expected}" in capsys.readouterr().out.splitlines()


@pytest.mark.parametrize(
    ("review_status", "rounds", "reworks", "expected"),
    [
        ("passed", 1, 0, "passed"),
        ("changes_requested", 2, 1, "changes_requested"),
        ("unavailable", 1, 0, "unavailable"),
        (None, 0, 0, "-"),
    ],
)
def test_print_result_includes_stable_review_accounting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    review_status: str | None,
    rounds: int,
    reworks: int,
    expected: str,
) -> None:
    from testpilot.cli import _print_result

    result = _result(
        success=review_status == "passed",
        review_status=review_status,
        review_rounds=rounds,
        review_reworks=reworks,
    )

    _print_result(result, tmp_path / "trace.jsonl")

    lines = capsys.readouterr().out.splitlines()
    assert f"review={expected}" in lines
    assert f"review_rounds={rounds}" in lines
    assert f"review_reworks={reworks}" in lines


def test_print_result_json_escapes_control_and_bidi_characters_in_changed_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from testpilot.cli import _print_result

    hostile_path = "safe.py\nSTATUS=SUCCESS\x1b[2J\u202e.py"
    result = _result(success=False)
    result.state.changed_files = {hostile_path}

    _print_result(result, tmp_path / "trace.jsonl")

    lines = capsys.readouterr().out.splitlines()
    assert 'changed_files=["safe.py\\nSTATUS=SUCCESS\\u001b[2J\\u202e.py"]' in lines
    assert lines.count("STATUS=SUCCESS") == 0


def test_build_agent_workspace_protects_verifier_and_trace_assets(tmp_path: Path) -> None:
    from testpilot.cli import CliConfig, build_agent

    checks = tmp_path / "checks"
    checks.mkdir()
    verification_test = checks / "case.py"
    verification_test.write_text("def test_gate():\n    assert False\n", encoding="utf-8")
    trace_path = tmp_path / "audit.jsonl"
    agent = build_agent(
        CliConfig(
            workspace=tmp_path,
            verifier=(sys.executable, "-m", "pytest", "checks/case.py", "-q"),
            task="Fix it",
            api_key="key",
            model="model",
            base_url=None,
            trace_path=trace_path,
            max_iterations=2,
        )
    )

    verifier_write = agent.registry.execute(
        "write_file", {"path": "checks/case.py", "content": "def test_gate():\n    assert True\n"}
    )
    trace_write = agent.registry.execute(
        "write_file", {"path": "audit.jsonl", "content": "forged\n"}
    )

    assert verifier_write.error_code == "protected_path"
    assert trace_write.error_code == "protected_path"
    assert verification_test.read_text(encoding="utf-8") == "def test_gate():\n    assert False\n"
    assert not trace_path.exists()


def test_positive_integer_rejects_zero_and_bool() -> None:
    from testpilot.cli import positive_integer

    assert positive_integer("3") == 3
    with pytest.raises(argparse.ArgumentTypeError):
        positive_integer("0")
