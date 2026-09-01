"""Behaviour tests for the public TestPilot command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from testpilot.types import AgentRunResult, RunState


def _result(
    *,
    success: bool,
    reason: str = "verified",
    approval_status: str | None = None,
) -> AgentRunResult:
    state = RunState(stop_reason=reason)
    state.changed_files.add("calculator.py")
    state.last_verify_exit_code = 0 if success else 1
    if approval_status is not None:
        state.record_approval(approval_status)
    return AgentRunResult(success, "private model text", reason, state, (), None)


class _Runner:
    def __init__(self, result: AgentRunResult) -> None:
        self.result = result
        self.task: str | None = None

    def run(self, task: str) -> AgentRunResult:
        self.task = task
        return self.result


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
    assert "changed_files=calculator.py" in output
    assert f"approval={'approved' if success else '-'}" in output
    assert "private model text" not in output
    assert "cli-key-sentinel" not in output
    assert "base-secret" not in output


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
    assert agent.max_iterations == 2


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
    assert agent.approval.journal.summaries() == (
        ChangeSummary("app.py", "created", additions=1, deletions=0),
    )
    assert agent.approval.request(changed_files=("app.py",), verification_exit_code=0)
    assert output == [
        "APPROVAL_REQUIRED",
        "verification_exit=0",
        "A app.py (+1/-0)",
    ]


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
