"""Behaviour tests for bounded command execution and JSONL tracing."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import testpilot.command as command_module
from testpilot.command import CommandRunner, FinishTool, RunCommandTool, Verifier
from testpilot.trace import JsonlTrace


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture
def runner(workspace: Path) -> CommandRunner:
    return CommandRunner(workspace, default_timeout=5, max_timeout=10)


def test_command_runner_rejects_unlisted_program(runner: CommandRunner) -> None:
    result = runner.run(["powershell", "-Command", "Write-Output bad"])

    assert not result.ok
    assert result.error_code == "command_not_allowed"


def test_command_runner_keeps_nonzero_exit_code(runner: CommandRunner) -> None:
    result = runner.run([sys.executable, "-c", "raise SystemExit(7)"])

    assert not result.ok
    assert result.error_code == "command_failed"
    assert result.exit_code == 7


def test_command_runner_keeps_stdout_and_stderr_separate(runner: CommandRunner) -> None:
    result = runner.run(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
    )

    assert result.ok
    assert result.exit_code == 0
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


def test_command_runner_times_out(runner: CommandRunner) -> None:
    result = runner.run([sys.executable, "-c", "import time; time.sleep(2)"], timeout_seconds=0.05)

    assert not result.ok
    assert result.error_code == "timeout"
    assert result.timed_out


def test_command_runner_truncates_each_output_stream(workspace: Path) -> None:
    runner = CommandRunner(workspace, output_limit=30)
    result = runner.run(
        [
            sys.executable,
            "-c",
            "import sys; print('x' * 200_000); print('y' * 200_000, file=sys.stderr)",
        ]
    )

    assert result.ok
    assert result.truncated
    assert len(result.stdout) <= 30
    assert len(result.stderr) <= 30


def test_command_runner_uses_only_its_workspace_as_cwd(
    runner: CommandRunner, workspace: Path
) -> None:
    result = runner.run([sys.executable, "-c", "import os; print(os.getcwd())"])

    assert result.ok
    assert Path(result.stdout.strip()).resolve() == workspace.resolve()


def test_pytest_can_import_a_workspace_module_from_a_tests_subdirectory(
    runner: CommandRunner, workspace: Path
) -> None:
    (workspace / "app.py").write_text("VALUE = 42\n", encoding="utf-8")
    tests_directory = workspace / "tests"
    tests_directory.mkdir()
    (tests_directory / "test_app.py").write_text(
        "from app import VALUE\n\n\ndef test_value():\n    assert VALUE == 42\n",
        encoding="utf-8",
    )

    result = runner.run([sys.executable, "-m", "pytest", "-q"])

    assert result.ok, result.stderr or result.stdout


def test_command_runner_rejects_missing_workspace(tmp_path: Path) -> None:
    result = CommandRunner(tmp_path / "missing").run([sys.executable, "-c", "print('never')"])

    assert not result.ok
    assert result.error_code == "invalid_workspace"


def test_command_runner_rejects_non_finite_timeout_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeouts"):
        CommandRunner(tmp_path, default_timeout=float("nan"))


def test_command_runner_never_runs_a_single_shell_string(runner: CommandRunner) -> None:
    result = runner.run([f"{sys.executable} -c \"print('bad')\""])

    assert not result.ok
    assert result.error_code == "command_not_allowed"


def test_command_runner_rejects_nul_arguments(runner: CommandRunner) -> None:
    result = runner.run([sys.executable, "-c", "print('bad')\x00"])

    assert not result.ok
    assert result.error_code == "invalid_arguments"


def test_command_runner_converts_process_start_type_error_to_a_result(
    runner: CommandRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raising_popen(*args: object, **kwargs: object) -> object:
        raise TypeError("bad subprocess parameters")

    monkeypatch.setattr(command_module.subprocess, "Popen", raising_popen)

    result = runner.run([sys.executable, "-c", "print('never')"])

    assert not result.ok
    assert result.error_code == "command_start_failed"


def test_command_runner_rejects_batch_pytest_alias_even_if_it_resolves_to_a_trusted_launcher(
    runner: CommandRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = Path(sys.executable).resolve()
    monkeypatch.setattr(command_module, "_resolved_program", lambda _: trusted)
    runner._pytest_launchers = frozenset({trusted})

    assert runner.canonical_command(["pytest.cmd", "--version"]) is None
    assert runner.canonical_command(["pytest.bat", "--version"]) is None


def test_command_runner_rejects_bare_pytest_that_resolves_to_a_batch_wrapper(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_wrapper = workspace.parent / "trusted-looking-pytest.cmd"
    batch_wrapper.write_text("@echo malicious", encoding="utf-8")

    monkeypatch.setattr(command_module, "_resolve_from_path", lambda _: batch_wrapper.resolve())
    runner = CommandRunner(workspace)

    assert not runner._pytest_launchers
    assert runner.canonical_command(["pytest", "--version"]) is None
    assert runner.canonical_model_command(["pytest", "--version"]) is None
    assert runner.canonical_command([str(batch_wrapper), "--version"]) is None


def test_command_runner_rejects_workspace_program_disguised_as_python(
    runner: CommandRunner, workspace: Path
) -> None:
    fake_python = workspace / "python.exe"
    fake_python.write_text("not an executable", encoding="utf-8")

    result = runner.run([str(fake_python), "-c", "print('bad')"])

    assert not result.ok
    assert result.error_code == "command_not_allowed"


def test_command_runner_executes_trusted_absolute_python_instead_of_workspace_shadow(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "python.exe").write_text("malicious", encoding="utf-8")
    trusted_python = str(Path(sys.executable).resolve())
    original_which = command_module.shutil.which
    captured: dict[str, object] = {}

    def trusted_which(program: str) -> str | None:
        if program.lower() in {"python", "python.exe"}:
            return trusted_python
        return original_which(program)

    class FakeProcess:
        stdout = __import__("io").StringIO("")
        stderr = __import__("io").StringIO("")
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(argv: object, **kwargs: object) -> FakeProcess:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(command_module.shutil, "which", trusted_which)
    monkeypatch.setattr(command_module.subprocess, "Popen", fake_popen)
    runner = CommandRunner(workspace)

    result = runner.run(["python", "-c", "print('never runs in this test')"])

    assert result.ok
    assert captured["argv"] == (trusted_python, "-c", "print('never runs in this test')")
    assert captured["kwargs"] is not None


@pytest.mark.parametrize("secret_name", ["DEMO_API_KEY", "DEMO_KEY"])
def test_command_runner_filters_secret_environment_values(
    runner: CommandRunner, monkeypatch: pytest.MonkeyPatch, secret_name: str
) -> None:
    monkeypatch.setenv(secret_name, "should-not-reach-child")

    result = runner.run(
        [
            sys.executable,
            "-c",
            f"import os; print(os.environ.get({secret_name!r}, 'missing'))",
        ]
    )

    assert result.ok
    assert result.stdout == "missing\n"


def test_command_runner_uses_a_fresh_pyc_prefix_to_ignore_matching_workspace_pyc(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    module = workspace / "cached_module.py"
    old_source = "VALUE = 'old'\n"
    new_source = "VALUE = 'new'\n"
    assert len(old_source) == len(new_source)
    timestamp = int(time.time()) - 10
    module.write_text(old_source, encoding="utf-8")
    os.utime(module, (timestamp, timestamp))
    direct = subprocess.run(
        [sys.executable, "-c", "import cached_module; print(cached_module.VALUE)"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    assert direct.stdout == "old\n"
    assert (workspace / "__pycache__").exists()

    module.write_text(new_source, encoding="utf-8")
    os.utime(module, (timestamp, timestamp))
    stale = subprocess.run(
        [sys.executable, "-c", "import cached_module; print(cached_module.VALUE)"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    assert stale.stdout == "old\n"

    result = CommandRunner(workspace).run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '.'); import cached_module; print(cached_module.VALUE)",
        ]
    )

    assert result.ok
    assert result.stdout == "new\n"


def test_command_runner_cleans_its_temporary_pyc_prefix(
    runner: CommandRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_path = tmp_path / "pyc-prefix"
    lifecycle: list[str] = []

    class RecordingTemporaryDirectory:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> str:
            created_path.mkdir()
            lifecycle.append("entered")
            return str(created_path)

        def __exit__(self, *args: object) -> None:
            lifecycle.append("exited")
            shutil.rmtree(created_path)

    monkeypatch.setattr(command_module, "TemporaryDirectory", RecordingTemporaryDirectory)

    result = runner.run([sys.executable, "-c", "print('ok')"])

    assert result.ok
    assert lifecycle == ["entered", "exited"]
    assert not created_path.exists()


def test_safe_environment_removes_python_import_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "untrusted-path")
    monkeypatch.setenv("PYTHONHOME", "untrusted-home")
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "../outside_test.py")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted_plugin")

    environment = command_module._safe_environment()

    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "PYTHONSAFEPATH" not in environment
    assert "PYTEST_ADDOPTS" not in environment
    assert "PYTEST_PLUGINS" not in environment
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


def test_run_command_tool_validates_arguments_and_delegates(runner: CommandRunner) -> None:
    tool = RunCommandTool(runner)

    invalid = tool.execute({"argv": "not-an-array"})
    forbidden = tool.execute({"argv": [sys.executable, "-c", "print('not a test')"]})
    result = tool.execute({"argv": [sys.executable, "-m", "pytest", "--version"]})

    assert invalid.error_code == "invalid_arguments"
    assert forbidden.error_code == "command_not_allowed"
    assert result.ok
    assert result.data == {"argv": [sys.executable, "-m", "pytest", "--version"]}
    assert tool.parameters["properties"]["argv"]["type"] == "array"


def test_run_command_tool_rejects_pytest_targets_outside_the_workspace(
    runner: CommandRunner, workspace: Path
) -> None:
    outside = workspace.parent / "outside_test.py"
    outside.write_text("def test_escape():\n    assert True\n", encoding="utf-8")

    result = RunCommandTool(runner).execute(
        {"argv": [sys.executable, "-m", "pytest", str(outside), "-q"]}
    )

    assert not result.ok
    assert result.error_code == "command_not_allowed"


@pytest.mark.parametrize(
    "arguments",
    [
        ["-p", "untrusted_plugin"],
        ["-c", "alternate.ini"],
        ["--rootdir", ".."],
        ["--junitxml", "source.py"],
        ["@pytest-arguments.txt"],
    ],
)
def test_model_pytest_rejects_options_that_can_load_or_overwrite_assets(
    runner: CommandRunner, arguments: list[str]
) -> None:
    assert runner.canonical_model_command([sys.executable, "-m", "pytest", *arguments]) is None


def test_model_pytest_accepts_workspace_targets_and_reporting_options(
    runner: CommandRunner, workspace: Path
) -> None:
    target = workspace / "check_example.py"
    target.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    canonical = runner.canonical_model_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "check_example.py::test_ok",
            "-q",
            "--tb=short",
        ]
    )

    assert canonical is not None


def test_finish_tool_only_records_a_request() -> None:
    result = FinishTool().execute({"summary": "tests are green"})

    assert result.ok
    assert result.data == {"finish_requested": True, "summary": "tests are green"}


def test_verifier_copies_command_and_always_uses_its_runner(runner: CommandRunner) -> None:
    command = [sys.executable, "-m", "pytest", "--version"]
    verifier = Verifier(runner, command)
    command[-1] = "../outside_test.py"

    result = verifier.verify()

    assert isinstance(verifier.command, tuple)
    assert result.ok
    assert result.stdout.startswith("pytest ")


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-c", "raise SystemExit(0)"],
        [sys.executable, "verify.py"],
        [sys.executable, "-m", "custom_verifier"],
    ],
)
def test_verifier_rejects_custom_python_entry_points(
    runner: CommandRunner, command: list[str]
) -> None:
    with pytest.raises(ValueError, match="pytest"):
        Verifier(runner, command)


def test_verifier_protects_an_explicit_pytest_node_target(workspace: Path) -> None:
    checks = workspace / "checks"
    checks.mkdir()
    (checks / "case.py").write_text("", encoding="utf-8")

    verifier = Verifier(
        CommandRunner(workspace),
        [
            sys.executable,
            "-m",
            "pytest",
            "checks/case.py::TestCase::test_it",
            "-q",
        ],
    )

    assert verifier.protected_patterns == ("checks/case.py",)


def test_verifier_protects_an_explicit_pytest_directory_subtree(workspace: Path) -> None:
    checks = workspace / "checks"
    checks.mkdir()

    verifier = Verifier(CommandRunner(workspace), [sys.executable, "-m", "pytest", "checks", "-q"])

    assert verifier.protected_patterns == ("checks/**",)


def test_verifier_rejects_pytest_assets_outside_the_workspace(
    workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="pytest"):
        Verifier(
            CommandRunner(workspace),
            [sys.executable, "-m", "pytest", str(outside)],
        )


def test_jsonl_trace_writes_parseable_events_without_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRACE_TEST_SECRET", "trace-secret-value")
    path = tmp_path / "nested" / "run.jsonl"
    trace = JsonlTrace(path)

    trace.record("tool_result", {"tool": "run_command", "exit_code": 0})
    trace.record("stop", {"reason": "verified"})

    lines = path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert [event["event"] for event in events] == ["tool_result", "stop"]
    assert all("timestamp" in event for event in events)
    assert "trace-secret-value" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("secret_name", ["DEMO_API_KEY", "DEMO_KEY"])
def test_jsonl_trace_redacts_sensitive_environment_values_in_nested_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secret_name: str
) -> None:
    monkeypatch.setenv(secret_name, "trace-output-sentinel")
    path = tmp_path / "trace.jsonl"
    trace = JsonlTrace(path)

    trace.record("tool_result", {"result": {"stdout": "token=trace-output-sentinel"}})

    contents = path.read_text(encoding="utf-8")
    assert "trace-output-sentinel" not in contents
    assert "[REDACTED]" in contents


@pytest.mark.parametrize(
    "unsafe_key",
    [
        "api_key",
        "accessToken",
        "nested_secret_name",
        "db_password",
        "credential_ref",
        "environment",
    ],
)
def test_jsonl_trace_rejects_nested_sensitive_payload_keys_without_altering_existing_file(
    tmp_path: Path, unsafe_key: str
) -> None:
    path = tmp_path / "trace.jsonl"
    trace = JsonlTrace(path)
    trace.record("safe", {"value": "kept"})
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="must not be recorded"):
        trace.record("unsafe", {"outer": {unsafe_key: "sentinel-secret"}})

    assert path.read_text(encoding="utf-8") == before
    assert "sentinel-secret" not in path.read_text(encoding="utf-8")


def test_jsonl_trace_rejects_non_json_payload_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = JsonlTrace(path)

    with pytest.raises(ValueError, match="JSON-native"):
        trace.record("bad", {"value": object()})

    assert not path.exists()
