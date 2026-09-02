from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path
from queue import Queue
from typing import Any

import pytest

from testpilot.types import AgentRunResult, RunState
from testpilot.web import WebApp, make_server


class _FakeRunner:
    def __init__(self, input_fn, output_fn, blocker: Queue[str]) -> None:
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.blocker = blocker
        self.task = ""

    def run(self, task: str, resume: object = None) -> AgentRunResult:
        del resume
        self.task = task
        self.output_fn("APPROVAL_REQUIRED")
        self.output_fn("verification_exit=0")
        self.output_fn('M "calculator.py" (+1/-1)')
        self.blocker.put("waiting")
        response = self.input_fn("Accept verified changes? [y/N]: ")
        approved = isinstance(response, str) and response.strip().lower() in {"y", "yes"}
        state = RunState()
        state.changed_files.add("calculator.py")
        state.last_verify_exit_code = 0
        state.review_status = "passed"
        state.approval_status = "approved" if approved else "rejected"
        return AgentRunResult(
            success=approved,
            final_text="",
            stop_reason="verified" if approved else "rejected",
            state=state,
            messages=(),
            memory_saved="yes" if approved else "no",
        )


def _start_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, str, int, Queue[str]]:
    monkeypatch.setenv("OPENAI_API_KEY", "web-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "web-test-model")
    blocker: Queue[str] = Queue()

    def factory(config, journal, checkpoint, input_fn, output_fn):
        del config, journal, checkpoint
        return _FakeRunner(input_fn, output_fn, blocker)

    app = WebApp(runner_factory=factory)
    server = make_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    (tmp_path / "keep.txt").write_text("ok", encoding="utf-8")
    return server, str(host), int(port), blocker


def _request(
    host: str,
    port: int,
    method: str,
    path: str,
    body: dict[str, object] | None = None,
) -> tuple[int, str, str]:
    connection = HTTPConnection(host, port, timeout=5)
    try:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        return response.status, response.getheader("Content-Type") or "", data
    finally:
        connection.close()


def test_pytest_temp_workspaces_are_not_remembered(tmp_path: Path) -> None:
    from testpilot.web import _usable_workspace_path, remember_workspace

    junk = tmp_path / "pytest-of-29148" / "pytest-1" / "test_x0"
    junk.mkdir(parents=True)
    remember_workspace(str(junk))
    assert _usable_workspace_path(str(junk)) is False


def test_saved_credentials_parser_reads_openai_fields_only(tmp_path: Path) -> None:
    from testpilot.cli import parse_saved_credentials

    env_file = tmp_path / "web.env"
    env_file.write_text(
        "OPENAI_API_KEY=file-secret-key\nOPENAI_MODEL=deepseek-chat\nOTHER=nope\n",
        encoding="utf-8",
    )
    parsed = parse_saved_credentials(env_file)
    assert parsed == {
        "OPENAI_API_KEY": "file-secret-key",
        "OPENAI_MODEL": "deepseek-chat",
    }


def test_bootstrap_hides_secrets_and_lists_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, host, port, _ = _start_app(tmp_path, monkeypatch)
    try:
        status, _, body = _request(host, port, "GET", "/api/bootstrap")
    finally:
        server.shutdown()

    assert status == 200
    payload = json.loads(body)
    assert payload["credentials_ready"] is True
    assert "web-test-key" not in body
    assert "OPENAI_API_KEY" not in body
    assert payload["default_verify"]


def test_console_page_does_not_embed_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, host, port, _ = _start_app(tmp_path, monkeypatch)
    try:
        status, content_type, body = _request(host, port, "GET", "/")
    finally:
        server.shutdown()

    assert status == 200
    assert "text/html" in content_type
    assert "web-test-key" not in body
    assert "TestPilot" in body


def test_missing_workspace_does_not_start_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, host, port, _ = _start_app(tmp_path, monkeypatch)
    try:
        status, _, body = _request(
            host,
            port,
            "POST",
            "/api/runs",
            {
                "workspace": str(tmp_path / "missing"),
                "verify": "python -m pytest -q",
                "task": "fix tests",
            },
        )
    finally:
        server.shutdown()

    assert status == 400
    payload = json.loads(body)
    assert payload["error"] == "workspace must be an existing directory"


def test_second_run_is_conflict_while_approval_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, host, port, blocker = _start_app(tmp_path, monkeypatch)
    try:
        first = _request(
            host,
            port,
            "POST",
            "/api/runs",
            {
                "workspace": str(tmp_path),
                "verify": f'"{__import__("sys").executable}" -m pytest -q',
                "task": "fix tests",
            },
        )
        blocker.get(timeout=5)
        second = _request(
            host,
            port,
            "POST",
            "/api/runs",
            {
                "workspace": str(tmp_path),
                "verify": f'"{__import__("sys").executable}" -m pytest -q',
                "task": "another task",
            },
        )
        _request(host, port, "POST", "/api/runs/current/approval", {"decision": "rejected"})
    finally:
        server.shutdown()

    assert first[0] == 200
    assert second[0] == 409


def test_reject_returns_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, host, port, blocker = _start_app(tmp_path, monkeypatch)
    try:
        started = _request(
            host,
            port,
            "POST",
            "/api/runs",
            {
                "workspace": str(tmp_path),
                "verify": f'"{__import__("sys").executable}" -m pytest -q',
                "task": "fix tests",
            },
        )
        blocker.get(timeout=5)
        decided = _request(
            host,
            port,
            "POST",
            "/api/runs/current/approval",
            {"decision": "rejected"},
        )
    finally:
        server.shutdown()

    assert started[0] == 200
    assert decided[0] == 200
    payload = json.loads(decided[2])
    assert payload["approval"] == "rejected"
    assert payload["STATUS"] == "FAILED"
