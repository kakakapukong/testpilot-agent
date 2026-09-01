"""Tests for the keyless, offline end-to-end demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_demo_runs_a_real_failure_then_verified_repair_without_input(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testpilot.demo import main

    def forbidden_input(prompt: str = "") -> str:
        raise AssertionError("offline demo must not request approval input")

    monkeypatch.setattr("builtins.input", forbidden_input)

    assert main([]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "BEFORE=FAIL",
        "INTERRUPTED=CHECKPOINTED",
        "RESUMED=SUCCESS",
        "VERIFIED=PASS",
        "REVIEWED=PASS",
        "APPROVED=SIMULATED",
        "AFTER=PASS",
    ]


@pytest.mark.parametrize("occupied_kind", ["file", "nonempty_directory"])
def test_demo_rejects_an_occupied_keep_path_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    occupied_kind: str,
) -> None:
    from testpilot.demo import main

    target = tmp_path / "occupied"
    if occupied_kind == "file":
        target.write_text("keep me", encoding="utf-8")
    else:
        target.mkdir()
        (target / "keep.txt").write_text("keep me", encoding="utf-8")

    assert main(["--keep", str(target)]) == 1
    assert "must be new or empty" in capsys.readouterr().out


def test_demo_accepts_an_existing_empty_keep_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testpilot.demo import main

    target = tmp_path / "empty"
    target.mkdir()

    def forbidden_input(prompt: str = "") -> str:
        raise AssertionError("offline demo must not request approval input")

    monkeypatch.setattr("builtins.input", forbidden_input)

    assert main(["--keep", str(target)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "BEFORE=FAIL",
        "INTERRUPTED=CHECKPOINTED",
        "RESUMED=SUCCESS",
        "VERIFIED=PASS",
        "REVIEWED=PASS",
        "APPROVED=SIMULATED",
        "AFTER=PASS",
    ]
    assert (target / "calculator.py").is_file()
    assert "return left - right" in (target / "calculator.py").read_text(encoding="utf-8")
    trace = target / ".testpilot" / "traces" / "offline-demo.jsonl"
    assert trace.is_file()
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    run_ids = {
        event["payload"]["run_id"]
        for event in events
        if event["event"] == "checkpoint"
    }
    assert len(run_ids) == 1
    assert not list((target / ".testpilot" / "checkpoints").glob("*.json"))
