"""Tests for the keyless, offline end-to-end demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_EXPECTED_OUTPUT = [
    "BEFORE=FAIL",
    "INTERRUPTED=CHECKPOINTED",
    "RESUMED=SUCCESS",
    "VERIFIED=PASS",
    "REVIEWED=PASS",
    "APPROVED=SIMULATED",
    "AFTER=PASS",
    "MEMORY_FIRST_SAVED=yes",
    "MEMORY_SECOND_RETRIEVED=1",
    "MEMORY_REUSED=yes",
]


def test_demo_shows_memory_saved_then_retrieved(tmp_path: Path) -> None:
    from testpilot.demo import run_memory_demo

    summary = run_memory_demo(tmp_path)

    assert summary.first.success is True
    assert summary.first.memory_saved == "yes"
    assert summary.first.memory_warning is None
    assert summary.second.success is True
    assert summary.second.memories_retrieved == 1
    assert summary.second.memory_saved == "duplicate"
    assert summary.second.memory_warning is None
    memory_path = tmp_path / ".testpilot" / "memories" / "entries.jsonl"
    assert memory_path.is_file()
    assert len(memory_path.read_text(encoding="utf-8").splitlines()) == 1


def test_demo_runs_a_real_failure_then_verified_repair_without_input(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testpilot.demo import main

    def forbidden_input(prompt: str = "") -> str:
        raise AssertionError("offline demo must not request approval input")

    monkeypatch.setattr("builtins.input", forbidden_input)

    assert main([]) == 0
    assert capsys.readouterr().out.splitlines() == _EXPECTED_OUTPUT


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
    assert capsys.readouterr().out.splitlines() == _EXPECTED_OUTPUT
    assert (target / "calculator.py").is_file()
    assert "return left - right" in (target / "calculator.py").read_text(encoding="utf-8")
    assert "def difference(left, right):\n    return left - right" in (
        target / "calculator.py"
    ).read_text(encoding="utf-8")
    trace = target / ".testpilot" / "traces" / "offline-demo.jsonl"
    assert trace.is_file()
    assert (target / ".testpilot" / "traces" / "offline-memory-second.jsonl").is_file()
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    run_ids = {
        event["payload"]["run_id"]
        for event in events
        if event["event"] == "checkpoint"
    }
    assert len(run_ids) == 1
    assert not list((target / ".testpilot" / "checkpoints").glob("*.json"))
    memory_path = target / ".testpilot" / "memories" / "entries.jsonl"
    assert len(memory_path.read_text(encoding="utf-8").splitlines()) == 1
