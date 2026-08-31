"""Tests for the keyless, offline end-to-end demonstration."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_demo_runs_a_real_failure_then_verified_repair(capsys) -> None:
    from testpilot.demo import main

    assert main([]) == 0
    output = capsys.readouterr().out
    assert "BEFORE=FAIL" in output
    assert "AGENT=SUCCESS" in output
    assert "AFTER=PASS" in output


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
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from testpilot.demo import main

    target = tmp_path / "empty"
    target.mkdir()

    assert main(["--keep", str(target)]) == 0
    output = capsys.readouterr().out
    assert "BEFORE=FAIL" in output
    assert "AGENT=SUCCESS" in output
    assert "AFTER=PASS" in output
    assert (target / "calculator.py").is_file()
