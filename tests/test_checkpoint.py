from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from testpilot.approval import JournalSnapshot
from testpilot.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointRequest,
    CheckpointStore,
    FileFingerprint,
    RunCheckpoint,
    decode_run_state,
    encode_run_state,
    workspace_identity,
)
from testpilot.context import BoundedContext
from testpilot.types import RunPhase, RunState


def _request(root: Path) -> CheckpointRequest:
    return CheckpointRequest(
        task="Fix app.py",
        verifier=(sys.executable, "-m", "pytest", "-q"),
        max_iterations=12,
        trace_path=".testpilot/traces/run.jsonl",
    )


def _context() -> BoundedContext:
    context = BoundedContext(
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": "Fix app.py"},
    )
    context.append_transaction({"role": "assistant", "content": "inspected"})
    return context


def _minimal_checkpoint(root: Path, run_id: str) -> RunCheckpoint:
    now = datetime.now(UTC).isoformat()
    return RunCheckpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        run_id=run_id,
        workspace_identity=workspace_identity(root),
        request=_request(root),
        context=_context().snapshot(),
        state=encode_run_state(RunState()),
        last_call_signature=None,
        journal=(),
        fingerprints=(),
        lifecycle_status="active",
        safe_point=1,
        created_at=now,
        updated_at=now,
    )


def test_checkpoint_store_round_trip_is_atomic_and_json_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = _minimal_checkpoint(tmp_path, store.new_run_id())
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
    ) -> None:
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr("testpilot.checkpoint.os.replace", recording_replace)

    store.save(checkpoint)
    loaded = store.load(checkpoint.run_id)

    assert loaded == checkpoint
    assert replacements[0][0].parent == replacements[0][1].parent
    assert list(store.directory.glob("*.tmp")) == []
    assert json.loads(store.path_for(checkpoint.run_id).read_text(encoding="utf-8"))[
        "schema_version"
    ] == 1


def test_checkpoint_codec_round_trips_binary_journal_and_fingerprints(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    base = _minimal_checkpoint(tmp_path, store.new_run_id())
    checkpoint = replace(
        base,
        journal=(JournalSnapshot("app.py", b"old\x00bytes\n", 0o640, ()),),
        fingerprints=(FileFingerprint("app.py", "file", 0o640, "a" * 64),),
        last_call_signature='[{"name":"read_file"}]',
    )

    store.save(checkpoint)
    loaded = store.load(checkpoint.run_id)

    assert loaded.journal[0].original == b"old\x00bytes\n"
    assert loaded.fingerprints == checkpoint.fingerprints
    assert loaded.last_call_signature == '[{"name":"read_file"}]'


def test_run_state_codec_round_trips_every_stable_field() -> None:
    state = RunState(
        phase=RunPhase.REVIEW,
        iteration=5,
        edit_count=2,
        source_edit_count=2,
        changed_files={"app.py", "src/lib.py"},
        last_verify_exit_code=0,
        verified_after_last_edit=False,
        consecutive_no_progress=1,
        stop_reason="model_transient_failure",
        approval_status=None,
        review_status="changes_requested",
        review_rounds=1,
        review_rework_count=1,
        reviewed_edit_count=2,
        reviewed_source_edit_count=2,
    )

    restored = decode_run_state(encode_run_state(state))

    assert restored == state
    payload = encode_run_state(state)
    payload["changed_files"].append("changed.py")
    assert restored.changed_files == {"app.py", "src/lib.py"}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"unknown": True}),
        lambda payload: payload.pop("iteration"),
        lambda payload: payload.update({"iteration": True}),
        lambda payload: payload.update({"phase": "unknown"}),
        lambda payload: payload.update({"changed_files": ["b.py", "a.py"]}),
        lambda payload: payload.update({"source_edit_count": 2, "edit_count": 1}),
        lambda payload: payload.update({"review_rework_count": 2}),
        lambda payload: payload.update({"approval_status": "pending"}),
    ],
)
def test_run_state_codec_rejects_invalid_payloads(mutation: object) -> None:
    payload = encode_run_state(RunState())
    mutation(payload)  # type: ignore[operator]

    with pytest.raises(CheckpointError) as caught:
        decode_run_state(payload)

    assert caught.value.code == "checkpoint_invalid"


@pytest.mark.parametrize(
    "run_id",
    ["", "abc", "../escape", "A" * 16, "0" * 15, "0" * 17, "0" * 15 + "g"],
)
def test_checkpoint_store_rejects_invalid_run_ids(tmp_path: Path, run_id: str) -> None:
    store = CheckpointStore(tmp_path)

    with pytest.raises(CheckpointError) as caught:
        store.load(run_id)

    assert caught.value.code == "checkpoint_invalid"


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        b"[]",
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":NaN}',
        b"\xff\xfe",
    ],
)
def test_checkpoint_store_rejects_malformed_or_noncanonical_json(
    tmp_path: Path,
    raw: bytes,
) -> None:
    store = CheckpointStore(tmp_path)
    run_id = store.new_run_id()
    store.path_for(run_id).write_bytes(raw)

    with pytest.raises(CheckpointError) as caught:
        store.load(run_id)

    assert caught.value.code == "checkpoint_invalid"


@pytest.mark.parametrize(
    "location",
    [
        ("root", "unknown"),
        ("request", "unknown"),
        ("runtime", "unknown"),
        ("workspace", "unknown"),
        ("journal", "unknown"),
        ("lifecycle", "unknown"),
    ],
)
def test_checkpoint_store_rejects_unknown_schema_fields(
    tmp_path: Path,
    location: tuple[str, str],
) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = _minimal_checkpoint(tmp_path, store.new_run_id())
    store.save(checkpoint)
    path = store.path_for(checkpoint.run_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    section, key = location
    if section == "root":
        payload[key] = True
    else:
        payload[section][key] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointError) as caught:
        store.load(checkpoint.run_id)

    assert caught.value.code == "checkpoint_invalid"


def test_checkpoint_store_rejects_invalid_nested_values(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = _minimal_checkpoint(tmp_path, store.new_run_id())
    store.save(checkpoint)
    path = store.path_for(checkpoint.run_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["lifecycle"]["safe_point"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointError) as caught:
        store.load(checkpoint.run_id)

    assert caught.value.code == "checkpoint_invalid"


def test_checkpoint_store_rejects_invalid_base64_and_duplicate_records(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    base = _minimal_checkpoint(tmp_path, store.new_run_id())
    checkpoint = replace(
        base,
        journal=(JournalSnapshot("app.py", b"old\n", 0o644, ()),),
        fingerprints=(FileFingerprint("app.py", "file", 0o644, "b" * 64),),
    )
    store.save(checkpoint)
    path = store.path_for(checkpoint.run_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["journal"]["snapshots"][0]["original"] = "***"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointError) as caught:
        store.load(checkpoint.run_id)
    assert caught.value.code == "checkpoint_invalid"

    store.save(checkpoint)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["journal"]["snapshots"].append(payload["journal"]["snapshots"][0])
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointError) as caught:
        store.load(checkpoint.run_id)
    assert caught.value.code == "checkpoint_invalid"


def test_checkpoint_file_contains_no_environment_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = ("api-key-sentinel", "base-url-secret", "token-sentinel")
    monkeypatch.setenv("OPENAI_API_KEY", secrets[0])
    monkeypatch.setenv("OPENAI_BASE_URL", f"https://user:{secrets[1]}@example.invalid")
    monkeypatch.setenv("CHECKPOINT_TEST_TOKEN", secrets[2])
    store = CheckpointStore(tmp_path)
    checkpoint = _minimal_checkpoint(tmp_path, store.new_run_id())

    store.save(checkpoint)

    contents = store.path_for(checkpoint.run_id).read_text(encoding="utf-8")
    assert all(secret not in contents for secret in secrets)


def test_checkpoint_store_enforces_size_limit_before_writing(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, max_checkpoint_bytes=100)
    checkpoint = _minimal_checkpoint(tmp_path, store.new_run_id())

    with pytest.raises(CheckpointError) as caught:
        store.save(checkpoint)

    assert caught.value.code == "checkpoint_too_large"
    assert not store.path_for(checkpoint.run_id).exists()


def test_checkpoint_store_rejects_an_oversized_existing_file(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, max_checkpoint_bytes=100)
    run_id = store.new_run_id()
    store.path_for(run_id).write_bytes(b"x" * 101)

    with pytest.raises(CheckpointError) as caught:
        store.load(run_id)

    assert caught.value.code == "checkpoint_too_large"


def test_failed_atomic_replace_preserves_previous_checkpoint_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = _minimal_checkpoint(tmp_path, store.new_run_id())
    store.save(checkpoint)
    path = store.path_for(checkpoint.run_id)
    before = path.read_bytes()

    def failing_replace(source: object, target: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("testpilot.checkpoint.os.replace", failing_replace)

    with pytest.raises(CheckpointError) as caught:
        store.save(replace(checkpoint, safe_point=2))

    assert caught.value.code == "checkpoint_save_failed"
    assert path.read_bytes() == before
    assert list(store.directory.glob("*.tmp")) == []


def test_failed_fsync_cleans_temp_without_creating_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = _minimal_checkpoint(tmp_path, store.new_run_id())

    def failing_fsync(file_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr("testpilot.checkpoint.os.fsync", failing_fsync)

    with pytest.raises(CheckpointError) as caught:
        store.save(checkpoint)

    assert caught.value.code == "checkpoint_save_failed"
    assert not store.path_for(checkpoint.run_id).exists()
    assert list(store.directory.glob("*.tmp")) == []


def test_checkpoint_store_rejects_checkpoint_file_symlink(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path)
    run_id = store.new_run_id()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        store.path_for(run_id).symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this system")

    with pytest.raises(CheckpointError) as caught:
        store.load(run_id)

    assert caught.value.code == "checkpoint_invalid"


def test_checkpoint_store_rejects_testpilot_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / ".testpilot"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this system")

    with pytest.raises(CheckpointError) as caught:
        CheckpointStore(tmp_path)

    assert caught.value.code == "checkpoint_invalid"


def test_checkpoint_delete_removes_only_the_validated_run_file(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = _minimal_checkpoint(tmp_path, store.new_run_id())
    store.save(checkpoint)

    store.delete(checkpoint.run_id)

    assert not store.path_for(checkpoint.run_id).exists()
    store.delete(checkpoint.run_id)
