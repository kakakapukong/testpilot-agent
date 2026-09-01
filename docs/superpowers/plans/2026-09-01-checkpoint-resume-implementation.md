# Checkpoint Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add safe-node checkpoint persistence so a TestPilot repair can resume under the same run ID after process or model interruption while preserving rollback, verification, review, and approval guarantees.

**Architecture:** A host-only checkpoint module serializes bounded Repair Agent context, validated RunState, durable ChangeJournal snapshots, repeated-call state, and current file fingerprints into an atomic JSON file under .testpilot/checkpoints. AgentRunner saves only at complete transaction boundaries; the CLI validates and rebuilds a run before any model call, invalidates old success evidence, and reuses the fixed verifier and trace. Workspace tools structurally hide checkpoint files from both Agents.

**Tech Stack:** Python 3.11+, dataclasses, pathlib, hashlib, base64, strict JSON, tempfile/os.replace, existing BoundedContext/ChangeJournal/AgentRunner/JsonlTrace, pytest, Ruff.

---

## File map

- Create src/testpilot/checkpoint.py: checkpoint value types, strict codec, atomic store, fingerprints, lifecycle, and resume reconstruction.
- Create tests/test_checkpoint.py: schema, store, fingerprint, lifecycle, corruption, secret, and restore coverage.
- Modify src/testpilot/context.py and tests/test_context_and_model.py: validated bounded-context snapshot and restoration.
- Modify src/testpilot/approval.py and tests/test_approval.py: durable ChangeJournal snapshot export/import.
- Modify src/testpilot/workspace.py and tests/test_workspace_tools.py: host-private checkpoint path boundary for every file tool.
- Modify src/testpilot/types.py and tests/test_types.py: checkpoint metadata in AgentRunResult and resume-proof invalidation in RunState.
- Modify src/testpilot/agent.py, tests/test_agent.py, and tests/test_agent_e2e.py: safe-point saves, cumulative iterations, resume entry, finalization, and end-to-end restart.
- Modify src/testpilot/cli.py and tests/test_cli.py: fresh/resume argument modes, checkpoint assembly, stable output, existing trace reuse, and pre-model validation.
- Modify src/testpilot/demo.py and tests/test_demo.py: deterministic keyless interruption and recovery across two runner instances.
- Modify README.md, submission/录屏与提交清单.md, and submission/README.txt.template: public usage and assessment evidence.

### Task 1: Make bounded Repair context round-trip safely

**Files:**
- Modify: src/testpilot/context.py:10-55
- Modify: tests/test_context_and_model.py:1-90

- [ ] **Step 1: Write failing context snapshot tests**

Add these tests:

```python
def test_context_snapshot_round_trip_preserves_complete_bounded_groups() -> None:
    from testpilot.context import BoundedContext

    context = BoundedContext(
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": "repair"},
        max_recent_groups=2,
        max_tool_content_chars=40,
    )
    context.append_transaction(_assistant_call("one"), [_tool_message("one", "first")])
    context.append_transaction(_assistant_call("two"), [_tool_message("two", "second")])

    restored = BoundedContext.from_snapshot(context.snapshot())

    assert restored.messages() == context.messages()
    copied = restored.snapshot()
    copied["groups"][0][0]["content"] = "changed"
    assert restored.messages()[2]["content"] == "I will inspect the file."


def test_context_restore_rejects_an_incomplete_tool_transaction() -> None:
    from testpilot.context import BoundedContext

    payload = {
        "developer": {"role": "developer", "content": "rules"},
        "user": {"role": "user", "content": "repair"},
        "groups": [[_assistant_call("one")]],
        "max_recent_groups": 8,
        "max_tool_content_chars": 8_000,
    }

    with pytest.raises(ValueError, match="tool"):
        BoundedContext.from_snapshot(payload)
```

Add parametrized cases for an unknown top-level key, missing key, non-mapping payload, invalid anchor role, duplicate tool result, non-JSON value, boolean numeric limit, and group count above max_recent_groups. Add one positive case proving oversized tool content is bounded through the existing visible truncation rule. Each invalid payload must raise TypeError or ValueError without mutating its input.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_context_and_model.py -q
```

Expected: failures report that BoundedContext has no snapshot or from_snapshot API.

- [ ] **Step 3: Add a strict snapshot shape**

Add these methods and reconstruct every group through append_transaction:

```python
_CONTEXT_SNAPSHOT_KEYS = frozenset(
    {
        "developer",
        "user",
        "groups",
        "max_recent_groups",
        "max_tool_content_chars",
    }
)


def snapshot(self) -> dict[str, Any]:
    return {
        "developer": _json_copy(self._developer),
        "user": _json_copy(self._user),
        "groups": [
            [_json_copy(message) for message in group]
            for group in self._groups
        ],
        "max_recent_groups": self._max_recent_groups,
        "max_tool_content_chars": self._max_tool_content_chars,
    }


@classmethod
def from_snapshot(cls, payload: Mapping[str, Any]) -> "BoundedContext":
    if not isinstance(payload, Mapping):
        raise TypeError("context snapshot must be an object")
    if set(payload) != _CONTEXT_SNAPSHOT_KEYS:
        raise ValueError("context snapshot fields are invalid")
    max_groups = payload["max_recent_groups"]
    max_chars = payload["max_tool_content_chars"]
    if type(max_groups) is not int or max_groups < 0:
        raise ValueError("context group limit is invalid")
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("context tool-content limit is invalid")
    groups = payload["groups"]
    if not isinstance(groups, list) or len(groups) > max_groups:
        raise ValueError("context groups are invalid")
    developer = payload["developer"]
    user = payload["user"]
    if not isinstance(developer, Mapping) or not isinstance(user, Mapping):
        raise TypeError("context anchors must be objects")
    context = cls(
        developer,
        user,
        max_recent_groups=max_groups,
        max_tool_content_chars=max_chars,
    )
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("context transaction is invalid")
        if not all(isinstance(message, Mapping) for message in group):
            raise TypeError("context transaction messages must be objects")
        context.append_transaction(group[0], group[1:])
    return context
```

Place snapshot and from_snapshot on BoundedContext. Keep _json_copy as the only copying/JSON-native gate so NaN, objects, and non-string mapping keys remain rejected.

- [ ] **Step 4: Run context tests and Ruff**

Run:

```powershell
python -m pytest tests/test_context_and_model.py -q
python -m ruff check src/testpilot/context.py tests/test_context_and_model.py
```

Expected: all selected tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/testpilot/context.py tests/test_context_and_model.py
git commit -m "feat: serialize bounded repair context"
```

### Task 2: Persist the rollback journal without weakening its path checks

**Files:**
- Modify: src/testpilot/approval.py:18-170
- Modify: tests/test_approval.py:1-180

- [ ] **Step 1: Write failing journal export/import tests**

Add these tests:

```python
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

    restored = ChangeJournal(root)
    restored.restore_snapshots(first.export_snapshots())
    restored.rollback()

    assert target.read_bytes() == b"original\x00bytes\n"
    assert stat.S_IMODE(target.stat().st_mode) == original_mode
    assert not (root / "new").exists()


def test_change_journal_restore_rejects_duplicate_or_escaping_paths(tmp_path: Path) -> None:
    from testpilot.approval import ApprovalError, JournalSnapshot

    journal = ChangeJournal(tmp_path)
    valid = JournalSnapshot("app.py", b"old\n", 0o644, ())

    with pytest.raises(ApprovalError):
        journal.restore_snapshots((valid, valid))
    with pytest.raises(ApprovalError):
        ChangeJournal(tmp_path).restore_snapshots(
            (JournalSnapshot("../outside.py", b"old\n", None, ()),)
        )
```

Add cases for a journal that is already populated, inconsistent original/mode pairs, non-bytes original data, non-ancestor missing parent, absolute paths, root paths, and mutation of exported values. Confirm import itself never reads or writes the tracked file.

- [ ] **Step 2: Run journal tests and verify RED**

Run:

```powershell
python -m pytest tests/test_approval.py -q
```

Expected: import fails because JournalSnapshot, export_snapshots, and restore_snapshots are absent.

- [ ] **Step 3: Add the immutable public journal record**

Add:

```python
@dataclass(frozen=True)
class JournalSnapshot:
    path: str
    original: bytes | None
    mode: int | None
    missing_parents: tuple[str, ...]
```

Export only normalized POSIX-relative paths:

```python
def export_snapshots(self) -> tuple[JournalSnapshot, ...]:
    return tuple(
        JournalSnapshot(
            path=relative,
            original=None if snapshot.original is None else bytes(snapshot.original),
            mode=snapshot.mode,
            missing_parents=tuple(
                parent.relative_to(self.root).as_posix()
                for parent in snapshot.missing_parents
            ),
        )
        for relative, snapshot in sorted(self._snapshots.items())
    )
```

Implement restore_snapshots as an all-or-nothing operation: validate the complete sequence into a temporary dictionary, call _normalize for every target and parent, require unique targets, require every missing parent to be a strict ancestor of its target inside root, require original and mode to be either both absent or a valid bytes/integer pair, then assign self._snapshots once. Convert all input/type/path failures to the content-free ApprovalError("could not restore workspace change journal").

- [ ] **Step 4: Run journal and workspace regressions**

Run:

```powershell
python -m pytest tests/test_approval.py tests/test_workspace_tools.py -q
python -m ruff check src/testpilot/approval.py tests/test_approval.py
```

Expected: all selected tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/testpilot/approval.py tests/test_approval.py
git commit -m "feat: persist rollback journal snapshots"
```

### Task 3: Make checkpoint storage invisible to both Agents

**Files:**
- Modify: src/testpilot/workspace.py:13-105, 110-470, 530-595
- Modify: tests/test_workspace_tools.py:1-70, 178-300, 705-835

- [ ] **Step 1: Write failing private-boundary tests**

Add:

```python
@pytest.mark.parametrize("operation", ["read", "search", "write", "edit", "list"])
def test_checkpoint_tree_is_host_private_for_every_workspace_operation(
    tmp_path: Path,
    operation: str,
) -> None:
    private = tmp_path / ".testpilot" / "checkpoints"
    private.mkdir(parents=True)
    (private / "0123456789abcdef.json").write_text('{"task":"private"}', encoding="utf-8")
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspaceError) as caught:
        if operation == "read":
            workspace.read_file(".testpilot/checkpoints/0123456789abcdef.json")
        elif operation == "search":
            workspace.search_text("private", ".testpilot/checkpoints")
        elif operation == "write":
            workspace.write_file(".testpilot/checkpoints/new.json", "{}")
        elif operation == "edit":
            workspace.edit_file(
                ".testpilot/checkpoints/0123456789abcdef.json",
                "private",
                "changed",
            )
        else:
            workspace.list_files(".testpilot/checkpoints")
    assert caught.value.code == "private_path"


def test_root_listing_and_search_prune_checkpoint_tree(tmp_path: Path) -> None:
    private = tmp_path / ".testpilot" / "checkpoints"
    private.mkdir(parents=True)
    (private / "0123456789abcdef.json").write_text("sentinel", encoding="utf-8")
    (tmp_path / "app.py").write_text("sentinel\n", encoding="utf-8")
    workspace = Workspace(tmp_path)

    assert workspace.list_files(".")["files"] == ["app.py"]
    assert workspace.search_text("sentinel")["matches"] == [
        {"path": "app.py", "line": 1, "text": "sentinel"}
    ]
```

Add a symlink-alias case when supported and constructor validation for a string, blank pattern, and explicitly empty private pattern tuple.

- [ ] **Step 2: Run workspace tests and verify RED**

Run:

```powershell
python -m pytest tests/test_workspace_tools.py -q
```

Expected: current read/list/search operations expose the checkpoint file.

- [ ] **Step 3: Add a separate host-private policy**

Add:

```python
DEFAULT_PRIVATE_PATTERNS = (".testpilot/checkpoints/**",)
```

Add private_patterns: Sequence[str] = DEFAULT_PRIVATE_PATTERNS to Workspace.__init__, validate it with the same strict sequence rules as protected_patterns, and store an immutable tuple. Keep protected_path for write-protected tests and traces; use private_path for paths hidden from every tool.

Add these helpers:

```python
def _is_private(self, resolved: Path) -> bool:
    return is_protected_relative_path(
        self._relative(resolved),
        self.private_patterns,
        case_insensitive=os.name == "nt",
    )


def _assert_visible(self, resolved: Path) -> None:
    if self._is_private(resolved):
        raise WorkspaceError("private_path", "path is reserved for host state")
```

Call _assert_visible on direct read/list/search/write/edit targets. In _iter_files, skip a private candidate before adding a file or descending into a directory. Use the canonical resolved path so symlink aliases cannot reveal the checkpoint tree.

- [ ] **Step 4: Run workspace, tool, and CLI-adjacent regressions**

Run:

```powershell
python -m pytest tests/test_workspace_tools.py tests/test_cli.py tests/test_reviewer.py -q
python -m ruff check src/testpilot/workspace.py tests/test_workspace_tools.py
```

Expected: all selected tests pass and neither Agent registry can observe checkpoint files.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/testpilot/workspace.py tests/test_workspace_tools.py
git commit -m "feat: hide host checkpoint state from agents"
```

### Task 4: Build the strict checkpoint codec and atomic store

**Files:**
- Create: src/testpilot/checkpoint.py
- Create: tests/test_checkpoint.py
- Modify: src/testpilot/types.py:149-225
- Modify: tests/test_types.py:145-255

- [ ] **Step 1: Write failing value, schema, and store tests**

Create tests/test_checkpoint.py with these core cases:

```python
def _request(root: Path, trace: Path) -> CheckpointRequest:
    return CheckpointRequest(
        task="Fix app.py",
        verifier=(sys.executable, "-m", "pytest", "-q"),
        max_iterations=12,
        trace_path=trace.relative_to(root).as_posix(),
    )


def _minimal_checkpoint(root: Path, run_id: str) -> RunCheckpoint:
    now = datetime.now(UTC).isoformat()
    context = BoundedContext(
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": "Fix app.py"},
    )
    state = {
        "phase": "discover",
        "iteration": 0,
        "edit_count": 0,
        "source_edit_count": 0,
        "changed_files": [],
        "last_verify_exit_code": None,
        "verified_after_last_edit": False,
        "consecutive_no_progress": 0,
        "stop_reason": None,
        "approval_status": None,
        "review_status": None,
        "review_rounds": 0,
        "review_rework_count": 0,
        "reviewed_edit_count": None,
        "reviewed_source_edit_count": None,
    }
    return RunCheckpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        run_id=run_id,
        workspace_identity=workspace_identity(root),
        request=_request(root, root / ".testpilot" / "traces" / "run.jsonl"),
        context=context.snapshot(),
        state=state,
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
    root = tmp_path / "repo"
    root.mkdir()
    store = CheckpointStore(root)
    checkpoint = _minimal_checkpoint(root, store.new_run_id())
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr("testpilot.checkpoint.os.replace", recording_replace)
    store.save(checkpoint)
    loaded = store.load(checkpoint.run_id)

    assert loaded == checkpoint
    assert replacements[0][0].parent == replacements[0][1].parent
    assert list(store.directory.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "run_id",
    ["", "abc", "../escape", "A" * 16, "0" * 15, "0" * 17],
)
def test_checkpoint_store_rejects_invalid_run_ids(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(CheckpointError) as caught:
        CheckpointStore(tmp_path).load(run_id)
    assert caught.value.code == "checkpoint_invalid"
```

Add explicit tests for duplicate JSON keys, NaN/Infinity, missing and unknown fields at every level, wrong scalar/container types, booleans used as integers, unknown phase/status, invalid base64, duplicate journal/fingerprint paths, more than the configured byte limit, checkpoint-file symlinks, a .testpilot parent symlink escaping the workspace, failed fsync/replace cleanup, and a failed replacement preserving the previous valid file.

Set OPENAI_API_KEY, OPENAI_BASE_URL, and a TOKEN-named environment value to sentinels; save a checkpoint and assert none appears in the file bytes.

- [ ] **Step 2: Run checkpoint tests and verify RED**

Run:

```powershell
python -m pytest tests/test_checkpoint.py tests/test_types.py -q
```

Expected: collection fails because testpilot.checkpoint and result metadata do not exist.

- [ ] **Step 3: Define the closed public value model**

Create these public shapes in checkpoint.py:

```python
CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_MAX_CHECKPOINT_BYTES = 16_000_000
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{16}\Z")


class CheckpointError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__("checkpoint operation failed")
        self.code = code


@dataclass(frozen=True)
class CheckpointRequest:
    task: str
    verifier: tuple[str, ...]
    max_iterations: int
    trace_path: str


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    kind: str
    mode: int | None
    sha256: str | None


@dataclass(frozen=True)
class RunCheckpoint:
    schema_version: int
    run_id: str
    workspace_identity: str
    request: CheckpointRequest
    context: Mapping[str, Any]
    state: Mapping[str, Any]
    last_call_signature: str | None
    journal: tuple[JournalSnapshot, ...]
    fingerprints: tuple[FileFingerprint, ...]
    lifecycle_status: str
    safe_point: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResumeData:
    context: BoundedContext
    state: RunState
    last_call_signature: str | None


def workspace_identity(root: Path) -> str:
    return os.path.normcase(str(root.resolve(strict=True)))
```

Validate all dataclass inputs at construction or decode time. The codec must use exact-key checks, json.loads(encoded, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicate_keys), json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(",", ":")), and validate RFC 3339 UTC timestamps through datetime.fromisoformat.

- [ ] **Step 4: Add explicit RunState serialization and resume invalidation**

Use a constant tuple containing every current RunState field. Encode phase with phase.value and changed_files as a sorted list. Decode into a fresh RunState only after exact field/type/status validation.

Add this method to RunState:

```python
def invalidate_for_resume(self) -> None:
    self.verified_after_last_edit = False
    self.approval_status = None
    self.stop_reason = None
    if self.review_status != "changes_requested":
        self.review_status = None
        self.reviewed_edit_count = None
        self.reviewed_source_edit_count = None
    self.phase = RunPhase.EDIT if self.changed_files else RunPhase.DISCOVER
```

Preserve iteration, edit counters, changed files, consecutive_no_progress, last verifier exit code as history, review_rounds, review_rework_count, and the reviewed edit counters for a pending changes_requested decision.

Extend AgentRunResult with backward-compatible defaults:

```python
run_id: str | None = None
checkpoint_path: Path | None = None
resume_available: bool = False
checkpoint_warning: str | None = None
```

Add tests proving the invalidation rule and defensive result defaults.

- [ ] **Step 5: Implement the strict JSON codec**

Encode JournalSnapshot bytes with validate=True-compatible base64 and decode them back before ChangeJournal.restore_snapshots:

```python
def _encode_original(value: bytes | None) -> str | None:
    if value is None:
        return None
    return base64.b64encode(value).decode("ascii")


def _decode_original(value: object) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CheckpointError("checkpoint_invalid")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise CheckpointError("checkpoint_invalid") from exc
```

Create one exact-key validator per object level: root, request, runtime, context, state, journal record, fingerprint, and lifecycle. Reject duplicate journal and fingerprint paths before constructing RunCheckpoint. Encode tuples as arrays, sort journal/fingerprint records by path, keep changed_files sorted, and return defensive JSON-native copies for context/state.

- [ ] **Step 6: Implement CheckpointStore**

CheckpointStore must:

1. resolve and retain the canonical workspace root;
2. reject a symlinked .testpilot or checkpoints component;
3. create .testpilot/checkpoints and re-check its canonical parent;
4. generate run IDs with secrets.token_hex(8);
5. derive file names only after RUN_ID_PATTERN.fullmatch;
6. serialize and size-check before changing the existing file;
7. write a sibling NamedTemporaryFile, flush, fsync, best-effort chmod 0o600, and os.replace;
8. remove every temporary file on failure;
9. read at most max_checkpoint_bytes + 1 bytes;
10. reject non-regular targets and convert public failures to checkpoint_invalid, checkpoint_too_large, checkpoint_load_failed, or checkpoint_save_failed.

Implement delete(run_id) with the same path validation and a content-free checkpoint_cleanup_failed error.

- [ ] **Step 7: Run codec/store tests and Ruff**

Run:

```powershell
python -m pytest tests/test_checkpoint.py tests/test_types.py -q
python -m ruff check src/testpilot/checkpoint.py src/testpilot/types.py tests/test_checkpoint.py tests/test_types.py
```

Expected: schema and store tests pass; no sentinel secret appears in the serialized file.

- [ ] **Step 8: Commit Task 4**

```powershell
git add src/testpilot/checkpoint.py src/testpilot/types.py tests/test_checkpoint.py tests/test_types.py
git commit -m "feat: add atomic checkpoint store"
```

### Task 5: Add fingerprints, resume reconstruction, and terminal lifecycle

**Files:**
- Modify: src/testpilot/checkpoint.py
- Modify: tests/test_checkpoint.py
- Modify: tests/test_approval.py

- [ ] **Step 1: Write failing fingerprint and session tests**

Add:

```python
def test_session_restores_context_state_journal_and_rework_limit(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("old\n", encoding="utf-8")
    journal = ChangeJournal(tmp_path)
    Workspace(tmp_path, change_recorder=journal).write_file("app.py", "new\n")
    context = BoundedContext(
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": "Fix app.py"},
    )
    context.append_transaction({"role": "assistant", "content": "inspected"})
    state = RunState(
        iteration=4,
        edit_count=1,
        source_edit_count=1,
        changed_files={"app.py"},
        verified_after_last_edit=True,
        approval_status="approved",
        review_status="changes_requested",
        review_rounds=1,
        review_rework_count=1,
        reviewed_edit_count=1,
        reviewed_source_edit_count=1,
    )
    store = CheckpointStore(tmp_path)
    session = CheckpointSession.create(
        store=store,
        journal=journal,
        request=_request(tmp_path, tmp_path / ".testpilot/traces/run.jsonl"),
    )
    session.save(context=context, state=state, last_call_signature="signature")

    restored_journal = ChangeJournal(tmp_path)
    restored_session, resume = CheckpointSession.restore(
        store=store,
        journal=restored_journal,
        run_id=session.run_id,
    )

    assert restored_session.run_id == session.run_id
    assert resume.context.messages() == context.messages()
    assert resume.state.iteration == 4
    assert resume.state.review_rework_count == 1
    assert resume.state.review_status == "changes_requested"
    assert not resume.state.verified_after_last_edit
    assert resume.state.approval_status is None
    restored_journal.rollback()
    assert target.read_text(encoding="utf-8") == "old\n"


def test_resume_refuses_an_external_file_change_before_model_use(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("old\n", encoding="utf-8")
    journal = ChangeJournal(tmp_path)
    Workspace(tmp_path, change_recorder=journal).write_file("app.py", "new\n")
    store = CheckpointStore(tmp_path)
    session = CheckpointSession.create(
        store=store,
        journal=journal,
        request=_request(tmp_path, tmp_path / ".testpilot/traces/run.jsonl"),
    )
    context = BoundedContext(
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": "Fix app.py"},
    )
    state = RunState(
        iteration=2,
        edit_count=1,
        source_edit_count=1,
        changed_files={"app.py"},
    )
    session.save(context=context, state=state, last_call_signature=None)
    target.write_text("external\n", encoding="utf-8")

    with pytest.raises(CheckpointError) as caught:
        CheckpointSession.restore(
            store=store,
            journal=ChangeJournal(tmp_path),
            run_id=session.run_id,
        )

    assert caught.value.code == "checkpoint_workspace_changed"
```

Add cases for file deletion, created-file replacement, mode change on POSIX, symlink/non-regular replacement, over-limit current content, wrong workspace identity, verifier/request round trip, safe-point increment, active-only restore, terminal marker written before deletion, cleanup warning after terminal deletion failure, and rollback failure retaining an active checkpoint.

- [ ] **Step 2: Run session tests and verify RED**

Run:

```powershell
python -m pytest tests/test_checkpoint.py tests/test_approval.py -q
```

Expected: failures report missing fingerprint and CheckpointSession behavior.

- [ ] **Step 3: Implement bounded current-file fingerprints**

For each exported JournalSnapshot, normalize root/path again and return:

```python
def fingerprint_path(root: Path, relative: str, *, max_bytes: int) -> FileFingerprint:
    candidate = root / relative
    if candidate.is_symlink():
        raise CheckpointError("checkpoint_workspace_changed")
    target = candidate.resolve(strict=False)
    target.relative_to(root)
    if not target.exists():
        return FileFingerprint(relative, "missing", None, None)
    stat_result = target.stat()
    if not stat.S_ISREG(stat_result.st_mode) or stat_result.st_size > max_bytes:
        raise CheckpointError("checkpoint_workspace_changed")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
    return FileFingerprint(
        relative,
        "file",
        stat.S_IMODE(stat_result.st_mode),
        digest.hexdigest(),
    )
```

Sort fingerprints by path and compare the full immutable tuples with hmac.compare_digest for each digest. An empty journal has an empty fingerprint tuple.

- [ ] **Step 4: Implement CheckpointSession creation and saving**

CheckpointSession.create binds the canonical workspace identity, request, journal, timestamps, a new run ID, safe_point zero, and an optional on_ready callback. save increments safe_point, exports context/state/journal, captures current fingerprints, writes an active RunCheckpoint, caches that exact value, and invokes on_ready exactly once after the first successful atomic save.

- [ ] **Step 5: Implement validated resume reconstruction**

CheckpointSession.restore must:

1. load an active checkpoint;
2. compare canonical workspace identity;
3. decode journal records and restore them into an empty ChangeJournal;
4. recalculate and compare every fingerprint;
5. rebuild BoundedContext and RunState;
6. call state.invalidate_for_resume();
7. retain last_call_signature and the original request;
8. return the session and ResumeData only after all checks pass.

- [ ] **Step 6: Implement terminal lifecycle**

Implement finalization with this contract:

```python
@dataclass(frozen=True)
class FinalizeResult:
    cleanup_warning: str | None


def finalize(self, outcome: str) -> FinalizeResult:
    if outcome not in {"completed", "approved", "rolled_back"}:
        raise ValueError("invalid checkpoint outcome")
    terminal = replace(
        self._latest,
        lifecycle_status="terminal",
        updated_at=_utc_now(),
    )
    self.store.save(terminal)
    self._latest = terminal
    try:
        self.store.delete(self.run_id)
    except CheckpointError:
        return FinalizeResult("checkpoint_cleanup_failed")
    return FinalizeResult(None)
```

Require a successfully saved _latest value before finalize. A terminal file left behind must be rejected by restore. If writing the terminal marker fails, immediately try deleting the active checkpoint: successful deletion completes finalization safely; if deletion also fails, raise checkpoint_finalize_failed and retain the active file for explicit inspection.

- [ ] **Step 7: Run checkpoint/journal tests and Ruff**

Run:

```powershell
python -m pytest tests/test_checkpoint.py tests/test_approval.py tests/test_context_and_model.py tests/test_types.py -q
python -m ruff check src/testpilot/checkpoint.py src/testpilot/approval.py tests/test_checkpoint.py
```

Expected: restore reproduces the exact rollback baseline and rejects all fingerprint mismatches before returning ResumeData.

- [ ] **Step 8: Commit Task 5**

```powershell
git add src/testpilot/checkpoint.py tests/test_checkpoint.py tests/test_approval.py
git commit -m "feat: restore validated checkpoint sessions"
```

### Task 6: Integrate safe points into AgentRunner

**Files:**
- Modify: src/testpilot/agent.py:20-145, 197-308, 547-652
- Modify: tests/test_agent.py:147-218, 1053-1110, 1423-1545
- Modify: tests/test_agent_e2e.py

- [ ] **Step 1: Add a deterministic fake checkpoint boundary and failing tests**

Use:

```python
class FakeCheckpointSession:
    def __init__(self) -> None:
        self.run_id = "0123456789abcdef"
        self.path = Path(".testpilot/checkpoints/0123456789abcdef.json")
        self.safe_point = 0
        self.saved: list[tuple[list[dict[str, object]], int, str | None]] = []
        self.finalized: list[str] = []

    def save(
        self,
        *,
        context: BoundedContext,
        state: RunState,
        last_call_signature: str | None,
    ) -> None:
        self.safe_point += 1
        self.saved.append((context.messages(), state.iteration, last_call_signature))

    def finalize(self, outcome: str) -> FinalizeResult:
        self.finalized.append(outcome)
        return FinalizeResult(None)
```

Write separate tests proving:

- the initial save occurs before FakeModel.complete;
- a write is not checkpointed until its complete assistant/tool transaction is appended;
- model failure saves a resumable stop;
- resume starts at stored iteration + 1 and max_iterations is a new invocation budget;
- restored messages, last signature, and consecutive_no_progress prevent restart bypass;
- a pending Reviewer changes_requested state still requires a new source edit;
- a previously passed review and verifier are rerun after resume;
- success finalizes as completed or approved;
- successful rejection rollback finalizes as rolled_back;
- rollback_failed keeps the session active;
- checkpoint save failure returns checkpoint_save_failed without recursive saving;
- optional checkpoint=None preserves every existing AgentRunner behavior;
- checkpoint trace events contain run ID, safe-point metadata, status, error code, and duration but no messages, paths, task, source, or journal bytes.

- [ ] **Step 2: Run Agent tests and verify RED**

Run:

```powershell
python -m pytest tests/test_agent.py tests/test_agent_e2e.py -q
```

Expected: AgentRunner rejects checkpoint/resume arguments and does not save state.

- [ ] **Step 3: Add optional checkpoint and resume protocols**

Add:

```python
class _CheckpointSession(Protocol):
    run_id: str
    path: Path
    safe_point: int

    def save(
        self,
        *,
        context: BoundedContext,
        state: RunState,
        last_call_signature: str | None,
    ) -> None:
        raise NotImplementedError

    def finalize(self, outcome: str) -> FinalizeResult:
        raise NotImplementedError
```

Add checkpoint: _CheckpointSession | None = None to AgentRunner.__init__ and resume: ResumeData | None = None as a keyword-only argument to run. Fresh runs construct the current context/state/signature. Resume runs require the stored user anchor to equal task, use the restored objects, and begin with start_iteration = state.iteration + 1.

Replace range(1, max_iterations + 1) with:

```python
start_iteration = state.iteration + 1
for iteration in range(start_iteration, start_iteration + self.max_iterations):
    state.iteration = iteration
```

- [ ] **Step 4: Save only complete safe nodes**

Add a helper that converts CheckpointError into a content-free code:

```python
def _save_checkpoint(
    self,
    context: BoundedContext,
    state: RunState,
    last_call_signature: str | None,
) -> str | None:
    if self.checkpoint is None:
        return None
    started_ns = monotonic_ns()
    try:
        self.checkpoint.save(
            context=context,
            state=state,
            last_call_signature=last_call_signature,
        )
    except CheckpointError as error:
        code = error.code
    except Exception:
        code = "checkpoint_save_failed"
    else:
        code = None
    self._trace(
        "checkpoint",
        {
            "stage": "save",
            "run_id": self.checkpoint.run_id,
            "safe_point": self.checkpoint.safe_point,
            "ok": code is None,
            "error_code": code,
            "duration_ms": _elapsed_ms(started_ns),
        },
    )
    return code
```

Call it once before the first model request and once after lines that append a complete context transaction and update no-progress accounting. _stop saves the latest non-terminal state, including stop_reason. A checkpoint error must stop with checkpoint_save_failed and use persist_checkpoint=False so _stop cannot retry recursively.

- [ ] **Step 5: Finalize only after commit or successful rollback**

On successful run completion call finalize("approved") when approval exists, otherwise finalize("completed"). After a rejection/unavailable decision, call finalize("rolled_back") only after approval.rollback returns successfully. Preserve active state for rollback_failed.

Populate AgentRunResult.run_id, checkpoint_path, resume_available, and checkpoint_warning. resume_available is true only when the session remains active and its last save succeeded. A terminal cleanup warning is reported as checkpoint_cleanup_failed but does not change an already determined success/rejection result.

- [ ] **Step 6: Add a real restart E2E test**

Create a temporary buggy module and a real Workspace/ChangeJournal/Verifier. First runner uses a FakeModel that reads and edits, then raises ModelError. Reconstruct a new journal and session from disk, create a second runner whose FakeModel calls finish, and assert:

```python
assert first.success is False
assert first.resume_available is True
assert second.success is True
assert second.state.iteration > first.state.iteration
assert verifier.verify().ok
assert not checkpoint_path.exists()
```

Add a rejection variant proving that the reconstructed journal restores exact original bytes and POSIX mode.

- [ ] **Step 7: Run Agent regressions and Ruff**

Run:

```powershell
python -m pytest tests/test_agent.py tests/test_agent_e2e.py tests/test_checkpoint.py -q
python -m ruff check src/testpilot/agent.py tests/test_agent.py tests/test_agent_e2e.py
```

Expected: all selected tests pass; existing no-checkpoint callers remain unchanged.

- [ ] **Step 8: Commit Task 6**

```powershell
git add src/testpilot/agent.py tests/test_agent.py tests/test_agent_e2e.py
git commit -m "feat: checkpoint complete agent transactions"
```

### Task 7: Add fresh and resume modes to the real CLI

**Files:**
- Modify: src/testpilot/cli.py:28-83, 149-207, 219-367
- Modify: tests/test_cli.py
- Modify: src/testpilot/trace.py:18-38
- Modify: tests/test_command_and_trace.py:468-535

- [ ] **Step 1: Write failing parser, assembly, and output tests**

Add tests for these accepted forms:

```python
fresh = build_parser().parse_args(
    ["--workspace", ".", "--verify", "python -m pytest -q", "Fix app.py"]
)
resumed = build_parser().parse_args(
    ["--workspace", ".", "--resume", "0123456789abcdef"]
)

assert fresh.resume is None
assert resumed.resume == "0123456789abcdef"
```

Add rejection cases for resume combined with task, verify, or trace; fresh mode missing task/verify; invalid run ID; blank task; and non-positive per-invocation budget.

Add CLI integration tests that patch model construction with a forbidden factory and prove corrupt checkpoint, workspace mismatch, fingerprint mismatch, invalid verifier, and terminal checkpoint all stop before any model client is built.

Add successful fresh/resume assembly tests proving one shared journal is used by Workspace, ConsoleApprovalWorkflow, and CheckpointSession; the trace path is reused; the resume task/verifier are loaded from checkpoint; current environment supplies both model clients; and an explicit resume --max-iterations overrides only the invocation budget.

- [ ] **Step 2: Run CLI/trace tests and verify RED**

Run:

```powershell
python -m pytest tests/test_cli.py tests/test_command_and_trace.py -q
```

Expected: parser requires fresh-only arguments and no checkpoint runtime is assembled.

- [ ] **Step 3: Make argparse defer mode-specific requirements**

Change --verify to optional at parser construction, add --resume, change task to nargs="?", and use default=None for --max-iterations so configuration can distinguish an omitted resume override. Validate combinations after _workspace_path and before API/model construction.

Resolved CliConfig remains fully populated:

```python
@dataclass(frozen=True)
class CliConfig:
    workspace: Path
    verifier: tuple[str, ...]
    task: str
    api_key: str
    model: str
    base_url: str | None
    trace_path: Path
    max_iterations: int
```

Add:

```python
@dataclass(frozen=True)
class RunSetup:
    config: CliConfig
    journal: ChangeJournal
    checkpoint: CheckpointSession
    resume: ResumeData | None
```

_fresh_setup validates task/verifier, reserves a new trace, creates journal/store/session, and uses 12 when max_iterations is omitted. _resume_setup validates and restores the checkpoint first, canonicalizes its stored verifier through CommandRunner.canonical_model_command, validates the existing trace as a regular non-symlink file inside workspace, and applies the stored or user-supplied invocation budget.

Catch CheckpointError separately at the CLI boundary:

```python
except CheckpointError as error:
    print("STATUS=FAILED")
    print(f"stop_reason={error.code}")
    return 1
```

Only the fixed CheckpointError.code is printed; the original exception, task, paths, and checkpoint payload remain hidden.

- [ ] **Step 4: Reuse injected runtime objects in build_agent**

Extend build_agent with keyword-only journal and checkpoint arguments. If journal is absent, retain current direct-library construction. The real main path passes one journal to Workspace and ConsoleApprovalWorkflow and passes checkpoint to AgentRunner.

Keep .testpilot/checkpoints/** in Workspace.private_patterns and include the exact trace target in protected_patterns. Do not expose a flag that disables either boundary.

- [ ] **Step 5: Print stable checkpoint metadata and record resume**

The session on_ready callback prints exactly these two safe lines after the first successful save and before the model request:

```text
run_id=0123456789abcdef
checkpoint=.testpilot/checkpoints/0123456789abcdef.json
```

Extend _print_result with:

```python
run_id = getattr(result, "run_id", None)
resume_available = getattr(result, "resume_available", False) is True
warning = getattr(result, "checkpoint_warning", None)
print(f"run_id={run_id if isinstance(run_id, str) else '-'}")
print(f"resume_available={'yes' if resume_available else 'no'}")
print(
    "checkpoint_warning="
    + (warning if warning == "checkpoint_cleanup_failed" else "-")
)
```

On restore, append a trace event with run_id, schema_version, safe_point, ok, error_code, and duration only. Adjust JsonlTrace initialization only as needed to append to a prevalidated existing trace; keep its payload and environment-redaction rules unchanged.

- [ ] **Step 6: Run CLI, trace, and assembly regressions**

Run:

```powershell
python -m pytest tests/test_cli.py tests/test_command_and_trace.py tests/test_workspace_tools.py -q
python -m ruff check src/testpilot/cli.py src/testpilot/trace.py tests/test_cli.py
```

Expected: fresh and resumed modes pass, failures occur before model construction, and output contains no task, source, token, API key, base URL credential, or checkpoint payload.

- [ ] **Step 7: Commit Task 7**

```powershell
git add src/testpilot/cli.py src/testpilot/trace.py tests/test_cli.py tests/test_command_and_trace.py
git commit -m "feat: resume checkpointed runs from cli"
```

### Task 8: Demonstrate, document, and verify the complete recovery path

**Files:**
- Modify: src/testpilot/demo.py
- Modify: tests/test_demo.py
- Modify: README.md
- Modify: submission/录屏与提交清单.md
- Modify: submission/README.txt.template
- Modify: docs/superpowers/plans/2026-09-01-checkpoint-resume-implementation.md

- [ ] **Step 1: Write the failing two-process-style demo test**

Update the main demo assertion to require this ordered output:

```python
lines = capsys.readouterr().out.splitlines()
assert lines == [
    "BEFORE=FAIL",
    "INTERRUPTED=CHECKPOINTED",
    "RESUMED=SUCCESS",
    "VERIFIED=PASS",
    "REVIEWED=PASS",
    "APPROVED=SIMULATED",
    "AFTER=PASS",
]
```

For --keep, additionally assert the trace exists, the terminal checkpoint was removed, calculator.py is fixed, and the JSONL events contain one run ID across the initial and resumed runners. Keep forbidden_input so the keyless demo never waits for terminal input.

- [ ] **Step 2: Run demo tests and verify RED**

Run:

```powershell
python -m pytest tests/test_demo.py -q
```

Expected: the current one-run demo lacks interruption and resume output.

- [ ] **Step 3: Split the scripted repair across two runtime instances**

The first FakeModel reads and edits, then exhausts its script so AgentRunner returns a resumable model_exhausted result. Load the saved run through a new ChangeJournal and CheckpointSession. The second FakeModel requests finish; a separate Reviewer FakeModel reads the repaired source and passes.

Use a demo-only approval object that records one simulated approval and delegates commit/rollback to the real ChangeJournal:

```python
class _DemoApproval:
    def __init__(self, journal: ChangeJournal) -> None:
        self.journal = journal
        self.approved = False

    def request(
        self,
        *,
        changed_files: Sequence[str],
        verification_exit_code: int,
    ) -> bool:
        self.approved = bool(changed_files) and verification_exit_code == 0
        return self.approved

    def commit(self) -> None:
        self.journal.commit()

    def rollback(self) -> None:
        self.journal.rollback()
```

Print APPROVED=SIMULATED so the demo clearly distinguishes deterministic test wiring from a real person's terminal decision.

- [ ] **Step 4: Run demo tests and the executable demo**

Run:

```powershell
python -m pytest tests/test_demo.py -q
$env:PYTHONPATH = "src"
python -m testpilot.demo
Remove-Item Env:PYTHONPATH
```

Expected: output matches the seven lines in Step 1 and exits zero without API configuration.

- [ ] **Step 5: Update public documentation with implemented behavior**

Update README with:

```powershell
python -m testpilot --workspace . --verify "python -m pytest -q" "修复这个项目"
python -m testpilot --workspace . --resume 0123456789abcdef
```

Explain safe-node persistence in plain Chinese, the host-private and gitignored checkpoint location, preserved rollback across restart, fingerprint refusal before model use, cumulative iteration with a per-invocation budget, re-verification/re-review before approval, trace continuity, terminal cleanup, and the new offline demo output. Replace the old sentence that says persistent checkpoint recovery is absent with a positive list of currently implemented capabilities; do not add a separate list of excluded future features.

Update the recording checklist to show: initial run ID, interrupted result with resume_available=yes, resume command, same run ID/trace, fixed pytest, fresh Reviewer pass, simulated demo approval label, real CLI human approval as a separate recording, and Git commits for each implementation stage.

Update submission/README.txt.template with the fresh command, resume command, offline demo command, and a warning that .testpilot/checkpoints contains local task/source context and must remain uncommitted.

- [ ] **Step 6: Run the full verification gate**

Run fresh commands:

```powershell
python -m pytest -q
python -m ruff check .
$env:PYTHONPATH = "src"
python -m testpilot.demo
Remove-Item Env:PYTHONPATH
git diff --check
```

Expected: every test passes, Ruff reports no errors, the deterministic demo prints the seven expected lines, and git diff --check has no output.

- [ ] **Step 7: Audit the approved specification against code and tests**

Confirm with direct file/test evidence that:

- every persisted object is rebuilt through strict validators;
- checkpoint paths cannot escape the workspace or be reached through Agent tools;
- API/environment credentials are absent from checkpoint and trace fixtures;
- the journal survives restart and exact rollback is tested;
- every successful tool batch is checkpointed only after a complete transaction;
- external file changes stop before model construction;
- old verification/review/approval evidence cannot reach success;
- Reviewer rework count and repeated-call protection survive resume;
- terminal runs reject reuse and clean sensitive checkpoint state;
- CLI and demo expose the same stable run ID without printing source or model text.

- [ ] **Step 8: Mark plan steps complete and commit documentation**

```powershell
git add README.md 'submission/录屏与提交清单.md' submission/README.txt.template src/testpilot/demo.py tests/test_demo.py docs/superpowers/plans/2026-09-01-checkpoint-resume-implementation.md
git commit -m "docs: explain checkpoint recovery workflow"
```

## Final review and integration

- [ ] Re-run the complete verification commands immediately before claiming completion.
- [ ] Review git diff main...HEAD against docs/superpowers/specs/2026-09-01-checkpoint-resume-design.md and resolve every critical or important discrepancy.
- [ ] Record that independent subagent review was unavailable if workspace credits remain exhausted, and perform the same spec-compliance, security, and code-quality checklists locally with fresh evidence.
- [ ] Present the finishing-a-development-branch integration options; do not merge, push, or discard without the user's choice.
