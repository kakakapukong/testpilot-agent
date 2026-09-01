# Human Approval Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require a human decision after immutable verification succeeds, keep approved edits, and restore the exact pre-run files when the human rejects or approval cannot be obtained.

**Architecture:** A `ChangeJournal` captures each file once immediately before its first successful workspace write and can summarize or atomically restore that original state. `AgentRunner` invokes an injected approval workflow only after a successful `finish` verification; approval preserves the edits, while rejection or an unavailable prompt fails closed and rolls them back. The CLI wires a console approval workflow into the real runtime, while direct library users and the offline demo remain backward-compatible unless they explicitly inject one.

**Tech Stack:** Python 3.11+, dataclasses, pathlib, tempfile, difflib, pytest, existing TestPilot tool/trace abstractions.

---

## File map

- Create `src/testpilot/approval.py`: reversible file journal, change summaries, console approval workflow.
- Modify `src/testpilot/workspace.py`: optional pre-write snapshot hook at the canonical file-write boundary.
- Modify `src/testpilot/types.py`: stable approval state in `RunState`.
- Modify `src/testpilot/agent.py`: verification-to-approval transition, fail-closed rollback, audit events.
- Modify `src/testpilot/cli.py`: assemble the journal and console workflow; print approval status.
- Create `tests/test_approval.py`: journal and console workflow behavior.
- Modify `tests/test_workspace_tools.py`: snapshot ordering and snapshot-failure protection.
- Modify `tests/test_agent.py`: approval state-machine unit tests and trace assertions.
- Modify `tests/test_agent_e2e.py`: real-file approve/reject behavior.
- Modify `tests/test_cli.py`: runtime wiring and result summary.
- Modify `README.md` and `submission/录屏与提交清单.md`: user-visible workflow and recording instructions.

### Task 1: Reversible workspace change journal

**Files:**
- Create: `src/testpilot/approval.py`
- Modify: `src/testpilot/workspace.py`
- Create: `tests/test_approval.py`
- Modify: `tests/test_workspace_tools.py`

- [ ] **Step 1: Write failing journal tests**

Add tests that express the public behavior before implementation:

```python
def test_change_journal_summarizes_and_restores_existing_and_new_files(tmp_path: Path) -> None:
    original = tmp_path / "app.py"
    original.write_text("value = 1\n", encoding="utf-8")
    journal = ChangeJournal(tmp_path)
    workspace = Workspace(tmp_path, change_recorder=journal)

    workspace.write_file("app.py", "value = 2\nextra = True\n")
    workspace.write_file("nested/new.py", "created = True\n")

    assert journal.summaries() == (
        ChangeSummary("app.py", "modified", additions=2, deletions=1),
        ChangeSummary("nested/new.py", "created", additions=1, deletions=0),
    )

    journal.rollback()
    assert original.read_text(encoding="utf-8") == "value = 1\n"
    assert not (tmp_path / "nested/new.py").exists()
    assert not (tmp_path / "nested").exists()
```

Also cover repeated writes capturing only the first state, preservation of POSIX mode, a missing/changed file at summary time, and rollback failures producing a safe `ApprovalError` without embedding file contents.

- [ ] **Step 2: Run the new journal tests and verify RED**

Run:

```powershell
python -m pytest tests/test_approval.py -q
```

Expected: collection fails because `testpilot.approval` does not exist.

- [ ] **Step 3: Implement the minimal journal and summary types**

Implement these stable public shapes in `approval.py`:

```python
@dataclass(frozen=True)
class ChangeSummary:
    path: str
    status: str
    additions: int
    deletions: int


class ApprovalError(RuntimeError):
    """A safe approval or rollback failure with no file contents."""


class ChangeJournal:
    def __init__(self, root: str | Path) -> None: ...
    def capture(self, path: Path) -> None: ...
    def summaries(self) -> tuple[ChangeSummary, ...]: ...
    def rollback(self) -> None: ...
```

`capture()` stores bytes and the original mode only on the first write. `summaries()` uses line-level opcodes to count additions/deletions without exposing source content. `rollback()` restores existing files through a same-directory temporary file and `os.replace`, deletes newly created files, and removes only empty parent directories below the workspace root.

- [ ] **Step 4: Add the workspace pre-write hook with failure protection**

Add a structural protocol and optional constructor dependency:

```python
class _ChangeRecorder(Protocol):
    def capture(self, path: Path) -> None: ...


def __init__(..., change_recorder: _ChangeRecorder | None = None) -> None:
    self.change_recorder = change_recorder
```

In `Workspace.write_file`, call the recorder after path/protection/no-op validation and immediately before creating directories or replacing the target. Convert recorder failures to `WorkspaceError("snapshot_failed", "could not snapshot file before writing")`, and do not write the requested content.

- [ ] **Step 5: Verify GREEN and run workspace regressions**

Run:

```powershell
python -m pytest tests/test_approval.py tests/test_workspace_tools.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add src/testpilot/approval.py src/testpilot/workspace.py tests/test_approval.py tests/test_workspace_tools.py
git commit -m "feat: add reversible workspace change journal"
```

### Task 2: Gate verified success on a human decision

**Files:**
- Modify: `src/testpilot/types.py`
- Modify: `src/testpilot/agent.py`
- Modify: `src/testpilot/approval.py`
- Modify: `tests/test_types.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_agent_e2e.py`

- [ ] **Step 1: Write failing approval state-machine tests**

Use an injected fake implementing this boundary:

```python
class _ApprovalWorkflow(Protocol):
    def request(self, *, changed_files: Sequence[str], verification_exit_code: int) -> bool: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
```

Add separate tests proving:

```python
def test_verified_repair_requires_and_records_approval() -> None:
    approval = FakeApproval(approved=True)
    result = _verified_runner(approval=approval).run("Fix app.py")
    assert result.success
    assert result.stop_reason == "verified"
    assert result.state.approval_status == "approved"
    assert approval.requests == [(('app.py',), 0)]


def test_rejected_repair_rolls_back_and_fails() -> None:
    approval = FakeApproval(approved=False)
    result = _verified_runner(approval=approval).run("Fix app.py")
    assert not result.success
    assert result.stop_reason == "approval_rejected"
    assert result.state.approval_status == "rejected"
    assert approval.rollback_calls == 1
```

Also assert: no approval on failed verification; prompt exceptions fail closed as `approval_unavailable` and roll back; rollback exceptions become `rollback_failed`; approval audit events contain only stage, decision, file count, verification exit, and rollback outcome.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_types.py tests/test_agent.py -q
```

Expected: failures show that `approval_status` and the `approval` runner dependency are absent.

- [ ] **Step 3: Add stable approval accounting**

Extend `RunState` with:

```python
approval_status: str | None = None

def record_approval(self, status: str) -> None:
    if status not in {"approved", "rejected", "unavailable"}:
        raise ValueError("invalid approval status")
    self.approval_status = status
```

- [ ] **Step 4: Implement the verification-to-approval transition**

Add `approval: _ApprovalWorkflow | None = None` to `AgentRunner.__init__`. When the full tool turn has finished and `state.verified_after_last_edit` is true:

```python
approved, rejection_reason = self._request_approval(state)
if not approved:
    return self._stop(False, final_text, rejection_reason, state, context)
return self._stop(True, final_text, "verified", state, context)
```

If no workflow was injected, preserve current library behavior. If one was injected, request exactly once. Approval commits and clears the accepted journal baseline; any request or commit error is treated as unavailable, and every non-approved path attempts rollback. A successful rollback also clears old snapshots so a reused runner starts fresh. A rollback exception overrides the stop reason with `rollback_failed`. Never include prompt text, source content, or diff content in JSONL events.

- [ ] **Step 5: Add real-file approve/reject end-to-end tests**

Build a real `Workspace` and `ChangeJournal` around a temporary buggy calculator. Verify that approval retains the passing repair and rejection restores the exact original bytes while returning failure.

- [ ] **Step 6: Verify GREEN and run Agent regressions**

Run:

```powershell
python -m pytest tests/test_types.py tests/test_agent.py tests/test_agent_e2e.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 2**

```powershell
git add src/testpilot/types.py src/testpilot/agent.py src/testpilot/approval.py tests/test_types.py tests/test_agent.py tests/test_agent_e2e.py
git commit -m "feat: require approval for verified repairs"
```

### Task 3: Wire safe console approval and document the workflow

**Files:**
- Modify: `src/testpilot/approval.py`
- Modify: `src/testpilot/cli.py`
- Modify: `tests/test_approval.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `submission/录屏与提交清单.md`

- [ ] **Step 1: Write failing console and CLI tests**

Test an injected input/output pair rather than the real terminal:

```python
def test_console_approval_displays_summary_and_accepts_yes(tmp_path: Path) -> None:
    lines: list[str] = []
    workflow = ConsoleApprovalWorkflow(
        journal_with_one_change(tmp_path),
        input_fn=lambda prompt: "yes",
        output_fn=lines.append,
    )
    assert workflow.request(changed_files=("app.py",), verification_exit_code=0)
    assert lines == [
        "APPROVAL_REQUIRED",
        "verification_exit=0",
        'M "app.py" (+1/-1)',
    ]
```

Add separate tests for `y`, upper-case input, default rejection for blank/unknown responses, EOF/KeyboardInterrupt rejection, and output that never contains file contents. Assert `build_agent` shares one journal between `Workspace` and `ConsoleApprovalWorkflow`, and `_print_result` emits `approval=approved|rejected|unavailable|-`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_approval.py tests/test_cli.py -q
```

Expected: failures show that `ConsoleApprovalWorkflow` and CLI wiring are absent.

- [ ] **Step 3: Implement fail-closed console interaction**

Implement:

```python
class ConsoleApprovalWorkflow:
    def __init__(
        self,
        journal: ChangeJournal,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None: ...

    def request(self, *, changed_files: Sequence[str], verification_exit_code: int) -> bool: ...
    def commit(self) -> None:
        self.journal.commit()
    def rollback(self) -> None:
        self.journal.rollback()
```

Print only status, immutable verification exit, and per-file `M/A "path" (+N/-N)` summaries. Render paths as ASCII-safe JSON strings so control and bidirectional characters cannot forge terminal lines. Prompt with `Accept verified changes? [y/N]: `; accept only `y` or `yes`. Treat blank input, other input, `EOFError`, and `KeyboardInterrupt` as rejection.

- [ ] **Step 4: Wire the real CLI**

In `build_agent`, create one `ChangeJournal(config.workspace)`, pass it to `Workspace(change_recorder=journal)`, and pass `ConsoleApprovalWorkflow(journal)` to `AgentRunner(approval=...)`. Keep the offline demo non-interactive by relying on the optional runner default. Add the `approval=` line to `_print_result`, and encode `changed_files=` as an ASCII-safe JSON array.

- [ ] **Step 5: Update user documentation**

Document this exact order:

```text
model tools -> host verification -> change summary -> human approve/reject
approve -> keep files and report success
reject/unavailable -> restore original files and report failure
```

Explain that approval occurs once per successful run, dangerous paths remain forbidden rather than approvable, non-interactive input rejects safely, and JSONL records the decision without source content. Add a recording checklist item demonstrating both accept and reject/rollback paths.

- [ ] **Step 6: Run full verification**

Run:

```powershell
python -m pytest -q
python -m ruff check .
python -m testpilot.demo
git diff --check
```

Expected: all tests pass, Ruff reports no errors, demo prints `BEFORE=FAIL`, `AGENT=SUCCESS`, `AFTER=PASS`, and the diff check is empty.

- [ ] **Step 7: Commit Task 3 as code and documentation history**

```powershell
git add src/testpilot/approval.py src/testpilot/cli.py tests/test_approval.py tests/test_cli.py
git commit -m "feat: add console approval workflow"
git add README.md submission/录屏与提交清单.md docs/superpowers/plans/2026-09-01-human-approval-implementation.md
git commit -m "docs: explain the human approval workflow"
```

## Final review and integration

- [ ] Compare all changes with this plan and the accepted flow.
- [ ] Run an independent spec-compliance review, then a separate code-quality review.
- [ ] Fix every critical or important finding and repeat the relevant review.
- [ ] Re-run the full verification commands immediately before integration.
- [ ] Merge `feature/human-approval` into `main` without squashing so every real commit remains visible.
- [ ] Push `main` and verify the remote commit hashes and public repository status.
