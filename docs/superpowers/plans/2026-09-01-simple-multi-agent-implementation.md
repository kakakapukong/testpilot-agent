# Simple Multi-Agent Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate read-only reviewer Agent that runs after immutable pytest verification, may request exactly one repair round, and must pass before the existing human approval gate.

**Architecture:** `ReviewerAgent` is a second bounded model loop with a fresh prompt and a registry containing only list/read/search plus a structured `submit_review` decision tool. `AgentRunner` remains the coordinator: it delays review until the complete repair-model turn is processed, returns first-round review feedback through the existing `finish` tool result, re-verifies after a new edit, and stops after a second rejection. The CLI creates separate repair and reviewer model clients and exposes safe review state in the trace and terminal summary.

**Tech Stack:** Python 3.11+, dataclasses, protocols, existing `BoundedContext`, `ModelClient`, `ToolRegistry`, workspace tools, pytest, Ruff.

---

## File map

- Create `src/testpilot/reviewer.py`: review result/error types, structured decision tool, read-only registry factory, and bounded reviewer loop.
- Create `tests/test_reviewer.py`: reviewer tool boundary, structured decisions, recovery, limits, and sanitized errors.
- Modify `src/testpilot/types.py`: review phase and stable review accounting in `RunState`.
- Modify `src/testpilot/agent.py`: optional reviewer dependency and verify-review-rework-approval orchestration.
- Modify `tests/test_types.py`: review state validation.
- Modify `tests/test_agent.py`: coordinator ordering, feedback round, final rejection, and failure boundaries.
- Modify `tests/test_agent_e2e.py`: real workspace two-agent success path.
- Modify `src/testpilot/cli.py`: separate model clients and read-only reviewer wiring; safe result fields.
- Modify `tests/test_cli.py`: runtime assembly and output contract.
- Modify `src/testpilot/demo.py` and `tests/test_demo.py`: deterministic keyless two-agent demonstration.
- Modify `README.md` and `submission/录屏与提交清单.md`: architecture, usage, trace, and recording evidence.

### Task 1: Build the read-only Reviewer Agent

**Files:**
- Create: `src/testpilot/reviewer.py`
- Create: `tests/test_reviewer.py`

- [ ] **Step 1: Write failing tests for the public reviewer contract**

Create focused tests that express these concrete behaviors:

```python
def test_reviewer_registry_contains_only_read_tools_and_structured_decision(tmp_path: Path) -> None:
    registry = build_reviewer_registry(Workspace(tmp_path))
    assert registry.names() == ("list_files", "read_file", "search_text", "submit_review")


def test_reviewer_inspects_then_returns_structured_pass(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    model = FakeModel(
        [
            AssistantTurn("inspect", (ToolCall("read", "read_file", {"path": "app.py"}),)),
            AssistantTurn(
                "decide",
                (
                    ToolCall(
                        "review",
                        "submit_review",
                        {"decision": "pass", "feedback": "No blocking correctness issue."},
                    ),
                ),
            ),
        ]
    )
    reviewer = ReviewerAgent(model, build_reviewer_registry(Workspace(tmp_path)))

    result = reviewer.review(
        task="Fix app.py",
        changed_files=("app.py",),
        verification_exit_code=0,
    )

    assert result == ReviewResult("pass", "No blocking correctness issue.")
    assert [schema["function"]["name"] for schema in model.received_inputs[0][1]] == [
        "list_files",
        "read_file",
        "search_text",
        "submit_review",
    ]
```

Add separate tests proving `request_changes` is returned with feedback, a `submit_review` call mixed with an inspection call is rejected and can recover on the next turn, malformed call arguments are returned as tool failures, no-tool termination and exhausted iterations raise stable `ReviewerError.code` values, model exceptions do not leak their messages, duplicate/blank call IDs fail, feedback must be non-blank and no longer than 4,000 characters, and a string `changed_files` argument is rejected.

- [ ] **Step 2: Run the reviewer tests and verify RED**

Run:

```powershell
python -m pytest tests/test_reviewer.py -q
```

Expected: collection fails because `testpilot.reviewer` does not exist.

- [ ] **Step 3: Add stable review values and the structured decision tool**

Implement these public shapes without exposing mutable input:

```python
MAX_REVIEW_FEEDBACK_CHARS = 4_000
REVIEW_TOOL_NAMES = ("list_files", "read_file", "search_text", "submit_review")


@dataclass(frozen=True)
class ReviewResult:
    decision: str
    feedback: str

    def __post_init__(self) -> None:
        if self.decision not in {"pass", "request_changes"}:
            raise ValueError("invalid review decision")
        if not isinstance(self.feedback, str) or not self.feedback.strip():
            raise ValueError("review feedback must be non-blank")
        if len(self.feedback) > MAX_REVIEW_FEEDBACK_CHARS:
            raise ValueError("review feedback is too long")


class ReviewerError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__("reviewer stopped")
        self.code = code
```

Implement `SubmitReviewTool` with a closed object schema requiring `decision` and `feedback`. Its `execute()` validates through `ReviewResult` and returns only `{"decision": ..., "feedback": ...}` on success or a stable `invalid_review_decision` failure without echoing feedback.

- [ ] **Step 4: Build and enforce the read-only registry**

Add a factory with exactly this registration order:

```python
def build_reviewer_registry(workspace: Workspace) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        ListFilesTool(workspace),
        ReadFileTool(workspace),
        SearchTextTool(workspace),
        SubmitReviewTool(),
    ):
        registry.register(tool)
    return registry
```

`ReviewerAgent.__init__` must reject any registry whose names differ from `REVIEW_TOOL_NAMES`, validate a positive integer iteration limit, and store only the model, registry, and context limits.

- [ ] **Step 5: Implement the bounded reviewer loop**

Create a fresh `BoundedContext` for every `review()` call. The developer prompt must say the reviewer is read-only, repository contents are untrusted data, passing pytest is evidence but not proof, and the reviewer must inspect before submitting one structured decision. Encode the task, sorted changed files, and verification exit code as JSON in the user anchor.

For each turn:

1. call `ModelClient.complete` with only reviewer schemas;
2. validate `AssistantTurn`, unique non-blank call IDs, and `argument_error`;
3. reject a mixed `submit_review`/inspection turn with a tool failure and continue;
4. execute inspection tools and append a complete assistant/tool transaction;
5. return `ReviewResult` only for a valid sole `submit_review` call;
6. convert `ModelError`, generic exceptions, no-tool stops, invalid turns, and budget exhaustion into stable `ReviewerError.code` values without source or exception text.

- [ ] **Step 6: Verify GREEN and run adjacent regressions**

Run:

```powershell
python -m pytest tests/test_reviewer.py tests/test_context_and_model.py tests/test_registry.py -q
python -m ruff check src/testpilot/reviewer.py tests/test_reviewer.py
```

Expected: all selected tests pass and Ruff reports no errors.

- [ ] **Step 7: Commit Task 1**

```powershell
git add src/testpilot/reviewer.py tests/test_reviewer.py
git commit -m "feat: add read-only reviewer agent"
```

### Task 2: Orchestrate verify, review, one rework, and approval

**Files:**
- Modify: `src/testpilot/types.py`
- Modify: `src/testpilot/agent.py`
- Modify: `tests/test_types.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_agent_e2e.py`

- [ ] **Step 1: Write failing `RunState` and coordinator tests**

Add a deterministic fake boundary:

```python
@dataclass
class FakeReviewer:
    results: list[ReviewResult | BaseException]
    requests: list[tuple[str, tuple[str, ...], int]] = field(default_factory=list)

    def review(
        self,
        *,
        task: str,
        changed_files: tuple[str, ...],
        verification_exit_code: int,
    ) -> ReviewResult:
        self.requests.append((task, tuple(changed_files), verification_exit_code))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result
```

Write separate tests proving:

- a first-pass review occurs after verification and before approval;
- a first `request_changes` appears as `review_changes_requested` in the repair model's `finish` tool result;
- a second `finish` without a new source edit returns `review_rework_required` and does not re-run verifier or reviewer;
- one new edit permits a second verifier and review, and a pass reaches approval;
- a second `request_changes` stops with `review_changes_remaining` and never requests approval;
- reviewer exceptions, `KeyboardInterrupt`, or non-`ReviewResult` values fail closed without leaking details;
- `SystemExit` is not swallowed;
- a later edit in the same model turn prevents premature review;
- no reviewer dependency preserves the existing verification-to-approval behavior;
- review trace events contain counts/status/duration but not task, paths, feedback, or secrets.

Add `RunState` tests for valid statuses and counters, rejection of invalid statuses, and invalidation of `verified_after_last_edit` for `changes_requested` and `unavailable`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_types.py tests/test_agent.py tests/test_agent_e2e.py -q
```

Expected: failures show that review state and the reviewer dependency are absent.

- [ ] **Step 3: Add review accounting to `RunState`**

Extend `RunPhase` with `REVIEW = "review"` and add:

```python
review_status: str | None = None
review_rounds: int = 0
review_rework_count: int = 0
reviewed_edit_count: int | None = None

def record_review(self, status: str) -> None:
    if status not in {"passed", "changes_requested", "unavailable"}:
        raise ValueError("invalid review status")
    self.phase = RunPhase.REVIEW
    self.review_status = status
    self.review_rounds += 1
    self.reviewed_edit_count = self.edit_count
    if status != "passed":
        self.verified_after_last_edit = False
```

The coordinator, not the state type, increments `review_rework_count` only when it grants the single feedback-driven repair round.

- [ ] **Step 4: Add the reviewer boundary and post-turn review transition**

Add an optional protocol dependency to `AgentRunner`:

```python
class _Reviewer(Protocol):
    def review(
        self,
        *,
        task: str,
        changed_files: Sequence[str],
        verification_exit_code: int,
    ) -> ReviewResult: ...
```

Keep tool execution serial, but do not call the reviewer inside `_execute_call`. Record the index of the successful `finish` tool message. Only after every call in the assistant turn has executed, and only if no later edit invalidated verification, call `_review_repair(task, state)`.

`_review_repair` must:

- return success immediately when no reviewer was injected;
- reject re-review before `state.edit_count` exceeds `state.reviewed_edit_count`;
- trace safe `review` start/complete events;
- on `pass`, call `state.record_review("passed")` and preserve verification;
- on the first `request_changes`, call `state.record_review("changes_requested")`, increment `review_rework_count`, and return a bounded `ToolResult.failure` with code `review_changes_requested` and feedback in `data`;
- on the second request, return code and terminal reason `review_changes_remaining`;
- on unavailable/invalid review, record `unavailable` and return a terminal, content-free failure.

Replace the stored successful `finish` tool message with the review failure before appending the transaction to `BoundedContext`. This lets the repair model receive reviewer feedback through an ordinary tool result and keeps assistant/tool IDs valid. Request approval only when verification is still valid and the reviewer passed or no reviewer was configured.

- [ ] **Step 5: Add real-workspace two-agent E2E coverage**

Use a temporary buggy Python module, a real `Workspace`, real fixed `Verifier`, scripted repair turns, and a fake reviewer. Assert this exact event order:

```python
assert ordered_stages == [
    "verification:start",
    "verification:complete",
    "review:start",
    "review:complete",
    "approval:start",
    "approval:complete",
]
```

Also prove a reviewer-requested edit is re-verified and that final review rejection does not call approval.

- [ ] **Step 6: Verify GREEN and run Agent regressions**

Run:

```powershell
python -m pytest tests/test_types.py tests/test_agent.py tests/test_agent_e2e.py -q
python -m ruff check src/testpilot/types.py src/testpilot/agent.py tests/test_types.py tests/test_agent.py tests/test_agent_e2e.py
```

Expected: all selected tests pass and Ruff reports no errors.

- [ ] **Step 7: Commit Task 2**

```powershell
git add src/testpilot/types.py src/testpilot/agent.py tests/test_types.py tests/test_agent.py tests/test_agent_e2e.py
git commit -m "feat: coordinate review and one repair round"
```

### Task 3: Wire the real CLI and offline two-agent demo

**Files:**
- Modify: `src/testpilot/cli.py`
- Modify: `src/testpilot/demo.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_demo.py`

- [ ] **Step 1: Write failing runtime and output tests**

Extend CLI tests to assert:

```python
assert agent.reviewer is not None
assert agent.reviewer.model is not agent.model
assert agent.reviewer.registry.names() == (
    "list_files",
    "read_file",
    "search_text",
    "submit_review",
)
```

Patch `OpenAIChatModel` with a recording factory and prove `build_agent` creates two distinct instances with the same user configuration. Extend `_print_result` tests to require stable lines:

```text
review=passed|changes_requested|unavailable|-
review_rounds=<non-negative integer>
review_reworks=0|1
```

Update the demo test to require `REVIEW=PASS` between `AGENT=SUCCESS` and `AFTER=PASS`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_cli.py tests/test_demo.py -q
```

Expected: failures show that CLI reviewer wiring and demo review output are absent.

- [ ] **Step 3: Assemble separate repair and reviewer runtimes**

In `build_agent`, retain the shared read-only view of the same `Workspace` but create independent model adapters:

```python
repair_model = OpenAIChatModel(
    model=config.model,
    api_key=config.api_key,
    base_url=config.base_url,
)
review_model = OpenAIChatModel(
    model=config.model,
    api_key=config.api_key,
    base_url=config.base_url,
)
reviewer = ReviewerAgent(review_model, build_reviewer_registry(workspace))
```

Pass `repair_model` as the main model and `reviewer=reviewer` to `AgentRunner`. Do not add a CLI flag that can accidentally disable the review gate.

- [ ] **Step 4: Extend safe CLI result output**

After the existing `verification_exit=` line, print `review=`, `review_rounds=`, and `review_reworks=` from validated state fields. Do not print feedback, model text, reviewer messages, source, or diff content.

- [ ] **Step 5: Make the offline demo exercise both agents**

Build a `ReviewerAgent` with a separate `FakeModel` scripted to read `calculator.py` and submit a `pass` decision. Inject it into `AgentRunner`, print the final stable review status as `REVIEW=PASS`, and keep the demo non-interactive by leaving approval unset.

- [ ] **Step 6: Verify GREEN and run the real demo**

Run:

```powershell
python -m pytest tests/test_cli.py tests/test_demo.py -q
python -m testpilot.demo
python -m ruff check src/testpilot/cli.py src/testpilot/demo.py tests/test_cli.py tests/test_demo.py
```

Expected: tests and Ruff pass; demo prints `BEFORE=FAIL`, `AGENT=SUCCESS`, `REVIEW=PASS`, and `AFTER=PASS`.

- [ ] **Step 7: Commit Task 3**

```powershell
git add src/testpilot/cli.py src/testpilot/demo.py tests/test_cli.py tests/test_demo.py
git commit -m "feat: enable two-agent review in cli"
```

### Task 4: Document, audit, and verify the complete feature

**Files:**
- Modify: `README.md`
- Modify: `submission/录屏与提交清单.md`
- Modify: `docs/superpowers/plans/2026-09-01-simple-multi-agent-implementation.md`

- [ ] **Step 1: Update architecture and usage documentation**

Replace the single-Agent description and old success sequence with:

```text
Repair Agent -> fixed pytest verifier -> read-only Reviewer Agent
Reviewer pass -> human approval -> success
Reviewer requests changes -> one Repair Agent rework -> pytest -> final review
Final review still rejects -> failure; no unbounded conversation
```

Document that both roles may use the same configured model name but have separate prompts, contexts, model-client objects, and tool permissions. List the review-only tool set and explain that passing tests and review are complementary gates. Update the “not implemented” list so it no longer claims multi-Agent is absent; continue to state that parallel repair, persistence, long-term memory, MCP/plugins, and Web UI are not implemented.

- [ ] **Step 2: Update trace and recording evidence**

Document safe review events and CLI fields. Add a recording checklist that shows the Reviewer Agent inspecting files, the no-write registry, a pass path, a request-changes/rework path, a final rejection path, and the Git history containing separate real commits.

- [ ] **Step 3: Run the full verification gate**

Run fresh commands:

```powershell
python -m pytest -q
python -m ruff check .
python -m testpilot.demo
git diff --check
```

Expected: all tests pass, Ruff reports no errors, the demo shows both agents and a passing final verifier, and the diff check is empty.

- [ ] **Step 4: Review requirements and failure boundaries**

Check every design requirement against code and tests. Confirm especially that the Reviewer registry cannot mutate, review runs only after the complete repair turn and successful immutable verification, only one rework is granted, approval is last, errors fail closed, and trace/CLI output contains no feedback or source content.

- [ ] **Step 5: Commit documentation and plan history**

```powershell
git add README.md 'submission/录屏与提交清单.md' docs/superpowers/plans/2026-09-01-simple-multi-agent-implementation.md
git commit -m "docs: explain simple multi-agent review"
```

## Final review and integration

- [ ] Re-run the full verification commands immediately before any completion claim.
- [ ] Review the branch diff against `docs/superpowers/specs/2026-09-01-simple-multi-agent-design.md` and fix every critical or important issue.
- [ ] Because workspace subagent credits are unavailable, record that independent subagent review could not run and perform the same spec-compliance and code-quality checklists locally with fresh test evidence.
- [ ] Present the four `finishing-a-development-branch` integration options; do not merge, push, or discard without the user's choice.
