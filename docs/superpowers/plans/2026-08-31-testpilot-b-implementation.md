# TestPilot B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python coding agent that owns its tool loop and only reports success after an independent post-edit verification command passes.

**Architecture:** A protocol-based model adapter feeds normalized turns to a single `AgentRunner`. A static registry executes workspace-confined tools and returns structured results; a separate verifier, state object, context limiter, and JSONL trace keep completion deterministic and auditable.

**Tech Stack:** Python 3.11+, standard library, OpenAI Python client as an optional API dependency, pytest, Ruff.

---

## File map

- `pyproject.toml`: package metadata, optional API dependency, test/lint configuration.
- `src/testpilot/types.py`: normalized tool calls, assistant turns, results, state, and run result.
- `src/testpilot/registry.py`: tool protocol, JSON-Schema validation, static registration and dispatch.
- `src/testpilot/workspace.py`: path confinement and atomic UTF-8 file operations.
- `src/testpilot/tools.py`: list/read/search/edit/write tool implementations.
- `src/testpilot/command.py`: allowlisted, timeout-bound subprocess execution.
- `src/testpilot/trace.py`: append-only JSONL event trace.
- `src/testpilot/context.py`: bounded message history that preserves system/user anchors.
- `src/testpilot/model.py`: `ModelClient`, scripted `FakeModel`, and OpenAI-compatible adapter.
- `src/testpilot/agent.py`: autonomous loop, state transitions and independent finish verification.
- `src/testpilot/cli.py`, `src/testpilot/__main__.py`: command-line entry point.
- `tests/`: behavior-first unit and integration tests.

### Task 1: Package skeleton and core value types

**Files:**
- Create: `pyproject.toml`
- Create: `src/testpilot/__init__.py`
- Create: `src/testpilot/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing value-type tests**

```python
from testpilot.types import RunPhase, RunState, ToolResult


def test_tool_result_serializes_structured_failure():
    result = ToolResult.failure("outside workspace", code="path_outside_workspace")
    assert result.to_dict() == {
        "ok": False,
        "data": None,
        "error": "outside workspace",
        "error_code": "path_outside_workspace",
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "truncated": False,
    }


def test_run_state_records_an_edit():
    state = RunState()
    state.record_edit("src/app.py")
    assert state.phase is RunPhase.EDIT
    assert state.edit_count == 1
    assert state.changed_files == {"src/app.py"}
    assert not state.verified_after_last_edit
```

- [ ] **Step 2: Run `python -m pytest tests/test_types.py -q` and confirm import failure**

- [ ] **Step 3: Implement enums and dataclasses**

Implement immutable `ToolCall` and `AssistantTurn`, plus `ToolResult.success`, `ToolResult.failure`, `ToolResult.to_dict`, `RunState.record_edit`, `RunState.record_verification`, and `AgentRunResult`. Do not add persistence or long-term memory fields.

- [ ] **Step 4: Re-run the focused test, then the full suite**

- [ ] **Step 5: Commit as `feat(core): define agent state and tool results`**

### Task 2: Static tool registry and schema validation

**Files:**
- Create: `src/testpilot/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write failing tests for schema export and dispatch errors**

```python
def test_registry_rejects_unknown_tool(registry):
    result = registry.execute("missing", {})
    assert not result.ok
    assert result.error_code == "unknown_tool"


def test_registry_rejects_missing_required_argument(registry):
    result = registry.execute("echo", {})
    assert not result.ok
    assert result.error_code == "invalid_arguments"
```

- [ ] **Step 2: Run the focused tests and confirm they fail for missing registry code**

- [ ] **Step 3: Implement `Tool` protocol and `ToolRegistry`**

Support object schemas with `required`, primitive `string`/`integer`/`boolean`/`array` checks, `additionalProperties=False`, duplicate-name rejection, OpenAI function-tool export, exception-to-`tool_exception` conversion, and sequential dispatch.

- [ ] **Step 4: Run focused and full tests**

- [ ] **Step 5: Commit as `feat(tools): add static registry and argument validation`**

### Task 3: Workspace-confined file tools

**Files:**
- Create: `src/testpilot/workspace.py`
- Create: `src/testpilot/tools.py`
- Test: `tests/test_workspace_tools.py`

- [ ] **Step 1: Write failing tests**

Cover: `../` rejection; absolute outside path rejection; symlink escape rejection where supported; UTF-8 read with line ranges; list result limit; plain-text search; atomic write; exact edit succeeds once; zero and multiple matches do not modify the file.

```python
def test_edit_rejects_ambiguous_match(toolset, workspace):
    target = workspace / "a.py"
    target.write_text("x = 1\nx = 1\n", encoding="utf-8")
    result = toolset["edit_file"].execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"}
    )
    assert not result.ok
    assert result.error_code == "ambiguous_edit"
    assert target.read_text(encoding="utf-8") == "x = 1\nx = 1\n"
```

- [ ] **Step 2: Run focused tests and observe expected failures**

- [ ] **Step 3: Implement `Workspace` and five file tools**

All paths pass through `Path.resolve()` and `relative_to(root)`. Limit individual reads to 50,000 characters, listings/searches to configurable result counts, reject binary/NUL content, create parent directories only inside the workspace, and write via `NamedTemporaryFile` followed by `os.replace`.

- [ ] **Step 4: Run focused and full tests**

- [ ] **Step 5: Commit as `feat(tools): add confined file exploration and editing`**

### Task 4: Safe command runner, verifier and trace

**Files:**
- Create: `src/testpilot/command.py`
- Create: `src/testpilot/trace.py`
- Test: `tests/test_command_and_trace.py`

- [ ] **Step 1: Write failing command tests**

```python
def test_command_runner_rejects_unlisted_program(runner):
    result = runner.run(["powershell", "-Command", "Write-Output bad"])
    assert not result.ok
    assert result.error_code == "command_not_allowed"


def test_command_runner_keeps_nonzero_exit_code(runner):
    result = runner.run([sys.executable, "-c", "raise SystemExit(7)"])
    assert not result.ok
    assert result.exit_code == 7
```

Also test timeout, stdout/stderr separation, truncation markers, cwd confinement, and that JSONL emits one valid object per event without environment values.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement `CommandRunner`, `RunCommandTool`, `FinishTool`, `Verifier`, and `JsonlTrace`**

Use argument arrays, `shell=False`, workspace cwd, pinned Python/pytest executables, configurable timeout, and a 20,000-character cap per stream. Model commands and the verifier accept only restricted pytest arguments and workspace targets; `finish` itself never marks success.

- [ ] **Step 4: Run focused and full tests**

- [ ] **Step 5: Commit as `feat(exec): add bounded commands verification and traces`**

### Task 5: Context management and model adapters

**Files:**
- Create: `src/testpilot/context.py`
- Create: `src/testpilot/model.py`
- Test: `tests/test_context_and_model.py`

- [ ] **Step 1: Write failing tests**

Test that context always retains the developer/system and original user task, keeps only the newest configured tail, never leaves a tool message without its assistant tool call, and truncates oversized tool content. Test `FakeModel` exhaustion. With a fake SDK response, test normalization of assistant content, tool-call IDs, names and JSON arguments; malformed JSON must remain a callable turn whose registry result is `invalid_arguments` rather than crashing.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement bounded context and adapters**

Define a `ModelClient` protocol, `FakeModel(scripted_turns)`, `OpenAIChatModel` using `client.chat.completions.create(messages=..., tools=..., tool_choice="auto")`, and a `ModelError`. Read API configuration only in the CLI; never log it. Retry transient SDK exceptions at most twice with short exponential delays and fail authentication-style errors immediately when detectable.

- [ ] **Step 4: Run focused and full tests without a network request**

- [ ] **Step 5: Commit as `feat(model): normalize tool calls and bound context`**

### Task 6: Verification-gated Agent loop

**Files:**
- Create: `src/testpilot/agent.py`
- Test: `tests/test_agent.py`
- Test: `tests/test_agent_e2e.py`

- [ ] **Step 1: Write failing deterministic loop tests**

Script FakeModel turns for: read -> edit -> finish; tool error -> corrected call; failed finish verification -> second edit -> successful finish; no-tool response failure; unknown tool recovery; maximum iterations; repeated identical calls. Assert tool results contain matching `tool_call_id` values and success requires verification after the last edit.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement `AgentRunner.run(task)`**

Initialize anchored messages and `RunState`; ask the model with registry schemas; append the full assistant turn; execute calls sequentially; append one JSON tool result per call; update edits only from successful write/edit results; treat `finish` as a verifier request; continue after failed verification; stop on explicit verified finish, no-tool output, maximum iterations, repeated identical calls, or `ModelError`. Emit trace events at every transition.

- [ ] **Step 4: Add the temporary buggy-repository end-to-end test**

Create a calculator with a failing subtraction test inside `tmp_path`, script FakeModel to inspect and uniquely edit the implementation, and use `(sys.executable, "-m", "pytest", "-q")` as the immutable verifier. Assert exit success, changed file tracking, and final green verification.

- [ ] **Step 5: Run focused and full tests**

- [ ] **Step 6: Commit as `feat(agent): enforce an independently verified tool loop`**

### Task 7: CLI, documentation and offline demonstration

**Files:**
- Create: `src/testpilot/cli.py`
- Create: `src/testpilot/__main__.py`
- Create: `.env.example`
- Create: `README.md`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Test `--help`, nonexistent workspace, missing `OPENAI_API_KEY`, missing model, verifier parsing, nonzero process exit on failed Agent result, and that secrets never appear in captured output.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement the CLI**

Expose `python -m testpilot --workspace PATH --verify "python -m pytest -q" "TASK"`. Read `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and `OPENAI_MODEL`; initialize the SDK lazily only after validation. Print a compact final status, stop reason, changed files, verification exit code and trace path.

- [ ] **Step 4: Document architecture, setup, security boundaries, commands and demo procedure**

The README must explicitly say no Agent framework is used, the API key is an environment variable, tool execution is local, and the current release intentionally omits multi-Agent, MCP, plugins and long-term memory.

- [ ] **Step 5: Run focused tests, full tests, Ruff, build metadata check and `python -m testpilot --help`**

- [ ] **Step 6: Commit as `docs: add CLI setup and defensible design notes`**

### Task 8: Final review and release evidence

**Files:**
- Modify only files identified by reviewers.

- [ ] **Step 1: Run a spec-compliance review against the approved B design**
- [ ] **Step 2: Fix all missing requirements with a new failing test for every behavior change**
- [ ] **Step 3: Run an independent code-quality and security review**
- [ ] **Step 4: Fix all critical/important findings and re-review**
- [ ] **Step 5: Run `python -m pytest -q`, `python -m ruff check .`, `python -m ruff format --check .`, and `python -m testpilot --help` from a clean environment**
- [ ] **Step 6: Inspect `git diff --check`, `git status`, commit history, and tracked files for secrets**
- [ ] **Step 7: Commit as `chore: prepare TestPilot B release` only after all evidence is green**
