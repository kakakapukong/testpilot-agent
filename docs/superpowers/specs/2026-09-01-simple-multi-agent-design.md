# Simple Multi-Agent Review Design

## Goal

Extend TestPilot from one repair agent into a small, explicit two-agent system without using an Agent framework. The repair agent keeps responsibility for inspecting and editing a Python workspace. After the host verifier passes, a separate read-only reviewer agent checks the current repair. The reviewer may request at most one repair round. Human approval remains the final gate before verified and reviewed changes are accepted.

## Accepted workflow

```text
repair agent edits
    -> immutable host pytest verifier
    -> read-only reviewer agent
       -> pass: human approval
       -> request changes (first review only): feedback to repair agent
            -> repair agent edits
            -> immutable host pytest verifier
            -> final read-only review
               -> pass: human approval
               -> request changes: stop as failure
```

The reviewer never replaces the immutable verifier. Pytest answers whether the configured executable checks pass; the reviewer looks for correctness risks, regressions, and missing coverage that a passing suite may not expose. Human approval is requested only after both gates pass.

## Approaches considered

1. **Repair agent self-review.** This is the smallest change, but it is still one agent with one context and does not demonstrate role separation.
2. **Dedicated read-only reviewer agent (chosen).** A second model loop receives a fresh reviewer prompt and only file-listing, file-reading, text-search, and structured-decision tools. It provides genuine role separation while keeping orchestration deterministic and explainable.
3. **Parallel repair agents with result selection.** This could explore multiple patches, but safe workspace isolation, conflict resolution, and candidate ranking would be much larger than the requested simple multi-agent feature.

## Components and boundaries

### Repair agent

`AgentRunner` remains the top-level coordinator and existing repair loop. Its model keeps the seven current tools. A new optional reviewer dependency preserves direct-library backward compatibility, while the real CLI always injects a reviewer.

### Reviewer agent

A new `ReviewerAgent` owns a fresh bounded context and an independently bounded loop. It receives only:

- the original user task;
- the sorted paths changed by the repair agent;
- the fact that the fixed host verifier exited successfully.

Its registry contains only `list_files`, `read_file`, `search_text`, and `submit_review`. It has no edit, write, command, finish, approval, or network tool. Repository text is treated as untrusted data rather than instructions.

`submit_review` requires a structured decision:

```python
ReviewResult(decision="pass" | "request_changes", feedback="...")
```

The decision must be submitted alone in its model turn so it cannot pretend to have considered tool results it has not yet received. Feedback is non-blank and bounded before it can be returned to the repair loop.

### Coordinator transition

When `finish` triggers a successful host verification, `AgentRunner` calls the reviewer before human approval.

- `pass` keeps verification valid and proceeds to approval.
- The first `request_changes` converts the `finish` result into a structured `review_changes_requested` tool failure containing bounded feedback. This naturally returns the feedback to the repair model in its existing tool-call context.
- Another `finish` is rejected until a new successful source edit occurs.
- After that edit, verification and review run again. A second `request_changes` stops with `review_changes_remaining`; it does not start an unbounded loop.
- Reviewer errors, invalid responses, or budget exhaustion fail closed and never reach human approval.

The reviewer is optional only for direct `AgentRunner` construction and the compatibility surface. The CLI wires it by default so real user runs follow the full two-agent workflow.

## State and observability

`RunState` gains a review phase and stable review accounting:

- final review status: `passed`, `changes_requested`, `unavailable`, or unset;
- number of review attempts;
- number of reviewer-requested repair rounds, limited to one.

JSONL adds review start and completion events with role, round, decision, duration, and feedback length. It never stores reviewer feedback, source text, diff text, prompts, API keys, or changed paths in review events. CLI output prints only the stable review status and counters.

## Error and safety behavior

- A reviewer cannot modify the workspace because its registry contains no mutating tools.
- Generic reviewer/model exceptions are converted to stable, content-free stop reasons.
- `KeyboardInterrupt` at the reviewer boundary fails closed; `SystemExit` is not swallowed.
- A reviewer pass never bypasses pytest, protected paths, command restrictions, or human approval.
- If a run stops before approval, existing TestPilot behavior is retained: partial files remain available for inspection. Human rejection or unavailable human input still uses the existing journal rollback.
- The two agents may use the same configured model name, but they are separate runtime objects with different prompts, contexts, tools, and responsibilities.

## Testing strategy

1. Reviewer unit tests prove its tool set is read-only, decisions are structured, mixed decision/tool turns are rejected, feedback is bounded, and failures are sanitized.
2. Coordinator tests prove the exact order `verify -> review -> approve`, one feedback-driven repair round, mandatory edit before re-review, final rejection, and fail-closed reviewer errors.
3. CLI tests prove separate repair/reviewer clients and registries are assembled and stable review fields are printed.
4. A deterministic offline demo and end-to-end test exercise a repair, passing pytest, reviewer inspection, reviewer pass, and final success without an API key.
5. Full verification runs pytest, Ruff, the offline demo, and `git diff --check`.
