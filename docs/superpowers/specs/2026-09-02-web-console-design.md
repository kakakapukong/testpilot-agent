# TestPilot Web Console Design

## Goal

Add a local, Codex-style web console that can operate a real TestPilot repair run from the browser: submit workspace, verifier, and task; stream compact run events; and approve or reject with buttons instead of terminal `y`/`n`. This is for答辩演示 and local use. It does not replace the CLI.

## Accepted workflow

```text
operator opens python -m testpilot.web
    -> browser form: workspace, verify command, task
    -> Run starts one AgentRunner in a background thread
    -> event stream shows tool names, phase, compact summaries
    -> pytest + Reviewer pass
    -> approval card: Approve or Reject
    -> same success/rollback/memory rules as CLI
```

API keys stay in process environment variables (`OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL`). The page never accepts, stores, or redisplays a key.

## Approaches considered

1. **Local stdlib HTTP server inside this repo (chosen).** `python -m testpilot.web` serves one dark page and a small JSON/SSE API. No React, no Streamlit, no extra Agent framework. Fits答辩, keeps the Python-only story, and reuses `AgentRunner`.
2. **Streamlit or Gradio.** Faster to sketch, but the layout looks like a dashboard, not a Codex session, and adds another product dependency.
3. **Separate React frontend.** Closer to a shipped product, but splits the repository and is too large for this round.

## UI

Desktop-first dark session console, IBM Plex Mono / system monospace, tight spacing, no marketing chrome.

- **Transcript:** user task, then compact events (tool name, path or status, phase). No source, no diff body, no memory text, no API key.
- **Composer (bottom):** workspace path, verify command, task, Run. Run is disabled while a run is active.
- **Approval card:** appears only after Reviewer pass, showing the same safe summary as CLI (`verification_exit` and `M/A "path" (+n/-n)`). Buttons: Approve, Reject.
- **Status strip:** `STATUS`, `stop_reason`, `review`, `approval`, `run_id`, `resume_available`, `memory_saved`.
- **Empty / error states:** missing env vars, invalid workspace, model failure, and “another run is active” each have a short, non-secret message.

One primary action per screen: Run, or Approve/Reject when a card is open.

## Components and boundaries

### Web server

New `src/testpilot/web.py` serves `src/testpilot/static/console.html` from a localhost-only `ThreadingHTTPServer`. Default bind: `127.0.0.1:8765`. It does not listen on `0.0.0.0`.

Routes:

- `GET /` — the console page
- `POST /api/runs` — start a fresh run; body `{workspace, verify, task}`; rejects if a run is already active
- `GET /api/runs/current/events` — Server-Sent Events for the active run
- `POST /api/runs/current/approval` — body `{decision: "approved"|"rejected"}`; only valid while waiting for approval

No resume-from-web in v1. Operators who need resume use the existing CLI `--resume`.

### Run coordinator

A single in-process coordinator owns at most one `AgentRunner`. It translates host events into JSON objects:

```text
{ "type": "phase"|"tool"|"verify"|"review"|"approval_required"|"status"|"error", ... }
```

Tool events include tool name and a length/path summary, never file contents. Approval events include the same compact summary the CLI prints.

### Approval adapter

CLI approval currently reads stdin. The web coordinator supplies an approval function that blocks on a thread-safe queue filled by `POST /api/runs/current/approval`. Timeout or server shutdown treats the decision as unavailable/rejected using the existing rollback path. Demo’s simulated approval is unchanged and unused by this console.

### Existing agents

Repair, Reviewer, Memory, checkpoint, verifier, and trace stay as they are. The console is another host, like CLI, not a fourth agent.

## Data flow

1. Browser posts run request.
2. Coordinator validates workspace exists, verify command parses, env vars are non-empty, and no run is active.
3. Background thread builds the same objects CLI uses (`build_agent`) and calls `run(task)`.
4. Trace/host callbacks enqueue compact events; SSE flushes them to the browser.
5. When the runner needs human approval, it waits on the queue; the page shows the card.
6. Approve or Reject unblocks the runner. Final status event closes the stream.

## Error handling

- Missing `OPENAI_API_KEY` or `OPENAI_MODEL`: HTTP 400 with `stop_reason=runtime_setup_failed`; page tells the operator to set env vars in the terminal.
- Invalid workspace or verify command: HTTP 400, no model call.
- Second concurrent Run: HTTP 409.
- Model/tool/verify failures: stream a `status` event with the same public fields CLI prints; never include SDK exception text that might contain URLs with credentials.
- Approval posted when not waiting: HTTP 409.
- Browser disconnect does not auto-approve; the run keeps waiting until Reject, Approve, or process exit.

## Testing

- Page `GET /` returns HTML and does not embed env secrets.
- Starting a run with missing env or bad workspace fails without creating a runner.
- A fake-model run streams tool/phase events and reaches `approval_required`.
- Approve keeps the edited file bytes; Reject restores the pre-run bytes (same journal semantics as CLI).
- A second `POST /api/runs` during an active run returns 409.
- Event payloads never contain API keys, file source, or memory body.

## Out of scope

- Offline `testpilot.demo` button
- Resume from the web
- Multiple concurrent sessions, accounts, or public deployment
- Showing source, diffs, traces, or memory JSONL
- Editing API keys in the browser
- Web UI for parallel repair

## Compliance

Still no LangChain / Agents SDK / AutoGen / CrewAI / Claude Code / Codex / OpenCode / MCP. The page is a local host UI over the existing in-repo agent. Optional API extra remains the only network client.
