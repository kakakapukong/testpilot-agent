# TestPilot Web Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a localhost Codex-style web console that starts one real TestPilot repair, streams compact events, and approves or rejects with buttons.

**Architecture:** `python -m testpilot.web` serves `static/console.html` and a small JSON/SSE API. A single in-process coordinator builds the same `AgentRunner` as CLI, wraps trace + stdin approval with a queue, and never binds off loopback.

**Tech Stack:** Python 3.11 stdlib `http.server`, existing `AgentRunner` / `FakeModel` tests, HTML/CSS/JS (no React).

---

### Task 1: Run coordinator HTTP API

**Files:**
- Create: `src/testpilot/web.py`
- Create: `tests/test_web.py`
- Modify: `pyproject.toml` (scripts + package data)

- [ ] Tests first for missing env, 409 on second run, approval queue, secret-free HTML.
- [ ] Coordinator + ThreadingHTTPServer on 127.0.0.1.
- [ ] Wrap `JsonlTrace.record` into public SSE events; approval `input_fn` blocks on a queue.
- [ ] Commit.

### Task 2: Codex-style console page

**Files:**
- Create: `src/testpilot/static/console.html`

- [ ] Dark session layout: transcript, composer, approval card, status strip.
- [ ] `EventSource` for `/api/runs/current/events`; POST run and approval.
- [ ] No source/diff/key in the UI.
- [ ] Commit.

### Task 3: Docs and demo wiring

**Files:**
- Modify: `README.md`
- Modify: `src/testpilot/__init__.py` if needed for export

- [ ] Document `python -m testpilot.web`.
- [ ] Full pytest + ruff.
- [ ] Commit.
