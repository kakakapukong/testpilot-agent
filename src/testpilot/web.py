"""Localhost Codex-style console for operating one TestPilot repair."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from typing import Any
from urllib.parse import urlparse

from .agent import AgentRunner
from .cli import (
    _ConfigError,
    _fresh_setup,
    _result_fields,
    _workspace_path,
    build_agent,
)

STATIC_PAGE = Path(__file__).resolve().parent / "static" / "console.html"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_APPROVAL_DECISIONS = frozenset({"approved", "rejected"})


class WebError(RuntimeError):
    """A safe, public web failure."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class FanoutTrace:
    """Copy trace records to the browser without changing on-disk JSONL rules."""

    def __init__(self, inner: Any, emit: Callable[[dict[str, Any]], None]) -> None:
        self._inner = inner
        self._emit = emit
        self.path = getattr(inner, "path", None)

    def record(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self._inner.record(event, payload)
        public = _public_trace_event(event, payload or {})
        if public is not None:
            self._emit(public)


class RunCoordinator:
    """Own at most one in-process Agent run and its approval queue."""

    def __init__(
        self,
        *,
        runner_factory: Callable[..., AgentRunner] | None = None,
    ) -> None:
        self._runner_factory = runner_factory or _default_runner
        self._lock = threading.Lock()
        self._busy = False
        self._waiting_approval = False
        self._events: Queue[dict[str, Any] | None] = Queue()
        self._approval: Queue[str] = Queue()
        self._thread: threading.Thread | None = None
        self._last_status: dict[str, Any] | None = None
        self._approval_lines: list[str] = []

    def start(self, workspace: str, verify: str, task: str) -> dict[str, Any]:
        with self._lock:
            if self._busy:
                raise WebError("a run is already active", 409)
            self._busy = True
            self._waiting_approval = False
            self._last_status = None
            self._approval_lines = []
            self._drain(self._events)
            self._drain(self._approval)
        try:
            path = _workspace_path(workspace)
            setup = _fresh_setup(
                path,
                verify=verify,
                task=task,
                trace=None,
                max_iterations=None,
                output_fn=self._output,
            )
        except _ConfigError as exc:
            with self._lock:
                self._busy = False
            raise WebError(str(exc), 400) from None
        except (OSError, RuntimeError, TypeError, ValueError):
            with self._lock:
                self._busy = False
            raise WebError("could not start run", 400) from None

        def worker() -> None:
            try:
                agent = self._runner_factory(
                    setup.config,
                    setup.journal,
                    setup.checkpoint,
                    self._input,
                    self._output,
                )
                if getattr(agent, "trace", None) is not None:
                    agent.trace = FanoutTrace(agent.trace, self._emit)
                result = agent.run(setup.config.task)
                status = _status_payload(result, setup.config.trace_path)
                self._last_status = status
                self._emit({"type": "status", **status})
            except WebError as exc:
                self._last_status = {"STATUS": "FAILED", "stop_reason": str(exc)}
                self._emit({"type": "error", "message": str(exc)})
            except Exception:  # noqa: BLE001 - host boundary; never leak SDK text.
                self._last_status = {
                    "STATUS": "FAILED",
                    "stop_reason": "runtime_setup_failed",
                }
                self._emit({"type": "error", "message": "run failed"})
            finally:
                self._events.put(None)
                with self._lock:
                    self._busy = False
                    self._waiting_approval = False

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        return {"ok": True, "workspace": str(path)}

    def decide(self, decision: str) -> dict[str, Any]:
        if decision not in _APPROVAL_DECISIONS:
            raise WebError("decision must be approved or rejected", 400)
        with self._lock:
            if not self._waiting_approval:
                raise WebError("approval is not required right now", 409)
        self._approval.put(decision)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=30)
        return self._last_status or {"ok": True, "decision": decision}

    def events(self) -> Queue[dict[str, Any] | None]:
        return self._events

    def _emit(self, event: dict[str, Any]) -> None:
        self._events.put(event)

    def _output(self, line: str) -> None:
        if not isinstance(line, str):
            return
        self._approval_lines.append(line)
        if line == "APPROVAL_REQUIRED":
            self._emit(
                {
                    "type": "approval_required",
                    "lines": list(self._approval_lines),
                }
            )
            return
        if line.startswith(("run_id=", "checkpoint=", "verification_exit=", "M ", "A ")):
            self._emit({"type": "log", "text": line})

    def _input(self, prompt: str) -> str:
        del prompt
        with self._lock:
            self._waiting_approval = True
        decision = self._approval.get()
        with self._lock:
            self._waiting_approval = False
        return "y" if decision == "approved" else "n"

    @staticmethod
    def _drain(queue: Queue[Any]) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return


class WebApp:
    """HTTP API plus the static console page."""

    def __init__(self, *, runner_factory: Callable[..., AgentRunner] | None = None) -> None:
        self.coordinator = RunCoordinator(runner_factory=runner_factory)

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        path = parsed.path
        method = handler.command
        try:
            if method == "GET" and path == "/":
                self._send(handler, 200, _read_page(), "text/html; charset=utf-8")
                return
            if method == "POST" and path == "/api/runs":
                body = _read_json(handler)
                result = self.coordinator.start(
                    str(body.get("workspace", "")),
                    str(body.get("verify", "")),
                    str(body.get("task", "")),
                )
                self._send_json(handler, 200, result)
                return
            if method == "POST" and path == "/api/runs/current/approval":
                body = _read_json(handler)
                result = self.coordinator.decide(str(body.get("decision", "")))
                self._send_json(handler, 200, result)
                return
            if method == "GET" and path == "/api/runs/current/events":
                self._stream_events(handler)
                return
        except WebError as exc:
            self._send_json(handler, exc.status, {"error": str(exc)})
            return
        self._send_json(handler, 404, {"error": "not found"})

    def _stream_events(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()
        queue = self.coordinator.events()
        while True:
            try:
                item = queue.get(timeout=15)
            except Empty:
                handler.wfile.write(b": keepalive\n\n")
                handler.wfile.flush()
                continue
            if item is None:
                handler.wfile.write(b"event: end\ndata: {}\n\n")
                handler.wfile.flush()
                return
            payload = json.dumps(item, ensure_ascii=True, separators=(",", ":"))
            handler.wfile.write(f"data: {payload}\n\n".encode())
            handler.wfile.flush()

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _send_json(self, handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        self._send(handler, status, body, "application/json; charset=utf-8")


def make_server(app: WebApp, *, host: str, port: int) -> ThreadingHTTPServer:
    if host not in _ALLOWED_HOSTS:
        raise WebError("web console only binds to localhost", 400)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            app.handle(self)

        def do_POST(self) -> None:
            app.handle(self)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TestPilot's local web console.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        server = make_server(WebApp(), host=args.host, port=args.port)
    except WebError as exc:
        print(str(exc))
        return 1
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"TestPilot console: {url}")
    print("API keys stay in the terminal environment. Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


def _default_runner(config, journal, checkpoint, input_fn, output_fn) -> AgentRunner:
    return build_agent(
        config,
        journal=journal,
        checkpoint=checkpoint,
        input_fn=input_fn,
        output_fn=output_fn,
    )


def _read_page() -> bytes:
    if not STATIC_PAGE.is_file():
        raise WebError("console page is missing", 500)
    return STATIC_PAGE.read_bytes()


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length < 0 or length > 32_768:
        raise WebError("request is too large", 400)
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise WebError("request must be JSON", 400) from None
    if not isinstance(payload, dict):
        raise WebError("request must be an object", 400)
    return payload


def _public_trace_event(event: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if event in {"run_start", "stop"}:
        return {"type": "phase", "event": event, "reason": payload.get("reason")}
    if event == "model_turn":
        return {"type": "phase", "event": "model", "stage": payload.get("stage")}
    if event in {"review", "memory_retrieval", "memory_saved", "checkpoint"}:
        return {
            "type": "phase",
            "event": event,
            "ok": payload.get("ok"),
            "stage": payload.get("stage"),
        }
    return None


def _status_payload(result: object, trace_path: Path) -> dict[str, Any]:
    return dict(_result_fields(result, trace_path))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
