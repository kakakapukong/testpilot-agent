"""Localhost Grok-style console for operating one TestPilot repair."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from collections.abc import Callable, Mapping
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from time import monotonic
from typing import Any
from urllib.parse import urlparse

from .agent import AgentRunner
from .cli import (
    _ConfigError,
    _fresh_setup,
    _result_fields,
    _workspace_path,
    build_agent,
    credentials_status,
    load_saved_credentials,
)

STATIC_PAGE = Path(__file__).resolve().parent / "static" / "console.html"
PREFS_PATH = Path.home() / ".testpilot" / "web-prefs.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_VERIFY = f'"{sys.executable}" -m pytest -q'
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_APPROVAL_DECISIONS = frozenset({"approved", "rejected"})
_TOOL_TITLES = {
    "list_files": "列出工作区文件",
    "read_file": "阅读文件",
    "search_text": "搜索代码",
    "edit_file": "修改文件",
    "write_file": "写入新文件",
    "run_command": "运行命令",
    "finish": "申请宿主验证",
}


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


class ToolEventRegistry:
    """Forward tools to the real registry and emit compact, content-free events."""

    def __init__(self, inner: Any, emit: Callable[[dict[str, Any]], None]) -> None:
        self._inner = inner
        self._emit = emit

    def names(self) -> Any:
        return self._inner.names()

    def schemas(self) -> Any:
        return self._inner.schemas()

    def execute(self, name: str, arguments: Mapping[str, Any]) -> Any:
        path = arguments.get("path") if isinstance(arguments, Mapping) else None
        if not isinstance(path, str) or len(path) > 200:
            path = None
        command = None
        raw_command = arguments.get("argv") if isinstance(arguments, Mapping) else None
        if isinstance(raw_command, (list, tuple)):
            command = " ".join(str(part) for part in raw_command)[:160]
        started = monotonic()
        self._emit(
            {
                "type": "tool",
                "stage": "start",
                "name": name,
                "path": path,
                "command": command,
            }
        )
        result = self._inner.execute(name, arguments)
        duration_ms = int((monotonic() - started) * 1000)
        ok = bool(getattr(result, "ok", False))
        error_code = getattr(result, "error_code", None)
        self._emit(
            {
                "type": "tool",
                "stage": "complete",
                "name": name,
                "path": path,
                "command": command,
                "ok": ok,
                "error_code": error_code if isinstance(error_code, str) else None,
                "duration_ms": duration_ms,
            }
        )
        return result


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
        self._started_at = monotonic()

    def start(self, workspace: str, verify: str, task: str) -> dict[str, Any]:
        with self._lock:
            if self._busy:
                raise WebError("a run is already active", 409)
            self._busy = True
            self._waiting_approval = False
            self._last_status = None
            self._approval_lines = []
            self._started_at = monotonic()
            self._drain(self._events)
            self._drain(self._approval)
        try:
            load_saved_credentials()
            path = _workspace_path(workspace)
            command = verify.strip() if isinstance(verify, str) else ""
            setup = _fresh_setup(
                path,
                verify=command or DEFAULT_VERIFY,
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
                if getattr(agent, "registry", None) is not None:
                    agent.registry = ToolEventRegistry(agent.registry, self._emit)
                self._emit(
                    {
                        "type": "note",
                        "title": "任务已开始",
                        "detail": (
                            "流程是：检索经验 → Repair 必须先改源码 → 宿主 pytest → "
                            "只读 Reviewer → 你批准或拒绝 → 成功后写入记忆。"
                            "若代码已经是对的，请先把 bug 改回去再运行。"
                        ),
                    }
                )
                remember_workspace(str(path))
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
        self._events.put(_decorate_event(event, self._started_at))

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
            if method == "GET" and path == "/api/bootstrap":
                self._send_json(handler, 200, bootstrap_payload())
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
    load_saved_credentials()
    print(f"TestPilot console: {url}")
    status = credentials_status()
    if status["credentials_ready"]:
        print(f"Loaded model={status['model']} from env or {status['credential_file']}")
    else:
        print(f"Missing API settings. Put them in {status['credential_file']} or set env vars.")
    print("Press Ctrl+C to stop.")
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
    stage = payload.get("stage")
    duration = payload.get("duration_ms")
    if event == "verification":
        if stage == "start":
            return {
                "type": "phase",
                "event": event,
                "stage": stage,
                "title": "宿主正在执行固定 pytest",
                "detail": "这是验证门：由系统跑测试，不是模型口头说通过。",
            }
        ok = payload.get("ok")
        exit_code = payload.get("exit_code")
        return {
            "type": "phase",
            "event": event,
            "stage": stage,
            "title": "宿主验证通过" if ok else "宿主验证未通过",
            "detail": (
                f"exit_code={exit_code}"
                if ok
                else f"exit_code={exit_code}。若尚未修改源码，finish 会被拒绝并要求继续修复。"
            ),
            "ok": ok,
            "duration_ms": duration,
        }
    if event == "run_start":
        return {
            "type": "phase",
            "event": event,
            "title": "Repair Agent 开始工作",
            "detail": "接下来会浏览代码、修改、然后交给宿主 pytest。",
        }
    if event == "stop":
        reason = payload.get("reason")
        return {
            "type": "phase",
            "event": event,
            "title": "本轮结束",
            "detail": f"停止原因：{reason}" if reason else None,
        }
    if event == "model_turn":
        if stage == "start":
            return {
                "type": "phase",
                "event": "model",
                "stage": stage,
                "title": "正在请求 Repair 模型",
                "detail": "模型根据当前任务和工具结果决定下一步：读文件、改文件，或申请验证。",
            }
        return {
            "type": "phase",
            "event": "model",
            "stage": stage,
            "title": "模型已返回",
            "detail": None if duration is None else f"耗时 {duration} ms",
            "duration_ms": duration,
        }
    if event == "memory_retrieval":
        if stage == "start":
            return {
                "type": "phase",
                "event": event,
                "stage": stage,
                "title": "正在检索本仓库的历史修复经验",
                "detail": "最多注入 3 条相关记忆；Reviewer 看不到这些内容。",
            }
        hits = payload.get("hit_count")
        return {
            "type": "phase",
            "event": event,
            "stage": stage,
            "title": "历史经验检索完成",
            "detail": f"命中 {hits} 条" if hits is not None else None,
        }
    if event == "checkpoint":
        if stage == "save":
            return {
                "type": "phase",
                "event": event,
                "stage": stage,
                "title": "已写入断点",
                "detail": "如果现在中断，可以用同一个 run_id 从命令行恢复。",
                "duration_ms": duration,
            }
        return {
            "type": "phase",
            "event": event,
            "stage": stage,
            "title": "断点已收尾",
            "detail": None,
            "duration_ms": duration,
        }
    if event == "review":
        error_code = payload.get("error_code")
        if stage == "start":
            title = "Reviewer 正在只读检查"
            detail = "Reviewer 不能改文件，只判断当前修复是否可进入人工审批。"
        elif error_code:
            title = "Reviewer 未能给出有效结论"
            detail = _stop_reason_text(error_code)
        else:
            title = "Reviewer 检查结束"
            detail = "Reviewer 不能改文件，只判断当前修复是否可进入人工审批。"
        return {
            "type": "phase",
            "event": event,
            "stage": stage,
            "title": title,
            "detail": detail,
            "ok": payload.get("ok"),
            "error_code": error_code,
            "duration_ms": duration,
        }
    if event == "memory_saved":
        return {
            "type": "phase",
            "event": event,
            "title": "经验已写入本地记忆库",
            "detail": "只在 pytest、Reviewer、人工批准都通过后才会保存。",
        }
    return None


def _decorate_event(event: dict[str, Any], started_at: float) -> dict[str, Any]:
    decorated = dict(event)
    decorated.setdefault("t", datetime.now().astimezone().strftime("%H:%M:%S"))
    decorated.setdefault("elapsed_ms", int((monotonic() - started_at) * 1000))
    if decorated.get("type") == "tool":
        name = str(decorated.get("name") or "tool")
        action = _TOOL_TITLES.get(name, name)
        path = decorated.get("path")
        stage = decorated.get("stage")
        duration = decorated.get("duration_ms")
        hint = path or decorated.get("command")
        if stage == "start":
            decorated["title"] = f"开始{action}"
            decorated["detail"] = hint
        else:
            flag = "完成" if decorated.get("ok") else "失败"
            suffix = f"（{duration} ms）" if isinstance(duration, int) else ""
            decorated["title"] = f"{flag}{action}{suffix}"
            error_code = decorated.get("error_code")
            if not decorated.get("ok") and isinstance(error_code, str):
                decorated["detail"] = "；".join(part for part in (hint, error_code) if part)
            else:
                decorated["detail"] = hint
    if decorated.get("type") == "approval_required":
        decorated.setdefault("title", "等待你批准改动")
        decorated.setdefault(
            "detail",
            "pytest 和 Reviewer 已通过。Approve 保留修改，Reject 回滚到运行前。",
        )
    if decorated.get("type") == "status":
        status = decorated.get("STATUS")
        reason = decorated.get("stop_reason")
        decorated.setdefault(
            "title",
            "修复成功" if status == "SUCCESS" else "本轮未接受为成功",
        )
        decorated.setdefault("detail", _stop_reason_text(reason))
    if decorated.get("type") == "error":
        decorated.setdefault("title", "运行失败")
        decorated.setdefault("detail", decorated.get("message"))
    decorated.setdefault("title", decorated.get("event") or decorated.get("type") or "event")
    return decorated


def bootstrap_payload() -> dict[str, Any]:
    load_saved_credentials()
    prefs = _read_prefs()
    suggestions = [
        item
        for item in prefs.get("recent_workspaces", [])
        if isinstance(item, str) and _usable_workspace_path(item)
    ]
    for candidate in (
        Path.home() / "Desktop" / "IRdrop" / "sample-calc",
        Path.cwd() / "demo-workspace",
    ):
        text = str(candidate)
        if candidate.is_dir() and _usable_workspace_path(text) and text not in suggestions:
            suggestions.append(text)
    return {
        **credentials_status(),
        "default_verify": prefs.get("verify")
        if isinstance(prefs.get("verify"), str) and prefs.get("verify")
        else DEFAULT_VERIFY,
        "recent_workspaces": suggestions[:8],
        "default_task": "修改 calculator.py 中的 subtract，使其做减法而不是加法；不要修改 tests",
    }


def _usable_workspace_path(workspace: str) -> bool:
    lowered = workspace.replace("/", "\\").lower()
    if "\\temp\\pytest-" in lowered or "\\pytest-of-" in lowered:
        return False
    path = Path(workspace)
    return path.is_dir()


def remember_workspace(workspace: str) -> None:
    if not _usable_workspace_path(workspace):
        return
    prefs = _read_prefs()
    recent = [
        item
        for item in prefs.get("recent_workspaces", [])
        if isinstance(item, str) and _usable_workspace_path(item)
    ]
    recent = [workspace, *[item for item in recent if item != workspace]][:8]
    prefs["recent_workspaces"] = recent
    prefs["verify"] = DEFAULT_VERIFY
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(
            json.dumps(prefs, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        return


def _read_prefs() -> dict[str, Any]:
    try:
        payload = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stop_reason_text(reason: object) -> str:
    if reason == "model_stopped_without_finish":
        return (
            "模型没有调用 finish 申请宿主验证就结束了。"
            "常见原因是测试已经通过，或它只自己跑了命令。"
            "请确认 subtract 现在是错的，然后再运行一次。"
        )
    if reason == "max_iterations":
        return "模型来回次数用完，还没有完成 pytest → Reviewer → 批准。"
    if reason == "review_unavailable":
        return "Reviewer 运行失败。Repair 可能已经改对了代码，但独立审查没有给出 pass/request_changes。"
    if reason == "reviewer_stopped_without_decision":
        return "Reviewer 看完代码后没有调用 submit_review，所以不能进入批准。"
    if reason == "review_model_failed":
        return "Reviewer 请求模型失败。请再运行一次。"
    if reason == "review_max_iterations":
        return "Reviewer 检查次数用完，仍未提交结论。"
    if reason == "review_invalid_response":
        return "Reviewer 返回的结论格式无效。"
    if isinstance(reason, str) and reason:
        return f"停止原因：{reason}"
    return "停止原因未知"


def _status_payload(result: object, trace_path: Path) -> dict[str, Any]:
    return dict(_result_fields(result, trace_path))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
