"""Append-only JSONL run traces that deliberately exclude environment values."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

_ENVIRONMENT_KEYS = frozenset({"env", "environ", "environment"})
_SENSITIVE_KEY_PARTS = ("API_KEY", "_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


class JsonlTrace:
    """Serialize JSON-native events one per line, safely across threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def record(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        """Append one event, refusing non-JSON or environment-bearing payloads."""
        if not isinstance(event, str) or not event:
            raise ValueError("event must be a non-empty string")
        normalized_payload = _json_native(payload or {})
        record = {
            "event": _redact_sensitive_values(event),
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": normalized_payload,
        }
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")


def _json_native(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_sensitive_key(key):
        raise ValueError("environment or credential values must not be recorded")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _redact_sensitive_values(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload must be JSON-native")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                raise TypeError("payload must use string keys")
            copied[child_key] = _json_native(child_value, key=child_key)
        return copied
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    raise ValueError(f"payload must be JSON-native, got {type(value).__name__}")


def _is_sensitive_key(key: str) -> bool:
    normalized = key.upper()
    return key.lower() in _ENVIRONMENT_KEYS or any(
        part in normalized for part in _SENSITIVE_KEY_PARTS
    )


def _redact_sensitive_values(value: str) -> str:
    redacted = value
    for secret in _sensitive_environment_values():
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _sensitive_environment_values() -> tuple[str, ...]:
    values = {
        value
        for key, value in os.environ.items()
        if value and any(part in key.upper() for part in _SENSITIVE_KEY_PARTS)
    }
    return tuple(sorted(values, key=len, reverse=True))
