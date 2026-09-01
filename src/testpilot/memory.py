"""Bounded repository memory values and deterministic local retrieval."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

MEMORY_SCHEMA_VERSION = 1
MAX_MEMORY_TEXT_CHARS = 800
MIN_MEMORY_KEYWORDS = 3
MAX_MEMORY_KEYWORDS = 12
MAX_MEMORY_KEYWORD_CHARS = 64
MAX_MEMORY_CHANGED_FILES = 50
MAX_RENDERED_MEMORY_CHARS = 6_000

_DRAFT_KEYS = frozenset(
    {"problem", "root_cause", "solution", "verification", "keywords"}
)
_ENTRY_KEYS = frozenset(
    {
        "schema_version",
        "memory_id",
        "created_at",
        "source_run_id",
        "problem",
        "root_cause",
        "solution",
        "verification",
        "keywords",
        "changed_files",
        "test_exit_code",
        "review_passed",
        "human_approved",
        "fingerprint",
    }
)
_MEMORY_ID_PATTERN = re.compile(r"mem_[0-9a-f]{16}\Z")
_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{16}\Z")
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_TEXT_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.\\/-]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
_CAMEL_TOKEN_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\Z)|[A-Z]?[a-z]+|\d+")


@dataclass(frozen=True)
class MemoryDraft:
    """One model-produced semantic repair summary with strict bounds."""

    problem: str
    root_cause: str
    solution: str
    verification: str
    keywords: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("problem", "root_cause", "solution", "verification"):
            value = _bounded_text(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)

        if not isinstance(self.keywords, tuple):
            raise TypeError("memory keywords must be a tuple")
        if not MIN_MEMORY_KEYWORDS <= len(self.keywords) <= MAX_MEMORY_KEYWORDS:
            raise ValueError("memory keyword count is invalid")
        normalized: list[str] = []
        seen: set[str] = set()
        for keyword in self.keywords:
            if not isinstance(keyword, str):
                raise TypeError("memory keywords must be strings")
            clean = keyword.strip()
            if not clean or len(clean) > MAX_MEMORY_KEYWORD_CHARS:
                raise ValueError("memory keyword length is invalid")
            key = clean.casefold()
            if key in seen:
                raise ValueError("memory keywords must be unique")
            seen.add(key)
            normalized.append(clean)
        object.__setattr__(self, "keywords", tuple(normalized))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MemoryDraft:
        """Build a draft from the exact JSON-facing field set."""
        if not isinstance(value, Mapping):
            raise TypeError("memory draft must be an object")
        if set(value) != _DRAFT_KEYS:
            raise ValueError("memory draft fields are invalid")
        raw_keywords = value["keywords"]
        if not isinstance(raw_keywords, list):
            raise TypeError("memory keywords must be a list")
        return cls(
            problem=value["problem"],  # type: ignore[arg-type]
            root_cause=value["root_cause"],  # type: ignore[arg-type]
            solution=value["solution"],  # type: ignore[arg-type]
            verification=value["verification"],  # type: ignore[arg-type]
            keywords=tuple(raw_keywords),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed JSON-native draft representation."""
        return {
            "problem": self.problem,
            "root_cause": self.root_cause,
            "solution": self.solution,
            "verification": self.verification,
            "keywords": list(self.keywords),
        }


@dataclass(frozen=True)
class MemoryEntry:
    """One validated durable memory plus host-controlled success evidence."""

    schema_version: int
    memory_id: str
    created_at: datetime
    source_run_id: str
    problem: str
    root_cause: str
    solution: str
    verification: str
    keywords: tuple[str, ...]
    changed_files: tuple[str, ...]
    test_exit_code: int
    review_passed: bool
    human_approved: bool
    fingerprint: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != MEMORY_SCHEMA_VERSION:
            raise ValueError("memory schema version is invalid")
        if not isinstance(self.memory_id, str) or not _MEMORY_ID_PATTERN.fullmatch(
            self.memory_id
        ):
            raise ValueError("memory id is invalid")
        if not isinstance(self.source_run_id, str) or not _RUN_ID_PATTERN.fullmatch(
            self.source_run_id
        ):
            raise ValueError("memory source run id is invalid")
        if not isinstance(self.created_at, datetime):
            raise TypeError("memory timestamp must be a datetime")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != UTC.utcoffset(
            self.created_at
        ):
            raise ValueError("memory timestamp must use UTC")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))

        draft = MemoryDraft(
            self.problem,
            self.root_cause,
            self.solution,
            self.verification,
            self.keywords,
        )
        object.__setattr__(self, "problem", draft.problem)
        object.__setattr__(self, "root_cause", draft.root_cause)
        object.__setattr__(self, "solution", draft.solution)
        object.__setattr__(self, "verification", draft.verification)
        object.__setattr__(self, "keywords", draft.keywords)

        if not isinstance(self.changed_files, tuple):
            raise TypeError("memory changed files must be a tuple")
        if not 1 <= len(self.changed_files) <= MAX_MEMORY_CHANGED_FILES:
            raise ValueError("memory changed file count is invalid")
        normalized_paths = tuple(_memory_path(path) for path in self.changed_files)
        if normalized_paths != tuple(sorted(set(normalized_paths))):
            raise ValueError("memory changed files must be sorted and unique")
        object.__setattr__(self, "changed_files", normalized_paths)

        if type(self.test_exit_code) is not int or self.test_exit_code != 0:
            raise ValueError("memory verification evidence is invalid")
        if type(self.review_passed) is not bool or self.review_passed is not True:
            raise ValueError("memory review evidence is invalid")
        if type(self.human_approved) is not bool or self.human_approved is not True:
            raise ValueError("memory approval evidence is invalid")
        if not isinstance(self.fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(
            self.fingerprint
        ):
            raise ValueError("memory fingerprint is invalid")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MemoryEntry:
        """Decode one exact JSON entry and reapply every invariant."""
        if not isinstance(value, Mapping):
            raise TypeError("memory entry must be an object")
        if set(value) != _ENTRY_KEYS:
            raise ValueError("memory entry fields are invalid")
        raw_keywords = value["keywords"]
        raw_paths = value["changed_files"]
        if not isinstance(raw_keywords, list) or not isinstance(raw_paths, list):
            raise TypeError("memory entry lists are invalid")
        timestamp = value["created_at"]
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise ValueError("memory timestamp is invalid")
        try:
            created_at = datetime.fromisoformat(f"{timestamp[:-1]}+00:00")
        except ValueError as exc:
            raise ValueError("memory timestamp is invalid") from exc
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            memory_id=value["memory_id"],  # type: ignore[arg-type]
            created_at=created_at,
            source_run_id=value["source_run_id"],  # type: ignore[arg-type]
            problem=value["problem"],  # type: ignore[arg-type]
            root_cause=value["root_cause"],  # type: ignore[arg-type]
            solution=value["solution"],  # type: ignore[arg-type]
            verification=value["verification"],  # type: ignore[arg-type]
            keywords=tuple(raw_keywords),
            changed_files=tuple(raw_paths),
            test_exit_code=value["test_exit_code"],  # type: ignore[arg-type]
            review_passed=value["review_passed"],  # type: ignore[arg-type]
            human_approved=value["human_approved"],  # type: ignore[arg-type]
            fingerprint=value["fingerprint"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-native record."""
        return {
            "schema_version": self.schema_version,
            "memory_id": self.memory_id,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "source_run_id": self.source_run_id,
            "problem": self.problem,
            "root_cause": self.root_cause,
            "solution": self.solution,
            "verification": self.verification,
            "keywords": list(self.keywords),
            "changed_files": list(self.changed_files),
            "test_exit_code": self.test_exit_code,
            "review_passed": self.review_passed,
            "human_approved": self.human_approved,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class MemoryMatch:
    """One memory paired with its deterministic local relevance score."""

    entry: MemoryEntry
    score: int

    def __post_init__(self) -> None:
        if not isinstance(self.entry, MemoryEntry):
            raise TypeError("memory match entry is invalid")
        if type(self.score) is not int or self.score < 1:
            raise ValueError("memory match score must be positive")


def retrieve_memories(
    task: str,
    entries: Sequence[MemoryEntry],
    *,
    limit: int = 3,
) -> tuple[MemoryMatch, ...]:
    """Return up to three positive deterministic keyword matches."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError("memory retrieval task must be non-blank")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("memory entries must be a sequence")
    if not all(isinstance(entry, MemoryEntry) for entry in entries):
        raise TypeError("memory entries must contain MemoryEntry values")
    if type(limit) is not int or not 1 <= limit <= 3:
        raise ValueError("memory retrieval limit must be between one and three")

    query = Counter(_tokens(task))
    matches: list[MemoryMatch] = []
    for entry in entries:
        score = _score(query, entry)
        if score > 0:
            matches.append(MemoryMatch(entry, score))
    matches.sort(
        key=lambda match: (
            -match.score,
            -match.entry.created_at.timestamp(),
            match.entry.memory_id,
        )
    )
    return tuple(matches[:limit])


def render_memory_block(
    matches: Sequence[MemoryMatch],
    *,
    max_chars: int = MAX_RENDERED_MEMORY_CHARS,
) -> str:
    """Render a bounded JSON list without cutting through JSON syntax."""
    if isinstance(matches, (str, bytes)) or not isinstance(matches, Sequence):
        raise TypeError("memory matches must be a sequence")
    if not all(isinstance(match, MemoryMatch) for match in matches):
        raise TypeError("memory matches contain an invalid value")
    if type(max_chars) is not int or max_chars < 2:
        raise ValueError("memory render limit must be at least two")

    payload: list[dict[str, Any]] = []
    for match in matches[:3]:
        candidate = {
            "memory_id": match.entry.memory_id,
            "problem": match.entry.problem,
            "root_cause": match.entry.root_cause,
            "solution": match.entry.solution,
            "verification": match.entry.verification,
            "keywords": list(match.entry.keywords),
        }
        rendered = _render_payload([*payload, candidate])
        if len(rendered) > max_chars:
            break
        payload.append(candidate)
    return _render_payload(payload)


def _bounded_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"memory {field_name} must be a string")
    clean = value.strip()
    if not clean or len(clean) > MAX_MEMORY_TEXT_CHARS:
        raise ValueError(f"memory {field_name} length is invalid")
    return clean


def _memory_path(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("memory changed file paths must be strings")
    if not value or "\\" in value or "\0" in value or ":" in value:
        raise ValueError("memory changed file path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError("memory changed file path is invalid")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError("memory changed file path is not canonical")
    return normalized


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value)
    tokens: set[str] = set()
    for raw in _TEXT_TOKEN_PATTERN.findall(normalized):
        if raw[0].isascii():
            lowered = raw.casefold()
            tokens.add(lowered)
            collapsed = re.sub(r"[^a-z0-9]", "", lowered)
            if len(collapsed) >= 2:
                tokens.add(collapsed)
            for part in re.split(r"[_.\\/-]+", raw):
                if not part:
                    continue
                folded_part = part.casefold()
                if len(folded_part) >= 2:
                    tokens.add(folded_part)
                for camel in _CAMEL_TOKEN_PATTERN.findall(part):
                    folded_camel = camel.casefold()
                    if len(folded_camel) >= 2:
                        tokens.add(folded_camel)
        else:
            tokens.add(raw)
            if len(raw) >= 2:
                tokens.update(raw[index : index + 2] for index in range(len(raw) - 1))
    return tuple(sorted(tokens))


def _score(query: Counter[str], entry: MemoryEntry) -> int:
    fields = (
        (5, " ".join(entry.keywords)),
        (3, entry.problem),
        (3, entry.root_cause),
        (1, entry.solution),
        (1, " ".join(entry.changed_files)),
    )
    score = 0
    query_tokens = set(query)
    for weight, value in fields:
        score += weight * len(query_tokens.intersection(_tokens(value)))
    return score


def _render_payload(payload: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
