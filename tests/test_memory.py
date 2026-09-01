from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from testpilot.memory import (
    MemoryDraft,
    MemoryEntry,
    render_memory_block,
    retrieve_memories,
)


def _entry(
    memory_id: str,
    *,
    keywords: tuple[str, ...] = ("path", "pytest", "windows"),
    problem: str = "Windows 路径错误",
    root_cause: str = "分隔符没有规范化",
    solution: str = "normalize_path converts separators",
    changed_files: tuple[str, ...] = ("src/path_utils.py",),
    created_at: datetime | None = None,
    fingerprint: str = "a" * 64,
) -> MemoryEntry:
    return MemoryEntry(
        schema_version=1,
        memory_id=memory_id,
        created_at=created_at or datetime(2026, 9, 2, tzinfo=UTC),
        source_run_id="0123456789abcdef",
        problem=problem,
        root_cause=root_cause,
        solution=solution,
        verification="fixed pytest verifier passed",
        keywords=keywords,
        changed_files=changed_files,
        test_exit_code=0,
        review_passed=True,
        human_approved=True,
        fingerprint=fingerprint,
    )


def test_memory_draft_round_trip_uses_only_fixed_fields() -> None:
    draft = MemoryDraft(
        problem="path bug",
        root_cause="raw separator",
        solution="normalize once",
        verification="pytest passed",
        keywords=("path", "pytest", "windows"),
    )

    restored = MemoryDraft.from_mapping(draft.to_dict())

    assert restored == draft
    with pytest.raises(ValueError, match="fields"):
        MemoryDraft.from_mapping({**draft.to_dict(), "extra": "not allowed"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("problem", ""),
        ("root_cause", " "),
        ("solution", "x" * 801),
        ("verification", 3),
    ],
)
def test_memory_draft_rejects_invalid_summary_fields(field: str, value: object) -> None:
    values: dict[str, object] = {
        "problem": "problem",
        "root_cause": "cause",
        "solution": "solution",
        "verification": "verification",
        "keywords": ["path", "pytest", "windows"],
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError)):
        MemoryDraft.from_mapping(values)


@pytest.mark.parametrize(
    "keywords",
    [
        ["one", "two"],
        [str(index) for index in range(13)],
        ["pytest", "pytest", "path"],
        ["one", "two", "x" * 65],
        ["one", "two", " "],
        "one,two,three",
    ],
)
def test_memory_draft_rejects_invalid_keywords(keywords: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryDraft.from_mapping(
            {
                "problem": "problem",
                "root_cause": "cause",
                "solution": "solution",
                "verification": "verification",
                "keywords": keywords,
            }
        )


def test_memory_entry_round_trip_preserves_utc_and_tuples() -> None:
    entry = _entry("mem_0000000000000001")

    encoded = entry.to_dict()
    restored = MemoryEntry.from_mapping(encoded)

    assert restored == entry
    assert encoded["created_at"] == "2026-09-02T00:00:00Z"
    assert encoded["keywords"] == ["path", "pytest", "windows"]


@pytest.mark.parametrize(
    ("override", "error_type"),
    [
        ({"schema_version": True}, ValueError),
        ({"memory_id": "bad"}, ValueError),
        ({"source_run_id": "bad"}, ValueError),
        ({"created_at": datetime(2026, 9, 2, tzinfo=UTC).replace(tzinfo=None)}, ValueError),
        ({"changed_files": ("../outside.py",)}, ValueError),
        ({"changed_files": ("C:/outside.py",)}, ValueError),
        ({"test_exit_code": False}, ValueError),
        ({"test_exit_code": 1}, ValueError),
        ({"review_passed": 1}, ValueError),
        ({"review_passed": False}, ValueError),
        ({"human_approved": False}, ValueError),
        ({"fingerprint": "not-sha256"}, ValueError),
    ],
)
def test_memory_entry_rejects_invalid_host_evidence(
    override: dict[str, object], error_type: type[Exception]
) -> None:
    values = {
        "schema_version": 1,
        "memory_id": "mem_0000000000000001",
        "created_at": datetime(2026, 9, 2, tzinfo=UTC),
        "source_run_id": "0123456789abcdef",
        "problem": "problem",
        "root_cause": "cause",
        "solution": "solution",
        "verification": "pytest",
        "keywords": ("path", "pytest", "windows"),
        "changed_files": ("src/path.py",),
        "test_exit_code": 0,
        "review_passed": True,
        "human_approved": True,
        "fingerprint": "a" * 64,
    }
    values.update(override)

    with pytest.raises(error_type):
        MemoryEntry(**values)  # type: ignore[arg-type]


def test_retrieve_memories_weights_keywords_and_limits_to_three() -> None:
    entries = (
        _entry("mem_0000000000000001", keywords=("windows", "pytest", "path")),
        _entry(
            "mem_0000000000000002",
            keywords=("pytest", "python", "testing"),
            problem="path handling failed",
            fingerprint="b" * 64,
        ),
        _entry(
            "mem_0000000000000003",
            keywords=("path", "python", "bug"),
            fingerprint="c" * 64,
        ),
        _entry(
            "mem_0000000000000004",
            keywords=("network", "http", "retry"),
            problem="request timeout",
            root_cause="remote server",
            solution="bounded retry",
            changed_files=("src/http.py",),
            fingerprint="d" * 64,
        ),
    )

    matches = retrieve_memories("修复 Windows pytest 路径 path 错误", entries)

    assert len(matches) == 3
    assert matches[0].entry.memory_id == "mem_0000000000000001"
    assert all(match.score > 0 for match in matches)
    assert "mem_0000000000000004" not in {match.entry.memory_id for match in matches}


def test_retrieve_memories_supports_chinese_bigrams_and_code_identifiers() -> None:
    chinese = _entry(
        "mem_0000000000000001",
        keywords=("路径处理", "分隔符", "测试"),
        fingerprint="a" * 64,
    )
    identifier = _entry(
        "mem_0000000000000002",
        keywords=("normalize_path", "camelcase", "python"),
        problem="normalizePath failed",
        fingerprint="b" * 64,
    )

    chinese_matches = retrieve_memories("修复路径处理逻辑", (chinese, identifier))
    identifier_matches = retrieve_memories("normalizePath in path_utils.py", (chinese, identifier))

    assert chinese_matches[0].entry.memory_id == chinese.memory_id
    assert identifier_matches[0].entry.memory_id == identifier.memory_id


def test_retrieve_memories_uses_newest_then_id_for_ties() -> None:
    older = _entry(
        "mem_0000000000000002",
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        fingerprint="a" * 64,
    )
    newer_high_id = _entry(
        "mem_0000000000000003",
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
        fingerprint="b" * 64,
    )
    newer_low_id = _entry(
        "mem_0000000000000001",
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
        fingerprint="c" * 64,
    )

    matches = retrieve_memories("path pytest windows", (older, newer_high_id, newer_low_id))

    assert [match.entry.memory_id for match in matches] == [
        "mem_0000000000000001",
        "mem_0000000000000003",
        "mem_0000000000000002",
    ]


def test_retrieve_memories_rejects_bad_limits_and_returns_no_zero_scores() -> None:
    unrelated = _entry(
        "mem_0000000000000001",
        keywords=("network", "http", "retry"),
        problem="remote timeout",
        root_cause="server busy",
        solution="backoff",
        changed_files=("src/http.py",),
    )

    assert retrieve_memories("database schema", (unrelated,)) == ()
    with pytest.raises(ValueError, match="limit"):
        retrieve_memories("path", (unrelated,), limit=0)
    with pytest.raises(ValueError, match="task"):
        retrieve_memories(" ", (unrelated,))


def test_render_memory_block_is_json_and_respects_total_limit() -> None:
    entries = tuple(
        _entry(
            f"mem_{index:016x}",
            problem=f"problem-{index}-" + "x" * 300,
            fingerprint=f"{index + 1:064x}",
            created_at=datetime(2026, 9, 2, tzinfo=UTC) + timedelta(seconds=index),
        )
        for index in range(4)
    )
    matches = retrieve_memories("path pytest windows", entries)

    rendered = render_memory_block(matches, max_chars=1_600)
    decoded = json.loads(rendered)

    assert len(rendered) <= 1_600
    assert len(decoded) <= 3
    assert set(decoded[0]) == {
        "keywords",
        "memory_id",
        "problem",
        "root_cause",
        "solution",
        "verification",
    }
