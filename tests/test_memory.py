from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import testpilot.memory as memory_module
from testpilot.memory import (
    MemoryDraft,
    MemoryEntry,
    MemoryError,
    MemoryStore,
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


def _draft(
    *,
    problem: str = "path bug",
    root_cause: str = "raw separator",
) -> MemoryDraft:
    return MemoryDraft(
        problem=problem,
        root_cause=root_cause,
        solution="normalize at the boundary",
        verification="fixed pytest passed",
        keywords=("path", "pytest", "windows"),
    )


def _save(store: MemoryStore, draft: MemoryDraft | None = None, **overrides: object):
    arguments: dict[str, object] = {
        "source_run_id": "0123456789abcdef",
        "changed_files": ("src/path.py",),
        "test_exit_code": 0,
        "review_passed": True,
        "human_approved": True,
    }
    arguments.update(overrides)
    return store.save(draft or _draft(), **arguments)  # type: ignore[arg-type]


def test_memory_store_missing_file_is_an_empty_library(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)

    assert store.load() == ()
    assert store.retrieve("path failure") == ()
    assert not store.path.exists()


def test_memory_store_round_trip_and_duplicate_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = iter(("0000000000000001", "0000000000000002"))
    monkeypatch.setattr(memory_module.secrets, "token_hex", lambda _: next(tokens))
    store = MemoryStore(tmp_path)

    first = _save(store)
    second = _save(store, source_run_id="fedcba9876543210")

    assert first.status == "saved"
    assert first.memory_id == "mem_0000000000000001"
    assert first.entry_count == 1
    assert first.pruned is False
    assert second.status == "duplicate"
    assert second.memory_id == first.memory_id
    assert len(store.load()) == 1
    assert store.retrieve("windows path pytest")[0].entry.memory_id == first.memory_id
    raw = store.path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert len(raw.splitlines()) == 1


def test_memory_store_redacts_environment_tokens_assignments_and_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMORY_TEST_TOKEN", "environment-secret-value")
    monkeypatch.setattr(memory_module.secrets, "token_hex", lambda _: "0000000000000001")
    store = MemoryStore(tmp_path)
    draft = MemoryDraft(
        problem="sk-secret1234567890 failed",
        root_cause="token=environment-secret-value",
        solution="replace https://person:password@example.invalid safely",
        verification="pytest passed with password=hunter2-value",
        keywords=("path", "pytest", "windows"),
    )

    _save(store, draft)

    contents = store.path.read_text(encoding="utf-8")
    for secret in (
        "sk-secret1234567890",
        "environment-secret-value",
        "password@example.invalid",
        "hunter2-value",
    ):
        assert secret not in contents
    assert "[REDACTED]" in contents


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"source_run_id": "bad"}, "memory_invalid"),
        ({"changed_files": ("../outside.py",)}, "memory_invalid"),
        ({"test_exit_code": 1}, "memory_invalid"),
        ({"test_exit_code": False}, "memory_invalid"),
        ({"review_passed": False}, "memory_invalid"),
        ({"human_approved": False}, "memory_invalid"),
    ],
)
def test_memory_store_requires_real_success_evidence(
    tmp_path: Path, override: dict[str, object], code: str
) -> None:
    with pytest.raises(MemoryError) as caught:
        _save(MemoryStore(tmp_path), **override)

    assert caught.value.code == code
    assert "path bug" not in str(caught.value)


@pytest.mark.parametrize(
    "contents",
    [
        "not-json\n",
        json.dumps({**_entry("mem_0000000000000001").to_dict(), "extra": True}) + "\n",
        "\n",
    ],
)
def test_memory_store_rejects_invalid_jsonl(tmp_path: Path, contents: str) -> None:
    path = tmp_path / ".testpilot" / "memories" / "entries.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(MemoryError) as caught:
        MemoryStore(tmp_path).load()

    assert caught.value.code == "memory_invalid"
    if contents.strip():
        assert contents.strip() not in str(caught.value)


def test_memory_store_rejects_oversized_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / ".testpilot" / "memories" / "entries.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * 2_000_001)

    with pytest.raises(MemoryError) as file_error:
        MemoryStore(tmp_path).load()
    assert file_error.value.code == "memory_too_large"

    path.write_bytes(b"x" * 8_193 + b"\n")
    with pytest.raises(MemoryError) as line_error:
        MemoryStore(tmp_path).load()
    assert line_error.value.code == "memory_too_large"


def test_memory_store_rejects_a_symlinked_memory_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text("private", encoding="utf-8")
    path = tmp_path / ".testpilot" / "memories" / "entries.jsonl"
    path.parent.mkdir(parents=True)
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(MemoryError) as caught:
        MemoryStore(tmp_path).load()

    assert caught.value.code == "memory_load_failed"
    assert "private" not in str(caught.value)


def test_memory_store_atomic_failure_preserves_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = iter(("0000000000000001", "0000000000000002"))
    monkeypatch.setattr(memory_module.secrets, "token_hex", lambda _: next(tokens))
    store = MemoryStore(tmp_path)
    _save(store)
    original = store.path.read_bytes()

    def fail_replace(source: object, target: object) -> None:
        del source, target
        raise OSError("private write detail")

    monkeypatch.setattr(memory_module.os, "replace", fail_replace)
    with pytest.raises(MemoryError) as caught:
        _save(store, _draft(problem="different bug"))

    assert caught.value.code == "memory_save_failed"
    assert "private" not in str(caught.value)
    assert store.path.read_bytes() == original
    assert not tuple(store.path.parent.glob(".entries-*.tmp"))


def test_memory_store_prunes_the_oldest_of_more_than_two_hundred_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".testpilot" / "memories" / "entries.jsonl"
    path.parent.mkdir(parents=True)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    entries = [
        _entry(
            f"mem_{index:016x}",
            problem=f"problem {index}",
            created_at=start + timedelta(seconds=index),
            fingerprint=f"{index + 1:064x}",
        )
        for index in range(200)
    ]
    path.write_text(
        "".join(
            json.dumps(entry.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
            for entry in entries
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(memory_module.secrets, "token_hex", lambda _: "ffffffffffffffff")

    result = _save(MemoryStore(tmp_path), _draft(problem="new unique issue"))
    loaded = MemoryStore(tmp_path).load()

    assert result.status == "saved"
    assert result.pruned is True
    assert result.entry_count == 200
    assert len(loaded) == 200
    assert entries[0].memory_id not in {entry.memory_id for entry in loaded}
    assert "mem_ffffffffffffffff" in {entry.memory_id for entry in loaded}
