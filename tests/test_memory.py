"""Transactional, scoped memory tests; no model, simulator or credential access."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from go2_demo.memory import MemoryLimits, MemoryScope, MemoryStore, MemoryStoreError
from go2_demo.provenance import digest
from go2_demo.robots.registry import get_profile


@pytest.fixture
def scope():
    return MemoryScope.from_profile(get_profile())


@pytest.fixture
def store(tmp_path, scope):
    value = MemoryStore(tmp_path / "memory.sqlite3", scope=scope, mode="read_write", episode_id="episode-1")
    yield value
    value.close()


def evidence(store, ref="episode-1/turn-1", *, delta=.31):
    result = store.register_public_evidence(ref, {
        "observation": {"image_id": "frame-1", "local_odometry": {"translation_m": [0., 0.] }},
        "calls": [{"call_id": "move-1", "name": "move_relative", "arguments": {"forward_m": .4, "lateral_m": 0.}}],
        "results": [{"status": "succeeded", "measured": {"translation_start_frame_m": [delta, 0.]}}],
    })
    assert result["status"] == "succeeded"
    return ref


def save(store, text="本回合请求0.4米，实测前进0.31米；应继续检查测量。", call_id="save-1"):
    return store.save(text, [evidence(store)], call_id=call_id)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scope_uses_same_profile_hash_as_runtime_provenance(scope):
    assert scope.profile_sha256 == digest(get_profile().to_dict())
    assert scope.to_dict() == {"robot_id": "go2", "controller_id": get_profile().controller_id,
                               "profile_sha256": digest(get_profile().to_dict())}
    with pytest.raises(FrozenInstanceError):
        scope.robot_id = "other"


@pytest.mark.parametrize("change", [{"robot_id": ""}, {"controller_id": "\n"},
                                   {"profile_sha256": "not-a-hash"}, {"profile_sha256": "A" * 64}])
def test_invalid_scope_rejected(scope, change):
    with pytest.raises(ValueError):
        replace(scope, **change)


def test_save_reopen_read_and_literal_search_preserve_evidence(tmp_path, scope):
    path = tmp_path / "memory.sqlite3"
    original = MemoryStore(path, scope=scope, mode="read_write", episode_id="first")
    result = save(original, "Observed: STOP drift needs measurement. 不要累加请求距离。")
    memory_id = result["memory_id"]
    snapshot = original.snapshot()
    original.close()
    reopened = MemoryStore(path, scope=scope, mode="read_write", episode_id="second")
    assert reopened.snapshot() == snapshot
    assert reopened.search("stop")["matches"][0]["memory_id"] == memory_id
    assert reopened.search("不要累加")["returned"] == 1
    assert reopened.search(".*")["matches"] == []  # literal, not regex or SQL wildcard
    item = reopened.read(memory_id)
    assert item["memory"]["version"] == item["current_version"] == 1
    assert item["available_versions"] == [1]
    assert item["evidence"][0]["public_record"]["results"][0]["measured"]["translation_start_frame_m"] == [.31, 0.]
    assert item["evidence"][0]["sha256"] == digest(item["evidence"][0]["public_record"])
    assert reopened.info()["record_count"] == reopened.info()["version_count"] == 1
    reopened.close()


@pytest.mark.parametrize("dimension", ["robot_id", "controller_id", "profile_sha256"])
def test_no_cross_robot_controller_or_profile_access(tmp_path, scope, dimension):
    path = tmp_path / "memory.sqlite3"
    source = MemoryStore(path, scope=scope, mode="read_write", episode_id="one")
    result = save(source)
    source.close()
    replacement = "b" * 64 if dimension == "profile_sha256" else "other"
    isolated = MemoryStore(path, scope=replace(scope, **{dimension: replacement}), mode="read_write", episode_id="one")
    assert isolated.search("")["matches"] == []
    assert isolated.read(result["memory_id"])["error_code"] == "MEMORY_NOT_FOUND"
    assert isolated.save("Cannot borrow another scope's evidence", ["episode-1/turn-1"], call_id="save-new")["error_code"] == "UNKNOWN_EVIDENCE"
    assert isolated.info()["record_count"] == 0
    assert isolated.snapshot()["content"]["records"] == []
    isolated.close()


def test_revision_is_cas_append_only_and_old_versions_remain_readable(store):
    first = save(store)
    ref = evidence(store, "episode-1/turn-2", delta=.4)
    revised = store.revise(first["memory_id"], 1, "后续一次实测达到0.4米，前次偏差不能概括为固定比例。",
                           "补充反例", [ref], call_id="revise-1")
    assert revised["status"] == "succeeded" and revised["version"] == 2
    current, old = store.read(first["memory_id"]), store.read(first["memory_id"], 1)
    assert current["memory"]["reason"] == "补充反例"
    assert current["available_versions"] == [1, 2]
    assert old["memory"]["text"] != current["memory"]["text"]
    assert old["memory"]["version"] == 1 and old["current_version"] == 2
    conflict = store.revise(first["memory_id"], 1, "stale rewrite", "test", [ref], call_id="revise-stale")
    assert conflict["error_code"] == "VERSION_CONFLICT" and conflict["current_version"] == 2
    assert store.info()["record_count"] == 1 and store.info()["version_count"] == 2
    assert store.read(first["memory_id"], 3)["error_code"] == "MEMORY_NOT_FOUND"


def test_revision_cannot_introduce_unregistered_evidence(store):
    first = save(store)
    before = store.snapshot()
    result = store.revise(first["memory_id"], 1, "Unsupported claim", "no source", ["runs/raw/metrics.json"], call_id="invalid-revision")
    assert result["error_code"] == "UNKNOWN_EVIDENCE"
    assert store.snapshot() == before
    assert store.read(first["memory_id"])["current_version"] == 1


def test_writes_are_durable_idempotent_and_conflicting_call_ids_are_rejected(tmp_path, scope):
    path = tmp_path / "memory.sqlite3"
    first = MemoryStore(path, scope=scope, mode="read_write", episode_id="episode-1")
    result = save(first)
    snapshot = first.snapshot()
    first.close()
    reopened = MemoryStore(path, scope=scope, mode="read_write", episode_id="episode-1")
    repeated = save(reopened)
    assert repeated == {**result, "deduplicated": True}
    assert reopened.info()["record_count"] == reopened.info()["version_count"] == 1
    assert reopened.snapshot() == snapshot
    assert save(reopened, "different text")["error_code"] == "CALL_ID_REUSED"
    assert reopened.update_working_note("different operation", call_id="save-1")["error_code"] == "CALL_ID_REUSED"
    revision = reopened.revise(result["memory_id"], 1, "updated", "correction", ["episode-1/turn-1"], call_id="rev-1")
    assert reopened.revise(result["memory_id"], 1, "updated", "correction", ["episode-1/turn-1"], call_id="rev-1") == {**revision, "deduplicated": True}
    reopened.close()


def test_failed_write_receipt_does_not_silently_succeed_after_evidence_arrives(store):
    failure = store.save("claim", ["future"], call_id="same")
    assert failure["error_code"] == "UNKNOWN_EVIDENCE"
    evidence(store, "future")
    assert store.save("claim", ["future"], call_id="same") == {**failure, "deduplicated": True}
    assert store.info()["record_count"] == 0
    assert store.save("claim", ["future"], call_id="new-call")["status"] == "succeeded"


def test_call_id_can_be_reused_in_a_distinct_episode(tmp_path, scope):
    path = tmp_path / "memory.sqlite3"
    first = MemoryStore(path, scope=scope, mode="read_write", episode_id="episode-1")
    a = save(first)
    first.close()
    second = MemoryStore(path, scope=scope, mode="read_write", episode_id="episode-2")
    b = save(second)
    assert a["memory_id"] != b["memory_id"]
    assert second.info()["record_count"] == 2
    second.close()


def test_public_evidence_is_immutable_and_detached_from_callers(store):
    payload = {"observation": {"image_id": "frame-1"}, "results": [{"status": "succeeded"}]}
    registered = store.register_public_evidence("ref", payload)
    payload["observation"]["image_id"] = "modified"
    assert store.register_public_evidence("ref", payload)["error_code"] == "EVIDENCE_CONFLICT"
    result = store.save("claim", ["ref"], call_id="save")
    read = store.read(result["memory_id"])
    assert read["evidence"][0]["sha256"] == registered["sha256"]
    assert read["evidence"][0]["public_record"]["observation"]["image_id"] == "frame-1"
    read["evidence"][0]["public_record"]["observation"]["image_id"] = "edited"
    assert store.read(result["memory_id"])["evidence"][0]["sha256"] == registered["sha256"]


@pytest.mark.parametrize("private", ["reasoning_content", "metrics", "evaluation", "trajectory", "agent_context", "target_xy_m", "camera_to_world"])
def test_private_or_memory_derived_objects_are_not_registered(store, private):
    assert store.register_public_evidence("bad", {"nested": [{private: "not public"}]})["error_code"] == "INVALID_ARGUMENTS"
    assert store.save("claim", ["bad"], call_id="save")["error_code"] == "UNKNOWN_EVIDENCE"


def test_evidence_reference_that_looks_like_a_path_is_never_dereferenced(store, tmp_path):
    source = tmp_path / "private.json"
    source.write_text('{"private":"not public evidence"}')
    result = store.save("claim", [str(source)], call_id="save")
    assert result["error_code"] == "UNKNOWN_EVIDENCE"


def test_off_mode_does_not_touch_database_or_create_directories(tmp_path, scope, monkeypatch):
    path = tmp_path / "absent" / "memory.sqlite3"
    def forbidden(*args, **kwargs):
        pytest.fail("Off mode must never connect to SQLite")
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    off = MemoryStore(path, scope=scope, mode="off", episode_id="off")
    assert off.info()["record_count"] == off.info()["version_count"] == 0
    assert off.snapshot()["content"]["records"] == []
    for result in (off.search(""), off.read("memory-id"), off.save("claim", ["ref"], call_id="one"),
                   off.revise("memory-id", 1, "claim", "reason", ["ref"], call_id="two"),
                   off.update_working_note("note", call_id="three"), off.register_public_evidence("ref", {})):
        assert result["error_code"] == "MEMORY_DISABLED"
    off.close()
    assert not path.parent.exists()


def test_read_only_keeps_db_bytes_and_snapshot_unchanged_but_allows_ram_note(tmp_path, scope):
    path = tmp_path / "memory.sqlite3"
    writable = MemoryStore(path, scope=scope, mode="read_write", episode_id="first")
    first = save(writable)
    initial = writable.snapshot()
    writable.close()
    before = file_hash(path)
    readonly = MemoryStore(path, scope=scope, mode="read_only", episode_id="readonly")
    assert readonly.search("")["returned"] == 1
    assert readonly.read(first["memory_id"])["status"] == "succeeded"
    assert readonly.register_public_evidence("fresh", {"observation": {"image_id": "new"}})["status"] == "succeeded"
    assert readonly.save("new", ["fresh"], call_id="save")["error_code"] == "MEMORY_READ_ONLY"
    assert readonly.revise(first["memory_id"], 1, "new", "reason", ["fresh"], call_id="rev")["error_code"] == "MEMORY_READ_ONLY"
    note = readonly.update_working_note("Check measured progress next.", call_id="note")
    assert note["status"] == "succeeded" and readonly.info()["working_note"] == note["working_note"]
    assert readonly.update_working_note("Check measured progress next.", call_id="note")["deduplicated"] is True
    assert readonly.info()["working_note_version"] == 1
    assert readonly.snapshot() == initial
    readonly.close()
    assert file_hash(path) == before
    reopened = MemoryStore(path, scope=scope, mode="read_only", episode_id="next")
    assert reopened.info()["working_note"] == "" and reopened.info()["working_note_version"] == 0
    reopened.close()


def test_read_only_missing_database_fails_without_creating_it(tmp_path, scope):
    path = tmp_path / "absent" / "memory.sqlite3"
    with pytest.raises(MemoryStoreError) as failure:
        MemoryStore(path, scope=scope, mode="read_only", episode_id="read")
    assert failure.value.code == "MEMORY_UNAVAILABLE"
    assert not path.parent.exists()


def test_snapshot_is_reproducible_and_ignores_unreferenced_evidence_and_ram(store):
    empty = store.snapshot()
    ref = evidence(store)
    assert store.snapshot() == empty
    store.update_working_note("temporary", call_id="note")
    assert store.snapshot() == empty
    result = store.save("measured evidence", [ref], call_id="save")
    snapshot = store.snapshot()
    assert snapshot != empty
    assert digest(snapshot["content"]) == snapshot["sha256"]
    assert snapshot["content"]["records"][0]["memory_id"] == result["memory_id"]
    assert [e["evidence_ref"] for e in snapshot["content"]["evidence"]] == [ref]
    evidence(store, "not-used")
    store.search("")
    store.read(result["memory_id"])
    assert store.snapshot() == snapshot
    assert "working_note" not in json.dumps(snapshot)


@pytest.mark.parametrize(("action", "arguments"), [
    ("save", {"text": "", "evidence_refs": ["ref"], "call_id": "s"}),
    ("save", {"text": "claim", "evidence_refs": [], "call_id": "s"}),
    ("save", {"text": "claim", "evidence_refs": ["ref", "ref"], "call_id": "s"}),
    ("save", {"text": "claim", "evidence_refs": "ref", "call_id": "s"}),
    ("read", {"memory_id": "memory", "version": True}),
    ("read", {"memory_id": "memory", "version": 0}),
    ("search", {"query": "claim", "limit": True}),
    ("search", {"query": "claim", "limit": 0}),
    ("search", {"query": "claim", "limit": 11}),
    ("revise", {"memory_id": "memory", "expected_version": True, "text": "claim", "reason": "reason", "evidence_refs": ["ref"], "call_id": "s"}),
])
def test_invalid_arguments_are_rejected_without_memory_changes(store, action, arguments):
    before = store.snapshot()
    assert getattr(store, action)(**arguments)["error_code"] == "INVALID_ARGUMENTS"
    assert store.snapshot() == before


def test_text_evidence_search_and_record_budgets(tmp_path, scope):
    limits = replace(MemoryLimits(), max_text_chars=5, max_note_chars=5, max_query_chars=5,
                     max_reason_chars=5, max_public_evidence_bytes=128, max_records=1,
                     max_versions=1, max_public_evidence_records=1)
    store = MemoryStore(tmp_path / "memory.sqlite3", scope=scope, mode="read_write", episode_id="limited", limits=limits)
    assert store.register_public_evidence("large", {"public": "x" * 200})["error_code"] == "MEMORY_LIMIT"
    assert store.register_public_evidence("ref", {"public": "ok"})["status"] == "succeeded"
    assert store.register_public_evidence("ref2", {"public": "ok"})["error_code"] == "MEMORY_LIMIT"
    assert store.save("x" * 6, ["ref"], call_id="long")["error_code"] == "INVALID_ARGUMENTS"
    assert store.update_working_note("x" * 6, call_id="long-note")["error_code"] == "INVALID_ARGUMENTS"
    assert store.search("x" * 6)["error_code"] == "INVALID_ARGUMENTS"
    first = store.save("short", ["ref"], call_id="first")
    assert first["status"] == "succeeded"
    assert store.save("short", ["ref"], call_id="second")["error_code"] == "MEMORY_LIMIT"
    assert store.revise(first["memory_id"], 1, "new", "cause", ["ref"], call_id="rev")["error_code"] == "MEMORY_LIMIT"
    store.close()


def test_write_rejects_evidence_combo_that_cannot_be_read(tmp_path, scope):
    store = MemoryStore(tmp_path / "memory.sqlite3", scope=scope, mode="read_write", episode_id="large",
                        limits=replace(MemoryLimits(), max_result_chars=1500))
    assert store.register_public_evidence("large", {"public": "x" * 2000})["status"] == "succeeded"
    before = store.snapshot()
    failure = store.save("short", ["large"], call_id="too-large")
    assert failure["error_code"] == "MEMORY_RESULT_LIMIT"
    assert store.info()["record_count"] == store.info()["version_count"] == 0
    assert store.snapshot() == before
    store.close()


def test_concurrent_revisions_from_same_version_commit_exactly_once(tmp_path, scope):
    path = tmp_path / "memory.sqlite3"
    first = MemoryStore(path, scope=scope, mode="read_write", episode_id="first")
    saved = save(first)
    other = MemoryStore(path, scope=scope, mode="read_write", episode_id="other")
    def revise(store, call_id):
        return store.revise(saved["memory_id"], 1, call_id, "correction", ["episode-1/turn-1"], call_id=call_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: revise(*item), [(first, "revision-a"), (other, "revision-b")]))
    assert sum(r["status"] == "succeeded" for r in results) == 1
    assert sum(r.get("error_code") == "VERSION_CONFLICT" for r in results) == 1
    assert first.info()["version_count"] == 2
    first.close()
    other.close()


def test_locked_database_reports_no_committed_write(store):
    ref = evidence(store)
    before = store.snapshot()
    blocker = sqlite3.connect(store._path)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        failure = store.save("claim", [ref], call_id="locked")
        assert failure["error_code"] == "MEMORY_LOCKED"
    finally:
        blocker.rollback()
        blocker.close()
    assert store.snapshot() == before and store.info()["version_count"] == 0
    assert store.save("claim", [ref], call_id="locked")["status"] == "succeeded"


def test_corrupt_database_and_unknown_schema_fail_fast(tmp_path, scope):
    path = tmp_path / "broken.sqlite3"
    path.write_text("not a SQLite database")
    with pytest.raises(MemoryStoreError):
        MemoryStore(path, scope=scope, mode="read_write", episode_id="broken")
    path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE schema_meta (version INTEGER)")
    connection.execute("INSERT INTO schema_meta VALUES (999)")
    connection.commit()
    connection.close()
    with pytest.raises(MemoryStoreError, match="Unsupported"):
        MemoryStore(path, scope=scope, mode="read_only", episode_id="future")


def test_closed_store_does_not_accept_operations(store):
    store.close()
    assert store.search("")["error_code"] == "MEMORY_CLOSED"
    assert store.update_working_note("note", call_id="closed")["error_code"] == "MEMORY_CLOSED"
    with pytest.raises(MemoryStoreError):
        store.snapshot()
