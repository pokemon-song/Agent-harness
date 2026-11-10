"""Storage invariants for asynchronous Markdown memory; no model or simulator."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from go2_demo.md_memory import store as module
from go2_demo.md_memory.store import MDMemoryStore, StoreError


def change(event, body="A ball was observed near N3.", path="objects/ball.md", status="confirmed"):
    return {"path": path, "body": body, "evidence_ids": [event["id"]], "status": status}


def assert_error(code, callback):
    with pytest.raises(StoreError) as caught:
        callback()
    assert caught.value.code == code


def test_round_trip_versions_frontmatter_and_reopen(tmp_path):
    store = MDMemoryStore(tmp_path)
    first = store.append_event("observation", {"object": "ball", "node": "N3"}, critical=True)
    assert store.snapshot()["pending_critical_events"][0]["id"] == first["id"]
    store.commit(0, first["seq"], [change(first)], call_id="commit-1")
    second = store.append_event("observation", {"object": "ball", "node": "N5"})
    store.commit(1, second["seq"], [change(second, "Ball has moved to N5.")], call_id="commit-2")
    old = store.read("objects/ball.md", revision=1)
    new = store.read("objects/ball.md")
    assert old["body"] == "A ball was observed near N3."
    assert new["body"] == "Ball has moved to N5."
    assert new["revision"] == 2 and old["revision"] == 1
    assert new["content"].startswith('---\npath: "objects/ball.md"')
    assert new["evidence_ids"] == [second["id"]]
    reopened = MDMemoryStore(tmp_path)
    assert reopened.snapshot()["processed_event_seq"] == second["seq"]
    assert reopened.snapshot()["pending_critical_events"] == []
    assert reopened.pending_events() == []
    assert reopened.get_event(first["id"]) == first


def test_empty_changes_advance_watermark_without_changing_document_revision(tmp_path):
    store = MDMemoryStore(tmp_path)
    event = store.append_event("observation", {})
    store.commit(0, 1, [change(event)], call_id="first")
    store.append_event("no_change", {})
    result = store.commit(1, 2, [], call_id="second")
    assert result["revision"] == 2
    assert store.read("objects/ball.md")["revision"] == 1
    assert store.read("objects/ball.md")["snapshot_revision"] == 2
    assert store.pending_events() == []


def test_late_event_remains_pending_after_old_window_commit(tmp_path):
    store = MDMemoryStore(tmp_path)
    first = store.append_event("observation", {})
    window = store.pending_events()
    later = store.append_event("user_instruction", {"text": "Return now"}, critical=True)
    store.commit(0, window[-1]["seq"], [change(first)], call_id="first-window")
    snapshot = store.snapshot()
    assert snapshot["latest_event_seq"] == later["seq"]
    assert snapshot["processed_event_seq"] == first["seq"]
    assert [event["id"] for event in store.pending_events()] == [later["id"]]
    assert snapshot["pending_critical_events"][0]["id"] == later["id"]


def test_idempotency_survives_reopen_and_rejects_changed_content(tmp_path):
    store = MDMemoryStore(tmp_path)
    event = store.append_event("observation", {})
    original = store.commit(0, 1, [change(event)], call_id="durable-id")
    reopened = MDMemoryStore(tmp_path)
    replay = reopened.commit(0, 1, [change(event)], call_id="durable-id")
    assert replay == {**original, "deduplicated": True}
    assert reopened.snapshot()["revision"] == 1
    assert_error("CALL_ID_REUSED", lambda: reopened.commit(0, 1, [change(event, "Different")], call_id="durable-id"))


def test_concurrent_stale_writers_have_exactly_one_winner(tmp_path):
    first = MDMemoryStore(tmp_path)
    event = first.append_event("observation", {})
    second = MDMemoryStore(tmp_path)
    barrier = threading.Barrier(2)

    def submit(store, number):
        barrier.wait()
        try:
            return store.commit(0, 1, [change(event, f"Writer {number}")], call_id=f"writer-{number}")["status"]
        except StoreError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(submit, store, i) for i, store in enumerate((first, second))]
        assert sorted(f.result() for f in futures) == ["VERSION_CONFLICT", "succeeded"]
    assert first.snapshot()["revision"] == 1
    assert len(first.snapshot()["files"]) == 1


def test_concurrent_event_append_uses_unique_contiguous_sequences(tmp_path):
    store = MDMemoryStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        events = list(executor.map(lambda number: store.append_event("observation", {"number": number}), range(20)))
    assert sorted(event["seq"] for event in events) == list(range(1, 21))
    assert [event["seq"] for event in store.pending_events(limit=20)] == list(range(1, 21))


def test_reader_sees_previous_revision_until_all_files_are_ready(tmp_path, monkeypatch):
    store = MDMemoryStore(tmp_path)
    event = store.append_event("observation", {})
    store.commit(0, 1, [change(event, "old")], call_id="old")
    next_event = store.append_event("observation", {})
    entered, release = threading.Event(), threading.Event()
    original = module._write_file

    def delayed(path, data):
        original(path, data)
        if path.name == "ball.md":
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(module, "_write_file", delayed)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.commit, 1, 2, [change(next_event, "new"), change(next_event, "task", "tasks/current.md")], call_id="new")
        assert entered.wait(5)
        snapshot = store.snapshot()
        assert snapshot["revision"] == 1
        assert [item["path"] for item in snapshot["files"]] == ["objects/ball.md"]
        assert store.read("objects/ball.md")["body"] == "old"
        release.set()
        assert future.result()["revision"] == 2
    assert len(store.snapshot()["files"]) == 2


def test_failed_file_write_never_publishes_partial_revision(tmp_path, monkeypatch):
    store = MDMemoryStore(tmp_path)
    event = store.append_event("observation", {})
    original = module._write_file

    def fail_manifest(path, data):
        if path.name == "manifest.json":
            raise OSError("injected disk failure")
        original(path, data)

    monkeypatch.setattr(module, "_write_file", fail_manifest)
    assert_error("STORE_UNAVAILABLE", lambda: store.commit(0, 1, [change(event)], call_id="disk-failure"))
    assert store.snapshot()["revision"] == 0
    assert store.pending_events()[0]["id"] == event["id"]
    monkeypatch.setattr(module, "_write_file", original)
    assert store.commit(0, 1, [change(event)], call_id="disk-failure")["revision"] == 1


def test_abandoned_directory_is_not_a_commit(tmp_path):
    store = MDMemoryStore(tmp_path)
    orphan = tmp_path / "revisions" / "00000099-abandoned"
    orphan.mkdir()
    (orphan / "manifest.json").write_text('{"revision":99}')
    reopened = MDMemoryStore(tmp_path)
    assert reopened.snapshot()["revision"] == 0
    assert_error("REVISION_NOT_FOUND", lambda: reopened.snapshot(99))


@pytest.mark.parametrize("path", ["../escape.md", "/tmp/escape.md", "objects/../escape.md", "objects//x.md", "objects/./x.md", "objects\\x.md", "objects/x.txt", "secrets/x.md", "x.md", "objects/.hidden.md"])
def test_unsafe_document_paths_rejected_without_commit(tmp_path, path):
    store = MDMemoryStore(tmp_path)
    event = store.append_event("observation", {})
    assert_error("INVALID_PATH", lambda: store.commit(0, 1, [change(event, path=path)], call_id="unsafe"))
    assert store.snapshot()["revision"] == 0


def test_missing_future_and_stale_evidence_are_rejected(tmp_path):
    store = MDMemoryStore(tmp_path)
    first = store.append_event("observation", {})
    second = store.append_event("observation", {})
    fake = {**first, "id": "event-missing"}
    assert_error("UNKNOWN_EVIDENCE", lambda: store.commit(0, 1, [change(fake)], call_id="missing"))
    assert_error("FUTURE_EVIDENCE", lambda: store.commit(0, 1, [change(second)], call_id="future"))
    store.commit(0, 2, [change(second)], call_id="newer")
    assert_error("STALE_EVIDENCE", lambda: store.commit(1, 2, [change(first, "Old location")], call_id="older"))
    assert store.snapshot()["revision"] == 1


def test_watermark_bounds_and_scope_isolation(tmp_path):
    store = MDMemoryStore(tmp_path, scope="site-A")
    event = store.append_event("observation", {})
    assert_error("INVALID_WATERMARK", lambda: store.commit(0, 2, [], call_id="skip-future"))
    store.commit(0, 1, [change(event)], call_id="one")
    assert_error("INVALID_WATERMARK", lambda: store.commit(1, 0, [], call_id="backwards"))
    assert_error("SCOPE_MISMATCH", lambda: MDMemoryStore(tmp_path, scope="site-B"))
    assert MDMemoryStore(tmp_path, scope="site-A").snapshot()["revision"] == 1


def revision_directory(root, revision):
    with sqlite3.connect(root / "index.sqlite3") as db:
        name = db.execute("SELECT directory FROM commits WHERE revision=?", (revision,)).fetchone()[0]
    return root / "revisions" / name


@pytest.mark.parametrize("file", ["manifest.json", "objects/ball.md"])
def test_corrupt_committed_files_fail_closed(tmp_path, file):
    store = MDMemoryStore(tmp_path)
    event = store.append_event("observation", {})
    store.commit(0, 1, [change(event)], call_id="one")
    (revision_directory(tmp_path, 1) / file).write_text("tampered")
    assert_error("MEMORY_CORRUPT", store.snapshot)
    assert_error("MEMORY_CORRUPT", lambda: store.read("objects/ball.md"))
    assert_error("MEMORY_CORRUPT", lambda: MDMemoryStore(tmp_path))
    assert store.snapshot(0)["revision"] == 0


def test_symlinked_committed_document_is_rejected(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("observation", {})
    store.commit(0, 1, [change(event)], call_id="one")
    document = revision_directory(store.root, 1) / "objects" / "ball.md"
    outside = tmp_path / "outside.md"
    outside.write_bytes(document.read_bytes())
    document.unlink()
    document.symlink_to(outside)
    assert_error("INVALID_PATH", store.snapshot)


def test_images_are_immutable_root_scoped_evidence(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    (store.root / "evidence").mkdir()
    image = store.root / "evidence" / "frame.png"
    image.write_bytes(b"png-test")
    event = store.append_event("frame", {}, image_path="evidence/frame.png")
    assert store.get_event(event["id"])["image_sha256"] == module._sha(b"png-test")
    image.write_bytes(b"changed")
    assert_error("EVIDENCE_CORRUPT", lambda: store.get_event(event["id"]))
    assert_error("EVIDENCE_CORRUPT", lambda: store.commit(0, 1, [change(event)], call_id="changed"))
    assert_error("INVALID_PATH", lambda: store.append_event("frame", {}, image_path="../outside.png"))
    image.unlink()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    image.symlink_to(outside)
    assert_error("INVALID_PATH", lambda: store.append_event("frame", {}, image_path="evidence/frame.png"))


def test_search_is_literal_bounded_and_revision_specific(tmp_path):
    store = MDMemoryStore(tmp_path)
    first = store.append_event("observation", {})
    store.commit(0, 1, [change(first, "Ball near OLD_NODE")], call_id="one")
    second = store.append_event("observation", {})
    store.commit(1, 2, [change(second, "Ball near NEW_NODE")], call_id="two")
    assert len(store.search("old_node", revision=1)) == 1
    assert store.search("OLD_NODE") == []
    assert store.search("new_node")[0]["revision"] == 2


def test_public_evidence_and_change_limits(tmp_path):
    store = MDMemoryStore(tmp_path)
    assert_error("NON_PUBLIC_EVIDENCE", lambda: store.append_event("frame", {"nested": {"api_key": "never-save"}}))
    assert_error("NON_PUBLIC_EVIDENCE", lambda: store.append_event("frame", {"source": "privileged_simulator_evaluator"}))
    event = store.append_event("observation", {})
    assert_error("INVALID_ARGUMENTS", lambda: store.commit(0, 1, [change(event, "a" * 6001)], call_id="large"))
    assert_error("INVALID_ARGUMENTS", lambda: store.commit(0, 1, [change(event, status="valid")], call_id="status"))
    assert_error("INVALID_ARGUMENTS", lambda: store.commit(0, 1, [change(event)] * 5, call_id="many"))
    assert_error("INVALID_ARGUMENTS", lambda: store.commit(0, 1, [change(event)] * 2, call_id="duplicate"))


def test_symlinked_root_ancestor_and_revision_directory_are_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    assert_error("INVALID_PATH", lambda: MDMemoryStore(alias / "memory"))
    root = tmp_path / "memory"
    root.mkdir()
    (root / "revisions").symlink_to(target, target_is_directory=True)
    assert_error("INVALID_PATH", lambda: MDMemoryStore(root))
