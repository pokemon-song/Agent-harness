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


def test_critical_events_preempt_feedback_without_acknowledging_the_gap(tmp_path):
    store = MDMemoryStore(tmp_path)
    feedback = [store.append_event("feedback", {"number": i}) for i in range(16)]
    notice = store.append_event("new_instruction", {"revision": 2}, critical=True)
    assert store.pending_events() == feedback[:8]
    assert store.pending_events(prioritize_critical=True) == [notice]
    result = store.commit(0, 17, [change(notice, "Active instruction revision 2", "tasks/dispatch.md")],
                          call_id="critical", event_seqs=[17])
    assert result["processed_event_seq"] == 0
    assert result["acknowledged_event_seqs"] == [17]
    snapshot = store.snapshot()
    assert snapshot["processed_event_seq"] == 0
    assert snapshot["processed_critical_event_seq"] == 17
    assert snapshot["pending_event_count"] == 16
    assert snapshot["pending_critical_events"] == []
    assert store.is_event_processed(17)
    assert not store.is_event_processed(1)
    assert not store.is_event_processed(18)
    assert store.pending_events(limit=100, prioritize_critical=True) == feedback
    # Draining the old feedback incorporates the already-acknowledged notice.
    store.commit(1, 8, [], call_id="old-feedback-1", event_seqs=list(range(1, 9)))
    result = store.commit(2, 16, [], call_id="old-feedback-2", event_seqs=list(range(9, 17)))
    assert result["processed_event_seq"] == 17
    assert result["acknowledged_event_seqs"] == []
    assert store.snapshot()["pending_event_count"] == 0
    assert store.snapshot()["processed_critical_event_seq"] == 17
    assert store.pending_events(prioritize_critical=True) == []
    assert store.snapshot(1)["acknowledged_event_seqs"] == [17]
    assert store.snapshot(1)["pending_event_count"] == 16
    assert store.snapshot(0)["processed_critical_event_seq"] == 0


def test_multiple_critical_events_sorted_bounded_and_late_notice_stays_pending(tmp_path):
    store = MDMemoryStore(tmp_path)
    events = [store.append_event("notice" if seq in {3, 5, 7} else "feedback", {},
                                 critical=seq in {3, 5, 7}) for seq in range(1, 8)]
    window = store.pending_events(limit=2, prioritize_critical=True)
    assert [e["seq"] for e in window] == [3, 5]
    later = store.append_event("notice", {}, critical=True)
    store.commit(0, 5, [], call_id="first-critical-window", event_seqs=[3, 5])
    assert [e["seq"] for e in store.pending_events(prioritize_critical=True)] == [7, 8]
    assert [e["seq"] for e in store.snapshot()["pending_critical_events"]] == [7, later["seq"]]
    assert store.snapshot()["processed_critical_event_seq"] == 5
    assert store.pending_events(limit=100) == [e for e in events + [later] if e["seq"] not in {3, 5}]
    store.commit(1, 8, [], call_id="second-critical-window", event_seqs=[7, 8])
    assert [e["seq"] for e in store.pending_events(prioritize_critical=True)] == [1, 2, 4, 6]
    # Legacy prefix processing remains compatible and collapses sparse suffixes.
    store.commit(2, 4, [], call_id="legacy-feedback")
    assert store.snapshot()["processed_event_seq"] == 5
    assert store.snapshot()["acknowledged_event_seqs"] == [7, 8]
    store.commit(3, 6, [], call_id="last-feedback", event_seqs=[6])
    assert store.snapshot()["processed_event_seq"] == 8
    assert store.snapshot()["pending_event_count"] == 0
    assert store.snapshot()["processed_critical_event_seq"] == 8


def test_sparse_evidence_requires_acknowledged_or_selected_events(tmp_path):
    store = MDMemoryStore(tmp_path)
    old = store.append_event("feedback", {})
    gap = store.append_event("feedback", {})
    critical = store.append_event("notice", {}, critical=True)
    unselected = store.append_event("notice", {}, critical=True)
    assert_error("FUTURE_EVIDENCE", lambda: store.commit(
        0, 3, [change(gap)], call_id="unprocessed-gap", event_seqs=[3]))
    assert_error("FUTURE_EVIDENCE", lambda: store.commit(
        0, 3, [change(unselected)], call_id="unselected-notice", event_seqs=[3]))
    store.commit(0, 3, [change(critical, "Updated instruction")], call_id="priority", event_seqs=[3])
    # Already-acknowledged critical evidence remains available even though its
    # sequence is higher than the currently processed old-feedback batch.
    mixed = change(critical, "Keep new instruction; summarize old feedback.")
    mixed["evidence_ids"].append(old["id"])
    store.commit(1, 1, [mixed], call_id="old-with-current-context", event_seqs=[1])
    assert store.snapshot()["processed_event_seq"] == 1
    assert store.snapshot()["acknowledged_event_seqs"] == [3]
    assert_error("STALE_EVIDENCE", lambda: store.commit(
        2, 2, [change(gap, "Replace with old instruction")], call_id="late-old-overwrite", event_seqs=[2]))
    assert not store.is_event_processed(2)
    # An old prefix caller may also retain newer acknowledged evidence.
    store.commit(2, 2, [change(critical, "Retain current instruction")], call_id="legacy-with-context")
    assert store.snapshot()["processed_event_seq"] == 3
    assert store.pending_events() == [unselected]


@pytest.mark.parametrize("selected", [[], [1, 1], [2, 1], [True], [0], [-1], [1.0], "1", list(range(1, 102))])
def test_invalid_explicit_selections_fail_without_acknowledgement(tmp_path, selected):
    store = MDMemoryStore(tmp_path)
    event = store.append_event("feedback", {})
    assert_error("INVALID_ARGUMENTS", lambda: store.commit(0, 1, [], call_id="invalid", event_seqs=selected))
    assert store.pending_events() == [event]
    assert store.snapshot()["revision"] == 0


def test_explicit_selection_bounds_and_already_processed_events(tmp_path):
    store = MDMemoryStore(tmp_path)
    for number in range(3):
        store.append_event("event", {"number": number})
    assert_error("INVALID_WATERMARK", lambda: store.commit(0, 2, [], call_id="wrong-max", event_seqs=[3]))
    assert_error("INVALID_WATERMARK", lambda: store.commit(0, 4, [], call_id="unregistered", event_seqs=[4]))
    store.commit(0, 3, [], call_id="sparse", event_seqs=[3])
    assert_error("EVENT_ALREADY_PROCESSED", lambda: store.commit(1, 3, [], call_id="repeat", event_seqs=[3]))
    store.commit(1, 1, [], call_id="first", event_seqs=[1])
    assert_error("EVENT_ALREADY_PROCESSED", lambda: store.commit(2, 1, [], call_id="prefix-repeat", event_seqs=[1]))
    assert_error("INVALID_ARGUMENTS", lambda: store.pending_events(prioritize_critical=1))
    assert_error("INVALID_ARGUMENTS", lambda: store.is_event_processed(True))
    assert store.snapshot()["pending_event_count"] == 1


def test_sparse_idempotency_and_readonly_reopen_do_not_change_any_file(tmp_path):
    store = MDMemoryStore(tmp_path)
    store.append_event("feedback", {})
    notice = store.append_event("notice", {}, critical=True)
    original = store.commit(0, 2, [change(notice)], call_id="durable-sparse", event_seqs=[2])
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    reopened = MDMemoryStore(tmp_path)
    assert reopened.snapshot()["acknowledged_event_seqs"] == [2]
    assert reopened.snapshot()["pending_event_count"] == 1
    assert reopened.is_event_processed(2)
    reopened.snapshot(0)
    reopened.read("objects/ball.md")
    reopened.search("ball")
    reopened.pending_events(prioritize_critical=True)
    replay = reopened.commit(0, 2, [change(notice)], call_id="durable-sparse", event_seqs=[2])
    assert replay == {**original, "deduplicated": True}
    assert_error("CALL_ID_REUSED", lambda: reopened.commit(
        0, 2, [change(notice)], call_id="durable-sparse", event_seqs=[1, 2]))
    assert_error("CALL_ID_REUSED", lambda: reopened.commit(0, 2, [change(notice)], call_id="durable-sparse"))
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_legacy_manifests_and_receipts_reopen_without_migration(tmp_path):
    store = MDMemoryStore(tmp_path)
    event = store.append_event("observation", {})
    store.commit(0, 1, [change(event)], call_id="legacy")
    # Recreate the pre-extension disk shape, including its original receipt.
    with sqlite3.connect(tmp_path / "index.sqlite3") as db:
        for revision, directory in db.execute("SELECT revision,directory FROM commits").fetchall():
            path = tmp_path / "revisions" / directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest.pop("acknowledged_event_seqs")
            data = module._json(manifest).encode()
            path.write_bytes(data)
            db.execute("UPDATE commits SET manifest_sha256=? WHERE revision=?", (module._sha(data), revision))
        legacy_result = {"status": "succeeded", "revision": 1, "processed_event_seq": 1}
        db.execute("UPDATE receipts SET result=? WHERE call_id='legacy'", (module._json(legacy_result),))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    reopened = MDMemoryStore(tmp_path)
    assert reopened.snapshot()["acknowledged_event_seqs"] == []
    assert reopened.snapshot()["pending_event_count"] == 0
    assert reopened.snapshot(0)["pending_event_count"] == 1
    assert reopened.commit(0, 1, [change(event)], call_id="legacy") == {**legacy_result, "deduplicated": True}
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    reopened.append_event("feedback", {})
    notice = reopened.append_event("notice", {}, critical=True)
    reopened.commit(1, 3, [change(notice)], call_id="new-format", event_seqs=[3])
    assert reopened.snapshot()["acknowledged_event_seqs"] == [3]
    assert reopened.snapshot(1)["acknowledged_event_seqs"] == []


@pytest.mark.parametrize("acknowledged", [[1], [3, 3], [3, 2], [999], [True], [0], "3"])
def test_invalid_sparse_manifest_metadata_fails_closed(tmp_path, acknowledged):
    store = MDMemoryStore(tmp_path)
    for _ in range(3):
        store.append_event("feedback", {})
    store.commit(0, 3, [], call_id="sparse", event_seqs=[3])
    path = revision_directory(tmp_path, 1) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["acknowledged_event_seqs"] = acknowledged
    data = module._json(manifest).encode()
    path.write_bytes(data)
    with sqlite3.connect(tmp_path / "index.sqlite3") as db:
        db.execute("UPDATE commits SET manifest_sha256=? WHERE revision=1", (module._sha(data),))
    assert_error("MEMORY_CORRUPT", store.snapshot)
    assert_error("MEMORY_CORRUPT", lambda: MDMemoryStore(tmp_path))


def test_concurrent_sparse_writers_do_not_acknowledge_the_losing_selection(tmp_path):
    first = MDMemoryStore(tmp_path)
    first.append_event("feedback", {})
    notices = [first.append_event("notice", {}, critical=True) for _ in range(2)]
    second = MDMemoryStore(tmp_path)
    barrier = threading.Barrier(2)

    def submit(store, event):
        barrier.wait()
        try:
            store.commit(0, event["seq"], [], call_id=event["id"], event_seqs=[event["seq"]])
            return "succeeded", event["seq"]
        except StoreError as exc:
            return exc.code, event["seq"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda pair: submit(*pair), zip([first, second], notices)))
    assert sorted(status for status, _ in outcomes) == ["VERSION_CONFLICT", "succeeded"]
    winner = next(seq for status, seq in outcomes if status == "succeeded")
    assert first.snapshot()["acknowledged_event_seqs"] == [winner]
    assert first.snapshot()["pending_event_count"] == 2
    assert first.snapshot()["pending_critical_events"][0]["seq"] != winner


def test_sparse_acknowledgement_publishes_atomically_with_its_document(tmp_path, monkeypatch):
    store = MDMemoryStore(tmp_path)
    store.append_event("feedback", {})
    notice = store.append_event("notice", {}, critical=True)
    entered, release = threading.Event(), threading.Event()
    original = module._write_file

    def delayed(path, data):
        original(path, data)
        if path.name == "ball.md":
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(module, "_write_file", delayed)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.commit, 0, 2, [change(notice)], call_id="priority", event_seqs=[2])
        assert entered.wait(5)
        assert not store.is_event_processed(2)
        assert store.snapshot()["acknowledged_event_seqs"] == []
        assert store.pending_events(prioritize_critical=True) == [notice]
        release.set()
        assert future.result()["processed_event_seq"] == 0
    assert store.is_event_processed(2)
    assert store.read("objects/ball.md")["evidence_ids"] == [notice["id"]]
