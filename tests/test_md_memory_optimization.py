"""Failures seen in the real complex mission, without paid model requests."""

import threading

import pytest

from go2_demo.md_memory.store import MDMemoryStore
from go2_demo.models import ModelError
from test_md_memory_worker import Scripted, change, decision, logs, make_worker, wait_for


def test_critical_notice_reaches_first_request_without_acknowledging_backlog(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    old = [store.append_event("robot_feedback", {"step": n}) for n in range(16)]
    notice = store.append_event("instruction", {"text": "Dispatch code 001037"}, critical=True)
    model = Scripted([decision("commit_memory", {"changes": [change(notice)]})])
    worker = make_worker(tmp_path, store, model, max_calls=1).start()
    try:
        assert worker.wait_event(notice["seq"], 2)
        assert not worker.wait_until(notice["seq"], .02)
    finally:
        assert worker.stop(2)
    assert model.inputs[0][0]["pending_events"] == [notice]
    assert store.pending_events(limit=100) == old
    assert store.snapshot()["processed_event_seq"] == 0
    assert store.snapshot()["pending_event_count"] == 16
    assert not store.snapshot()["pending_critical_events"]


def test_inflight_feedback_is_preempted_at_response_boundary(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    old = store.append_event("observation", {"text": "Old room"})
    entered, release = threading.Event(), threading.Event()

    def blocked(*_):
        entered.set()
        assert release.wait(3)
        return decision("commit_memory", {"changes": [change(old, "OLD_REPLY_MUST_NOT_PUBLISH")]})

    def urgent(observation, *_):
        event = observation["pending_events"][0]
        assert event["critical"]
        return decision("commit_memory", {"changes": [change(event, "New notice")]})

    model = Scripted([blocked, urgent, decision("commit_memory", {"changes": []})])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert entered.wait(2)
        notice = store.append_event("instruction", {"text": "New notice"}, critical=True)
        worker.notify()
        release.set()
        assert worker.wait_until(notice["seq"], 2)
    finally:
        release.set()
        assert worker.stop(2)
    assert [o[0]["batch_event_seqs"] for o in model.inputs] == [[1], [2], [1]]
    assert store.read("tasks/current.md")["body"] == "New notice"
    assert worker.stats["preempted_batches"] == 1
    assert worker.stats["api_requests"] == 3
    assert worker.stats["committed_batches"] == 2


def test_failed_background_request_does_not_swallow_new_critical_notification(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    store.append_event("observation", {"text": "Old room"})
    entered, release = threading.Event(), threading.Event()

    def failed(*_):
        entered.set()
        assert release.wait(3)
        raise ModelError("private transport detail", "MODEL_TRANSPORT_ERROR")

    model = Scripted([failed, decision("commit_memory", {"changes": []}),
                      decision("commit_memory", {"changes": []})])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert entered.wait(2)
        notice = store.append_event("instruction", {"text": "Urgent update"}, critical=True)
        worker.notify()
        release.set()
        assert worker.wait_until(notice["seq"], 2)
    finally:
        release.set()
        assert worker.stop(2)
    assert [o[0]["batch_event_seqs"] for o in model.inputs] == [[1], [2], [1]]
    assert worker.stats["errors"] == []
    assert worker.stats["superseded_errors"][0]["code"] == "MODEL_TRANSPORT_ERROR"
    assert worker.stats["api_requests"] == 3 and worker.stats["completed_responses"] == 2
    assert any(row["event"] == "model_error" for row in logs(worker))
    assert "private transport detail" not in worker.log_path.read_text()


def test_field_errors_support_correction_without_echoing_content(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("instruction", {"text": "Store notice"})
    bad_path = {**change(event), "path": "world/layout.md"}
    too_long = change(event, "PRIVATE_BODY_SENTINEL" + "x" * 6000)
    model = Scripted([decision("commit_memory", {"changes": [bad_path]}),
                      decision("commit_memory", {"changes": [too_long]}),
                      decision("commit_memory", {"changes": [change(event)]})])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert worker.wait_until(event["seq"], 2)
    finally:
        assert worker.stop(2)
    path_error = model.inputs[1][3][-1]["results"][0]
    length_error = model.inputs[2][3][-1]["results"][0]
    assert path_error["error_code"] == "INVALID_PATH"
    assert path_error["details"]["field"] == "changes[0].path"
    assert length_error["details"]["actual"] == len(too_long["body"])
    assert length_error["details"]["max"] == 6000
    assert "PRIVATE_BODY_SENTINEL" not in str(length_error)


def test_patch_retains_unedited_identity_and_evidence_and_requires_read(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    source = store.append_event("instruction", {"text": "Initial order"})
    original = "Order order-00-ab.\nReceipt prefix: 001.\nACTIVE R1: left.\n"
    store.commit(0, 1, [change(source, original)], call_id="seed")
    notice = store.append_event("instruction", {"text": "Use right under R2"}, critical=True)
    patch = {"path": "tasks/current.md", "edits": [{"old": "ACTIVE R1: left.", "new": "ACTIVE R2: right."}],
             "evidence_ids": [notice["id"]], "status": "confirmed"}
    model = Scripted([decision("patch_memory", patch), decision("memory_read", {"path": "tasks/current.md"}),
                      decision("patch_memory", patch, call_id="after-read")])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert worker.wait_until(notice["seq"], 2)
    finally:
        assert worker.stop(2)
    document = store.read("tasks/current.md")
    assert document["body"] == original.replace("ACTIVE R1: left.", "ACTIVE R2: right.")
    assert document["evidence_ids"] == [source["id"], notice["id"]]
    assert model.inputs[1][3][-1]["results"][0]["error_code"] == "MEMORY_READ_REQUIRED"
    assert "content" not in model.inputs[2][3][-1]["results"][0]["document"]


def test_full_replacement_also_requires_reading_existing_document(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    source = store.append_event("instruction", {"text": "Initial order"})
    store.commit(0, 1, [change(source)], call_id="seed")
    later = store.append_event("observation", {})
    model = Scripted([decision("commit_memory", {"changes": [change(later, "Replacement")]}),
                      decision("memory_read", {"path": "tasks/current.md"}),
                      decision("commit_memory", {"changes": [change(later, "Replacement")]}, call_id="read")])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert worker.wait_until(later["seq"], 2)
    finally:
        assert worker.stop(2)
    assert model.inputs[1][3][-1]["results"][0]["error_code"] == "MEMORY_READ_REQUIRED"
    assert store.read("tasks/current.md")["body"] == "Replacement"


def test_unselected_pending_event_cannot_be_used_as_current_evidence(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    old = store.append_event("observation", {"text": "Not selected"})
    urgent = store.append_event("instruction", {"text": "Urgent"}, critical=True)
    model = Scripted([decision("commit_memory", {"changes": [change(old)]}),
                      decision("commit_memory", {"changes": [change(urgent)]}),
                      decision("commit_memory", {"changes": []})])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert worker.wait_until(urgent["seq"], 2)
    finally:
        assert worker.stop(2)
    assert model.inputs[1][3][-1]["results"][0]["error_code"] == "EVENT_OUTSIDE_BATCH"
    assert store.read("tasks/current.md")["evidence_ids"] == [urgent["id"]]
