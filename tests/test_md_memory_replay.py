"""Portable queue/context regression fixtures; no API and no physics."""
from copy import deepcopy
import json

import pytest

from go2_demo.md_memory.context import document_context, project_event
from go2_demo.md_memory.store import MDMemoryStore
from scripts.replay_md_memory_regressions import build_replay, tree_hashes


ORDER_LITERAL = "order-FIXTURE-73d9a1"


def feedback_payload():
    observation = {
        "image_id": "frame-fixture-3", "sim_time_s": 12.5,
        "task": "A repeated navigation instruction. " * 60,
        "capabilities": {"capability_description": "Repeated configuration. " * 60},
        "local_odometry": {"x_m": 1.2, "y_m": -0.4},
        "robot_state": {"fallen": False}, "sensors": {"clearance_m": 0.6},
        "tracked_points": [{"point_id": "target-1", "distance_m": 0.7}],
        "new_sensor_field": {"dynamic_evidence": "fixture"},
    }
    return {"episode_id": "fixture-episode", "scene_id": "fixture-scene",
            "coordinate_reference": "episode-start-local-frame",
            "call": {"name": "move_relative", "arguments": {"forward_m": 0.2}},
            "result": {"status": "succeeded", "measured_distance_m": 0.18},
            "observation": deepcopy(observation), "next_observation": observation}


def seed_backlog(tmp_path):
    source_run = tmp_path / "historical-run"
    store = MDMemoryStore(source_run / "memory/fragments", scope="fixture-world")
    first = store.append_event("public_announcement", {"text": ORDER_LITERAL}, critical=True)
    for _ in range(17):
        store.append_event("robot_feedback", feedback_payload())
    store.commit(0, 18, [{"path": "tasks/current.md", "body": f"Order: {ORDER_LITERAL}\nRoute: left",
                         "status": "confirmed", "evidence_ids": [first["id"]]}], call_id="fixture-seed")
    for _ in range(16):
        store.append_event("robot_feedback", feedback_payload())
    notice = store.append_event("public_announcement", {
        "text": f"Keep order ID exactly {ORDER_LITERAL}; revised route: right. 公告必须逐字保留。"}, critical=True)
    assert notice["seq"] == 35
    return source_run, store, notice


def test_replay_prioritizes_announcement_without_acknowledging_old_feedback(tmp_path):
    source, store, _ = seed_backlog(tmp_path)
    before = tree_hashes(store.root)
    report = build_replay(source, tmp_path / "replay")
    assert report["passed"] is True
    assert report["is_llm_result"] is False
    assert report["model_api_requests"] == 0
    assert report["simulator_used"] is False
    assert report["fixture_acknowledgements_only"] is True
    assert report["old_fifo_first_batch"] == list(range(19, 27))
    assert report["new_priority_first_batch"] == [35]
    assert report["fifo_batch_number_for_target"] == 3
    assert report["older_pending_feedback_count"] == 16
    assert report["after_priority"]["processed_event_seq"] == 18
    assert report["after_priority"]["acknowledged_event_seqs"] == [35]
    assert report["after_priority"]["pending_event_count"] == 16
    assert report["pending_after_priority"] == list(range(19, 35))
    assert report["fixture_acknowledgement_batches"] == [[35], list(range(19, 27)), list(range(27, 35))]
    assert report["final"]["processed_event_seq"] == 35
    assert report["final"]["acknowledged_event_seqs"] == []
    assert report["final"]["pending_event_count"] == 0
    assert tree_hashes(store.root) == before
    assert store.snapshot()["pending_event_count"] == 17
    assert not store.is_event_processed(35)


def test_replay_reports_same_event_projection_separately_and_omits_source_bodies(tmp_path):
    source, store, _ = seed_backlog(tmp_path)
    output = tmp_path / "replay"
    report = build_replay(source, output)
    sizes = report["input_sizes"]
    assert sizes["same_fifo_batch_projection_only"]["byte_reduction_fraction"] > 0.5
    assert sizes["all_pending_projection_only"]["byte_reduction_fraction"] > 0.5
    assert sizes["document_read_context"]["after"]["utf8_bytes"] < sizes["document_read_context"]["before"]["utf8_bytes"]
    saved = json.loads((output / "replay.json").read_text())
    assert saved == report
    assert ORDER_LITERAL not in (output / "replay.json").read_text()
    assert ORDER_LITERAL not in (output / "REPORT.zh-CN.md").read_text()
    copied = MDMemoryStore(output / "memory-copy", scope=store.scope)
    assert copied.read("tasks/current.md")["body"] == store.read("tasks/current.md")["body"]
    assert "JSON" in (output / "REPORT.zh-CN.md").read_text()


def test_replay_cannot_write_inside_historical_source_or_overwrite_output(tmp_path):
    source, store, _ = seed_backlog(tmp_path)
    before = tree_hashes(source)
    with pytest.raises(ValueError, match="outside"):
        build_replay(source, source / "replay")
    assert tree_hashes(source) == before
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "do-not-replace.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError, match="new directory"):
        build_replay(source, output)
    assert sentinel.read_text() == "preserve"
    assert store.snapshot()["revision"] == 1


def test_replay_rejects_symlink_sources_and_open_sqlite_journals(tmp_path):
    source, store, _ = seed_backlog(tmp_path)
    link = store.root / "linked-evidence"
    link.symlink_to(tmp_path / "unrelated")
    with pytest.raises(ValueError, match="symlinks"):
        build_replay(source, tmp_path / "symlink-output")
    assert not (tmp_path / "symlink-output").exists()
    link.unlink()
    (store.root / "index.sqlite3-wal").write_bytes(b"fixture-active-journal")
    with pytest.raises(ValueError, match="closed historical"):
        build_replay(source, tmp_path / "journal-output")
    assert not (tmp_path / "journal-output").exists()


def test_projection_preserves_measured_feedback_and_keeps_original_ledger(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("robot_feedback", feedback_payload())
    original = deepcopy(event)
    projected = project_event(event)
    assert event == original == store.get_event(event["id"])
    for key in ("call", "result", "coordinate_reference", "episode_id", "scene_id"):
        assert projected["payload"][key] == event["payload"][key]
    assert "observation" not in projected["payload"]
    after = projected["payload"]["next_observation"]
    assert set(after) == {"image_id", "sim_time_s", "local_odometry", "robot_state", "sensors",
                          "tracked_points", "new_sensor_field"}
    assert all(value == event["payload"]["next_observation"][key] for key, value in after.items())
    assert projected["context_projection"]["full_event_tool"] == "inspect_event"
    assert projected["context_projection"]["event_id"] == event["id"]


def test_projection_never_summarizes_critical_or_unknown_event_envelopes(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    ordinary_notice = store.append_event("public_announcement", {"text": ORDER_LITERAL})
    critical_feedback = store.append_event("robot_feedback", feedback_payload(), critical=True)
    unfamiliar = store.append_event("robot_feedback", {**feedback_payload(), "external_constraint": ORDER_LITERAL})
    legacy = store.append_event("robot_feedback", {"result": {"status": "failed"}, "text": ORDER_LITERAL})
    for event in (ordinary_notice, critical_feedback, unfamiliar, legacy):
        assert project_event(event) == event


def test_projection_preserves_changed_task_and_calibration(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    payload = feedback_payload()
    payload["next_observation"]["task"] = "New instruction: wait beside the revised target."
    payload["observation"]["sensor_calibration"] = {"camera_yaw": 0.0}
    payload["next_observation"]["sensor_calibration"] = {"camera_yaw": 0.1}
    event = store.append_event("robot_feedback", payload)
    after = project_event(event)["payload"]["next_observation"]
    assert after["task"] == payload["next_observation"]["task"]
    assert after["sensor_calibration"] == {"camera_yaw": 0.1}
    assert "capabilities" not in after


def test_document_context_retains_literal_body_and_revision_metadata(tmp_path):
    _, store, _ = seed_backlog(tmp_path)
    document = store.read("tasks/current.md")
    reduced = document_context(document)
    assert "content" in document and "content" not in reduced
    assert reduced["body"] == document["body"]
    assert ORDER_LITERAL in reduced["body"]
    assert reduced["snapshot_revision"] == document["snapshot_revision"]
    assert reduced["sha256"] == document["sha256"]
    reduced["evidence_ids"].append("fixture-independent-copy")
    assert reduced["evidence_ids"] != document["evidence_ids"]
