"""Real demo orchestration with explicit scripted models and no network access."""
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest

from go2_demo.md_memory import fixtures
from go2_demo.md_memory.store import MDMemoryStore
from scripts import run_md_memory_demo as demo


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unit test attempted network/model API access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(demo.APIModel, "from_env", forbidden)


def read_json(path):
    return json.loads(path.read_text())


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_scripted_fixture_uses_both_agent_protocols_and_preserves_old_md(tmp_path):
    directory = tmp_path / "run"
    summary = demo.run_demo(directory, record=False, api_timeout=1, stage_timeout=5)
    assert summary["demonstrated"], summary
    assert summary["is_llm_result"] is False
    assert summary["simulator_used"] is False
    assert summary["api_requests"] == 0
    assert summary["known_total_tokens"] == 0
    assert summary["formal_score"] is None
    assert summary["model_requests"] >= 6
    assert len(summary["phases"]) == 2
    first, revised = summary["phases"]
    assert first["execution_match"] and revised["execution_match"]
    assert first["plan"]["arguments"]["lateral_m"] == -revised["plan"]["arguments"]["lateral_m"]
    assert summary["pending_events"] == 0
    assert summary["stale_plan_probe"]["blocked"]
    assert summary["stale_plan_probe"]["code"] in {"MEMORY_PENDING", "MEMORY_CHANGED"}
    assert summary["cleanup_errors"] == []
    assert summary["final_stop"]["status"] == "succeeded"
    assert summary["memory_agent"]["committed_batches"] >= 2
    assert [p["completed_responses"] for p in summary["planners"]] == [2, 2]
    snapshot = read_json(directory / "memory-final.json")
    before = hashes(directory / "memory")
    store = MDMemoryStore(directory / "memory", scope=snapshot["scope"])
    assert store.snapshot() == snapshot
    old = store.read("tasks/current_order.md", revision=first["memory_revision"])
    new = store.read("tasks/current_order.md", revision=revised["memory_revision"])
    assert "ACTIVE R1" in old["body"] and "ACTIVE R2" not in old["body"]
    assert "ACTIVE R2" in new["body"] and "ARCHIVED R1" in new["body"]
    assert old["sha256"] != new["sha256"]
    store.search("ACTIVE", revision=first["memory_revision"])
    assert hashes(directory / "memory") == before, "Reopen and read operations changed persistent files"
    assert (directory / "REPORT.zh-CN.md").exists()
    assert (directory / "watch.html").exists()
    for phase in ("initial", "revised"):
        records = [json.loads(line) for line in (directory / f"planner-{phase}.jsonl").read_text().splitlines()]
        calls = [call["name"] for r in records if r["event"] == "model_response" for call in r["calls"]]
        assert calls == ["memory_read", "submit_plan"]
        inputs = [r["observation"] for r in records if r["event"] == "model_input"]
        assert all("contract.private.json" not in json.dumps(observation) for observation in inputs)
        assert all("revised_lateral_m" not in json.dumps(observation) for observation in inputs)


def test_planner_failure_stops_backend_and_retains_request_counts_without_secret_text(tmp_path, monkeypatch):
    class TrackedBackend(fixtures.FixtureBackend):
        def __init__(self):
            super().__init__()
            self.stop_count = 0

        def stop(self):
            self.stop_count += 1
            return super().stop()

    class FailedPlanner(fixtures.FixturePlannerModel):
        def decide(self, *args, **kwargs):
            raise RuntimeError("SECRET_SENTINEL_DO_NOT_LOG")

    backend = TrackedBackend()
    monkeypatch.setattr(fixtures, "FixtureBackend", lambda: backend)
    monkeypatch.setattr(fixtures, "FixturePlannerModel", FailedPlanner)
    directory = tmp_path / "failed"
    summary = demo.run_demo(directory, record=False, api_timeout=1, stage_timeout=5)
    assert not summary["demonstrated"]
    assert summary["error"]["type"] == "RuntimeError"
    assert backend.closed and backend.stop_count >= 1
    assert summary["memory_agent"]["stopped"]
    assert summary["planners"][0]["api_requests"] == 1
    assert summary["planners"][0]["completed_responses"] == 0
    assert summary["model_requests"] >= 2
    assert summary["api_requests"] == 0
    for path in directory.rglob("*"):
        if path.suffix in {".json", ".jsonl", ".md", ".html"}:
            assert "SECRET_SENTINEL_DO_NOT_LOG" not in path.read_text()


@pytest.mark.parametrize("recording_fails", [False, True])
def test_physics_adapter_trajectory_and_recording_cleanup(tmp_path, monkeypatch, recording_fails):
    class FailedRecording:
        calls = 0

        def finish(self):
            self.calls += 1
            if recording_fails:
                raise OSError("SECRET_RECORDING_PATH")

    class RecordingBackend(fixtures.FixtureBackend):
        def __init__(self):
            super().__init__()
            self.data = SimpleNamespace(time=0.)
            self.trajectory = []
            self.recording = FailedRecording()
            self.evaluator = SimpleNamespace(evaluate=lambda: {"collision": False, "fallen": False,
                                                               "stopped": True, "speed_mps": 0.})

        def execute(self, call):
            result = super().execute(call)
            self.data.time = self.time_s
            self.trajectory.append({"base_xyz_m": [*self.xy, .35], "sim_time_s": self.time_s})
            return result

        def start_recording(self, path):
            return self.recording

    backend = RecordingBackend()
    monkeypatch.setattr(demo, "create_backend", lambda *args, **kwargs: backend)
    directory = tmp_path / "cleanup"
    summary = demo.run_demo(directory, backend_kind="mujoco", record=True, api_timeout=1, stage_timeout=5)
    assert summary["demonstrated"] is (not recording_fails)
    assert backend.closed
    assert backend.recording.calls == 1
    assert summary["path_length_m"] == pytest.approx(.5)
    assert summary["sim_elapsed_s"] == pytest.approx(2.)
    assert len(read_json(directory / "trajectory.json")["points"]) == 2
    assert [error["operation"] for error in summary["cleanup_errors"]] == (["recording"] if recording_fails else [])
    assert "SECRET_RECORDING_PATH" not in (directory / "summary.json").read_text()
    assert (directory / "memory-final.json").exists()


def test_planner_close_failure_keeps_both_planner_request_counts(tmp_path, monkeypatch):
    class CloseFailedPlanner(fixtures.FixturePlannerModel):
        def close(self):
            raise RuntimeError("PRIVATE_CLOSE_DETAIL")

    monkeypatch.setattr(fixtures, "FixturePlannerModel", CloseFailedPlanner)
    summary = demo.run_demo(tmp_path / "close-failure", record=False, api_timeout=1, stage_timeout=5)
    assert not summary["demonstrated"]
    assert len(summary["phases"]) == 2
    assert [stats["api_requests"] for stats in summary["planners"]] == [2, 2]
    assert [error["operation"] for error in summary["cleanup_errors"]] == ["planner.close", "planner.close"]
    assert "PRIVATE_CLOSE_DETAIL" not in json.dumps(summary)


def test_reported_worker_join_failure_keeps_request_counts(tmp_path, monkeypatch):
    class JoinReportedFailedWorker(demo.MemoryWorker):
        def stop(self, timeout_s=5):
            assert super().stop(timeout_s), "Test worker must actually exit"
            return False

    monkeypatch.setattr(demo, "MemoryWorker", JoinReportedFailedWorker)
    summary = demo.run_demo(tmp_path / "join-failure", record=False, api_timeout=1, stage_timeout=5)
    assert not summary["demonstrated"]
    assert not summary["worker_stopped"]
    assert summary["memory_agent"]["api_requests"] >= 2
    assert summary["model_requests"] == summary["memory_agent"]["api_requests"] + 4
    assert not summary["usage_complete"]
    assert any(error["operation"] == "memory_worker.stop" for error in summary["cleanup_errors"])


def test_final_source_change_invalidates_demo(tmp_path, monkeypatch):
    calls = []

    def identity():
        calls.append(True)
        return {"source": "original" if len(calls) <= 3 else "changed"}

    monkeypatch.setattr(demo, "source_identity", identity)
    summary = demo.run_demo(tmp_path / "source-change", record=False, api_timeout=1, stage_timeout=5)
    assert not summary["source_unchanged"]
    assert not summary["demonstrated"]
    assert len(summary["phases"]) == 2


def test_fast_memory_commit_before_stale_probe_is_still_a_valid_block(tmp_path, monkeypatch):
    workers = []
    original_check = demo.check_plan

    class CapturedWorker(demo.MemoryWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            workers.append(self)

    def after_commit_check(plan, store, profile):
        # Force the legitimate schedule where memory finishes R2 before the
        # probe observes it. The same real worker/model protocol performs it.
        target = store.snapshot()["latest_event_seq"]
        workers[0].notify()
        assert workers[0].wait_until(target, timeout_s=5)
        return original_check(plan, store, profile)

    monkeypatch.setattr(demo, "MemoryWorker", CapturedWorker)
    monkeypatch.setattr(demo, "check_plan", after_commit_check)
    summary = demo.run_demo(tmp_path / "fast-worker", record=False, api_timeout=1, stage_timeout=5)
    assert summary["demonstrated"], summary
    assert summary["stale_plan_probe"] == {"blocked": True, "code": "MEMORY_CHANGED"}


def test_existing_experiment_directory_is_never_overwritten(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "previous.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        demo.run_demo(directory, record=False)
    assert marker.read_text() == "original"


@pytest.mark.parametrize("incomplete", [{}, {"total_tokens": 8},
                                      {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3}])
def test_one_complete_response_cannot_hide_another_missing_usage(tmp_path, incomplete):
    records = [{"event": "model_response", "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
               {"event": "model_response", "usage": incomplete}]
    (tmp_path / "memory-agent.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    assert demo.complete_usage_responses(tmp_path) == 1
