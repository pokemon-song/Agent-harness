"""Offline report fixtures; no model requests, MuJoCo stepping, or video encoding."""
import json

import pytest

from go2_demo.md_memory.store import MDMemoryStore
from scripts import report_md_roundtrip as report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def write_events(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def source_hashes(directory):
    return {str(path.relative_to(directory)): report.sha256(path)
            for path in directory.rglob("*") if path.is_file()}


def fixture_run(tmp_path, *, return_runs=True, timeout=False, llm=False):
    directory = tmp_path / "run"
    directory.mkdir()
    store = MDMemoryStore(directory / "memory", scope="fixture-shared-frame")
    event = store.append_event("observation", {"text": "Measured route source"})
    store.commit(0, 1, [{"path": "places/route.md", "body": "SEMANTIC_BODY_SENTINEL: route chosen by fixture.",
                         "evidence_ids": [event["id"]], "status": "confirmed"}], call_id="fixture-seed")
    document = store.read("places/route.md")
    write_json(directory / "experiment.json", {"experiment_kind": "md-roundtrip-v1", "is_llm_result": llm,
        "model": "fixture-not-llm", "git": {"commit": "fixture", "dirty": False},
        "coordinate_frame_id": "fixture-shared-frame"})
    phases = []
    for phase in ("outbound", "return") if return_runs else ("outbound",):
        returning = phase == "return"
        usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        second_usage = {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}
        events = [{"event": "model_input", "turn": 0, "image_path": "frames/000.png"},
                  {"event": "model_response", "turn": 0, "usage": usage}]
        if returning:
            events.append({"event": "tool_results", "turn": 0, "results": [
                {"tool": "memory_read", "status": "succeeded", "document": document}]})
            events.append({"event": "model_input", "turn": 1, "image_path": "frames/000.png"})
            if timeout:
                events.append({"event": "model_error", "turn": 1, "error": {"code": "MODEL_TIMEOUT"}, "latency_s": .5})
            else:
                events += [{"event": "model_response", "turn": 1, "usage": second_usage},
                    {"event": "tool_results", "turn": 1, "results": [{"tool": "move_relative", "status": "succeeded"}]}]
        else:
            events.append({"event": "tool_results", "turn": 0, "results": [{"tool": "move_relative", "status": "succeeded"}]})
        memory_usage = {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        memory = {"api_requests": 0 if returning else 1, "completed_responses": 0 if returning else 1,
                  "usage": {} if returning else memory_usage, "committed_batches": 0 if returning else 1}
        if not returning:
            write_events(directory / phase / "memory-agent.jsonl", [
                {"event": "model_request", "attempt": 1},
                {"event": "model_response", "attempt": 1, "usage": memory_usage}])
        planner_usage = dict(usage)
        if returning and not timeout:
            planner_usage = {key: value + second_usage[key] for key, value in usage.items()}
        evaluation = {"success": not (returning and timeout), "collision": False, "fallen": False, "stopped": True}
        if returning:
            evaluation.update(return_distance_to_start_m=.08, retrace_max_deviation_m=.2,
                retrace_mean_deviation_m=.09, retrace_p95_deviation_m=.18, outbound_coverage=.96,
                return_corridor_fraction=1., same_cabinet_side=True, return_stages_in_order=True,
                source="fixture-evaluator")
        metrics = {"success": not (returning and timeout), "termination": "model_error" if returning and timeout else "finish",
            "error": {"code": "MODEL_TIMEOUT"} if returning and timeout else None,
            "model_calls": 2 if returning else 1, "completed_responses": 1 if not returning or timeout else 2,
            "usage": planner_usage, "memory_agent": memory, "memory_agent_stopped": True,
            "memory_mode": "read_only" if returning else "read_write", "wall_elapsed_s": 2., "sim_elapsed_s": 1.,
            "path_length_m": 1.1, "evaluation": evaluation, "cleanup_errors": []}
        write_events(directory / phase / "events.jsonl", events)
        (directory / phase / "frames").mkdir()
        (directory / phase / "frames/000.png").write_bytes(b"fixture-frame-not-a-video")
        write_json(directory / phase / "metrics.json", metrics)
        for label in ("initial", "final"):
            write_json(directory / phase / ("memory-" + label + ".json"), store.snapshot())
        phases.append({"phase": phase, "metrics": metrics})
    summary = {"experiment_kind": "md-roundtrip-v1", "is_llm_result": llm,
        "mission_success": return_runs and not timeout, "phases": phases,
        "checks": {"shared_coordinate_frame": True, "return_memory_unchanged": True},
        "api_requests": 4 if return_runs else 2, "complete_responses": (3 if timeout else 4) if return_runs else 2,
        "known_usage": {"prompt_tokens": 25 if timeout else 45, "completion_tokens": 5 if timeout else 8,
                        "total_tokens": 30 if timeout else 53} if return_runs else
                       {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18},
        "unknown_usage_attempts": int(timeout and return_runs),
        "return_not_run_reason": None if return_runs else "outbound_failed", "error": None, "cleanup_errors": []}
    write_json(directory / "summary.json", summary)
    return directory, summary


def test_report_keeps_fixture_identity_and_real_reads_without_copying_semantic_body(tmp_path):
    directory, _ = fixture_run(tmp_path)
    before = source_hashes(directory)
    result = report.build_report(directory)
    analysis = json.loads((directory / "analysis.json").read_text())
    assert result["mission_success"] is True
    assert analysis["is_llm_result"] is False and analysis["api_requests"] == 0
    assert analysis["model_request_attempts"] == 4
    returning = analysis["phases"][1]
    assert returning["read_before_first_motion"] is True
    assert returning["memory_reads"][0]["href"].startswith("memory/revisions/")
    assert returning["return_evaluation"]["outbound_coverage"] == .96
    assert returning["return_evaluation"]["retrace_max_deviation_m"] == .2
    assert analysis["causal_memory_benefit_established"] is False
    for name in report.OUTPUTS:
        assert "SEMANTIC_BODY_SENTINEL" not in (directory / name).read_text()
    assert all(report.sha256(directory / name) == digest for name, digest in before.items())
    assert "同一个起点坐标系" in (directory / "REPORT.zh-CN.md").read_text()


def test_timeout_usage_remains_unknown_and_lifecycle_log_is_not_second_request(tmp_path):
    directory, _ = fixture_run(tmp_path, timeout=True, llm=True)
    path = directory / "return/events.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"event": "failure", "error": {"code": "MODEL_TIMEOUT"}}) + "\n")
    result = report.build_report(directory)
    data = json.loads((directory / "analysis.json").read_text())
    assert result["api_requests"] == 4 and result["unknown_usage_attempts"] == 1
    assert result["known_usage"]["total_tokens"] == 30 and result["usage_complete"] is False
    role = data["phases"][1]["planner"]
    assert len(role["request_errors"]) == 1 and len(role["worker_lifecycle_errors"]) == 1
    assert data["mission_success"] is False


def test_missing_return_is_not_passed_and_existing_report_is_protected(tmp_path):
    directory, summary = fixture_run(tmp_path, return_runs=False)
    summary["mission_success"] = True  # A report must reject a contradictory summary.
    write_json(directory / "summary.json", summary)
    result = report.build_report(directory)
    data = json.loads((directory / "analysis.json").read_text())
    assert result["mission_success"] is False and data["not_run"] == ["return"]
    assert data["return_not_run_reason"] == "outbound_failed"
    before = source_hashes(directory)
    with pytest.raises(FileExistsError):
        report.build_report(directory, render=True)
    assert source_hashes(directory) == before


def test_early_initialization_failure_still_has_report_when_video_cannot_render(tmp_path, monkeypatch):
    directory = tmp_path / "partial"
    directory.mkdir()
    write_json(directory / "experiment.json", {"experiment_kind": "md-roundtrip-v1", "is_llm_result": False})
    write_json(directory / "summary.json", {"experiment_kind": "md-roundtrip-v1", "phases": [],
        "mission_success": False, "error": {"code": "FIXTURE_INIT_FAILED"}, "return_not_run_reason": "outbound_not_completed"})
    def missing_archive(_):
        raise FileNotFoundError("fixture has no archive")
    monkeypatch.setattr(report, "render_continuous", missing_archive)
    report.build_report(directory, render=True)
    data = json.loads((directory / "analysis.json").read_text())
    assert data["not_run"] == ["outbound", "return"]
    assert data["error"]["code"] == "FIXTURE_INIT_FAILED"
    assert data["render_error"]["type"] == "FileNotFoundError"
    assert data["video"] is None and data["mission_success"] is False


def test_render_then_report_verifies_one_continuous_archive(tmp_path, monkeypatch):
    directory, _ = fixture_run(tmp_path)
    archive = directory / "third_person"
    archive.mkdir()
    for name in ("model.mjb", "states.npz", "metadata.json"):
        (archive / name).write_bytes(b"fixture-archive-" + name.encode())
    before = source_hashes(directory)
    calls = []
    def fake_render(target):
        calls.append(target)
        (archive / "third-person-sim.mp4").write_bytes(b"fixture-video")
        write_json(archive / "video-manifest.json", {
            "input_sha256": {name: report.sha256(archive / name) for name in ("model.mjb", "states.npz", "metadata.json")},
            "state_reconstruction": "fixture verification metadata; no simulation",
            "videos": {"sim": {"file": "third-person-sim.mp4", "sha256": report.sha256(archive / "third-person-sim.mp4"),
                "decoded_all_frames": True, "first_and_final_states_included": True, "frame_count": 25, "duration_s": 1.}}})
    monkeypatch.setattr(report, "render_continuous", fake_render)
    result = report.build_report(directory, render=True)
    data = json.loads((directory / "analysis.json").read_text())
    assert calls == [directory.resolve()] and result["video_verified"] is True
    assert data["video"]["reset_cards"] is False
    assert data["video"]["src"] == "third_person/third-person-sim.mp4"
    assert all(report.sha256(directory / name) == digest for name, digest in before.items())


def test_source_mutation_during_render_aborts_before_writing_reports(tmp_path, monkeypatch):
    directory, _ = fixture_run(tmp_path)
    def corrupt_source(_):
        (directory / "outbound/metrics.json").write_text("changed")
    monkeypatch.setattr(report, "render_continuous", corrupt_source)
    with pytest.raises(RuntimeError, match="Raw artifacts changed"):
        report.build_report(directory, render=True)
    assert all(not (directory / name).exists() for name in report.OUTPUTS)


def test_summary_metrics_disagreement_and_missing_memory_reads_cannot_pass(tmp_path):
    directory, summary = fixture_run(tmp_path)
    summary["phases"][1]["metrics"]["path_length_m"] = 9.0
    write_json(directory / "summary.json", summary)
    with pytest.raises(ValueError, match="disagree"):
        report.build_report(directory)
    summary["phases"][1]["metrics"]["path_length_m"] = 1.1
    write_json(directory / "summary.json", summary)
    path = directory / "return/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        if event.get("event") == "tool_results" and event.get("turn") == 0:
            event["results"] = []
    write_events(path, events)
    result = report.build_report(directory)
    assert result["mission_success"] is False
