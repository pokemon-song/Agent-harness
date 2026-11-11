"""Offline lifecycle checks, deliberately not model or robot success scores."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts/run_md_roundtrip.py"
    spec = importlib.util.spec_from_file_location("md_roundtrip_driver_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "frozen_sources", lambda: {"fixture.py": "unchanged"})
    monkeypatch.setattr(module, "git_identity", lambda root: {"commit": "fixture", "dirty": False})
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    from go2_demo.models import APIModel

    def forbidden(*args, **kwargs):
        raise AssertionError("Round-trip driver tests must not use network or API credentials")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(APIModel, "from_env", forbidden)


def args(tmp_path):
    return SimpleNamespace(runs_dir=tmp_path, api_key_file=tmp_path / "not-a-key",
        outbound_calls=60, return_calls=90, memory_calls=48, stage_timeout=1800.,
        sim_timeout=300., api_timeout=180., max_output_tokens=16384)


class Backend:
    def __init__(self):
        self.xyz, self.yaw, self.time = [.03, -.02, .3], .11, 1.
        self.trajectory, self.close_count, self.stop_count = [], 0, 0
        self.evaluator = object()
        self.close_error = False

    def capture(self):
        return SimpleNamespace(base_xy_yaw=(*self.xyz[:2], self.yaw))

    def _evaluation_state(self):
        return SimpleNamespace(position_xyz_m=list(self.xyz), yaw_rad=self.yaw, sim_time_s=self.time,
            velocity_xy_mps=[0., 0.], yaw_rate_radps=0., fallen=False, collision_steps=0,
            trajectory=self.trajectory)

    def advance(self, xyz):
        self.xyz, self.time = list(xyz), self.time + 10.
        self.trajectory.append({"sim_time_s": self.time, "base_xyz_m": list(self.xyz), "yaw_rad": self.yaw})

    def reset(self, *args, **kwargs):
        raise AssertionError("The physical world must not reset at B")

    def stop(self):
        self.stop_count += 1
        return {"status": "succeeded"}

    def close(self):
        self.close_count += 1
        if self.close_error:
            raise RuntimeError("fixture backend cleanup failed")


class Model:
    def __init__(self, role):
        self.role, self.close_count = role, 0

    def close(self):
        self.close_count += 1


def write_events(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


class Wiring:
    """Replace all model/physics execution; retain the real disk memory store."""

    def __init__(self, driver, monkeypatch, **options):
        self.driver, self.options = driver, options
        self.backend, self.models, self.calls, self.created = Backend(), [], [], []
        self.evaluator_arguments = None
        monkeypatch.setattr(driver, "run_md_episode", self.episode)
        monkeypatch.setattr(driver, "ReturnEvaluator", self.evaluator)

    def make_backend(self, scene):
        self.created.append(scene.task_id)
        return self.backend

    def make_model(self, role):
        if role == self.options.get("model_failure"):
            raise RuntimeError("fixture model construction failed")
        model = Model(role)
        self.models.append(model)
        return model

    def evaluator(self, *values):
        self.evaluator_arguments = values
        return object()

    def episode(self, backend, planner, **kwargs):
        phase = planner.role
        self.calls.append({"phase": phase, "backend": backend, "planner": planner,
                           "before": deepcopy(backend._evaluation_state()), "kwargs": kwargs})
        if self.options.get("episode_interrupt") == phase:
            raise KeyboardInterrupt()
        store, directory = kwargs["store"], kwargs["run_dir"]
        directory.mkdir(parents=True, exist_ok=True)
        observation = {"sim_time_s": backend.time,
            "agent_context": {"coordinate_frame_id": kwargs["coordinate_frame_id"]},
            "local_odometry": {"reference": "mission_start_body_horizontal"}}
        history = 0
        if phase == "return":
            observation["sim_time_s"] += self.options.get("return_clock_offset", 0.)
            observation["agent_context"]["coordinate_frame_id"] = self.options.get("return_frame", kwargs["coordinate_frame_id"])
            observation["local_odometry"]["reference"] = self.options.get("return_reference", "mission_start_body_horizontal")
            history = self.options.get("return_history", 0)
        write_events(directory / "events.jsonl", [
            {"event": "model_input", "observation": observation, "history_turns": history},
            {"event": "model_response", "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}])
        if phase == "outbound":
            if self.options.get("publish_memory", True):
                event = store.append_event("robot_feedback", {"fixture": "measured-route placeholder"}, critical=True)
                store.commit(0, event["seq"], [{"path": "places/route.md", "body": "Offline wiring fixture only",
                    "evidence_ids": [event["id"]], "status": "confirmed"}], call_id="fixture-publish")
            if self.options.get("pending_critical"):
                store.append_event("robot_feedback", {"fixture": "unprocessed"}, critical=True)
            backend.advance([4.5, 0., .3])
            kwargs["memory_model"].close()
        else:
            if self.options.get("mutate_readonly"):
                (store.root / "unexpected-write.txt").write_text("fixture read-only violation")
            backend.advance([.03, -.02, .3])
        planner.close()
        metrics = {"success": self.options.get("outbound_success", True) if phase == "outbound" else True,
            "termination": "finish", "mode": kwargs.get("mode", "api"),
            "is_llm_result": kwargs.get("mode", "api") == "api",
            "memory_read_calls": self.options.get("return_reads", 1) if phase == "return" else 0,
            "path_length_m": 5.}
        self.driver.write_json(directory / "metrics.json", metrics)
        hook = self.options.get("after_" + phase)
        if hook:
            hook()
        return metrics

    def run(self, tmp_path):
        return self.driver.experiment(args(tmp_path), backend_factory=self.make_backend, model_factory=self.make_model)


def test_one_backend_two_fresh_planners_same_origin_and_actual_readonly_memory(driver, monkeypatch, tmp_path):
    wiring = Wiring(driver, monkeypatch)
    directory, summary = wiring.run(tmp_path)
    assert wiring.created == ["composite_drawers_gate_b"]
    assert [m.role for m in wiring.models] == ["outbound", "memory", "return"]
    assert all(m.close_count == 1 for m in wiring.models)
    assert wiring.backend.close_count == 1
    first, second = wiring.calls
    assert first["backend"] is second["backend"] is wiring.backend
    assert first["planner"] is not second["planner"]
    assert first["kwargs"]["odometry_origin"] == second["kwargs"]["odometry_origin"] == (.03, -.02, .11)
    assert first["kwargs"]["coordinate_frame_id"] == second["kwargs"]["coordinate_frame_id"]
    assert second["before"].position_xyz_m == [4.5, 0., .3] and second["before"].sim_time_s == 11.
    assert first["kwargs"]["store"] is second["kwargs"]["store"]
    assert second["kwargs"]["read_only"] is True and second["kwargs"]["require_memory_for_motion"] is True
    assert "memory_model" not in second["kwargs"] and "memory_notice" not in second["kwargs"]
    assert summary["mission_success"] and summary["checks"]["return_memory_unchanged"]
    assert summary["return_not_run_reason"] is None
    config = driver.read_json(directory / "experiment.json")
    assert config["budgets"]["outbound"]["max_model_calls"] == 60
    assert config["budgets"]["return"]["max_model_calls"] == 90
    assert config["budgets"]["return"]["max_wall_seconds"] == 1800.
    assert config["memory_calls"] == 48


def test_fixture_results_are_never_labelled_as_real_llm_scores(driver, monkeypatch, tmp_path):
    directory, summary = Wiring(driver, monkeypatch).run(tmp_path)
    config = driver.read_json(directory / "experiment.json")
    assert config["is_llm_result"] is False and config["model"] == "fixture-not-llm"
    assert summary["is_llm_result"] is False and summary["formal_score"] is None
    for row in summary["phases"]:
        assert row["metrics"]["is_llm_result"] is False
        assert driver.read_json(directory / row["phase"] / "metrics.json")["is_llm_result"] is False


@pytest.mark.parametrize("options,reason", [
    ({"outbound_success": False}, "outbound_failed"),
    ({"publish_memory": False}, "published_memory_unavailable"),
    ({"pending_critical": True}, "published_memory_unavailable"),
])
def test_return_is_not_invented_after_failure_or_missing_memory(driver, monkeypatch, tmp_path, options, reason):
    wiring = Wiring(driver, monkeypatch, **options)
    directory, summary = wiring.run(tmp_path)
    assert [c["phase"] for c in wiring.calls] == ["outbound"]
    assert not (directory / "return").exists()
    assert not summary["mission_success"] and summary["return_not_run_reason"] == reason
    assert wiring.evaluator_arguments is None


@pytest.mark.parametrize("options,check", [
    ({"mutate_readonly": True}, "return_memory_unchanged"),
    ({"return_history": 1}, "fresh_return_history"),
    ({"return_clock_offset": .02}, "same_simulation_clock"),
    ({"return_frame": "different-world"}, "shared_coordinate_frame"),
    ({"return_reference": "episode_start_body_horizontal"}, "shared_coordinate_frame"),
    ({"return_reads": 0}, "return_read_memory"),
])
def test_context_reset_and_readonly_violations_override_phase_success(driver, monkeypatch, tmp_path, options, check):
    _, summary = Wiring(driver, monkeypatch, **options).run(tmp_path)
    assert all(p["metrics"]["success"] for p in summary["phases"])
    assert summary["checks"][check] is False and not summary["mission_success"]


def test_source_change_after_outbound_blocks_new_planner(driver, monkeypatch, tmp_path):
    state = {"sha": "initial"}
    monkeypatch.setattr(driver, "frozen_sources", lambda: {"fixture.py": state["sha"]})
    wiring = Wiring(driver, monkeypatch, after_outbound=lambda: state.update(sha="modified"))
    _, summary = wiring.run(tmp_path)
    assert [m.role for m in wiring.models] == ["outbound", "memory"]
    assert not summary["checks"]["source_frozen"] and not summary["mission_success"]
    assert summary["error"]["type"] == "RuntimeError"


def test_source_change_during_return_cannot_be_recorded_as_success(driver, monkeypatch, tmp_path):
    state = {"sha": "initial"}
    monkeypatch.setattr(driver, "frozen_sources", lambda: {"fixture.py": state["sha"]})
    wiring = Wiring(driver, monkeypatch, after_return=lambda: state.update(sha="modified"))
    _, summary = wiring.run(tmp_path)
    assert len(summary["phases"]) == 2
    assert not summary["checks"]["source_frozen"] and not summary["mission_success"]


def test_model_construction_failure_stops_backend_and_closes_existing_model(driver, monkeypatch, tmp_path):
    wiring = Wiring(driver, monkeypatch, model_failure="memory")
    directory, summary = wiring.run(tmp_path)
    assert not wiring.calls and not summary["mission_success"]
    assert wiring.backend.stop_count == wiring.backend.close_count == 1
    assert len(wiring.models) == 1 and wiring.models[0].close_count == 1
    assert driver.read_json(directory / "summary.json")["error"]["type"] == "RuntimeError"


@pytest.mark.parametrize("phase", ["outbound", "return"])
def test_interrupted_episode_preserves_failure_and_cleans_owned_resources(driver, monkeypatch, tmp_path, phase):
    wiring = Wiring(driver, monkeypatch, episode_interrupt=phase)
    _, summary = wiring.run(tmp_path)
    assert not summary["mission_success"] and summary["error"]["type"] == "KeyboardInterrupt"
    assert wiring.backend.stop_count == wiring.backend.close_count == 1
    assert all(m.close_count == 1 for m in wiring.models)


def test_backend_cleanup_failure_overrides_success(driver, monkeypatch, tmp_path):
    wiring = Wiring(driver, monkeypatch)
    wiring.backend.close_error = True
    _, summary = wiring.run(tmp_path)
    assert not summary["mission_success"]
    assert any(e["operation"] == "backend.close" for e in summary["cleanup_errors"])


@pytest.mark.parametrize("operation", ["source", "accounting", "memory_snapshot"])
def test_postrun_audit_failure_still_writes_an_explicit_failed_summary(driver, monkeypatch, tmp_path, operation):
    def fail(*args, **kwargs):
        raise OSError("fixture postrun audit failure")

    wiring = Wiring(driver, monkeypatch)

    def inject():
        if operation == "source":
            monkeypatch.setattr(driver, "frozen_sources", fail)
        elif operation == "accounting":
            monkeypatch.setattr(driver, "usage_totals", fail)
        else:
            monkeypatch.setattr(wiring.calls[0]["kwargs"]["store"], "snapshot", fail)

    wiring.options["after_return"] = inject
    directory, summary = wiring.run(tmp_path)
    saved = driver.read_json(directory / "summary.json")
    assert not summary["mission_success"] and saved == summary
    assert wiring.backend.close_count == 1
    if operation == "accounting":
        assert summary["api_requests"] is None and summary["unknown_usage_attempts"] is None
        assert summary["usage_complete"] is False and summary["accounting_errors"]
    else:
        assert summary["cleanup_errors"]


def test_backend_startup_failure_leaves_a_non_llm_failure_record(driver, tmp_path):
    def unavailable(scene):
        raise RuntimeError("fixture backend unavailable")

    directory, summary = driver.experiment(args(tmp_path), backend_factory=unavailable,
                                          model_factory=lambda role: Model(role))
    assert not summary["mission_success"] and summary["is_llm_result"] is False
    assert summary["phases"] == [] and summary["api_requests"] == 0
    assert driver.read_json(directory / "summary.json")["error"]["type"] == "RuntimeError"


def test_accounting_preserves_unknown_attempts_and_partial_usage(driver, tmp_path):
    write_events(tmp_path / "outbound/events.jsonl", [
        {"event": "model_input"},
        {"event": "model_response", "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
        {"event": "tool_results", "usage": {"total_tokens": 999}},
        {"event": "model_input"}, {"event": "model_error"}, {"event": "failure"},
    ])
    write_events(tmp_path / "outbound/memory-agent.jsonl", [
        {"event": "model_request"}, {"event": "model_response", "usage": {"prompt_tokens": 4}},
        {"event": "model_request"},
        {"event": "model_response", "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}},
    ])
    result = driver.usage_totals(tmp_path)
    assert result["api_requests"] == 4 and result["complete_responses"] == 3
    assert result["known_usage"] == {"prompt_tokens": 34, "completion_tokens": 5, "total_tokens": 35}
    assert result["unknown_usage_attempts"] == 2 and result["usage_complete"] is False


def test_malformed_log_and_boolean_usage_do_not_appear_complete(driver, tmp_path):
    log = tmp_path / "return/events.jsonl"
    write_events(log, [{"event": "model_input"}, {"event": "model_response",
                      "usage": {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3}}])
    with log.open("a") as stream:
        stream.write('{"event":"model_input"')
    result = driver.usage_totals(tmp_path)
    assert result["known_usage"] == {"prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 3}
    assert result["unknown_usage_attempts"] == 1 and result["accounting_errors"]
    assert result["usage_complete"] is False


def test_only_one_fixture_factory_is_rejected_before_creating_artifacts(driver, tmp_path):
    with pytest.raises(ValueError, match="both"):
        driver.experiment(args(tmp_path), backend_factory=lambda scene: Backend())
    assert list(tmp_path.iterdir()) == []
