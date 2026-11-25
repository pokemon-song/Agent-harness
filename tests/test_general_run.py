"""Offline adapter/runner safety, cleanup, and independent scoring regression tests."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from go2_demo import general_episode_run as runner
from go2_demo.general_agent import AgentLimits, GeneralAgent
from go2_demo.types import ToolCall


def run_stubs(monkeypatch, tmp_path, *, evaluation=None, termination="request_budget",
              harness_termination=None, collisions=0, fallen=False, failure=None, run_error=None,
              exceeded_budget=None):
    """Exercise the real run_episode orchestration with no simulator or API."""
    calls = []
    evaluation = evaluation or {"success": False}

    def mark(name):
        calls.append(name)
        if name == failure:
            raise OSError("injected " + name + " failure")

    class Trajectory:
        def __iter__(self):
            mark("trajectory")
            yield {"sim_time_s": 1., "base_xyz_m": [0., 0., .3]}

    class Adapter:
        def __init__(self, *args):
            self.backend = SimpleNamespace(
                data=SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1), time=1.),
                deadline_monotonic=time.monotonic() + (-1. if exceeded_budget == "wall" else 60.),
                sim_deadline_s=.5 if exceeded_budget == "simulation" else 100.,
                collision_events=[{} for _ in range(collisions)], _fallen_ever=fallen,
                trajectory=Trajectory(), stop=lambda: mark("final_stop"))
            self.harness = SimpleNamespace(termination=harness_termination)

        def evaluation(self):
            return deepcopy(evaluation)

        def feedback(self):
            return {}

        def close(self):
            mark("physical_recording")

    class Client:
        def __init__(self, role):
            self.role = role

        def close(self):
            mark(self.role + "_close")

    class Memory:
        def write(self, output):
            mark("memory_archive")

    class Agent:
        def __init__(self, *args, **kwargs):
            self.memory = Memory()

        def result(self, reason):
            return {"termination": reason, "requests": {"planner": 1}, "task_state": {
                "active": {"status": "completed", "verification": {
                    "state": "completed", "source": "local_skill_not_task_success"}}}}

        def run(self):
            if run_error:
                raise RuntimeError(run_error)
            return self.result(termination)

    class Log(runner.RunLog):
        def close(self):
            try:
                super().close()
            finally:
                mark("event_log")

    def client(**kwargs):
        return Client("planner" if kwargs["retain_context"] else "reader")

    # Temporary import stub remains valid while the independent execution module
    # is under development; monkeypatch restores the prior module afterwards.
    execution = ModuleType("go2_demo.general_execution")
    execution.GenericExecutor = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, execution.__name__, execution)
    monkeypatch.setattr(runner, "MujocoTaskAdapter", Adapter)
    monkeypatch.setattr(runner, "GeneralAgent", Agent)
    monkeypatch.setattr(runner, "RunLog", Log)
    monkeypatch.setattr(runner.MazeClient, "from_env", client)
    config = Path(__file__).resolve().parents[1] / "configs/general_agent/single_object.json"
    output = tmp_path / "episode"
    args = runner.parser().parse_args(["--task-config", str(config), "--output", str(output)])
    result = runner.run_episode(args)
    assert json.loads((output / "metrics.json").read_text()) == result
    return result, calls, output


@pytest.mark.parametrize("failure", [
    "memory_archive", "final_stop", "physical_recording", "trajectory", "planner_close", "reader_close", "event_log",
])
def test_cleanup_failure_still_attempts_every_later_cleanup_and_saves_metrics(monkeypatch, tmp_path, failure):
    result, calls, output = run_stubs(monkeypatch, tmp_path, failure=failure)
    assert calls == ["memory_archive", "final_stop", "physical_recording", "trajectory",
                     "planner_close", "reader_close", "event_log"]
    assert result["termination"] == "infrastructure_error" and result["success"] is False
    assert len(result["cleanup_errors"]) == 1
    assert "injected " + failure + " failure" in result["cleanup_errors"][0]
    assert (output / "events.jsonl").exists()


def test_cleanup_error_does_not_erase_primary_run_exception(monkeypatch, tmp_path):
    result, calls, _ = run_stubs(monkeypatch, tmp_path, failure="memory_archive", run_error="primary failure")
    assert result["error"] == "RuntimeError: primary failure"
    assert result["success"] is False and result["termination"] == "infrastructure_error"
    assert calls[-3:] == ["planner_close", "reader_close", "event_log"]
    assert result["requests"] == {"planner": 1}


@pytest.mark.parametrize("evaluation,harness_termination,termination,fallen", [
    ({"success": False}, "finish", "finish", False),
    ({"success": True}, None, "finish", False),
    ({"success": True}, "finish", "abort", False),
    ({"success": True}, "finish", "finish", True),
])
def test_local_skill_completion_cannot_override_independent_acceptance(
        monkeypatch, tmp_path, evaluation, harness_termination, termination, fallen):
    result, _, _ = run_stubs(monkeypatch, tmp_path, evaluation=evaluation,
        harness_termination=harness_termination, termination=termination, fallen=fallen)
    assert result["task_state"]["active"]["status"] == "completed"
    assert result["success"] is False and result["goal_points"] == result["score"] == 0


@pytest.mark.parametrize("passed,evaluator_success,termination,expected_points,expected_success", [
    ((True, False), False, "request_budget", 50., False),
    ((True, True), True, "request_budget", 50., False),
    ((True, True), True, "finish", 100., True),
    ((False, False), False, "finish", 0., False),
])
def test_ordered_partial_credit_and_collision_penalty_remain_independent(
        monkeypatch, tmp_path, passed, evaluator_success, termination, expected_points, expected_success):
    evaluation = {"success": evaluator_success, "checkpoints": {
        "checkpoint_one": {"passed": passed[0]}, "checkpoint_two": {"passed": passed[1]}},
        "checkpoint_order": ["checkpoint_one", "checkpoint_two"]}
    result, _, output = run_stubs(monkeypatch, tmp_path, evaluation=evaluation, termination=termination,
        harness_termination="finish" if termination == "finish" else None, collisions=2)
    assert result["success"] is expected_success
    assert result["goal_points"] == expected_points
    assert result["collision_count"] == 2 and result["collision_penalty"] == 10
    assert result["score"] == max(0., expected_points - 10)
    assert json.loads((output / "evaluation.evaluator-only.json").read_text()) == evaluation


def test_fall_does_not_issue_cleanup_motion(monkeypatch, tmp_path):
    result, calls, _ = run_stubs(monkeypatch, tmp_path, fallen=True)
    assert result["success"] is False
    assert "final_stop" not in calls and "physical_recording" in calls


@pytest.mark.parametrize("budget", ["wall", "simulation"])
def test_independent_success_after_finish_cannot_override_exceeded_budget(monkeypatch, tmp_path, budget):
    result, _, _ = run_stubs(monkeypatch, tmp_path, evaluation={"success": True},
        termination="finish", harness_termination="finish", exceeded_budget=budget)
    assert result["termination"] == "budget_exhausted" and result["success"] is False
    assert result["goal_points"] == result["score"] == 0


def test_finish_request_cannot_override_executor_budget_failure():
    """Stopping can cross a deadline even after the declaration was dispatched."""
    feedbacks = []

    class Planner:
        raw_usage = None

        def choose(self, *args, **kwargs):
            self.raw_usage = {"prompt_tokens": 1, "completion_tokens": 1}
            return ToolCall("finish-at-boundary", "finish", {"note": "stop now"})

        def finish(self, call, feedback):
            feedbacks.append(feedback)

        def context_stats(self):
            return {}

    class Executor:
        harness = SimpleNamespace(frame=SimpleNamespace(rgb=np.zeros((2, 2, 3), np.uint8)), termination=None)

        def observation(self):
            return {"image_id": "frame-1", "local_odometry": {"translation_m": [0., 0.], "yaw_rad": 0.},
                    "grounded_targets": [], "robot_state": {"fallen": False}}

        def finish(self, name, reason=None):
            self.harness.termination = "finish"
            return {"status": "budget_exhausted", "error_code": "simulation_time_limit"}

    task = {"id": "deadline-boundary", "instruction": "finish when accepted", "targets": [], "requirements": "test"}
    agent = GeneralAgent(task, Planner(), Executor(), limits=AgentLimits(max_requests=1),
                         feedback_provider=lambda: {"task_completed": True})
    result = agent.run()
    assert result["termination"] == "budget_exhausted"
    assert feedbacks[0]["status"] == "budget_exhausted"


@dataclass
class Scene:
    collision_penalty_points: int = 0
    floor_half_size_m: tuple = (8., 6.)


def install_constructor_stubs(monkeypatch, failure):
    calls = []

    class Backend:
        def __init__(self, *args, **kwargs):
            calls.append("backend_created")
            self.data = SimpleNamespace(time=0.)

        def evaluate(self):
            if failure == "evaluation":
                raise RuntimeError("injected evaluation failure")
            return {"success": False, "checkpoints": {"unexpected": {"passed": False}}}

        def close(self):
            calls.append("backend_closed")

    def harness(*args, **kwargs):
        if failure == "harness":
            raise RuntimeError("injected harness failure")
        return SimpleNamespace(termination=None)

    class Recording:
        def __init__(self, *args, **kwargs):
            if failure == "recording":
                raise RuntimeError("injected recording failure")
            calls.append("recording_created")

        def finish(self):
            calls.append("recording_closed")

    for name, values in {
        "backend": {"MujocoGo2Backend": Backend}, "harness": {"Harness": harness},
        "occluded_video": {"OccludedRecording": Recording}, "scenarios": {"get_scenario": lambda name: Scene()},
    }.items():
        module = ModuleType("go2_demo." + name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.mark.parametrize("failure", ["harness", "recording", "evaluation", "checkpoint_mapping"])
def test_adapter_constructor_failure_closes_allocated_resources(monkeypatch, tmp_path, failure):
    calls = install_constructor_stubs(monkeypatch, failure)
    config = {"task": {"instruction": "test"}, "environment": {"scenario": "test"}}
    with pytest.raises((ValueError, RuntimeError)):
        runner.MujocoTaskAdapter(config, tmp_path, 0, tmp_path, 10., 10.)
    assert calls[0] == "backend_created" and calls[-1] == "backend_closed"
    if "recording_created" in calls:
        assert calls.index("recording_closed") < calls.index("backend_closed")


def test_recording_export_failure_is_not_retried_before_backend_resource_release():
    calls = []

    class Recording:
        def finish(self):
            calls.append("recording_export")
            raise OSError("recording export failed")

    class Backend:
        def close(self):
            # Match the legacy backend's ordering: an attached recording is
            # finished before its renderer and viewer are released.
            if self._recording is not None:
                self._recording.finish()
            calls.append("resources_released")

    adapter = object.__new__(runner.MujocoTaskAdapter)
    adapter.recorder = Recording()
    adapter.backend = Backend()
    adapter.backend._recording = adapter.recorder
    with pytest.raises(OSError, match="recording export failed"):
        adapter.close()
    assert calls == ["recording_export", "resources_released"]
    assert adapter.backend._recording is None


def test_constructor_cleanup_failure_preserves_original_initialization_exception(monkeypatch, tmp_path):
    calls = install_constructor_stubs(monkeypatch, "harness")
    backend_type = sys.modules["go2_demo.backend"].MujocoGo2Backend

    def failing_close(self):
        calls.append("backend_close_attempted")
        raise OSError("secondary cleanup failure")

    monkeypatch.setattr(backend_type, "close", failing_close)
    config = {"task": {"instruction": "test"}, "environment": {"scenario": "test"}}
    with pytest.raises(RuntimeError, match="injected harness failure") as caught:
        runner.MujocoTaskAdapter(config, tmp_path, 0, tmp_path, 10., 10.)
    assert calls == ["backend_created", "backend_close_attempted"]
    assert any("secondary cleanup failure" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("overrides,expected", [
    ({}, True), ({"distance_error_m": .19}, False), ({"heading_error_rad": .26}, False),
    ({"stopped": False}, False), ({"fallen": True}, False),
])
def test_single_target_collision_policy_preserves_measured_acceptance(overrides, expected):
    source = {"success": False, "goal_kind": "standoff", "collision": True,
              "distance_error_m": .10, "distance_tolerance_m": .18,
              "heading_error_rad": .05, "stopped": True, "fallen": False, **overrides}
    adapter = object.__new__(runner.MujocoTaskAdapter)
    adapter.backend = SimpleNamespace(evaluate=lambda: source)
    result = adapter.evaluation()
    assert result["success"] is expected
    assert source["success"] is False  # Never mutates the independent raw sample.


def test_ordered_evaluation_is_retained_without_single_target_reinterpretation():
    source = {"success": False, "checkpoints": {"arbitrary": {"passed": True, "private_xy": [7., 9.]}},
              "checkpoint_order": ["arbitrary"], "stopped": True, "fallen": False}
    adapter = object.__new__(runner.MujocoTaskAdapter)
    adapter.backend = SimpleNamespace(evaluate=lambda: source)
    assert adapter.evaluation() == source
    adapter.config = {"task": {"targets": [{"target_id": "target"}]},
                      "environment": {"checkpoint_target_ids": {"arbitrary": "target"}}}
    assert adapter.feedback() == {"milestones": [{"target_id": "target", "passed": True}], "task_completed": False}


def test_runner_defaults_to_measured_subgoal_updates_and_records_policy(monkeypatch, tmp_path):
    _, _, output = run_stubs(monkeypatch, tmp_path)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["subgoal_updates"] == "measured"
    assert protocol["subgoal_judgment_source"] == "measured_rules"
    assert protocol["route_progress_source"] == "executor_measured_prefix"
    assert protocol["task_pass_source"] == "independent_environment_evaluator"
    assert protocol["max_consecutive_errors"] == 0
    assert protocol["error_count_termination_enabled"] is False
    args = runner.parser().parse_args(["--task-config", "unused", "--subgoal-updates", "agent"])
    assert args.subgoal_updates == "agent"
    assert args.max_consecutive_errors == 0
    args = runner.parser().parse_args(["--task-config", "unused", "--max-consecutive-errors", "6"])
    assert args.max_consecutive_errors == 6
