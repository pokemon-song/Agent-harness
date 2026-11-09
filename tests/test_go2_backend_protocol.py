"""Public robot dispatch and independent task evaluation, without model APIs."""

from dataclasses import replace
import subprocess
import sys
import threading

import numpy as np
import pytest

from go2_demo.backend import MujocoGo2Backend as LegacyGo2Backend
from go2_demo.robots.contracts import RobotBackend
from go2_demo.robots.go2.backend import MujocoGo2Backend
from go2_demo.robots.go2.profile import GO2_PROFILE
from go2_demo.scenarios import get_scenario
from go2_demo.task_evaluation import TaskEvaluator, TaskEvaluationState
from go2_demo.types import ToolCall


@pytest.fixture
def adapter():
    robot = object.__new__(MujocoGo2Backend)
    robot._command_lock = threading.Lock()
    dispatched = []

    def action(name):
        def perform(**arguments):
            dispatched.append((name, arguments))
            return {"status": "succeeded", "error_code": None, "measured": {"source": name}}
        return perform

    for name in ("move_relative", "turn", "set_posture", "reset_posture", "stop"):
        setattr(robot, name, action(name))
    return robot, dispatched


def test_legacy_backend_is_the_registered_go2_adapter_and_matches_contract(adapter):
    robot, _ = adapter
    assert LegacyGo2Backend is MujocoGo2Backend
    assert isinstance(robot, RobotBackend)
    assert robot.profile is GO2_PROFILE
    assert robot.dt == GO2_PROFILE.physics_dt_s == .002
    assert robot.decimation == GO2_PROFILE.control_decimation == 10
    assert 1 / (robot.dt * robot.decimation) == GO2_PROFILE.trajectory_hz == 50


@pytest.mark.parametrize("name,arguments", [
    ("move_relative", {"forward_m": .4, "lateral_m": -.2}),
    ("turn", {"yaw_rad": .6}),
    ("set_posture", {"body_height_m": .28, "body_pitch_rad": .1}),
    ("reset_posture", {}), ("stop", {}),
])
def test_execute_dispatches_only_validated_physical_actions_and_keeps_measured_result(adapter, name, arguments):
    robot, dispatched = adapter
    result = robot.execute(ToolCall("action-1", name, arguments))
    assert dispatched == [(name, arguments)]
    assert result == {"status": "succeeded", "error_code": None, "measured": {"source": name}}
    assert "call_id" not in result and "tool" not in result


@pytest.mark.parametrize("name,arguments", [
    ("move_relative", {"forward_m": .6001, "lateral_m": 0.}),
    ("move_relative", {"forward_m": .5, "lateral_m": .5}),
    ("move_relative", {"forward_m": True, "lateral_m": 0.}),
    ("turn", {"yaw_rad": .6001}),
    ("turn", {"yaw_rad": float("nan")}),
    ("turn", {"yaw_rad": .3, "timeout_s": 20}),
    ("set_posture", {"body_height_m": .36, "body_pitch_rad": .2}),
    ("stop", {"unexpected": 1}),
    ("close", {}), ("_make_model", {}),
])
def test_execute_rejects_invalid_arguments_before_any_action(adapter, name, arguments):
    robot, dispatched = adapter
    assert robot.execute(ToolCall("rejected", name, arguments))["error_code"] == "INVALID_ARGUMENTS"
    assert dispatched == []


@pytest.mark.parametrize("name,arguments", [
    ("observe", {}), ("get_geometry", {}), ("finish", {}),
    ("abort", {"reason": "no route"}),
])
def test_execute_cannot_take_over_harness_perception_or_task_declarations(adapter, name, arguments):
    robot, dispatched = adapter
    assert robot.execute(ToolCall("harness-only", name, arguments))["error_code"] == "UNSUPPORTED_CAPABILITY"
    assert dispatched == []


def test_execute_respects_profile_capability_instead_of_legacy_method_presence(adapter):
    robot, dispatched = adapter
    robot.profile = replace(GO2_PROFILE, height_pitch_control=False)
    result = robot.execute(ToolCall("unsupported", "set_posture", {"body_height_m": .28, "body_pitch_rad": 0.}))
    assert result["error_code"] == "UNSUPPORTED_CAPABILITY"
    assert dispatched == []


def test_cancel_reports_idle_measured_stop_without_claiming_an_action_was_cancelled(adapter):
    robot, dispatched = adapter
    result = robot.cancel()
    assert result == {"status": "succeeded", "cancelled": False, "reason": "no_active_action",
                      "stop_result": {"status": "succeeded", "error_code": None, "measured": {"source": "stop"}}}
    assert dispatched == [("stop", {})]


def test_cancel_cannot_claim_idle_if_the_real_stop_fails(adapter):
    robot, _ = adapter
    robot.stop = lambda: {"status": "failed", "error_code": "STOP_NOT_CONFIRMED", "speed_mps": .2}
    result = robot.cancel()
    assert result["status"] == "failed" and result["cancelled"] is False
    assert result["error_code"] == "STOP_NOT_CONFIRMED"
    assert result["stop_result"]["speed_mps"] == .2


def test_synchronous_adapter_rejects_concurrent_dispatch_or_preemption(adapter):
    robot, dispatched = adapter
    robot._command_lock.acquire()
    try:
        assert robot.execute(ToolCall("busy", "stop", {}))["error_code"] == "BACKEND_BUSY"
        result = robot.cancel()
        assert result["error_code"] == "CANCEL_UNSUPPORTED"
        assert result["cancelled"] is False
        assert result["reason"] == "synchronous_action_in_progress"
        assert dispatched == []
    finally:
        robot._command_lock.release()


def test_interrupted_action_releases_dispatch_guard_without_swallowing_interrupt(adapter):
    robot, _ = adapter

    def interrupted(**arguments):
        raise KeyboardInterrupt("simulator interrupted")

    robot.turn = interrupted
    with pytest.raises(KeyboardInterrupt):
        robot.execute(ToolCall("interrupted", "turn", {"yaw_rad": .3}))
    assert robot.cancel()["status"] == "succeeded"


@pytest.mark.parametrize("task_id", ["go_to_blue_ball", "go_through_red_triangle_tunnel", "composite_drawers_gate_b"])
def test_public_task_rules_never_read_or_reveal_privileged_state(task_id):
    scenario = get_scenario(task_id)

    def private_snapshot():
        pytest.fail("Public rules and archived criteria must not sample live state")

    evaluator = TaskEvaluator(scenario, scenario.target.xy, private_snapshot)
    public = evaluator.public_task_spec
    archived = evaluator.criteria()
    forbidden = {"passage_x_m", "passage_center_y_m", "passage_half_width_m", "finish_x_min_m",
                 "cabinet_exit_x_m", "target_xy_m", "base_xy_m", "success", "stages", "events"}
    assert not forbidden.intersection(public)
    assert archived["requires_finish"] and archived["no_collision_or_fall"]
    if scenario.goal_kind in {"passage", "composite"}:
        assert archived["passage_x_m"] == scenario.passage_x
        assert archived["finish_x_min_m"] == scenario.finish_x_min
    if scenario.goal_kind == "passage":
        assert archived["standoff_m"] is None and archived["heading_tolerance_rad"] is None
    if scenario.goal_kind == "composite":
        assert archived["requires_stage_order"] is True


def test_independent_evaluator_observes_explicit_state_changes_and_cumulative_safety():
    scenario = get_scenario("go_to_blue_ball")
    state = TaskEvaluationState(np.array([scenario.target.xy[0] - 1, scenario.target.xy[1], .34]),
                                0., np.zeros(2), 0., False, 0, [], 10.)
    evaluator = TaskEvaluator(scenario, scenario.target.xy, lambda: state)
    assert evaluator.evaluate()["success"] is True
    state = replace(state, collision_steps=1)
    result = evaluator.evaluate()
    assert result["collision"] is True and result["success"] is False
    state = replace(state, collision_steps=0, yaw_rate_radps=.25)
    assert evaluator.evaluate()["stopped"] is False


def test_task_evaluator_and_legacy_posture_limits_do_not_import_simulator_or_policy():
    subprocess.run([sys.executable, "-c", "import sys; import go2_demo.task_evaluation; "
                    "from go2_demo.posture import validate_posture; validate_posture(.28, .1); "
                    "assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules"], check=True)
