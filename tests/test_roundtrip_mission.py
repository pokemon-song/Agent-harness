"""Return acceptance uses fake measured snapshots, never model or simulator APIs."""

from copy import deepcopy
import json
import math
from types import SimpleNamespace

import pytest

from go2_demo.roundtrip_mission import (
    ReturnEvaluator, memory_notice, outbound_instruction, return_instruction,
)
from go2_demo.scenarios import get_scenario


POINTS = [(0., 0.), (.65, -.72), (1.65, -.72), (2.4, -.72), (2.9, -.12),
          (3.4, 0.), (4.3, 0.), (4.5, 0.)]


def trajectory(points, first_time=0.):
    return [{"base_xyz_m": [x, y, .34], "sim_time_s": first_time + index}
            for index, (x, y) in enumerate(points)]


def fixture(*, seeded_return=False):
    outbound = trajectory(POINTS)
    state = SimpleNamespace(position_xyz_m=[4.5, 0., .34], yaw_rad=0., velocity_xy_mps=[0., 0.],
                            yaw_rate_radps=0., fallen=False, collision_steps=0,
                            trajectory=deepcopy(outbound), sim_time_s=7.)
    evaluator = ReturnEvaluator(lambda: state, outbound, [0., 0.], [4.5, 0.], len(outbound),
                                get_scenario("composite_drawers_gate_b"))
    returning = trajectory(list(reversed(POINTS)), 7.)
    state.trajectory.extend(returning if seeded_return else returning[1:])
    state.position_xyz_m, state.sim_time_s = [0., 0., .34], 14.
    return evaluator, state, outbound


@pytest.mark.parametrize("seeded_return", [False, True])
def test_same_backend_return_scores_new_suffix_and_real_boundary(seeded_return):
    evaluator, state, outbound = fixture(seeded_return=seeded_return)
    original = deepcopy(state.trajectory)
    value = evaluator.evaluate()
    assert value["success"] and value["geometry_success"]
    assert value["same_cabinet_side"] and value["return_stages_in_order"]
    assert value["return_path_length_m"] == pytest.approx(value["outbound_path_length_m"])
    assert value["return_distance_to_start_m"] == 0.
    assert state.trajectory == original


def test_outbound_reference_is_copied_before_backend_continues():
    evaluator, state, outbound = fixture()
    outbound[1]["base_xyz_m"] = [100., 100., .34]
    assert evaluator.evaluate()["success"]


@pytest.mark.parametrize("field,value", [
    ("velocity_xy_mps", [.12, 0.]), ("velocity_xy_mps", [0., -.12]),
    ("yaw_rate_radps", .25), ("yaw_rate_radps", -.25),
    ("fallen", True), ("collision_steps", 1),
])
def test_geometric_return_does_not_override_motion_or_cumulative_safety(field, value):
    evaluator, state, _ = fixture()
    setattr(state, field, value)
    result = evaluator.evaluate()
    assert result["geometry_success"] and not result["success"]


def test_finish_orientation_is_not_a_hidden_requirement():
    evaluator, state, _ = fixture()
    state.yaw_rad = math.pi
    assert evaluator.evaluate()["success"]
    assert evaluator.public_task_spec["heading_required"] is False


def test_current_position_overrides_a_stale_successful_trace_endpoint():
    evaluator, state, _ = fixture()
    state.position_xyz_m = [.31, 0., .34]
    result = evaluator.evaluate()
    assert result["geometry_success"]
    assert not result["returned_to_start"] and not result["success"]
    assert result["return_distance_to_start_m"] == pytest.approx(.31)


def test_zero_return_motion_cannot_pass_by_reusing_outbound_history():
    outbound = trajectory(POINTS)
    state = SimpleNamespace(position_xyz_m=[4.5, 0., .34], yaw_rad=0., velocity_xy_mps=[0., 0.],
                            yaw_rate_radps=0., fallen=False, collision_steps=0,
                            trajectory=deepcopy(outbound), sim_time_s=7.)
    evaluator = ReturnEvaluator(lambda: state, outbound, [0., 0.], [4.5, 0.], len(outbound),
                                get_scenario("composite_drawers_gate_b"))
    result = evaluator.evaluate()
    assert not result["success"] and result["return_path_length_m"] == 0


def test_cumulative_trace_cannot_be_cleared_between_legs():
    evaluator, state, _ = fixture()
    state.trajectory.clear()
    with pytest.raises(ValueError, match="cleared or truncated"):
        evaluator.evaluate()


def test_public_spec_contains_no_true_route_or_goal_coordinates():
    evaluator, _, _ = fixture()
    public = evaluator.public_task_spec
    assert public["distance_tolerance_m"] == .30
    assert public["retrace_tolerance_m"] == .50
    text = json.dumps(public)
    for field in ("start_xy_m", "turnaround_xy_m", "passage_x_m", "cabinet_x_m", "trajectory", "success"):
        assert field not in text
    private = evaluator.criteria()
    assert private["start_xy_m"] == [0., 0.] and private["turnaround_xy_m"] == [4.5, 0.]
    assert private["passage_x_m"] == 3.4 and private["cabinet_x_m"] == 1.65
    assert private["cabinet_half_width_m"] == .33


def test_instructions_keep_original_scene_and_explain_fresh_agent_without_world_reset():
    scene = get_scenario("composite_drawers_gate_b")
    outbound = outbound_instruction(scene)
    assert outbound.startswith(scene.instruction)
    assert "只结束去程" in outbound and "均不重置" in outbound
    assert "0.50米" in return_instruction() and "0.30米" in return_instruction()
    assert "95%" in return_instruction() and "不要求特定终点朝向" in return_instruction()
    assert "coordinate_frame_id" in memory_notice()
    assert "不要从本任务描述虚构路线坐标或预选绕行侧" in memory_notice()
    for text in (outbound, return_instruction(), memory_notice()):
        assert "order_id" not in text and "receipt_code" not in text
        assert "1.65" not in text and "3.4" not in text


def test_boundary_requires_actual_measured_turnaround_position():
    outbound = trajectory(POINTS)
    state = SimpleNamespace(position_xyz_m=[2., 0., .34], trajectory=outbound, sim_time_s=7.)
    with pytest.raises(ValueError, match="disagrees"):
        ReturnEvaluator(lambda: state, outbound, [0., 0.], [4.5, 0.], len(outbound),
                        get_scenario("composite_drawers_gate_b"))


@pytest.mark.parametrize("field,value", [
    ("position_xyz_m", [float("nan"), 0., .34]),
    ("velocity_xy_mps", [float("inf"), 0.]),
    ("yaw_rate_radps", float("nan")),
    ("collision_steps", -1),
])
def test_invalid_current_measurement_fails_closed(field, value):
    evaluator, state, _ = fixture()
    setattr(state, field, value)
    with pytest.raises(ValueError):
        evaluator.evaluate()
