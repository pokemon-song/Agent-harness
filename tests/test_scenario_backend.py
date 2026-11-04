"""Independent scene scoring and geometry contracts, without a viewer or model API."""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from go2_demo.backend import MujocoGo2Backend, crossed_passage
from go2_demo.composite import evaluate_composite_trajectory
from go2_demo.scenarios import get_scenario, list_scenarios


def _trace(*points):
    return [{"base_xyz_m": [x, y, .34]} for x, y in points]


def test_passage_requires_crossing_the_opening_not_merely_finishing_beyond_it():
    around_frame = _trace((0, 0), (1, 1.4), (2, 1.4), (3, 0))
    assert not crossed_passage(around_frame, 1.5, 0.0, .65)
    through_opening = _trace((0, 0), (1.4, .2), (1.6, .1), (3, 0))
    assert crossed_passage(through_opening, 1.5, 0.0, .65)


def test_passage_interpolates_actual_gate_plane_and_requires_forward_motion():
    assert crossed_passage(_trace((1, -.8), (2, .8)), 1.5, 0.0, .1)
    assert not crossed_passage(_trace((2, 0), (1, 0)), 1.5, 0.0, .65)


def test_invalid_pose_cannot_bridge_a_gap_and_claim_passage():
    trace = _trace((1.4, 0), (float("nan"), 0), (1.6, 0))
    assert not crossed_passage(trace, 1.5, 0.0, .65)


@pytest.mark.parametrize("letter", ["A", "B", "C"])
def test_letter_panels_have_visible_front_ink_without_extra_collision_geometry(letter):
    world = ET.Element("worldbody")
    MujocoGo2Backend._add_letter(world, "panel", (3, .4, .6), (.06, .3, .4), letter)
    geoms = list(world)
    assert len(geoms) >= 3
    for geom in geoms:
        center = np.fromstring(geom.attrib["pos"], sep=" ")
        assert center[0] < 3 - .06
        assert abs(center[1] - .4) < .3
        assert abs(center[2] - .6) < .4
        assert geom.attrib["rgba"] == ".015 .015 .015 1"
        assert geom.attrib["contype"] == geom.attrib["conaffinity"] == "0"


def _evaluator(task_id, position, trajectory, *, collision=False, fallen=False):
    backend = object.__new__(MujocoGo2Backend)
    backend.scenario = get_scenario(task_id)
    backend._target_xy = np.asarray(backend.scenario.target.xy)
    backend._base_pose = lambda: (np.asarray([*position, .34]), 0., np.eye(3))
    backend.data = SimpleNamespace(qvel=np.zeros(6), time=10.)
    backend.trajectory = trajectory
    backend._collision_steps = int(collision)
    backend._fallen_ever = fallen
    return backend


def test_passage_scorer_rejects_a_route_around_the_frame_even_when_final_position_is_correct():
    backend = _evaluator("go_through_green_rectangle_tunnel", (3.2, 0),
                         _trace((0, 0), (2, 1.4), (2.5, 1.4), (3.2, 0)))
    result = backend.evaluate()
    assert result["beyond_exit"]
    assert not result["passage_crossed"]
    assert not result["success"]


@pytest.mark.parametrize("collision,fallen", [(False, False), (True, False), (False, True)])
def test_passage_scorer_uses_actual_crossing_and_keeps_collision_and_fall_requirements(collision, fallen):
    backend = _evaluator("go_through_red_triangle_tunnel", (3.2, 0),
                         _trace((0, 0), (2, .1), (2.4, .1), (3.2, 0)),
                         collision=collision, fallen=fallen)
    result = backend.evaluate()
    assert result["passage_crossed"]
    assert result["beyond_exit"]
    assert result["success"] is (not collision and not fallen)


def test_standoff_scorer_targets_the_selected_object_not_the_old_red_marker():
    target = get_scenario("go_to_blue_ball").target.xy
    backend = _evaluator("go_to_blue_ball", (target[0] - 1, target[1]), [])
    assert backend.evaluate()["success"]
    backend._target_xy = np.asarray([3.0, .55])
    assert not backend.evaluate()["success"]


@pytest.mark.parametrize("scenario", [item for item in list_scenarios() if item.supported],
                         ids=lambda item: item.task_id)
def test_each_runnable_scene_compiles_and_registers_physical_environment_collisions(scenario):
    assets = Path(__file__).resolve().parents[1] / "assets"
    if not (assets / "go2_rl_lab/resources/robots/go2/go2.xml").is_file():
        pytest.skip("Pinned robot assets are missing; run python scripts/fetch_assets.py")
    backend = object.__new__(MujocoGo2Backend)
    backend.asset_dir = assets
    backend.scenario = scenario
    backend._target_xy = np.asarray(scenario.target.xy)
    backend._environment_geom_names = []
    model = backend._make_model()
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "scenario_target")
    assert target_id >= 0
    assert model.geom_contype[target_id] != 0
    assert bool(model.geom_type[target_id] == mujoco.mjtGeom.mjGEOM_SPHERE) is (scenario.target.kind == "sphere")
    for item in scenario.objects:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, item.name)
        assert geom_id >= 0
        if item.role == "decoration":
            assert item.name not in backend._environment_geom_names
            assert model.geom_contype[geom_id] == 0
        else:
            assert item.name in backend._environment_geom_names
            assert model.geom_contype[geom_id] != 0


def test_unsupported_crawling_is_rejected_before_any_assets_or_physics_are_loaded(tmp_path):
    with pytest.raises(ValueError, match="UNSUPPORTED_CAPABILITY"):
        MujocoGo2Backend(tmp_path / "missing-assets", scenario=get_scenario("crawl_red_gate2"))


def test_public_camera_mount_comes_from_the_robot_model_without_a_world_pose():
    backend = object.__new__(MujocoGo2Backend)
    backend.asset_dir = Path(__file__).resolve().parents[1] / "assets"
    backend.scenario = get_scenario("go_through_red_triangle_tunnel")
    backend._target_xy = np.asarray(backend.scenario.target.xy)
    backend._environment_geom_names = []
    backend.model = backend._make_model()
    backend._camera_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_rgbd")
    calibration = backend.sensor_calibration
    np.testing.assert_allclose(calibration["camera_position_in_body_m"], [.32, 0, .08])
    np.testing.assert_allclose(calibration["camera_optical_to_body_rotation"],
                               [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-12)
    assert "camera_xyz_m" not in calibration and "world_pose" not in calibration


def test_declared_passage_rule_matches_existing_score_without_exposing_positions():
    backend = _evaluator("go_through_red_triangle_tunnel", (2.11924, .114),
                         _trace((0, 0), (1.6, .1), (2.11924, .114)))
    specification = backend.public_task_spec
    assert specification["required_clearance_m"] == pytest.approx(.8)
    assert specification["reference_body"] == "base"
    assert not {"passage_x", "finish_x_min", "target_xy_m", "success"}.intersection(specification)
    assert not backend.evaluate()["success"]


def _composite_path(points):
    scenario = get_scenario("composite_drawers_gate_b")
    return evaluate_composite_trajectory(
        _trace(*points), cabinet_exit_x_m=scenario.cabinet_exit_x,
        passage_x_m=scenario.passage_x, passage_center_y_m=scenario.passage_center_y,
        passage_half_width_m=scenario.passage_half_width, finish_x_min_m=scenario.finish_x_min)


def _completed_composite_trace():
    return _trace((0, 0), (1.1, .7), (2.1, .7), (2.5, .5),
                  (3.3, .2), (3.5, .1), (4.3, 0), (4.6, 0))


def test_shared_composite_checks_interpolated_order_and_complete_exit():
    result = _composite_path(((0, 0), (4.6, 0)))
    assert all(result[key] for key in ("cabinet_cleared", "passage_crossed", "beyond_exit", "stages_in_order"))
    assert [event["kind"] for event in result["events"]] == ["cabinet_clearance", "passage_crossing", "passage_exit"]
    assert [event["fraction"] for event in result["events"]] == sorted(event["fraction"] for event in result["events"])
    # This helper proves the base-point stages, not obstacle-free travel.
    backend = _evaluator("composite_drawers_gate_b", (4.6, 0), _trace((0, 0), (4.6, 0)), collision=True)
    assert not backend.evaluate()["success"]


def test_composite_requires_cabinet_clearance_before_the_valid_opening_crossing():
    result = _composite_path(((3.0, 0), (3.5, 0), (4.3, 0),
                              (2.0, 1.2), (2.5, 1.2), (4.6, 1.2), (4.6, 0)))
    assert result["cabinet_cleared"] and result["passage_crossed"] and result["beyond_exit"]
    assert not result["stages_in_order"]


@pytest.mark.parametrize("y,expected", [(-.6, True), (.6, True), (.6001, False), (-.6001, False)])
def test_composite_opening_is_checked_at_the_interpolated_base_crossing(y, expected):
    result = _composite_path(((0, y), (2.5, y), (3.5, y), (4.3, y)))
    assert result["passage_crossed"] is expected
    assert result["stages_in_order"] is expected


def test_composite_missing_or_invalid_crossing_cannot_be_inferred_from_the_final_pose():
    final_only = _composite_path(((4.3, 0), (4.6, 0)))
    assert final_only["beyond_exit"]
    assert not final_only["cabinet_cleared"] and not final_only["stages_in_order"]
    gap = _composite_path(((2.2, 0), (float("nan"), 0), (2.4, 0), (3.5, 0), (4.6, 0)))
    assert gap["passage_crossed"]
    assert not gap["cabinet_cleared"] and not gap["stages_in_order"]


def test_composite_partial_gate_crossing_followed_by_retreat_and_bypass_does_not_clear_gate_stage():
    result = _composite_path(((0, .7), (2.5, .7), (3.5, 0), (3.0, 1.2), (4.6, 1.2), (4.6, 0)))
    assert result["cabinet_cleared"] and result["passage_crossed"] and result["beyond_exit"]
    assert not result["stages_in_order"]
    retry = _composite_path(((0, .7), (2.5, .7), (3.5, 0), (3.0, .2), (3.5, .1), (4.6, 0)))
    assert retry["stages_in_order"]


def test_composite_completed_stages_can_survive_a_later_collision_free_retry():
    result = _composite_path(((0, .7), (2.5, .7), (3.5, 0), (4.3, 0),
                              (2.1, 1.2), (2.5, .7), (3.5, .1), (4.6, 0)))
    assert result["stages_in_order"] and result["beyond_exit"]


def test_composite_event_timestamps_interpolate_and_reversed_time_cannot_supply_gate_evidence():
    trace = [{**row, "sim_time_s": float(index)} for index, row in enumerate(_trace((0, 0), (4.6, 0)))]
    result = evaluate_composite_trajectory(trace, cabinet_exit_x_m=2.3, passage_x_m=3.4,
                                           passage_center_y_m=0, passage_half_width_m=.6, finish_x_min_m=4.2)
    assert result["events"][0]["sim_time_s"] == pytest.approx(.5)
    trace[-1]["sim_time_s"] = -1
    result = evaluate_composite_trajectory(trace, cabinet_exit_x_m=2.3, passage_x_m=3.4,
                                           passage_center_y_m=0, passage_half_width_m=.6, finish_x_min_m=4.2)
    assert not result["passage_crossed"] and not result["stages_in_order"]


@pytest.mark.parametrize("changes", [
    {"cabinet_exit_x_m": None}, {"cabinet_exit_x_m": True}, {"passage_half_width_m": 0},
    {"passage_half_width_m": float("nan")}, {"finish_x_min_m": 3.0},
])
def test_composite_invalid_scoring_geometry_is_rejected(changes):
    arguments = dict(cabinet_exit_x_m=2.3, passage_x_m=3.4, passage_center_y_m=0,
                     passage_half_width_m=.6, finish_x_min_m=4.2)
    arguments.update(changes)
    with pytest.raises(ValueError):
        evaluate_composite_trajectory([], **arguments)


@pytest.mark.parametrize("collision,fallen", [(False, False), (True, False), (False, True)])
def test_composite_acceptance_requires_all_three_stages_and_global_safety(collision, fallen):
    backend = _evaluator("composite_drawers_gate_b", (4.6, 0), _completed_composite_trace(),
                         collision=collision, fallen=fallen)
    result = backend.evaluate()
    assert result["stages"] == {"avoid_drawers": True, "pass_gate": True, "approach_b": True}
    assert result["stage_completion_rate"] == 1.0
    assert result["success"] is (not collision and not fallen)


def test_composite_final_b_without_route_is_partial_progress_only():
    backend = _evaluator("composite_drawers_gate_b", (4.6, 0), _trace((4.3, 0), (4.6, 0)))
    result = backend.evaluate()
    assert result["stages"] == {"avoid_drawers": False, "pass_gate": False, "approach_b": True}
    assert result["stage_completion_rate"] == pytest.approx(1 / 3)
    assert not result["success"]


@pytest.mark.parametrize("position,yaw,speed,yaw_rate", [
    ((4.6, -1.0), 0, 0, 0), ((4.6, 1.0), 0, 0, 0),
    ((4.6, 0), .251, 0, 0), ((4.6, 0), 0, .12, 0), ((4.6, 0), 0, 0, -.25),
])
def test_composite_last_stage_checks_the_correct_letter_heading_and_measured_stop(position, yaw, speed, yaw_rate):
    backend = _evaluator("composite_drawers_gate_b", position, _completed_composite_trace())
    backend._base_pose = lambda: (np.asarray([*position, .34]), yaw, np.eye(3))
    backend.data.qvel[0], backend.data.qvel[5] = speed, yaw_rate
    result = backend.evaluate()
    assert result["stages"]["avoid_drawers"] and result["stages"]["pass_gate"]
    assert not result["stages"]["approach_b"] and not result["success"]


def test_composite_public_rules_expose_stage_contract_without_world_geometry_or_evaluator_flags():
    backend = _evaluator("composite_drawers_gate_b", (4.6, 0), _completed_composite_trace())
    public = backend.public_task_spec
    assert public["goal_kind"] == "composite" and public["stages_ordered"] is True
    assert [s["stage_id"] for s in public["stage_rules"]] == ["avoid_drawers", "pass_gate", "approach_b"]
    assert public["stage_rules"][1]["required_clearance_m"] == .8
    for stage in public["stage_rules"]:
        assert set(stage) <= {"stage_id", "instruction", "required_clearance_m"}
    forbidden = {"cabinet_exit_x_m", "passage_x_m", "gate_x_m", "finish_x_min_m", "target_xy_m",
                 "base_xy_m", "success", "stages", "stage_completion_rate", "events"}
    assert not forbidden.intersection(public)
