"""Public geometry, real Harness/PointTracker and bounded execution regressions."""
from copy import deepcopy
import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from go2_demo.general_execution import GenericExecutor
from go2_demo.harness import Harness
from go2_demo.types import SensorFrame


class SyntheticBackend:
    """A textured static plane; no semantic detector or task scorer."""
    def __init__(self, plane_x=2.):
        self.pose = np.zeros(3)
        self.sim_time = 0.
        self.plane_x = plane_x
        self.frame_number = 0
        self.rgb = np.random.default_rng(19).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        self.depth_invalid = False
        self.fallen = False
        self.fall_on_stop = False
        self.stop_failed = False
        self.stop_drift = 0.
        self.turn_drift = 0.
        self.move_gain = 1.
        self.move_status = "succeeded"
        self.moves, self.turns = [], []
        self.stops = 0
        self.deadline_monotonic = None
        self.sim_deadline_s = None
        self.collision_terminated = False
        self.collision_reported = False
        self.sensor_calibration = {
            "camera_position_in_body_m": [0., 0., 0.],
            "camera_optical_to_body_rotation": [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]],
        }
        self.public_task_spec = {"standoff_m": 1., "target_position": "PRIVATE_SENTINEL"}

    def capture(self):
        self.frame_number += 1
        yaw = self.pose[2]
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.],
                             [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
        transform = np.eye(4)
        transform[:3, :3] = rotation @ np.asarray(self.sensor_calibration["camera_optical_to_body_rotation"])
        transform[:3, 3] = [*self.pose[:2], .4]
        depth = float("nan") if self.depth_invalid else self.plane_x - self.pose[0]
        return SensorFrame(
            frame_id=f"image-{self.frame_number}", sim_time_s=self.sim_time,
            rgb=self.rgb.copy(), depth_m=np.full((80, 100), depth, np.float32),
            intrinsics=np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]]),
            camera_to_world=transform, base_xy_yaw=self.pose.copy(),
            robot_state={"speed_mps": 0., "yaw_rate_radps": 0., "fallen": self.fallen,
                         "body_height_m": .4, "body_roll_rad": 0., "body_pitch_rad": 0.,
                         "target_position": "PRIVATE_SENTINEL"})

    def move_relative(self, forward_m, lateral_m):
        self.moves.append((forward_m, lateral_m))
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        motion = self.move_gain * np.array([forward_m, lateral_m])
        self.pose[:2] += np.array([[c, -s], [s, c]]) @ motion
        self.sim_time += 1.
        return {"status": self.move_status, "error_code": None if self.move_status == "succeeded" else "MOTION_TIMEOUT",
                "collision": self.collision_reported, "hidden_rule": "PRIVATE_SENTINEL",
                "measured": {"translation_start_frame_m": motion.tolist(), "private": "PRIVATE_SENTINEL"},
                "stop": {"status": "succeeded", "private": "PRIVATE_SENTINEL"}}

    def turn(self, yaw_rad):
        self.turns.append(yaw_rad)
        self.pose[2] = math.remainder(self.pose[2] + yaw_rad, math.tau)
        self.pose[0] += self.turn_drift
        self.turn_drift = 0.
        self.sim_time += 1.
        return {"status": "succeeded"}

    def stop(self):
        self.stops += 1
        self.sim_time += .6
        self.pose[0] += self.stop_drift
        self.stop_drift = 0.
        self.fallen = self.fallen or self.fall_on_stop
        return {"status": "failed" if self.stop_failed or self.fallen else "succeeded",
                "error_code": "STOP_NOT_CONFIRMED" if self.stop_failed else None,
                "speed_mps": .5 if self.stop_failed else 0., "yaw_rate_radps": 0.}


def make_executor(**kwargs):
    backend = SyntheticBackend()
    harness = Harness(backend, "Visit the described object; semantic identity is model-proposed.")
    executor = GenericExecutor(harness, kwargs.pop("target_ids", ["parcel", "marker"]), **kwargs)
    return executor, harness, backend


def ground(executor, target_id="parcel"):
    return executor.ground_target(target_id, executor.observation()["image_id"], 50., 40.)


def route(*points):
    return {"waypoints": [{"x": x, "y": 0.} for x in points], "after": "inspect"}


def test_actual_point_tracker_binds_model_label_without_semantic_certification():
    events = []
    executor, harness, backend = make_executor(on_action=events.append)
    result = ground(executor)
    assert result["status"] == "ok" and result["native_actions"] == 1
    assert result["geometry_only"] is True and result["identity_basis"] == "model_proposed"
    assert result["grounding"]["relative_xy_m"] == pytest.approx([2., 0.])
    assert result["grounding"]["range_m"] == 2.
    assert set(harness.tracker.points) == {result["grounding"]["point_id"]}
    assert events[0]["after"]["grounded_targets"][0]["target_id"] == "parcel"
    assert "semantic_verified" not in json.dumps(result)
    assert backend.moves == [] and backend.stops == 0
    # Caller mutation cannot alter the persistent association or recorded events.
    result["grounding"]["target_id"] = "invented"
    events[0]["after"]["grounded_targets"].clear()
    assert executor.observation()["grounded_targets"][0]["target_id"] == "parcel"
    assert executor.actions[0]["after"]["grounded_targets"]


@pytest.mark.parametrize("bad,code", [("unknown", "UNKNOWN_TARGET_ID"),
                                     ("stale", "STALE_IMAGE"), ("depth", "INVALID_DEPTH"),
                                     ("pixel", "PIXEL_OUT_OF_BOUNDS")])
def test_bad_groundings_do_not_move_or_create_a_target(bad, code):
    executor, harness, backend = make_executor()
    image_id = executor.observation()["image_id"]
    if bad == "stale":
        harness.refresh()
    if bad == "depth":
        backend.depth_invalid = True
        harness.refresh()
        image_id = executor.observation()["image_id"]
    result = executor.ground_target("unknown" if bad == "unknown" else "parcel", image_id,
                                    -1. if bad == "pixel" else 50., 40.)
    assert result["status"] == "replan" and result["error_code"] == code
    assert executor.observation()["grounded_targets"] == []
    assert not harness.tracker.points and backend.moves == [] and backend.turns == []


def test_same_target_can_be_regrounded_repeatedly_without_exhausting_point_slots():
    executor, harness, _ = make_executor()
    other = ground(executor, "marker")["grounding"]["point_id"]
    for _ in range(12):
        assert ground(executor)["status"] == "ok"
    assert len(harness.tracker.points) == 2 and other in harness.tracker.points
    assert executor.native_count == 13
    assert len({e["call_id"] for e in executor.actions}) == 13


def test_active_point_capacity_is_explicit_and_does_not_erase_other_targets():
    executor, harness, _ = make_executor(target_ids=[f"item-{n}" for n in range(9)])
    for n in range(8):
        assert ground(executor, f"item-{n}")["status"] == "ok"
    result = ground(executor, "item-8")
    assert result["error_code"] == "TRACK_CAPACITY_REACHED"
    assert len(harness.tracker.points) == 8
    assert len(executor.observation()["grounded_targets"]) == 8


def test_long_public_target_id_matches_task_contract():
    target = "T" * 256
    executor, _, _ = make_executor(target_ids=[target])
    assert ground(executor, target)["target_id"] == target


def test_lost_point_nulls_geometry_and_approach_never_searches_or_blindly_moves():
    executor, harness, backend = make_executor()
    ground(executor)
    backend.depth_invalid = True
    harness.refresh()
    point = executor.observation()["grounded_targets"][0]
    assert not point["valid"]
    assert all(point[k] is None for k in ("pixel_uv", "relative_xy_m", "range_m", "bearing_rad"))
    result = executor.approach("parcel", 1.)
    assert result["error_code"] == "TARGET_GROUNDING_REQUIRED" and result["native_actions"] == 0
    assert backend.moves == [] and backend.turns == [] and backend.stops == 0


def test_approach_slices_keep_global_budget_and_complete_only_a_surface_skill():
    executor, harness, backend = make_executor(actions_per_slice=1, max_native_actions=8)
    ground(executor)
    first = executor.approach("parcel", 1.)
    second = executor.approach("parcel", 1.)
    third = executor.approach("parcel", 1.)
    assert [first["error_code"], second["error_code"]] == ["execution_slice_complete"] * 2
    assert [first["total_native_actions"], second["total_native_actions"], third["total_native_actions"]] == [2, 3, 4]
    assert third["status"] == "skill_completed" and third["skill"] == "approach_target"
    assert third["reference"] == "tracked_surface_point" and third["geometry_only"]
    assert third["measured_range_m"] == pytest.approx(1., abs=.001)
    assert backend.stops == 1 and harness.termination is None
    assert "finish" not in [e["tool"] for e in executor.actions]
    assert max(math.hypot(*move) for move in backend.moves) <= .53


def test_stop_is_remeasured_before_skill_completion():
    executor, harness, backend = make_executor(actions_per_slice=1)
    backend.pose[0] = 1.
    harness.refresh()
    ground(executor)
    backend.stop_drift = .15
    result = executor.approach("parcel", 1.)
    assert result["status"] == "replan" and result["error_code"] == "execution_slice_complete"
    assert result["measured_displacement_m"] == pytest.approx(.15)
    assert harness.termination is None


def test_loss_during_stop_does_not_count_as_completed(monkeypatch):
    executor, harness, backend = make_executor()
    ground(executor)
    original = backend.stop

    def lose_depth():
        backend.depth_invalid = True
        return original()

    monkeypatch.setattr(backend, "stop", lose_depth)
    result = executor.approach("parcel", 2.)
    assert result["status"] == "replan" and result["error_code"] == "TARGET_GROUNDING_REQUIRED"
    assert backend.stops == 1 and harness.termination is None


def test_routes_report_only_measured_completed_prefixes_across_slices():
    executor, _, backend = make_executor(actions_per_slice=2, max_native_actions=3)
    first = executor.navigate("follow_waypoints", route(.53, 1.06, 1.59))
    assert first["error_code"] == "execution_slice_complete"
    assert first["completed_waypoints"] == 2 and first["native_actions"] == 2
    second = executor.navigate("follow_waypoints", route(1.59))
    assert second["status"] == "ok" and second["completed_waypoints"] == 1
    assert second["total_native_actions"] == 3
    exhausted = executor.navigate("scan", {"yaw_rad": .2})
    assert exhausted["status"] == "budget_exhausted" and exhausted["native_actions"] == 0
    assert backend.turns == []


def test_requested_travel_and_successful_tool_reply_do_not_prove_a_waypoint():
    executor, _, backend = make_executor(actions_per_slice=1)
    backend.move_gain = .2
    result = executor.navigate("follow_waypoints", route(.53))
    assert result["completed_waypoints"] == 0
    assert result["measured_displacement_m"] == pytest.approx(.106)
    assert result["error_code"] == "execution_slice_complete"


def test_settled_partial_move_is_remeasured_not_counted_as_success():
    executor, _, backend = make_executor(actions_per_slice=1)
    backend.move_gain, backend.move_status = .2, "timed_out"
    result = executor.navigate("follow_waypoints", route(.53))
    assert result["completed_waypoints"] == 0 and result["error_code"] == "execution_slice_complete"
    assert executor.actions[0]["reply"]["error_code"] == "MOTION_TIMEOUT"


def test_turn_drift_cannot_confirm_a_waypoint_from_two_different_poses():
    executor, _, backend = make_executor(actions_per_slice=1)
    backend.turn_drift = .3
    result = executor.navigate("follow_waypoints", {"waypoints": [{"x": 0., "y": 0., "yaw": .5}], "after": "inspect"})
    assert result["completed_waypoints"] == 0 and result["error_code"] == "execution_slice_complete"
    assert result["current_odometry"]["translation_m"] == pytest.approx([.3, 0.])


def test_endpoint_resume_rechecks_pose_and_does_not_count_the_waypoint_twice():
    executor, _, _ = make_executor()
    result = executor.navigate("follow_waypoints", {"waypoints": [{"x": 0., "y": 0., "yaw": .4}], "after": "inspect"}, resume_endpoint=True)
    assert result["status"] == "ok" and result["completed_waypoints"] == 0
    assert result["native_actions"] == 1


@pytest.mark.parametrize("after", ["dock", "finish", "search"])
def test_routes_cannot_request_implicit_docking_or_search(after):
    executor, harness, backend = make_executor()
    result = executor.navigate("follow_waypoints", {"waypoints": [{"x": 1., "y": 0.}], "after": after})
    assert result["error_code"] == "INVALID_NAVIGATION_ARGUMENTS"
    assert not executor.actions and backend.moves == [] and harness.termination is None


def test_visible_depth_obstacle_blocks_translation_without_choosing_another_path():
    executor, harness, backend = make_executor()
    backend.plane_x = .55
    harness.refresh()
    result = executor.navigate("follow_waypoints", route(.53))
    assert result["error_code"] == "current_depth_blocks_requested_translation"
    assert result["depth_feedback"]["error_code"] == "VISIBLE_OBSTACLE_IN_STEP"
    assert backend.moves == [] and backend.turns == [] and result["native_actions"] == 0


def test_missing_depth_cannot_be_interpreted_as_free_space():
    executor, harness, backend = make_executor()
    backend.depth_invalid = True
    harness.refresh()
    result = executor.navigate("follow_waypoints", route(.53))
    assert result["depth_feedback"]["error_code"] == "LOCAL_DEPTH_UNAVAILABLE"
    assert backend.moves == []


def test_collision_feedback_is_recorded_without_stopping_a_nonterminal_environment():
    executor, _, backend = make_executor()
    backend.collision_reported = True
    assert executor.navigate("follow_waypoints", route(.53))["status"] == "ok"
    assert executor.actions[0]["reply"]["collision"] is True


def test_budget_prevents_any_terminal_or_ground_dispatch():
    executor, harness, backend = make_executor(max_native_actions=0)
    assert ground(executor)["error_code"] == "native_action_limit"
    assert executor.finish()["error_code"] == "native_action_limit"
    assert not executor.actions and backend.stops == 0 and harness.termination is None


@pytest.mark.parametrize("limit", ["wall", "simulation"])
def test_time_budget_is_global_and_checked_before_each_operation(limit):
    executor, _, backend = make_executor()
    if limit == "wall":
        backend.deadline_monotonic = 0.
    else:
        backend.sim_deadline_s = 0.
    assert executor.navigate("scan", {"yaw_rad": .2})["error_code"] == f"{limit}_time_limit"
    assert not executor.actions


def test_constructor_never_extends_an_existing_environment_budget():
    backend = SyntheticBackend()
    backend.deadline_monotonic, backend.sim_deadline_s = 5., 7.
    harness = Harness(backend, "General inspection")
    GenericExecutor(harness, [], max_wall_seconds=100., max_sim_seconds=100.)
    assert backend.deadline_monotonic == 5. and backend.sim_deadline_s == 7.


@pytest.mark.parametrize("name", ["finish", "abort"])
def test_terminal_stop_over_deadline_does_not_report_success(name):
    executor, harness, backend = make_executor()
    backend.sim_deadline_s = .5
    result = executor.finish(name, reason="requested end")
    assert result["status"] == "budget_exhausted" and result["error_code"] == "simulation_time_limit"
    assert result["native_actions"] == 1 and backend.stops == 1
    assert harness.termination == name


def test_fall_during_terminal_stop_is_not_finish_success():
    executor, _, backend = make_executor()
    backend.fall_on_stop = True
    result = executor.finish()
    assert result["status"] == "fall" and result["error_code"] == "ROBOT_FELL"


def test_failed_terminal_stop_reports_ended_control_error_instead_of_replan():
    executor, harness, backend = make_executor()
    backend.stop_failed = True
    result = executor.finish()
    assert result["status"] == "control_error" and result["error_code"] == "STOP_NOT_CONFIRMED"
    assert result["episode_ended"] is True and harness.termination == "finish"
    assert executor.navigate("scan", {"yaw_rad": .1})["error_code"] == "EPISODE_ENDED"


def test_finish_is_an_explicit_declaration_separate_from_skill_success():
    executor, harness, backend = make_executor()
    ground(executor)
    assert executor.approach("parcel", 2.)["status"] == "skill_completed"
    assert harness.termination is None
    result = executor.finish()
    assert result["status"] == "finish" and harness.termination == "finish"
    assert backend.stops == 2 and "success" not in result


def test_call_ids_remain_unique_if_a_second_executor_shares_the_harness():
    first, harness, backend = make_executor()
    second = GenericExecutor(harness, [])
    first.navigate("scan", {"yaw_rad": .2})
    second.navigate("scan", {"yaw_rad": .2})
    assert first.actions[0]["call_id"] != second.actions[0]["call_id"]
    assert backend.turns == [.2, .2]


def test_public_observations_and_callbacks_recursively_drop_backend_metadata(monkeypatch):
    events = []
    executor, harness, _ = make_executor(on_action=events.append)
    original = harness.observation

    def contaminated():
        observation = deepcopy(original())
        observation["hidden_rules"] = "PRIVATE_SENTINEL"
        observation["local_odometry"]["hidden_world_pose"] = "PRIVATE_SENTINEL"
        observation["robot_state"]["expected_station"] = "PRIVATE_SENTINEL"
        observation["sensor_calibration"]["target_position"] = "PRIVATE_SENTINEL"
        for point in observation["tracked_points"]:
            point["world"] = "PRIVATE_SENTINEL"
        return observation

    monkeypatch.setattr(harness, "observation", contaminated)
    ground(executor)
    executor.navigate("follow_waypoints", route(.53))
    serialized = json.dumps([executor.observation(), events], allow_nan=False)
    assert "PRIVATE_SENTINEL" not in serialized
    assert '"world"' not in serialized and '"hidden_rules"' not in serialized


def test_executor_does_not_import_task_detectors_planners_or_scoring():
    path = Path(__file__).parents[1] / "go2_demo/general_execution.py"
    tree = ast.parse(path.read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    for module in imports:
        assert not any(token in (module or "") for token in ("fresh", "memory_perception", "return_route", "paired", "evaluation", "scenarios"))
