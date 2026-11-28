import math
from types import SimpleNamespace

import numpy as np

from go2_demo.fresh_exploration import StageExploration


def floor_frame():
    rows, cols = np.indices((120, 180))
    k = np.array([[105., 0., 89.5], [0., 105., 59.5], [0., 0., 1.]])
    v = (rows-k[1, 2])/k[1, 1]
    depth = np.divide(-.38, -v, out=np.full(v.shape, np.nan), where=v>.001)
    obs = {"local_odometry": {"translation_m": [0., 0.], "yaw_rad": 0.},
           "robot_state": {"body_roll_rad": 0., "body_pitch_rad": 0., "body_height_m": .30},
           "sensor_calibration": {"camera_optical_to_body_rotation": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                                  "camera_position_in_body_m": [.32, 0., .08]},
           "target_candidates": []}
    return SimpleNamespace(depth_m=depth, intrinsics=k), obs


def test_floor_proposals_are_local_reachable_stances_and_rotate_with_odometry():
    frame, obs = floor_frame()
    initial = StageExploration().observe(frame, obs, "A")
    assert initial["viewpoints"] and initial["observed_depth_edges_rad"] == []
    obs["local_odometry"] = {"translation_m": [2., -1.], "yaw_rad": math.pi/2}
    rotated = StageExploration().observe(frame, obs, "A")
    for old, new in zip(initial["viewpoints"], rotated["viewpoints"], strict=True):
        assert new["floor_support_from_m"] <= new["distance_m"] - .23
        p, q = old["waypoint"], new["waypoint"]
        assert np.allclose([2.-p["y"], -1.+p["x"]], [q["x"], q["y"]], atol=.002)
        assert math.dist([q["x"], q["y"]], [2., -1.]) > .19


def test_only_measured_floor_and_clear_path_can_produce_candidate():
    frame, obs = floor_frame()
    obs.pop("sensor_calibration")
    assert not StageExploration().observe(frame, obs, "A")["viewpoints"]
    for depth in (np.nan, .85):
        frame, obs = floor_frame()
        frame.depth_m[:] = depth
        assert not StageExploration().observe(frame, obs, "A")["viewpoints"]
    frame, obs = floor_frame()
    frame.depth_m[(frame.depth_m > .8) & (frame.depth_m < 1.22)] = np.nan
    assert not StageExploration().observe(frame, obs, "A")["viewpoints"]


def test_measured_edge_next_to_farther_floor_can_set_viewing_direction():
    frame, obs = floor_frame()
    rows, cols = np.indices(frame.depth_m.shape)
    # An extended vertical plate on the left. It stops at the centre of the image.
    hit = (cols < 90) & (rows < 74) & ((frame.depth_m > 2.5) | ~np.isfinite(frame.depth_m))
    frame.depth_m[hit] = 2.5
    result = StageExploration().observe(frame, obs, "A")
    assert result["observed_depth_edges_rad"]
    first = result["viewpoints"][0]
    assert first["depth_edge"]["measured_depth_jump_m"] >= .90
    assert first["depth_edge"]["open_side_bearing_rad"] < 0.
    assert "yaw" in first["waypoint"]


def test_repeat_detection_uses_goal_and_position_and_resets_with_new_stage():
    frame, obs = floor_frame()
    frame.depth_m[:] = np.nan
    obs["target_candidates"] = [{"task_label": "B"}]
    stage = StageExploration()
    results = [stage.observe(frame, obs, "A") for _ in range(12)]
    assert not any(r["stalled"] for r in results[:-1])
    assert results[-1]["stalled"] and not results[-1]["target_visible"]
    obs["local_odometry"]["translation_m"][0] += .50
    assert not stage.observe(frame, obs, "A")["stalled"]
    assert not StageExploration().observe(frame, obs, "B")["stalled"]
    assert StageExploration().observe(frame, obs, "B")["target_visible"]


def test_refused_candidate_does_not_immediately_reappear():
    frame, obs = floor_frame()
    stage = StageExploration()
    first = stage.observe(frame, obs, "A")["viewpoints"][0]["waypoint"]
    stage.mark_blocked(first)
    proposals = stage.observe(frame, obs, "A")["viewpoints"]
    assert all(math.dist((first["x"], first["y"]), (p["waypoint"]["x"], p["waypoint"]["y"])) >= .31
               for p in proposals)
