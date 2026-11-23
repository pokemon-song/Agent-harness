import math

import pytest

from go2_demo.fresh_subgoals import StageSubgoals, validate_goal
from go2_demo.models import ModelProtocolError


def obs(x=0., y=0., yaw=0., target=None, frame="now"):
    return {"image_id": frame, "sim_time_s": 1.,
            "local_odometry": {"translation_m": [x, y], "yaw_rad": yaw},
            "target_candidates": [] if target is None else [{"task_label": target}],
            "secret_evaluator": "hidden coordinates must never enter state"}


def goal(criterion="dock_target", **extra):
    return {"description": "continue approaching the stage target", "criterion": criterion, **extra}


def test_goal_completion_requires_measured_pose_and_requested_heading():
    state = StageSubgoals("A")
    state.set_goal(goal("reach_pose", target_pose={"x": 1., "y": 2., "yaw": math.pi - .03}), obs())
    state.observe(obs(1., 2., 0.))
    assert state.active["status"] == "active"
    snapshot = state.observe(obs(1.1, 2., -math.pi + .02))
    assert snapshot["active"]["status"] == "completed"
    assert snapshot["active"]["verification"]["source"] == "public_odometry"
    assert "hidden coordinates" not in str(snapshot)


@pytest.mark.parametrize("arguments", [
    goal("reach_pose"), goal("reach_pose", target_pose={"x": True, "y": 0.}),
    goal("reach_pose", target_pose={"x": float("nan"), "y": 0.}),
    goal("reach_pose", target_pose={"x": 0., "y": 0., "id": "private"}),
    goal("observe_target", target_pose={"x": 0., "y": 0.}),
    goal("finish"), {"description": " ", "criterion": "dock_target"},
    goal(reason=""), goal(secret_evaluator="leak"),
])
def test_invalid_goal_rejected_before_execution(arguments):
    with pytest.raises(ModelProtocolError):
        validate_goal(arguments)


def test_target_sighting_survives_occlusion_without_completing_dock():
    state = StageSubgoals("B")
    state.set_goal(goal(), obs())
    sight = obs(2., .5, .2, target="B", frame="seen-B")
    sight["target_candidates"][0].update(pixel_uv=[30, 40], camera_xyz_m=[.2, 0., 3.],
                                           coordinate_frame="current_camera_optical", secret_truth="hidden")
    state.observe(sight)
    later = state.observe(obs(2.1, .5, .5, target="A", frame="occluded"))
    assert later["active"]["status"] == "active"
    assert later["current_target_visible"] is False
    assert later["last_target_sighting"]["image_id"] == "seen-B"
    assert later["last_target_sighting"]["candidate"]["camera_xyz_m"] == [.2, 0., 3.]
    assert "secret_truth" not in str(later)
    state.set_goal(goal("observe_target", reason="inspect from current stance"), obs(target="A"))
    assert state.active["status"] == "active"
    assert state.observe(obs(target="B"))["active"]["status"] == "completed"


def test_replacement_needs_reason_but_does_not_erase_stuck_state():
    state = StageSubgoals("B")
    state.set_goal(goal(), obs())
    feedback = {"status": "replan", "error_code": "current_depth_blocks_requested_translation"}
    state.record_action("follow_waypoints", {}, obs(), obs(), feedback, "approach B")
    with pytest.raises(ModelProtocolError):
        state.set_goal(goal("observe_target"), obs())
    state.set_goal(goal("observe_target", reason="need another view of B"), obs())
    assert state.recovery_required
    assert state.active["status"] == "recovering"
    assert state.recent_goals[-1]["status"] == "replaced"
    assert state.recent_goals[-1]["replacement_reason"] == "need another view of B"


def test_recovery_retains_goal_and_clearance_failure_then_real_detour_makes_progress():
    state = StageSubgoals("B")
    state.set_goal(goal(), obs(target="B"))
    original_id = state.active["id"]
    state.record_action("follow_waypoints", {}, obs(target="B"), obs(),
                        {"status": "replan", "error_code": "current_depth_blocks_requested_translation"}, "dock B")
    assert state.recovery_required and state.active["id"] == original_id
    # Queries and repeated observations are not new physical evidence.
    state.record_action("memory_query", {}, obs(), obs(), {"status": "ok"})
    for _ in range(10):
        state.observe(obs())
    assert state.action_count == 1 and state.recovery_required
    after = state.record_action("follow_waypoints", {}, obs(), obs(.5), {"status": "ok"},
                                "move out of blocked approach", recovery=True)
    assert not after["recovery_required"]
    assert after["active"]["id"] == original_id and after["active"]["status"] == "active"
    assert after["last_action"]["intention"] == "move out of blocked approach"


def test_rotation_oscillation_requests_recovery_but_explicit_heading_progress_counts():
    state = StageSubgoals("A")
    state.set_goal(goal("observe_target"), obs())
    for index, (before, after) in enumerate(((0., .5), (.5, 0.), (0., .5))):
        state.record_action("scan", {"yaw_rad": after-before}, obs(yaw=before), obs(yaw=after), {"status": "ok"})
    assert state.recovery_required and state.no_progress_actions == 3
    reach = StageSubgoals("A")
    reach.set_goal(goal("reach_pose", target_pose={"x": 0., "y": 0., "yaw": 1.}), obs())
    result = reach.record_action("scan", {"yaw_rad": .5}, obs(), obs(yaw=.5), {"status": "ok"})
    assert result["last_action"]["made_progress"] and not result["recovery_required"]
    assert result["active"]["status"] == "active"


def test_verified_route_prefix_survives_scan_and_recovery_detour():
    state = StageSubgoals("B")
    state.set_goal(goal(), obs())
    points = [{"x": 1., "y": 0.}, {"x": 2., "y": 0.}, {"x": 3., "y": 0.}]
    arguments = {"waypoints": points, "after": "dock"}
    state.note_route(arguments)
    result = state.record_action("follow_waypoints", arguments, obs(), obs(1.5),
                                 {"status": "replan", "error_code": "execution_slice_complete", "completed_waypoints": 1})
    assert not result["recovery_required"]
    assert state.pending_route["waypoints"] == points[1:]
    assert state.pending_route["completed_waypoints"] == 1
    detour = {"waypoints": [{"x": 1.5, "y": .5}], "after": "inspect"}
    state.note_route(detour, recovery=True)
    state.record_action("follow_waypoints", detour, obs(1.5), obs(1.5, .5),
                        {"status": "ok", "completed_waypoints": 1}, recovery=True)
    state.record_action("scan", {"yaw_rad": .5}, obs(1.5, .5), obs(1.5, .5, .5), {"status": "ok"})
    assert state.pending_route["waypoints"] == points[1:]
    # A candidate seen mid-route is not a waypoint completion claim.
    continuation = {"waypoints": points[1:], "after": "dock"}
    state.record_action("follow_waypoints", continuation, obs(1.5, .5), obs(1.7, .5, target="B"),
                        {"status": "replan", "error_code": "target_visible_in_current_view", "completed_waypoints": 0})
    assert state.pending_route["completed_waypoints"] == 1


@pytest.mark.parametrize("after,endpoint", [("dock", {"x": 1., "y": 0.}),
                                          ("inspect", {"x": 1., "y": 0., "yaw": 1.})])
def test_endpoint_retry_keeps_docking_or_final_heading_without_counting_twice(after, endpoint):
    state = StageSubgoals("B")
    state.set_goal(goal(), obs())
    arguments = {"waypoints": [endpoint], "after": after}
    state.note_route(arguments)
    state.record_action("follow_waypoints", arguments, obs(), obs(1.),
                        {"status": "replan", "error_code": "execution_slice_complete", "completed_waypoints": 1})
    route = state.pending_route
    assert route["waypoints"] == [] and route["endpoint_pending"]
    assert route["completed_waypoints"] == route["total_waypoints"] == 1
    state.record_action("follow_waypoints", arguments, obs(1.), obs(1., yaw=.5),
                        {"status": "replan", "error_code": "execution_slice_complete", "completed_waypoints": 1})
    assert state.pending_route["completed_waypoints"] == 1
    state.record_action("follow_waypoints", arguments, obs(1., yaw=.5), obs(1., yaw=1.),
                        {"status": "finish" if after == "dock" else "ok", "completed_waypoints": 1})
    assert state.pending_route is None


def test_docking_requires_controller_confirmation_and_distinguishes_independent_score():
    for stage in ("A", "B"):
        state = StageSubgoals(stage)
        state.set_goal(goal(), obs(target=stage))
        state.record_action("follow_waypoints", {}, obs(), obs(target=stage), {"status": "ok"})
        assert state.active["status"] == "active"
        if stage == "A":
            state.record_action("follow_waypoints", {}, obs(), obs(), {"status": "finish"})
            assert state.active["status"] != "completed"
            feedback = {"status": "handoff", "error_code": "A_stop_confirmed"}
            source = "independent_A_handoff"
        else:
            feedback, source = {"status": "finish"}, "controller_finish"
        state.record_action("follow_waypoints", {}, obs(), obs(), feedback)
        assert state.active["status"] == "completed"
        assert state.active["verification"]["source"] == source
        assert state.active["verification"]["independent_task_success"] is None


def test_state_is_bounded_copied_and_reset_for_new_stage():
    state = StageSubgoals("A")
    for _ in range(20):
        state.set_goal(goal("observe_target"), obs(target="A"))
    result = state.snapshot()
    assert len(result["recent_goals"]) == 4
    result["active"]["description"] = "external mutation"
    assert state.active["description"] != "external mutation"
    returning = StageSubgoals("B").snapshot()
    assert returning["active"] is None and returning["last_target_sighting"] is None
    assert returning["recent_goals"] == [] and returning["action_count"] == 0
