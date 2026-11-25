"""Task-independent goal verification, recovery and route-prefix regressions."""
from copy import deepcopy
import json
import math

import pytest

from go2_demo.general_subgoals import SubgoalState, validate_goal
from go2_demo.models import ModelProtocolError


TARGET = "valve:west-42"


def obs(x=0., y=0., yaw=0., *, target=None, frame="current", valid=True, target_frame=None):
    return {"image_id": frame, "sim_time_s": 1.,
            "local_odometry": {"translation_m": [x, y], "yaw_rad": yaw, "private": "DO_NOT_COPY_ODOM"},
            "grounded_targets": [] if target is None else [{"target_id": target, "valid": valid,
                "image_id": frame if target_frame is None else target_frame,
                "pixel_uv": [30, 40], "camera_xyz_m": [.2, 0., 3.], "relative_xy_m": [3., -.2],
                "coordinate_frame": "current_camera_optical", "world_xyz": [991., 992., 993.],
                "private_evaluator": "DO_NOT_COPY_TARGET"}],
            "private_evaluator": "DO_NOT_COPY_OBSERVATION"}


def goal(criterion="approach_target", **extra):
    return {"description": "Complete the next measured local objective", "criterion": criterion,
            **({"target_id": TARGET} if criterion in {"observe_target", "approach_target"} else {}), **extra}


def route(*points):
    return {"waypoints": list(points), "after": "inspect"}


def active(task="inventory/wing-3", **goal_fields):
    state = SubgoalState(task)
    state.set_goal(goal(**goal_fields), obs())
    return state


@pytest.mark.parametrize("task,target", [("room-19", "vent-7"), ("巡检/重启", "door:42"), ("", "component_x")])
def test_arbitrary_tasks_and_targets_have_no_stage_or_task_verdict(task, target):
    state = active(task, target_id=target, desired_distance_m=1.3)
    before, after = obs(target=target), obs(.2, target=target, frame="next")
    result = state.record_action("approach_target", {"target_id": target, "desired_distance_m": 1.3}, before, after,
        {"status": "skill_completed", "skill": "approach_target", "target_id": target,
         "private_evaluator": "DO_NOT_COPY_RESULT"})
    assert result["task_id"] == task and "stage" not in result
    assert result["active"]["status"] == "completed"
    assert result["active"]["desired_distance_m"] == 1.3
    assert result["active"]["verification"] == {"state": "completed", "source": "local_skill_not_task_success",
        "skill": "approach_target", "target_id": target, "independent_task_success": None}
    serialized = json.dumps(result)
    assert "DO_NOT_COPY" not in serialized and "world_xyz" not in serialized


@pytest.mark.parametrize("arguments", [
    {}, {"description": " ", "criterion": "observe_target", "target_id": TARGET},
    goal("reach_pose"), goal("reach_pose", target_pose={"x": True, "y": 0.}),
    goal("reach_pose", target_pose={"x": float("nan"), "y": 0.}),
    goal("reach_pose", target_pose={"x": 0., "y": 0., "private": 2.}),
    goal("reach_pose", target_pose={"x": 0., "y": 0.}, target_id=TARGET),
    goal("reach_pose", target_pose={"x": 0., "y": 0.}, desired_distance_m=1.),
    goal("observe_target", target_pose={"x": 0., "y": 0.}),
    goal("observe_target", target_id=""), goal("observe_target", target_id=7),
    {"description": "Look", "criterion": "observe_target"},
    {"description": "Approach", "criterion": "approach_target"},
    goal("observe_target", desired_distance_m=1.), goal(desired_distance_m=True),
    goal(desired_distance_m=float("inf")), goal(desired_distance_m=0.), goal(desired_distance_m=-.2),
    goal(reason=""), goal(secret_evaluator="leak"), goal("dock_target"), goal("finish"),
])
def test_goal_validation_rejects_missing_targets_conflicting_fields_and_private_payloads(arguments):
    with pytest.raises(ModelProtocolError):
        validate_goal(arguments)


def test_goal_arguments_and_returned_state_do_not_alias_caller_data():
    arguments = goal("reach_pose", target_pose={"x": 1., "y": 2., "yaw": .3})
    state = SubgoalState("a-task")
    result = state.set_goal(arguments, obs())
    arguments["target_pose"]["x"] = 99.
    result["task_state"]["active"]["target_pose"]["x"] = 88.
    assert state.active["target_pose"]["x"] == 1.


def test_reach_requires_measured_pose_and_wrapped_heading_not_claimed_completion():
    state = SubgoalState("route-around-object")
    state.set_goal(goal("reach_pose", target_pose={"x": 1., "y": 2., "yaw": math.pi - .03}), obs())
    state.record_action("finish", {}, obs(), obs(1., 2.), {"status": "finish", "success": True})
    assert state.active["status"] == "active"
    result = state.observe(obs(1.1, 2., -math.pi + .02))
    assert result["active"]["status"] == "completed"
    assert result["active"]["verification"]["source"] == "public_odometry"


@pytest.mark.parametrize("candidate", [
    {"target_id": TARGET, "valid": True, "image_id": "old"},
    {"target_id": TARGET, "valid": False, "image_id": "current"},
    {"target_id": TARGET, "valid": 1, "image_id": "current"},
    {"target_id": TARGET, "image_id": "current"},
    {"target_id": TARGET, "valid": True},
    {"target_id": "different-target", "valid": True, "image_id": "current"},
])
def test_stale_invalid_or_other_groundings_cannot_complete_observe(candidate):
    state = active(criterion="observe_target")
    observation = obs()
    observation["grounded_targets"] = [candidate]
    result = state.observe(observation)
    assert result["active"]["status"] == "active"
    assert result["current_target_visible"] is False
    assert result["last_target_sighting"] is None
    good = state.observe(obs(target=TARGET))
    assert good["active"]["status"] == "completed"
    assert good["active"]["verification"]["image_id"] == "current"


def test_current_frame_is_required_and_old_sighting_is_historical_only():
    state = active()
    first = state.observe(obs(2., .5, .2, target=TARGET, frame="seen"))
    assert first["active"]["status"] == "active"  # Sight is not approach completion.
    assert first["last_target_sighting"]["candidate"]["camera_xyz_m"] == [.2, 0., 3.]
    stale = state.observe(obs(2.1, .5, .3, target=TARGET, frame="later", target_frame="seen"))
    assert stale["current_target_visible"] is False
    assert stale["last_target_sighting"]["image_id"] == "seen"
    assert stale["last_target_sighting"]["coordinates_at_sighting_only"]
    state.set_goal(goal("observe_target", reason="Get a fresh sighting"), obs(frame="later"))
    invalid = obs(target=TARGET)
    invalid.pop("image_id")
    assert state.observe(invalid)["active"]["status"] == "active"
    assert state.observe(obs(target=TARGET, frame="fresh"))["active"]["status"] == "completed"
    state.set_goal(goal(target_id="another-component"), obs(target=TARGET, frame="fresh"))
    assert state.last_target_sighting is None


@pytest.mark.parametrize("name,arguments,feedback", [
    ("finish", {}, {"status": "finish"}),
    ("approach_target", {"target_id": TARGET}, {"status": "finish", "skill": "approach_target", "target_id": TARGET}),
    ("approach_target", {"target_id": TARGET}, {"status": "handoff", "error_code": "A_stop_confirmed"}),
    ("approach_target", {"target_id": TARGET}, {"status": "ok", "skill": "approach_target", "target_id": TARGET}),
    ("approach_target", {"target_id": TARGET}, {"status": "skill_completed", "skill": "other", "target_id": TARGET}),
    ("approach_target", {"target_id": TARGET}, {"status": "skill_completed", "skill": "approach_target", "target_id": "other"}),
    ("approach_target", {"target_id": "other"}, {"status": "skill_completed", "skill": "approach_target", "target_id": TARGET}),
    ("follow_waypoints", route({"x": 0., "y": 0.}), {"status": "skill_completed", "skill": "approach_target", "target_id": TARGET}),
])
def test_wrong_skill_feedback_or_finish_never_completes_approach(name, arguments, feedback):
    state = active()
    result = state.record_action(name, arguments, obs(target=TARGET), obs(target=TARGET), feedback)
    assert result["active"]["status"] != "completed"


@pytest.mark.parametrize("distance", [None, 1.0, True, float("nan"), float("inf"), "1.3"])
def test_approach_completion_requires_declared_distance_to_match_executed_request(distance):
    state = active(desired_distance_m=1.3)
    arguments = {"target_id": TARGET}
    if distance is not None:
        arguments["desired_distance_m"] = distance
    feedback = {"status": "skill_completed", "skill": "approach_target", "target_id": TARGET}
    result = state.record_action("approach_target", arguments, obs(target=TARGET), obs(target=TARGET), feedback)
    assert result["active"]["status"] != "completed"
    assert result["no_progress_actions"] == 1
    arguments["desired_distance_m"] = 1.3
    result = state.record_action("approach_target", arguments, obs(target=TARGET), obs(target=TARGET), feedback)
    assert result["active"]["status"] == "completed"
    assert result["active"]["verification"]["source"] == "local_skill_not_task_success"


def test_failed_attempt_queries_replacements_and_repeated_view_do_not_reset_stagnation():
    state = active()
    original_id = state.active["id"]
    blocked = {"status": "replan", "error_code": "current_depth_blocks_requested_translation"}
    state.record_action("follow_waypoints", route({"x": 1., "y": 0.}), obs(), obs(), blocked)
    with pytest.raises(ModelProtocolError):
        state.set_goal(goal("observe_target"), obs())
    for i in range(3):
        state.set_goal(goal(reason="Try a revised local goal"), obs())
        state.record_action("scan", {"yaw_rad": .2}, obs(), obs(), {"status": "ok"})
        state.record_action("memory_query", {}, obs(), obs(), {"status": "ok"})
        state.observe(obs())
    assert state.active["id"] != original_id
    assert state.action_count == state.no_progress_actions == 4
    assert state.recovery_required and state.active["status"] == "recovering"
    # Even a trivial goal achieved at the same pose cannot erase the counter.
    state.set_goal(goal("reach_pose", target_pose={"x": 0., "y": 0.}, reason="Check current pose"), obs())
    assert state.active["status"] == "completed"
    assert state.no_progress_actions == 4 and state.recovery_required


def test_real_detour_preserves_goal_then_clears_recovery_after_measured_progress():
    state = active()
    identifier = state.active["id"]
    state.record_action("approach_target", {"target_id": TARGET}, obs(), obs(),
                        {"status": "failed", "error_code": "PATH_BLOCKED"})
    assert state.recovery_required
    result = state.record_action("follow_waypoints", route({"x": .5, "y": .3}), obs(), obs(.5, .3),
                                 {"status": "ok", "completed_waypoints": 1}, "Move around the blocker", recovery=True)
    assert not result["recovery_required"] and result["no_progress_actions"] == 0
    assert result["active"]["id"] == identifier
    assert result["last_action"]["role"] == "recovery"


def test_rotation_oscillation_stalls_but_rotation_toward_requested_heading_progresses():
    state = active(criterion="observe_target")
    for before, after in ((0., .5), (.5, 0.), (0., .5)):
        state.record_action("scan", {"yaw_rad": after-before}, obs(yaw=before), obs(yaw=after), {"status": "ok"})
    assert state.no_progress_actions == 3 and state.recovery_required
    reach = SubgoalState("turn-to-inspect")
    reach.set_goal(goal("reach_pose", target_pose={"x": 0., "y": 0., "yaw": 1.}), obs())
    result = reach.record_action("scan", {"yaw_rad": .5}, obs(), obs(yaw=.5), {"status": "ok"})
    assert result["last_action"]["made_progress"] and not result["recovery_required"]


def test_primary_prefix_survives_multiple_recoveries_and_only_matching_continuations_advance():
    state = active()
    points = [{"x": float(x), "y": 0.} for x in (1, 2, 3)]
    arguments = route(*points)
    state.note_route(arguments)
    state.record_action("follow_waypoints", arguments, obs(), obs(1.4),
                        {"status": "replan", "error_code": "execution_slice_complete", "completed_waypoints": 1})
    for y in (.4, .8):
        recovery = route({"x": 1.4, "y": y})
        state.note_route(recovery, recovery=True)
        state.record_action("follow_waypoints", recovery, obs(1.4, y-.4), obs(1.4, y),
                            {"status": "ok", "completed_waypoints": 1}, recovery=True)
        assert state.pending_route["waypoints"] == points[1:]
        assert state.pending_route["completed_waypoints"] == 1
    continuation = route(*points[1:])
    state.record_action("follow_waypoints", continuation, obs(1.4, .8), obs(2.2),
                        {"status": "replan", "error_code": "execution_slice_complete", "completed_waypoints": 1})
    assert state.pending_route["waypoints"] == points[2:]
    assert state.pending_route["completed_waypoints"] == 2
    state.record_action("follow_waypoints", route(*points[2:]), obs(2.2), obs(3.),
                        {"status": "ok", "completed_waypoints": 1})
    assert state.pending_route is None
    assert state.active["status"] != "completed"  # Route inspection is not approach.


@pytest.mark.parametrize("submitted,count", [
    (route({"x": 99., "y": 0.}), 1),
    (route({"x": 1., "y": 0.}, {"x": 99., "y": 0.}), 1),
    (route({"x": 2., "y": 0.}, {"x": 1., "y": 0.}), 1),
    ({"waypoints": [{"x": 1., "y": 0.}], "after": "dock"}, 1),
    (route({"x": 1., "y": 0.}), True),
    (route({"x": 1., "y": 0.}), -1),
    (route({"x": 1., "y": 0.}), 2),
    (route({"x": 1., "y": 0.}), 1.0),
])
def test_forged_route_or_prefix_count_cannot_advance_saved_route(submitted, count):
    state = active()
    state.note_route(route({"x": 1., "y": 0.}, {"x": 2., "y": 0.}))
    before = state.pending_route
    state.record_action("follow_waypoints", submitted, obs(), obs(.3),
                        {"status": "ok", "completed_waypoints": count})
    assert state.pending_route == before


@pytest.mark.parametrize("endpoint", [{"x": 1., "y": 0.}, {"x": 1., "y": 0., "yaw": 1.}])
def test_endpoint_continuation_retains_inspection_and_heading_without_counting_twice(endpoint):
    state = active()
    arguments = route(endpoint)
    state.note_route(arguments)
    state.record_action("follow_waypoints", arguments, obs(), obs(1.),
        {"status": "replan", "error_code": "execution_slice_complete", "completed_waypoints": 1})
    assert state.pending_route["endpoint_pending"] and state.pending_route["waypoints"] == []
    assert state.pending_route["completed_waypoints"] == 1
    # finish is not completion of inspect; the prefix stays exactly one.
    state.record_action("follow_waypoints", arguments, obs(1.), obs(1.),
                        {"status": "finish", "completed_waypoints": 1})
    assert state.pending_route["completed_waypoints"] == 1
    # A wrong endpoint or an incorrect measured final pose cannot clear it.
    state.record_action("follow_waypoints", route({"x": 99., "y": 0.}), obs(1.), obs(1.),
                        {"status": "ok", "completed_waypoints": 1})
    assert state.pending_route is not None
    state.record_action("follow_waypoints", arguments, obs(1.), obs(1.5),
                        {"status": "ok", "completed_waypoints": 0})
    assert state.pending_route is not None
    state.record_action("follow_waypoints", arguments, obs(1.5), obs(1., yaw=endpoint.get("yaw", 0.)),
                        {"status": "skill_completed", "skill": "follow_waypoints", "completed_waypoints": 0})
    assert state.pending_route is None


@pytest.mark.parametrize("arguments", [
    {"waypoints": [{"x": 1., "y": 0.}], "after": "dock"},
    {"waypoints": [{"x": 1., "y": 0.}], "after": "approach_target"},
    {"waypoints": [{"x": 1., "y": 0.}], "after": "inspect", "completed_waypoints": 1},
    route(), route({"x": 1., "y": float("inf")}),
])
def test_routes_cannot_embed_task_specialized_docking_or_progress_claims(arguments):
    state = active()
    for recovery in (False, True):
        with pytest.raises(ModelProtocolError):
            state.note_route(arguments, recovery=recovery)


def test_route_and_snapshot_copies_are_bounded_and_extra_observation_fields_never_escape():
    state = SubgoalState("many-goals")
    for i in range(20):
        state.set_goal(goal("observe_target", target_id=f"sensor-{i}"), obs(target=f"sensor-{i}"))
    state.set_goal(goal(), obs())
    arguments = route({"x": 1., "y": 0.})
    state.note_route(arguments)
    arguments["waypoints"][0]["x"] = 99.
    pending = state.pending_route
    pending["waypoints"][0]["x"] = 88.
    result = state.observe(obs(target=TARGET))
    result["active"]["description"] = "external mutation"
    result["last_target_sighting"]["candidate"]["pixel_uv"][0] = 999
    clean = state.snapshot()
    assert state.pending_route["waypoints"][0]["x"] == 1.
    assert state.active["description"] != "external mutation"
    assert clean["last_target_sighting"]["candidate"]["pixel_uv"] == [30, 40]
    assert len(clean["recent_goals"]) == 4
    assert "DO_NOT_COPY" not in json.dumps(clean) and "world_xyz" not in json.dumps(clean)
    fresh = SubgoalState("unrelated").snapshot()
    assert fresh["active"] is None and fresh["last_target_sighting"] is None
    assert fresh["recent_goals"] == [] and fresh["action_count"] == 0


def agent_active(**goal_fields):
    state = SubgoalState("generic-agent-owned-task", update_policy="agent")
    state.set_goal(goal(**goal_fields), obs())
    return state


def judgment(state, status="active", **extra):
    return {"goal_id": state.active["id"], "status": status,
            "reason": "Reviewed the current observations and execution feedback", **extra}


@pytest.mark.parametrize("criterion,fields,observed", [
    ("reach_pose", {"target_pose": {"x": 1., "y": 2.}}, obs(1., 2.)),
    ("observe_target", {}, obs(target=TARGET)),
])
def test_agent_observations_report_evidence_without_completing_goal(criterion, fields, observed):
    state = agent_active(criterion=criterion, **fields)
    state.note_route(route({"x": 1., "y": 2.}))
    saved = state.pending_route
    result = state.observe(observed)
    assert result["active"]["status"] == "active"
    assert state.pending_route == saved
    assert result["active"]["observed_evidence"]["criterion_supported_by_observation"] is True
    assert result["active"]["observed_evidence"]["state"] == "diagnostic_only"
    assert result["active"]["verification"]["state"] == "awaiting_agent"
    state.update_goal(judgment(state, "completed"), observed)
    assert state.active["status"] == "completed" and state.pending_route is None
    assert state.active["verification"]["source"] == "agent_judgment"


def test_agent_skill_success_and_executor_prefix_are_diagnostics_not_plan_mutations():
    state = agent_active(desired_distance_m=1.3)
    points = [{"x": 1., "y": 0.}, {"x": 2., "y": 0.}]
    state.note_route(route(*points))
    saved = state.pending_route
    state.record_action("follow_waypoints", route(*points), obs(), obs(2.),
                        {"status": "ok", "completed_waypoints": 2})
    assert state.pending_route == saved
    assert state.last_action["executor_completed_waypoints"] == 2
    assert state.last_action["diagnostic_only"] is True
    state.record_action("approach_target", {"target_id": TARGET, "desired_distance_m": 1.3},
        obs(target=TARGET), obs(target=TARGET),
        {"status": "skill_completed", "skill": "approach_target", "target_id": TARGET})
    assert state.last_action["local_skill_confirmed"] is True
    assert state.active["status"] == "active" and state.pending_route == saved
    state.update_goal(judgment(state, "completed"), obs(target=TARGET))
    assert state.active["status"] == "completed" and state.pending_route is None
    assert state.active["verification"]["independent_task_success"] is None


def test_agent_decides_recovery_even_when_stagnation_or_motion_diagnostics_change():
    state = agent_active()
    for _ in range(4):
        state.record_action("scan", {"yaw_rad": .2}, obs(), obs(), {"status": "ok"})
    assert state.no_progress_actions == 4 and state.last_action["repeated_view"]
    assert state.active["status"] == "active" and not state.recovery_required
    state.record_action("approach_target", {"target_id": TARGET}, obs(), obs(),
                        {"status": "failed", "error_code": "PATH_BLOCKED"})
    assert state.last_action["blocked_feedback"] and not state.recovery_required
    reason = "The repeated observations show no useful progress; try a detour"
    state.update_goal(judgment(state, "recovering", reason=reason), obs())
    state.record_action("follow_waypoints", route({"x": .5, "y": .3}), obs(), obs(.5, .3),
                        {"status": "ok", "completed_waypoints": 1}, recovery=True)
    assert state.active["status"] == "recovering" and state.recovery_required
    assert state.recovery_reason == reason
    assert state.no_progress_actions == 0  # A diagnostic, not a status transition.
    state.update_goal(judgment(state, "active"), obs(.5, .3))
    assert state.active["status"] == "active" and not state.recovery_required


def test_agent_cumulative_route_prefix_and_endpoint_require_explicit_updates():
    state = agent_active()
    points = [{"x": float(x), "y": 0.} for x in (1, 2, 3)]
    points[-1]["yaw"] = 1.
    state.note_route(route(*points))
    state.update_goal(judgment(state, completed_waypoints=1), obs(1.))
    assert state.pending_route["completed_waypoints"] == 1
    assert state.pending_route["waypoints"] == points[1:]
    state.record_action("follow_waypoints", route(*points[1:]), obs(1.), obs(3., yaw=1.),
                        {"status": "ok", "completed_waypoints": 2})
    assert state.pending_route["completed_waypoints"] == 1
    state.update_goal(judgment(state, completed_waypoints=3), obs(3., yaw=1.))
    endpoint = state.pending_route
    assert endpoint["completed_waypoints"] == 3 and endpoint["waypoints"] == []
    assert endpoint["endpoint_pending"] and endpoint["endpoint"] == points[-1]
    state.observe(obs(3., yaw=1.))
    assert state.pending_route == endpoint  # Arrival alone cannot clear the route.
    result = state.update_goal(judgment(state, route_done=True), obs(3., yaw=1.))
    assert state.pending_route is None and state.active["status"] == "active"
    result["task_state"]["active"]["description"] = "Mutated outside state"
    assert state.active["description"] != "Mutated outside state"


@pytest.mark.parametrize("patch", [
    {"goal_id": "previous-goal"}, {"goal_id": 1}, {"status": "passed"},
    {"status": None}, {"reason": " "}, {"reason": "x" * 241},
    {"completed_waypoints": True}, {"completed_waypoints": 1.0},
    {"completed_waypoints": -1}, {"completed_waypoints": 3},
    {"route_done": 1}, {"route_done": True, "completed_waypoints": 1},
    {"success": True},
])
def test_invalid_agent_judgments_are_atomic(patch):
    state = agent_active()
    state.note_route(route({"x": 1., "y": 0.}, {"x": 2., "y": 0.}))
    before = state.snapshot()
    with pytest.raises(ModelProtocolError):
        state.update_goal({**judgment(state), **patch}, obs(target=TARGET))
    assert state.snapshot() == before


def test_agent_route_counts_cannot_regress_or_apply_after_route_replacement():
    state = agent_active()
    state.note_route(route({"x": 1., "y": 0.}, {"x": 2., "y": 0.}))
    state.update_goal(judgment(state, completed_waypoints=1), obs())
    before = state.snapshot()
    with pytest.raises(ModelProtocolError):
        state.update_goal(judgment(state, completed_waypoints=0), obs())
    assert state.snapshot() == before
    state.note_route(route({"x": 4., "y": 0.}))
    assert state.pending_route["completed_waypoints"] == 0
    before = state.snapshot()
    with pytest.raises(ModelProtocolError):
        state.update_goal(judgment(state, completed_waypoints=2), obs())
    assert state.snapshot() == before
    state.update_goal(judgment(state, route_done=True), obs())
    before = state.snapshot()
    for extra in ({"completed_waypoints": 0}, {"route_done": True}):
        with pytest.raises(ModelProtocolError):
            state.update_goal(judgment(state, **extra), obs())
        assert state.snapshot() == before


@pytest.mark.parametrize("terminal", ["completed", "abandoned"])
def test_agent_terminal_claim_is_retained_without_faking_independent_acceptance(terminal):
    state = agent_active(criterion="reach_pose", target_pose={"x": 5., "y": 0.})
    state.note_route(route({"x": 5., "y": 0.}))
    observation = obs()
    observation["task_feedback"] = {"task_completed": False}
    state.update_goal(judgment(state, terminal), observation)
    result = state.observe(obs(.1, target=TARGET))
    assert result["active"]["status"] == terminal and state.pending_route is None
    assert result["active"]["verification"]["independent_task_success"] is None
    assert result["active"]["observed_evidence"]["criterion_supported_by_observation"] is False
    assert "DO_NOT_COPY" not in json.dumps(result) and "world_xyz" not in json.dumps(result)
    with pytest.raises(ModelProtocolError):
        state.note_route(route({"x": 1., "y": 0.}))
    state.set_goal(goal("observe_target"), obs(target=TARGET))
    assert state.recent_goals[-1]["status"] == terminal
    assert state.active["status"] == "active"  # Sighting alone does not complete it.


def test_agent_replacement_starts_active_and_keeps_stagnation_as_diagnostic():
    state = agent_active()
    state.record_action("scan", {"yaw_rad": .2}, obs(), obs(), {"status": "ok"})
    state.update_goal(judgment(state, "recovering"), obs())
    with pytest.raises(ModelProtocolError):
        state.set_goal(goal("observe_target"), obs())
    state.set_goal(goal("observe_target", reason="Change the objective after reviewing feedback"), obs())
    assert state.active["status"] == "active" and not state.recovery_required
    assert state.no_progress_actions == 1


def test_agent_update_requires_explicit_mode_and_current_goal():
    with pytest.raises(ValueError):
        SubgoalState("task", update_policy="unknown")
    measured = active()
    before = measured.snapshot()
    with pytest.raises(ModelProtocolError):
        measured.update_goal(judgment(measured, "completed"), obs())
    assert measured.snapshot() == before
    fresh = SubgoalState("task", update_policy="agent")
    with pytest.raises(ModelProtocolError):
        fresh.update_goal({"goal_id": "goal-1", "status": "active", "reason": "No goal yet"}, obs())
