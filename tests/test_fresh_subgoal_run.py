"""Offline integration checks for stage-local goals and interrupted execution."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from go2_demo.fresh_episode_run import StageFollower, play_stage
from go2_demo.types import ToolCall
from test_fresh_episode_run import StageHarness


class ScriptedPlanner:
    raw_usage = {"total_tokens": 1}
    response_finish_reason = "tool_calls"

    def __init__(self, choices):
        self.choices = iter(choices)
        self.inputs = []

    def choose(self, observation, image, card, history, *, stage, timeout_s):
        self.inputs.append(deepcopy({"stage": stage, "observation": observation, "history": history}))
        name, arguments = next(self.choices)
        return ToolCall("script", name, arguments)


def services(**overrides):
    events = []
    log = SimpleNamespace(write=lambda event, **fields: events.append((event, deepcopy(fields))),
                          save_frame=lambda frame, name: name)
    memory = SimpleNamespace(card=None, observe=lambda *args: None,
                             query_access=lambda: {"allowed": True, "intents": ["target_B"]},
                             after_physical_action=lambda: None)
    cfg = SimpleNamespace(navigation="subgoal", max_requests=10, max_consecutive_errors=1,
                          max_native_actions=20, max_sim_seconds=470, max_wall_seconds=1117,
                          **overrides)
    return log, memory, cfg, events


def no_proposals(*args, **kwargs):
    pytest.fail("The subgoal runner must not instantiate a viewpoint proposal mechanism")


def test_live_stage_slices_resume_only_remaining_route_and_reset_at_handoff(monkeypatch):
    harness = StageHarness(confirm_stop=True)
    log, memory, cfg, events = services()
    route = [{"x": i / 2, "y": 0.} for i in range(1, 9)]
    planner = ScriptedPlanner([
        ("set_subgoal", {"description": "Reach and stop at A", "criterion": "dock_target"}),
        ("follow_waypoints", {"waypoints": route, "after": "dock", "route_role": "primary",
                              "intention": "Traverse the selected route toward A"}),
        ("continue_subgoal", {"intention": "Complete the remaining route and stop at A"}),
        ("abort", {"reason": "End the offline B-stage check"}),
    ])
    monkeypatch.setattr("go2_demo.fresh_episode_run.StageExploration", no_proposals)
    monkeypatch.setattr("go2_demo.paired_return_run.check_clearance", lambda *args, **kwargs: None)
    monkeypatch.setattr("go2_demo.fresh_episode_run.TaskLabelRGBD.inspect",
                        lambda *args: ({"target_candidates": [{"task_label": "A", "pixel_uv": [8, 8]}]}, {}))
    monkeypatch.setattr(StageFollower, "measure_docking_target", lambda self: (.96, 0., .96, 0.))

    outbound = play_stage("A", "empty", harness, log, memory, planner, cfg)
    assert (outbound["termination"], outbound["detail"]) == ("handoff", "A_stop_confirmed")
    continuation_input = planner.inputs[2]["observation"]["task_state"]
    pending = continuation_input["active"]["pending_route"]
    assert pending["completed_waypoints"] == 6
    assert pending["waypoints"] == route[6:]
    assert not continuation_input["recovery_required"]
    assert outbound["task_state"]["active"]["verification"]["source"] == "independent_A_handoff"
    native = [fields["detail"] for event, fields in events if event == "native_action"]
    assert [row["tool"] for row in native] == ["move_relative"] * 8 + ["stop"]
    assert all(row["args"]["forward_m"] == pytest.approx(.5) for row in native[:-1])

    returning = play_stage("B", "empty", harness, log, memory, planner, cfg)
    first_b = planner.inputs[3]
    assert returning["termination"] == "abort"
    assert first_b["history"] == []
    state = first_b["observation"]["task_state"]
    assert state["stage"] == "B" and state["active"] is None
    assert state["last_target_sighting"] is None and state["recent_goals"] == []
    assert state["last_action"] is None and state["action_count"] == 0
    assert all("exploration" not in item["observation"] for item in planner.inputs)


def test_memory_query_and_recovery_keep_blocked_goal_and_last_B_sighting(monkeypatch):
    harness = StageHarness(confirm_stop=False)
    log, memory, cfg, events = services()
    queried = []

    def query(question, **kwargs):
        queried.append(question)
        return {"requests": 0, "unknown_usage_requests": 0, "known_usage": {},
                "status": "unavailable", "error": None, "items": {}, "cache_hit": False, "card": None}

    memory.handle = query
    primary = [{"x": 1., "y": 0.}]
    planner = ScriptedPlanner([
        ("set_subgoal", {"description": "Approach B", "criterion": "dock_target"}),
        ("follow_waypoints", {"waypoints": primary, "after": "dock", "route_role": "primary",
                              "intention": "Approach the currently visible B"}),
        ("memory_query", {"question": "Where was B visible?", "intents": ["target_B"]}),
        ("follow_waypoints", {"waypoints": [{"x": .5, "y": 0.}], "after": "inspect", "route_role": "recovery",
                              "intention": "Move to a different approach position"}),
        ("abort", {"reason": "End the offline recovery check"}),
    ])
    navigate = StageFollower.navigate
    attempts = []

    def block_first(follower, name, arguments, turn, target_visible, **kwargs):
        attempts.append(deepcopy(arguments))
        if len(attempts) == 1:
            return {"status": "replan", "error_code": "current_depth_blocks_requested_translation",
                    "completed_waypoints": 0, "native_actions": 0,
                    "measured_displacement_m": 0., "measured_yaw_change_rad": 0.}
        return navigate(follower, name, arguments, turn, target_visible, **kwargs)

    monkeypatch.setattr(StageFollower, "navigate", block_first)
    monkeypatch.setattr("go2_demo.paired_return_run.check_clearance", lambda *args, **kwargs: None)
    monkeypatch.setattr("go2_demo.fresh_episode_run.TaskLabelRGBD.inspect",
                        lambda *args: ({"target_candidates": [{"task_label": "B", "pixel_uv": [8, 8]}]
                                       if harness.x < .1 else []}, {}))

    result = play_stage("B", "query", harness, log, memory, planner, cfg)
    blocked = planner.inputs[2]["observation"]["task_state"]
    after_query = planner.inputs[3]["observation"]["task_state"]
    after_recovery = planner.inputs[4]["observation"]["task_state"]
    assert queried == ["Where was B visible?"]
    assert blocked["recovery_required"] and after_query["recovery_required"]
    assert blocked["active"]["id"] == after_query["active"]["id"] == after_recovery["active"]["id"]
    assert blocked["action_count"] == after_query["action_count"] == 1
    assert after_recovery["action_count"] == 2
    assert after_recovery["active"]["criterion"] == "dock_target"
    assert after_recovery["active"]["status"] != "completed"
    assert after_recovery["active"]["pending_route"]["waypoints"] == primary
    assert after_recovery["last_target_sighting"] == blocked["last_target_sighting"]
    assert after_recovery["last_action"]["role"] == "recovery"
    assert result["termination"] == "abort"
    assert any(event == "subgoal_verification" for event, _ in events)


def test_slice_limit_yields_before_stage_budget_and_budget_survives_continuation(monkeypatch):
    harness = StageHarness(confirm_stop=False)
    log, memory, _, _ = services()
    follower = StageFollower(harness, log, memory, "B", max_actions=7, actions_per_slice=6)
    monkeypatch.setattr("go2_demo.paired_return_run.check_clearance", lambda *args, **kwargs: None)
    monkeypatch.setattr(follower.sensor, "inspect", lambda *args: ({"target_candidates": []}, {}))
    route = [{"x": i / 2, "y": 0.} for i in range(1, 10)]
    first = follower.navigate("follow_waypoints", {"waypoints": route, "after": "inspect"}, 1, False)
    assert (first["status"], first["error_code"]) == ("replan", "execution_slice_complete")
    assert first["native_actions"] == first["completed_waypoints"] == 6
    assert first["measured_displacement_m"] == 3.
    resumed = follower.navigate("follow_waypoints", {"waypoints": route[6:], "after": "inspect"}, 2, False)
    assert (resumed["status"], resumed["error_code"]) == ("budget_exhausted", "native_action_limit")
    assert resumed["native_actions"] == 1 and resumed["completed_waypoints"] == 1
    assert len(follower.actions) == 7 and harness.x == 3.5


def test_docking_continuation_keeps_approach_progress_after_route_endpoint(monkeypatch):
    harness = StageHarness(confirm_stop=False)
    log, memory, cfg, events = services()
    planner = ScriptedPlanner([
        ("set_subgoal", {"description": "Approach and stop at B", "criterion": "dock_target"}),
        ("follow_waypoints", {"waypoints": [{"x": .5, "y": 0.}], "after": "dock", "route_role": "primary",
                              "intention": "Reach the selected observation position then approach B"}),
        ("continue_subgoal", {"intention": "Continue visual approach from the present body position"}),
        ("continue_subgoal", {"intention": "Confirm a stable stop at B"}),
    ])
    monkeypatch.setattr("go2_demo.paired_return_run.check_clearance", lambda *args, **kwargs: None)
    monkeypatch.setattr("go2_demo.fresh_episode_run.TaskLabelRGBD.inspect",
                        lambda *args: ({"target_candidates": [{"task_label": "B"}]}, {}))

    def visual_measurement(follower):
        forward = 6. - harness.x
        return forward, 0., abs(forward), 0.

    visits = []
    go_to = StageFollower.go_to

    def recorded_go_to(follower, waypoint):
        visits.append(deepcopy(waypoint))
        return go_to(follower, waypoint)

    monkeypatch.setattr(StageFollower, "measure_docking_target", visual_measurement)
    monkeypatch.setattr(StageFollower, "go_to", recorded_go_to)
    result = play_stage("B", "empty", harness, log, memory, planner, cfg)
    first_resume = planner.inputs[2]["observation"]
    assert first_resume["local_odometry"]["translation_m"][0] == pytest.approx(2.9)
    pending = first_resume["task_state"]["active"]["pending_route"]
    assert pending["waypoints"] == [] and pending["endpoint_pending"]
    assert pending["completed_waypoints"] == pending["total_waypoints"] == 1
    assert len(visits) == 1 and visits[0]["x"] == .5
    assert result["termination"] == "finish" and harness.x == pytest.approx(5.04)
    native = [fields["detail"] for event, fields in events if event == "native_action"]
    assert all(row["tool"] != "turn" for row in native)
    assert result["task_state"]["active"]["verification"]["source"] == "controller_finish"
