from types import SimpleNamespace
from dataclasses import replace

import numpy as np

from go2_demo.experimental_boundary_gate import experimental_boundary_gate
from go2_demo.fresh_episode_run import StageFollower, episode_scene, play_stage
from go2_demo.types import ToolCall
from test_return_route_control import PublicHarness


class StageHarness(PublicHarness):
    def __init__(self, confirm_stop):
        super().__init__()
        self.confirm_stop = confirm_stop
        self.backend._ordered_evaluator = SimpleNamespace(a_witness=None)
        self.backend.deadline_monotonic = self.backend.sim_deadline_s = float("inf")
        self.backend.trajectory = []
        self.frame = SimpleNamespace(
            sim_time_s=0., base_xy_yaw=np.zeros(3), frame_id="start",
            rgb=np.zeros((16, 16, 3), dtype=np.uint8), depth_m=np.ones((16, 16)),
            intrinsics=np.array([[16., 0, 8.], [0, 16., 8.], [0, 0, 1.]]),
        )
        self.dispatched = []

    def dispatch(self, call):
        self.dispatched.append(call.name)
        result = super().dispatch(call)
        if call.name == "stop" and self.confirm_stop:
            self.backend._ordered_evaluator.a_witness = {"sim_time_s": self.clock}
        self.frame.sim_time_s = self.clock
        self.frame.frame_id = f"frame-{self.clock}"
        self.frame.base_xy_yaw[:] = self.x, 0., self.yaw
        self.backend.trajectory.append({"base_xyz_m": [self.x, 0., 0.]})
        return result


DOCK_HERE = {"waypoints": [{"x": 0., "y": 0.}], "after": "dock"}


def test_wider_episode_moves_the_platform_and_its_boundary_gate_together():
    original = episode_scene()
    wider = episode_scene((24, 18))
    assert original.floor_half_size_m == (8., 6.)
    assert wider == replace(original, floor_half_size_m=(12., 9.))
    for scene, x, blocked in ((original, 7.5, True), (wider, 7.5, False), (wider, 11.5, True)):
        result = experimental_boundary_gate(scene, (x, 0., 0.), forward_m=.3, lateral_m=0.)
        assert (result is not None) == blocked


def test_native_action_budget_applies_across_planner_decisions():
    harness = StageHarness(confirm_stop=False)
    follower = StageFollower(harness, SimpleNamespace(write=lambda *a, **kw: None),
                             SimpleNamespace(observe=lambda *a: None), "A", max_actions=1)
    assert follower.execute("scan", {"yaw_rad": .1}, turn=1)["status"] == "ok"
    result = follower.execute("scan", {"yaw_rad": .1}, turn=2)
    assert (result["status"], result["error_code"]) == ("budget_exhausted", "native_action_limit")
    assert harness.dispatched == ["turn"]


def test_A_visual_dock_cannot_dispatch_finish_without_independent_stop(monkeypatch):
    harness = StageHarness(confirm_stop=False)
    follower = StageFollower(harness, SimpleNamespace(write=lambda *a, **kw: None),
                             SimpleNamespace(observe=lambda *a: None), "A")
    monkeypatch.setattr(follower, "measure_docking_target", lambda: (.96, 0., .96, 0.))
    result = follower.execute("follow_waypoints", DOCK_HERE, turn=1)
    assert (result["status"], result["error_code"]) == ("replan", "A_stop_not_yet_confirmed")
    assert harness.dispatched == ["stop"]


def test_independent_A_stop_interrupts_macro_and_first_B_decision_has_empty_history(monkeypatch):
    harness = StageHarness(confirm_stop=True)
    log = SimpleNamespace(write=lambda *a, **kw: None, save_frame=lambda frame, name: name)
    memory = SimpleNamespace(card=None, observe=lambda *a: None, query_access=lambda: {"allowed": False, "intents": []},
                             after_physical_action=lambda: None)
    config = SimpleNamespace(max_requests=4, max_consecutive_errors=1, max_native_actions=2,
                             max_sim_seconds=470, max_wall_seconds=1117, navigation="viewpoints")
    choices = iter([("scan", {"yaw_rad": .1}), ("follow_waypoints", DOCK_HERE), ("abort", {"reason": "end"})])
    turns = []

    def choose(obs, image, card, history, *, stage, timeout_s):
        turns.append((stage, [entry["tool"] for entry in history], "depth_sectors" in obs))
        name, args = next(choices)
        return ToolCall("test", name, args)

    planner = SimpleNamespace(choose=choose, raw_usage={"total_tokens": 1})
    measurements = []

    def measure(follower):
        measurements.append(follower.target_label)
        return .96, 0., .96, 0.

    monkeypatch.setattr(StageFollower, "measure_docking_target", measure)
    monkeypatch.setattr("go2_demo.fresh_episode_run.TaskLabelRGBD.inspect",
                        lambda *a: ({"target_candidates": []}, {}))
    outbound = play_stage("A", "empty", harness, log, memory, planner, config)
    assert (outbound["termination"], outbound["detail"]) == ("handoff", "A_stop_confirmed")
    assert harness.dispatched == ["turn", "stop"]
    # Dock would otherwise measure again and attempt finish after its stop.
    assert measurements == ["A"]

    returning = play_stage("B", "empty", harness, log, memory, planner, config)
    assert returning["termination"] == "abort"
    assert turns == [("A", [], True), ("A", ["scan"], True), ("B", [], False)]


def test_inspect_near_identical_waypoint_does_not_pretend_to_change_view():
    harness = StageHarness(confirm_stop=False)
    follower = StageFollower(harness, SimpleNamespace(write=lambda *a, **kw: None),
                             SimpleNamespace(observe=lambda *a: None), "A")
    result = follower.navigate("follow_waypoints", {"waypoints": [{"x": .008, "y": 0.}], "after": "inspect"},
                               1, False)
    assert result["error_code"] == "inspect_waypoint_already_reached" and result["native_actions"] == 0
    assert not harness.dispatched


def test_inspect_stops_a_multi_step_route_at_a_new_target_view(monkeypatch):
    harness = StageHarness(confirm_stop=False)
    follower = StageFollower(harness, SimpleNamespace(write=lambda *a, **kw: None),
                             SimpleNamespace(observe=lambda *a: None), "A")
    monkeypatch.setattr(follower.sensor, "inspect", lambda *a: ({"target_candidates": [{"task_label": "A"}]}, {}))
    monkeypatch.setattr("go2_demo.paired_return_run.check_clearance", lambda *a, **kw: None)
    result = follower.navigate("follow_waypoints", {"waypoints": [{"x": 1.4, "y": 0.}], "after": "inspect"},
                               1, False)
    assert result["error_code"] == "target_visible_in_current_view" and result["status"] == "replan"
    assert result["native_actions"] == 1 and harness.dispatched == ["move_relative"]
    assert harness.backend._ordered_evaluator.a_witness is None
