from types import SimpleNamespace

import numpy as np

from go2_demo.fresh_episode_run import StageFollower, play_stage
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
    memory = SimpleNamespace(card=None, observe=lambda *a: None)
    config = SimpleNamespace(max_requests=4, max_consecutive_errors=1)
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
