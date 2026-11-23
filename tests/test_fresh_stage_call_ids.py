"""A/B share real Harness dispatch deduplication without reusing action IDs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from go2_demo.fresh_episode_run import StageFollower
from go2_demo.harness import Harness
from go2_demo.return_route_control import _End, _Follower
from go2_demo.types import ToolCall
from test_harness import RecordingBackend


class StageBackend(RecordingBackend):
    def __init__(self):
        super().__init__()
        self.collision_terminated = False
        self.deadline_monotonic = self.sim_deadline_s = float("inf")
        self._ordered_evaluator = SimpleNamespace(a_witness=None)

    def capture(self):
        frame = super().capture()
        frame.robot_state["fallen"] = False
        return frame

    def turn(self, yaw_rad):
        self.pose[2] += yaw_rad
        return super().turn(yaw_rad)

    def stop(self):
        self._ordered_evaluator.a_witness = {"sim_time_s": self.capture_count * .1}
        return super().stop()


@pytest.mark.parametrize("return_yaw", [.2, -.2])
def test_retained_harness_dispatch_executes_return_actions_and_preserves_cache(return_yaw):
    backend = StageBackend()
    harness = Harness(backend, "先停靠 A，再返回 B")
    log = SimpleNamespace(write=lambda *args, **kwargs: None)
    memory = SimpleNamespace(observe=lambda *args: None)
    outbound = StageFollower(harness, log, memory, "A", max_actions=2)
    outbound.act("turn", yaw_rad=.2)
    with pytest.raises(_End) as ended:
        outbound.act("stop")
    assert (ended.value.reason, ended.value.detail) == ("handoff", "A_stop_confirmed")
    cache = harness.cache
    outbound_cache = deepcopy(cache)
    tracker = harness.tracker

    returning = StageFollower(harness, log, memory, "B", max_actions=2)
    turn = returning.act("turn", yaw_rad=return_yaw)
    stop = returning.act("stop")
    assert turn["status"] == stop["status"] == "succeeded"
    assert not turn.get("deduplicated") and not stop.get("deduplicated")
    assert backend.turns == [.2, return_yaw] and backend.stops == 2
    assert len(outbound.actions) == len(returning.actions) == 2
    assert harness.cache is cache and harness.tracker is tracker
    assert all(cache[key] == value for key, value in outbound_cache.items())
    assert set(cache) == {"stage-A-000", "stage-A-001", "stage-B-000", "stage-B-001"}

    # Keeping the cache still prevents a repeated call from executing twice.
    replay = harness.dispatch(ToolCall(turn["call_id"], "turn", {"yaw_rad": return_yaw}))
    assert replay["deduplicated"] is True
    conflict = harness.dispatch(ToolCall(turn["call_id"], "turn", {"yaw_rad": .4}))
    assert conflict["error_code"] == "CALL_ID_REUSED"
    assert backend.turns == [.2, return_yaw]


def test_standalone_route_follower_keeps_default_call_id_namespace():
    backend = StageBackend()
    harness = Harness(backend, "原生路线控制")
    follower = _Follower(harness, max_actions=2, on_action=None)
    follower.act("turn", yaw_rad=.2)
    follower.act("turn", yaw_rad=-.2)
    assert set(harness.cache) == {"route-control-000", "route-control-001"}
