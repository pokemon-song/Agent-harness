from types import SimpleNamespace

from go2_demo.paired_return_run import ReturnFollower
from test_return_route_control import PublicHarness


def test_uncited_waypoints_replan_after_depth_refusal_then_walk_twice(monkeypatch):
    harness = PublicHarness()
    harness.frame = object()
    harness.backend.sim_deadline_s = harness.backend.deadline_monotonic = float("inf")
    follower = ReturnFollower(harness, SimpleNamespace(write=lambda *args, **kwargs: None))
    depths = iter([{"error_code": "VISIBLE_OBSTACLE_IN_STEP"}, None, None])
    monkeypatch.setattr("go2_demo.paired_return_run.check_clearance", lambda *args, **kwargs: next(depths))
    def inspect(x, index):
        return follower.execute("follow_waypoints", {"waypoints": [{"x": x, "y": 0.}], "after": "inspect"}, index)
    first = inspect(.5, 1)
    assert first["status"] == "replan" and first["native_actions"] == 0
    assert first["depth_feedback"]["error_code"] == "VISIBLE_OBSTACLE_IN_STEP"
    assert inspect(.5, 2)["status"] == "ok" and inspect(1., 3)["status"] == "ok"
    assert harness.x == 1. and follower.refusals == 1
    assert len(set(a["call_id"] for a in follower.actions)) == len(follower.actions) == 2
