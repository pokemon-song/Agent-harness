"""A fresh planner may inherit an explicit measured frame, never a hidden route."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from go2_demo.harness import Harness
from go2_demo.md_memory.navigation import run_md_episode
from go2_demo.md_memory.store import MDMemoryStore
from test_md_navigation import Model, call, seed
from test_runner import EpisodeBackend


class ContinuousBackend(EpisodeBackend):
    def __init__(self):
        super().__init__()
        self.pose = np.array([4., -2., math.pi / 2])
        self.clock = 10.

    def capture(self):
        frame = super().capture()
        frame.base_xy_yaw = self.pose.copy()
        frame.sim_time_s = self.clock
        return frame

    def move_relative(self, forward_m, lateral_m):
        self.moves.append({"forward_m": forward_m, "lateral_m": lateral_m})
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        self.pose[:2] += [c * forward_m - s * lateral_m, s * forward_m + c * lateral_m]
        self.clock += 1.
        self.trajectory.append({"sim_time_s": self.clock, "base_xyz_m": [*self.pose[:2], .3]})
        return {"status": "succeeded"}


def test_shared_harness_origin_preserves_rotation_and_new_default_still_rebases():
    backend = ContinuousBackend()
    shared = Harness(backend, "return", odometry_origin=(4., -4., math.pi / 2))
    odom = shared.observation()["local_odometry"]
    assert odom["translation_m"] == pytest.approx([2., 0.])
    assert odom["yaw_rad"] == 0.
    assert odom["reference"] == "mission_start_body_horizontal"
    assert Harness(backend, "other").observation()["local_odometry"]["translation_m"] == [0., 0.]


@pytest.mark.parametrize("origin", [(0., 0.), (0., False, 0.), (0., 0., float("nan")), "0,0,0"])
def test_invalid_shared_origin_rejected(origin):
    with pytest.raises(ValueError):
        Harness(ContinuousBackend(), "return", odometry_origin=origin)


def test_continuous_legs_keep_frame_clear_history_and_count_only_new_path(tmp_path):
    backend = ContinuousBackend()
    origin = tuple(backend.pose)
    store = MDMemoryStore(tmp_path / "memory")
    seed(store, "Actual route observations in the shared mission frame.")
    first = Model([call("move_relative", {"forward_m": .5, "lateral_m": 0.}), call("finish")])
    before = deepcopy(backend.pose)
    common = dict(store=store, read_only=True, task="Use route memory.", odometry_origin=origin,
                  coordinate_frame_id="same-world")
    one = run_md_episode(backend, first, run_dir=tmp_path / "outbound", **common)
    second = Model([
        call("move_relative", {"forward_m": -.5, "lateral_m": 0.}, identifier="premature"),
        call("memory_read", {"path": "tasks/order.md"}),
        call("move_relative", {"forward_m": -.5, "lateral_m": 0.}, identifier="return"), call("finish")])
    two = run_md_episode(backend, second, run_dir=tmp_path / "return", require_memory_for_motion=True, **common)
    assert one["success"] and two["success"]
    assert len(backend.moves) == 2 and backend.pose == pytest.approx(before)
    assert second.inputs[0][1] == []
    initial = second.inputs[0][0]
    assert initial["local_odometry"]["translation_m"] == pytest.approx([.5, 0.])
    assert initial["agent_context"]["coordinate_frame_id"] == "same-world"
    assert "previous episodes reset local origin" not in initial["agent_context"]["coordinate_reference"]
    assert one["path_length_m"] == pytest.approx(.5)
    assert two["path_length_m"] == pytest.approx(.5)
    assert two["trajectory_start_index"] == 1
    rows = [json.loads(line) for line in (tmp_path / "return/trajectory.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["sim_time_s"] == 12.
    events = [json.loads(line) for line in (tmp_path / "return/events.jsonl").read_text().splitlines()]
    failure = next(e for e in events if e["event"] == "tool_results")["results"][0]
    assert failure["error_code"] == "MEMORY_NOT_READ"
    assert backend.closed is False


def test_missing_memory_does_not_allow_finish_but_abort_remains_available(tmp_path):
    backend = ContinuousBackend()
    model = Model([call("finish"), call("abort", {"reason": "No route memory"})])
    metrics = run_md_episode(backend, model, run_dir=tmp_path / "run",
        store=MDMemoryStore(tmp_path / "empty"), read_only=True, task="return",
        require_memory_for_motion=True)
    assert not metrics["success"] and metrics["termination"] == "abort"
    assert not backend.moves


def test_shared_frame_requires_both_identity_and_anchor(tmp_path):
    with pytest.raises(ValueError):
        run_md_episode(ContinuousBackend(), Model([]), run_dir=tmp_path / "run",
            store=MDMemoryStore(tmp_path / "memory"), task="return", coordinate_frame_id="missing-anchor")
