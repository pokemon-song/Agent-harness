"""A real MuJoCo collision checks the last pose, time and private recording."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from go2_demo.backend import MujocoGo2Backend
from go2_demo.occluded_video import OccludedRecording
from go2_demo.scenarios import get_scenario


def test_strict_contact_freezes_physics_and_records_exact_last_pose(tmp_path, monkeypatch):
    assets = Path(__file__).resolve().parents[1] / "assets"
    if not (assets / "go2_rl_lab/resources/robots/go2/go2.xml").exists():
        pytest.skip("Go2 assets unavailable")
    # This test never captures pixels, so it also runs on machines without EGL.
    monkeypatch.setattr(mujoco, "Renderer", lambda *_, **__: SimpleNamespace(close=lambda: None))
    backend = MujocoGo2Backend(assets, scenario=get_scenario("memory_ab_occluded_v1"),
                               stop_on_first_collision=True)
    try:
        assert backend.public_task_spec["collision_free_required"]
        recording = OccludedRecording(backend, tmp_path / "capture")
        recording.mark_handoff()
        for _ in range(20):
            result = backend.move_relative(.55, 0.)
            if result["error_code"] == "COLLISION":
                break
        assert backend.collision_terminated and backend._collision_steps == 1
        collision_time = backend.data.time
        pose = backend.data.qpos.copy()
        assert backend.collision_events[0]["geom_names"]
        assert backend.collision_events[0]["sim_time_s"] == collision_time
        assert backend.trajectory[-1]["sim_time_s"] == collision_time
        assert backend.stop()["error_code"] == "COLLISION"
        assert backend.turn(.1)["error_code"] == "COLLISION"
        assert backend.data.time == collision_time
        np.testing.assert_array_equal(backend.data.qpos, pose)
        assert backend.evaluate()["success"] is False
    finally:
        backend.close()
    directory = tmp_path / "capture"
    meta = json.loads((directory / "capture.json").read_text())
    with np.load(directory / "poses.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["qpos"][-1], pose)
        assert np.all(np.diff(archive["time"]) > 0)
        assert meta["start_s"] == meta["handoff_s"] == archive["time"][0]
        assert meta["end_s"] == collision_time == archive["time"][-1]
