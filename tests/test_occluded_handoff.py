import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from go2_demo.occluded_handoff import _OutboundNotes, _compare_observations, _verify_trajectory


def test_notes_use_current_public_surface_and_keep_evidence_ids_unique():
    h = SimpleNamespace(frame=SimpleNamespace(rgb=None, depth_m=None, intrinsics=None))
    o = {"image_id": "frame-01", "sim_time_s": 1., "image_size": {"width": 1, "height": 1},
         "local_odometry": {"translation_m": [3., 1.], "yaw_rad": np.pi / 2},
         "tracked_points": [{"point_id": "p1", "relative_xy_m": [1., .25]}]}
    h.observation = lambda: o
    notes = _OutboundNotes("memory_ab_occluded_v1")
    candidate = {"task_label": "B", "description": "绿方块", "pixel_uv": [0, 0]}
    notes.sensor = SimpleNamespace(inspect=lambda *args: ({"target_candidates": [candidate]}, {}),
                                   associate=lambda *args: candidate)
    notes.observe(h, 0)
    notes.observe(h, 1)
    docs = notes.documents()
    target = docs["targets/target-B.md"]
    assert "(2.750,2.000)" in target and "表面" in target
    evidence_ids = re.findall(r"^\[(E\d+)\]", "\n".join(docs.values()), flags=re.M)
    assert len(evidence_ids) == len(set(evidence_ids))
    assert all(len(line) < 300 for doc in docs.values() for line in doc.splitlines())


def test_only_changed_collision_policy_is_normalized():
    old = {"completion_requirements": {"collision_free_required": True, "standoff_m": 1.}, "pose": [0, 0]}
    new = {"completion_requirements": {"collision_free_required": False, "standoff_m": 1.,
                                        "collision_penalty_points": 5}, "pose": [0, 0]}
    changes = set()
    _compare_observations(old, new, changes)
    assert changes == {"completion_requirements.collision_policy"}
    with pytest.raises(ValueError):
        _compare_observations(old, {**new, "pose": [8, 8]}, set())


def test_verified_prefix_stops_before_bad_tail(tmp_path):
    row = {"base_xyz_m": [1., 2., .3], "yaw_rad": .2, "sim_time_s": 1.,
           "velocity_world_mps": [0., 0., 0.], "command": [0., 0., 0.], "fallen": False, "collision_steps": 0}
    (tmp_path / "trajectory.jsonl").write_text(json.dumps(row) + "\nfuture invalid data")
    assert len(_verify_trajectory([row], tmp_path)) == 64
    with pytest.raises(ValueError, match="yaw_rad"):
        _verify_trajectory([{**row, "yaw_rad": 1.}], tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        _verify_trajectory([{**row, "collision_steps": 1}], tmp_path)
