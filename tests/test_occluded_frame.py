import os
from pathlib import Path
from zipfile import ZipFile

import pytest

from go2_demo.occluded_frame import replay_return_prefix, restore_frame_six


def test_wrong_memory_revision_rejected_before_executing_any_action():
    with pytest.raises(ValueError, match="memory revision"):
        replay_return_prefix(None, {"different.md": "# Other episode"})


def test_original_frame_and_five_completed_public_turns(tmp_path):
    from go2_demo.backend import MujocoGo2Backend
    from go2_demo.scenarios import get_scenario

    root = Path(__file__).resolve().parents[1]
    assets = Path(os.environ.get("GO2_ASSETS_DIR", root / "assets"))
    if not (assets / "go2_rl_lab").is_dir():
        pytest.skip("Locked Go2 assets unavailable")
    with ZipFile(root / "datasets/occluded-return-seed0.zip") as src:
        src.extractall(tmp_path)
    backend = MujocoGo2Backend(assets, seed=0, scenario=get_scenario("memory_ab_occluded_v1"),
                               stop_on_first_collision=True)
    try:
        harness, _, obs, history, audit = restore_frame_six(backend, tmp_path)
        assert audit["public_history_sha256"] == "4b7f67a8440d33f151905e51b6b26435ab11430f5ab12defe5a2ab11098d8a33"
        assert audit["public_observation_sha256"] == "ba709c1b96d249d191fe23a1891f346b7613fdf4b2fb7290dca22145a6f8030c"
        assert obs["image_id"] == harness.frame.frame_id == "frame-00066"
        assert obs["sim_time_s"] == pytest.approx(133.2)
        assert not obs["target_candidates"] and backend.data.time == obs["sim_time_s"]
        assert [p["point_id"] for p in obs["tracked_points"] if p["valid"]] == ["point-003"]
        assert [turn["calls"][0]["name"] for turn in history] == [
            "memory_query", "track_point", "track_point", "turn", "track_point"]
        assert all(set(turn) == {"observation", "calls", "results"} for turn in history)
        assert all("return_memory" not in turn["observation"] for turn in history)
        assert len(obs["return_memory"]["current"]["evidence"]) == 3
    finally:
        backend.close()
