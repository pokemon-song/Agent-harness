"""Offline movie timing and real encode/decode tests; never calls a model API."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def video():
    path = Path(__file__).resolve().parents[1] / "scripts/render_memory_mission_video.py"
    spec = importlib.util.spec_from_file_location("memory_mission_video", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resampling_holds_inference_pause_and_preserves_both_endpoints(video):
    wall = np.array([0., .08, .08, 1.08, 1.10])
    indices, times = video.frame_schedule(wall, 25)
    assert indices[0] == 0 and indices[-1] == 4
    assert np.all(indices[(times >= .08) & (times < 1.08)] == 2)
    assert len(indices) == 29 and times[-1] == 1.10
    # The same paused simulation has many duplicate timestamps.
    sim_indices, _ = video.frame_schedule([1., 1.02, 1.02, 1.02, 1.04], 25)
    assert list(sim_indices) == [0, 4]


@pytest.mark.parametrize("times", ([], [0, float("nan")], [0, 2, 1], [[0, 1]], [0, float("inf")]))
def test_rejects_invalid_clock_before_render(video, times):
    with pytest.raises(ValueError):
        video.frame_schedule(times)


def make_recording(video, directory):
    import mujoco
    directory.mkdir(parents=True)
    model = mujoco.MjModel.from_xml_string('''<mujoco><visual><global offwidth="160" offheight="120"/></visual>
      <worldbody><light pos="0 0 3"/><geom type="plane" size="4 4 .1"/>
      <body pos="0 0 .3"><freejoint/><geom type="box" size=".2 .1 .1" rgba="1 0 0 1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    states = []
    for x, time in ((0., 1.), (0., 1.), (.5, 1.04), (.5, 1.04)):
        data.qpos[0], data.time = x, time
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        states.append(state)
    mujoco.mj_saveModel(model, str(directory / "model.mjb"))
    np.savez_compressed(directory / "states.npz", state=states,
                        sim_time_s=[1., 1., 1.04, 1.04], wall_time_s=[0., .12, .16, .24])
    metadata = {"state_spec": spec, "sample_count": 4, "mujoco_version": mujoco.__version__,
                "model_sha256": video.sha256(directory / "model.mjb"),
                "states_sha256": video.sha256(directory / "states.npz")}
    (directory / "metadata.json").write_text(json.dumps(metadata))


def test_rejects_tampered_state_archive(video, tmp_path):
    make_recording(video, tmp_path / "recording")
    with (tmp_path / "recording/states.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        video.load_recording(tmp_path / "recording")


def test_real_mujoco_render_and_video_decode_without_physics_step(video, tmp_path, monkeypatch):
    pytest.importorskip("imageio_ffmpeg", reason="Optional video dependencies not installed")
    import mujoco
    for phase in video.PHASES:
        make_recording(video, tmp_path / phase / "third_person")
    # Rendering must reconstruct archived states, never rerun the controller/physics.
    monkeypatch.setattr(mujoco, "mj_step", lambda *args, **kwargs: pytest.fail("Offline renderer stepped physics"))
    report = video.render_mission(tmp_path, width=320, height=240, fps=25)
    assert report["all_phases_recorded"] and report["missing_phases"] == []
    for phase in report["phases"]:
        assert phase["videos"]["wall"]["frame_count"] == 7
        for clip in phase["videos"].values():
            assert clip["decoded_all_frames"] and clip["first_and_final_states_included"]
            assert clip["first_source_index"] == 0 and clip["last_source_index"] == 3
    assert report["combined"]["wall"]["frame_count"] == 3 * 7 + 4 * 25
    assert report["combined"]["wall"]["decoded_all_frames"]
    with pytest.raises(FileExistsError):
        video.render_mission(tmp_path, width=320, height=240)
