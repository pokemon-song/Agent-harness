"""Offline video evidence must preserve physics and the agent's actual camera."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from go2_demo.recording import SimulationRecording


def free_body():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep="0.002"/>
          <worldbody><body pos="0 0 1"><freejoint/>
            <geom type="box" size=".1 .2 .3" mass="1"/>
          </body></worldbody>
        </mujoco>
    """)
    return model, mujoco.MjData(model)


def load_archive(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(directory / metadata["model_file"]))
    with np.load(directory / metadata["states_file"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return metadata, model, arrays


def test_binary_model_and_every_saved_state_restore_exact_time_and_pose(tmp_path):
    model, data = free_body()
    data.qvel[:3] = [.2, -.1, 0.]
    mujoco.mj_forward(model, data)
    directory = tmp_path / "third_person"
    recording = SimulationRecording(model, data, directory)
    expected = [(float(data.time), data.qpos.copy(), data.qvel.copy())]
    for _ in range(3):
        for _ in range(10):
            mujoco.mj_step(model, data)
        recording.sample(model, data)
        expected.append((float(data.time), data.qpos.copy(), data.qvel.copy()))
    recording.finish()

    metadata, replay_model, arrays = load_archive(directory)
    assert metadata["sample_count"] == len(expected)
    assert metadata["state_spec"] == int(mujoco.mjtState.mjSTATE_INTEGRATION)
    assert metadata["sim_start_s"] == expected[0][0]
    assert metadata["sim_end_s"] == expected[-1][0]
    assert arrays["state"].shape == (len(expected), metadata["state_size"])
    assert np.all(np.diff(arrays["wall_time_s"]) >= 0)
    assert metadata["wall_duration_s"] == arrays["wall_time_s"][-1]
    for stem in ("model", "states"):
        assert metadata[stem + "_sha256"] == hashlib.sha256(
            (directory / metadata[stem + "_file"]).read_bytes()).hexdigest()
    replay_data = mujoco.MjData(replay_model)
    for saved, (sim_time, qpos, qvel) in zip(arrays["state"], expected):
        mujoco.mj_setState(replay_model, replay_data, saved, metadata["state_spec"])
        mujoco.mj_forward(replay_model, replay_data)
        assert replay_data.time == sim_time
        np.testing.assert_array_equal(replay_data.qpos, qpos)
        np.testing.assert_array_equal(replay_data.qvel, qvel)


def test_finish_is_idempotent_and_late_sampling_cannot_change_evidence(tmp_path):
    model, data = free_body()
    directory = tmp_path / "third_person"
    recording = SimulationRecording(model, data, directory)
    recording.finish()
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    mujoco.mj_step(model, data)
    recording.sample(model, data)
    recording.finish()
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == original
    assert len(recording.states) == 1
    with pytest.raises(FileExistsError):
        SimulationRecording(model, data, directory)


def test_archive_write_failure_can_be_retried_without_losing_samples(tmp_path, monkeypatch):
    model, data = free_body()
    recording = SimulationRecording(model, data, tmp_path / "third_person")
    mujoco.mj_step(model, data)
    recording.sample(model, data)
    original_save = np.savez_compressed

    def fail_save(*args, **kwargs):
        raise OSError("synthetic archive write failure")

    monkeypatch.setattr(np, "savez_compressed", fail_save)
    with pytest.raises(OSError, match="synthetic"):
        recording.finish()
    assert not recording.closed and len(recording.states) == 2
    monkeypatch.setattr(np, "savez_compressed", original_save)
    recording.finish()
    assert recording.closed
    assert load_archive(recording.directory)[0]["sample_count"] == 2


@pytest.mark.parametrize("failing_operation", ["sample", "finish", "viewer"])
def test_close_releases_render_resources_even_if_recording_or_viewer_fails(failing_operation):
    from go2_demo.robots.go2.backend import MujocoGo2Backend

    called = []

    def operation(name):
        def run(*args):
            called.append(name)
            if name == failing_operation:
                raise OSError("synthetic " + name + " failure")
        return run

    # No GL or policy is needed to exercise the cleanup exception path.
    backend = object.__new__(MujocoGo2Backend)
    backend.model = backend.data = None
    backend._recording = SimpleNamespace(sample=operation("sample"), finish=operation("finish"))
    backend._viewer = SimpleNamespace(close=operation("viewer"))
    backend._renderer = SimpleNamespace(close=operation("renderer"))
    with pytest.raises(OSError, match="synthetic " + failing_operation):
        backend.close()
    assert called[-2:] == ["viewer", "renderer"]
    assert backend._renderer is None
    if failing_operation != "viewer":
        assert backend._viewer is None


def test_real_go2_recording_preserves_physics_front_camera_and_final_stop(tmp_path):
    from go2_demo.memory_mission import delivery_scenario
    from go2_demo.robots.registry import create_backend

    root = Path(__file__).resolve().parents[1]
    if not (root / "assets/go2_rl_lab/deploy/pre_train/go2/go2_cts_150k.pt").exists():
        pytest.skip("Checksum-verified local Go2 assets are unavailable")
    baseline = recorded = None
    try:
        baseline = create_backend("go2", "mujoco", asset_dir=root / "assets", seed=0,
                                  scenario=delivery_scenario())
        recorded = create_backend("go2", "mujoco", asset_dir=root / "assets", seed=0,
                                  scenario=delivery_scenario())
        initial_qpos, initial_time = recorded.data.qpos.copy(), float(recorded.data.time)
        directory = tmp_path / "third_person"
        recording = recorded.start_recording(directory)
        for _ in range(6):
            command = np.array([.1, .03, 0.], dtype=np.float32)
            baseline._step_control(command)
            recorded._step_control(command)
        np.testing.assert_array_equal(baseline.data.qpos, recorded.data.qpos)
        np.testing.assert_array_equal(baseline.data.qvel, recorded.data.qvel)
        assert baseline.trajectory == recorded.trajectory
        plain_frame, recorded_frame = baseline.capture(), recorded.capture()
        np.testing.assert_array_equal(plain_frame.rgb, recorded_frame.rgb)
        np.testing.assert_array_equal(plain_frame.depth_m, recorded_frame.depth_m)
        assert plain_frame.robot_state == recorded_frame.robot_state
        before_stop_time = float(recorded.data.time)
        assert baseline.stop() == recorded.stop()
        np.testing.assert_array_equal(baseline.data.qpos, recorded.data.qpos)
        final_qpos, final_time = recorded.data.qpos.copy(), float(recorded.data.time)
        recorded.close()
        file_bytes = {p.name: p.read_bytes() for p in directory.iterdir()}
        recorded.close()
        assert {p.name: p.read_bytes() for p in directory.iterdir()} == file_bytes
        assert recording.closed

        metadata, replay_model, arrays = load_archive(directory)
        assert metadata["sim_start_s"] == initial_time
        assert metadata["sim_end_s"] == final_time
        assert final_time - before_stop_time == pytest.approx(.6)
        # Initial + both sides of every 50 Hz step + capture + final close.
        assert metadata["sample_count"] == 1 + 2 * (6 + 30) + 1 + 1
        assert np.all(np.diff(arrays["sim_time_s"]) >= 0)
        assert np.max(np.diff(arrays["sim_time_s"])) <= .0200000001
        replay_data = mujoco.MjData(replay_model)
        for index, expected_qpos in ((0, initial_qpos), (-1, final_qpos)):
            mujoco.mj_setState(replay_model, replay_data, arrays["state"][index], metadata["state_spec"])
            mujoco.mj_forward(replay_model, replay_data)
            np.testing.assert_array_equal(replay_data.qpos, expected_qpos)
    finally:
        if baseline is not None:
            baseline.close()
        if recorded is not None:
            recorded.close()
