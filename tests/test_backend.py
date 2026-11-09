"""Backend state contracts can be checked without a viewer or model API."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from go2_demo.backend import MujocoGo2Backend


@pytest.mark.parametrize("height,tilt_cos,position_value,expected", [
    (0.34, 1.0, 0.0, False),
    (0.15, 1.0, 0.0, True),
    (0.34, 0.5, 0.0, True),
    (0.34, 1.0, float("nan"), True),
    (0.34, 1.0, float("inf"), True),
])
def test_fall_state_is_correct_and_serializable_for_api_observations(height, tilt_cos, position_value, expected):
    backend = object.__new__(MujocoGo2Backend)
    backend.data = SimpleNamespace(qpos=np.array([position_value, 0.0, height]))
    tilt_sin = np.sqrt(1 - tilt_cos**2)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, tilt_cos, -tilt_sin], [0.0, tilt_sin, tilt_cos]])
    backend._base_pose = lambda: (np.array([0.0, 0.0, height]), 0.0, rotation)

    state = backend._fallen()
    assert state is expected
    assert json.loads(json.dumps({"fallen": state}))["fallen"] is expected


def test_capture_explicitly_declares_simulated_clock_depth_and_localization():
    class Renderer:
        depth = False

        def disable_depth_rendering(self):
            self.depth = False

        def enable_depth_rendering(self):
            self.depth = True

        def update_scene(self, data, camera):
            assert camera == "front_rgbd"

        def render(self):
            return (np.ones((360, 640), dtype=np.float32) if self.depth
                    else np.zeros((360, 640, 3), dtype=np.uint8))

    backend = object.__new__(MujocoGo2Backend)
    backend._renderer = Renderer()
    backend._frame_counter = 0
    backend._camera_id = 0
    backend._fallen_ever = False
    backend._posture_controller = backend._posture_target = None
    backend.data = SimpleNamespace(time=5., qvel=np.zeros(6), cam_xmat=np.eye(3).reshape(1, 9),
                                   cam_xpos=np.array([[.32, 0, .42]]))
    backend.model = SimpleNamespace(cam_fovy=np.array([60.]))
    backend._base_pose = lambda: (np.array([0., 0., .34]), 0., np.eye(3))
    frame = backend.capture()
    assert frame.timestamp_s == frame.sim_time_s == 5.
    assert frame.time_source == "simulation"
    assert frame.time_mode == backend.profile.time_mode == "paused_between_tools"
    assert frame.localization_available is True
    assert frame.localization_source == "simulated_proprioception"
    assert frame.depth_available is True and frame.depth_source == "simulated_rgbd"
    assert not {"target_xy_m", "success", "passage_x_m"}.intersection(frame.robot_state)
