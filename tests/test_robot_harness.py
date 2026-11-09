"""Generic contracts use explicit mock profiles, without a real robot or simulator."""

import json
import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from go2_demo.harness import Harness
from go2_demo.robots.profile import MotionLimits, RobotDimensions, RobotProfile, SensorSpec
from go2_demo.types import SensorFrame, ToolCall


def mock_profile(*, depth=True, odometry=True, lateral=True, turn=True, planar=True):
    return RobotProfile(
        version="test-1", robot_id="mock-platform", backend_id="mock", display_name="Mock contract",
        controller_id="mock-executor", model_dimensions=RobotDimensions(
            torso_box_size_m=(.2, .1, .1), nominal_base_height_m=.2, leg_count=0,
            actuated_joints=0, dimension_note="Synthetic unit-test dimensions; no hardware adapter."),
        motion=MotionLimits(
            max_translation_m=.25, max_yaw_rad=.3, settle_position_threshold_m=.01,
            settle_yaw_threshold_rad=.01, position_tolerance_m=.02, yaw_tolerance_rad=.02,
            max_forward_speed_mps=.2, max_lateral_speed_mps=.1, max_yaw_rate_radps=.3,
            stopped_speed_mps=.03, stopped_yaw_rate_radps=.03, stop_settle_s=.1,
            move_timeout_s=2., turn_timeout_s=2., max_action_timeout_s=3.),
        posture=None, sensor=SensorSpec(
            rgb_enabled=True, depth_enabled=depth, rgb_width_px=8, rgb_height_px=6,
            depth_min_m=.5, depth_max_m=3., depth_convention="optical_z_m",
            camera_frame="mock_optical", body_frame="mock_body",
            optical_axes="x right, y down, z forward", body_axes="x forward, y left, z up",
            camera_position_in_body_m=(0., 0., 0.),
            camera_optical_to_body_rotation=((0., 0., 1.), (-1., 0., 0.), (0., -1., 0.)),
            vertical_fov_deg=60.),
        planar_motion=planar, lateral_motion=lateral, turn_in_place=turn,
        height_pitch_control=False, moving_height_pitch_control=False,
        point_tracking=depth and odometry, odometry=odometry,
        simulation_mode="mock_only", time_mode="externally_timestamped",
        odometry_source="mock_relative_localization", localization_source="mock_localization",
        physics_dt_s=.01, control_decimation=1, trajectory_hz=100.)


class MockBackend:
    """The only action entry point is execute; there are no ad-hoc motion methods."""

    def __init__(self, profile):
        self.profile = profile
        self.calls = []
        self.captures = 0
        self.pose = np.array([5., -2., 0.]) if profile.odometry else None
        self.depth = 2. if profile.sensor.depth_enabled else None
        self.intrinsics_available = True
        self.timestamps_available = True
        self.same_frame_id = False
        self.localization_quality = {}
        self.depth_quality = {}

    def capture(self):
        self.captures += 1
        transform = np.eye(4) if self.pose is not None else None
        if transform is not None:
            transform[:2, 3] = self.pose[:2]
            transform[:3, :3] = self.profile.sensor.camera_optical_to_body_rotation
        return SensorFrame(
            frame_id="same-frame" if self.same_frame_id else f"frame-{self.captures}",
            sim_time_s=None, rgb=np.zeros((6, 8, 3), np.uint8),
            depth_m=None if self.depth is None else np.full((6, 8), self.depth),
            intrinsics=np.array([[8., 0., 4.], [0., 8., 3.], [0., 0., 1.]])
                       if self.intrinsics_available else None,
            camera_to_world=transform, base_xy_yaw=None if self.pose is None else self.pose.copy(),
            robot_state={"speed_mps": 0., "target_position": [99., 99.]},
            timestamp_s=100. + self.captures if self.timestamps_available else None,
            time_source="mock_sensor_clock", time_mode="externally_timestamped",
            localization_available=self.pose is not None, localization_source="mock_localization",
            localization_quality=dict(self.localization_quality), depth_available=self.depth is not None,
            depth_source="mock_depth", depth_quality=dict(self.depth_quality))

    def execute(self, call):
        self.profile.validate_action(call.name, call.arguments)
        self.calls.append(call)
        if call.name == "move_relative" and self.pose is not None:
            f, l = call.arguments["forward_m"], call.arguments["lateral_m"]
            yaw = self.pose[2]
            self.pose[:2] += .9 * np.array([math.cos(yaw) * f - math.sin(yaw) * l,
                                           math.sin(yaw) * f + math.cos(yaw) * l])
        elif call.name == "turn" and self.pose is not None:
            self.pose[2] += call.arguments["yaw_rad"]
        return {"status": "succeeded", "call_id": "untrusted-backend-id", "tool": "untrusted-label"}


def names(harness):
    return {tool["function"]["name"] for tool in harness.tools}


def test_generic_import_does_not_load_go2_simulator_or_policy():
    code = (
        "import sys; from go2_demo.harness import Harness; "
        "assert not {'mujoco', 'torch', 'go2_demo.posture', "
        "'go2_demo.robots.go2.profile'}.intersection(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_profile_or_executor_is_never_treated_as_an_implicit_go2():
    with pytest.raises(TypeError, match="explicit RobotProfile"):
        Harness(SimpleNamespace(capture=lambda: pytest.fail("capture")), "task")
    with pytest.raises(TypeError, match="execute"):
        Harness(SimpleNamespace(profile=mock_profile(), capture=lambda: pytest.fail("capture")), "task")


def test_rgb_only_profile_omits_depth_posture_and_turn_tools_and_does_not_fake_localization():
    backend = MockBackend(mock_profile(depth=False, odometry=False, lateral=False, turn=False))
    harness = Harness(backend, "Use available sensors.")
    assert names(harness) == {"observe", "move_relative", "stop", "finish", "abort"}
    observation = harness.observation()
    assert observation["local_odometry"] is None
    assert observation["tracked_points"] == []
    assert observation["sensors"]["depth"]["available"] is False
    assert observation["sensors"]["localization"]["available"] is False
    for tool, arguments in (("track_point", {"image_id": "frame-1", "u": 4., "v": 3.}),
                            ("get_geometry", {}), ("reset_posture", {}), ("turn", {"yaw_rad": .1}),
                            ("move_relative", {"forward_m": .1, "lateral_m": .1})):
        result = harness.execute(ToolCall(tool, tool, arguments))
        assert result["error_code"] == "UNSUPPORTED_CAPABILITY"
    assert backend.calls == []
    result = harness.execute(ToolCall("bad-type", "move_relative", {"forward_m": True, "lateral_m": .1}))
    assert result["error_code"] == "INVALID_ARGUMENTS"


def test_profile_limits_drive_both_instance_schema_and_validation():
    backend = MockBackend(mock_profile(lateral=False))
    harness = Harness(backend, "Move safely.")
    spec = next(t["function"]["parameters"] for t in harness.tools if t["function"]["name"] == "move_relative")
    assert spec["properties"]["forward_m"]["maximum"] == .25
    assert spec["properties"]["lateral_m"]["minimum"] == spec["properties"]["lateral_m"]["maximum"] == 0.
    assert harness.execute(ToolCall("too-far", "move_relative", {"forward_m": .26, "lateral_m": 0.}))["error_code"] == "INVALID_ARGUMENTS"
    call = ToolCall("move", "move_relative", {"forward_m": .25, "lateral_m": 0.})
    result = harness.execute(call)
    assert result["call_id"] == call.call_id and result["tool"] == call.name
    assert harness.observation()["local_odometry"]["translation_m"] == pytest.approx([.225, 0.])
    assert harness.dispatch(call)["deduplicated"] is True
    assert len(backend.calls) == 1
    spec["properties"]["forward_m"]["maximum"] = 999.
    again = next(t["function"]["parameters"] for t in harness.tools if t["function"]["name"] == "move_relative")
    assert again["properties"]["forward_m"]["maximum"] == .25


def test_motion_disabled_profile_does_not_publish_or_dispatch_motion():
    backend = MockBackend(mock_profile(planar=False, lateral=False, turn=False))
    harness = Harness(backend, "Observe.")
    assert not {"move_relative", "turn"}.intersection(names(harness))
    result = harness.execute(ToolCall("move", "move_relative", {"forward_m": .1, "lateral_m": 0.}))
    assert result["error_code"] == "UNSUPPORTED_CAPABILITY" and backend.calls == []


def test_missing_depth_or_calibration_has_explicit_unknown_geometry():
    backend = MockBackend(mock_profile())
    backend.depth = None
    harness = Harness(backend, "Observe without depth.")
    result = harness.execute(ToolCall("point", "track_point", {"image_id": "frame-1", "u": 4., "v": 3.}))
    assert result["error_code"] == "PERCEPTION_ERROR" and "Depth" in result["message"]
    assert harness.observation()["tracked_points"] == []
    backend.depth = 2.
    backend.intrinsics_available = False
    observation = harness.refresh()
    assert observation["sensors"]["depth"]["available"] is True
    assert observation["sensors"]["metric_geometry"]["available"] is False
    result = harness.execute(ToolCall("point-2", "track_point", {"image_id": "frame-2", "u": 4., "v": 3.}))
    assert result["error_code"] == "PERCEPTION_ERROR" and "intrinsics" in result["message"]


def test_profile_depth_range_is_used_by_metric_tracking():
    backend = MockBackend(mock_profile())
    backend.depth = .2
    harness = Harness(backend, "Measure.")
    result = harness.execute(ToolCall("point", "track_point", {"image_id": "frame-1", "u": 4., "v": 3.}))
    assert result["error_code"] == "PERCEPTION_ERROR" and harness.tracker.points == {}


def test_missing_initial_localization_never_becomes_a_fabricated_episode_origin():
    backend = MockBackend(mock_profile())
    backend.pose = None
    harness = Harness(backend, "Wait for localization.")
    assert harness.observation()["local_odometry"] is None
    backend.pose = np.array([10., 20., .5])
    observation = harness.refresh()
    assert observation["sensors"]["localization"]["available"] is True
    assert observation["local_odometry"] is None
    assert "Episode-start" in observation["sensors"]["localization"]["unavailable_reason"]


def test_localization_loss_invalidates_tracks_and_recovery_preserves_the_known_origin():
    backend = MockBackend(mock_profile())
    harness = Harness(backend, "Track.")
    result = harness.execute(ToolCall("point", "track_point", {"image_id": "frame-1", "u": 4., "v": 3.}))
    assert result["point"]["valid"]
    backend.pose = None
    observation = harness.refresh()
    assert observation["local_odometry"] is None
    point = observation["tracked_points"][0]
    assert not point["valid"] and point["relative_xy_m"] is None and point["range_m"] is None
    backend.pose = np.array([5.3, -1.8, .1])
    observation = harness.refresh()
    assert observation["local_odometry"]["translation_m"] == pytest.approx([.3, .2])
    assert not observation["tracked_points"][0]["valid"]


def test_sensor_time_and_quality_do_not_invent_measurements_or_expose_arbitrary_metadata():
    backend = MockBackend(mock_profile())
    backend.timestamps_available = False
    backend.localization_quality = {"confidence": .6, "age_s": .02, "target_xy_m": [99., 99.]}
    backend.depth_quality = {"valid_fraction": .9, "oracle_success": True}
    observation = Harness(backend, "Observe.").observation()
    assert observation["sim_time_s"] is None and observation["sensors"]["time"]["timestamp_s"] is None
    assert observation["sensors"]["localization"]["quality"] == {"confidence": .6, "age_s": .02}
    assert observation["sensors"]["depth"]["quality"] == {"valid_fraction": .9}
    encoded = json.dumps(observation, allow_nan=False)
    assert "target_xy_m" not in encoded and "oracle_success" not in encoded
    assert observation["local_odometry"]["source"] == "mock_localization"


def test_finish_uses_the_same_executor_and_a_retried_terminal_call_is_idempotent():
    backend = MockBackend(mock_profile())
    harness = Harness(backend, "Finish after checking.")
    call = ToolCall("finish", "finish", {})
    result = harness.execute(call)
    assert result["status"] == "succeeded" and result["stop_result"]["status"] == "succeeded"
    assert [(c.name, c.arguments) for c in backend.calls] == [("stop", {})]
    assert harness.dispatch(call) == {**result, "deduplicated": True}
    assert len(backend.calls) == 1
    assert harness.execute(ToolCall("later", "stop", {}))["error_code"] == "EPISODE_ENDED"


def test_unsupported_calls_are_cached_without_invoking_the_backend():
    backend = MockBackend(mock_profile(depth=False))
    harness = Harness(backend, "Observe.")
    call = ToolCall("geometry", "get_geometry", {})
    first = harness.execute(call)
    assert first["error_code"] == "UNSUPPORTED_CAPABILITY"
    assert harness.execute(call) == {**first, "deduplicated": True}
    assert backend.calls == []
