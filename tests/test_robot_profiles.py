"""Profiles validate real capability boundaries without constructing a simulator."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from go2_demo.robots import (RobotBackend, UnsupportedCapabilityError, create_backend,
                             get_profile, list_robots)
from go2_demo.robots.go2.profile import GO2_PROFILE
from go2_demo.robots import registry
from go2_demo.robots.contracts import IndependentTaskEvaluator, SimulationEpisodeBackend


def test_go2_profile_keeps_public_limits_and_model_measurements():
    profile = get_profile("go2", "mujoco")
    assert profile is GO2_PROFILE
    assert (profile.version, profile.robot_id, profile.backend_id) == ("1", "go2", "mujoco")
    assert profile.display_name == "Unitree Go2"
    assert profile.controller_id == "go2_cts_150k_torchscript_pd_v1"
    assert profile.motion.max_translation_m == profile.motion.max_yaw_rad == .6
    assert profile.motion.settle_position_threshold_m == .06
    assert profile.motion.settle_yaw_threshold_rad == .045
    assert profile.motion.position_tolerance_m == profile.motion.yaw_tolerance_rad == .1
    assert profile.motion.stopped_speed_mps == .12
    assert profile.motion.stopped_yaw_rate_radps == .25
    assert profile.motion.stop_settle_s == .6
    assert (profile.motion.move_timeout_s, profile.motion.turn_timeout_s) == (8, 6)
    assert profile.motion.low_speed_deadband_mps is None
    assert profile.physics_dt_s * profile.control_decimation == 1 / profile.trajectory_hz == .02
    assert profile.simulation_mode == "paused_during_model_inference"
    assert profile.time_mode == "paused_between_tools"
    assert profile.odometry_source == "idealized_simulated_proprioception_relative_to_episode_start"
    assert profile.localization_source == "simulated_proprioception"
    assert profile.model_dimensions.actuated_joints == 12
    assert profile.model_dimensions.leg_count == 4
    assert "not a full-body clearance envelope" in profile.model_dimensions.dimension_note

    # These are the pinned model's torso-box dimensions, not catalogue dimensions
    # or a foot/limb envelope inferred from a convenient robot pose.
    xml = Path(__file__).resolve().parents[1] / "assets/go2_rl_lab/resources/robots/go2/go2.xml"
    if xml.exists():
        geom = ET.parse(xml).getroot().find(".//geom[@name='base_box']")
        assert profile.model_dimensions.torso_box_size_m == tuple(2 * float(n) for n in geom.attrib["size"].split())


def test_profile_and_nested_values_are_immutable_and_serialization_is_detached():
    with pytest.raises(FrozenInstanceError):
        GO2_PROFILE.robot_id = "other"
    with pytest.raises(FrozenInstanceError):
        GO2_PROFILE.motion.max_translation_m = 10
    capabilities = GO2_PROFILE.capabilities
    capabilities["depth"] = False
    assert GO2_PROFILE.capabilities["depth"] is True
    first = GO2_PROFILE.to_dict()
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    first["sensor"]["camera_position_in_body_m"][0] = 99
    first["motion"]["max_translation_m"] = 99
    assert GO2_PROFILE.sensor.camera_position_in_body_m == (.32, 0, .08)
    assert GO2_PROFILE.motion.max_translation_m == .6


def test_mutable_input_vectors_are_copied_into_immutable_values():
    position = [.1, 0, .2]
    rotation = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    sensor = replace(GO2_PROFILE.sensor, camera_position_in_body_m=position,
                     camera_optical_to_body_rotation=rotation)
    position[0] = 100
    rotation[0][2] = 100
    assert sensor.camera_position_in_body_m == (.1, 0, .2)
    assert sensor.camera_optical_to_body_rotation[0] == (0, 0, 1)


@pytest.mark.parametrize(("name", "arguments"), [
    ("observe", {}), ("get_geometry", {}), ("stop", {}), ("finish", {}),
    ("abort", {"reason": "Cannot safely continue"}),
    ("track_point", {"image_id": "frame-1", "u": 100.5, "v": 40}),
    ("move_relative", {"forward_m": .6, "lateral_m": 0}),
    ("move_relative", {"forward_m": 0, "lateral_m": -.6}),
    ("turn", {"yaw_rad": -.6}),
    ("set_posture", {"body_height_m": .24, "body_pitch_rad": -.2}),
    ("set_posture", {"body_height_m": .36, "body_pitch_rad": 0}),
    ("reset_posture", {}),
])
def test_valid_action_boundaries(name, arguments):
    original = dict(arguments)
    assert GO2_PROFILE.validate_action(name, arguments) is None
    assert arguments == original


@pytest.mark.parametrize(("name", "arguments"), [
    ("move_relative", {"forward_m": .5, "lateral_m": .5}),
    ("move_relative", {"forward_m": .6000001, "lateral_m": 0}),
    ("turn", {"yaw_rad": .6000001}),
    ("move_relative", {"forward_m": .1}),
    ("stop", {"speed": 1}),
    ("turn", {"yaw_rad": True}),
    ("turn", {"yaw_rad": float("nan")}),
    ("turn", {"yaw_rad": float("inf")}),
    ("turn", {"yaw_rad": 10 ** 10000}),
    ("turn", {"yaw_rad": "0.1"}),
    ("track_point", {"image_id": "frame-1", "u": False, "v": 1}),
    ("track_point", {"image_id": "", "u": 1, "v": 1}),
    ("abort", {"reason": "x" * 301}),
    ("set_posture", {"body_height_m": .36, "body_pitch_rad": .2}),
    ("set_posture", {"body_height_m": .23999, "body_pitch_rad": 0}),
    ("set_posture", {"body_height_m": .3, "body_pitch_rad": .20001}),
    ("set_posture", {"body_height_m": True, "body_pitch_rad": 0}),
    ("unknown_action", {}), (None, {}), ("stop", []),
])
def test_invalid_actions_raise_value_error(name, arguments):
    with pytest.raises(ValueError) as error:
        GO2_PROFILE.validate_action(name, arguments)
    assert not isinstance(error.value, UnsupportedCapabilityError)


def test_posture_joint_domain_matches_existing_controller_contract():
    limits = GO2_PROFILE.posture
    assert (limits.min_height_m, limits.max_height_m, limits.max_pitch_rad) == (.24, .36, .2)
    assert (limits.half_body_length_m, limits.max_end_height_m) == (.1934, .38)
    boundary_height = limits.max_end_height_m - limits.half_body_length_m * .2
    limits.validate(boundary_height, .2)
    limits.validate(boundary_height + 5e-13, .2)
    with pytest.raises(ValueError):
        limits.validate(boundary_height + 2e-12, .2)


def test_unavailable_lateral_motion_is_rejected_without_an_implicit_curve():
    profile = replace(GO2_PROFILE, robot_id="test_differential", lateral_motion=False)
    profile.validate_action("move_relative", {"forward_m": .3, "lateral_m": 0})
    with pytest.raises(UnsupportedCapabilityError, match="lateral") as error:
        profile.validate_action("move_relative", {"forward_m": .3, "lateral_m": .01})
    assert error.value.error_code == "UNSUPPORTED_CAPABILITY"
    assert profile.capabilities["lateral_motion"] is False


@pytest.mark.parametrize(("changes", "name", "arguments"), [
    ({"planar_motion": False, "lateral_motion": False}, "move_relative", {"forward_m": .1, "lateral_m": 0}),
    ({"turn_in_place": False}, "turn", {"yaw_rad": .1}),
    ({"posture": None, "height_pitch_control": False}, "set_posture", {"body_height_m": .3, "body_pitch_rad": 0}),
    ({"posture": None, "height_pitch_control": False}, "reset_posture", {}),
    ({"point_tracking": False}, "track_point", {"image_id": "frame-1", "u": 1, "v": 1}),
    ({"point_tracking": False}, "get_geometry", {}),
])
def test_missing_capabilities_explicitly_reject_actions(changes, name, arguments):
    profile = replace(GO2_PROFILE, **changes)
    with pytest.raises(UnsupportedCapabilityError):
        profile.validate_action(name, arguments)


def test_bad_arguments_are_rejected_before_missing_capabilities():
    profile = replace(GO2_PROFILE, posture=None, height_pitch_control=False, turn_in_place=False)
    for name, arguments in [("set_posture", {"body_height_m": True, "body_pitch_rad": 0}),
                            ("reset_posture", {"extra": 1}), ("turn", {"yaw_rad": .7})]:
        with pytest.raises(ValueError) as error:
            profile.validate_action(name, arguments)
        assert not isinstance(error.value, UnsupportedCapabilityError)


@pytest.mark.parametrize("changes", [
    {"lateral_motion": True, "planar_motion": False},
    {"height_pitch_control": True, "posture": None},
    {"moving_height_pitch_control": True, "height_pitch_control": False},
    {"odometry": False}, {"odometry": 1},
    {"planar_motion": 1}, {"physics_dt_s": True}, {"control_decimation": 0},
])
def test_inconsistent_or_non_boolean_profiles_cannot_be_constructed(changes):
    with pytest.raises(ValueError):
        replace(GO2_PROFILE, **changes)


@pytest.mark.parametrize("rotation", [
    [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 1, 0]],
])
def test_sensor_rejects_non_rigid_or_reflected_calibration(rotation):
    with pytest.raises(ValueError):
        replace(GO2_PROFILE.sensor, camera_optical_to_body_rotation=rotation)


def test_registry_only_lists_implemented_robot():
    assert list_robots() == ("go2",)


@pytest.mark.parametrize(("robot_id", "backend_id"), [
    ("unknown", "mujoco"), ("go2", "hardware"), ("GO2", "mujoco"), (None, "mujoco"),
])
def test_unknown_selection_fails_before_adapter_import(monkeypatch, robot_id, backend_id):
    def unexpected_import(*_):
        pytest.fail("Unknown selection must not import an adapter")
    monkeypatch.setattr(registry, "import_module", unexpected_import)
    with pytest.raises(ValueError):
        get_profile(robot_id, backend_id)
    with pytest.raises(ValueError):
        create_backend(robot_id, backend_id, asset_dir="unused")


def test_factory_lazily_constructs_only_the_selected_adapter(monkeypatch):
    imported, constructed = [], []
    backend = object()

    def make_backend(**kwargs):
        constructed.append(kwargs)
        return backend

    def load(name):
        imported.append(name)
        return SimpleNamespace(MujocoGo2Backend=make_backend)

    monkeypatch.setattr(registry, "import_module", load)
    assert get_profile() is GO2_PROFILE
    assert imported == []
    assert create_backend(asset_dir="assets", seed=7) is backend
    assert imported == ["go2_demo.robots.go2.backend"]
    assert constructed == [{"asset_dir": "assets", "seed": 7}]


def test_importing_registry_needs_no_simulator_or_numeric_libraries():
    script = """
import sys
from importlib.abc import MetaPathFinder
class RejectHeavy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'torch', 'numpy', 'cv2'}:
            raise AssertionError('Unexpected heavyweight import: ' + fullname)
sys.meta_path.insert(0, RejectHeavy())
from go2_demo.robots import RobotBackend, get_profile, list_robots
assert get_profile().robot_id == 'go2'
assert list_robots() == ('go2',)
assert 'go2_demo.robots.go2.backend' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=10)
    assert result.returncode == 0, result.stderr


def test_backend_protocol_checks_required_surface_at_runtime():
    class Adapter:
        profile = GO2_PROFILE

        def capture(self):
            return None

        def execute(self, call):
            return {"status": "succeeded"}

        def stop(self):
            return {"status": "succeeded"}

        def cancel(self):
            return {"status": "succeeded", "cancelled": False, "reason": "no_active_action"}

        def close(self):
            pass

    assert isinstance(Adapter(), RobotBackend)
    assert not isinstance(SimpleNamespace(profile=GO2_PROFILE), RobotBackend)


def test_simulation_episode_contract_is_stricter_than_generic_backend():
    class GenericAdapter:
        profile = GO2_PROFILE

        def capture(self):
            return None

        def execute(self, call):
            return {"status": "succeeded"}

        def stop(self):
            return {"status": "succeeded"}

        def cancel(self):
            return {"status": "succeeded", "cancelled": False}

        def close(self):
            pass

    class Evaluator:
        def criteria(self, standoff_m=1., tolerance_m=.18):
            return {"goal_kind": "standoff", "standoff_m": standoff_m}

        def evaluate(self, standoff_m=1., tolerance_m=.18):
            return {"success": False}

    class SimulationAdapter(GenericAdapter):
        evaluator = Evaluator()
        trajectory = []
        deadline_monotonic = None
        sim_deadline_s = None

    assert isinstance(GenericAdapter(), RobotBackend)
    assert not isinstance(GenericAdapter(), SimulationEpisodeBackend)
    simulation = SimulationAdapter()
    assert isinstance(simulation, SimulationEpisodeBackend)
    assert isinstance(simulation.evaluator, IndependentTaskEvaluator)
    assert not isinstance(SimpleNamespace(evaluate=lambda: {}), IndependentTaskEvaluator)
    # A structural check must not be mistaken for checking attribute value types.
    simulation.evaluator = None
    assert isinstance(simulation, SimulationEpisodeBackend)
    assert not isinstance(simulation.evaluator, IndependentTaskEvaluator)
