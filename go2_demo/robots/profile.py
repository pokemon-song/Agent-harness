"""Immutable, simulator-independent descriptions of a robot's public contract.

Limits describe supported requests, not guaranteed physical execution. In particular,
posture validation cannot replace a controller's reachability and safety checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


class UnsupportedCapabilityError(ValueError):
    """A well-formed request needs a capability this profile does not provide."""

    error_code = "UNSUPPORTED_CAPABILITY"


def _number(value: Any, name: str, *, minimum: float | None = None,
            positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _text(value: Any, name: str, *, maximum: int | None = None) -> None:
    if (not isinstance(value, str) or not value.strip()
            or (maximum is not None and len(value) > maximum)):
        raise ValueError(f"{name} must be a nonempty string"
                         + (f" of at most {maximum} characters" if maximum else ""))


def _boolean(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _integer(value: Any, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _vector(value: Any, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    for item in value:
        _number(item, name)
    return tuple(float(item) for item in value)


def _json_lists(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_lists(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_lists(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class MotionLimits:
    max_translation_m: float
    max_yaw_rad: float
    settle_position_threshold_m: float
    settle_yaw_threshold_rad: float
    position_tolerance_m: float
    yaw_tolerance_rad: float
    max_forward_speed_mps: float
    max_lateral_speed_mps: float
    max_yaw_rate_radps: float
    stopped_speed_mps: float
    stopped_yaw_rate_radps: float
    stop_settle_s: float
    move_timeout_s: float
    turn_timeout_s: float
    max_action_timeout_s: float
    low_speed_deadband_mps: float | None = None
    deadband_note: str = "The controller's low-speed deadband has not been measured."

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name not in {"deadband_note", "low_speed_deadband_mps"}:
                _number(getattr(self, name), name, minimum=0)
        for name in ("max_translation_m", "max_yaw_rad", "max_action_timeout_s"):
            _number(getattr(self, name), name, positive=True)
        if self.low_speed_deadband_mps is not None:
            _number(self.low_speed_deadband_mps, "low_speed_deadband_mps", minimum=0)
        _text(self.deadband_note, "deadband_note")
        if self.settle_position_threshold_m > self.position_tolerance_m:
            raise ValueError("Settling position threshold exceeds final position tolerance")
        if self.settle_yaw_threshold_rad > self.yaw_tolerance_rad:
            raise ValueError("Settling yaw threshold exceeds final yaw tolerance")
        if not (0 < self.move_timeout_s <= self.max_action_timeout_s
                and 0 < self.turn_timeout_s <= self.max_action_timeout_s):
            raise ValueError("Default action timeouts must be positive and within their limit")


@dataclass(frozen=True, slots=True)
class PostureLimits:
    min_height_m: float
    max_height_m: float
    max_pitch_rad: float
    half_body_length_m: float
    max_end_height_m: float
    height_tolerance_m: float
    pitch_tolerance_rad: float
    max_height_rate_mps: float
    max_pitch_rate_radps: float
    stable_duration_s: float
    stopped_speed_mps: float
    stopped_angular_speed_radps: float
    timeout_s: float
    height_reference: str = "base_above_flat_ground"
    pitch_convention: str = "nose_up_positive"

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name not in {"height_reference", "pitch_convention"}:
                _number(getattr(self, name), name, minimum=0)
        if self.min_height_m > self.max_height_m:
            raise ValueError("Posture height limits are reversed")
        _text(self.height_reference, "height_reference")
        _text(self.pitch_convention, "pitch_convention")

    def validate(self, body_height_m: float, body_pitch_rad: float) -> None:
        _number(body_height_m, "body_height_m")
        _number(body_pitch_rad, "body_pitch_rad")
        if not self.min_height_m <= body_height_m <= self.max_height_m:
            raise ValueError(f"body_height_m must be between {self.min_height_m} and {self.max_height_m} m")
        if abs(body_pitch_rad) > self.max_pitch_rad:
            raise ValueError(f"Absolute body_pitch_rad must not exceed {self.max_pitch_rad} rad")
        if body_height_m + self.half_body_length_m * abs(body_pitch_rad) > self.max_end_height_m + 1e-12:
            raise ValueError("Height and pitch together exceed the static stance reach limit")


@dataclass(frozen=True, slots=True)
class SensorSpec:
    rgb_enabled: bool
    depth_enabled: bool
    rgb_width_px: int
    rgb_height_px: int
    depth_min_m: float
    depth_max_m: float
    depth_convention: str
    camera_frame: str
    body_frame: str
    optical_axes: str
    body_axes: str
    camera_position_in_body_m: tuple[float, float, float]
    camera_optical_to_body_rotation: tuple[tuple[float, float, float], ...]
    vertical_fov_deg: float

    def __post_init__(self) -> None:
        _boolean(self.rgb_enabled, "rgb_enabled")
        _boolean(self.depth_enabled, "depth_enabled")
        _integer(self.rgb_width_px, "rgb_width_px")
        _integer(self.rgb_height_px, "rgb_height_px")
        _number(self.depth_min_m, "depth_min_m", minimum=0)
        _number(self.depth_max_m, "depth_max_m", positive=True)
        if self.depth_min_m >= self.depth_max_m:
            raise ValueError("Depth limits must increase")
        _number(self.vertical_fov_deg, "vertical_fov_deg", positive=True)
        if self.vertical_fov_deg >= 180:
            raise ValueError("Vertical field of view must be less than 180 degrees")
        for name in ("depth_convention", "camera_frame", "body_frame", "optical_axes", "body_axes"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "camera_position_in_body_m",
                           _vector(self.camera_position_in_body_m, "camera_position_in_body_m", 3))
        rotation = self.camera_optical_to_body_rotation
        if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
            raise ValueError("camera_optical_to_body_rotation must contain three rows")
        rows = tuple(_vector(row, "camera_optical_to_body_rotation", 3) for row in rotation)
        # Keep the contract a proper rigid rotation without importing numeric libraries.
        for i in range(3):
            for j in range(3):
                if abs(sum(rows[i][k] * rows[j][k] for k in range(3)) - (i == j)) > 1e-6:
                    raise ValueError("camera_optical_to_body_rotation must be orthonormal")
        determinant = (rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
                       - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
                       + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]))
        if abs(determinant - 1) > 1e-6:
            raise ValueError("camera_optical_to_body_rotation must be right-handed")
        object.__setattr__(self, "camera_optical_to_body_rotation", rows)


@dataclass(frozen=True, slots=True)
class RobotDimensions:
    """Model measurements, explicitly distinguished from a motion clearance envelope."""

    torso_box_size_m: tuple[float, float, float]
    nominal_base_height_m: float
    leg_count: int
    actuated_joints: int
    dimension_note: str

    def __post_init__(self) -> None:
        dimensions = _vector(self.torso_box_size_m, "torso_box_size_m", 3)
        if any(value <= 0 for value in dimensions):
            raise ValueError("Torso dimensions must be positive")
        object.__setattr__(self, "torso_box_size_m", dimensions)
        _number(self.nominal_base_height_m, "nominal_base_height_m", positive=True)
        _integer(self.leg_count, "leg_count", minimum=0)
        _integer(self.actuated_joints, "actuated_joints", minimum=0)
        _text(self.dimension_note, "dimension_note")


@dataclass(frozen=True, slots=True)
class RobotProfile:
    version: str
    robot_id: str
    backend_id: str
    display_name: str
    controller_id: str
    model_dimensions: RobotDimensions
    motion: MotionLimits
    posture: PostureLimits | None
    sensor: SensorSpec
    planar_motion: bool
    lateral_motion: bool
    turn_in_place: bool
    height_pitch_control: bool
    moving_height_pitch_control: bool
    point_tracking: bool
    simulation_mode: str
    time_mode: str
    odometry_source: str
    localization_source: str
    physics_dt_s: float
    control_decimation: int
    trajectory_hz: float
    odometry: bool = True

    def __post_init__(self) -> None:
        for name in ("version", "robot_id", "backend_id", "display_name", "controller_id",
                     "simulation_mode", "time_mode", "odometry_source", "localization_source"):
            _text(getattr(self, name), name)
        for name, cls in (("model_dimensions", RobotDimensions), ("motion", MotionLimits), ("sensor", SensorSpec)):
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"{name} must be a {cls.__name__}")
        if self.posture is not None and not isinstance(self.posture, PostureLimits):
            raise ValueError("posture must be PostureLimits or None")
        for name in ("planar_motion", "lateral_motion", "turn_in_place", "height_pitch_control",
                     "moving_height_pitch_control", "point_tracking", "odometry"):
            _boolean(getattr(self, name), name)
        if self.lateral_motion and not self.planar_motion:
            raise ValueError("Lateral motion requires planar motion")
        if self.height_pitch_control and self.posture is None:
            raise ValueError("Height and pitch control requires posture limits")
        if self.moving_height_pitch_control and not (self.planar_motion and self.height_pitch_control):
            raise ValueError("Moving posture control requires planar and posture control")
        if self.point_tracking and not (self.sensor.rgb_enabled and self.sensor.depth_enabled and self.odometry):
            raise ValueError("Current point tracking requires RGB, metric depth and odometry")
        _number(self.physics_dt_s, "physics_dt_s", positive=True)
        _integer(self.control_decimation, "control_decimation")
        _number(self.trajectory_hz, "trajectory_hz", positive=True)

    @property
    def capabilities(self) -> dict[str, bool]:
        """Return a detached dictionary so callers cannot mutate the profile."""
        return {"planar_motion": self.planar_motion, "lateral_motion": self.lateral_motion,
                "turn_in_place": self.turn_in_place, "height_pitch_control": self.height_pitch_control,
                "moving_height_pitch_control": self.moving_height_pitch_control,
                "rgb": self.sensor.rgb_enabled, "depth": self.sensor.depth_enabled,
                "point_tracking": self.point_tracking, "odometry": self.odometry}

    def to_dict(self) -> dict:
        result = _json_lists(asdict(self))
        result["capabilities"] = self.capabilities
        return result

    def validate_action(self, name: str, args: dict) -> None:
        """Validate exact public tool arguments and capabilities without executing.

        This deliberately does not validate call IDs, selected-image freshness,
        current mode, task truth, stopping success or inverse-kinematic reachability.
        Those require live episode/controller state.
        """
        schemas = {
            "observe": (), "track_point": ("image_id", "u", "v"), "get_geometry": (),
            "move_relative": ("forward_m", "lateral_m"), "turn": ("yaw_rad",),
            "set_posture": ("body_height_m", "body_pitch_rad"), "reset_posture": (),
            "stop": (), "finish": (), "abort": ("reason",),
        }
        if not isinstance(name, str) or name not in schemas:
            raise ValueError("Unknown or unsupported tool")
        if not isinstance(args, dict):
            raise ValueError("Arguments must be an object")
        if set(args) != set(schemas[name]):
            raise ValueError("Arguments must match exactly: " + ", ".join(schemas[name]))
        for key, value in args.items():
            if key in {"image_id", "reason"}:
                _text(value, key, maximum=300)
            else:
                _number(value, key)
        if name == "move_relative":
            if math.hypot(args["forward_m"], args["lateral_m"]) > self.motion.max_translation_m:
                raise ValueError(f"Translation length exceeds {self.motion.max_translation_m} m")
            if not self.planar_motion:
                raise UnsupportedCapabilityError("This robot does not support planar motion")
            if args["lateral_m"] != 0 and not self.lateral_motion:
                raise UnsupportedCapabilityError("This robot does not support lateral translation")
        elif name == "turn":
            if abs(args["yaw_rad"]) > self.motion.max_yaw_rad:
                raise ValueError(f"Yaw exceeds {self.motion.max_yaw_rad} rad")
            if not self.turn_in_place:
                raise UnsupportedCapabilityError("This robot does not support turning in place")
        elif name in {"set_posture", "reset_posture"}:
            if name == "set_posture" and self.posture is not None:
                self.posture.validate(**args)
            if not self.height_pitch_control:
                raise UnsupportedCapabilityError("This robot does not support static height and pitch control")
        elif name in {"track_point", "get_geometry"} and not self.point_tracking:
            raise UnsupportedCapabilityError("This robot does not support RGB-D point tracking")
