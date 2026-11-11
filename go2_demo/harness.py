"""Whitelisted synchronous tools. No task planning or hidden-target access."""

from copy import deepcopy
import json
import math

from .perception import PerceptionError, PointTracker, depth_frame_error, localization_pose, metric_frame_error
from .robots.profile import RobotProfile, UnsupportedCapabilityError
from .types import ToolCall


def _tool(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": required or [], "additionalProperties": False}}}


def build_tools(profile: RobotProfile) -> list[dict]:
    """Describe only the actions admitted by this explicit robot profile."""
    if not isinstance(profile, RobotProfile):
        raise TypeError("An explicit RobotProfile is required")
    motion, capabilities = profile.motion, profile.capabilities
    tools = [_tool("observe", "Refresh the current sensor observation; no movement.")]
    if capabilities["point_tracking"] and capabilities["depth"]:
        tools.extend([
            _tool("track_point", "Select a static visible point in the latest RGB image. Pixels use origin top-left, u right, v down. Depth gives geometry at that pixel, not semantic identity. Valid tracked points update after motion; lost points have unknown geometry. Requires available calibrated depth and localization.",
                  {"image_id": {"type": "string"}, "u": {"type": "number"}, "v": {"type": "number"}},
                  ["image_id", "u", "v"]),
            _tool("get_geometry", "Read tracked points; lost points have no current metric geometry."),
        ])
    if capabilities["planar_motion"]:
        lateral_limit = motion.max_translation_m if capabilities["lateral_motion"] else 0.
        description = (f"Target displacement in the base horizontal frame at action start: forward positive, left positive, metres; vector length at most {motion.max_translation_m:g} m. "
                       f"Settling starts within {motion.settle_position_threshold_m:g} m; final position tolerance is {motion.position_tolerance_m:g} m. "
                       "Small requests can succeed with almost no motion. Use measured feedback and available local_odometry; do not sum requested distances. No path planning or obstacle avoidance.")
        if not capabilities["lateral_motion"]:
            description += " Lateral motion is unsupported; lateral_m must be zero."
        if capabilities["height_pitch_control"] and not capabilities["moving_height_pitch_control"]:
            description += " In static posture mode explicitly call reset_posture before moving."
        tools.append(_tool("move_relative", description,
                           {"forward_m": {"type": "number", "minimum": -motion.max_translation_m, "maximum": motion.max_translation_m},
                            "lateral_m": {"type": "number", "minimum": -lateral_limit, "maximum": lateral_limit}},
                           ["forward_m", "lateral_m"]))
    if capabilities["turn_in_place"]:
        description = f"Relative yaw in radians, positive left; absolute angle at most {motion.max_yaw_rad:g} rad."
        if capabilities["height_pitch_control"] and not capabilities["moving_height_pitch_control"]:
            description += " Static posture requires explicit reset_posture before turning."
        tools.append(_tool("turn", description,
                           {"yaw_rad": {"type": "number", "minimum": -motion.max_yaw_rad, "maximum": motion.max_yaw_rad}}, ["yaw_rad"]))
    if capabilities["height_pitch_control"]:
        posture = profile.posture
        tools.extend([
            _tool("set_posture", f"Request a stationary absolute body height ({posture.height_reference}) and pitch ({posture.pitch_convention}). Physical execution requires measured confirmation. Stop keeps posture; walking or turning requires explicit reset_posture. Height and pitch must satisfy body_height_m + {posture.half_body_length_m:g} * abs(body_pitch_rad) <= {posture.max_end_height_m:g}, plus actual stance reachability.",
                  {"body_height_m": {"type": "number", "minimum": posture.min_height_m, "maximum": posture.max_height_m},
                   "body_pitch_rad": {"type": "number", "minimum": -posture.max_pitch_rad, "maximum": posture.max_pitch_rad}},
                  ["body_height_m", "body_pitch_rad"]),
            _tool("reset_posture", "Explicitly recover the configured standard standing posture and locomotion controller. Check clearance and measured feedback before moving."),
        ])
    tools.extend([
        _tool("stop", "Request a physical stop and check measured feedback; an active static posture is retained."),
        _tool("finish", "Declare completion only after checking public requirements against measurements. The camera alone does not prove passage clearance. Independent evaluation decides success."),
        _tool("abort", "Stop the episode without claiming success.", {"reason": {"type": "string"}}, ["reason"]),
    ])
    return tools


def __getattr__(name):
    if name == "TOOLS":
        # Deprecated Go2 compatibility export. Importing the generic Harness
        # alone never chooses a robot or loads a simulator/controller.
        from .robots.go2.profile import GO2_PROFILE
        return build_tools(GO2_PROFILE)
    raise AttributeError(name)

_ROBOT_FIELDS = {"body_height_m", "body_pitch_rad", "body_roll_rad", "speed_mps",
                 "yaw_rate_radps", "fallen", "motion_state", "linear_velocity_mps",
                 "angular_velocity_radps", "posture_control_supported"}

_CALIBRATION_TEXT_FIELDS = {"camera_frame", "body_frame", "optical_axes", "body_axes"}
_COMPLETION_TEXT_FIELDS = {"goal_kind", "reference_body"}
_COMPLETION_BOOL_FIELDS = {"stop_required", "collision_free_required", "stages_ordered", "heading_required"}
_COMPLETION_NUMBER_FIELDS = {
    "required_clearance_m", "standoff_m", "distance_tolerance_m",
    "heading_tolerance_rad", "max_stopped_speed_mps", "max_stopped_yaw_rate_radps",
    "retrace_tolerance_m", "min_outbound_coverage",
}


def _finite_number(value) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _public_calibration(value) -> dict:
    """Copy the supported calibration schema without arbitrary backend metadata."""
    if not isinstance(value, dict):
        return {}
    result = {key: item for key, item in value.items()
              if key in _CALIBRATION_TEXT_FIELDS and isinstance(item, str)}
    position = value.get("camera_position_in_body_m")
    if isinstance(position, (list, tuple)) and len(position) == 3 and all(_finite_number(x) for x in position):
        result["camera_position_in_body_m"] = [float(x) for x in position]
    rotation = value.get("camera_optical_to_body_rotation")
    if (isinstance(rotation, (list, tuple)) and len(rotation) == 3
            and all(isinstance(row, (list, tuple)) and len(row) == 3
                    and all(_finite_number(x) for x in row) for row in rotation)):
        result["camera_optical_to_body_rotation"] = [[float(x) for x in row] for row in rotation]
    return result


def _public_completion_requirements(value) -> dict:
    """Expose declared tolerances, never gate locations or evaluator state."""
    if not isinstance(value, dict):
        return {}
    result = {key: float(item) if key in _COMPLETION_NUMBER_FIELDS else item
            for key, item in value.items()
            if ((key in _COMPLETION_TEXT_FIELDS and isinstance(item, str))
                or (key in _COMPLETION_BOOL_FIELDS and isinstance(item, bool))
                or (key in _COMPLETION_NUMBER_FIELDS and _finite_number(item) and item >= 0))}
    rules = value.get("stage_rules")
    if isinstance(rules, (list, tuple)):
        public_rules = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if not all(isinstance(rule.get(key), str) and rule[key].strip()
                       for key in ("stage_id", "instruction")):
                continue
            public_rule = {key: rule[key] for key in ("stage_id", "instruction")}
            clearance = rule.get("required_clearance_m")
            if _finite_number(clearance) and clearance >= 0:
                public_rule["required_clearance_m"] = float(clearance)
            public_rules.append(public_rule)
        if public_rules:
            result["stage_rules"] = public_rules
    return result


def _public_robot_state(value: dict) -> dict:
    result = {key: item for key, item in value.items() if key in _ROBOT_FIELDS}
    if value.get("pitch_convention") == "nose_up_positive":
        result["pitch_convention"] = "nose_up_positive"
    mode = value.get("control_mode")
    if isinstance(mode, str) and mode in {"locomotion", "posture"}:
        result["control_mode"] = mode
    target = value.get("posture_target")
    if (isinstance(target, dict)
            and all(_finite_number(target.get(key)) for key in ("body_height_m", "body_pitch_rad"))):
        result["posture_target"] = {key: float(target[key]) for key in ("body_height_m", "body_pitch_rad")}
    return result


def _public_quality(value) -> dict:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("valid", "stale"):
        if type(value.get(key)) is bool:
            result[key] = value[key]
    for key in ("confidence", "age_s", "position_std_m", "yaw_std_rad", "valid_fraction"):
        if _finite_number(value.get(key)) and value[key] >= 0:
            result[key] = float(value[key])
    for key in ("status", "reason"):
        if isinstance(value.get(key), str):
            result[key] = value[key]
    return result


class Harness:
    def __init__(self, backend, task: str, *, odometry_origin=None):
        profile = getattr(backend, "profile", None)
        if not isinstance(profile, RobotProfile):
            raise TypeError("Backend must provide an explicit RobotProfile; implicit Go2 defaults are unsupported")
        if not callable(getattr(backend, "execute", None)):
            raise TypeError("Backend must implement execute(ToolCall)")
        self.backend, self.task = backend, task
        self.profile = profile
        self._tools = build_tools(profile)
        self.tracker = PointTracker(min_depth_m=profile.sensor.depth_min_m, max_depth_m=profile.sensor.depth_max_m)
        self.frame = None
        if odometry_origin is not None and (
                not isinstance(odometry_origin, (list, tuple)) or len(odometry_origin) != 3
                or not all(_finite_number(value) for value in odometry_origin)):
            raise ValueError("Odometry origin must be three finite measured x/y/yaw values")
        self._initial_base_xy_yaw: tuple[float, float, float] | None = (
            tuple(float(value) for value in odometry_origin) if odometry_origin is not None else None)
        self._odometry_reference = ("mission_start_body_horizontal" if odometry_origin is not None
                                    else "episode_start_body_horizontal")
        self._initial_capture_complete = False
        self.termination: str | None = None
        self.cache: dict[str, tuple[str, dict]] = {}
        self.refresh()

    @property
    def tools(self) -> list[dict]:
        return deepcopy(self._tools)

    def refresh(self) -> dict:
        self.frame = self.backend.capture()
        if not self._initial_capture_complete:
            if self.profile.capabilities.get("odometry", False) and self._initial_base_xy_yaw is None:
                try:
                    self._initial_base_xy_yaw = tuple(float(x) for x in localization_pose(self.frame))
                except PerceptionError:
                    pass
            self._initial_capture_complete = True
        if self.profile.capabilities["point_tracking"]:
            self.tracker.update(self.frame)
        return self.observation()

    def observation(self) -> dict:
        frame = self.frame
        capabilities = self.profile.capabilities
        pose, localization_problem = None, None
        if capabilities.get("odometry", False):
            try:
                pose = localization_pose(frame)
            except PerceptionError as exc:
                localization_problem = str(exc)
        else:
            localization_problem = "Localization is unsupported by the robot profile"
        odometry = None
        source = (frame.localization_source if frame.localization_source != "unspecified"
                  else self.profile.localization_source)
        if pose is not None and self._initial_base_xy_yaw is not None:
            initial_x, initial_y, initial_yaw = self._initial_base_xy_yaw
            x, y, yaw = (float(value) for value in pose)
            dx, dy = x - initial_x, y - initial_y
            c, s = math.cos(initial_yaw), math.sin(initial_yaw)
            yaw_delta = yaw - initial_yaw
            odometry = {"translation_m": [c * dx + s * dy, -s * dx + c * dy],
                        "yaw_rad": math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)),
                        "reference": self._odometry_reference, "source": source}
        elif pose is not None:
            localization_problem = "Episode-start localization was unavailable; relative odometry is unknown"
        depth_problem = depth_frame_error(frame) if capabilities["depth"] else "Depth is unsupported by the robot profile"
        geometry_problem = (metric_frame_error(frame) if capabilities["point_tracking"]
                            else "Metric point tracking is unsupported by the robot profile")
        observation = {"task": self.task, "image_id": frame.frame_id,
                "sim_time_s": frame.sim_time_s if _finite_number(frame.sim_time_s) else None,
                "image_size": {"width": frame.rgb.shape[1], "height": frame.rgb.shape[0]},
                "coordinate_convention": "tracked_points.relative_xy_m: origin at the current base body reference, horizontal forward and left axes, metres; yaw positive left. local_odometry: origin and horizontal axes fixed at the episode-start base body reference.",
                "local_odometry": odometry,
                "robot_state": _public_robot_state(frame.robot_state),
                "tracked_points": self.tracker.geometry(),
                "capabilities": capabilities,
                "sensors": {
                    "time": {"timestamp_s": frame.timestamp_s if _finite_number(frame.timestamp_s) else None,
                             "source": frame.time_source,
                             "mode": frame.time_mode if frame.time_mode != "unspecified" else self.profile.time_mode},
                    "depth": {"available": depth_problem is None, "source": frame.depth_source,
                              "quality": _public_quality(frame.depth_quality), "unavailable_reason": depth_problem},
                    "localization": {"available": pose is not None, "source": source,
                                     "quality": _public_quality(frame.localization_quality),
                                     "unavailable_reason": localization_problem},
                    "metric_geometry": {"available": geometry_problem is None,
                                        "unavailable_reason": geometry_problem},
                },
                "simulation_mode": self.profile.simulation_mode}
        if self._odometry_reference == "mission_start_body_horizontal":
            observation["coordinate_convention"] = (
                "tracked_points.relative_xy_m: origin at the current base body reference, horizontal forward and left axes, metres; yaw positive left. "
                "local_odometry: origin and horizontal axes fixed at the continuous mission's initial base pose; preserved across planning-context resets.")
        if self._posture_supported():
            posture = self.profile.posture
            observation["posture_limits"] = {
                "body_height_m": {"minimum": posture.min_height_m, "maximum": posture.max_height_m},
                "body_pitch_rad": {"minimum": -posture.max_pitch_rad, "maximum": posture.max_pitch_rad},
                "height_reference": posture.height_reference,
                "pitch_convention": posture.pitch_convention,
                "combined_constraint": f"body_height_m + {posture.half_body_length_m} * abs(body_pitch_rad) <= {posture.max_end_height_m}",
            }
        calibration = _public_calibration(getattr(self.backend, "sensor_calibration", None))
        if calibration:
            observation["sensor_calibration"] = calibration
        requirements = _public_completion_requirements(getattr(self.backend, "public_task_spec", None))
        if requirements:
            observation["completion_requirements"] = requirements
        return observation

    def _posture_supported(self) -> bool:
        return self.profile.capabilities["height_pitch_control"]

    @staticmethod
    def _result(call, status="succeeded", **kwargs):
        return {"call_id": call.call_id, "tool": call.name, "status": status, **kwargs}

    def _validate(self, call: ToolCall) -> None:
        if not isinstance(call.call_id, str) or not 0 < len(call.call_id) <= 200:
            raise ValueError("Invalid call_id")
        if not isinstance(call.arguments, dict):
            raise ValueError("Arguments must be an object")
        self.profile.validate_action(call.name, call.arguments)

    def dispatch_many(self, calls: list[ToolCall]) -> list[dict]:
        if len(calls) != 1:
            return [self._result(c, "failed", error_code="ONE_TOOL_PER_TURN",
                                 message="No calls executed. Request exactly one tool per turn.") for c in calls]
        return [self.execute(calls[0])]

    def dispatch(self, call: ToolCall) -> dict:
        """Compatibility alias; all calls use the same validation/cache path."""
        return self.execute(call)

    def execute(self, call: ToolCall) -> dict:
        unsupported = None
        try:
            try:
                self._validate(call)
            except UnsupportedCapabilityError as exc:
                unsupported = str(exc)
            fingerprint = json.dumps([call.name, call.arguments], sort_keys=True, allow_nan=False)
        except (ValueError, TypeError, OverflowError) as exc:
            return self._result(call, "failed", error_code="INVALID_ARGUMENTS", message=str(exc))
        if call.call_id in self.cache:
            old, result = self.cache[call.call_id]
            if old != fingerprint:
                return self._result(call, "failed", error_code="CALL_ID_REUSED")
            return {**deepcopy(result), "deduplicated": True}
        if self.termination is not None:
            return self._result(call, "failed", error_code="EPISODE_ENDED")
        try:
            if unsupported is not None:
                result = self._result(call, "failed", error_code="UNSUPPORTED_CAPABILITY",
                                      message=unsupported)
            elif call.name == "track_point":
                result = self._result(call, point=self.tracker.add(**call.arguments))
            elif call.name == "get_geometry":
                result = self._result(call, points=self.tracker.geometry())
            elif call.name == "observe":
                self.refresh()
                result = self._result(call, image_id=self.frame.frame_id)
            elif call.name in {"move_relative", "turn", "stop", "set_posture", "reset_posture"}:
                data = self.backend.execute(call)
                if not isinstance(data, dict) or not isinstance(data.get("status"), str):
                    raise TypeError("Backend execute must return a result dictionary with status")
                result = {**data, "call_id": call.call_id, "tool": call.name}
                self.refresh()
            else:
                stopped = self.backend.execute(ToolCall(call.call_id, "stop", {}))
                self.refresh()
                self.termination = call.name
                result = self._result(call, stop_result=stopped, message="Declaration recorded; evaluator runs separately.")
        except PerceptionError as exc:
            result = self._result(call, "failed", error_code="PERCEPTION_ERROR", message=str(exc))
        self.cache[call.call_id] = (fingerprint, deepcopy(result))
        return result
