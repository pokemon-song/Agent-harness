"""Task-neutral execution over public odometry and model-selected RGB-D points.

The executor never identifies objects, chooses destinations, or scores a task.
A grounding associates a caller's label with a tracked *surface point*, not with
an independently verified semantic identity or the centre of an object.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time
import uuid

from .harness import Harness, _public_calibration
from .short_step_clearance import check_clearance
from .types import ToolCall


POSITION_TOLERANCE_M = .19
HEADING_TOLERANCE_RAD = .12
MAX_TRANSLATION_M = .53
MAX_TURN_RAD = .59
SURFACE_DISTANCE_TOLERANCE_M = .075
SURFACE_BEARING_TOLERANCE_RAD = .09


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value, size):
    return (isinstance(value, (list, tuple)) and len(value) == size
            and all(_number(item) for item in value))


def _scalars(source, *, numbers=(), strings=(), booleans=()):
    if not isinstance(source, dict):
        return {}
    return {key: value for key, value in source.items()
            if ((key in numbers and _number(value))
                or (key in strings and isinstance(value, str))
                or (key in booleans and type(value) is bool))}


def _public_point(point):
    """Never copy a tracker's internal world point or arbitrary metadata."""
    result = _scalars(point, numbers=("timestamp_s", "range_m", "bearing_rad", "confidence"),
                      strings=("point_id", "image_id", "reason"), booleans=("valid",))
    for name in ("pixel_uv", "relative_xy_m"):
        result[name] = list(point[name]) if _vector(point.get(name), 2) else None
    if not result.get("valid"):
        result.update(valid=False, pixel_uv=None, relative_xy_m=None, range_m=None, bearing_rad=None)
    return result


def _public_reply(reply):
    result = _scalars(reply, strings=("call_id", "tool", "status", "error_code"),
                      booleans=("collision", "deduplicated"))
    measured = _scalars(reply.get("measured"), numbers=(
        "yaw_delta_rad", "position_error_m", "heading_error_rad", "elapsed_sim_s"))
    if isinstance(reply.get("measured"), dict) and _vector(reply["measured"].get("translation_start_frame_m"), 2):
        measured["translation_start_frame_m"] = list(reply["measured"]["translation_start_frame_m"])
    if measured:
        result["measured"] = measured
    for name in ("stop", "stop_result"):
        if isinstance(reply.get(name), dict):
            result[name] = _scalars(reply[name], strings=("status", "error_code"),
                                    numbers=("speed_mps", "yaw_rate_radps"), booleans=("collision",))
    if isinstance(reply.get("point"), dict):
        result["point"] = _public_point(reply["point"])
    if result.get("error_code") == "PERCEPTION_ERROR":
        message = reply.get("message", "")
        for prefix, code in (("STALE_IMAGE:", "STALE_IMAGE"),
                             ("No usable depth", "INVALID_DEPTH"),
                             ("Pixel outside", "PIXEL_OUT_OF_BOUNDS"),
                             ("Maximum tracked points", "TRACK_CAPACITY_REACHED")):
            if isinstance(message, str) and message.startswith(prefix):
                result["error_code"] = code
                break
    return result


class _Halt(Exception):
    def __init__(self, status, code, **details):
        self.status, self.code, self.details = status, code, details


class GenericExecutor:
    """One run's cumulative native budget, with short independently bounded slices.

    ``on_action`` receives public before/after observations and the actual native
    call/result. A fresh executor uses fresh call IDs even on the same Harness.
    No state is automatically restored across process restarts.
    """

    def __init__(self, harness: Harness, target_ids, *, max_native_actions=230,
                 actions_per_slice=6, on_action=None, max_wall_seconds=None,
                 max_sim_seconds=None):
        if (not isinstance(target_ids, (list, tuple, set, frozenset))
                or any(not isinstance(item, str) or not item.strip() or len(item) > 256 for item in target_ids)
                or len(set(target_ids)) != len(target_ids)):
            raise ValueError("target_ids must be distinct nonempty public identifiers")
        if type(max_native_actions) is not int or max_native_actions < 0:
            raise ValueError("max_native_actions must be a nonnegative integer")
        if actions_per_slice is not None and (type(actions_per_slice) is not int or actions_per_slice < 1):
            raise ValueError("actions_per_slice must be positive or None")
        if on_action is not None and not callable(on_action):
            raise ValueError("on_action must be callable")
        for value in (max_wall_seconds, max_sim_seconds):
            if value is not None and (not _number(value) or value <= 0):
                raise ValueError("Optional time budgets must be positive finite numbers")
        self.harness = harness
        self.target_ids = frozenset(target_ids)
        self.max_native_actions, self.actions_per_slice = max_native_actions, actions_per_slice
        self.on_action = on_action
        self.actions = []
        self.native_count = 0
        self._bindings = {}
        self._prefix = "generic-" + uuid.uuid4().hex
        self._slice_start = 0
        self._completed = 0
        backend = harness.backend
        for name, candidate in (
                ("deadline_monotonic", None if max_wall_seconds is None else time.monotonic() + max_wall_seconds),
                ("sim_deadline_s", None if max_sim_seconds is None else self.observation()["sim_time_s"] + max_sim_seconds)):
            if candidate is not None:
                existing = getattr(backend, name, None)
                setattr(backend, name, candidate if existing is None else min(existing, candidate))

    def observation(self):
        raw = self.harness.observation()
        result = _scalars(raw, strings=("task", "image_id", "coordinate_convention", "simulation_mode"),
                          numbers=("sim_time_s",))
        result["image_size"] = _scalars(raw.get("image_size"), numbers=("width", "height"))
        odometry = raw.get("local_odometry", {})
        result["local_odometry"] = _scalars(odometry, numbers=("yaw_rad",), strings=("reference", "source"))
        if _vector(odometry.get("translation_m"), 2):
            result["local_odometry"]["translation_m"] = list(odometry["translation_m"])
        result["robot_state"] = _scalars(raw.get("robot_state"), numbers=(
            "body_height_m", "body_pitch_rad", "body_roll_rad", "speed_mps", "yaw_rate_radps"),
            strings=("motion_state",), booleans=("fallen", "posture_control_supported"))
        result["capabilities"] = _scalars(raw.get("capabilities"), booleans=("planar_motion", "height_pitch_control"))
        calibration = _public_calibration(raw.get("sensor_calibration"))
        if calibration:
            result["sensor_calibration"] = calibration
        requirements = _scalars(raw.get("completion_requirements"),
            strings=("goal_kind", "reference_body"),
            numbers=("required_clearance_m", "standoff_m", "distance_tolerance_m", "heading_tolerance_rad",
                     "max_stopped_speed_mps", "max_stopped_yaw_rate_radps", "collision_penalty_points"),
            booleans=("stop_required", "collision_free_required"))
        if requirements:
            result["completion_requirements"] = requirements
        points = [_public_point(point) for point in raw.get("tracked_points", []) if isinstance(point, dict)]
        result["tracked_points"] = points
        by_id = {point.get("point_id"): point for point in points}
        targets = []
        for target_id, binding in self._bindings.items():
            point = deepcopy(by_id.get(binding["point_id"], {}))
            # A geometry record must belong to this current image, not a cached sighting.
            if not point.get("valid") or point.get("image_id") != result.get("image_id"):
                point.update(valid=False, image_id=result.get("image_id"), pixel_uv=None,
                             relative_xy_m=None, range_m=None, bearing_rad=None,
                             confidence=0., reason=point.get("reason") or "tracking_unavailable")
            point.update(target_id=target_id, point_id=binding["point_id"],
                         identity_basis="model_proposed", geometry_only=True,
                         proposed_image_id=binding["image_id"])
            targets.append(point)
        result["grounded_targets"] = targets
        result["native_actions_remaining"] = max(0, self.max_native_actions - self.native_count)
        return result

    def _pose(self):
        odometry = self.observation()["local_odometry"]
        if not _vector(odometry.get("translation_m"), 2) or not _number(odometry.get("yaw_rad")):
            raise _Halt("replan", "PUBLIC_ODOMETRY_UNAVAILABLE")
        return (*odometry["translation_m"], odometry["yaw_rad"])

    def _check_state(self, *, allow_ended=False):
        observation = self.observation()
        backend = self.harness.backend
        if observation["robot_state"].get("fallen"):
            raise _Halt("fall", "ROBOT_FELL")
        if getattr(backend, "collision_terminated", False):
            raise _Halt("collision", "ENVIRONMENT_TERMINATED_ON_COLLISION")
        wall = getattr(backend, "deadline_monotonic", None)
        sim = getattr(backend, "sim_deadline_s", None)
        if wall is not None and time.monotonic() >= wall:
            raise _Halt("budget_exhausted", "wall_time_limit")
        if sim is not None and observation["sim_time_s"] >= sim:
            raise _Halt("budget_exhausted", "simulation_time_limit")
        if not allow_ended and getattr(self.harness, "termination", None) is not None:
            raise _Halt("ended", "EPISODE_ENDED")

    def _act(self, name, *, _bind_target=None, **arguments):
        self._check_state()
        if self.native_count >= self.max_native_actions:
            raise _Halt("budget_exhausted", "native_action_limit")
        if self.actions_per_slice is not None and self.native_count - self._slice_start >= self.actions_per_slice:
            raise _Halt("replan", "execution_slice_complete")
        before = self.observation()
        if name == "move_relative":
            block = check_clearance(self.harness.frame, before, **arguments)
            if block is not None:
                detail = _scalars(block, strings=("error_code", "reason"), numbers=("clearance_along_command_m",))
                if _vector(block.get("obstacle_forward_left_m"), 2):
                    detail["obstacle_forward_left_m"] = list(block["obstacle_forward_left_m"])
                raise _Halt("replan", "current_depth_blocks_requested_translation", depth_feedback=detail)
        call = ToolCall(f"{self._prefix}-{self.native_count:06d}", name, deepcopy(arguments))
        self.native_count += 1  # Count every dispatched attempt, including failed native tools.
        reply = _public_reply(self.harness.dispatch(call))
        if _bind_target is not None and reply.get("status") == "succeeded":
            point = reply.get("point", {})
            if point.get("valid") and point.get("image_id") == self.harness.frame.frame_id:
                self._bindings[_bind_target] = {"point_id": point["point_id"], "image_id": arguments["image_id"]}
        after = self.observation()
        entry = {"call_id": call.call_id, "tool": name, "args": deepcopy(arguments),
                 "action": {"tool": name, **deepcopy(arguments)}, "before": before, "after": after, "reply": reply}
        self.actions.append(deepcopy(entry))
        if self.on_action is not None:
            self.on_action(deepcopy(entry))
        # A declaration is only an execution result; task evaluation stays outside.
        self._check_state(allow_ended=name in {"finish", "abort"})
        if name in {"finish", "abort"} and reply.get("stop_result", {}).get("status") != "succeeded":
            raise _Halt("control_error", "STOP_NOT_CONFIRMED", reply=reply,
                        episode_ended=getattr(self.harness, "termination", None) is not None)
        if reply.get("status") == "succeeded":
            return reply
        if (name == "move_relative" and reply.get("error_code") in {"MOTION_TIMEOUT", "DID_NOT_SETTLE_AT_TARGET"}
                and reply.get("stop", {}).get("status") == "succeeded"):
            return reply  # Remeasure; never mark a waypoint reached from this reply alone.
        raise _Halt("replan", reply.get("error_code") or "CONTROL_FAILURE", reply=reply)

    def _operate(self, operation):
        before = self.observation()
        self._slice_start, self._completed = self.native_count, 0
        try:
            self._check_state()
            result = operation()
        except _Halt as stop:
            result = {"status": stop.status, "error_code": stop.code, **stop.details}
        after = self.observation()
        old, new = before["local_odometry"], after["local_odometry"]
        moved = (math.dist(old["translation_m"], new["translation_m"])
                 if _vector(old.get("translation_m"), 2) and _vector(new.get("translation_m"), 2) else None)
        rotated = (math.remainder(new["yaw_rad"] - old["yaw_rad"], math.tau)
                   if _number(old.get("yaw_rad")) and _number(new.get("yaw_rad")) else None)
        return {"error_code": None, **result, "native_actions": self.native_count - self._slice_start,
                "total_native_actions": self.native_count, "completed_waypoints": self._completed,
                "current_odometry": new, "measured_displacement_m": moved, "measured_yaw_change_rad": rotated}

    def ground_target(self, target_id, image_id, u, v):
        def ground():
            if not isinstance(target_id, str) or target_id not in self.target_ids:
                raise _Halt("replan", "UNKNOWN_TARGET_ID")
            if not isinstance(image_id, str) or not image_id or not _number(u) or not _number(v):
                raise _Halt("replan", "INVALID_GROUNDING_ARGUMENTS")
            # An unsuccessful new proposal cannot silently retain the old association.
            old = self._bindings.pop(target_id, None)
            if old is not None:
                # Release only this executor's superseded association. Other targets,
                # other users of the tracker, and historical evidence are untouched.
                self.harness.tracker.points.pop(old["point_id"], None)
            reply = self._act("track_point", _bind_target=target_id, image_id=image_id, u=u, v=v)
            point = reply.get("point", {})
            if not point.get("valid") or point.get("image_id") != self.observation().get("image_id"):
                raise _Halt("replan", "TARGET_GROUNDING_REQUIRED")
            self._bindings[target_id] = {"point_id": point["point_id"], "image_id": image_id}
            grounding = next(row for row in self.observation()["grounded_targets"] if row["target_id"] == target_id)
            return {"status": "ok", "target_id": target_id, "identity_basis": "model_proposed",
                    "geometry_only": True, "grounding": grounding}
        return self._operate(ground)

    def _face(self, heading, tolerance):
        while True:
            delta = math.remainder(heading - self._pose()[2], math.tau)
            if abs(delta) <= tolerance:
                return
            self._act("turn", yaw_rad=max(-MAX_TURN_RAD, min(MAX_TURN_RAD, delta)))

    def _go_to(self, waypoint):
        while True:
            x, y, yaw = self._pose()
            dx, dy = waypoint["x"] - x, waypoint["y"] - y
            distance = math.hypot(dx, dy)
            if distance <= POSITION_TOLERANCE_M:
                if "yaw" in waypoint:
                    self._face(waypoint["yaw"], HEADING_TOLERANCE_RAD)
                # Turning/settling can translate the body. Verify the same final
                # snapshot, rather than joining position and yaw from two times.
                x, y, yaw = self._pose()
                if (math.hypot(waypoint["x"] - x, waypoint["y"] - y) <= POSITION_TOLERANCE_M
                        and ("yaw" not in waypoint or abs(math.remainder(waypoint["yaw"] - yaw, math.tau)) <= HEADING_TOLERANCE_RAD)):
                    return
                continue
            bearing = math.atan2(dy, dx)
            if abs(math.remainder(bearing - yaw, math.tau)) > .62:
                self._face(bearing, .19)
                continue
            scale = min(MAX_TRANSLATION_M, distance) / distance
            self._act("move_relative", forward_m=(dx * math.cos(yaw) + dy * math.sin(yaw)) * scale,
                      lateral_m=(-dx * math.sin(yaw) + dy * math.cos(yaw)) * scale)

    def navigate(self, name, args, resume_endpoint=False):
        def navigate_route():
            if not isinstance(args, dict) or type(resume_endpoint) is not bool:
                raise _Halt("replan", "INVALID_NAVIGATION_ARGUMENTS")
            if name == "scan":
                if set(args) != {"yaw_rad"} or not _number(args["yaw_rad"]) or abs(args["yaw_rad"]) > MAX_TURN_RAD:
                    raise _Halt("replan", "INVALID_NAVIGATION_ARGUMENTS")
                self._act("turn", yaw_rad=args["yaw_rad"])
            elif name == "follow_waypoints":
                points = args.get("waypoints")
                if (set(args) != {"waypoints", "after"} or args["after"] != "inspect"
                        or not isinstance(points, list) or not 1 <= len(points) <= 32
                        or any(not isinstance(p, dict) or set(p) not in ({"x", "y"}, {"x", "y", "yaw"})
                               or not all(_number(v) for v in p.values()) for p in points)
                        or (resume_endpoint and len(points) != 1)):
                    raise _Halt("replan", "INVALID_NAVIGATION_ARGUMENTS")
                for waypoint in points:
                    self._go_to(waypoint)
                    if not resume_endpoint:
                        self._completed += 1
            else:
                raise _Halt("replan", "UNKNOWN_NAVIGATION_TOOL")
            return {"status": "ok"}
        return self._operate(navigate_route)

    def _target_geometry(self, target_id):
        if not isinstance(target_id, str) or target_id not in self.target_ids:
            raise _Halt("replan", "UNKNOWN_TARGET_ID")
        point = next((row for row in self.observation()["grounded_targets"] if row["target_id"] == target_id), None)
        if not point or not point.get("valid") or not _vector(point.get("relative_xy_m"), 2):
            raise _Halt("replan", "TARGET_GROUNDING_REQUIRED", target_id=target_id)
        forward, left = point["relative_xy_m"]
        distance = math.hypot(forward, left)
        if not _number(distance) or distance <= 1e-6:
            raise _Halt("replan", "TARGET_GROUNDING_REQUIRED", target_id=target_id)
        return forward, left, distance, math.atan2(left, forward)

    def approach(self, target_id, desired_distance_m):
        def approach_point():
            if not _number(desired_distance_m) or not 0 < desired_distance_m <= 12.:
                raise _Halt("replan", "INVALID_DESIRED_DISTANCE")
            while True:
                forward, left, distance, bearing = self._target_geometry(target_id)
                error = distance - desired_distance_m
                if abs(bearing) > SURFACE_BEARING_TOLERANCE_RAD:
                    self._act("turn", yaw_rad=max(-MAX_TURN_RAD, min(MAX_TURN_RAD, bearing)))
                elif abs(error) > SURFACE_DISTANCE_TOLERANCE_M:
                    step = max(-MAX_TRANSLATION_M, min(MAX_TRANSLATION_M, error))
                    self._act("move_relative", forward_m=step * forward / distance, lateral_m=step * left / distance)
                else:
                    self._act("stop")
                    _, _, distance, bearing = self._target_geometry(target_id)
                    state = self.observation()["robot_state"]
                    stopped = (_number(state.get("speed_mps")) and state["speed_mps"] <= .12
                               and _number(state.get("yaw_rate_radps")) and abs(state["yaw_rate_radps"]) <= .25
                               and not state.get("fallen", False))
                    if not stopped:
                        raise _Halt("replan", "STOP_NOT_CONFIRMED")
                    if (abs(distance - desired_distance_m) <= SURFACE_DISTANCE_TOLERANCE_M
                            and abs(bearing) <= SURFACE_BEARING_TOLERANCE_RAD):
                        return {"status": "skill_completed", "skill": "approach_target", "target_id": target_id,
                                "identity_basis": "model_proposed", "geometry_only": True,
                                "reference": "tracked_surface_point", "desired_distance_m": desired_distance_m,
                                "measured_range_m": distance, "measured_bearing_rad": bearing, "stopped": True}
        return self._operate(approach_point)

    def act(self, name, **arguments):
        """One bounded native tool. Task scoring and semantic success are external."""
        def perform():
            if name not in {"observe", "get_geometry", "move_relative", "turn", "stop", "finish", "abort"}:
                raise _Halt("replan", "UNSUPPORTED_DIRECT_TOOL")
            try:
                Harness._validate(ToolCall("validation", name, arguments))
            except (ValueError, TypeError, OverflowError):
                raise _Halt("replan", "INVALID_ARGUMENTS") from None
            if ((name == "move_relative" and math.hypot(arguments["forward_m"], arguments["lateral_m"]) > MAX_TRANSLATION_M)
                    or (name == "turn" and abs(arguments["yaw_rad"]) > MAX_TURN_RAD)):
                raise _Halt("replan", "INVALID_ARGUMENTS")
            reply = self._act(name, **arguments)
            return {"status": name if name in {"finish", "abort"} else "ok", "reply": reply}
        return self._operate(perform)

    def finish(self, name="finish", reason=None):
        if name not in {"stop", "finish", "abort"}:
            return self._operate(lambda: {"status": "replan", "error_code": "INVALID_FINISH_TOOL"})
        arguments = {"reason": reason} if name == "abort" else {}
        return self.act(name, **arguments)
