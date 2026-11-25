"""Task-independent, public-observation subgoal state and execution progress.

The state stores model-selected goals and routes. It neither chooses targets nor
generates routes, and local goal completion is never an overall task verdict.
In measured mode, trusted executor feedback advances local state. In agent mode,
observations remain diagnostics and only explicit agent updates advance state.
"""
from collections import deque
from copy import deepcopy
import math

from .models import ModelProtocolError


POSITION_TOLERANCE_M = .19
HEADING_TOLERANCE_RAD = .12
CRITERIA = ("reach_pose", "observe_target", "approach_target")
PHYSICAL_ACTIONS = ("scan", "follow_waypoints", "approach_target")
UPDATE_POLICIES = ("measured", "agent")
AGENT_STATUSES = ("active", "recovering", "completed", "abandoned")
TERMINAL_STATUSES = ("completed", "abandoned")


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _text(value, maximum=240):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _valid_pose(value):
    return (isinstance(value, dict) and set(value) in ({"x", "y"}, {"x", "y", "yaw"})
            and all(_finite(number) for number in value.values()))


def validate_goal(arguments):
    """Validate a model proposal without task names, scene labels or hidden state."""
    allowed = {"description", "criterion", "target_id", "target_pose", "desired_distance_m", "reason"}
    if not isinstance(arguments, dict) or set(arguments) - allowed:
        raise ModelProtocolError("Unexpected subgoal fields.")
    if not _text(arguments.get("description")) or arguments.get("criterion") not in CRITERIA:
        raise ModelProtocolError("Provide a description and reach_pose, observe_target or approach_target.")
    if "reason" in arguments and not _text(arguments["reason"]):
        raise ModelProtocolError("A replacement reason must contain 1 to 240 characters.")
    criterion = arguments["criterion"]
    if criterion == "reach_pose":
        if not _valid_pose(arguments.get("target_pose")) or set(arguments) & {"target_id", "desired_distance_m"}:
            raise ModelProtocolError("reach_pose requires finite target_pose x/y and optional yaw, without a target ID.")
    else:
        if not _text(arguments.get("target_id"), 256) or "target_pose" in arguments:
            raise ModelProtocolError("observe_target and approach_target require an explicit target_id, not target_pose.")
        if "desired_distance_m" in arguments:
            distance = arguments["desired_distance_m"]
            if criterion != "approach_target" or not _finite(distance) or distance <= 0:
                raise ModelProtocolError("Only approach_target accepts a finite positive desired_distance_m.")
    return deepcopy(arguments)


def _pose(observation):
    odometry = observation.get("local_odometry", {})
    if not isinstance(odometry, dict):
        return None
    xy, yaw = odometry.get("translation_m"), odometry.get("yaw_rad")
    if (not isinstance(xy, (list, tuple)) or len(xy) != 2
            or not all(_finite(value) for value in (*xy, yaw))):
        return None
    return {"x": float(xy[0]), "y": float(xy[1]), "yaw": float(yaw)}


def _distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _yaw_distance(a, b):
    # Reduce before subtraction so very large finite angles cannot overflow.
    return abs(math.remainder(math.remainder(a["yaw"], math.tau) - math.remainder(b["yaw"], math.tau), math.tau))


def _route(arguments):
    if not isinstance(arguments, dict) or set(arguments) != {"waypoints", "after"} or arguments["after"] != "inspect":
        raise ModelProtocolError("A route requires waypoints and after=inspect; approach is a separate skill.")
    points = arguments["waypoints"]
    if not isinstance(points, list) or not 1 <= len(points) <= 32 or not all(_valid_pose(point) for point in points):
        raise ModelProtocolError("A route requires 1 to 32 finite numeric waypoints.")
    return points


def _rounded(value):
    return round(value, 3) if math.isfinite(value) else None


class SubgoalState:
    """One task's bounded execution state, compatible with generic motion tools."""

    def __init__(self, task_id, update_policy="measured"):
        if not isinstance(task_id, str):
            raise ValueError("task_id must be a string.")
        if update_policy not in UPDATE_POLICIES:
            raise ValueError("update_policy must be measured or agent.")
        self.task_id = task_id
        self.update_policy = update_policy
        self.active = None
        self.recent_goals = deque(maxlen=4)
        self.last_target_sighting = None
        self.last_action = None
        self.action_count = self.goal_count = self.no_progress_actions = 0
        self.recovery_required = False
        self.recovery_reason = None
        self._poses = deque(maxlen=6)

    @property
    def pending_route(self):
        return deepcopy(self.active.get("pending_route")) if self.active else None

    def set_goal(self, arguments, observation):
        goal = validate_goal(arguments)
        if self.active and self.active["status"] not in TERMINAL_STATUSES and not goal.get("reason"):
            raise ModelProtocolError("Replacing an unfinished subgoal requires an explicit reason.")
        if self.active:
            fields = ("id", "description", "criterion", "status", "verification", "target_id", "target_pose", "desired_distance_m")
            previous = {key: deepcopy(self.active[key]) for key in fields if key in self.active}
            if previous["status"] not in TERMINAL_STATUSES:
                previous.update(status="replaced", replacement_reason=goal["reason"])
            self.recent_goals.append(previous)
        if self.update_policy == "agent":
            # Choosing a new goal is an explicit agent decision; old recovery
            # declarations do not silently become the new goal's status.
            self.recovery_required, self.recovery_reason = False, None
        self.goal_count += 1
        self.active = {"id": f"goal-{self.goal_count}", "description": goal["description"],
                       "criterion": goal["criterion"], "status": "recovering" if self.recovery_required else "active",
                       "attempts": 0, "verification": {"state": "pending"}, "pending_route": None}
        if self.update_policy == "agent":
            self.active["verification"] = {"state": "awaiting_agent", "source": "agent_required",
                                           "independent_task_success": None}
        for key in ("target_id", "target_pose", "desired_distance_m"):
            if key in goal:
                self.active[key] = goal[key]
        if self.last_target_sighting and self.last_target_sighting["target_id"] != goal.get("target_id"):
            self.last_target_sighting = None
        # Goal changes and repeated observations never reset action-based stagnation.
        self.observe(observation)
        return {"status": "ok", "goal_id": self.active["id"], "task_state": self.snapshot()}

    def update_goal(self, arguments, observation):
        """Apply one explicit agent judgment, without promoting it to task truth.

        Prefix counts are cumulative for the currently saved primary route.
        Reaching its last point keeps the endpoint pending until ``route_done``
        or a terminal goal judgment is explicitly submitted. All argument
        validation happens before mutating the stored state.
        """
        if self.update_policy != "agent":
            raise ModelProtocolError("Explicit subgoal judgments require agent update policy.")
        required = {"goal_id", "status", "reason"}
        allowed = required | {"completed_waypoints", "route_done"}
        if (not isinstance(arguments, dict) or set(arguments) - allowed
                or not required <= set(arguments)):
            raise ModelProtocolError("Subgoal updates need goal_id, status and reason only, with optional route progress.")
        if (not self.active or not _text(arguments["goal_id"])
                or arguments["goal_id"] != self.active["id"]):
            raise ModelProtocolError("Subgoal update must name the current goal_id.")
        status, reason = arguments["status"], arguments["reason"]
        if status not in AGENT_STATUSES or not _text(reason):
            raise ModelProtocolError("Provide a supported subgoal status and a concise nonempty reason.")
        if "route_done" in arguments and type(arguments["route_done"]) is not bool:
            raise ModelProtocolError("route_done must be a boolean.")
        route = deepcopy(self.active["pending_route"])
        if "completed_waypoints" in arguments:
            count = arguments["completed_waypoints"]
            if (route is None or type(count) is not int
                    or not route["completed_waypoints"] <= count <= route["total_waypoints"]):
                raise ModelProtocolError("Route progress must be a monotonic cumulative integer for the current route.")
            if arguments.get("route_done") and count != route["total_waypoints"]:
                raise ModelProtocolError("route_done conflicts with an incomplete prefix count.")
            delta = count - route["completed_waypoints"]
            del route["waypoints"][:delta]
            route["completed_waypoints"] = count
            route["endpoint_pending"] = not route["waypoints"]
        if arguments.get("route_done") and route is None:
            raise ModelProtocolError("There is no saved route to complete.")
        if arguments.get("route_done") or status in TERMINAL_STATUSES:
            route = None

        # The evidence is a model judgment, even if its interpretation is wrong.
        # Independent acceptance remains solely in the environment evaluator.
        self.active["pending_route"] = route
        self.active["status"] = status
        self.active["verification"] = {"state": status, "source": "agent_judgment",
            "reason": reason, "independent_task_success": None}
        self.recovery_required = status == "recovering"
        self.recovery_reason = reason if self.recovery_required else None
        self.observe(observation)
        return {"status": "ok", "goal_id": self.active["id"], "task_state": self.snapshot()}

    def note_route(self, arguments, *, recovery=False):
        """Save a new primary route; continuations skip this method.

        A recovery route is validated but cannot replace the primary route or its
        already verified prefix. Every route ends in observation, never docking.
        """
        points = _route(arguments)
        if self.active is None or self.active["status"] in TERMINAL_STATUSES:
            raise ModelProtocolError("Set an unfinished subgoal before submitting a route.")
        if recovery:
            return
        self.active["pending_route"] = {"waypoints": deepcopy(points), "after": "inspect",
                                        "completed_waypoints": 0, "total_waypoints": len(points),
                                        "endpoint": deepcopy(points[-1]), "endpoint_pending": False}

    def _grounding(self, observation):
        target_id = self.active.get("target_id") if self.active else None
        image_id = observation.get("image_id")
        if target_id is None or not isinstance(image_id, str) or not image_id:
            return None
        candidates = observation.get("grounded_targets", [])
        if not isinstance(candidates, (list, tuple)):
            return None
        return next((item for item in candidates if isinstance(item, dict)
                     and item.get("target_id") == target_id and item.get("valid") is True
                     and item.get("image_id") == image_id), None)

    def observe(self, observation):
        """Use only current-frame valid grounding and public local odometry."""
        grounding, pose = self._grounding(observation), _pose(observation)
        visible = grounding is not None
        if visible:
            candidate = {"target_id": grounding["target_id"], "valid": True, "image_id": grounding["image_id"]}
            for key, size in (("pixel_uv", 2), ("camera_xyz_m", 3), ("relative_xy_m", 2)):
                values = grounding.get(key)
                if isinstance(values, (list, tuple)) and len(values) == size and all(_finite(value) for value in values):
                    candidate[key] = list(values)
            for key in ("source", "status", "coordinate_frame"):
                if isinstance(grounding.get(key), str):
                    candidate[key] = grounding[key][:120]
            self.last_target_sighting = {"target_id": grounding["target_id"], "image_id": observation["image_id"],
                "sim_time_s": observation.get("sim_time_s") if _finite(observation.get("sim_time_s")) else None,
                "observer_pose": pose, "candidate": candidate, "coordinates_at_sighting_only": True}
        if self.active and (self.update_policy == "agent" or self.active["status"] not in TERMINAL_STATUSES):
            criterion = self.active["criterion"]
            verified, evidence = False, {"state": "pending", "current_target_visible": visible}
            if criterion == "observe_target":
                verified = visible
                evidence.update(source="current_valid_grounding", target_id=self.active["target_id"])
                if visible:
                    evidence["image_id"] = observation["image_id"]
            elif criterion == "reach_pose":
                evidence["source"] = "public_odometry"
                if pose is not None:
                    target = self.active["target_pose"]
                    distance = _distance(pose, target)
                    heading = _yaw_distance(pose, target) if "yaw" in target else None
                    verified = distance <= POSITION_TOLERANCE_M and (heading is None or heading <= HEADING_TOLERANCE_RAD)
                    evidence.update(distance_m=_rounded(distance), heading_error_rad=None if heading is None else _rounded(heading))
            else:
                evidence.update(source="awaiting_local_skill_confirmation", target_id=self.active["target_id"])
            if self.update_policy == "agent":
                # Preserve even a mistaken declaration. Facts help the model
                # reconsider; they do not silently change its plan judgment.
                evidence.update(state="diagnostic_only", criterion_supported_by_observation=verified)
                self.active["observed_evidence"] = evidence
            elif verified:
                evidence.update(state="completed", independent_task_success=None)
                self.active["status"] = "completed"
                self.active["pending_route"] = None
            else:
                self.active["status"] = "recovering" if self.recovery_required else "active"
            if self.update_policy == "measured":
                self.active["verification"] = evidence
        result = self.snapshot()
        result["current_target_visible"] = visible
        return result

    def _record_prefix(self, arguments, after, feedback):
        route = self.active.get("pending_route")
        if route is None:
            return
        try:
            points = _route(arguments)
        except ModelProtocolError:
            return
        expected = route["waypoints"]
        if route["endpoint_pending"]:
            if points != [route["endpoint"]]:
                return
        elif len(points) > len(expected) or points != expected[:len(points)]:
            return
        count = feedback.get("completed_waypoints", 0)
        if type(count) is not int or not 0 <= count <= len(points):
            return
        if not route["endpoint_pending"]:
            route["completed_waypoints"] += count
            del route["waypoints"][:count]
        if route["waypoints"]:
            return
        # Counted endpoint arrival does not itself prove final yaw/inspect finished.
        endpoint = route["endpoint"]
        at_endpoint = (after is not None and _distance(after, endpoint) <= POSITION_TOLERANCE_M
                       and ("yaw" not in endpoint or _yaw_distance(after, endpoint) <= HEADING_TOLERANCE_RAD))
        completed = feedback.get("status") == "ok" or (
            feedback.get("status") == "skill_completed" and feedback.get("skill") == "follow_waypoints")
        if completed and at_endpoint:
            self.active["pending_route"] = None
        else:
            route["endpoint_pending"] = True

    def record_action(self, name, arguments, before_observation, after_observation, feedback,
                      intention="", *, recovery=False):
        """Verify a physical slice from executor feedback, never from declarations."""
        if name not in PHYSICAL_ACTIONS:
            return self.observe(after_observation)
        before, after = _pose(before_observation), _pose(after_observation)
        moved = _distance(before, after) if before and after else 0.
        rotated = _yaw_distance(before, after) if before and after else 0.
        self.action_count += 1
        if self.active:
            self.active["attempts"] += 1
            if name == "follow_waypoints" and not recovery and self.update_policy == "measured":
                self._record_prefix(arguments, after, feedback)
        confirmed = bool(self.active and self.active["criterion"] == "approach_target"
            and self.active["status"] != "completed" and name == "approach_target"
            and arguments.get("target_id") == self.active["target_id"]
            and feedback.get("status") == "skill_completed" and feedback.get("skill") == "approach_target"
            and feedback.get("target_id") == self.active["target_id"])
        if confirmed and "desired_distance_m" in self.active:
            distance = arguments.get("desired_distance_m")
            confirmed = bool(_finite(distance) and math.isclose(
                distance, self.active["desired_distance_m"], rel_tol=1e-9, abs_tol=1e-9))
        new_sighting = self._grounding(after_observation) is not None and self._grounding(before_observation) is None
        toward_goal = False
        if self.active and self.active["criterion"] == "reach_pose" and before and after:
            target = self.active["target_pose"]
            toward_goal = _distance(before, target) - _distance(after, target) >= .05
            if "yaw" in target and _distance(after, target) <= POSITION_TOLERANCE_M:
                toward_goal |= _yaw_distance(before, target) - _yaw_distance(after, target) >= .12
        made_progress = bool((math.isfinite(moved) and moved >= .08) or new_sighting or toward_goal or confirmed)
        self.no_progress_actions = 0 if made_progress else self.no_progress_actions + 1
        if after:
            self._poses.append(after)
        repeated_view = bool(after and sum(_distance(pose, after) < .20 and _yaw_distance(pose, after) < .20
                                          for pose in self._poses) >= 3)
        code = feedback.get("error_code") if isinstance(feedback.get("error_code"), str) else None
        status = feedback.get("status") if isinstance(feedback.get("status"), str) else None
        ordinary_yield = code in (None, "execution_slice_complete", "target_visible_in_current_view")
        blocked = status in {"failed", "blocked", "error"} or (status == "replan" and not ordinary_yield)
        if self.update_policy == "measured" and (blocked or self.no_progress_actions >= 3 or repeated_view):
            self.recovery_required = True
            self.recovery_reason = (code or status)[:200] if blocked else ("repeated_view" if repeated_view else "no_measured_progress")
        elif self.update_policy == "measured" and made_progress:
            self.recovery_required, self.recovery_reason = False, None
        self.last_action = {"tool": name, "intention": intention[:240] if isinstance(intention, str) else "",
                           "role": "recovery" if recovery else "primary", "status": status,
                           "error_code": code[:200] if code is not None else None,
                           "measured_displacement_m": _rounded(moved), "measured_yaw_change_rad": _rounded(rotated),
                           "made_progress": made_progress, "repeated_view": repeated_view}
        if self.update_policy == "agent":
            self.last_action.update(diagnostic_only=True,
                executor_completed_waypoints=feedback.get("completed_waypoints")
                    if type(feedback.get("completed_waypoints")) is int and feedback["completed_waypoints"] >= 0 else None,
                local_skill_confirmed=confirmed, blocked_feedback=blocked)
        self.observe(after_observation)
        if confirmed and self.update_policy == "measured":
            self.active["status"] = "completed"
            self.active["pending_route"] = None
            self.active["verification"] = {"state": "completed", "source": "local_skill_not_task_success",
                "skill": "approach_target", "target_id": self.active["target_id"], "independent_task_success": None}
            self.recovery_required, self.recovery_reason = False, None
        return self.snapshot()

    def snapshot(self):
        return deepcopy({"task_id": self.task_id, "phase": "subgoal_execution", "update_policy": self.update_policy,
            "status": "recovery_required" if self.recovery_required else (self.active["status"] if self.active else "needs_subgoal"),
            "active": self.active, "recent_goals": list(self.recent_goals),
            "last_target_sighting": self.last_target_sighting, "last_action": self.last_action,
            "action_count": self.action_count, "goal_count": self.goal_count,
            "recovery_required": self.recovery_required, "recovery_reason": self.recovery_reason,
            "no_progress_actions": self.no_progress_actions})
