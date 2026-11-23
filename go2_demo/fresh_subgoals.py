"""Public, stage-local task state for a planner/executor/checker loop.

The checker verifies model-selected goals; it never generates a destination or
search direction.  A new stage starts its own executable goal. A retained
previous-stage snapshot can remain available as historical context.
"""
from collections import deque
from copy import deepcopy
import math

from .models import ModelProtocolError


POSITION_TOLERANCE_M = .19
HEADING_TOLERANCE_RAD = .12
CRITERIA = ("reach_pose", "observe_target", "dock_target")


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_goal(arguments):
    """Validate a model proposal without consulting environment internals."""
    if not isinstance(arguments, dict) or set(arguments) - {"description", "criterion", "target_pose", "reason"}:
        raise ModelProtocolError("Subgoal requires description, criterion, optional target_pose and reason.")
    text = arguments.get("description")
    if not isinstance(text, str) or not text.strip() or len(text) > 240:
        raise ModelProtocolError("Subgoal description must contain 1 to 240 characters.")
    if arguments.get("criterion") not in CRITERIA:
        raise ModelProtocolError("Choose reach_pose, observe_target, or dock_target.")
    if "reason" in arguments:
        reason = arguments["reason"]
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
            raise ModelProtocolError("Replacement reason must contain 1 to 240 characters.")
    if arguments["criterion"] == "reach_pose":
        pose = arguments.get("target_pose")
        if (not isinstance(pose, dict) or set(pose) not in ({"x", "y"}, {"x", "y", "yaw"})
                or not all(_finite(value) for value in pose.values())):
            raise ModelProtocolError("reach_pose requires finite numeric target_pose x, y and optional yaw.")
    elif "target_pose" in arguments:
        raise ModelProtocolError("Only reach_pose accepts target_pose; target labels follow the current stage.")
    return deepcopy(arguments)


def _pose(observation):
    odometry = observation.get("local_odometry", {})
    xy, yaw = odometry.get("translation_m"), odometry.get("yaw_rad")
    if (not isinstance(xy, (list, tuple)) or len(xy) < 2
            or not all(_finite(v) for v in (*xy[:2], yaw))):
        return None
    return {"x": float(xy[0]), "y": float(xy[1]), "yaw": float(yaw)}


def _distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _yaw_distance(a, b):
    return abs(math.remainder(a["yaw"] - b["yaw"], math.tau))


class StageSubgoals:
    """Bounded working state shared by empty, ledger, and query conditions."""

    def __init__(self, stage, *, prior_stage_state=None):
        if stage not in ("A", "B"):
            raise ValueError("Expected task stage A or B.")
        if prior_stage_state is not None and (
                stage != "B" or not isinstance(prior_stage_state, dict)
                or prior_stage_state.get("stage") != "A"
                or prior_stage_state.get("previous_stage_state") is not None):
            raise ValueError("Only a single preceding A-stage snapshot can be retained by B.")
        self.stage = stage
        # Historical state stays visible without making an A route executable as
        # a B route or treating an old A sighting as current B evidence.
        self.previous_stage_state = deepcopy(prior_stage_state)
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
        if self.active and self.active["status"] != "completed" and not goal.get("reason"):
            raise ModelProtocolError("Replacing an unfinished subgoal requires an explicit reason.")
        if self.active:
            previous = {key: self.active[key] for key in ("id", "description", "criterion", "status", "verification")}
            if previous["status"] != "completed":
                previous.update(status="replaced", replacement_reason=goal["reason"])
            self.recent_goals.append(deepcopy(previous))
        self.goal_count += 1
        self.active = {"id": f"{self.stage}-{self.goal_count}", "description": goal["description"],
                       "criterion": goal["criterion"], "status": "active", "attempts": 0,
                       "target_label": self.stage, "verification": {"state": "pending"},
                       "pending_route": None}
        if "target_pose" in goal:
            self.active["target_pose"] = goal["target_pose"]
        self.observe(observation)
        return {"status": "ok", "goal_id": self.active["id"], "task_state": self.snapshot()}

    def note_route(self, arguments, *, recovery=False):
        """Store exactly the model's primary route before an execution slice.

        Continuations must skip this method. Recovery detours preserve the
        uncompleted primary route and its verified prefix.
        """
        if recovery:
            return
        if self.active is None or self.active["status"] == "completed":
            raise ModelProtocolError("Set an unfinished subgoal before submitting its route.")
        points = arguments.get("waypoints")
        if not isinstance(points, list) or not 1 <= len(points) <= 32 or arguments.get("after") not in ("inspect", "dock"):
            raise ModelProtocolError("A primary route needs 1 to 32 waypoints and after=inspect or dock.")
        for point in points:
            if (not isinstance(point, dict) or set(point) not in ({"x", "y"}, {"x", "y", "yaw"})
                    or not all(_finite(value) for value in point.values())):
                raise ModelProtocolError("Route waypoints require finite numeric x, y and optional yaw.")
        self.active["pending_route"] = {"waypoints": deepcopy(points), "after": arguments["after"],
                                        "completed_waypoints": 0, "total_waypoints": len(points),
                                        "endpoint": deepcopy(points[-1]), "endpoint_pending": False}

    def _visible(self, observation):
        return any(isinstance(candidate, dict) and candidate.get("task_label") == self.stage
                   for candidate in observation.get("target_candidates", []))

    def observe(self, observation):
        """Refresh current evidence. Re-reading a frame never counts as an action."""
        visible, pose = self._visible(observation), _pose(observation)
        if visible:
            source = next(candidate for candidate in observation["target_candidates"]
                          if isinstance(candidate, dict) and candidate.get("task_label") == self.stage)
            candidate = {}
            for key, size in (("pixel_uv", 2), ("camera_xyz_m", 3)):
                values = source.get(key)
                if isinstance(values, (list, tuple)) and len(values) == size and all(_finite(v) for v in values):
                    candidate[key] = list(values)
            for key in ("source", "status", "coordinate_frame"):
                if isinstance(source.get(key), str):
                    candidate[key] = source[key][:120]
            self.last_target_sighting = {"target_label": self.stage, "image_id": observation.get("image_id"),
                                         "sim_time_s": observation.get("sim_time_s"), "observer_pose": pose,
                                         "candidate": candidate, "coordinates_at_sighting_only": True}
        if self.active and self.active["status"] != "completed":
            criterion = self.active["criterion"]
            verified, evidence = False, {"state": "pending", "current_target_visible": visible}
            if criterion == "observe_target":
                verified = visible
                evidence["source"] = "current_rgbd_target_candidate"
            elif criterion == "reach_pose" and pose:
                target = self.active["target_pose"]
                distance = _distance(pose, target)
                heading = _yaw_distance(pose, target) if "yaw" in target else None
                verified = distance <= POSITION_TOLERANCE_M and (heading is None or heading <= HEADING_TOLERANCE_RAD)
                evidence.update(source="public_odometry", distance_m=round(distance, 3),
                                heading_error_rad=None if heading is None else round(heading, 3))
            elif criterion == "dock_target":
                evidence["source"] = "awaiting_controller_stop_confirmation"
            if verified:
                evidence["state"] = "completed"
                self.active["status"] = "completed"
                self.recovery_required, self.recovery_reason = False, None
            else:
                self.active["status"] = "recovering" if self.recovery_required else "active"
            self.active["verification"] = evidence
        result = self.snapshot()
        result["current_target_visible"] = visible
        return result

    def record_action(self, name, arguments, before_observation, after_observation, feedback,
                      intention="", *, recovery=False):
        """Check observed execution and retain the same goal after a failed attempt.

        ``completed_waypoints`` is the executor's measured prefix for this call.
        A sighting interruption, requested distance, or model claim cannot advance
        that prefix. ``finish`` is only controller confirmation, not B task scoring.
        """
        if name not in ("scan", "follow_waypoints"):
            return self.observe(after_observation)
        before, after = _pose(before_observation), _pose(after_observation)
        moved = _distance(before, after) if before and after else 0.
        rotated = _yaw_distance(before, after) if before and after else 0.
        self.action_count += 1
        if self.active:
            self.active["attempts"] += 1
            route = self.active.get("pending_route")
            if name == "follow_waypoints" and route and not recovery:
                count = feedback.get("completed_waypoints", 0)
                if (not route["endpoint_pending"] and type(count) is int
                        and 0 <= count <= min(len(arguments.get("waypoints", [])), len(route["waypoints"]))):
                    # The executor must have been given the actual pending prefix.
                    if arguments.get("waypoints", [])[:count] == route["waypoints"][:count]:
                        route["completed_waypoints"] += count
                        del route["waypoints"][:count]
                if not route["waypoints"]:
                    completed = feedback.get("status") in ("finish", "handoff") or (
                        route["after"] == "inspect" and feedback.get("status") == "ok")
                    route["endpoint_pending"] = not completed and (route["after"] == "dock" or "yaw" in route["endpoint"])
                    if completed or not route["endpoint_pending"]:
                        self.active["pending_route"] = None
        new_sighting = self._visible(after_observation) and not self._visible(before_observation)
        toward_goal = False
        if self.active and self.active["criterion"] == "reach_pose" and before and after:
            target = self.active["target_pose"]
            toward_goal = _distance(before, target) - _distance(after, target) >= .05
            if "yaw" in target and _distance(after, target) <= POSITION_TOLERANCE_M:
                toward_goal |= _yaw_distance(before, target) - _yaw_distance(after, target) >= .12
        made_progress = moved >= .08 or new_sighting or toward_goal
        self.no_progress_actions = 0 if made_progress else self.no_progress_actions + 1
        if after:
            self._poses.append(after)
        repeated_view = bool(after and sum(_distance(pose, after) < .20 and _yaw_distance(pose, after) < .20
                                          for pose in self._poses) >= 3)
        code = feedback.get("error_code")
        ordinary_yield = code in (None, "execution_slice_complete", "target_visible_in_current_view")
        blocked = feedback.get("status") == "replan" and not ordinary_yield
        if blocked or self.no_progress_actions >= 3 or repeated_view:
            self.recovery_required = True
            self.recovery_reason = str(code)[:200] if blocked else ("repeated_view" if repeated_view else "no_measured_progress")
        elif made_progress:
            self.recovery_required, self.recovery_reason = False, None
        self.last_action = {"tool": name, "intention": str(intention)[:240], "role": "recovery" if recovery else "primary",
                            "status": feedback.get("status"), "error_code": code,
                            "measured_displacement_m": round(moved, 3), "measured_yaw_change_rad": round(rotated, 3),
                            "made_progress": made_progress, "repeated_view": repeated_view}
        self.observe(after_observation)
        if self.active and self.active["criterion"] == "dock_target":
            confirmed_a = self.stage == "A" and feedback.get("status") == "handoff" and code == "A_stop_confirmed"
            reported_b = self.stage == "B" and feedback.get("status") == "finish"
            if confirmed_a or reported_b:
                self.active["status"] = "completed"
                self.active["verification"] = {
                    "state": "completed", "source": "independent_A_handoff" if confirmed_a else "controller_finish",
                    "independent_task_success": None,
                }
                self.recovery_required, self.recovery_reason = False, None
        return self.snapshot()

    def snapshot(self):
        state = {"stage": self.stage, "phase": "subgoal_execution",
                 "status": "recovery_required" if self.recovery_required else (self.active["status"] if self.active else "needs_subgoal"),
                 "active": self.active, "recent_goals": list(self.recent_goals),
                 "last_target_sighting": self.last_target_sighting, "last_action": self.last_action,
                 "action_count": self.action_count, "recovery_required": self.recovery_required,
                 "recovery_reason": self.recovery_reason, "no_progress_actions": self.no_progress_actions}
        if self.previous_stage_state is not None:
            state["previous_stage_state"] = self.previous_stage_state
        return deepcopy(state)
