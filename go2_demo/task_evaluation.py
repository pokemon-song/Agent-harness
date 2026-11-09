"""Independent task acceptance; privileged state never enters observations.

This module has no MuJoCo or robot-policy dependency. Completion tolerances,
base-point crossing rules and cumulative safety criteria match the original Go2
scorer; finish and episode budgets remain orchestration requirements.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .composite import evaluate_composite_trajectory


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def crossed_passage(trajectory: list[dict], gate_x: float, center_y: float, half_width: float) -> bool:
    """Check a forward gate crossing from measured poses, rather than final position alone."""
    previous = None
    for row in trajectory:
        point = np.asarray(row.get("base_xyz_m", ()), dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            previous = None
            continue
        if previous is not None and previous[0] < gate_x <= point[0]:
            fraction = (gate_x - previous[0]) / (point[0] - previous[0])
            crossing_y = float(previous[1] + fraction * (point[1] - previous[1]))
            if abs(crossing_y - center_y) <= half_width:
                return True
        previous = point
    return False


@dataclass(frozen=True)
class TaskEvaluationState:
    position_xyz_m: np.ndarray
    yaw_rad: float
    velocity_xy_mps: np.ndarray
    yaw_rate_radps: float
    fallen: bool
    collision_steps: int
    trajectory: list[dict]
    sim_time_s: float


class TaskEvaluator:
    """Own task criteria and score an explicit, private simulator snapshot."""

    def __init__(self, scenario, target_xy, snapshot: Callable[[], TaskEvaluationState]):
        self.scenario = scenario
        self._target_xy = np.asarray(target_xy, dtype=float).copy()
        self._snapshot = snapshot

    def criteria(self, standoff_m: float = 1.0, tolerance_m: float = 0.18) -> dict:
        """Privileged archived criteria; never use as a model observation."""
        scenario = self.scenario
        goal_kind = scenario.goal_kind if scenario is not None else "standoff"
        if scenario is not None:
            standoff_m, tolerance_m = scenario.standoff_m, scenario.distance_tolerance_m
        passage = goal_kind == "passage"
        result = {"goal_kind": goal_kind, "standoff_m": None if passage else standoff_m,
                  "distance_tolerance_m": None if passage else tolerance_m,
                  "heading_tolerance_rad": None if passage else 0.25,
                  "max_speed_mps": 0.12, "max_yaw_rate_radps": 0.25,
                  "requires_finish": True, "no_collision_or_fall": True}
        if passage or goal_kind == "composite":
            result.update(passage_x_m=scenario.passage_x,
                          passage_center_y_m=scenario.passage_center_y,
                          passage_half_width_m=scenario.passage_half_width,
                          finish_x_min_m=scenario.finish_x_min,
                          requires_forward_crossing=True)
        if goal_kind == "composite":
            result.update(cabinet_exit_x_m=scenario.cabinet_exit_x, requires_stage_order=True)
        return result

    @property
    def public_task_spec(self) -> dict:
        """Declared acceptance rules; no privileged positions or score flags."""
        scenario = self.scenario
        goal_kind = scenario.goal_kind if scenario is not None else "standoff"
        result = {"goal_kind": goal_kind, "reference_body": "base",
                  "stop_required": True, "collision_free_required": True,
                  "max_stopped_speed_mps": 0.12, "max_stopped_yaw_rate_radps": 0.25}
        if goal_kind == "passage":
            result["required_clearance_m"] = scenario.passage_clearance_m
        else:
            result.update(standoff_m=scenario.standoff_m if scenario is not None else 1.0,
                          distance_tolerance_m=scenario.distance_tolerance_m if scenario is not None else 0.18,
                          heading_tolerance_rad=0.25)
        if goal_kind == "composite":
            result.update(stages_ordered=True, stage_rules=[
                {"stage_id": "avoid_drawers", "instruction": "先绕过抽屉柜，让机身base原点通过柜体后方并留出机身余量。"},
                {"stage_id": "pass_gate", "instruction": "随后让机身base原点从绿色门框开口内向前穿过，继续前进规定余量。",
                 "required_clearance_m": scenario.passage_clearance_m},
                {"stage_id": "approach_b", "instruction": "最后接近字母B至公开的目标距离与朝向容差内，停稳后调用finish。"},
            ])
        return result

    def evaluate(self, standoff_m: float = 1.0, tolerance_m: float = 0.18) -> dict:
        """Privileged state: only the independent evaluator may consume this result."""
        scenario = self.scenario
        if scenario is not None:
            standoff_m, tolerance_m = scenario.standoff_m, scenario.distance_tolerance_m
        state = self._snapshot()
        position, yaw = state.position_xyz_m, state.yaw_rad
        difference = self._target_xy - position[:2]
        distance = float(np.linalg.norm(difference))
        heading_error = abs(wrap_angle(math.atan2(difference[1], difference[0]) - yaw))
        speed = float(np.linalg.norm(state.velocity_xy_mps))
        stopped = speed < 0.12 and abs(float(state.yaw_rate_radps)) < 0.25
        goal_kind = scenario.goal_kind if scenario is not None else "standoff"
        goal_reached = abs(distance-standoff_m) <= tolerance_m and heading_error <= 0.25
        passage = {}
        if goal_kind == "passage":
            if scenario.passage_x is None or scenario.finish_x_min is None:
                raise ValueError("Passage scenarios require a gate plane and an exit threshold")
            crossed = crossed_passage(state.trajectory, scenario.passage_x, scenario.passage_center_y,
                                      scenario.passage_half_width)
            beyond_exit = bool(position[0] >= scenario.finish_x_min)
            goal_reached = crossed and beyond_exit
            passage = {"passage_crossed": crossed, "beyond_exit": beyond_exit,
                       "gate_x_m": scenario.passage_x, "finish_x_min_m": scenario.finish_x_min}
        elif goal_kind == "composite":
            measured_path = evaluate_composite_trajectory(
                state.trajectory, cabinet_exit_x_m=scenario.cabinet_exit_x,
                passage_x_m=scenario.passage_x, passage_center_y_m=scenario.passage_center_y,
                passage_half_width_m=scenario.passage_half_width, finish_x_min_m=scenario.finish_x_min)
            # Use the measured current position as well as the recorded path.
            measured_path["beyond_exit"] = bool(measured_path["beyond_exit"] and position[0] >= scenario.finish_x_min)
            stages = {"avoid_drawers": measured_path["cabinet_cleared"],
                      "pass_gate": (measured_path["passage_crossed"] and measured_path["beyond_exit"]
                                    and measured_path["stages_in_order"]),
                      "approach_b": bool(goal_reached and stopped)}
            goal_reached = all(stages.values())
            passage = {**measured_path, "cabinet_exit_x_m": scenario.cabinet_exit_x,
                       "gate_x_m": scenario.passage_x, "finish_x_min_m": scenario.finish_x_min,
                       "stages": stages, "stage_completion_rate": sum(stages.values()) / len(stages)}
        return {"success": bool(goal_reached and stopped and not state.fallen and not state.collision_steps),
                "goal_kind": goal_kind, "distance_to_target_center_m": distance,
                "distance_error_m": abs(distance-standoff_m), "heading_error_rad": heading_error,
                "standoff_m": standoff_m, "distance_tolerance_m": tolerance_m,
                "stopped": bool(stopped), "speed_mps": speed, "fallen": bool(state.fallen),
                "collision": bool(state.collision_steps), "collision_physics_steps": state.collision_steps,
                "target_xy_m": self._target_xy.tolist(), "base_xy_m": position[:2].tolist(),
                "sim_time_s": float(state.sim_time_s), "source": "privileged_simulator_evaluator", **passage}

