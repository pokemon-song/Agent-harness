"""Offline trajectory measurements and success-gated benchmark components.

Path efficiency compares measured travel to a straight-line lower bound. It is
not SPL: obstacle-aware shortest feasible paths are not available here.
"""

from __future__ import annotations

import math


_MAX_SAMPLE_GAP_S = 0.020001
_TIME_TOLERANCE_S = 1e-5
_POSITION_TOLERANCE_M = 1e-5

SCORING_V2 = {
    "weights": {
        "completion": 60,
        "path_efficiency": 20,
        "wall_time": 10,
        "sim_time": 5,
        "model_calls": 5,
    },
    "formula": "strict_success * (60 + 20*P + 10*W + 5*T + 5*C)",
    "failure_score": 0,
    "normalization": {
        "wall_time": "clip(1 - wall_elapsed_s / 300, 0, 1)",
        "sim_time": "clip(1 - sim_elapsed_s / 120, 0, 1)",
        "model_calls": "clip((30 - model_calls) / 29, 0, 1)",
        "path_efficiency": "min(1, reference_distance_m / max(path_length_m, reference_distance_m))",
        "zero_path": "P=1 if L=D=0; P=0 if exactly one of L,D is zero",
        "standoff_reference": "max(0, distance(start_xy, target_xy) - standoff_m - distance_tolerance_m)",
        "passage_reference": "max(0, finish_x_min_m - start_x)",
        "path_reference_kind": "straight_line_lower_bound_not_obstacle_aware_shortest_path_or_SPL",
    },
    "trajectory": {
        "nominal_sample_period_s": 0.02,
        "maximum_sample_gap_s": _MAX_SAMPLE_GAP_S,
        "final_time_tolerance_s": _TIME_TOLERANCE_S,
        "final_xy_tolerance_m": _POSITION_TOLERANCE_M,
        "start": "prepend the exact episode-start XY pose after controller warmup",
        "length": "sum of successive world-XY Euclidean distances, including all settling and zero-command drift",
        "time": "strictly increasing from episode start through the final recorded simulation state",
        "yaw": "sum of absolute wrapped successive yaw changes including start; null if any yaw is missing",
        "noise_filtering": "none",
    },
}


def _finite(value, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite" + (f" and at least {minimum}" if minimum is not None else ""))
    return result


def _vector(value, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} finite numbers")
    return tuple(_finite(item, name) for item in value)


def _optional_yaw(value, name: str) -> float | None:
    return None if value is None else _finite(value, name)


def _reference_distance(reference: dict, start_xy: tuple[float, ...]) -> float:
    if not isinstance(reference, dict):
        raise ValueError("reference must be an object")
    if reference.get("goal_kind") == "standoff":
        target = _vector(reference.get("target_xy_m"), 2, "target_xy_m")
        standoff = _finite(reference.get("standoff_m"), "standoff_m", minimum=0)
        tolerance = _finite(reference.get("distance_tolerance_m"), "distance_tolerance_m", minimum=0)
        distance = math.dist(start_xy, target)
        if not math.isfinite(distance):
            raise ValueError("Reference distance overflowed")
        return max(0.0, distance - standoff - tolerance)
    if reference.get("goal_kind") == "passage":
        finish_x = _finite(reference.get("finish_x_min_m"), "finish_x_min_m")
        return _finite(max(0.0, finish_x - start_xy[0]), "reference_distance_m")
    raise ValueError("reference.goal_kind must be standoff or passage")


def trajectory_metrics(trajectory: list[dict], *, start: dict, final_xy: list,
                       sim_elapsed_s: float, reference: dict) -> dict:
    """Measure a complete 50 Hz trace, raising for missing or sparse evidence.

    The initial warmup is excluded. The exact starting point is prepended, and
    every subsequent XY displacement is counted, including stopping drift.
    Missing yaw only makes ``total_yaw_rad`` unknown; provided yaw must be finite.
    A short failed path is allowed here: the caller owns goal/success checks.
    """
    if not isinstance(start, dict):
        raise ValueError("start must be an object")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("trajectory must contain at least one recorded point")
    start_time = _finite(start.get("sim_time_s"), "start.sim_time_s", minimum=0)
    start_xy = _vector(start.get("base_xy_m"), 2, "start.base_xy_m")
    final_xy = _vector(final_xy, 2, "final_xy")
    elapsed = _finite(sim_elapsed_s, "sim_elapsed_s", minimum=0)
    if elapsed == 0:
        raise ValueError("sim_elapsed_s must be positive")
    expected_end = _finite(start_time + elapsed, "final simulation time")
    reference_distance = _reference_distance(reference, start_xy)
    previous_time, previous_xy = start_time, start_xy
    previous_yaw = _optional_yaw(start.get("yaw_rad"), "start.yaw_rad")
    yaw_complete = previous_yaw is not None
    lengths, yaw_changes = [], []
    for index, point in enumerate(trajectory):
        if not isinstance(point, dict):
            raise ValueError("Every trajectory point must be an object")
        timestamp = _finite(point.get("sim_time_s"), f"trajectory[{index}].sim_time_s", minimum=0)
        gap = timestamp - previous_time
        if gap <= 0 or gap > _MAX_SAMPLE_GAP_S:
            raise ValueError("Trajectory times must increase with gaps no greater than 0.020001 s")
        xyz = _vector(point.get("base_xyz_m"), 3, f"trajectory[{index}].base_xyz_m")
        xy = xyz[:2]
        lengths.append(_finite(math.dist(previous_xy, xy), "trajectory displacement", minimum=0))
        yaw = _optional_yaw(point.get("yaw_rad"), f"trajectory[{index}].yaw_rad")
        if yaw is None:
            yaw_complete = False
        elif previous_yaw is not None:
            change = yaw - previous_yaw
            _finite(change, "yaw change")
            yaw_changes.append(abs(math.atan2(math.sin(change), math.cos(change))))
        previous_time, previous_xy, previous_yaw = timestamp, xy, yaw
    if abs(previous_time - expected_end) > _TIME_TOLERANCE_S:
        raise ValueError("Trajectory final time disagrees with sim_elapsed_s")
    if math.dist(previous_xy, final_xy) > _POSITION_TOLERANCE_M:
        raise ValueError("Trajectory final XY disagrees with the evaluated final position")
    try:
        length = _finite(math.fsum(lengths), "path_length_m", minimum=0)
        total_yaw = _finite(math.fsum(yaw_changes), "total_yaw_rad", minimum=0) if yaw_complete else None
    except OverflowError:
        raise ValueError("Trajectory measurements overflowed") from None
    if length == 0:
        efficiency = 1.0 if reference_distance == 0 else 0.0
    elif reference_distance == 0:
        efficiency = 0.0
    else:
        efficiency = min(1.0, reference_distance / max(length, reference_distance))
    return {
        "path_length_m": length,
        "reference_distance_m": reference_distance,
        "path_efficiency": efficiency,
        "net_displacement_m": _finite(math.dist(start_xy, previous_xy), "net_displacement_m", minimum=0),
        "excess_path_m": max(0.0, length - reference_distance),
        "total_yaw_rad": total_yaw,
    }


def score_components(*, strict_success: bool, path_efficiency: float,
                     wall_elapsed_s: float, sim_elapsed_s: float, model_calls: int) -> dict:
    """Return five weighted components and their total; failed trials score zero."""
    if type(strict_success) is not bool:
        raise ValueError("strict_success must be a boolean")
    efficiency = _finite(path_efficiency, "path_efficiency", minimum=0)
    if efficiency > 1:
        raise ValueError("path_efficiency must not exceed 1")
    wall = _finite(wall_elapsed_s, "wall_elapsed_s", minimum=0)
    simulation = _finite(sim_elapsed_s, "sim_elapsed_s", minimum=0)
    if type(model_calls) is not int or model_calls < 0:
        raise ValueError("model_calls must be a nonnegative integer")
    if strict_success and (not 1 <= model_calls <= 30 or wall > 300 or simulation > 120):
        raise ValueError("A successful trial must stay within all frozen budgets and make at least one model call")
    normalized = {
        "completion": 1.0,
        "path_efficiency": efficiency,
        "wall_time": max(0.0, min(1.0, 1.0 - wall / 300.0)),
        "sim_time": max(0.0, min(1.0, 1.0 - simulation / 120.0)),
        "model_calls": max(0.0, min(1.0, (30 - model_calls) / 29.0)),
    }
    components = {name: float(weight * normalized[name]) if strict_success else 0.0
                  for name, weight in SCORING_V2["weights"].items()}
    return {**components, "total_100": math.fsum(components.values())}
