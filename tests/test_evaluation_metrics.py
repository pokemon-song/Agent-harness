"""Measurement boundaries and success gates for the multidimensional score."""

from copy import deepcopy
import math

import pytest

from go2_demo.evaluation_metrics import SCORING_V2, score_components, trajectory_metrics


START = {"sim_time_s": 1.0, "base_xy_m": [0.0, 0.0], "yaw_rad": 0.0}
REFERENCE = {"goal_kind": "standoff", "target_xy_m": [0.03, 0.0],
             "standoff_m": 0.01, "distance_tolerance_m": 0.0}


def trace():
    return [
        {"sim_time_s": 1.02, "base_xyz_m": [0.02, 0.0, 0.34], "yaw_rad": 0.1, "command": [0, 0, 0]},
        {"sim_time_s": 1.04, "base_xyz_m": [0.04, 0.0, 0.8], "yaw_rad": 0.2, "command": [0, 0, 0]},
    ]


def measure(points=None, **overrides):
    arguments = {"start": deepcopy(START), "final_xy": [0.04, 0.0], "sim_elapsed_s": 0.04,
                 "reference": deepcopy(REFERENCE)}
    arguments.update(overrides)
    return trajectory_metrics(trace() if points is None else points, **arguments)


def test_counts_initial_interval_zero_command_drift_and_only_horizontal_distance():
    result = measure()
    assert result == pytest.approx({"path_length_m": 0.04, "reference_distance_m": 0.02,
                                    "path_efficiency": 0.5, "net_displacement_m": 0.04,
                                    "excess_path_m": 0.02, "total_yaw_rad": 0.2})
    assert sum(SCORING_V2["weights"].values()) == 100


def test_backtracking_increases_path_without_increasing_net_displacement():
    points = trace()
    points[-1]["base_xyz_m"] = [0.0, 0.0, 0.34]
    result = measure(points, final_xy=[0.0, 0.0])
    assert result["path_length_m"] == pytest.approx(0.04)
    assert result["net_displacement_m"] == 0.0
    assert result["path_efficiency"] == pytest.approx(0.5)


def test_passage_lower_bound_uses_exit_threshold_and_actual_start_x():
    start = {**START, "base_xy_m": [-0.01, 4.0]}
    points = trace()
    for point in points:
        point["base_xyz_m"][1] = 4.0
    result = measure(points, start=start, final_xy=[0.04, 4.0],
                     reference={"goal_kind": "passage", "finish_x_min_m": 3.0})
    assert result["reference_distance_m"] == pytest.approx(3.01)
    assert result["path_efficiency"] == 1.0  # Goal attainment belongs to the caller.


@pytest.mark.parametrize("reference_distance,motion,expected", [
    (0.0, 0.0, 1.0), (0.02, 0.0, 0.0), (0.0, 0.04, 0.0), (0.08, 0.04, 1.0),
])
def test_zero_length_and_short_failed_paths(reference_distance, motion, expected):
    points = trace()
    points[0]["base_xyz_m"][0] = motion / 2
    points[1]["base_xyz_m"][0] = motion
    result = measure(points, final_xy=[motion, 0.0],
                     reference={"goal_kind": "passage", "finish_x_min_m": reference_distance})
    assert result["path_efficiency"] == expected


def test_yaw_wrap_does_not_turn_two_degree_crossing_into_full_revolution():
    start = {**START, "yaw_rad": math.pi - 0.01}
    points = trace()
    points[0]["yaw_rad"] = -math.pi + 0.01
    points[1]["yaw_rad"] = -math.pi + 0.03
    assert measure(points, start=start)["total_yaw_rad"] == pytest.approx(0.04)


@pytest.mark.parametrize("missing", ["start", 0, 1])
def test_missing_yaw_preserves_path_and_reports_unknown_rotation(missing):
    start, points = deepcopy(START), trace()
    if missing == "start":
        start.pop("yaw_rad")
    else:
        points[missing].pop("yaw_rad")
    result = measure(points, start=start)
    assert result["total_yaw_rad"] is None
    assert result["path_length_m"] == pytest.approx(0.04)


@pytest.mark.parametrize("points", [[], None, {}, [None]])
def test_empty_or_non_object_trace_is_invalid(points):
    with pytest.raises(ValueError):
        trajectory_metrics(points, start=START, final_xy=[0, 0], sim_elapsed_s=0.04, reference=REFERENCE)


@pytest.mark.parametrize("mutation", [
    lambda p: p[0].update(sim_time_s=1.0),
    lambda p: p[0].update(sim_time_s=1.03),
    lambda p: p[1].update(sim_time_s=1.02),
    lambda p: p[1].update(sim_time_s=1.01),
    lambda p: p[1].update(sim_time_s=1.06),
    lambda p: p[0].update(base_xyz_m=[0.0, 0.0]),
    lambda p: p[0].update(base_xyz_m=[0.0, 0.0, True]),
    lambda p: p[0].update(base_xyz_m=[math.inf, 0.0, 0.3]),
    lambda p: p[0].update(yaw_rad=math.nan),
    lambda p: p[0].update(yaw_rad=False),
    lambda p: p[0].update(sim_time_s=True),
])
def test_sparse_unordered_or_nonfinite_trace_is_invalid(mutation):
    points = trace()
    mutation(points)
    with pytest.raises(ValueError):
        measure(points)


@pytest.mark.parametrize("overrides", [
    {"start": {**START, "base_xy_m": [False, 0.0]}},
    {"start": {**START, "sim_time_s": -1.0}},
    {"start": {**START, "yaw_rad": math.inf}},
    {"final_xy": [0.04, math.nan]},
    {"final_xy": [0.0401, 0.0]},
    {"sim_elapsed_s": 0.0},
    {"sim_elapsed_s": True},
    {"sim_elapsed_s": 0.0401},
    {"reference": {"goal_kind": "other"}},
    {"reference": {**REFERENCE, "target_xy_m": [0.03, True]}},
    {"reference": {**REFERENCE, "standoff_m": -0.01}},
    {"reference": {"goal_kind": "passage", "finish_x_min_m": math.inf}},
])
def test_start_endpoint_duration_and_reference_are_validated(overrides):
    with pytest.raises(ValueError):
        measure(**overrides)


def test_small_timestamp_roundoff_and_endpoint_tolerance_are_accepted():
    points = trace()
    points[0]["sim_time_s"] += 0.0000005
    result = measure(points, final_xy=[0.040005, 0.0], sim_elapsed_s=0.040005)
    assert result["path_length_m"] == pytest.approx(0.04)


def test_success_score_uses_fixed_weights_and_normalization():
    result = score_components(strict_success=True, path_efficiency=0.5,
                              wall_elapsed_s=150, sim_elapsed_s=60, model_calls=15)
    assert result == pytest.approx({"completion": 60, "path_efficiency": 10, "wall_time": 5,
                                    "sim_time": 2.5, "model_calls": 5 * 15 / 29,
                                    "total_100": 77.5 + 5 * 15 / 29})


@pytest.mark.parametrize("path,wall,simulation,calls", [(1, 0, 0, 0), (1, 1, 1, 1), (0, 500, 150, 35)])
def test_failed_stationary_or_early_terminated_trials_receive_no_component_points(path, wall, simulation, calls):
    result = score_components(strict_success=False, path_efficiency=path,
                              wall_elapsed_s=wall, sim_elapsed_s=simulation, model_calls=calls)
    assert set(result) == {*SCORING_V2["weights"], "total_100"}
    assert all(value == 0 for value in result.values())


def test_score_extremes_and_exact_budget_boundary():
    assert score_components(strict_success=True, path_efficiency=1, wall_elapsed_s=0,
                            sim_elapsed_s=0, model_calls=1)["total_100"] == 100
    assert score_components(strict_success=True, path_efficiency=0, wall_elapsed_s=300,
                            sim_elapsed_s=120, model_calls=30)["total_100"] == 60


@pytest.mark.parametrize("overrides", [
    {"strict_success": 1}, {"path_efficiency": math.nan}, {"path_efficiency": True},
    {"path_efficiency": -0.1}, {"path_efficiency": 1.01}, {"wall_elapsed_s": math.inf},
    {"wall_elapsed_s": -1}, {"sim_elapsed_s": False}, {"model_calls": True},
    {"model_calls": 2.5}, {"model_calls": -1}, {"model_calls": 0},
    {"model_calls": 31}, {"wall_elapsed_s": 300.01}, {"sim_elapsed_s": 120.01},
])
def test_score_rejects_malformed_inputs_and_false_success_outside_budgets(overrides):
    arguments = {"strict_success": True, "path_efficiency": 0.5,
                 "wall_elapsed_s": 50, "sim_elapsed_s": 20, "model_calls": 5}
    arguments.update(overrides)
    with pytest.raises(ValueError):
        score_components(**arguments)
