"""Route acceptance follows measured crossings and keeps forbidden crossings."""

import math
from pathlib import Path

import numpy as np
import pytest

from go2_demo.complex_world import (
    DESTINATION_XY, EXIT_X, FIRST_GATE_X, GREEN_GATE_X, ROUTE_CENTER_Y,
    evaluate_route, final_scenario, learning_scenario,
)


def samples(points):
    return [{"base_xyz_m": [x, y, .32], "sim_time_s": 1. + index * .02}
            for index, (x, y) in enumerate(points)]


def valid_path(route):
    y = ROUTE_CENTER_Y[route]
    return samples([(1.7, y), (2.5, y), (2.7, y), (4.5, y), (4.5, 0.),
                    (5.1, 0.), (5.3, 0.), (5.99, 0.), (6.1, 0.)])


@pytest.mark.parametrize("route", ["blue", "yellow"])
def test_ordered_permitted_gate_green_and_full_clearance(route):
    result = evaluate_route(valid_path(route), route)
    assert result["success"]
    assert result["first_gate"] == route and not result["wrong_gate_first"]
    assert [e["gate"] for e in result["ordered_gate_events"]] == [route, "green", "clearance"]
    assert all(e["direction"] == "forward" for e in result["ordered_gate_events"])
    assert [e["crossing_xy_m"][0] for e in result["ordered_gate_events"]] == [FIRST_GATE_X, GREEN_GATE_X, EXIT_X]


def test_first_gate_interpolates_y_and_event_time():
    path = samples([(2.3, 1.0), (2.9, 1.6), (4.5, 0.), (6.2, 0.)])
    result = evaluate_route(path, "blue")
    assert result["success"]
    first = result["ordered_gate_events"][0]
    assert first["fraction"] == pytest.approx(.5)
    assert first["crossing_xy_m"] == pytest.approx([2.6, 1.3])
    assert first["sim_time_s"] == pytest.approx(1.01)
    last_two = result["ordered_gate_events"][-2:]
    assert last_two[0]["trajectory_index"] == last_two[1]["trajectory_index"]
    assert last_two[0]["fraction"] < last_two[1]["fraction"]


def test_wrong_first_gate_is_permanent_even_after_correct_reentry():
    path = samples([(2.7, -1.25), (2.4, -1.25), (2.4, 1.25),
                    (2.7, 1.25), (4.5, 1.25), (4.5, 0.), (6.2, 0.)])
    result = evaluate_route(path, "blue")
    assert not result["success"] and result["wrong_gate_first"]
    assert result["wrong_gate_crossings"] == 2
    assert result["green_clearance_reached"]
    assert any(e["direction"] == "reverse" for e in result["ordered_gate_events"])


def test_outside_first_barrier_cannot_be_repaired():
    path = samples([(2.7, 0.), (2.4, 0.), (2.4, 1.25),
                    (2.7, 1.25), (4.5, 1.25), (4.5, 0.), (6.2, 0.)])
    result = evaluate_route(path, "blue")
    assert not result["success"]
    assert result["first_barrier_outside_crossings"] == 2
    assert result["first_gate"] == "outside"


def test_green_crossing_without_prior_authorized_gate_is_out_of_order():
    result = evaluate_route(samples([(5.3, 0.), (6.2, 0.)]), "blue", start_xy=(4.0, 0.))
    assert not result["success"]
    assert result["out_of_order_crossings"] == 1
    assert result["green_crossed"] and not result["green_clearance_reached"]


def test_final_position_alone_or_empty_path_does_not_prove_route():
    assert not evaluate_route([], "blue")["success"]
    result = evaluate_route(samples([(6.2, 0.)]), "blue")
    assert not result["success"]
    assert result["first_barrier_outside_crossings"] == 1
    assert not evaluate_route(samples([(6.2, 0.)]), "blue", start_xy=(6.2, 0.))["success"]


def test_incomplete_clearance_and_return_after_clearance_fail():
    route = valid_path("blue")
    incomplete = evaluate_route(route[:-1], "blue")
    assert not incomplete["success"] and not incomplete["green_clearance_reached"]
    back = route + [{"base_xyz_m": [5.8, 0., .32], "sim_time_s": 2.0}]
    result = evaluate_route(back, "blue")
    assert not result["success"] and result["green_clearance_reached"]
    assert not result["beyond_green_clearance"]


def test_reverse_before_clearance_requires_new_green_crossing():
    path = samples([(2.5, 1.25), (2.7, 1.25), (4.5, 1.25), (4.5, 0.),
                    (5.3, 0.), (5.1, 0.), (5.4, 0.), (6.2, 0.)])
    result = evaluate_route(path, "blue")
    assert result["success"]
    greens = [e["direction"] for e in result["ordered_gate_events"] if e["gate"] == "green"]
    assert greens == ["forward", "reverse", "forward"]


def test_bypassing_green_gate_is_failure_even_if_later_crossing_is_correct():
    path = samples([(2.5, 1.25), (2.7, 1.25), (5.3, 1.25), (5.0, 1.25),
                    (5.0, 0.), (5.3, 0.), (6.2, 0.)])
    result = evaluate_route(path, "blue")
    assert not result["success"]
    assert result["green_outside_crossings"] == 2


@pytest.mark.parametrize("bad", [None, {}, {"base_xyz_m": [1, 2], "sim_time_s": 1.03},
    {"base_xyz_m": [math.nan, 0, .32], "sim_time_s": 1.03},
    {"base_xyz_m": [2.6, 1.25, .32], "sim_time_s": math.inf},
    {"base_xyz_m": [2.6, True, .32], "sim_time_s": 1.03}])
def test_invalid_samples_fail_without_bridging_unobserved_motion(bad):
    path = valid_path("blue")
    path.insert(2, bad)
    result = evaluate_route(path, "blue")
    assert not result["success"] and result["invalid_path"]


@pytest.mark.parametrize("bad_time", [1.0, 1.02])
def test_reversed_or_duplicate_times_fail(bad_time):
    path = valid_path("blue")
    path[2]["sim_time_s"] = bad_time
    result = evaluate_route(path, "blue")
    assert not result["success"] and result["invalid_path"]


@pytest.mark.parametrize("route", ["blue", "yellow"])
def test_body_center_margin_is_stricter_than_geometric_opening(route):
    y = ROUTE_CENTER_Y[route] + .60
    result = evaluate_route(samples([(2.5, y), (2.7, y), (4.5, y), (4.5, 0.), (6.2, 0.)]), route)
    assert not result["success"] and result["first_barrier_outside_crossings"] == 1


@pytest.mark.parametrize("route,start,trajectory", [("red", (0, 0), []), ("blue", (0, math.nan), []),
    ("blue", (0, 0), None)])
def test_invalid_configuration_is_rejected(route, start, trajectory):
    with pytest.raises(ValueError):
        evaluate_route(trajectory, route, start_xy=start)


def test_fixed_world_has_both_open_doors_distinct_signs_and_stable_geometry():
    scene = final_scenario()
    assert scene == final_scenario()
    assert scene.category == "memory_complex" and scene.goal_kind == "standoff"
    assert scene.target.xy == DESTINATION_XY["B"] and scene.target.letter == "B"
    objects = {obj.name: obj for obj in scene.objects}
    assert len(objects) == len(scene.objects)
    assert {objects["sign_" + letter].letter for letter in "AC"} == {"A", "C"}
    for route, center in ROUTE_CENTER_Y.items():
        left, right = objects[route + "_gate_left"], objects[route + "_gate_right"]
        assert left.position[0] == right.position[0] == FIRST_GATE_X
        assert (left.position[1] + right.position[1]) / 2 == center
        assert left.position[1] - left.size[1] - right.position[1] - right.size[1] == pytest.approx(1.4)
    assert objects["green_gate_left"].position[0] == GREEN_GATE_X


def test_learning_worlds_are_distinct_and_use_public_b_for_patrol_only():
    worlds = [learning_scenario(stage) for stage in ("dispatch", "fragments", "revision")]
    assert len({scene.task_id for scene in worlds}) == 3
    assert [scene.target.xy[0] for scene in worlds] == [5.6, 4.2, 3.5]
    assert all(scene.target.letter == "B" for scene in worlds)
    with pytest.raises(ValueError):
        learning_scenario("unrecognized")


def test_new_worlds_render_in_actual_go2_backend_without_api():
    from go2_demo.robots.registry import create_backend

    root = Path(__file__).resolve().parents[1]
    if not (root / "assets/go2_rl_lab/deploy/pre_train/go2/go2_cts_150k.pt").exists():
        pytest.skip("Checksum-verified Go2 assets unavailable")
    first_images = []
    for scene in (final_scenario(), learning_scenario("fragments"), learning_scenario("revision")):
        backend = create_backend("go2", "mujoco", asset_dir=root / "assets", seed=0, scenario=scene)
        try:
            frame = backend.capture()
            assert frame.rgb.shape == (360, 640, 3)
            assert np.isfinite(frame.depth_m).all()
            assert frame.robot_state["fallen"] is False
            assert backend._collision_steps == 0
            first_images.append(frame.rgb)
        finally:
            backend.close()
    assert not np.array_equal(first_images[0], first_images[1])
    assert not np.array_equal(first_images[1], first_images[2])
