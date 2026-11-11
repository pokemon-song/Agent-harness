"""Acceptance failures which a final-position-only return check would miss."""

from copy import deepcopy
import math

import pytest

from go2_demo.roundtrip import evaluate_roundtrip_trajectory


OUTBOUND = [(0., 0.), (.65, -.72), (1.65, -.72), (2.4, -.72), (2.9, -.12),
            (3.4, 0.), (4.3, 0.), (4.5, 0.)]


def path(points, start_time=0.):
    return [{"sim_time_s": start_time + i, "base_xyz_m": [x, y, .3]}
            for i, (x, y) in enumerate(points)]


def score(outbound=None, returning=None, **kwargs):
    outbound = path(OUTBOUND) if outbound is None else outbound
    returning = path(list(reversed(OUTBOUND)), len(OUTBOUND) - 1) if returning is None else returning
    return evaluate_roundtrip_trajectory(outbound, returning, start_xy_m=[0., 0.], **kwargs)


def test_exact_reverse_in_same_world_accepts_repeated_seam_and_records_distance():
    value = score()
    assert value["success"] and value["same_cabinet_side"]
    assert value["outbound_cabinet_side"] == "negative_y"
    assert value["retrace_max_deviation_m"] < 1e-12
    assert value["outbound_coverage"] == 1.
    assert value["return_path_length_m"] == pytest.approx(value["outbound_path_length_m"])
    reverse = [e for e in value["events"] if e["leg"] == "return"]
    assert [e["kind"] for e in reverse] == ["gate", "cabinet"]


def test_positive_side_is_learned_from_outbound_not_fixed_in_code():
    outbound = [(x, -y) for x, y in OUTBOUND]
    value = score(path(outbound), path(list(reversed(outbound)), 7.))
    assert value["success"]
    assert value["outbound_cabinet_side"] == "positive_y"


def test_small_retrace_variation_and_start_tolerance_are_allowed():
    points = [(4.5, 0.), (4.3, .10), (3.4, .10), (2.9, -.02),
              (2.4, -.62), (1.65, -.62), (.65, -.62), (.15, .08)]
    value = score(returning=path(points, 7.))
    assert value["success"]
    assert 0 < value["return_distance_to_start_m"] < .30
    assert 0 < value["retrace_max_deviation_m"] < .50


def test_same_side_detour_is_not_retracing():
    points = [(4.5, 0.), (3.4, 0.), (2.9, -.1), (2.4, -1.7), (1.65, -1.7), (.65, -1.7), (0., 0.)]
    value = score(returning=path(points, 7.))
    assert value["same_cabinet_side"] and value["return_stages_in_order"]
    assert not value["success"] and not value["retrace_corridor_passed"]
    assert value["retrace_max_deviation_m"] > .5


def test_wrong_side_return_fails_even_with_relaxed_corridor():
    points = [(4.5, 0.), (3.4, 0.), (2.4, .72), (1.65, .72), (.65, .72), (0., 0.)]
    value = score(returning=path(points, 7.), retrace_tolerance_m=3.)
    assert not value["success"] and not value["same_cabinet_side"]


def test_bypassing_gate_fails_even_with_relaxed_corridor():
    points = [(4.5, 0.), (3.8, 1.5), (3., 1.5), (2.4, -.72), (1.65, -.72), (.65, -.72), (0., 0.)]
    value = score(returning=path(points, 7.), retrace_tolerance_m=3.)
    assert value["outside_gate_crossings"] == 1
    assert not value["return_stages_in_order"] and not value["success"]


def test_direct_shortcut_through_cabinet_fails():
    value = score(returning=path([(4.5, 0.), (3.3, 0.), (0., 0.)], 7.))
    assert value["returned_to_start"] and not value["same_cabinet_side"]
    assert not value["success"]


def test_return_without_reverse_gate_evidence_fails_boundary_and_order():
    value = score(returning=path([(3., 0.), (2.4, -.72), (1.65, -.72), (0., 0.)], 7.))
    assert not value["boundaries_continuous"] and not value["return_stages_in_order"]
    assert not value["success"]


def test_stopping_short_of_origin_fails_even_when_all_crossings_are_valid():
    points = list(reversed(OUTBOUND))[:-1] + [(.35, 0.)]
    value = score(returning=path(points, 7.))
    assert value["return_stages_in_order"]
    assert not value["returned_to_start"] and not value["success"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1", 10 ** 1000])
def test_invalid_position_cannot_bridge_a_gate_crossing(bad):
    returning = path(list(reversed(OUTBOUND)), 7.)
    returning[2]["base_xyz_m"][0] = bad
    value = score(returning=returning)
    assert value["invalid_path"] and value["invalid_samples"] == 1
    assert not value["success"] and not value["return_stages_in_order"]


@pytest.mark.parametrize("timestamp", [7., 8., float("nan"), None, True])
def test_nonincreasing_or_invalid_time_is_not_silently_ignored(timestamp):
    returning = path(list(reversed(OUTBOUND)), 7.)
    returning[2]["sim_time_s"] = timestamp
    value = score(returning=returning)
    assert value["invalid_path"] and not value["success"]


def test_bidirectional_coverage_rejects_skipped_outbound_loop():
    # A large outward loop near the start is skipped on return. Every return
    # point can still be on the outbound route, so one-way corridor checks pass.
    points = [(0., 0.), (-1., 0.), (-1., -1.8), (0., 0.)] + OUTBOUND[1:]
    outbound = path(points)
    value = score(outbound, path(list(reversed(OUTBOUND)), len(points) - 1))
    assert value["retrace_max_deviation_m"] < 1e-12
    assert value["outbound_coverage"] < .95
    assert not value["success"]


def test_stationary_sampling_density_does_not_change_route_coverage():
    base = score()
    outbound = path(OUTBOUND[:1] * 1000 + OUTBOUND[1:])
    returning = path(list(reversed(OUTBOUND)), outbound[-1]["sim_time_s"])
    value = score(outbound, returning)
    assert value["success"]
    for key in ("outbound_coverage", "return_corridor_fraction", "outbound_path_length_m",
                "outbound_sample_count", "return_sample_count"):
        assert value[key] == base[key]


def test_empty_or_only_stationary_trajectories_are_not_success():
    assert not score([], [])["success"]
    assert not score(path([(0., 0.), (0., 0.)]), path([(0., 0.), (0., 0.)], 1.))["success"]


def test_late_turnaround_or_wrong_turnaround_coordinate_fails():
    value = score(turnaround_xy_m=[4.7, 0.])
    assert not value["boundaries_continuous"] and not value["success"]
    value = score(returning=path(list(reversed(OUTBOUND)), 6.9))
    assert not value["boundaries_continuous"] and not value["success"] and value["invalid_path"]


@pytest.mark.parametrize("kwargs", [{"retrace_tolerance_m": 0.}, {"sample_spacing_m": -1.},
    {"min_outbound_coverage": 1.1}, {"passage_x_m": 1.}, {"cabinet_clearance_m": -1.},
    {"return_tolerance_m": math.inf}])
def test_invalid_acceptance_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        score(**kwargs)


def test_inputs_remain_unchanged():
    outbound, returning = path(OUTBOUND), path(list(reversed(OUTBOUND)), 7.)
    before = deepcopy((outbound, returning))
    score(outbound, returning)
    assert (outbound, returning) == before
