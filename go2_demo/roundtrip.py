"""Private, measured-path acceptance for returning along an observed route.

This is a geometry evaluator, not a planner. It never supplies waypoints to an
agent and does not execute or infer motion. The caller separately checks arrival
at B, stopping, collisions, falls, finish and budgets. Both paths must contain
their measured boundary poses in one unchanged world coordinate frame.
"""

from __future__ import annotations

import math
from numbers import Real


def _finite(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _xy(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2 or not all(_finite(v) for v in value):
        raise ValueError(name + " must contain two finite coordinates")
    return tuple(float(v) for v in value)


def _read_path(rows, leg):
    if not isinstance(rows, list):
        raise ValueError(leg + " trajectory must be a list")
    points, errors, last_time = [], [], None
    for index, row in enumerate(rows):
        xyz = row.get("base_xyz_m") if isinstance(row, dict) else None
        timestamp = row.get("sim_time_s") if isinstance(row, dict) else None
        reason = None
        if (not isinstance(xyz, (tuple, list)) or len(xyz) != 3
                or not all(_finite(v) for v in xyz)):
            reason = "INVALID_POSITION"
        if not _finite(timestamp):
            reason = "INVALID_TIME"
        elif last_time is not None and timestamp <= last_time:
            reason = "NONINCREASING_TIME"
        if _finite(timestamp):
            last_time = max(last_time, float(timestamp)) if last_time is not None else float(timestamp)
        if reason:
            points.append(None)  # Never bridge a missing or disordered sample.
            errors.append({"leg": leg, "trajectory_index": index, "code": reason})
        else:
            points.append(((float(xyz[0]), float(xyz[1])), float(timestamp)))
    return points, errors


def _segments(points):
    return [(a[0], b[0]) for a, b in zip(points, points[1:])
            if a is not None and b is not None and math.dist(a[0], b[0]) > 1e-12]


def _sample_path(points, spacing):
    """Uniform arc-length samples, plus both endpoints of each valid component.

    Stationary periods do not dominate coverage. Invalid samples break the path;
    the entire result is invalid even though diagnostic distances remain useful.
    """
    result, previous, to_next = [], None, spacing
    for item in points:
        if item is None:
            if previous is not None and result[-1] != previous:
                result.append(previous)
            previous, to_next = None, spacing
            continue
        point = item[0]
        if previous is None:
            result.append(point)
            previous = point
            continue
        length = math.dist(previous, point)
        travelled = 0.
        while length - travelled >= to_next and length > 1e-12:
            travelled += to_next
            fraction = travelled / length
            result.append(tuple(a + fraction * (b - a) for a, b in zip(previous, point)))
            to_next = spacing
        to_next -= length - travelled
        previous = point
    if previous is not None and (not result or result[-1] != previous):
        result.append(previous)
    return result


def _distance_to_path(point, segments, fallback_points):
    best = math.inf
    for a, b in segments:
        dx, dy = b[0] - a[0], b[1] - a[1]
        fraction = max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(point[0] - a[0] - fraction * dx,
                                    point[1] - a[1] - fraction * dy))
    if math.isinf(best):
        best = min((math.dist(point, other) for other in fallback_points), default=math.inf)
    return best


def _crossings(points, leg, cabinet_x, cabinet_y, cabinet_clearance, gate_x, gate_y, gate_half_width):
    events = []
    for index, (start, end) in enumerate(zip(points, points[1:]), 1):
        if start is None or end is None or start[0][0] == end[0][0]:
            continue
        a, ta = start
        b, tb = end
        forward = b[0] > a[0]
        planes = ((cabinet_x, "cabinet"), (gate_x, "gate"))
        for plane, kind in planes if forward else reversed(planes):
            crossed = a[0] < plane <= b[0] if forward else b[0] < plane <= a[0]
            if not crossed:
                continue
            fraction = (plane - a[0]) / (b[0] - a[0])
            y = a[1] + fraction * (b[1] - a[1])
            event = {"leg": leg, "kind": kind, "direction": "forward" if forward else "reverse",
                     "trajectory_index": index, "fraction": fraction,
                     "sim_time_s": ta + fraction * (tb - ta), "crossing_xy_m": [plane, y]}
            if kind == "gate":
                event["inside_opening"] = abs(y - gate_y) <= gate_half_width
            else:
                event["side"] = ("positive_y" if y - cabinet_y >= cabinet_clearance else
                                 "negative_y" if y - cabinet_y <= -cabinet_clearance else "insufficient_clearance")
            events.append(event)
    return events


def _ordered(events, first_kind, second_kind, direction, side=None):
    first = None
    for event in events:
        if event["direction"] != direction:
            continue
        valid = (event["inside_opening"] if event["kind"] == "gate"
                 else event["side"] != "insufficient_clearance"
                 and (side is None or event["side"] == side))
        if valid and event["kind"] == first_kind and first is None:
            first = event
        if valid and event["kind"] == second_kind and first is not None:
            return first, event
    return None


def evaluate_roundtrip_trajectory(
        outbound_trajectory, return_trajectory, *, start_xy_m, turnaround_xy_m=None,
        passage_x_m=3.4, passage_center_y_m=0., passage_half_width_m=.60,
        cabinet_x_m=1.65, cabinet_center_y_m=0., cabinet_half_width_m=.33,
        cabinet_clearance_m=.12, return_tolerance_m=.30, retrace_tolerance_m=.50,
        min_outbound_coverage=.95, sample_spacing_m=.05, boundary_tolerance_m=.05):
    """Evaluate reverse gate/cabinet order and an observed-route corridor.

    The default geometry is the original drawers/gate/B scene. Retracing permits
    up to 0.50 m sampled deviation from the outbound measured polyline, and at
    least 95% of its arc-length samples must lie within 0.50 m of the return
    polyline. Distances are sampled every 0.05 m (not weighted by inference waits
    or stationary timesteps); the sample interval is reported. This is a route
    criterion, not a claim of identical footprints or proof of using memory.

    All samples require finite positions and strictly increasing times within
    each leg. The seam can repeat the same measured pose and timestamp. Invalid
    samples and noncontinuous boundary poses make success false. Stop and safety
    acceptance are deliberately left to the physical evaluator.
    """
    start = _xy(start_xy_m, "start_xy_m")
    values = (passage_x_m, passage_center_y_m, passage_half_width_m, cabinet_x_m,
              cabinet_center_y_m, cabinet_half_width_m, cabinet_clearance_m,
              return_tolerance_m, retrace_tolerance_m, min_outbound_coverage,
              sample_spacing_m, boundary_tolerance_m)
    if (not all(_finite(value) for value in values)
            or min(passage_half_width_m, cabinet_half_width_m, return_tolerance_m,
                   retrace_tolerance_m, sample_spacing_m, boundary_tolerance_m) <= 0
            or cabinet_clearance_m < 0 or not 0 < min_outbound_coverage <= 1
            or not start[0] < cabinet_x_m < passage_x_m):
        raise ValueError("Round-trip geometry and tolerances must be finite, positive and ordered")
    outbound, out_errors = _read_path(outbound_trajectory, "outbound")
    returning, return_errors = _read_path(return_trajectory, "return")
    out_first, out_last = (outbound[0], outbound[-1]) if outbound else (None, None)
    ret_first, ret_last = (returning[0], returning[-1]) if returning else (None, None)
    turnaround = (_xy(turnaround_xy_m, "turnaround_xy_m") if turnaround_xy_m is not None
                  else out_last[0] if out_last is not None else None)
    boundaries_ok = bool(out_first and out_last and ret_first and ret_last and turnaround
        and math.dist(out_first[0], start) <= boundary_tolerance_m
        and math.dist(out_last[0], turnaround) <= boundary_tolerance_m
        and math.dist(ret_first[0], turnaround) <= boundary_tolerance_m
        and ret_first[1] >= out_last[1] and turnaround[0] > passage_x_m)
    clearance = cabinet_half_width_m + cabinet_clearance_m
    out_events = _crossings(outbound, "outbound", cabinet_x_m, cabinet_center_y_m, clearance,
                            passage_x_m, passage_center_y_m, passage_half_width_m)
    ret_events = _crossings(returning, "return", cabinet_x_m, cabinet_center_y_m, clearance,
                            passage_x_m, passage_center_y_m, passage_half_width_m)
    outbound_chain = _ordered(out_events, "cabinet", "gate", "forward")
    side = outbound_chain[0]["side"] if outbound_chain else None
    return_chain = _ordered(ret_events, "gate", "cabinet", "reverse", side)
    cabinet_events = [e for e in out_events + ret_events if e["kind"] == "cabinet"]
    same_side = bool(side and return_chain and all(e["side"] == side for e in cabinet_events))
    outside_gate = sum(not e["inside_opening"] for e in out_events + ret_events if e["kind"] == "gate")
    out_segments, ret_segments = _segments(outbound), _segments(returning)
    out_samples, ret_samples = _sample_path(outbound, sample_spacing_m), _sample_path(returning, sample_spacing_m)
    deviations = [_distance_to_path(p, out_segments, out_samples) for p in ret_samples] if out_samples else []
    coverage_distances = [_distance_to_path(p, ret_segments, ret_samples) for p in out_samples] if ret_samples else []
    coverage = (sum(d <= retrace_tolerance_m for d in coverage_distances) / len(coverage_distances)
                if coverage_distances else 0.)
    fraction = sum(d <= retrace_tolerance_m for d in deviations) / len(deviations) if deviations else 0.
    max_deviation = max(deviations) if deviations else None
    mean_deviation = sum(deviations) / len(deviations) if deviations else None
    p95_deviation = sorted(deviations)[math.ceil(.95 * len(deviations)) - 1] if deviations else None
    distance = math.dist(ret_last[0], start) if ret_last else None
    corridor_ok = bool(max_deviation is not None and max_deviation <= retrace_tolerance_m
                       and coverage >= min_outbound_coverage)
    errors = out_errors + return_errors
    if out_last is not None and ret_first is not None and ret_first[1] < out_last[1]:
        errors.append({"leg": "return", "trajectory_index": 0, "code": "REVERSED_PHASE_TIME"})
    return {
        "success": bool(boundaries_ok and not errors and outbound_chain and return_chain and same_side
                        and not outside_gate and corridor_ok and distance is not None and distance <= return_tolerance_m),
        "source": "privileged_measured_roundtrip_trajectory",
        "acceptance_scope": "Geometry only; caller must check outbound B arrival, stop, collision, fall and budgets",
        "boundaries_continuous": boundaries_ok,
        "outbound_stages_in_order": bool(outbound_chain), "return_stages_in_order": bool(return_chain),
        "outbound_cabinet_side": side, "same_cabinet_side": same_side,
        "outside_gate_crossings": outside_gate, "return_distance_to_start_m": distance,
        "returned_to_start": bool(distance is not None and distance <= return_tolerance_m),
        "outbound_path_length_m": sum(math.dist(a, b) for a, b in out_segments),
        "return_path_length_m": sum(math.dist(a, b) for a, b in ret_segments),
        "retrace_max_deviation_m": max_deviation, "retrace_mean_deviation_m": mean_deviation,
        "retrace_p95_deviation_m": p95_deviation, "outbound_coverage": coverage,
        "return_corridor_fraction": fraction, "retrace_corridor_passed": corridor_ok,
        "outbound_sample_count": len(out_samples), "return_sample_count": len(ret_samples),
        "sample_spacing_m": sample_spacing_m, "retrace_tolerance_m": retrace_tolerance_m,
        "min_outbound_coverage": min_outbound_coverage, "return_tolerance_m": return_tolerance_m,
        "invalid_path": bool(errors), "invalid_samples": len(errors), "invalid_details": errors,
        "events": out_events + ret_events,
    }
