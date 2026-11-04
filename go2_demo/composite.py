"""Shared measured-trajectory checks for the local drawers/gate/B task.

This module has no simulator or model dependencies. Crossings concern the base
reference point; physical collision and final target acceptance are checked
separately. An ordered chain can be completed after a collision-free retry.
"""

from __future__ import annotations

import math


def _finite(value) -> bool:
    try:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value))
    except OverflowError:
        return False


def _pose(row):
    if not isinstance(row, dict):
        return None
    xyz = row.get("base_xyz_m")
    if (not isinstance(xyz, (list, tuple)) or len(xyz) != 3
            or not all(_finite(value) for value in xyz)):
        return None
    timestamp = row.get("sim_time_s")
    if timestamp is not None and not _finite(timestamp):
        return None
    return tuple(float(value) for value in xyz), timestamp


def evaluate_composite_trajectory(trajectory: list[dict], *, cabinet_exit_x_m: float,
                                  passage_x_m: float, passage_center_y_m: float,
                                  passage_half_width_m: float, finish_x_min_m: float) -> dict:
    """Find cabinet clearance, an opening crossing and a subsequent exit event.

    Crossings must be forward and interpolate between adjacent finite measured
    poses. Segment index and interpolation fraction define order, including
    multiple events within one segment. Timestamps, when present, are reported
    for diagnostics. An invalid pose cannot bridge an unobserved crossing.
    """
    values = (cabinet_exit_x_m, passage_x_m, passage_center_y_m,
              passage_half_width_m, finish_x_min_m)
    if (not all(_finite(value) for value in values) or passage_half_width_m <= 0
            or not cabinet_exit_x_m < passage_x_m < finish_x_min_m):
        raise ValueError("Composite planes must be finite, ordered and have a positive opening half-width")
    if not isinstance(trajectory, list):
        raise ValueError("Composite trajectory must be a list")
    events = []
    previous = final_pose = None
    cabinet_marker = gate_marker = exit_marker = None
    passage_crossed = False
    planes = ((cabinet_exit_x_m, "cabinet_clearance"),
              (passage_x_m, "passage_crossing"), (finish_x_min_m, "passage_exit"))
    for index, row in enumerate(trajectory):
        pose = _pose(row)
        final_pose = pose
        if pose is None:
            previous = None
            continue
        if previous is not None:
            start, start_time = previous
            end, end_time = pose
            # A timestamp reversal cannot supply a measured forward segment.
            time_ordered = start_time is None or end_time is None or end_time > start_time
            if (time_ordered and exit_marker is None
                    and start[0] >= passage_x_m > end[0]):
                # Returning to the entry side before completing clearance
                # requires another genuine opening crossing on a retry.
                gate_marker = None
            if end[0] > start[0] and time_ordered:
                for plane, kind in planes:
                    if not start[0] < plane <= end[0]:
                        continue
                    fraction = (plane - start[0]) / (end[0] - start[0])
                    y = start[1] + fraction * (end[1] - start[1])
                    if kind == "passage_crossing" and abs(y - passage_center_y_m) > passage_half_width_m:
                        continue
                    marker = (index, fraction)
                    timestamp = (float(start_time + fraction * (end_time - start_time))
                                 if start_time is not None and end_time is not None else None)
                    events.append({"kind": kind, "trajectory_index": index, "fraction": fraction,
                                   "sim_time_s": timestamp, "crossing_y_m": y})
                    if kind == "cabinet_clearance" and cabinet_marker is None:
                        cabinet_marker = marker
                    elif kind == "passage_crossing":
                        passage_crossed = True
                        if cabinet_marker is not None and cabinet_marker < marker and gate_marker is None:
                            gate_marker = marker
                    elif kind == "passage_exit" and gate_marker is not None and gate_marker < marker:
                        exit_marker = marker
        previous = pose
    return {"cabinet_cleared": cabinet_marker is not None,
            "passage_crossed": passage_crossed,
            "beyond_exit": bool(final_pose is not None and final_pose[0][0] >= finish_x_min_m),
            "stages_in_order": exit_marker is not None,
            "events": events}
