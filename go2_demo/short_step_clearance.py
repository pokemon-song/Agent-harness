"""Translation check against obstacles observed by the current front depth camera.

None means no visible evidence blocked this translation. The front-camera blind
zones, hidden surfaces, cliff edges, rotations and gait deviations are not certified.
"""
import math

import numpy as np


# Padded horizontal envelope for this Go2's gait (nose at x=0.34 m).
HALF_LENGTH_M = 0.40
HALF_WIDTH_M = 0.25


def check_clearance(frame, public_observation, forward_m, lateral_m):
    if forward_m == 0 and lateral_m == 0:
        return None
    unknown = {"error_code": "LOCAL_DEPTH_UNAVAILABLE",
               "reason": "Cannot evaluate this step from the current calibrated front depth view."}
    cal = public_observation.get("sensor_calibration", {})
    state = public_observation.get("robot_state", {})
    try:
        rotation = np.asarray(cal["camera_optical_to_body_rotation"], dtype=float)
        offset = np.asarray(cal["camera_position_in_body_m"], dtype=float)
        roll, pitch, height = (float(state[k]) for k in ("body_roll_rad", "body_pitch_rad", "body_height_m"))
        K = np.asarray(frame.intrinsics, dtype=float)
        depth = np.asarray(frame.depth_m)
    except (KeyError, TypeError, ValueError):
        return unknown
    if (rotation.shape != (3, 3) or offset.shape != (3,) or K.shape != (3, 3) or depth.ndim != 2
            or not np.isfinite(rotation).all() or not np.isfinite(offset).all() or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0 or not all(map(math.isfinite, (roll, pitch, height, forward_m, lateral_m)))):
        return unknown

    # Subsample the calibrated depth image; optical depth is along +z, not ray range.
    stride = 3
    z = depth[::stride, ::stride]
    row, col = np.nonzero(np.isfinite(z) & (z >= .08) & (z <= 12.))
    if len(row) == 0:
        return unknown
    z = z[row, col]
    xyz = np.column_stack(((col * stride - K[0, 2]) * z / K[0, 0],
                           (row * stride - K[1, 2]) * z / K[1, 1], z))
    body = xyz @ rotation.T + offset
    # Remove yaw while keeping measured roll and nose-up pitch, so z is
    # gravity-aligned and matches height above this flat platform.
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    horizontal = body @ np.array([[cp, 0., sp], [-sp * sr, cr, cp * sr], [-sp * cr, -sr, cp * cr]])
    elevation = height + horizontal[:, 2]
    xy = horizontal[(elevation > .04) & (elevation < height + .16), :2]

    # Find the time interval in [0,1] where each point is inside the moving
    # rectangle on both axes. Includes oblique translations without accepting
    # the untraversed corners of their bounding box.
    enter = np.zeros(len(xy))
    leave = np.ones(len(xy))
    for axis, (step, half) in enumerate(((forward_m, HALF_LENGTH_M), (lateral_m, HALF_WIDTH_M))):
        centered = xy[:, axis]
        if abs(step) < 1e-9:
            leave[np.abs(centered) > half] = -1.
        else:
            first, last = (centered - half) / step, (centered + half) / step
            enter = np.maximum(enter, np.minimum(first, last))
            leave = np.minimum(leave, np.maximum(first, last))
    hit = np.flatnonzero(enter <= leave)
    if len(hit) == 0:
        return None
    idx = hit[np.argmin(enter[hit])]
    return {"error_code": "VISIBLE_OBSTACLE_IN_STEP",
            "reason": "Measured front depth intersects the swept body footprint. Change heading or waypoint.",
            "obstacle_forward_left_m": np.round(xy[idx], 3).tolist(),
            "clearance_along_command_m": round(float(enter[idx]) * math.hypot(forward_m, lateral_m), 3)}
