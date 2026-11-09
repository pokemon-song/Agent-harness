"""Static Go2 stance inverse kinematics; episode state is never repositioned.

The caller must ramp the requested pose and track the resulting joint targets
with torque control. This controller does not provide a crawling gait.
"""

from __future__ import annotations

import math

import numpy as np


MIN_HEIGHT_M = 0.24
MAX_HEIGHT_M = 0.36
MAX_PITCH_RAD = 0.2
# Limit the raised end of the body so the stance does not straighten a leg
# against its knee joint limit. Exact reachability is also checked by IK.
MAX_END_HEIGHT_M = 0.38
HALF_BODY_LENGTH_M = 0.1934


def validate_posture(body_height_m: float, body_pitch_rad: float) -> None:
    """Validate absolute body height and pitch, with nose-up-positive radians."""
    for name, value in (("body_height_m", body_height_m), ("body_pitch_rad", body_pitch_rad)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if not MIN_HEIGHT_M <= body_height_m <= MAX_HEIGHT_M:
        raise ValueError(f"body_height_m must be between {MIN_HEIGHT_M} and {MAX_HEIGHT_M} m")
    if abs(body_pitch_rad) > MAX_PITCH_RAD:
        raise ValueError(f"Absolute body_pitch_rad must not exceed {MAX_PITCH_RAD} rad")
    # The conservative linear bound keeps the feasible set convex, so a
    # synchronized interpolation between accepted poses stays in the domain.
    if body_height_m + HALF_BODY_LENGTH_M * abs(body_pitch_rad) > MAX_END_HEIGHT_M + 1e-12:
        raise ValueError("Height and pitch together exceed the static stance reach limit")


