"""Compatibility imports for Go2 static posture limits and inverse kinematics."""

from .robots.go2.posture_limits import (
    HALF_BODY_LENGTH_M, MAX_END_HEIGHT_M, MAX_HEIGHT_M, MAX_PITCH_RAD,
    MIN_HEIGHT_M, validate_posture,
)


def __getattr__(name):
    if name == "StaticPostureController":
        from .robots.go2.posture import StaticPostureController
        return StaticPostureController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MIN_HEIGHT_M", "MAX_HEIGHT_M", "MAX_PITCH_RAD", "MAX_END_HEIGHT_M",
           "HALF_BODY_LENGTH_M", "validate_posture", "StaticPostureController"]
