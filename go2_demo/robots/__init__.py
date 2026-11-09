"""Robot descriptions and lazy backend selection; importing this package is lightweight."""

from .contracts import RobotBackend
from .profile import (MotionLimits, PostureLimits, RobotDimensions, RobotProfile,
                      SensorSpec, UnsupportedCapabilityError)
from .registry import create_backend, get_profile, list_robots

__all__ = [
    "MotionLimits", "PostureLimits", "RobotBackend", "RobotDimensions", "RobotProfile",
    "SensorSpec", "UnsupportedCapabilityError", "create_backend", "get_profile", "list_robots",
]
