"""Compatibility imports for the Go2-specific MuJoCo adapter.

New integrations resolve robots through the registry. Scene construction and
independent acceptance live outside the robot controller.
"""

import mujoco  # Keep the historical Renderer patch point for local callers.

from .robots.go2.backend import MujocoGo2Backend
from .task_evaluation import crossed_passage, wrap_angle

__all__ = ["MujocoGo2Backend", "crossed_passage", "wrap_angle"]
