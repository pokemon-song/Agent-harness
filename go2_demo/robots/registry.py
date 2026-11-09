"""Explicit robot/backend selection. Only Go2 in MuJoCo is implemented."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .profile import RobotProfile

if TYPE_CHECKING:
    from .contracts import RobotBackend


def list_robots() -> tuple[str, ...]:
    return ("go2",)


def get_profile(robot_id: str = "go2", backend_id: str = "mujoco") -> RobotProfile:
    if robot_id != "go2":
        raise ValueError(f"Unknown or unsupported robot_id: {robot_id!r}")
    if backend_id != "mujoco":
        raise ValueError(f"Unknown or unsupported backend_id for go2: {backend_id!r}")
    from .go2.profile import GO2_PROFILE
    return GO2_PROFILE


def create_backend(robot_id: str = "go2", backend_id: str = "mujoco", **kwargs: Any) -> RobotBackend:
    """Import the selected physical adapter only when constructing it.

    Invalid selections are rejected before importing simulator dependencies or
    touching assets. There is deliberately no fallback for unknown robot models.
    """
    get_profile(robot_id, backend_id)
    adapter = import_module("go2_demo.robots.go2.backend")
    return adapter.MujocoGo2Backend(**kwargs)
