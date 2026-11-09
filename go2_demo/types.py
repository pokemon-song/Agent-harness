"""Internal sensor data and model decisions. SensorFrame is never sent as-is to a model."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SensorFrame:
    frame_id: str
    sim_time_s: float | None
    rgb: np.ndarray
    depth_m: np.ndarray | None
    intrinsics: np.ndarray | None
    camera_to_world: np.ndarray | None
    base_xy_yaw: np.ndarray | None
    robot_state: dict[str, Any]
    timestamp_s: float | None = None
    time_source: str = "unspecified"
    time_mode: str = "unspecified"
    localization_available: bool | None = None
    localization_source: str = "unspecified"
    localization_quality: dict[str, Any] = field(default_factory=dict)
    depth_available: bool | None = None
    depth_source: str = "unspecified"
    depth_quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Decision:
    calls: list[ToolCall]
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    model: str = "unknown"
    raw_text: str = ""
    reasoning_content: str | None = field(default=None, repr=False)
    assistant_content: str | None = field(default=None, repr=False)
