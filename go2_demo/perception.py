"""RGB-D point tracking. No task labels, target poses or simulator object IDs enter here."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .types import SensorFrame


class PerceptionError(ValueError):
    pass


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise PerceptionError(f"Missing or invalid {name}") from None
    if result.shape != shape or not np.isfinite(result).all():
        raise PerceptionError(f"Missing or invalid {name}")
    return result


def localization_pose(frame: SensorFrame) -> np.ndarray:
    """Return supplied localization, never substitute a zero pose."""
    quality = frame.localization_quality if isinstance(frame.localization_quality, dict) else {}
    if frame.localization_available is False or quality.get("valid") is False or quality.get("stale") is True:
        raise PerceptionError("Localization is unavailable or stale")
    return _array(frame.base_xy_yaw, (3,), "localization pose")


def depth_frame_error(frame: SensorFrame) -> str | None:
    """Check the depth channel independently of localization availability."""
    quality = frame.depth_quality if isinstance(frame.depth_quality, dict) else {}
    if frame.depth_available is False or quality.get("valid") is False or quality.get("stale") is True:
        return "Depth is unavailable or stale"
    if not isinstance(frame.depth_m, np.ndarray) or frame.depth_m.ndim != 2 or not frame.depth_m.size:
        return "Depth is unavailable"
    if frame.depth_m.shape != frame.rgb.shape[:2]:
        return "Depth does not match the current RGB image"
    return None


def metric_frame_error(frame: SensorFrame) -> str | None:
    """Availability check without sampling or inventing a metric position."""
    problem = depth_frame_error(frame)
    if problem:
        return problem
    try:
        _array(frame.intrinsics, (3, 3), "camera intrinsics")
        _array(frame.camera_to_world, (4, 4), "camera pose")
        localization_pose(frame)
    except PerceptionError as exc:
        return str(exc)
    return None


def backproject(frame: SensorFrame, u: float, v: float, *,
                min_depth_m: float = .08, max_depth_m: float = 12.) -> np.ndarray:
    """Use optical-z depth; return a world point internally (never exposed to the model)."""
    problem = metric_frame_error(frame)
    if problem:
        raise PerceptionError(problem)
    h, w = frame.depth_m.shape
    try:
        valid_pixel = (not isinstance(u, (bool, np.bool_)) and not isinstance(v, (bool, np.bool_))
                       and math.isfinite(u) and math.isfinite(v) and 0 <= u < w and 0 <= v < h)
    except (TypeError, ValueError, OverflowError):
        valid_pixel = False
    if not valid_pixel:
        raise PerceptionError("Pixel outside the current image")
    x, y = int(round(u)), int(round(v))
    x, y = min(x, w - 1), min(y, h - 1)
    try:
        depth = float(frame.depth_m[y, x])
    except (TypeError, ValueError, OverflowError):
        raise PerceptionError("No usable depth at the selected pixel") from None
    if not math.isfinite(depth) or not min_depth_m <= depth <= max_depth_m:
        raise PerceptionError("No usable depth at the selected pixel")
    intrinsics = _array(frame.intrinsics, (3, 3), "camera intrinsics")
    camera_to_world = _array(frame.camera_to_world, (4, 4), "camera pose")
    try:
        ray = np.linalg.solve(intrinsics, np.array([u, v, 1.0]))
    except (np.linalg.LinAlgError, TypeError, ValueError):
        raise PerceptionError("Invalid camera intrinsics") from None
    point = camera_to_world @ np.r_[ray * depth, 1.0]
    if not np.isfinite(point).all():
        raise PerceptionError("Invalid camera calibration")
    return point[:3]


def relative_xy(frame: SensorFrame, world: np.ndarray) -> np.ndarray:
    x, y, yaw = localization_pose(frame)
    world = _array(world, (3,), "tracked world point")
    dx, dy = world[:2] - [x, y]
    return np.array([math.cos(yaw) * dx + math.sin(yaw) * dy,
                     -math.sin(yaw) * dx + math.cos(yaw) * dy])


@dataclass
class TrackedPoint:
    point_id: str
    pixel: np.ndarray
    world: np.ndarray
    valid: bool = True
    confidence: float = 1.0
    reason: str | None = None


class PointTracker:
    """Sparse LK tracking between decision frames, with explicit loss instead of oracle recovery."""

    def __init__(self, max_points: int = 8, *, min_depth_m: float = .08, max_depth_m: float = 12.):
        self.max_points = max_points
        self.min_depth_m, self.max_depth_m = min_depth_m, max_depth_m
        self.frame: SensorFrame | None = None
        self.gray: np.ndarray | None = None
        self.points: dict[str, TrackedPoint] = {}
        self.counter = 0

    def update(self, frame: SensorFrame) -> None:
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        active = [p for p in self.points.values() if p.valid]
        problem = metric_frame_error(frame)
        if problem:
            for point in active:
                self._lose(point, problem)
            self.frame, self.gray = frame, gray
            return
        if self.frame is not None and self.gray is not None and active:
            if frame.frame_id == self.frame.frame_id:
                return
            previous = np.array([p.pixel for p in active], np.float32).reshape(-1, 1, 2)
            params = dict(winSize=(31, 31), maxLevel=3,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            try:
                current, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, previous, None, **params)
                if current is None:
                    raise ValueError("No flow")
                backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, current, None, **params)
                if backward is None:
                    raise ValueError("No reverse flow")
                for i, point in enumerate(active):
                    if not status[i, 0] or not back_status[i, 0]:
                        self._lose(point, "optical_flow_lost")
                        continue
                    error = float(np.linalg.norm(previous[i] - backward[i]))
                    if not math.isfinite(error) or error > 1.5:
                        self._lose(point, "inconsistent_flow")
                        continue
                    uv = current[i, 0]
                    try:
                        world = backproject(frame, float(uv[0]), float(uv[1]),
                                            min_depth_m=self.min_depth_m, max_depth_m=self.max_depth_m)
                    except PerceptionError:
                        self._lose(point, "invalid_depth_or_out_of_view")
                        continue
                    # This demo tracks static scene points. Reject depth jumps / occluders.
                    if float(np.linalg.norm(world - point.world)) > 0.25:
                        self._lose(point, "geometry_jump")
                        continue
                    point.pixel = uv.copy()
                    point.world = world
                    point.confidence = max(0.0, 1.0 - error / 1.5)
            except (ValueError, cv2.error):
                for point in active:
                    self._lose(point, "optical_flow_lost")
        self.frame, self.gray = frame, gray

    @staticmethod
    def _lose(point: TrackedPoint, reason: str) -> None:
        point.valid, point.confidence, point.reason = False, 0.0, reason

    def add(self, image_id: str, u: float, v: float) -> dict:
        if self.frame is None or self.frame.frame_id != image_id:
            raise PerceptionError("STALE_IMAGE: select a pixel in the latest image_id")
        world = backproject(self.frame, u, v, min_depth_m=self.min_depth_m, max_depth_m=self.max_depth_m)
        if len(self.points) >= self.max_points:
            lost = next((k for k, p in self.points.items() if not p.valid), None)
            if lost is None:
                raise PerceptionError("Maximum tracked points reached")
            del self.points[lost]
        self.counter += 1
        point = TrackedPoint(f"point-{self.counter:03d}", np.array([u, v], np.float32), world)
        self.points[point.point_id] = point
        return self.describe(point)

    def describe(self, point: TrackedPoint) -> dict:
        assert self.frame is not None
        try:
            xy = relative_xy(self.frame, point.world) if point.valid else None
        except PerceptionError as exc:
            self._lose(point, str(exc))
            xy = None
        return {
            "point_id": point.point_id,
            "image_id": self.frame.frame_id,
            "timestamp_s": self.frame.timestamp_s if self.frame.timestamp_s is not None else self.frame.sim_time_s,
            "valid": point.valid,
            "pixel_uv": [round(float(x), 2) for x in point.pixel] if point.valid else None,
            "relative_xy_m": [round(float(x), 4) for x in xy] if xy is not None else None,
            "range_m": round(float(np.linalg.norm(xy)), 4) if xy is not None else None,
            "bearing_rad": round(math.atan2(float(xy[1]), float(xy[0])), 4) if xy is not None else None,
            "confidence": round(point.confidence, 3),
            "loss_reason": point.reason,
            "source": "rgbd_optical_flow",
        }

    def geometry(self) -> list[dict]:
        return [self.describe(p) for p in self.points.values()]


