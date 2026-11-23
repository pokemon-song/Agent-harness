"""Short-lived exploration feedback from public camera depth and episode odometry."""
from collections import deque
import math

import numpy as np

from .short_step_clearance import HALF_WIDTH_M, check_clearance


def _public_cloud(frame, obs):
    """Return forward/left/gravity-up points using only calibrated, public inputs."""
    try:
        k = np.asarray(frame.intrinsics, dtype=float)
        depth = np.asarray(frame.depth_m, dtype=float)
        cal, state = obs["sensor_calibration"], obs["robot_state"]
        rotation = np.asarray(cal["camera_optical_to_body_rotation"], dtype=float)
        offset = np.asarray(cal["camera_position_in_body_m"], dtype=float)
        roll, pitch, height = (float(state[key]) for key in ("body_roll_rad", "body_pitch_rad", "body_height_m"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if (depth.ndim != 2 or k.shape != (3, 3) or rotation.shape != (3, 3) or offset.shape != (3,)
            or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.isfinite(rotation).all() or not np.isfinite(offset).all()
            or not all(map(math.isfinite, (roll, pitch, height))) or height <= 0):
        return None
    stride = 3
    depth = depth[::stride, ::stride]
    rows, cols = np.nonzero(np.isfinite(depth) & (depth >= .08) & (depth <= 12.))
    if not len(rows):
        return None
    z = depth[rows, cols]
    camera = np.column_stack(((cols*stride-k[0, 2])*z/k[0, 0], (rows*stride-k[1, 2])*z/k[1, 1], z))
    body = camera @ rotation.T + offset
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    gravity = np.array([[cp, 0., sp], [-sp*sr, cr, cp*sr], [-sp*cr, -sr, cp*cr]])
    horizontal = body @ gravity
    rays = horizontal[:, :2] - (offset @ gravity)[:2]
    landmarks = np.zeros(len(camera), dtype=bool)
    for candidate in obs.get("target_candidates", []):
        point = np.asarray(candidate.get("camera_xyz_m", []), dtype=float)
        if point.shape == (3,) and np.isfinite(point).all():
            # Suppress a small neighbourhood around the measured task landmark
            # only for edge selection. All depth still participates in collision checks.
            landmarks |= np.sum((camera - point)**2, axis=1) < .36**2
    horizontal[:, 2] += height
    return horizontal, rays, landmarks


def _depth_edges(cloud, rays, floor_mask, obstacle_mask):
    """An observed near vertical surface next to observed, substantially farther floor."""
    bearings = np.arctan2(rays[:, 1], rays[:, 0])
    bins = np.floor(bearings / .09).astype(int)
    ranges = np.hypot(rays[:, 0], rays[:, 1])
    near, far = {}, {}
    for index in np.unique(bins):
        band = bins == index
        segment = np.flatnonzero(band & obstacle_mask & (ranges < 4.5))
        if len(segment) >= 8:
            radius = float(np.percentile(ranges[segment], 20))
            near[index] = (radius, np.median(cloud[segment[ranges[segment] <= radius + .18], :2], axis=0))
        floor_ranges = ranges[band & floor_mask]
        if len(floor_ranges) >= 8:
            far[index] = float(np.percentile(floor_ranges, 95))
    edges = []
    for index, (radius, xy) in near.items():
        for side in (-1, 1):
            adjacent = index + side
            far_range = far.get(adjacent, 0.)
            if far_range < radius + .90 or near.get(adjacent, (math.inf,))[0] <= radius + .70:
                continue
            # Aim slightly beyond the actual edge, on the side where floor was measured.
            direction = (adjacent + .5) * .09
            tangent = np.array([-math.sin(direction), math.cos(direction)])
            look_xy = xy + side * .26 * tangent
            edges.append((direction, float(look_xy[0]), float(look_xy[1]), round(far_range - radius, 2)))
    return edges


def _support_strip(floor, forward, left, distance):
    """Look for floor across the destination and visible approach; record the front-camera blind strip."""
    # Sample floor in three lateral bands. A thin slice of floor behind an occluder
    # or an isolated return at the waypoint does not establish destination support.
    leading = floor[:, 0]*forward + floor[:, 1]*left
    lateral = -floor[:, 0]*left + floor[:, 1]*forward
    stations = np.arange(distance + .24, .35, -.12)
    support_from = None
    gap = False
    for station in stations:
        strip = lateral[np.abs(leading - station) <= .075]
        if all(np.count_nonzero(np.abs(strip - side) <= .075) >= 2 for side in (-.18, 0., .18)):
            if gap:
                return None
            support_from = float(station)
        else:
            gap = True
    # Require the entire destination footprint plus some approach. The camera
    # cannot see directly under its nose. Do not present that strip as observed.
    if support_from is None or support_from > distance - .23 or support_from > 1.22:
        return None
    return round(support_from, 2)


def _angle(a, b):
    return abs(math.remainder(a - b, math.tau))


class StageExploration:
    """One instance per stage. Only same-stage visits and failed waypoints influence ranking."""
    def __init__(self):
        self.windows = deque(maxlen=12)
        self.visits = deque(maxlen=48)
        self.blocked = deque(maxlen=6)

    def mark_blocked(self, waypoint):
        self.blocked.append((waypoint["x"], waypoint["y"]))

    def _scan(self, xy, yaw):
        nearby_yaws = [a for x, y, a in self.visits if math.dist(xy, (x, y)) < .48]
        return max((.58, -.58), key=lambda delta: min((_angle(yaw + delta, a) for a in nearby_yaws), default=math.pi))

    def observe(self, frame, observation, stage):
        odom = observation["local_odometry"]
        x, y = odom["translation_m"]
        yaw = odom["yaw_rad"]
        visible = any(c["task_label"] == stage for c in observation.get("target_candidates", []))
        self.windows.append((x, y, yaw, visible))
        if not self.visits or math.dist((x, y), self.visits[-1][:2]) >= .16 or _angle(yaw, self.visits[-1][2]) >= .15:
            self.visits.append((x, y, yaw))
        window = list(self.windows)
        stalled = (len(window) == self.windows.maxlen and not any(seen for *_, seen in window)
                   and any(_angle(yaw, prev[2]) <= .15 for prev in window[:-1])
                   and all(math.dist(a[:2], b[:2]) <= .30 for i, a in enumerate(window) for b in window[i+1:]))
        result = {"stalled": stalled, "target_visible": visible, "viewpoints": [],
                  "scan_yaw_rad": self._scan((x, y), yaw), "source": "current_depth_and_stage_local_odometry"}
        if visible:
            return result
        sample = _public_cloud(frame, observation)
        if sample is None:
            return result
        cloud, rays, landmarks = sample
        floor_mask = np.abs(cloud[:, 2]) <= .035
        floor = cloud[floor_mask, :2]
        if not len(floor):
            return result
        obstacle_mask = (cloud[:, 2] > .04) & (cloud[:, 2] < observation["robot_state"]["body_height_m"] + .16)
        obstacle = cloud[obstacle_mask, :2]
        edges = _depth_edges(cloud, rays, floor_mask, obstacle_mask & ~landmarks)
        result["observed_depth_edges_rad"] = sorted({round(e[0], 3) for e in edges})
        previous = next((p for p in reversed(self.visits) if math.dist((x, y), p[:2]) > .40), None)
        choices = []
        for offset in (-.52, -.28, 0., .28, .52):
            forward, left = math.cos(offset), math.sin(offset)
            floor_along = floor[:, 0]*forward + floor[:, 1]*left
            floor_lane = floor_along[np.abs(-floor[:, 0]*left + floor[:, 1]*forward) < HALF_WIDTH_M]
            obstacle_along = obstacle[:, 0]*forward + obstacle[:, 1]*left
            obstacle_lane = obstacle_along[(np.abs(-obstacle[:, 0]*left + obstacle[:, 1]*forward) <= HALF_WIDTH_M + .10)
                                          & (obstacle_along > .30)]
            # Estimate only the length of *observed* floor ahead. Missing depth
            # is never interpreted as infinite free space.
            observed_reach = float(np.percentile(floor_lane, 93)) if len(floor_lane) else 0.
            if len(obstacle_lane):
                observed_reach = min(observed_reach, float(np.min(obstacle_lane)))
            for distance in (1.55, 1.25, 1.05):
                blind = _support_strip(floor, forward, left, distance)
                if blind is None or check_clearance(frame, observation, distance*forward, distance*left) is not None:
                    continue
                heading = yaw + offset
                px, py = x + distance*math.cos(heading), y + distance*math.sin(heading)
                if any(math.dist((px, py), blocked) < .31 for blocked in self.blocked):
                    continue
                nearest = min(math.dist((px, py), visit[:2]) for visit in self.visits)
                # Prefer an unexplored stance with floor continuing into view.
                # Recent measured travel breaks close ties toward the same branch.
                continuation = (math.cos(heading - math.atan2(y-previous[1], x-previous[0])) if previous else 0.)
                score = min(nearest, 1.8) + .39*min(observed_reach, 3.5) + .25*continuation
                point = {"x": round(px, 3), "y": round(py, 3)}
                cue = None
                if edges:
                    angle, ex, ey, gain = min(edges, key=lambda edge: abs(edge[0] - offset))
                    proximity = max(0., 1. - abs(angle-offset)/.75)
                    if proximity and math.hypot(ex-forward*distance, ey-left*distance) > .45:
                        look = yaw + math.atan2(ey-left*distance, ex-forward*distance)
                        point["yaw"] = round(math.remainder(look, math.tau), 3)
                        cue = {"open_side_bearing_rad": round(angle, 3), "measured_depth_jump_m": gain}
                        score += 1.75 * proximity
                choices.append((score, {"waypoint": point, "depth_edge": cue,
                    "bearing_offset_rad": offset, "suggested_heading_rad": round(math.remainder(heading, math.tau), 3),
                    "distance_m": distance, "floor_support_from_m": blind,
                    "observed_floor_reach_m": round(observed_reach, 2),
                    "nearest_stage_visit_m": round(nearest, 2)}))
                break
        for _, item in sorted(choices, key=lambda row: row[0], reverse=True):
            # Show different directions instead of several nearly identical points.
            if all(abs(item["bearing_offset_rad"] - other["bearing_offset_rad"]) >= .27
                   for other in result["viewpoints"]):
                result["viewpoints"].append(item)
            if len(result["viewpoints"]) == 3:
                break
        return result
