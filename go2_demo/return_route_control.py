"""Execute a planner-selected route with public odometry, then dock using current RGB-D."""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from .memory_perception import TaskLabelRGBD, task_targets
from .types import ToolCall


@dataclass(frozen=True)
class RouteControlParameters:
    waypoint_tolerance_m: float = .19
    max_translation_m: float = .53
    max_turn_rad: float = .59
    route_reorient_rad: float = .62
    route_face_tolerance_rad: float = .19
    last_view_face_tolerance_rad: float = .12
    search_turn_rad: float = .55
    visual_surface_standoff_m: float = .96
    visual_distance_tolerance_m: float = .075
    visual_bearing_tolerance_rad: float = .09
    visual_max_translation_m: float = .48
    stopped_visual_distance_tolerance_m: float = .09
    stopped_visual_bearing_tolerance_rad: float = .11
    extra_waypoint_iterations: int = 7
    max_facing_iterations: int = 12
    max_search_turns: int = 14
    max_docking_iterations: int = 16


class _End(Exception):
    def __init__(self, reason, detail):
        self.reason = reason
        self.detail = detail


def _limit(current, proposed):
    return proposed if current is None else min(current, proposed)


def _bounded(value, limit):
    return max(-limit, min(limit, value))


class _Follower:
    target_label = 'B'

    def __init__(self, harness, max_actions, on_action):
        self.harness = harness
        self.max_actions = max_actions
        self.on_action = on_action
        self.params = RouteControlParameters()
        self.sensor = TaskLabelRGBD(task_targets('memory_ab_occluded_v1'))
        self.actions = []
        self.completed = []
        self.phase = None

    def act(self, name, **args):
        self.check_state()
        if len(self.actions) >= self.max_actions:
            raise _End('budget_exhausted', 'native_action_limit')
        before = self.harness.observation()
        call = ToolCall(f'route-control-{len(self.actions):03}', name, args)
        reply = self.harness.dispatch(call)
        after = self.harness.observation()
        entry = {'phase': self.phase, 'call_id': call.call_id, 'tool': name, 'args': args, 'reply': reply,
                 'time_before': before['sim_time_s'], 'time_after': after['sim_time_s'],
                 'odom_before': before['local_odometry'], 'odom_after': after['local_odometry']}
        self.actions.append(entry)
        if self.on_action is not None:
            self.on_action(entry)
        self.check_state()
        if reply['status'] == 'succeeded':
            return reply
        # A timed-out translation may have covered most of its requested distance.
        # Only resume from a verified stop; the caller remeasures its waypoint error.
        if (name == 'move_relative' and reply.get('error_code') in {'MOTION_TIMEOUT', 'DID_NOT_SETTLE_AT_TARGET'}
                and reply.get('stop', {}).get('status') == 'succeeded'):
            return reply
        raise _End('control_error', f"{name}: {reply.get('error_code') or reply['status']}")

    def check_state(self):
        backend = self.harness.backend
        obs = self.harness.observation()
        if backend.collision_terminated:
            raise _End('collision', 'first_scene_contact')
        if obs['robot_state']['fallen']:
            raise _End('fall', 'robot_fell')
        if time.monotonic() >= backend.deadline_monotonic:
            raise _End('budget_exhausted', 'wall_time_limit')
        if obs['sim_time_s'] >= backend.sim_deadline_s:
            raise _End('budget_exhausted', 'simulation_time_limit')

    def pose(self):
        odom = self.harness.observation()['local_odometry']
        return *odom['translation_m'], odom['yaw_rad']

    def face(self, required, tolerance):
        for _ in range(self.params.max_facing_iterations):
            delta = math.remainder(required - self.pose()[2], math.tau)
            if abs(delta) < tolerance:
                return
            self.act('turn', yaw_rad=_bounded(delta, self.params.max_turn_rad))
        raise _End('control_error', 'heading_did_not_converge')

    def go_to(self, waypoint):
        self.phase = waypoint['id']
        x, y, _ = self.pose()
        initial_distance = math.hypot(waypoint['x'] - x, waypoint['y'] - y)
        attempts = self.params.extra_waypoint_iterations + math.ceil(initial_distance / self.params.max_translation_m)
        for _ in range(attempts):
            x, y, yaw = self.pose()
            dx, dy = waypoint['x'] - x, waypoint['y'] - y
            distance = math.hypot(dx, dy)
            if distance < self.params.waypoint_tolerance_m:
                self.completed.append(waypoint['id'])
                return
            bearing = math.atan2(dy, dx)
            if abs(math.remainder(bearing - yaw, math.tau)) > self.params.route_reorient_rad:
                self.face(bearing, self.params.route_face_tolerance_rad)
                continue
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            left = -dx * math.sin(yaw) + dy * math.cos(yaw)
            scale = min(self.params.max_translation_m, distance) / distance
            self.act('move_relative', forward_m=forward * scale, lateral_m=left * scale)
        raise _End('control_error', f"waypoint_did_not_converge: {waypoint['id']}")

    def measure_docking_target(self):
        frame = self.harness.frame
        obs = self.harness.observation()
        view, masks = self.sensor.inspect(obs, frame.rgb, frame.depth_m, frame.intrinsics)
        target = 'task:' + self.target_label
        if target not in masks:
            return None
        depth = frame.depth_m
        pixels = np.argwhere(masks[target] & np.isfinite(depth) & (depth >= .08) & (depth <= 12.))
        middle = np.median(pixels, axis=0)
        v, u = pixels[np.argmin(np.linalg.norm(pixels - middle, axis=1))]
        point = self.act('track_point', image_id=obs['image_id'], u=int(u), v=int(v))['point']
        association = self.sensor.associate(point, view, masks, depth, frame.intrinsics)
        if not association or association['task_label'] != self.target_label:
            raise _End('control_error', f'current_{self.target_label}_region_and_tracked_point_disagree')
        forward, left = point['relative_xy_m']
        return forward, left, math.hypot(forward, left), math.atan2(left, forward)

    def dock(self, last_yaw):
        self.phase = 'return_visual_docking' if self.target_label == 'B' else 'outbound_visual_docking'
        p = self.params
        if last_yaw is not None:
            self.face(last_yaw, p.last_view_face_tolerance_rad)
        sample = self.measure_docking_target()
        for _ in range(p.max_search_turns):
            if sample is not None:
                break
            self.act('turn', yaw_rad=p.search_turn_rad)
            sample = self.measure_docking_target()
        for _ in range(p.max_docking_iterations):
            if sample is None:
                raise _End('control_error', f'target_{self.target_label}_not_visible')
            forward, left, distance, heading = sample
            if distance <= 1e-6:
                raise _End('control_error', 'invalid_target_range')
            error = distance - p.visual_surface_standoff_m
            if abs(heading) > p.visual_bearing_tolerance_rad:
                self.act('turn', yaw_rad=_bounded(heading, p.max_turn_rad))
            elif abs(error) > p.visual_distance_tolerance_m:
                delta = _bounded(error, p.visual_max_translation_m)
                self.act('move_relative', forward_m=delta * forward / distance, lateral_m=delta * left / distance)
            else:
                self.act('stop')
                sample = self.measure_docking_target()
                if (sample is not None and
                        abs(sample[2] - p.visual_surface_standoff_m) <= p.stopped_visual_distance_tolerance_m and
                        abs(sample[3]) < p.stopped_visual_bearing_tolerance_rad):
                    self.act('finish')
                    return
                continue
            sample = self.measure_docking_target()
        raise _End('control_error', 'visual_docking_did_not_converge')


def follow_route(harness, waypoints, *, max_actions=115, max_sim_seconds=470., max_wall_seconds=1117., on_action=None):
    """Follow every supplied public waypoint in order, then dock on current visual evidence.

    Each waypoint supplies ``id``, ``x``, ``y`` and optionally its recorded ``yaw``.
    Only the last waypoint's yaw is used to recover its published viewing direction.
    This function does not read documents, choose waypoints, or score its own run.
    The caller constructs the same first-contact backend used for independent scoring.
    """
    if not isinstance(waypoints, (tuple, list)) or not waypoints:
        raise ValueError('Expected a nonempty list of public waypoints')
    for w in waypoints:
        if not isinstance(w, dict) or not isinstance(w.get('id'), str) or not w['id']:
            raise ValueError('Every waypoint must carry its public evidence ID')
        keys = ['x', 'y', 'yaw'] if 'yaw' in w else ['x', 'y']
        if any(type(w.get(k)) not in (int, float) or not math.isfinite(w[k]) for k in keys):
            raise ValueError('Expected finite recorded waypoint coordinates and heading')
    if (type(max_actions) is not int or not 0 <= max_actions <= 115 or
            not math.isfinite(max_sim_seconds) or not 0 <= max_sim_seconds <= 470 or
            not math.isfinite(max_wall_seconds) or not 0 <= max_wall_seconds <= 1117):
        raise ValueError('Expected budgets within the frozen return envelope')
    follower = _Follower(harness, max_actions, on_action)
    start_wall = time.monotonic()
    start_sim = harness.observation()['sim_time_s']
    backend = harness.backend
    backend.deadline_monotonic = _limit(backend.deadline_monotonic, start_wall + max_wall_seconds)
    backend.sim_deadline_s = _limit(backend.sim_deadline_s, start_sim + max_sim_seconds)
    termination, error = 'finish', None
    try:
        follower.check_state()
        for w in waypoints:
            follower.go_to(w)
        follower.dock(waypoints[-1].get('yaw'))
    except _End as exc:
        termination, error = exc.reason, exc.detail
    return {'termination': termination, 'error': error, 'actions': follower.actions,
            'completed_waypoints': follower.completed, 'parameters': asdict(follower.params),
            'limits': {'max_native_actions': max_actions, 'max_sim_seconds': max_sim_seconds,
                       'max_wall_seconds': max_wall_seconds}, 'sim_seconds': harness.observation()['sim_time_s'] - start_sim,
            'wall_seconds': time.monotonic() - start_wall}
