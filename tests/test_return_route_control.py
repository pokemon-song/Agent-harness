from types import SimpleNamespace

import pytest

from go2_demo.return_route_control import _Follower, follow_route


class PublicHarness:
    def __init__(self, collision=False, timeout=False):
        self.backend = SimpleNamespace(deadline_monotonic=None, sim_deadline_s=None, collision_terminated=False)
        self.clock = self.x = self.yaw = 0.
        self.fail_on_contact = collision
        self.partial_move = timeout

    def observation(self):
        return {'sim_time_s': self.clock, 'local_odometry': {'translation_m': [self.x, 0.], 'yaw_rad': self.yaw},
                'robot_state': {'fallen': False}}

    def dispatch(self, call):
        self.clock += 1.
        if self.fail_on_contact:
            self.backend.collision_terminated = True
        if call.name == 'move_relative':
            self.x += call.arguments['forward_m']
            if self.partial_move:
                self.partial_move = False
                return {'status': 'timed_out', 'error_code': 'MOTION_TIMEOUT', 'stop': {'status': 'succeeded'}}
        if call.name == 'turn':
            self.yaw += call.arguments['yaw_rad']
        return {'status': 'succeeded'}


W = [{'id': 'E20', 'x': 0., 'y': 0.}, {'id': 'E05', 'x': .5, 'y': 0.}, {'id': 'E18', 'x': 1., 'y': 0.}]


def test_follows_provided_order_and_checks_current_target_again_after_stop(monkeypatch):
    samples = iter([(.96, 0., .96, 0.), (.96, 0., .96, 0.)])
    monkeypatch.setattr(_Follower, 'measure_green_target', lambda self: next(samples))
    harness = PublicHarness(timeout=True)
    log = []
    d = follow_route(harness, W, on_action=log.append, max_actions=4)
    assert d['termination'] == 'finish' and d['completed_waypoints'] == ['E20', 'E05', 'E18']
    assert d['actions'] == log
    assert [x['tool'] for x in log] == ['move_relative', 'move_relative', 'stop', 'finish']
    assert log[0]['reply']['status'] == 'timed_out'
    assert log[0]['odom_after']['translation_m'] == [.5, 0.]


def test_first_collision_ends_route_without_later_waypoint_or_docking():
    d = follow_route(PublicHarness(collision=True), W)
    assert d['termination'] == 'collision' and d['completed_waypoints'] == ['E20']
    assert len(d['actions']) == 1


def test_budget_halts_visual_search_and_never_finishes_without_target(monkeypatch):
    monkeypatch.setattr(_Follower, 'measure_green_target', lambda self: None)
    d = follow_route(PublicHarness(), W[:1], max_actions=2)
    assert d['termination'] == 'budget_exhausted'
    assert [x['tool'] for x in d['actions']] == ['turn', 'turn']
