"""Exercise actual MuJoCo joint dynamics without a renderer or model API."""

from pathlib import Path
import time

import numpy as np
import pytest

from go2_demo.backend import MujocoGo2Backend
from go2_demo.posture import StaticPostureController, validate_posture
from go2_demo.robots.go2.posture import StaticPostureController as Go2PostureController
from go2_demo.types import ToolCall


@pytest.fixture
def backend(monkeypatch):
    assets = Path(__file__).resolve().parents[1] / "assets"
    if not (assets / "go2_rl_lab/deploy/pre_train/go2/go2_cts_150k.pt").is_file():
        pytest.skip("Pinned robot assets missing; run python scripts/fetch_assets.py")
    # Physics and the pinned gait run normally; these tests never capture RGB.
    monkeypatch.setattr("go2_demo.backend.mujoco.Renderer", lambda *args, **kwargs: None)
    robot = MujocoGo2Backend(assets)
    try:
        yield robot
    finally:
        robot.close()


@pytest.mark.parametrize("height,pitch", [
    (True, 0.), (.3, False), ("0.3", 0.), (None, 0.),
    (float("nan"), 0.), (.3, float("inf")),
    (.239, 0.), (.361, 0.), (.3, .201), (.36, .2),
])
def test_invalid_posture_is_rejected_before_backend_state_is_accessed(height, pitch):
    robot = object.__new__(MujocoGo2Backend)
    result = robot.set_posture(height, pitch)
    assert result["error_code"] == "INVALID_POSTURE"


def test_inverse_kinematics_does_not_reposition_the_live_robot(backend):
    assert StaticPostureController is Go2PostureController
    controller = StaticPostureController(backend.model, backend._qpos_ids, backend._dof_ids, backend._base_id)
    original = backend.data.qpos.copy()
    controller.begin(backend.data)
    angles = controller.target(.28, .15)
    assert angles.shape == (12,)
    assert np.isfinite(angles).all()
    np.testing.assert_array_equal(backend.data.qpos, original)
    assert backend.data.time == pytest.approx(1.)


def test_public_execute_and_idle_cancel_preserve_real_static_hold_and_recovery(backend):
    result = backend.execute(ToolCall("pose", "set_posture", {"body_height_m": .28, "body_pitch_rad": .1}))
    assert result["status"] == "succeeded", result
    reference = backend._posture_reference.copy()
    before = backend.data.time
    cancelled = backend.cancel()
    assert cancelled["status"] == "succeeded" and cancelled["cancelled"] is False
    assert cancelled["reason"] == "no_active_action"
    assert backend.data.time - before == pytest.approx(.6)
    np.testing.assert_array_equal(backend._posture_reference, reference)
    assert backend.control_mode == "posture"
    assert backend.execute(ToolCall("reset", "reset_posture", {}))["status"] == "succeeded"
    assert backend.control_mode == "locomotion"
    assert backend.evaluator.evaluate() == backend.evaluate()


def test_posture_after_real_move_turn_and_stop_accepts_measured_stance_outside_request_bounds(backend, monkeypatch):
    for name, arguments in (("move_relative", {"forward_m": .3, "lateral_m": 0.}),
                            ("turn", {"yaw_rad": .3}), ("stop", {})):
        result = backend.execute(ToolCall(name, name, arguments))
        assert result["status"] == "succeeded", result

    # Invalid user targets still fail before stop, control-mode change or physics.
    before = backend.data.qpos.copy()
    started = backend.data.time
    for height, pitch in ((.361, 0.), (.28, .201), (.36, .2)):
        assert backend.set_posture(height, pitch)["error_code"] == "INVALID_POSTURE"
    np.testing.assert_array_equal(backend.data.qpos, before)
    assert backend.data.time == started and backend.control_mode == "locomotion"

    references = []
    solve = StaticPostureController.target

    def record_reference(controller, height, pitch, *, validate_request=True):
        if not validate_request:
            references.append((height, pitch))
        return solve(controller, height, pitch, validate_request=validate_request)

    monkeypatch.setattr(StaticPostureController, "target", record_reference)
    result = backend.execute(ToolCall("pose-after-walking", "set_posture",
                                      {"body_height_m": .28, "body_pitch_rad": .08}))
    assert result["status"] == "succeeded", result
    # The pinned gait reproduces the original bug: its real starting height is
    # above 0.36m. Never clamp or teleport it to make the request appear valid.
    assert references[0][0] > .36
    assert any(height > .36 for height, _ in references[1:])
    assert abs(result["measured"]["height_error_m"]) <= .02
    assert abs(result["measured"]["pitch_error_rad"]) <= .04
    assert not backend._fallen_ever and not backend._collision_steps
    assert backend.reset_posture()["status"] == "succeeded"


@pytest.mark.parametrize("height,pitch", [
    (float("nan"), 0.), (.3, float("inf")), (True, 0.), (.3, False), (".3", 0.), (None, 0.),
])
def test_internal_posture_references_still_require_finite_numeric_values(height, pitch):
    controller = object.__new__(StaticPostureController)
    with pytest.raises(ValueError, match="finite numbers"):
        controller.target(height, pitch, validate_request=False)


def test_internal_reference_bypassing_request_bounds_still_requires_real_ik_reachability(backend):
    controller = StaticPostureController(backend.model, backend._qpos_ids, backend._dof_ids, backend._base_id)
    controller.begin(backend.data)
    live = backend.data.qpos.copy()
    scratch = controller.data.qpos.copy()
    with pytest.raises(ValueError, match="between 0.24 and 0.36"):
        controller.target(.8, 0.)
    with pytest.raises(ValueError, match="unreachable"):
        controller.target(.8, 0., validate_request=False)
    np.testing.assert_array_equal(backend.data.qpos, live)
    np.testing.assert_array_equal(controller.data.qpos, scratch)


def test_height_pitch_corners_stop_hold_and_explicit_recovery(backend):
    initial_height = backend._base_pose()[0][2]
    for height, pitch in [(.24, .2), (.24, -.2), (.34, -.2), (.34, .2), (.36, 0.)]:
        start_row = len(backend.trajectory)
        result = backend.set_posture(height, pitch)
        assert result["status"] == "succeeded", result
        assert result["control_mode"] == "posture"
        assert abs(result["measured"]["height_error_m"]) <= backend.posture_height_tolerance_m
        assert abs(result["measured"]["pitch_error_rad"]) <= backend.posture_pitch_tolerance_rad
        assert result["measured"]["stable_duration_s"] >= .3
        references = []
        for row in backend.trajectory[start_row:]:
            if row["control_mode"] == "posture":
                reference = row["posture_reference"]
                validate_posture(**reference)
                references.append([reference["body_height_m"], reference["body_pitch_rad"]])
        steps = np.abs(np.diff(references, axis=0))
        assert np.all(steps <= np.array([.06, .15]) * backend.dt * backend.decimation + 1e-10)

    held = backend._posture_reference.copy()
    assert backend.stop()["status"] == "succeeded"
    np.testing.assert_array_equal(backend._posture_reference, held)
    stopped_time = backend.data.time
    assert backend.move_relative(.2, 0.)["error_code"] == "POSTURE_MODE_ACTIVE"
    assert backend.turn(.3)["error_code"] == "POSTURE_MODE_ACTIVE"
    assert backend.data.time == stopped_time

    result = backend.reset_posture()
    assert result["status"] == "succeeded", result
    assert result["control_mode"] == "locomotion"
    assert backend._posture_target is None
    assert abs(result["measured"]["body_height_m"] - initial_height) < .02
    position, _, _ = backend._base_pose()
    move = backend.move_relative(.4, 0.)
    # The original gait's tolerance/deadband still applies. Verify actual travel
    # and controller handoff rather than treating a submitted command as motion.
    assert move["error_code"] not in {"POSTURE_MODE_ACTIVE", "ROBOT_FELL", "COLLISION"}
    assert backend._base_pose()[0][0] - position[0] > .2
    assert not backend._fallen_ever
    assert not backend._collision_steps


def test_posture_timeout_freezes_transition_and_stop_preserves_it(backend):
    result = backend.set_posture(.24, .2, timeout_s=.9)
    assert result["status"] == "timed_out"
    assert result["error_code"] == "POSTURE_TIMEOUT"
    assert backend.control_mode == "posture"
    reference = backend._posture_reference.copy()
    assert reference[0] > .3
    np.testing.assert_array_equal(backend._posture_target, reference)
    assert backend.stop()["status"] == "succeeded"
    np.testing.assert_array_equal(backend._posture_reference, reference)
    assert backend.reset_posture()["status"] == "succeeded"


@pytest.mark.parametrize("deadline", ["wall", "simulation"])
def test_expired_deadline_cannot_start_a_posture_action(backend, deadline):
    if deadline == "wall":
        backend.deadline_monotonic = time.monotonic() - 1.
        expected = "EPISODE_DEADLINE"
    else:
        backend.sim_deadline_s = backend.data.time
        expected = "SIMULATION_DEADLINE"
    started = backend.data.time
    for result in (backend.set_posture(.28, .1), backend.reset_posture()):
        assert result["error_code"] == expected
    assert backend.data.time == started
    assert backend.control_mode == "locomotion"


def test_simulation_deadline_interrupts_transition_and_keeps_last_reference(backend):
    backend.sim_deadline_s = backend.data.time + .85
    result = backend.set_posture(.24, .2)
    assert result["status"] == "failed"
    assert result["error_code"] == "SIMULATION_DEADLINE"
    reference = backend._posture_reference.copy()
    np.testing.assert_array_equal(backend._posture_target, reference)
    backend.stop()
    np.testing.assert_array_equal(backend._posture_reference, reference)


@pytest.mark.parametrize("field,error", [("_fallen_ever", "ROBOT_FELL"), ("_collision_steps", "COLLISION")])
def test_existing_failure_prevents_posture_motion(backend, field, error):
    setattr(backend, field, 1)
    started = backend.data.time
    assert backend.set_posture(.28, 0.)["error_code"] == error
    assert backend.reset_posture()["error_code"] == error
    assert backend.data.time == started


def test_unreachable_initial_stance_does_not_install_a_broken_controller(backend, monkeypatch):
    def unreachable(*args, **kwargs):
        raise ValueError("Foot stance outside reachable joint limits")

    monkeypatch.setattr(StaticPostureController, "target", unreachable)
    result = backend.set_posture(.28, .1)
    assert result["error_code"] == "INFEASIBLE_POSTURE"
    assert backend.control_mode == "locomotion"
    assert backend.stop()["status"] == "succeeded"


def test_unreachable_new_target_preserves_the_current_hold(backend, monkeypatch):
    assert backend.set_posture(.28, .1)["status"] == "succeeded"
    original_target = backend._posture_target.copy()
    original_reference = backend._posture_reference.copy()
    target = backend._posture_controller.target

    def reject_requested(height, pitch, *, validate_request=True):
        if height == .36:
            raise ValueError("Foot stance outside reachable joint limits")
        return target(height, pitch, validate_request=validate_request)

    monkeypatch.setattr(backend._posture_controller, "target", reject_requested)
    started = backend.data.time
    assert backend.set_posture(.36, 0.)["error_code"] == "INFEASIBLE_POSTURE"
    assert backend.data.time == started
    np.testing.assert_array_equal(backend._posture_target, original_target)
    np.testing.assert_array_equal(backend._posture_reference, original_reference)
    assert backend.stop()["status"] == "succeeded"
