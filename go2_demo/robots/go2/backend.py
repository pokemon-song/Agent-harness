"""Go2 MuJoCo adapter with the unchanged pretrained policy and PD controller.

Only resets write live qpos. Actions remain synchronous torque-driven physics;
policy/trajectory frequency is 50 Hz and physics/PD frequency is 500 Hz.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import torch

from ...environments.mujoco import MujocoEnvironment, add_letter
from ...scenarios import bind_robot
from ...task_evaluation import TaskEvaluator, TaskEvaluationState, wrap_angle
from ...types import SensorFrame, ToolCall
from .posture import StaticPostureController
from .posture_limits import validate_posture
from .profile import GO2_PROFILE


class MujocoGo2Backend:
    """Bounded locomotion and a separate, stationary joint posture controller."""

    profile = GO2_PROFILE
    posture_control_supported = True
    moving_posture_control_supported = False
    posture_height_tolerance_m = 0.02
    posture_pitch_tolerance_rad = 0.04
    width = 640
    height = 360
    dt = GO2_PROFILE.physics_dt_s
    decimation = GO2_PROFILE.control_decimation
    joint_names = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]

    @property
    def sensor_calibration(self) -> dict:
        """Rigid sensor mounting, with no world pose or task geometry."""
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.model.cam_quat[self._camera_id])
        optical_to_body = rotation.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        return {"camera_frame": "front_rgbd_optical", "body_frame": "base",
                "camera_position_in_body_m": self.model.cam_pos[self._camera_id].tolist(),
                "camera_optical_to_body_rotation": optical_to_body.tolist(),
                "optical_axes": "x right, y down, z forward",
                "body_axes": "x forward, y left, z up"}

    def __init__(self, asset_dir: Path, seed: int = 0, viewer: bool = False, scenario=None):
        self.asset_dir = Path(asset_dir).resolve()
        scenario = bind_robot(scenario, self.profile) if scenario is not None else None
        self.scenario = scenario
        if scenario is not None and not scenario.supported:
            raise ValueError(f"UNSUPPORTED_CAPABILITY: {scenario.unsupported_reason}")
        self.task_instruction = (scenario.instruction if scenario is not None else
                                 "找到红色标记，移动到距离标记约 1 米的位置，朝向标记并停止，然后调用 finish。距离按机身参考点到标记的水平距离理解。")
        self.deadline_monotonic: float | None = None
        self.sim_deadline_s: float | None = None
        self._viewer = None
        self._renderer = None
        self._recording = None
        self._stopping = False
        self._command_lock = threading.Lock()
        self._frame_counter = 0
        self.trajectory: list[dict] = []
        self._fallen_ever = False
        self._collision_steps = 0
        self._collided_geom_ids: set[int] = set()
        self.default_angles = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self._posture_controller: StaticPostureController | None = None
        self._posture_reference: np.ndarray | None = None
        self._posture_target: np.ndarray | None = None
        self._verify_assets()
        self.environment = MujocoEnvironment(scenario, seed)
        self._target_xy = self.environment.target_xy
        self._environment_geom_names = []
        self.evaluator = TaskEvaluator(scenario, self._target_xy, self._evaluation_state)
        self.model = self._make_model()
        self.data = mujoco.MjData(self.model)
        self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_rgbd")
        self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._environment_geom_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                                      for name in self._environment_geom_names}
        self._qpos_ids = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.joint_names])
        self._dof_ids = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.joint_names])
        policy_bytes = (self.asset_dir / "go2_rl_lab/deploy/pre_train/go2/go2_cts_150k.pt").read_bytes()
        self.policy = torch.jit.load(io.BytesIO(policy_bytes), map_location="cpu")
        self.policy.eval()
        # This is an episode reset, not an action implementation.
        self.data.qpos[:3] = [0, 0, 0.34]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self._qpos_ids] = self.default_angles
        mujoco.mj_forward(self.model, self.data)
        self._advance_zero(1.0, record=False)
        if self._fallen():
            raise RuntimeError("Go2 fell while initializing the pretrained controller")
        position, _, rotation = self._base_pose()
        self._nominal_posture = np.array([position[2], self._pitch(rotation)])
        self._fallen_ever = False
        self._collision_steps = 0
        self.trajectory.clear()
        self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        if viewer:
            from mujoco import viewer as viewer_module
            self._viewer = viewer_module.launch_passive(self.model, self.data)
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._viewer.cam.trackbodyid = self._base_id
            self._viewer.cam.distance = 3.5
            self._viewer.cam.azimuth = 135
            self._viewer.cam.elevation = -25

    def start_recording(self, directory):
        """Opt-in private state capture; no camera rendering in the action loop."""
        from ...recording import SimulationRecording
        if self._recording is not None:
            raise RuntimeError("A recording has already been started")
        self._recording = SimulationRecording(self.model, self.data, directory,
                                               control_hz=1 / (self.dt * self.decimation))
        return self._recording

    def _record_state(self):
        recording = getattr(self, "_recording", None)
        if recording is not None:
            recording.sample(self.model, self.data)

    def _verify_assets(self) -> None:
        manifest_path = Path(__file__).resolve().parents[3] / "assets.lock.json"
        self.asset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in self.asset_manifest["files"]:
            path = (self.asset_dir / entry["path"]).resolve()
            if not path.is_relative_to(self.asset_dir) or not path.is_file():
                raise FileNotFoundError("Go2 assets are missing; run: python scripts/fetch_assets.py")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Go2 asset checksum mismatch: {entry['path']}; run scripts/fetch_assets.py")

    def _base_pose(self) -> tuple[np.ndarray, float, np.ndarray]:
        position = self.data.xpos[self._base_id].copy()
        rotation = self.data.xmat[self._base_id].reshape(3, 3).copy()
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        return position, yaw, rotation

    def _fallen(self) -> bool:
        position, _, rotation = self._base_pose()
        # This state enters API observations and must remain a native JSON bool.
        return bool(not np.isfinite(self.data.qpos).all() or position[2] < 0.16 or rotation[2, 2] < 0.55)

    @staticmethod
    def _pitch(rotation: np.ndarray) -> float:
        return float(math.asin(np.clip(rotation[2, 0], -1, 1)))

    @property
    def control_mode(self) -> str:
        return "posture" if self._posture_controller is not None else "locomotion"

    def _step_control(self, command: np.ndarray, *, record: bool = True) -> None:
        started = time.monotonic()
        if record:
            self._record_state()
        if self._posture_controller is not None:
            target = self._posture_controller.target(*self._posture_reference, validate_request=False)
            kp, kd = 120.0, 2.0
        else:
            _, _, rotation = self._base_pose()
            observation = np.concatenate((
                self.data.qvel[3:6] * 0.25,
                rotation.T @ np.array([0.0, 0.0, -1.0]),
                command * np.array([2.0, 2.0, 0.25]),
                self.data.qpos[self._qpos_ids] - self.default_angles,
                self.data.qvel[self._dof_ids] * 0.05,
                self.last_action,
            )).astype(np.float32)
            with torch.inference_mode():
                action = self.policy(torch.from_numpy(observation).unsqueeze(0))
                if isinstance(action, tuple):
                    action = action[0]
                self.last_action = action.detach().cpu().numpy().reshape(12)
            target = self.default_angles + 0.25 * self.last_action
            kp, kd = 20.0, 0.5
        for _ in range(self.decimation):
            self.data.ctrl[:] = kp * (target - self.data.qpos[self._qpos_ids]) - kd * self.data.qvel[self._dof_ids]
            mujoco.mj_step(self.model, self.data)
            if record:
                touched = False
                for contact in self.data.contact:
                    for geom_id in (int(contact.geom1), int(contact.geom2)):
                        if geom_id in self._environment_geom_ids:
                            touched = True
                            self._collided_geom_ids.add(geom_id)
                self._collision_steps += int(touched)
        if record:
            self._fallen_ever |= self._fallen()
            position, yaw, rotation = self._base_pose()
            self.trajectory.append({"sim_time_s": float(self.data.time), "base_xyz_m": position.tolist(), "yaw_rad": yaw, "body_pitch_rad": self._pitch(rotation), "control_mode": self.control_mode, "posture_reference": self._posture_values(self._posture_reference), "posture_target": self._posture_values(self._posture_target), "velocity_world_mps": self.data.qvel[:3].tolist(), "command": command.tolist(), "fallen": self._fallen_ever, "collision_steps": self._collision_steps})
            self._record_state()
        if self._viewer is not None:
            if not self._viewer.is_running():
                closed_viewer, self._viewer = self._viewer, None
                closed_viewer.close()
                if not self._stopping:
                    raise KeyboardInterrupt("MuJoCo viewer was closed")
            else:
                self._viewer.sync()
                time.sleep(max(0.0, self.dt * self.decimation - (time.monotonic() - started)))

    def _advance_zero(self, seconds: float, *, record: bool = True) -> None:
        for _ in range(math.ceil(seconds / (self.dt * self.decimation))):
            self._step_control(np.zeros(3, dtype=np.float32), record=record)

    def capture(self) -> SensorFrame:
        self._record_state()
        self._frame_counter += 1
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self.data, camera="front_rgbd")
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera="front_rgbd")
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        camera_to_world = np.eye(4)
        # MuJoCo camera is x-right/y-up/z-back; SensorFrame uses x-right/y-down/z-forward.
        camera_to_world[:3, :3] = self.data.cam_xmat[self._camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        camera_to_world[:3, 3] = self.data.cam_xpos[self._camera_id]
        focal = 0.5 * self.height / math.tan(math.radians(float(self.model.cam_fovy[self._camera_id])) / 2)
        intrinsics = np.array([[focal, 0., (self.width-1)/2], [0., focal, (self.height-1)/2], [0., 0., 1.]])
        position, yaw, rotation = self._base_pose()
        state = {"motion_state": "idle", "body_height_m": float(position[2]), "body_pitch_rad": self._pitch(rotation), "pitch_convention": "nose_up_positive", "body_roll_rad": float(math.atan2(rotation[2, 1], rotation[2, 2])), "speed_mps": float(np.linalg.norm(self.data.qvel[:2])), "yaw_rate_radps": float(self.data.qvel[5]), "fallen": self._fallen_ever, "posture_control_supported": True, "control_mode": self.control_mode, "posture_target": self._posture_values(self._posture_target), "source": "simulated_proprioception", "time_mode": "paused_between_tools"}
        return SensorFrame(frame_id=f"frame-{self._frame_counter:05d}", sim_time_s=float(self.data.time), rgb=rgb, depth_m=depth, intrinsics=intrinsics, camera_to_world=camera_to_world, base_xy_yaw=np.array([position[0], position[1], yaw]), robot_state=state,
                           timestamp_s=float(self.data.time), time_source="simulation",
                           time_mode=self.profile.time_mode, localization_available=True,
                           localization_source="simulated_proprioception", depth_available=True,
                           depth_source="simulated_rgbd")

    def _motion(self, translation: np.ndarray, turn_rad: float, timeout_s: float) -> dict:
        start_pos, start_yaw, _ = self._base_pose()
        initial_rotation = np.array([[math.cos(start_yaw), -math.sin(start_yaw)], [math.sin(start_yaw), math.cos(start_yaw)]])
        target_xy = start_pos[:2] + initial_rotation @ translation
        target_yaw = wrap_angle(start_yaw + turn_rad)
        initial_sim_time = float(self.data.time)
        status, error = "timed_out", "MOTION_TIMEOUT"
        while self.data.time - initial_sim_time < timeout_s:
            if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
                error = "EPISODE_DEADLINE"
                break
            if self.sim_deadline_s is not None and self.data.time >= self.sim_deadline_s:
                error = "SIMULATION_DEADLINE"
                break
            if self._fallen_ever or self._fallen():
                status, error = "failed", "ROBOT_FELL"
                break
            if self._collision_steps:
                status, error = "failed", "COLLISION"
                break
            position, yaw, _ = self._base_pose()
            error_world = target_xy - position[:2]
            yaw_error = wrap_angle(target_yaw - yaw)
            # The pretrained gait has a low-speed deadband; this explicit local
            # tolerance is looser than ideal kinematics and is reported below.
            if np.linalg.norm(error_world) < 0.06 and abs(yaw_error) < 0.045:
                status, error = "succeeded", None
                break
            current_rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            error_local = current_rotation.T @ error_world
            command = np.array([np.clip(1.4 * error_local[0], -0.4, 0.4), np.clip(1.4 * error_local[1], -0.25, 0.25), np.clip(1.8 * yaw_error, -0.8, 0.8)], dtype=np.float32)
            self._step_control(command)
        stop_result = self.stop()
        final_position, final_yaw, _ = self._base_pose()
        position_error = float(np.linalg.norm(target_xy - final_position[:2]))
        heading_error = abs(wrap_angle(target_yaw - final_yaw))
        if status == "succeeded" and (position_error > 0.10 or heading_error > 0.10 or stop_result["status"] != "succeeded"):
            status, error = "failed", "DID_NOT_SETTLE_AT_TARGET"
        if self._fallen_ever:
            status, error = "failed", "ROBOT_FELL"
        elif self._collision_steps:
            status, error = "failed", "COLLISION"
        return {"status": status, "error_code": error, "measured": {"translation_start_frame_m": (initial_rotation.T @ (final_position[:2]-start_pos[:2])).tolist(), "yaw_delta_rad": wrap_angle(final_yaw-start_yaw), "position_error_m": position_error, "heading_error_rad": heading_error, "elapsed_sim_s": float(self.data.time-initial_sim_time)}, "stop": stop_result}

    def move_relative(self, forward_m: float, lateral_m: float, timeout_s: float = 8.0) -> dict:
        values = np.array([forward_m, lateral_m], dtype=float)
        if not np.isfinite(values).all() or np.linalg.norm(values) > 1.0 or not 0 < timeout_s <= 20:
            return {"status": "failed", "error_code": "INVALID_MOTION"}
        if self._posture_controller is not None:
            return {"status": "failed", "error_code": "POSTURE_MODE_ACTIVE",
                    "message": "Call reset_posture before locomotion; moving posture control is unsupported."}
        return self._motion(values, 0., timeout_s)

    def turn(self, yaw_rad: float, timeout_s: float = 6.0) -> dict:
        if not math.isfinite(yaw_rad) or abs(yaw_rad) > math.pi/2 or not 0 < timeout_s <= 20:
            return {"status": "failed", "error_code": "INVALID_MOTION"}
        if self._posture_controller is not None:
            return {"status": "failed", "error_code": "POSTURE_MODE_ACTIVE",
                    "message": "Call reset_posture before locomotion; moving posture control is unsupported."}
        return self._motion(np.zeros(2), yaw_rad, timeout_s)

    @staticmethod
    def _posture_values(values: np.ndarray | None) -> dict | None:
        if values is None:
            return None
        return {"body_height_m": float(values[0]), "body_pitch_rad": float(values[1])}

    def _posture_error(self) -> str | None:
        if self._fallen_ever or self._fallen():
            return "ROBOT_FELL"
        if self._collision_steps:
            return "COLLISION"
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            return "EPISODE_DEADLINE"
        if self.sim_deadline_s is not None and self.data.time >= self.sim_deadline_s:
            return "SIMULATION_DEADLINE"
        return None

    def set_posture(self, body_height_m: float, body_pitch_rad: float, timeout_s: float = 8.0) -> dict:
        """Smoothly reach a stationary absolute posture through joint torques."""
        try:
            validate_posture(body_height_m, body_pitch_rad)
            if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 20:
                raise ValueError("timeout_s must be a finite number in (0, 20]")
        except (ValueError, TypeError, OverflowError) as exc:
            return {"status": "failed", "error_code": "INVALID_POSTURE", "message": str(exc)}
        error = self._posture_error()
        if error:
            return {"status": "failed", "error_code": error}
        started = float(self.data.time)
        requested = np.array([body_height_m, body_pitch_rad], dtype=float)
        if self._posture_controller is None:
            stopped = self.stop()
            if stopped["status"] != "succeeded":
                return {"status": "failed", "error_code": "STOP_NOT_CONFIRMED", "stop": stopped}
            error = self._posture_error()
            if error:
                return {"status": "failed", "error_code": error}
            controller = StaticPostureController(self.model, self._qpos_ids, self._dof_ids, self._base_id)
            controller.begin(self.data)
            position, _, rotation = self._base_pose()
            reference = np.array([position[2], self._pitch(rotation)])
            try:
                controller.target(*reference, validate_request=False)
                controller.target(*requested)
            except ValueError as exc:
                return {"status": "failed", "error_code": "INFEASIBLE_POSTURE", "message": str(exc),
                        "control_mode": self.control_mode}
            self._posture_reference = reference
            self._posture_controller = controller
        else:
            # Failed scratch IK leaves both the live pose and the old hold target intact.
            try:
                self._posture_controller.target(*requested)
            except ValueError as exc:
                return {"status": "failed", "error_code": "INFEASIBLE_POSTURE", "message": str(exc),
                        "control_mode": self.control_mode}
        self._posture_target = requested.copy()
        period = self.dt * self.decimation
        rates = np.array([0.06, 0.15])
        initial_reference = self._posture_reference.copy()
        transition_s = float(np.max(np.abs(requested - initial_reference) / rates))
        transition_started = float(self.data.time)
        stable_s = 0.0
        error, status = "POSTURE_TIMEOUT", "timed_out"
        while self.data.time - started < timeout_s:
            problem = self._posture_error()
            if problem:
                error, status = problem, "failed"
                break
            fraction = min(1.0, (self.data.time - transition_started + period) / max(transition_s, period))
            next_reference = initial_reference + fraction * (requested - initial_reference)
            previous_reference = self._posture_reference.copy()
            self._posture_reference = next_reference
            try:
                self._step_control(np.zeros(3, dtype=np.float32))
            except ValueError:
                self._posture_reference = previous_reference
                error, status = "INFEASIBLE_POSTURE", "failed"
                break
            position, _, rotation = self._base_pose()
            within_target = (abs(position[2] - body_height_m) <= self.posture_height_tolerance_m
                             and abs(self._pitch(rotation) - body_pitch_rad) <= self.posture_pitch_tolerance_rad)
            still = (np.linalg.norm(self.data.qvel[:3]) < 0.06
                     and np.linalg.norm(self.data.qvel[3:6]) < 0.15)
            stable_s = stable_s + period if within_target and still and np.allclose(next_reference, requested) else 0.0
            if stable_s >= 0.3:
                status, error = "succeeded", None
                break
        # On interruption/timeout keep the last feasible reference, never continue a transition in stop().
        if status != "succeeded":
            self._posture_target = self._posture_reference.copy()
        stop_result = self.stop()
        position, _, rotation = self._base_pose()
        measured = np.array([position[2], self._pitch(rotation)])
        problem = self._posture_error()
        if problem:
            status, error = "failed", problem
        elif status == "succeeded" and (stop_result["status"] != "succeeded"
                or abs(measured[0] - requested[0]) > self.posture_height_tolerance_m
                or abs(measured[1] - requested[1]) > self.posture_pitch_tolerance_rad):
            status, error = "failed", "POSTURE_NOT_CONFIRMED"
        return {"status": status, "error_code": error, "requested": self._posture_values(requested),
                "measured": {**self._posture_values(measured), "height_error_m": float(measured[0] - requested[0]),
                             "pitch_error_rad": float(measured[1] - requested[1]),
                             "elapsed_sim_s": float(self.data.time - started), "stable_duration_s": stable_s},
                "control_mode": self.control_mode, "stop": stop_result}

    def reset_posture(self) -> dict:
        """Explicitly restore the initial standing posture before resuming the gait."""
        error = self._posture_error()
        if error:
            return {"status": "failed", "error_code": error}
        started = float(self.data.time)
        if self._posture_controller is None:
            result = {**self.stop(), "control_mode": self.control_mode}
            position, _, rotation = self._base_pose()
            result["measured"] = {"body_height_m": float(position[2]), "body_pitch_rad": self._pitch(rotation),
                                  "elapsed_sim_s": float(self.data.time - started)}
            error = self._posture_error()
            if error:
                result.update(status="failed", error_code=error)
            return result
        result = self.set_posture(*self._nominal_posture)
        if result["status"] != "succeeded":
            return result
        error = self._posture_error()
        if error:
            return {**result, "status": "failed", "error_code": error}
        self._posture_controller = None
        self._posture_reference = self._posture_target = None
        self.last_action[:] = 0
        stopped = self.stop()
        position, _, rotation = self._base_pose()
        measured = np.array([position[2], self._pitch(rotation)])
        result = {**result, **stopped, "control_mode": self.control_mode, "stop": stopped,
                  "measured": {**result["measured"], **self._posture_values(measured),
                               "height_error_m": float(measured[0] - self._nominal_posture[0]),
                               "pitch_error_rad": float(measured[1] - self._nominal_posture[1]),
                               "elapsed_sim_s": float(self.data.time - started)}}
        error = self._posture_error()
        if error:
            result.update(status="failed", error_code=error)
        elif result["status"] == "succeeded" and (
                abs(measured[0] - self._nominal_posture[0]) > self.posture_height_tolerance_m
                or abs(measured[1] - self._nominal_posture[1]) > self.posture_pitch_tolerance_rad):
            result.update(status="failed", error_code="POSTURE_NOT_CONFIRMED")
        return result

    def stop(self) -> dict:
        # Keep the active controller/reference; stopping never restores a standing posture.
        if self._viewer is not None and not self._viewer.is_running():
            closed_viewer, self._viewer = self._viewer, None
            closed_viewer.close()
        self._stopping = True
        try:
            self._advance_zero(0.6)
        finally:
            self._stopping = False
        speed = float(np.linalg.norm(self.data.qvel[:2]))
        angular_speed = abs(float(self.data.qvel[5]))
        stopped = speed < 0.12 and angular_speed < 0.25 and not self._fallen_ever
        return {"status": "succeeded" if stopped else "failed", "error_code": None if stopped else "STOP_NOT_CONFIRMED", "speed_mps": speed, "yaw_rate_radps": angular_speed}

    def close(self) -> None:
        try:
            self._record_state()
            recording = getattr(self, "_recording", None)
            if recording is not None:
                recording.finish()
        finally:
            try:
                if self._viewer is not None:
                    self._viewer.close()
                    self._viewer = None
            finally:
                if self._renderer is not None:
                    self._renderer.close()
                    self._renderer = None

    # Private helpers remain thin compatibility adapters for existing local code.
    _add_letter = staticmethod(add_letter)

    def _make_model(self):
        environment = getattr(self, "environment", None)
        if environment is None:
            environment = self.environment = MujocoEnvironment(self.scenario, target_xy=self._target_xy)
        model = environment.make_model(
            self.asset_dir / "go2_rl_lab/resources/robots/go2/go2.xml", dt=self.dt,
            width=self.width, height=self.height, camera_body_name="base",
            camera_attributes={"name": "front_rgbd", "pos": ".32 0 .08",
                               "xyaxes": "0 -1 0 0 0 1", "fovy": "60"})
        self._environment_geom_names = list(environment.collision_geom_names)
        return model

    def _evaluation_state(self) -> TaskEvaluationState:
        position, yaw, _ = self._base_pose()
        return TaskEvaluationState(position_xyz_m=position, yaw_rad=yaw,
                                   velocity_xy_mps=self.data.qvel[:2].copy(),
                                   yaw_rate_radps=float(self.data.qvel[5]),
                                   fallen=self._fallen_ever, collision_steps=self._collision_steps,
                                   trajectory=self.trajectory, sim_time_s=float(self.data.time))

    def _task_evaluator(self) -> TaskEvaluator:
        evaluator = getattr(self, "evaluator", None)
        if evaluator is None:
            # Older local callers may construct a state-only test double.
            evaluator = TaskEvaluator(getattr(self, "scenario", None), self._target_xy,
                                      self._evaluation_state)
        return evaluator

    @property
    def public_task_spec(self) -> dict:
        return self._task_evaluator().public_task_spec

    def evaluate(self, standoff_m: float = 1.0, tolerance_m: float = 0.18) -> dict:
        """Legacy scoring entry point; production scoring uses evaluator directly."""
        return self._task_evaluator().evaluate(standoff_m, tolerance_m)

    def execute(self, call: ToolCall) -> dict:
        """Validate the public profile, then dispatch only physical Go2 actions."""
        if not isinstance(call, ToolCall):
            return {"status": "failed", "error_code": "INVALID_ARGUMENTS", "message": "Expected ToolCall"}
        try:
            self.profile.validate_action(call.name, call.arguments)
        except ValueError as exc:
            return {"status": "failed", "error_code": getattr(exc, "error_code", "INVALID_ARGUMENTS"),
                    "message": str(exc)}
        actions = {"move_relative": self.move_relative, "turn": self.turn,
                   "set_posture": self.set_posture, "reset_posture": self.reset_posture,
                   "stop": self.stop}
        if call.name not in actions:
            return {"status": "failed", "error_code": "UNSUPPORTED_CAPABILITY",
                    "message": "Only physical actions are executed by this robot adapter"}
        if not self._command_lock.acquire(blocking=False):
            return {"status": "failed", "error_code": "BACKEND_BUSY"}
        try:
            return actions[call.name](**call.arguments)
        finally:
            self._command_lock.release()

    def cancel(self) -> dict:
        """Settle an idle synchronous adapter; no cross-thread preemption or emergency stop."""
        if not self._command_lock.acquire(blocking=False):
            return {"status": "failed", "error_code": "CANCEL_UNSUPPORTED", "cancelled": False,
                    "reason": "synchronous_action_in_progress"}
        try:
            stopped = self.stop()
        finally:
            self._command_lock.release()
        result = {"status": "succeeded" if stopped.get("status") == "succeeded" else "failed",
                  "cancelled": False, "reason": "no_active_action", "stop_result": stopped}
        if result["status"] == "failed":
            result["error_code"] = stopped.get("error_code") or "STOP_NOT_CONFIRMED"
        return result

