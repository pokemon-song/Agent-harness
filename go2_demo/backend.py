"""Real MuJoCo Go2 dynamics with a pinned, pretrained velocity locomotion policy.

Only resets write qpos. Every episode action is a torque-driven physics step.
The simulator is intentionally paused between tool calls (including API waits).
The 45-D policy observation and PD/action scaling derive from go2_rl_lab;
see THIRD_PARTY_NOTICES.md and the downloaded upstream LICENSE.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from .types import SensorFrame
from .scenarios import COLORS, LETTER_PATTERNS
from .posture import StaticPostureController, validate_posture
from .composite import evaluate_composite_trajectory


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def crossed_passage(trajectory: list[dict], gate_x: float, center_y: float, half_width: float) -> bool:
    """Check a forward gate crossing from measured poses, rather than final position alone."""
    previous = None
    for row in trajectory:
        point = np.asarray(row.get("base_xyz_m", ()), dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            previous = None
            continue
        if previous is not None and previous[0] < gate_x <= point[0]:
            fraction = (gate_x - previous[0]) / (point[0] - previous[0])
            crossing_y = float(previous[1] + fraction * (point[1] - previous[1]))
            if abs(crossing_y - center_y) <= half_width:
                return True
        previous = point
    return False


class MujocoGo2Backend:
    """Bounded locomotion and a separate, stationary joint posture controller."""

    posture_control_supported = True
    moving_posture_control_supported = False
    posture_height_tolerance_m = 0.02
    posture_pitch_tolerance_rad = 0.04
    width = 640
    height = 360
    dt = 0.002
    decimation = 10
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

    @property
    def public_task_spec(self) -> dict:
        """Declared acceptance rules; no privileged positions or score flags."""
        scenario = self.scenario
        goal_kind = scenario.goal_kind if scenario is not None else "standoff"
        result = {"goal_kind": goal_kind, "reference_body": "base",
                  "stop_required": True, "collision_free_required": True,
                  "max_stopped_speed_mps": 0.12, "max_stopped_yaw_rate_radps": 0.25}
        if goal_kind == "passage":
            result["required_clearance_m"] = scenario.passage_clearance_m
        else:
            result.update(standoff_m=scenario.standoff_m if scenario is not None else 1.0,
                          distance_tolerance_m=scenario.distance_tolerance_m if scenario is not None else 0.18,
                          heading_tolerance_rad=0.25)
        if goal_kind == "composite":
            result.update(stages_ordered=True, stage_rules=[
                {"stage_id": "avoid_drawers", "instruction": "先绕过抽屉柜，让机身base原点通过柜体后方并留出机身余量。"},
                {"stage_id": "pass_gate", "instruction": "随后让机身base原点从绿色门框开口内向前穿过，继续前进规定余量。",
                 "required_clearance_m": scenario.passage_clearance_m},
                {"stage_id": "approach_b", "instruction": "最后接近字母B至公开的目标距离与朝向容差内，停稳后调用finish。"},
            ])
        return result

    @staticmethod
    def _add_letter(world, name: str, position, size, letter: str) -> None:
        """Place readable black 7x5 glyph cells on the front of a white sign."""
        if letter not in LETTER_PATTERNS:
            raise ValueError(f"Unsupported scene letter: {letter}")
        x, y, z = position
        half_x, half_y, half_z = size
        cell_w, cell_h = 2 * half_y * .14, 2 * half_z * .105
        count = 0
        for row, pattern in enumerate(LETTER_PATTERNS[letter]):
            for column, bit in enumerate(pattern):
                if bit != "1":
                    continue
                # Camera forward is +x; increasing image u points toward world -y.
                position = (x - half_x - .003, y - (column - 2) * cell_w, z + (3 - row) * cell_h)
                ET.SubElement(world, "geom", name=f"{name}_glyph_{count}", type="box",
                              pos=" ".join(str(value) for value in position),
                              size=f".002 {cell_w / 2} {cell_h / 2}", rgba=".015 .015 .015 1",
                              contype="0", conaffinity="0")
                count += 1

    def __init__(self, asset_dir: Path, seed: int = 0, viewer: bool = False, scenario=None):
        self.asset_dir = Path(asset_dir).resolve()
        self.scenario = scenario
        if scenario is not None and not scenario.supported:
            raise ValueError(f"UNSUPPORTED_CAPABILITY: {scenario.unsupported_reason}")
        self.task_instruction = (scenario.instruction if scenario is not None else
                                 "找到红色标记，移动到距离标记约 1 米的位置，朝向标记并停止，然后调用 finish。距离按机身参考点到标记的水平距离理解。")
        self.deadline_monotonic: float | None = None
        self.sim_deadline_s: float | None = None
        self._viewer = None
        self._renderer = None
        self._stopping = False
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
        rng = np.random.default_rng(seed)
        self._target_xy = (np.asarray(scenario.target.xy, dtype=float).copy() if scenario is not None else
                           np.array([3.0 + rng.uniform(-0.15, 0.15), 0.55 + rng.uniform(-0.15, 0.15)]))
        if scenario is not None and scenario.target_jitter_m:
            self._target_xy += rng.uniform(-scenario.target_jitter_m, scenario.target_jitter_m, size=2)
        self._environment_geom_names = []
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

    def _verify_assets(self) -> None:
        manifest_path = Path(__file__).resolve().parents[1] / "assets.lock.json"
        self.asset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in self.asset_manifest["files"]:
            path = (self.asset_dir / entry["path"]).resolve()
            if not path.is_relative_to(self.asset_dir) or not path.is_file():
                raise FileNotFoundError("Go2 assets are missing; run: python scripts/fetch_assets.py")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Go2 asset checksum mismatch: {entry['path']}; run scripts/fetch_assets.py")

    def _make_model(self):
        source = self.asset_dir / "go2_rl_lab/resources/robots/go2"
        root = ET.parse(source / "go2.xml").getroot()
        # Load meshes through MuJoCo's virtual filesystem. This also supports
        # Windows workspaces whose paths contain non-ASCII characters.
        root.find("compiler").set("meshdir", "")
        mesh_assets = {path.name: path.read_bytes() for path in (source / "assets").glob("*.obj")}
        ET.SubElement(root, "option", timestep=str(self.dt), gravity="0 0 -9.81", integrator="implicitfast")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth=str(self.width), offheight=str(self.height))
        ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.3 0.3 0.3", specular="0.2 0.2 0.2")
        ET.SubElement(visual, "map", znear="0.01", zfar="20")
        asset = root.find("asset")
        ET.SubElement(asset, "texture", name="floor_texture", type="2d", builtin="checker", rgb1="0.28 0.32 0.36", rgb2="0.42 0.46 0.50", width="128", height="128")
        ET.SubElement(asset, "material", name="floor_material", texture="floor_texture", texrepeat="12 12", texuniform="true", reflectance="0")
        world = root.find("worldbody")
        ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", diffuse="0.8 0.8 0.8", directional="true")
        ET.SubElement(world, "geom", name="floor", type="plane", size="10 10 .1", material="floor_material", friction="1 .005 .0001")
        if self.scenario is None:
            target_x, target_y = self._target_xy
            ET.SubElement(world, "geom", name="red_target", type="box", pos=f"{target_x} {target_y} .5", size=".08 .20 .5", rgba=".85 .035 .025 1", friction="1 .005 .0001")
            # A small white cross makes the red target center a genuine visual feature.
            # These decorations have no physical collision geometry.
            ET.SubElement(world, "geom", name="target_cross_v", type="box", pos=f"{target_x-.082} {target_y} .5", size=".002 .018 .075", rgba=".95 .95 .95 1", contype="0", conaffinity="0")
            ET.SubElement(world, "geom", name="target_cross_h", type="box", pos=f"{target_x-.083} {target_y} .5", size=".002 .075 .018", rgba=".95 .95 .95 1", contype="0", conaffinity="0")
            ET.SubElement(world, "geom", name="blue_distractor", type="box", pos="3 -1.3 .4", size=".12 .18 .4", rgba=".04 .12 .85 1")
            self._environment_geom_names = ["red_target", "blue_distractor"]
        else:
            self._populate_scenario(world)
        base = world.find("body[@name='base']")
        ET.SubElement(base, "camera", name="front_rgbd", pos=".32 0 .08", xyaxes="0 -1 0 0 0 1", fovy="60")
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=mesh_assets)

    def _populate_scenario(self, world) -> None:
        target = self.scenario.target
        color = COLORS[target.color] if isinstance(target.color, str) else target.color
        position = (*self._target_xy, target.z)
        ET.SubElement(world, "geom", name="scenario_target", type="box" if target.kind in {"marker", "sign"} else target.kind,
                      pos=" ".join(str(value) for value in position),
                      size=" ".join(str(value) for value in target.half_size),
                      rgba=" ".join(str(value) for value in color), friction="1 .005 .0001")
        self._environment_geom_names.append("scenario_target")
        if target.letter:
            self._add_letter(world, "scenario_target", position, target.half_size, target.letter)
        elif self.scenario.category in {"baseline", "go_avoid", "go_through"}:
            front = self._target_xy[0] - target.half_size[0] - .002
            y, z = position[1:]
            ET.SubElement(world, "geom", name="target_cross_v", type="box", pos=f"{front} {y} {z}",
                          size=".002 .018 .075", rgba=".95 .95 .95 1", contype="0", conaffinity="0")
            ET.SubElement(world, "geom", name="target_cross_h", type="box", pos=f"{front - .001} {y} {z}",
                          size=".002 .075 .018", rgba=".95 .95 .95 1", contype="0", conaffinity="0")
        for item in self.scenario.objects:
            attributes = {"name": item.name, "type": "box" if item.kind == "sign" else item.kind,
                          "pos": " ".join(str(value) for value in item.position),
                          "size": " ".join(str(value) for value in item.size),
                          "rgba": " ".join(str(value) for value in item.rgba), "friction": "1 .005 .0001"}
            if item.euler is not None:
                attributes["euler"] = " ".join(str(value) for value in item.euler)
            if item.role == "decoration":
                attributes.update(contype="0", conaffinity="0")
            else:
                self._environment_geom_names.append(item.name)
            ET.SubElement(world, "geom", **attributes)
            if item.letter:
                self._add_letter(world, item.name, item.position, item.size, item.letter)

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
        if self._posture_controller is not None:
            target = self._posture_controller.target(*self._posture_reference)
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
        return SensorFrame(frame_id=f"frame-{self._frame_counter:05d}", sim_time_s=float(self.data.time), rgb=rgb, depth_m=depth, intrinsics=intrinsics, camera_to_world=camera_to_world, base_xy_yaw=np.array([position[0], position[1], yaw]), robot_state=state)

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
                controller.target(*reference)
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

    def evaluate(self, standoff_m: float = 1.0, tolerance_m: float = 0.18) -> dict:
        """Privileged state: only the independent evaluator may consume this result."""
        scenario = getattr(self, "scenario", None)
        if scenario is not None:
            standoff_m, tolerance_m = scenario.standoff_m, scenario.distance_tolerance_m
        position, yaw, _ = self._base_pose()
        difference = self._target_xy - position[:2]
        distance = float(np.linalg.norm(difference))
        heading_error = abs(wrap_angle(math.atan2(difference[1], difference[0]) - yaw))
        speed = float(np.linalg.norm(self.data.qvel[:2]))
        stopped = speed < 0.12 and abs(float(self.data.qvel[5])) < 0.25
        goal_kind = scenario.goal_kind if scenario is not None else "standoff"
        goal_reached = abs(distance-standoff_m) <= tolerance_m and heading_error <= 0.25
        passage = {}
        if goal_kind == "passage":
            if scenario.passage_x is None or scenario.finish_x_min is None:
                raise ValueError("Passage scenarios require a gate plane and an exit threshold")
            crossed = crossed_passage(self.trajectory, scenario.passage_x, scenario.passage_center_y,
                                      scenario.passage_half_width)
            beyond_exit = bool(position[0] >= scenario.finish_x_min)
            goal_reached = crossed and beyond_exit
            passage = {"passage_crossed": crossed, "beyond_exit": beyond_exit,
                       "gate_x_m": scenario.passage_x, "finish_x_min_m": scenario.finish_x_min}
        elif goal_kind == "composite":
            measured_path = evaluate_composite_trajectory(
                self.trajectory, cabinet_exit_x_m=scenario.cabinet_exit_x,
                passage_x_m=scenario.passage_x, passage_center_y_m=scenario.passage_center_y,
                passage_half_width_m=scenario.passage_half_width, finish_x_min_m=scenario.finish_x_min)
            # Use the measured current position as well as the recorded path.
            measured_path["beyond_exit"] = bool(measured_path["beyond_exit"] and position[0] >= scenario.finish_x_min)
            stages = {"avoid_drawers": measured_path["cabinet_cleared"],
                      "pass_gate": (measured_path["passage_crossed"] and measured_path["beyond_exit"]
                                    and measured_path["stages_in_order"]),
                      "approach_b": bool(goal_reached and stopped)}
            goal_reached = all(stages.values())
            passage = {**measured_path, "cabinet_exit_x_m": scenario.cabinet_exit_x,
                       "gate_x_m": scenario.passage_x, "finish_x_min_m": scenario.finish_x_min,
                       "stages": stages, "stage_completion_rate": sum(stages.values()) / len(stages)}
        return {"success": bool(goal_reached and stopped and not self._fallen_ever and not self._collision_steps),
                "goal_kind": goal_kind, "distance_to_target_center_m": distance,
                "distance_error_m": abs(distance-standoff_m), "heading_error_rad": heading_error,
                "standoff_m": standoff_m, "distance_tolerance_m": tolerance_m,
                "stopped": bool(stopped), "speed_mps": speed, "fallen": bool(self._fallen_ever),
                "collision": bool(self._collision_steps), "collision_physics_steps": self._collision_steps,
                "target_xy_m": self._target_xy.tolist(), "base_xy_m": position[:2].tolist(),
                "sim_time_s": float(self.data.time), "source": "privileged_simulator_evaluator", **passage}

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
