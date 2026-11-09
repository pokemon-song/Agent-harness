"""Go2-specific static stance IK in scratch MuJoCo data."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .posture_limits import validate_posture


class StaticPostureController:
    """Hold fixed foot contacts while changing a flat-ground standing pose.

    ``begin(data)`` captures the stance. ``target(height, pitch)`` then solves
    joint angles in private scratch data. Only the caller's torque-driven
    simulation can change the actual robot pose.
    """

    def __init__(self, model, qpos_ids, dof_ids, base_id):
        self.model = model
        self.qpos_ids = np.asarray(qpos_ids, dtype=int).copy()
        self.dof_ids = np.asarray(dof_ids, dtype=int).copy()
        self.base_id = int(base_id)
        self.data = mujoco.MjData(model)
        joint_ids = model.dof_jntid[self.dof_ids]
        self._joint_limits = model.jnt_range[joint_ids].copy()
        feet = []
        for leg in ("FL", "FR", "RL", "RR"):
            calf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
            candidates = np.flatnonzero(
                (model.geom_bodyid == calf)
                & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
            )
            if calf < 0 or len(candidates) != 1:
                raise ValueError(f"Cannot identify the {leg} foot contact sphere")
            feet.append(int(candidates[0]))
        self._feet = np.asarray(feet)
        self._anchors = None
        self._base_xy = None
        self._yaw = 0.0

    def begin(self, data) -> None:
        """Capture a stationary stance; do not modify the supplied live data."""
        self.data.qpos[:] = data.qpos
        self.data.qvel[:] = 0.0
        self._anchors = data.geom_xpos[self._feet].copy()
        # All four contact spheres stand on the flat z=0 floor. A millimetre
        # of contact penetration matches the soft MuJoCo ground contact.
        self._anchors[:, 2] = self.model.geom_size[self._feet, 0] - 0.001
        self._base_xy = data.xpos[self.base_id, :2].copy()
        rotation = data.xmat[self.base_id].reshape(3, 3)
        self._yaw = math.atan2(rotation[1, 0], rotation[0, 0])

    def target(self, body_height_m: float, body_pitch_rad: float, *,
               validate_request: bool = True) -> np.ndarray:
        """Solve a user target or an internal reference in scratch data.

        A measured starting stance can lie just outside the public request
        domain. Internal references skip request bounds, while retaining finite
        numeric inputs, joint limits and the same physical IK reachability test.
        """
        if validate_request:
            validate_posture(body_height_m, body_pitch_rad)
        elif not all(not isinstance(value, (bool, np.bool_))
                     and isinstance(value, (int, float, np.integer, np.floating))
                     and math.isfinite(value) for value in (body_height_m, body_pitch_rad)):
            raise ValueError("Posture references must contain finite numbers")
        if self._anchors is None:
            raise ValueError("Call begin(data) before requesting a posture target")
        previous = self.data.qpos.copy()
        self.data.qpos[:3] = [*self._base_xy, body_height_m]
        cy, sy = math.cos(self._yaw / 2), math.sin(self._yaw / 2)
        cp, sp = math.cos(body_pitch_rad / 2), math.sin(body_pitch_rad / 2)
        # Rz(yaw) Ry(-pitch): the robot's +x axis rises for positive pitch.
        self.data.qpos[3:7] = [cy * cp, sy * sp, -cy * sp, sy * cp]
        for _ in range(40):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            error = (self._anchors - self.data.geom_xpos[self._feet]).reshape(12)
            if np.max(np.abs(error)) < 1e-6:
                return self.data.qpos[self.qpos_ids].copy()
            rows = []
            for foot in self._feet:
                jacobian = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jacobian, None,
                              self.data.geom_xpos[foot], self.model.geom_bodyid[foot])
                rows.append(jacobian[:, self.dof_ids])
            jacobian = np.concatenate(rows)
            increment = np.linalg.solve(
                jacobian.T @ jacobian + 1e-6 * np.eye(12), jacobian.T @ error
            )
            self.data.qpos[self.qpos_ids] = np.clip(
                self.data.qpos[self.qpos_ids] + np.clip(increment, -0.1, 0.1),
                self._joint_limits[:, 0] + 0.001,
                self._joint_limits[:, 1] - 0.001,
            )
        mujoco.mj_kinematics(self.model, self.data)
        error = self._anchors - self.data.geom_xpos[self._feet]
        if not np.isfinite(error).all() or np.max(np.linalg.norm(error, axis=1)) > 0.002:
            self.data.qpos[:] = previous
            raise ValueError("Requested posture is unreachable from the current foot stance")
        return self.data.qpos[self.qpos_ids].copy()

