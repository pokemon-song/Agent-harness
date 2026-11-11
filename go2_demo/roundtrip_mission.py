"""Public round-trip instructions and private return acceptance.

The language model learns the outbound route from public observations. Nothing
in this module sends the measured route or evaluator coordinates to either
agent, generates motion commands, or replays recorded actions.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Real

from .roundtrip import evaluate_roundtrip_trajectory


RETURN_TOLERANCE_M = .30
RETRACE_TOLERANCE_M = .50

OUTBOUND_NOTE = (
    "\n这是一次连续往返任务的去程。到达B并停稳后调用finish，只结束去程。"
    "随后会清空规划Agent的对话历史、临时笔记和跟踪点，并启动新的返程规划上下文；"
    "机器人、场景、仿真时间和当前姿态均不重置，local_odometry继续使用本次往返共同的起点坐标系。"
    "返程需要读取记忆Agent根据本次真实观测和执行反馈保存的MD路线，再自行规划原路返回。"
    "继续用当前RGB、实测几何与执行反馈完成去程；不要把动作请求量当作实际位移。"
)

RETURN_INSTRUCTION = (
    "去程已经结束，现在仍处于同一个物理场景的B处，开始原路返回。"
    "你的对话历史、临时笔记和跟踪点已清空；机器人没有复位，local_odometry的共同起点坐标系保持不变。"
    "先检索并读取去程保存的MD路线，根据其中有证据的地标、实际绕行侧、转折点和位置误差，"
    "结合当前RGB、实测几何及动作反馈，自主按去程相反的顺序走回出发处。"
    "必须反向通过实际经过的通道，从抽屉柜去程经过的同一侧返回；"
    "不能以一条直接指向起点的直线替代原路，也不要把历史运动指令机械倒放。"
    "返程应保持在实际去程路线周围0.50米内，并覆盖至少95%的去程路线；"
    "最后让机身base回到去程起点0.30米范围内，停稳后调用finish，不要求特定终点朝向。"
    "全程不得接触障碍物或跌倒。用实际位移确认进度；记忆不充分或当前观测不支持时，"
    "先重新观察核实，无法可靠继续则如实abort。"
)

MEMORY_NOTICE = (
    "本任务需要你为同一场景中的原路返回保留去程路线记忆。"
    "规划Agent先绕过抽屉柜、穿过绿色门框并到达B；到B后它的对话历史和跟踪点会清空，"
    "返程只能从你发布的MD中恢复去程经历，并结合当前感知自行规划。"
    "机器人世界、实际姿态和local_odometry共同起点坐标系不会在B处重置；"
    "请记录观测中的coordinate_frame_id、起点参考及测量来源，明确不同阶段共用此坐标系。"
    "你仍是记忆Agent，负责保存证据和更新文档，不负责输出下一步运动指令。"
    "从登记的RGB观测、几何测量、local_odometry和实际执行反馈识别真正走过的路线："
    "包括可辨认的地标、经过柜体的实际一侧、少量有用的实测转折点、门框通过位置及余量、"
    "已知测量误差和需要返程重新观察的事项。区分尚未执行的计划、已请求的动作和已经测得的位移。"
    "不要从本任务描述虚构路线坐标或预选绕行侧，也不要把未验证的导航计划写成已完成经历。"
    "自行选择清晰的MD文件名，把本次路线整理成可逆序理解的简短记录，并引用公开证据。"
    "在关键经过点得到证据后及时保存；已有文档先读取，优先局部更新，保留之前仍有效的路线段。"
    "去程结束前保存最近测得的B处状态和路线不确定性，供新规划上下文使用。"
)


def outbound_instruction(scene):
    """Keep the original physical patrol task; add only the handoff contract."""
    if not isinstance(getattr(scene, "instruction", None), str):
        raise ValueError("The outbound scene requires a public instruction")
    return scene.instruction + OUTBOUND_NOTE


def return_instruction():
    return RETURN_INSTRUCTION


def memory_notice():
    return MEMORY_NOTICE


def _finite(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except (TypeError, OverflowError):
        return False


def _vector(value, length, name):
    try:
        if len(value) != length or not all(_finite(item) for item in value):
            raise ValueError(name + " requires finite measured coordinates")
        return tuple(float(item) for item in value)
    except TypeError as exc:
        raise ValueError(name + " requires finite measured coordinates") from exc


class ReturnEvaluator:
    """Score the new return trajectory while retaining cumulative safety state.

    Construct at the actual turnaround before return motion. The caller supplies
    a privately archived outbound path including its measured start pose. The
    backend trajectory remains cumulative and unmodified; only its new suffix is
    scored as the return leg. The captured turnaround pose closes the first
    measured segment when control-step rows begin after the phase boundary.
    """

    def __init__(self, snapshot, outbound_trajectory, start_xy_m, turnaround_xy_m,
                 return_start_index, scene):
        if not callable(snapshot):
            raise TypeError("snapshot must be callable")
        if not isinstance(outbound_trajectory, list):
            raise TypeError("outbound_trajectory must be a measured list")
        if type(return_start_index) is not int or return_start_index < 0:
            raise ValueError("return_start_index must be nonnegative")
        self._snapshot = snapshot
        self._outbound = deepcopy(outbound_trajectory)
        self._start = _vector(start_xy_m, 2, "start_xy_m")
        self._turnaround = _vector(turnaround_xy_m, 2, "turnaround_xy_m")
        self._return_start_index = return_start_index
        self.scene = scene
        cabinet = next((item for item in scene.objects if item.name == "drawers_body"), None)
        if cabinet is None or len(cabinet.position) != 3 or len(cabinet.size) < 2:
            raise ValueError("The original drawers scene requires its cabinet geometry")
        self._geometry = {
            "passage_x_m": scene.passage_x,
            "passage_center_y_m": scene.passage_center_y,
            "passage_half_width_m": scene.passage_half_width,
            "cabinet_x_m": cabinet.position[0],
            "cabinet_center_y_m": cabinet.position[1],
            "cabinet_half_width_m": cabinet.size[1],
            "cabinet_clearance_m": .12,
            "return_tolerance_m": RETURN_TOLERANCE_M,
            "retrace_tolerance_m": RETRACE_TOLERANCE_M,
            "min_outbound_coverage": .95,
        }
        if not all(_finite(value) for value in self._geometry.values()):
            raise ValueError("Return geometry requires finite scene values")
        boundary = snapshot()
        if not isinstance(boundary.trajectory, list) or return_start_index > len(boundary.trajectory):
            raise ValueError("Return boundary lies beyond the measured trajectory")
        position = _vector(boundary.position_xyz_m, 3, "turnaround position")
        if not _finite(boundary.sim_time_s):
            raise ValueError("Turnaround requires a measured simulation time")
        if math.dist(position[:2], self._turnaround) > .05:
            raise ValueError("Supplied turnaround disagrees with its measured snapshot")
        self._boundary = {"base_xyz_m": list(position), "sim_time_s": float(boundary.sim_time_s)}

    def criteria(self, standoff_m=1., tolerance_m=.18):
        """Private acceptance record, never a planner observation."""
        encoded = json.dumps(self._outbound, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        return {"goal_kind": "roundtrip_return", "start_xy_m": list(self._start),
                "turnaround_xy_m": list(self._turnaround), "return_start_index": self._return_start_index,
                "outbound_reference_samples": len(self._outbound),
                "outbound_trajectory_sha256": hashlib.sha256(encoded).hexdigest(),
                **self._geometry, "max_speed_mps": .12, "max_yaw_rate_radps": .25,
                "requires_finish": True, "no_collision_or_fall": True,
                "heading_required": False, "reference_body": "base"}

    @property
    def public_task_spec(self):
        """Only general requirements; no route, target position, or truth flags."""
        return {"goal_kind": "roundtrip_return", "reference_body": "base",
                "stop_required": True, "collision_free_required": True,
                "max_stopped_speed_mps": .12, "max_stopped_yaw_rate_radps": .25,
                "distance_tolerance_m": RETURN_TOLERANCE_M,
                "retrace_tolerance_m": RETRACE_TOLERANCE_M, "min_outbound_coverage": .95,
                "heading_required": False, "stages_ordered": True,
                "stage_rules": [
                    {"stage_id": "retrace_route", "instruction": "依据去程记忆逆序返回，反向通过实际经过的通道，并经过柜体去程相同一侧；保持0.50米路线走廊并覆盖至少95%的去程路线。"},
                    {"stage_id": "return_to_start", "instruction": "机身base回到原起点0.30米范围内，停稳后finish；不要求特定朝向。"},
                ]}

    def evaluate(self, standoff_m=1., tolerance_m=.18):
        state = self._snapshot()
        if not isinstance(state.trajectory, list) or self._return_start_index > len(state.trajectory):
            raise ValueError("Cumulative trajectory was cleared or truncated after turnaround")
        returning = deepcopy(state.trajectory[self._return_start_index:])
        if not returning:
            returning = [deepcopy(self._boundary)]
        else:
            first_time = returning[0].get("sim_time_s") if isinstance(returning[0], dict) else None
            if _finite(first_time) and first_time > self._boundary["sim_time_s"]:
                returning.insert(0, deepcopy(self._boundary))
        route = evaluate_roundtrip_trajectory(
            self._outbound, returning, start_xy_m=list(self._start),
            turnaround_xy_m=list(self._turnaround), **self._geometry)
        position = _vector(state.position_xyz_m, 3, "current position")
        velocity = _vector(state.velocity_xy_mps, 2, "current velocity")
        speed = math.hypot(*velocity)
        if not _finite(state.yaw_rate_radps):
            raise ValueError("Current yaw rate must be finite")
        if type(state.fallen) is not bool or type(state.collision_steps) is not int or state.collision_steps < 0:
            raise ValueError("Safety history must contain native cumulative indicators")
        yaw_rate = float(state.yaw_rate_radps)
        stopped = speed < .12 and abs(yaw_rate) < .25
        distance = math.dist(position[:2], self._start)
        returned = distance <= RETURN_TOLERANCE_M
        return {**route, "geometry_success": bool(route["success"]),
                "success": bool(route["success"] and returned and stopped
                                and not state.fallen and not state.collision_steps),
                "goal_kind": "roundtrip_return", "returned_to_start": returned,
                "return_distance_to_start_m": distance, "return_tolerance_m": RETURN_TOLERANCE_M,
                "retrace_tolerance_m": RETRACE_TOLERANCE_M, "state_valid": True,
                "stopped": stopped, "speed_mps": speed, "yaw_rate_radps": yaw_rate,
                "fallen": state.fallen, "collision": bool(state.collision_steps),
                "collision_physics_steps": state.collision_steps,
                "heading_required": False, "source": "privileged_roundtrip_evaluator"}
