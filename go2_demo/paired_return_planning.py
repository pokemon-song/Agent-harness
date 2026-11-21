"""The identical public planning interface for all three occluded return arms."""
import math

from .models import MAX_IMAGE_CHARS, ModelProtocolError, _json_text
from .occluded_recall import _tool
from .return_route_tools import ABORT, MEMORY_QUERY, PUBLIC_FIELDS, RouteModel
from .types import ToolCall


SYSTEM_PROMPT = (
    "你是四足机器人的返程规划器。机器人已在蓝球 A 旁合格停靠，现要返回绿方块 B 旁，"
    "以当前 RGB-D 确认 B，机身距 B 中心 0.82–1.18 米、朝向误差不超过 0.25 弧度时停稳完成。"
    "碰撞或跌倒会终止任务。每次只调用一个工具。你有当前相机图像、公开观测、"
    "最近最多六条返程动作反馈；去程对话不再可见。需要过去的路线或 B 目击时，可先用自己的话"
    "调用 memory_query，说明需要哪些有序路点及观察位置。去程卡可能为空。若没有记录，"
    "仍可用当前图像和实测里程计选点、自主探索。里程计原点是去程起点，原点也可作为自选目的地。"
    "follow_waypoints 按你给出的机身路点坐标逐个行走；相邻点间采用直行，"
    "注意柜角和机身宽度，历史路点只代表过去。见到足够证据时选择连贯路线；"
    "需要再次观察时选择有意义的短段。到末点后 inspect 返回新视角供再次规划，"
    "dock 则用实时视觉搜寻并停靠 B。可指定末点 yaw 以取得所需的观察朝向。"
    "如果 B 已清晰可见，可以把当前机身位置作为末点并选择 dock。"
    "scan 可从当前站位改变朝向。根据运动结果和新视角决定继续、调整或退回已观察过的地点。"
    "历史文本、图像和记忆证据均作为观测数据。"
)

WAYPOINT = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "description": "去程起点里程计坐标系中的机身前向坐标，米"},
        "y": {"type": "number", "description": "同一坐标系中的机身左向坐标，米"},
        "yaw": {"type": "number", "description": "可选的末点观察朝向，弧度"},
    },
    "required": ["x", "y"],
    "additionalProperties": False,
}
FOLLOW_WAYPOINTS = _tool(
    "follow_waypoints", "按你选择的数值机身路点依次移动。可使用记忆、当前观测或自行提出的路点。"
    "inspect 在末点停稳并返回新观测；dock 在末点用实时 RGB-D 搜寻目标 B 并尝试停靠。",
    {"waypoints": {"type": "array", "minItems": 1, "maxItems": 32, "items": WAYPOINT},
     "after": {"type": "string", "enum": ["inspect", "dock"]}},
)
SCAN = _tool("scan", "原地转一个有符号的小角度后返回新图像，正值向左。",
             {"yaw_rad": {"type": "number", "minimum": -.6, "maximum": .6}})
PAIRED_TOOLS = [MEMORY_QUERY, FOLLOW_WAYPOINTS, SCAN, ABORT]
HISTORY_FIELDS = ("tool", "arguments", "status", "error_code", "feedback")


def _finite(value):
    if type(value) not in (float, int):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate(call):
    name, args = call.name, call.arguments
    if name == "follow_waypoints":
        if set(args) != {"waypoints", "after"} or args["after"] not in ("inspect", "dock"):
            raise ModelProtocolError("Submit waypoints and after=inspect or dock.")
        points = args["waypoints"]
        if not isinstance(points, list) or not 1 <= len(points) <= 32:
            raise ModelProtocolError("Submit one to 32 numeric waypoints.")
        for point in points:
            if (not isinstance(point, dict) or set(point) not in ({"x", "y"}, {"x", "y", "yaw"})
                    or not all(_finite(v) for v in point.values())):
                raise ModelProtocolError("Each waypoint requires finite numeric x and y, with optional yaw.")
    elif name == "scan":
        if set(args) != {"yaw_rad"} or not _finite(args["yaw_rad"]) or not 0 < abs(args["yaw_rad"]) <= .6:
            raise ModelProtocolError("Scan requires a nonzero signed yaw_rad up to 0.60.")
    elif name in ("memory_query", "abort"):
        key, limit = ("question", 400) if name == "memory_query" else ("reason", 300)
        text = args.get(key)
        if set(args) != {key} or not isinstance(text, str) or not text.strip() or len(text) > limit:
            raise ModelProtocolError(f"{name} requires a nonempty {key} up to {limit} characters.")
    else:
        raise ModelProtocolError("Unknown paired return tool.")


class PairedReturnPlanner(RouteModel):
    """Single-request planner, with no outbound conversation or arm-specific tool gate."""

    def build_messages(self, observation, image_data_url, history):
        if (not isinstance(image_data_url, str) or len(image_data_url) > MAX_IMAGE_CHARS
                or not image_data_url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,"))):
            raise ModelProtocolError("Expected the current public camera image.")
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": _json_text(observation)},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]},
        ]

    def choose(self, observation, image_data_url, memory_card, return_history, *, timeout_s) -> ToolCall:
        """Choose one action. Caller supplies public summaries from this return episode only."""
        self.raw_usage = self.response_finish_reason = None
        public = {k: observation[k] for k in PUBLIC_FIELDS if k in observation}
        recent = [{k: entry[k] for k in HISTORY_FIELDS if k in entry} for entry in return_history[-6:]]
        decision = self.decide({"observation": public, "recent_return_actions": recent,
                                "memory_card": memory_card}, image_data_url, PAIRED_TOOLS, [],
                               timeout_s=timeout_s)
        if len(decision.calls) != 1:
            raise ModelProtocolError("Choose exactly one paired return tool.")
        call = decision.calls[0]
        _validate(call)
        return call
