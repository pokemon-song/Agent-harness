"""A cross-shift delivery task whose order is available only in the first shift.

The delivery world, public instructions and submission feedback are independent
of the expected answer. Only final, private evaluation compares a receipt with
the order. This module neither calls a model nor saves an agent's memories.
"""

from copy import deepcopy
import re
import secrets

from .scenarios import COLORS, SceneObject, Scenario, TargetSpec, get_scenario
from .types import ToolCall


DESTINATION_XY = {"A": (3.2, 0.8), "B": (3.2, 0.0), "C": (3.2, -0.8)}
_ORDER_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z", re.ASCII)
_RECEIPT_CODE = re.compile(r"[0-9]{6}\Z", re.ASCII)
_FIELDS = frozenset({"order_id", "destination", "receipt_code"})
_DELIVERY_INSTRUCTION = (
    "这里有 A、B、C 三块白色收货标牌。根据先前班次的订单选择目的地，"
    "移动到对应标牌中心水平距离约 1 米处，面向它并停稳。"
    "所有收货标牌都只是位置标签，不会显示配送单或回执码。"
    "停稳后提交订单回执，再调用 finish；全程不得接触任何标牌或场景物体。"
)


def _valid_order_id(value):
    return isinstance(value, str) and _ORDER_ID.fullmatch(value) is not None


def _valid_receipt(value):
    return (
        isinstance(value, dict)
        and set(value) == _FIELDS
        and _valid_order_id(value["order_id"])
        and isinstance(value["destination"], str)
        and value["destination"] in DESTINATION_XY
        and isinstance(value["receipt_code"], str)
        and _RECEIPT_CODE.fullmatch(value["receipt_code"]) is not None
    )


def _checked_order(order):
    if not _valid_receipt(order):
        raise ValueError("An order requires an opaque ID, destination A/B/C and six ASCII digits")
    return dict(order)


def make_order() -> dict:
    """Draw the ID, destination and code independently of scenario seeds."""
    return {
        "order_id": "order-" + secrets.token_hex(8),
        "destination": secrets.choice(tuple(DESTINATION_XY)),
        "receipt_code": "".join(secrets.choice("0123456789") for _ in range(6)),
    }


def briefing_instruction(order) -> str:
    """The public order remains available throughout this first shift only."""
    order = _checked_order(order)
    return get_scenario("composite_drawers_gate_b").instruction + (
        "\n\n这是跨班次任务的第一班次。当前巡检任务的终点仍为 B；下面的配送目的地仅用于下一班次，"
        "不要用它替换当前巡检终点。\n"
        f"公开调度单：订单 ID={order['order_id']}；下一班次目的地={order['destination']}；"
        f"回执码={order['receipt_code']}（六位数字，前导零必须保留）。\n"
        "这张调度单只在第一班次提供。完成当前巡检并调用 finish 后，机器人和模型会重新启动，"
        "全部对话历史、推理历史、跟踪点和临时工作笔记都会清空。下一班次只保留长期记忆库；"
        "交付指令只给订单 ID，不再给目的地和回执码。你需要自行决定保存内容与时机，"
        "通过有公开来源的长期记忆交接订单，下一班次才能完成正确目的地的交付。"
        "不要预先声称独立验收成功，也不要把一次运动反馈的推测当作可靠控制规律。"
    )


def recall_instruction(order_id) -> str:
    """Only the public lookup ID survives the shift boundary in the prompt."""
    if not _valid_order_id(order_id):
        raise ValueError("Invalid order ID")
    return (
        f"这是新班次的交付任务，订单 ID={order_id}。上一班次已经提供过完整调度单，"
        "当前对话、跟踪点及临时笔记均从空白开始。"
        + _DELIVERY_INSTRUCTION
        + "请根据长期记忆恢复本订单的目的地和六位回执码，并依据当前图像与实测几何导航。"
        "不要把历史位置或默认场景目标当作当前订单答案。"
        "到正确目的地并停稳后，调用 submit_delivery(order_id, destination, receipt_code) 提交完整回执，"
        "然后调用 finish。第一次格式有效的提交会锁定，无法改答案；提交反馈只确认收件，"
        "不会告诉你答案是否正确。若无法恢复订单，请如实说明并调用 abort，不要编造回执码。"
    )


def delivery_scenario() -> Scenario:
    """Return the same physical ABC layout for every hidden order.

    B is only a fixed geometry-building placeholder. The mission driver selects
    the private evaluator's target coordinates separately; no public geometry,
    task ID or instruction may be changed based on the order's destination.
    """
    objects = tuple(
        SceneObject("sign_" + letter, "sign", (*DESTINATION_XY[letter], 0.65),
                    (0.025, 0.20, 0.25), COLORS["white"], "distractor", letter=letter)
        for letter in ("A", "C")
    )
    objects += tuple(
        SceneObject("sign_support_" + letter, "box", (*xy, 0.20),
                    (0.025, 0.025, 0.20), (0.25, 0.25, 0.25, 1.0))
        for letter, xy in DESTINATION_XY.items()
    )
    return Scenario(
        task_id="memory_delivery_abc", title="跨班次订单交付", category="memory_mission",
        instruction=_DELIVERY_INSTRUCTION,
        target=TargetSpec("sign", "white", DESTINATION_XY["B"], z=0.65,
                          half_size=(0.025, 0.20, 0.25), letter="B"),
        objects=objects, goal_kind="standoff", target_jitter_m=0.0,
        source="local_memory_mission_mujoco",
    )


class DeliveryReceipt:
    """One immutable submitted answer; no correctness oracle in the tool path."""

    extension_id = "delivery-receipt-v1"

    def __init__(self, order):
        self._expected = _checked_order(order)
        self._submission = None

    @property
    def tools(self):
        return [{"type": "function", "function": {
            "name": "submit_delivery",
            "description": (
                "Submit the order ID, destination letter and six-digit receipt code after reaching "
                "the destination and stopping. The first format-valid submission is locked. "
                "Feedback only confirms receipt; independent evaluation later checks correctness. "
                "Call finish separately after submitting."
            ),
            "parameters": {"type": "object", "properties": {
                "order_id": {"type": "string", "minLength": 1, "maxLength": 128,
                             "pattern": "^[A-Za-z0-9_-]+$"},
                "destination": {"type": "string", "enum": ["A", "B", "C"]},
                "receipt_code": {"type": "string", "minLength": 6, "maxLength": 6,
                                 "pattern": "^[0-9]{6}$"},
            }, "required": ["order_id", "destination", "receipt_code"],
                "additionalProperties": False},
        }}]

    @staticmethod
    def _result(call, **fields):
        return {"call_id": call.call_id, "tool": call.name, **fields}

    def execute(self, call: ToolCall) -> dict:
        if not isinstance(call, ToolCall):
            raise TypeError("DeliveryReceipt requires a ToolCall")
        if call.name != "submit_delivery":
            return self._result(call, status="failed", error_code="UNKNOWN_TOOL")
        if not _valid_receipt(call.arguments):
            return self._result(call, status="failed", error_code="INVALID_ARGUMENTS",
                                message="Expected an order ID, destination A/B/C and six ASCII digits")
        duplicate = self._submission is not None
        if duplicate and self._submission != call.arguments:
            return self._result(call, status="failed", error_code="RECEIPT_ALREADY_SUBMITTED",
                                message="The first format-valid receipt is locked")
        if not duplicate:
            self._submission = dict(call.arguments)
        result = self._result(call, status="succeeded", submitted=True,
                              message="Receipt recorded. Independent evaluation runs after finish.")
        if duplicate:
            result["deduplicated"] = True
        return result

    def evaluate(self) -> dict:
        """Private final evaluation only; never feed this result to the model."""
        submitted = self._submission is not None
        matches = {
            "order_id_match": bool(submitted and self._submission["order_id"] == self._expected["order_id"]),
            "destination_match": bool(submitted and self._submission["destination"] == self._expected["destination"]),
            "code_match": bool(submitted and self._submission["receipt_code"] == self._expected["receipt_code"]),
        }
        return {"extension_id": self.extension_id, "success": submitted and all(matches.values()),
                "submitted": submitted, **matches, "submission": deepcopy(self._submission)}
