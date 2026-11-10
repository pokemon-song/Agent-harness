"""Three-shift dispatch evidence and an independently evaluated final receipt.

Announcements are public throughout their learning shift. The final prompt only
contains the order identifier. Nothing here writes an agent's memory, chooses its
tools or exposes a correctness oracle while the episode is running.
"""

from __future__ import annotations

from copy import deepcopy
import re
import secrets

from .complex_world import DESTINATION_XY, evaluate_route, final_scenario, learning_scenario
from .types import ToolCall


ALIASES = ("冷藏区", "质检区", "回收区")
ROUTES = ("blue", "yellow")
LEARNING_STAGES = ("dispatch", "fragments", "revision")
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z", re.ASCII)
_THREE_DIGITS = re.compile(r"[0-9]{3}\Z", re.ASCII)
_SIX_DIGITS = re.compile(r"[0-9]{6}\Z", re.ASCII)
_SUBMISSION_FIELDS = frozenset({"order_id", "dispatch_revision", "destination", "receipt_code", "route"})
_CONTRACT_FIELDS = frozenset({
    "order_id", "decoy_order_id", "alias_to_destination", "old_alias", "new_alias",
    "old_route", "new_route", "code_prefix", "code_suffix", "receipt_code",
    "old_revision", "new_revision", "decoy_alias", "decoy_route", "decoy_receipt_code",
})

_VERSION_RULES = (
    "同一订单的生效调度公告按修订号比较，较高修订号优先；出现得更晚不等于版本更高。"
    "标记 ARCHIVED 的公告只是历史参考，不重新生效。修订只覆盖明确变更的字段，"
    "未变更字段继续沿用；不同订单的信息不可混用。"
)
_MEMORY_HANDOFF = (
    "本班公告在本班任务文字中持续可见，不会自动写入长期记忆。"
    "完成本班实际巡检并 finish 后，会启动新进程并重置机器人："
    "对话历史、跟踪点和临时工作笔记全部清空，只有你自行保存的长期记忆能够交班。"
    "你自行决定保存、读取或修订的时机和内容；用正常机器人观测与反馈提供的公开证据引用记录事实，"
    "不要预先声称独立验收成功。"
)


def _matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _checked_contract(contract):
    if not isinstance(contract, dict) or set(contract) != _CONTRACT_FIELDS:
        raise ValueError("Invalid complex mission contract fields")
    if (not _matches(_IDENTIFIER, contract["order_id"])
            or not _matches(_IDENTIFIER, contract["decoy_order_id"])
            or contract["order_id"] == contract["decoy_order_id"]):
        raise ValueError("Orders require distinct opaque identifiers")
    mapping = contract["alias_to_destination"]
    if (not isinstance(mapping, dict) or set(mapping) != set(ALIASES)
            or any(not isinstance(value, str) for value in mapping.values())
            or set(mapping.values()) != set(DESTINATION_XY)):
        raise ValueError("Area aliases must map bijectively to A/B/C")
    if (not isinstance(contract["old_alias"], str) or contract["old_alias"] not in mapping
            or not isinstance(contract["new_alias"], str) or contract["new_alias"] not in mapping
            or contract["old_alias"] == contract["new_alias"]):
        raise ValueError("The revision must change the destination alias")
    if (contract["old_route"] not in ROUTES or contract["new_route"] not in ROUTES
            or contract["old_route"] == contract["new_route"]):
        raise ValueError("The revision must change the permitted route")
    if (type(contract["old_revision"]) is not int or type(contract["new_revision"]) is not int
            or not 1 <= contract["old_revision"] < contract["new_revision"] <= 100):
        raise ValueError("Revision numbers must increase within 1..100")
    if (not _matches(_THREE_DIGITS, contract["code_prefix"])
            or not _matches(_THREE_DIGITS, contract["code_suffix"])
            or not _matches(_SIX_DIGITS, contract["receipt_code"])
            or contract["receipt_code"] != contract["code_prefix"] + contract["code_suffix"]):
        raise ValueError("Receipt code must concatenate the two three-digit fragments")
    if (not isinstance(contract["decoy_alias"], str) or contract["decoy_alias"] not in mapping
            or contract["decoy_route"] not in ROUTES
            or not _matches(_SIX_DIGITS, contract["decoy_receipt_code"])):
        raise ValueError("Invalid independent distractor order")
    return deepcopy(contract)


def _digits(count):
    return "".join(secrets.choice("0123456789") for _ in range(count))


def make_contract() -> dict:
    """Independently sample identifiers, area mapping and code fragments."""
    order_id = "order-" + secrets.token_hex(8)
    decoy_order_id = "order-" + secrets.token_hex(8)
    while decoy_order_id == order_id:
        decoy_order_id = "order-" + secrets.token_hex(8)
    remaining = list(DESTINATION_XY)
    mapping = {alias: remaining.pop(secrets.randbelow(len(remaining))) for alias in ALIASES}
    old_alias = secrets.choice(ALIASES)
    new_alias = secrets.choice(tuple(alias for alias in ALIASES if alias != old_alias))
    old_route = secrets.choice(ROUTES)
    new_route = next(route for route in ROUTES if route != old_route)
    prefix, suffix = _digits(3), _digits(3)
    return _checked_contract({
        "order_id": order_id, "decoy_order_id": decoy_order_id,
        "alias_to_destination": mapping, "old_alias": old_alias, "new_alias": new_alias,
        "old_route": old_route, "new_route": new_route,
        "code_prefix": prefix, "code_suffix": suffix, "receipt_code": prefix + suffix,
        "old_revision": 1, "new_revision": 2,
        "decoy_alias": secrets.choice(ALIASES), "decoy_route": secrets.choice(ROUTES),
        "decoy_receipt_code": _digits(6),
    })


def learning_instruction(stage: str, contract: dict) -> str:
    """Expose only this shift's announcement, not a serialized full contract."""
    if stage not in LEARNING_STAGES:
        raise ValueError("Unknown learning stage")
    c = _checked_contract(contract)
    if stage == "dispatch":
        announcement = (
            f"第一班次公开调度公告：[ACTIVE DISPATCH 修订号={c['old_revision']}]。"
            f"订单 ID={c['order_id']}；最终配送目标区域别名={c['old_alias']}；"
            f"许可分流通道={c['old_route']}。"
            f"本订单六位回执码的前半段={c['code_prefix']}（恰好三位，保留前导零）；"
            "后半段与区域别名到实体字母标牌的对应关系将在另一班次提供。"
            "最终回执码按前半段接后半段拼成六位字符串，不作数字加法。"
        )
    elif stage == "fragments":
        mapping_text = "；".join(f"{alias}→{c['alias_to_destination'][alias]}" for alias in ALIASES)
        announcement = (
            f"第二班次仓储公告：当前交班订单 ID={c['order_id']}，"
            f"其六位回执码后半段={c['code_suffix']}（恰好三位，保留前导零）。"
            "前半段沿用第一班次提供的内容，本班不重复。按前半段接后半段拼接。"
            f"区域别名与实体标牌的有效对应表：{mapping_text}。"
            "本公告补充代码片段和区域映射，不更改当前订单的目的地区域别名或许可通道。\n"
            f"旁边另有一张不同订单的有效记录：订单 ID={c['decoy_order_id']}；"
            f"目标区域别名={c['decoy_alias']}；许可分流通道={c['decoy_route']}；"
            f"六位回执码={c['decoy_receipt_code']}。这张记录只属于它自己的订单号。"
        )
    else:
        announcement = (
            f"第三班次调度修订：[ACTIVE DISPATCH 修订号={c['new_revision']}]。"
            f"订单 ID={c['order_id']}；目的地区域别名改为={c['new_alias']}；"
            f"许可分流通道改为={c['new_route']}。"
            "这两个字段替代该订单的旧调度值；回执码两段及区域映射保持有效，本班均不重复。\n"
            f"随后屏幕回放历史公告：[ARCHIVED DISPATCH 修订号={c['old_revision']}]。"
            f"订单 ID={c['order_id']}；旧目的地区域别名={c['old_alias']}；"
            f"旧许可分流通道={c['old_route']}。这段历史回放没有重新激活旧调度。"
        )
    return (learning_scenario(stage).instruction
            + "\n\n以下公告仅用于跨班次最终配送，本班巡检仍按上面的本班目标完成。\n"
            + announcement + "\n" + _VERSION_RULES + "\n" + _MEMORY_HANDOFF)


def recall_instruction(order_id: str) -> str:
    if not _matches(_IDENTIFIER, order_id):
        raise ValueError("Invalid order ID")
    return (
        f"这是新的最终作业班次，订单 ID={order_id}。对话历史、跟踪点和临时工作笔记均已清空。"
        + final_scenario().instruction
        + "\n请从长期记忆恢复该订单当前有效的调度修订号、目的地区域别名、许可分流通道，"
        "将区域别名按已获得的对应表解析为实体 A/B/C 标牌，并把两段代码按前半段接后半段组成六位回执码。"
        + _VERSION_RULES
        + "结合当前图像与实测几何完成有效路线和最终停靠；不要把历史坐标当作当前测量，"
        "也不要把所有通道都走一遍作为选择路线的替代。"
        "停稳后调用 submit_mission(order_id, dispatch_revision, destination, receipt_code, route)，"
        "其中 dispatch_revision 是保存公告中的生效修订号，destination 为 A/B/C，"
        "route 为保存订单指定的 blue 或 yellow，receipt_code 是保留前导零的六位字符串。"
        "第一次格式有效的提交会锁定，反馈只确认收件，不提供答案或路线正确性；随后单独调用 finish。"
        "无法可靠恢复必要信息时，如实说明并 abort，不要编造缺失字段。"
    )


def _valid_submission(arguments):
    return (isinstance(arguments, dict) and set(arguments) == _SUBMISSION_FIELDS
            and _matches(_IDENTIFIER, arguments["order_id"])
            and type(arguments["dispatch_revision"]) is int and 1 <= arguments["dispatch_revision"] <= 100
            and isinstance(arguments["destination"], str) and arguments["destination"] in DESTINATION_XY
            and _matches(_SIX_DIGITS, arguments["receipt_code"])
            and isinstance(arguments["route"], str) and arguments["route"] in ROUTES)


class ComplexMissionReceipt:
    """Single immutable receipt plus private evaluation of the executed route."""

    extension_id = "complex-memory-mission-receipt-v1"

    def __init__(self, contract, snapshot):
        self._expected = _checked_contract(contract)
        if not callable(snapshot):
            raise TypeError("A callable physical state snapshot is required")
        self._snapshot = snapshot
        self._submission = None

    @property
    def tools(self):
        return [{"type": "function", "function": {
            "name": "submit_mission",
            "description": (
                "Record the reconstructed current dispatch after completing its route and stopping. "
                "The first format-valid receipt is locked. Feedback confirms receipt only; "
                "answer correctness and actual route are evaluated independently after finish."
            ),
            "parameters": {"type": "object", "properties": {
                "order_id": {"type": "string", "minLength": 1, "maxLength": 128,
                             "pattern": "^[A-Za-z0-9_-]+$"},
                "dispatch_revision": {"type": "integer", "minimum": 1, "maximum": 100},
                "destination": {"type": "string", "enum": ["A", "B", "C"]},
                "receipt_code": {"type": "string", "minLength": 6, "maxLength": 6,
                                 "pattern": "^[0-9]{6}$"},
                "route": {"type": "string", "enum": list(ROUTES)},
            }, "required": ["order_id", "dispatch_revision", "destination", "receipt_code", "route"],
                "additionalProperties": False},
        }}]

    @staticmethod
    def _result(call, **fields):
        return {"call_id": call.call_id, "tool": call.name, **fields}

    def execute(self, call):
        if not isinstance(call, ToolCall):
            raise TypeError("ComplexMissionReceipt requires a ToolCall")
        if call.name != "submit_mission":
            return self._result(call, status="failed", error_code="UNKNOWN_TOOL")
        if not _valid_submission(call.arguments):
            return self._result(call, status="failed", error_code="INVALID_ARGUMENTS",
                                message="Expected order ID, revision 1..100, A/B/C destination, six ASCII digits, and blue/yellow route")
        duplicate = self._submission is not None
        if duplicate and self._submission != call.arguments:
            return self._result(call, status="failed", error_code="RECEIPT_ALREADY_SUBMITTED",
                                message="The first format-valid receipt is locked")
        if not duplicate:
            self._submission = deepcopy(call.arguments)
        result = self._result(call, status="succeeded", submitted=True,
                              message="Receipt recorded. Independent evaluation runs after finish.")
        if duplicate:
            result["deduplicated"] = True
        return result

    def evaluate(self):
        """Called after the episode; none of these verdicts is an online reply."""
        c = self._expected
        submitted = self._submission is not None
        actual = self._submission or {}
        destination = c["alias_to_destination"][c["new_alias"]]
        code = actual.get("receipt_code", "")
        matches = {
            "order_id_match": bool(submitted and actual["order_id"] == c["order_id"]),
            "revision_match": bool(submitted and actual["dispatch_revision"] == c["new_revision"]),
            "destination_match": bool(submitted and actual["destination"] == destination),
            "code_match": bool(submitted and code == c["receipt_code"]),
            "prefix_match": bool(submitted and code[:3] == c["code_prefix"]),
            "suffix_match": bool(submitted and code[3:] == c["code_suffix"]),
            "route_match": bool(submitted and actual["route"] == c["new_route"]),
        }
        state = self._snapshot()
        route_evaluation = evaluate_route(state.trajectory, c["new_route"])
        if not isinstance(route_evaluation, dict) or type(route_evaluation.get("success")) is not bool:
            raise TypeError("Route evaluator must return a dictionary with native bool success")
        route_success = route_evaluation["success"]
        return {
            "extension_id": self.extension_id,
            "success": bool(submitted and all(matches.values()) and route_success),
            "submitted": submitted, **matches,
            "route_success": route_success, "route_evaluation": deepcopy(route_evaluation),
            "old_revision_selected": bool(submitted and actual["dispatch_revision"] == c["old_revision"]),
            "old_destination_selected": bool(submitted and actual["destination"] == c["alias_to_destination"][c["old_alias"]]),
            "old_route_selected": bool(submitted and actual["route"] == c["old_route"]),
            "submission": deepcopy(self._submission),
        }
