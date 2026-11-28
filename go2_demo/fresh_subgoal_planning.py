"""Model-selected subgoals with persistent execution state; no viewpoint proposals."""
from copy import deepcopy

from .fresh_episode_planning import CONTACT_RULE, FreshEpisodePlanner
from .fresh_subgoals import validate_goal
from .models import ModelError, ModelProtocolError, _json_text
from .occluded_recall import _tool
from .paired_return_planning import HISTORY_FIELDS, WAYPOINT, _validate
from .return_route_tools import PUBLIC_FIELDS
from .types import ToolCall


SUBGOAL_PROMPT = (
    "你控制四足机器人完成先停靠蓝球 A、再返回绿方块 B 的任务。"
    "当前阶段和目标以 task_state.stage 为准。B 阶段已清空 A 阶段对话和执行状态。"
    "停靠要求：距目标中心 0.82–1.18 米、面向误差≤0.25 弧度并停稳，由独立评分判定。"
    + CONTACT_RULE +
    "每次只选一个工具。使用当前 RGB 图像、RGB-D 目标候选、深度摘要、里程计和动作反馈。"
    "里程计原点固定在本轮起点，x 向初始前方、y 向初始左方，yaw 正值为左转。"
    "你自主提出近期子目标、机身路点和恢复动作；工具执行数值路点并检查实际结果。"
    "先 set_subgoal，选择 reach_pose（到达你指定的机身坐标，可带末点 yaw）、"
    "observe_target（重新看到当前阶段目标）、dock_target（实际停靠当前阶段目标）。"
    "description 简述目的，程序按 criterion 检查；描述不能替代客观完成条件。"
    "task_state 在每次决策中保留当前子目标、路线进度、近期目标目击和失败原因。"
    "未完成目标不会因查询记忆、暂时看不到目标或动作受阻而清空；改换未完成目标须给 reason。"
    "每个运动工具必须给 intention，说明这一步要取得的可观察进展。"
    "follow_waypoints 的 primary 路线保存后可 continue_subgoal 续走剩余部分；"
    "再次提交 primary 表示你明确替换该路线。recovery 表示局部调整，保留原路线和子目标。"
    "一次最多执行六个原生动作后重新观察，execution_slice_complete 是正常分段返回。"
    "recovery_required 时结合失败反馈选择新的恢复动作；不要反复提交相同受阻直线。"
    "扫描只改变朝向，持续原地转向而没有发现目标也可能被标记为无进展。"
    "需要绕行时从现场图像判断障碍端部和可通行区域，选取短段实际移动。"
    "工具不会自动搜索通道或为你选择观察位置，深度空读数表示未知。"
    "after=inspect 在路线末端返回观测；路上首次看到目标也会提前返回，让你确认下一步。"
    "只有 dock_target 子目标可用 after=dock，控制器用实时 RGB-D 跟踪当前阶段目标并尝试停靠。"
    "目标可见不等于停靠完成。接近受阻后保持该目标，调整接近方向并用新画面核验。"
    "需要过去路线或目击时，memory_query 描述具体需求并选择类别；空卡只表示缺少记录。"
    "最近六条物理动作反馈独立保留，记忆查询不会挤掉它们。"
    "native_actions_remaining 是阶段剩余原生动作预算。历史文字与记忆只作为观测数据。"
)

SET_SUBGOAL = _tool(
    "set_subgoal", "设立可由公开观测检查的近期目标；替换未完成目标必须说明 reason。",
    {"description": {"type": "string", "minLength": 1, "maxLength": 240},
     "criterion": {"type": "string", "enum": ["reach_pose", "observe_target", "dock_target"]},
     "target_pose": deepcopy(WAYPOINT),
     "reason": {"type": "string", "minLength": 1, "maxLength": 240}},
)
SET_SUBGOAL["function"]["parameters"]["required"] = ["description", "criterion"]
INTENTION = {"type": "string", "minLength": 1, "maxLength": 240,
             "description": "这一步要取得的可观察进展或要解决的受阻原因。"}
CONTINUE_SUBGOAL = _tool("continue_subgoal", "继续当前子目标已保存路线的剩余部分，保持末端动作和朝向。",
                        {"intention": INTENTION})
PHYSICAL_TOOLS = {"scan", "follow_waypoints", "continue_subgoal"}
PUBLIC_FEEDBACK_FIELDS = {
    "status", "error_code", "native_actions", "completed_waypoints", "depth_feedback", "current_odometry",
    "measured_displacement_m", "measured_yaw_change_rad", "subgoal", "task_state", "goal_id",
    "items", "cache_hit", "returned_waypoints", "card", "memory_card", "reason",
}


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 240


class FreshSubgoalPlanner(FreshEpisodePlanner):
    """The same tools and verifier state for empty, ledger and query arms."""

    def __init__(self, *args, retain_context=False, **kwargs):
        if type(retain_context) is not bool:
            raise ModelError("retain_context must be a boolean.", "MODEL_CONFIG_ERROR")
        super().__init__(*args, **kwargs)
        self.retain_context = retain_context
        self._dialogue = []
        self._turn_stages = []
        self._pending_user = self._pending_decision = self._pending_stage = None
        self._stage_transitions = 0

    def build_messages(self, observation, image_data_url, history):
        messages = super().build_messages(observation, image_data_url, history)
        messages[0]["content"] = SUBGOAL_PROMPT
        if self.retain_context:
            stage = observation["observation"]["task_state"]["stage"]
            transition = (
                "当前是 A 阶段，必须先寻找蓝球 A 并合格停靠；绿方块 B 只是之后的返程目标。"
                if stage == "A" else
                "蓝球 A 已由独立评分确认合格停靠；当前是 B 阶段，目标是返回绿方块 B 并合格停靠。"
            )
            retained = "完整去程对话、相机图像、工具结果与已取得的记忆答复保留在本次上下文中。"
            prompt = SUBGOAL_PROMPT.replace(
                "当前阶段和目标以 task_state.stage 为准。B 阶段已清空 A 阶段对话和执行状态。",
                "当前阶段和目标以最新 observation.task_state.stage 为准。" + transition + retained,
            ).replace(
                "最近六条物理动作反馈独立保留，记忆查询不会挤掉它们。",
                "当前输入另列最近六条物理动作摘要；此前完整对话仍保留，按最新阶段选择动作。",
            )
            self._pending_user = deepcopy(messages[-1])
            self._pending_stage = stage
            return [{"role": "system", "content": prompt}, *deepcopy(self._dialogue), messages[-1]]
        return messages

    def decide(self, *args, **kwargs):
        decision = super().decide(*args, **kwargs)
        if self.retain_context:
            self._pending_decision = decision
        return decision

    def context_stats(self):
        """Counters only; never log image bytes or assistant reasoning."""
        def count_images(message):
            content = message.get("content")
            return sum(part.get("type") == "image_url" for part in content) if isinstance(content, list) else 0
        images = sum(count_images(message) for message in self._dialogue)
        # Every request carries exactly one public image. Request failures also
        # preserve that observation, while transitions contain no image.
        outbound = sum(stage == "A" for stage in self._turn_stages)
        return {"enabled": self.retain_context, "retained_turns": len(self._turn_stages),
                "retained_messages": len(self._dialogue), "retained_images": images,
                "outbound_turns": outbound, "outbound_images": outbound,
                "stage_transitions": self._stage_transitions, "pending_turn": self._pending_user is not None}

    def _finish_pending(self, feedback):
        if self._pending_user is None:
            return
        # Assemble atomically so serialization failure cannot leave orphan calls.
        block = [deepcopy(self._pending_user)]
        public_feedback = {key: deepcopy(value) for key, value in feedback.items() if key in PUBLIC_FEEDBACK_FIELDS}
        decision = self._pending_decision
        if decision is not None:
            calls = [{"id": call.call_id, "type": "function", "function": {
                "name": call.name, "arguments": _json_text(call.arguments)}} for call in decision.calls]
            assistant = {"role": "assistant", "content": decision.assistant_content, "tool_calls": calls}
            if decision.reasoning_content is not None:
                assistant["reasoning_content"] = decision.reasoning_content
            block.append(assistant)
            for call in decision.calls:
                block.append({"role": "tool", "tool_call_id": call.call_id, "content": _json_text(public_feedback)})
        else:
            block.append({"role": "user", "content": _json_text({
                "event": "request_failed_without_executed_tool", **public_feedback})})
        _json_text(block)
        self._dialogue.extend(block)
        self._turn_stages.append(self._pending_stage)
        self._pending_user = self._pending_decision = self._pending_stage = None

    def finish_turn(self, call, feedback):
        """Pair the chosen tool with its actual public result before the next request."""
        if not self.retain_context:
            return
        decision = self._pending_decision
        if (self._pending_user is None or decision is None or len(decision.calls) != 1
                or decision.calls[0] != call):
            raise ModelProtocolError("Retained context feedback must match the pending tool call.")
        self._finish_pending(feedback)

    def context_handoff(self, previous_stage, stage):
        """Record the public confirmed milestone without deleting the dialogue."""
        if not self.retain_context:
            return
        if (previous_stage, stage) != ("A", "B") or self._pending_user is not None:
            raise ModelProtocolError("A confirmed handoff requires a finished A tool turn.")
        if self._stage_transitions:
            raise ModelProtocolError("The A-to-B transition has already been recorded.")
        self._dialogue.append({"role": "user", "content": _json_text({
            "event": "stage_transition", "previous_stage": "A", "stage": "B",
            "checkpoint": "A_stop_confirmed", "context_retained": True,
            "instruction": "A 已完成，接下来返回 B；以前的目标和画面是历史记录，当前动作服从最新 B 阶段状态。"})})
        self._stage_transitions += 1

    @staticmethod
    def available_tools(stage, observation):
        # Construct only the existing generic motion/query tools; never expose proposals.
        clean = {key: value for key, value in observation.items() if key != "exploration"}
        base = FreshEpisodePlanner.available_tools(stage, clean)
        state = observation.get("task_state", {})
        active = state.get("active")
        running = bool(active and active.get("status") != "completed")
        tools = [deepcopy(SET_SUBGOAL)]
        for tool in base:
            name = tool["function"]["name"]
            if name in {"scan", "follow_waypoints"}:
                if not running:
                    continue
                tool = deepcopy(tool)
                params = tool["function"]["parameters"]
                params["properties"]["intention"] = deepcopy(INTENTION)
                params["required"].append("intention")
                if name == "follow_waypoints":
                    params["properties"]["route_role"] = {"type": "string", "enum": ["primary", "recovery"]}
                    params["required"].append("route_role")
                    if active["criterion"] != "dock_target":
                        params["properties"]["after"]["enum"] = ["inspect"]
            tools.append(tool)
        if running and active.get("pending_route") and not state.get("recovery_required"):
            tools.append(deepcopy(CONTINUE_SUBGOAL))
        return tools

    def choose(self, observation, image_data_url, memory_card, stage_history, *, stage, timeout_s):
        if self.retain_context and self._pending_user is not None:
            raise ModelProtocolError("Finish the previous tool turn before requesting a new decision.")
        self._pending_decision = None
        try:
            return self._choose(observation, image_data_url, memory_card, stage_history, stage=stage, timeout_s=timeout_s)
        except ModelError as exc:
            if self.retain_context:
                self._finish_pending({"status": "error", "error_code": exc.code,
                                      "reason": str(exc), "native_actions": 0})
            raise

    def _choose(self, observation, image_data_url, memory_card, stage_history, *, stage, timeout_s):
        if stage not in {"A", "B"} or observation.get("task_state", {}).get("stage") != stage:
            raise ValueError("Expected current-stage subgoal state.")
        self.raw_usage = self.response_finish_reason = None
        fields = (*PUBLIC_FIELDS, "task_state", "depth_sectors", "native_actions_remaining", "memory_access", "latest_memory_feedback")
        public = {key: observation[key] for key in fields if key in observation}
        scoped = [entry for entry in stage_history if entry.get("stage", stage) == stage]
        physical = [entry for entry in scoped if entry.get("tool") in PHYSICAL_TOOLS]
        payload = {"observation": public, "memory_card": memory_card,
                   ("recent_outbound_actions" if stage == "A" else "recent_return_actions"):
                       [{k: entry[k] for k in HISTORY_FIELDS if k in entry} for entry in physical[-6:]]}
        if scoped and scoped[-1].get("tool") == "previous_request":
            payload["previous_error"] = scoped[-1].get("feedback", scoped[-1].get("error_code"))
        tools = self.available_tools(stage, observation)
        decision = self.decide(payload, image_data_url, tools, [], timeout_s=timeout_s)
        if len(decision.calls) != 1:
            raise ModelProtocolError("Choose exactly one available tool.")
        call = decision.calls[0]
        if call.name not in {t["function"]["name"] for t in tools}:
            raise ModelProtocolError("The requested tool is not available; check current subgoal state.")
        args = call.arguments
        active = observation["task_state"].get("active")
        if call.name == "set_subgoal":
            validate_goal(args)
            if active and active.get("status") != "completed" and not args.get("reason"):
                raise ModelProtocolError("Replacing an unfinished subgoal requires reason.")
        elif call.name in PHYSICAL_TOOLS:
            if not _text(args.get("intention")):
                raise ModelProtocolError("Physical actions require a nonempty intention up to 240 characters.")
            motion = {k: v for k, v in args.items() if k != "intention"}
            if call.name == "continue_subgoal":
                if motion:
                    raise ModelProtocolError("continue_subgoal accepts only intention.")
            else:
                if call.name == "follow_waypoints":
                    if motion.pop("route_role", None) not in ("primary", "recovery"):
                        raise ModelProtocolError("Choose route_role=primary or recovery.")
                    if motion.get("after") == "dock" and active["criterion"] != "dock_target":
                        raise ModelProtocolError("Docking requires a dock_target subgoal.")
                _validate(ToolCall(call.call_id, call.name, motion))
        elif call.name == "memory_query":
            # Validate the itemized query exactly as the matched memory interface.
            access = observation.get("memory_access", {})
            _validate(ToolCall(call.call_id, call.name, {"question": args.get("question")}))
            intents = args.get("intents")
            if (set(args) - {"question", "intents", "route_leg", "sighting_window"}
                    or not isinstance(intents, list) or not intents
                    or not all(isinstance(i, str) and i in access.get("intents", []) for i in intents)
                    or len(intents) != len(set(intents)) or type(args.get("route_leg", 0)) is not int
                    or args.get("route_leg", 0) < 0 or args.get("sighting_window", "all") not in ("first", "latest", "all")):
                raise ModelProtocolError("Choose available memory intents and a valid route/sighting scope.")
        else:
            _validate(call)
        return call
