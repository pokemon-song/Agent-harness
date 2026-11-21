"""Single-step recovery for a return planner that exhausted its thinking budget."""

from copy import copy

from .models import ModelProtocolError


_PUBLIC_FIELDS = (
    "task", "image_id", "sim_time_s", "image_size", "coordinate_convention",
    "local_odometry", "robot_state", "tracked_points", "target_candidates",
    "sensor_calibration", "completion_requirements", "return_memory",
)
_FEEDBACK_FIELDS = (
    "status", "error_code", "measured", "stop", "stop_result", "speed_mps",
    "yaw_rate_radps", "collision", "constraint_feedback", "deduplicated", "cache_hit",
)


def _recent_feedback(turns):
    feedback = []
    for turn in turns[-3:]:
        # A multi-call turn is rejected as a unit by this experiment.
        calls, results = turn["calls"], turn["results"]
        if len(calls) != 1 or len(results) != 1:
            continue
        call, result = calls[0], results[0]
        if result["call_id"] != call["call_id"] or result["tool"] != call["name"]:
            raise ModelProtocolError("Return feedback did not match the original call.")
        summary = {key: result[key] for key in _FEEDBACK_FIELDS if key in result}
        if "point" in result:
            summary["point_id"] = result["point"]["point_id"]
        feedback.append({"tool": call["name"], "arguments": call["arguments"], "result": summary})
    return feedback


def decide_after_length(planner, observation, image_data_url, tools, last_results, *, timeout_s):
    """Issue one fresh RGB + tools request using the current public return state.

    The caller selects this mode after a provider length finish and counts each
    call against the shared budget. No previous reasoning is sent or fabricated.
    """
    public = {key: observation[key] for key in _PUBLIC_FIELDS if key in observation}
    public["recent_tool_feedback"] = _recent_feedback(last_results)
    public["decision_instruction"] = (
        "只决定眼前的一步。目标被遮挡时，可在去程机身路点中选取邻近的上一段路点，"
        "结合当前图像和测距确认下一步净空，再分段折返。用实测里程计更新进度。"
        "根据当前证据选择一个安全且推进任务的工具调用，随后会获得新的观测。"
    )
    # Reuse the normal parser, transport, output cap and timeout. The copy
    # limits the different wire-level thinking protocol to this one request.
    request = copy(planner)
    request.extra_body = {**planner.extra_body, "thinking": {"type": "disabled"}}
    request.extra_body.pop("reasoning_effort", None)
    request.requires_reasoning_content = False
    request.raw_usage = request.response_finish_reason = None
    try:
        decision = request.decide(public, image_data_url, tools, [], timeout_s=timeout_s)
        if len(decision.calls) != 1 or decision.calls[0].name not in {tool["function"]["name"] for tool in tools}:
            raise ModelProtocolError("Return recovery requires exactly one supplied tool.")
        return decision
    finally:
        # Preserve actual provider accounting even for rejected/truncated replies.
        planner.raw_usage = request.raw_usage
        planner.response_finish_reason = request.response_finish_reason
