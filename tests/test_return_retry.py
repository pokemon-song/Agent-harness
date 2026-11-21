import json

import httpx
import pytest

from go2_demo.harness import TOOLS
from go2_demo.models import ModelError, ModelProtocolError
from go2_demo.occluded_return import ReturnPlanner, TASK
from go2_demo.return_retry import decide_after_length


IMAGE = "data:image/png;base64,ZnJhbWU="
USAGE = {"prompt_tokens": 100, "completion_tokens": 9, "total_tokens": 109}
QUERY = {"type": "function", "function": {"name": "memory_query", "parameters": {"type": "object"}}}
OBS = {
    "task": TASK, "return_memory": {"current": {"question": "B?", "evidence": []}},
    "image_id": "image-4", "local_odometry": {"translation_m": [1, 2], "yaw_rad": 0.1},
    "target_candidates": [], "completion_requirements": {"stop_required": True},
    "private_target": "undisclosed",
}


def turn(index):
    return {
        "observation": {"previous_image": "omitted"},
        "reasoning_content": "previous private reasoning",
        "assistant_content": "previous free text",
        "calls": [{"call_id": str(index), "name": "turn", "arguments": {"yaw_rad": .1 * index}}],
        "results": [{"call_id": str(index), "tool": "turn", "status": "succeeded",
                     "measured": {"yaw_delta_rad": index / 11}}],
    }


def wire_call(name="stop", index=1):
    return {"id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": "{}"}}


def planner_with_reply(calls, *, reason="tool_calls"):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        response = {"choices": [{"message": {"tool_calls": calls}, "finish_reason": reason}], "usage": USAGE}
        return httpx.Response(200, json=response)

    planner = ReturnPlanner("https://api.deepseek.com/v1", "deepseek-flash", "",
                            transport=httpx.MockTransport(handler), max_output_tokens=8192)
    return planner, requests


def test_wire_is_single_nonthinking_step_with_public_context_and_full_tools():
    planner, requests = planner_with_reply([wire_call()])
    tools = [*TOOLS, QUERY]
    decision = decide_after_length(planner, OBS, IMAGE, tools, [turn(i) for i in range(4)], timeout_s=3)
    assert decision.calls[0].name == "stop" and decision.reasoning_content is None
    assert planner.raw_usage == USAGE and planner.response_finish_reason == "tool_calls"
    assert len(requests) == 1
    payload = requests[0]
    assert payload["model"] == "deepseek-flash" and payload["tools"] == tools
    assert payload["max_tokens"] == 8192 and payload["thinking"] == {"type": "disabled"}
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    text, image = payload["messages"][-1]["content"]
    assert image["image_url"]["url"] == IMAGE
    state = json.loads(text["text"])
    assert all(state[key] == OBS[key] for key in ("task", "return_memory", "local_odometry", "completion_requirements"))
    assert state["recent_tool_feedback"] == [
        {"tool": "turn", "arguments": {"yaw_rad": .1 * i},
         "result": {"status": "succeeded", "measured": {"yaw_delta_rad": i / 11}}} for i in range(1, 4)
    ]
    assert not any(s in text["text"] for s in ("previous private reasoning", "previous free text", "omitted", "undisclosed"))
    with pytest.raises(ModelProtocolError, match="reasoning_content"):
        planner.decide(OBS, IMAGE, tools, [], timeout_s=3)
    assert requests[-1]["thinking"] == {"type": "enabled"}


@pytest.mark.parametrize("calls,reason", [
    ([wire_call()], "length"), ([], "stop"), ([wire_call("unknown")], "tool_calls"),
    ([wire_call(), wire_call(index=2)], "tool_calls"),
])
def test_invalid_or_truncated_reply_never_dispatches_or_auto_retries(calls, reason):
    planner, requests = planner_with_reply(calls, reason=reason)
    with pytest.raises(ModelProtocolError):
        decide_after_length(planner, OBS, IMAGE, TOOLS, [], timeout_s=4)
    assert len(requests) == 1 and planner.raw_usage == USAGE and planner.response_finish_reason == reason


def test_exhausted_deadline_prevents_request_and_clears_stale_usage():
    planner, requests = planner_with_reply([wire_call()])
    planner.raw_usage, planner.response_finish_reason = USAGE, "length"
    with pytest.raises(ModelError) as exc:
        decide_after_length(planner, OBS, IMAGE, TOOLS, [], timeout_s=0)
    assert exc.value.code == "MODEL_TIMEOUT"
    assert not requests and planner.raw_usage is planner.response_finish_reason is None
