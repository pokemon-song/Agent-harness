import json

import httpx
import pytest

from go2_demo.models import ModelProtocolError
from go2_demo.paired_return_planning import PairedReturnPlanner


IMAGE = "data:image/png;base64,cHVibGlj"
OBS = {"local_odometry": {"translation_m": [2., -.4], "yaw_rad": .3},
       "secret_evaluator": "PRIVATE", "return_memory": "OUTBOUND"}
ROUTE = {"waypoints": [{"x": 1.8, "y": -.25}, {"x": 1.4, "y": .1, "yaw": 0.}], "after": "inspect"}


def planner_reply(name, args):
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        call = {"id": "action", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        return httpx.Response(200, json={
            "choices": [{"message": {"tool_calls": [call]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        })

    planner = PairedReturnPlanner("https://api.deepseek.com", "deepseek-flash", "",
                                  transport=httpx.MockTransport(handler))
    return planner, payloads


def test_shared_planner_accepts_self_chosen_route_with_any_memory_card():
    planner, payloads = planner_reply("follow_waypoints", ROUTE)
    history = [{"tool": "scan", "arguments": {"yaw_rad": .5}, "status": "ok",
                "feedback": f"return-{i}", "private_outbound": "OUTBOUND"} for i in range(8)]
    cards = [None, {"waypoints": [{"x": .1, "y": .2}]}, {"target": "B", "route": []}]
    for card in cards:
        call = planner.choose(OBS, IMAGE, card, history, timeout_s=5)
        assert call.name == "follow_waypoints" and call.arguments == ROUTE
        assert planner.raw_usage["total_tokens"] == 13 and planner.response_finish_reason == "tool_calls"
    assert len(payloads) == 3
    shared = []
    for card, payload in zip(cards, payloads):
        system, user = payload.pop("messages")
        content, image = user["content"]
        body = json.loads(content["text"])
        assert body.pop("memory_card") == card
        assert [r["feedback"] for r in body["recent_return_actions"]] == [f"return-{i}" for i in range(2, 8)]
        assert body["observation"] == {"local_odometry": OBS["local_odometry"]}
        assert "OUTBOUND" not in content["text"] and "PRIVATE" not in content["text"]
        assert payload["tool_choice"] == "required" and not payload["parallel_tool_calls"]
        assert image["image_url"]["url"] == IMAGE
        shared.append((payload, system, body, image))
    assert shared[0] == shared[1] == shared[2]


@pytest.mark.parametrize("name,args", [
    ("follow_waypoints", {"waypoints": [], "after": "inspect"}),
    ("follow_waypoints", {"waypoints": [{"x": True, "y": .1}], "after": "dock"}),
    ("follow_waypoints", {"waypoints": [{"id": "E999", "x": 2., "y": 0.}], "after": "dock"}),
    ("follow_waypoints", {"waypoints": [{"x": 0., "y": 0.}]}),
    ("scan", {"yaw_rad": -.7}),
    ("memory_query", {"question": " "}),
])
def test_bad_tool_is_charged_and_not_executed(name, args):
    planner, payloads = planner_reply(name, args)
    with pytest.raises(ModelProtocolError):
        planner.choose(OBS, IMAGE, None, [], timeout_s=5)
    assert len(payloads) == 1 and planner.raw_usage["total_tokens"] == 13
