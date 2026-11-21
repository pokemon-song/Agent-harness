import json
import time

import httpx
import pytest

from go2_demo.return_route_tools import RouteModel, plan_return_route


IMG = "data:image/png;base64,cHVibGlj"
ODOM = {"translation_m": [2.1, .9], "yaw_rad": -.1}
OBS = {"local_odometry": ODOM, "secret_evaluator": "hidden_state"}
DOCS = {
    "route/leg-01.md": """
[E001] 去程步 0 / frame-00001 / t=1.00s：实测机身里程计 x=0.000, y=0.000 m，yaw=0.000 rad；同一去程起点坐标系，x 向前、y 向左。
[E002] 去程步 5 / frame-00002 / t=2.00s：实测机身里程计 x=1.000, y=0.500 m，yaw=-0.470 rad；同一去程起点坐标系，x 向前、y 向左。
[E009] 这是一段没有实测机身几何的记录。
""",
    "route/leg-02.md": """
[E003] 去程步 20 / frame-00010 / t=5.00s：实测机身里程计 x=2.100, y=0.900 m，yaw=-0.100 rad；同一去程起点坐标系，x 向前、y 向左。
""",
    "targets/target-B.md": "[E004] 去程步 0 / frame-00001 / t=1.00s：目标 B（绿方块）同色区域上的有效 RGB-D 跟踪点 point-001，机身前/左=(1.7,0.7) m。",
    "handoff/last-observation.md": "[E005] Never send this document to the planning agent.",
}
ASK = ("memory_query", {"question": "请给我按返程方向排列的完整机身路点、目标 B 目击。"})
READ = ("select_route_evidence", {"route_ids": ["E003", "E002", "E001"], "target_id": "E004"})
PLAN = ("follow_route", {"route_ids": ["E002", "E001"], "target_label": "B", "target_id": "E004"})
ABORT = ("abort", {"reason": "没有充分的来源。"})


def invoke(replies, **kwargs):
    payloads = []
    queue = iter(replies)

    def handler(request):
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        payloads.append(json.loads(request.content))
        name, args = next(queue)
        call = {"id": "call-" + str(len(payloads)), "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}
        response = {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [call]}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
        return httpx.Response(200, json=response)

    def client():
        return RouteModel("https://api.deepseek.com", "deepseek-flash", "", transport=httpx.MockTransport(handler))

    params = {"timeout_s": 10, "deadline": time.monotonic() + 60, **kwargs}
    events = []
    state = plan_return_route(client(), client(), OBS, IMG, DOCS, emit=events.append, **params)
    assert events == state["events"]
    return state, payloads


def test_real_wire_model_choose_route_card_and_planner_subset_without_hidden_state():
    state, payloads = invoke([ASK, READ, PLAN])
    assert state["status"] == "follow_route" and state["target_label"] == "B" and state["error"] is None
    assert state["waypoints"] == [{"id": "E002", "x": 1., "y": .5, "yaw": -.47},
                                  {"id": "E001", "x": 0., "y": 0., "yaw": 0.}]
    assert state["target_evidence"]["id"] == "E004" and state["target_evidence"]["path"] == "targets/target-B.md"
    assert state["requests"] == 3 and state["known_usage"]["total_tokens"] == 30
    assert state["unknown_usage_requests"] == 0
    assert [e["calls"][0]["arguments"] for e in state["events"]] == [ASK[1], READ[1], PLAN[1]]
    assert [e["stage"] for e in state["events"]] == ["ask", "read", "plan"]
    for payload in payloads:
        assert payload["thinking"] == {"type": "disabled"} and payload["tool_choice"] == "required"
        assert len(payload["messages"]) == 2 and "hidden_state" not in json.dumps(payload)
    assert json.loads(payloads[1]["messages"][1]["content"]) == {"question": ASK[1]["question"], "documents": DOCS}
    for i in (0, 2):
        text, image = payloads[i]["messages"][1]["content"]
        assert image["image_url"]["url"] == IMG
        assert "Never send this document" not in text["text"]
        assert json.loads(text["text"])["observation"]["local_odometry"] == ODOM
    card = json.loads(payloads[2]["messages"][1]["content"][0]["text"])["memory_card"]
    assert [row["id"] for row in card["route"]] == READ[1]["route_ids"]


def test_reader_and_planner_orders_are_preserved_even_if_different():
    choice = ("follow_route", {**PLAN[1], "route_ids": ["E001", "E003", "E002"]})
    state, _ = invoke([ASK, READ, choice])
    assert [p["id"] for p in state["waypoints"]] == choice[1]["route_ids"]


@pytest.mark.parametrize("invalid", [
    {"route_ids": ["E009"], "target_id": "E004"},
    {"route_ids": ["E002"], "target_id": "E005"},
    {"route_ids": ["E004"], "target_id": "E002"},
    {"route_ids": ["E001", "E001"], "target_id": "E004"},
    {"route_ids": ["E001"] * 33, "target_id": "E004"},
])
def test_bad_memory_citation_requires_a_new_reader_decision(invalid):
    state, payloads = invoke([ASK, (READ[0], invalid), READ, PLAN])
    assert state["status"] == "follow_route" and state["requests"] == 4
    assert state["events"][1]["error"] == "INVALID_ROUTE_TOOL"
    assert "previous_error" in json.loads(payloads[2]["messages"][1]["content"])


def test_planner_cannot_recover_an_uncited_route_id_from_the_source_archive():
    read = (READ[0], {**READ[1], "route_ids": ["E002"]})
    state, _ = invoke([ASK, read, PLAN, PLAN])
    assert state["status"] == "error" and state["error"] == "INVALID_ROUTE_TOOL"
    assert state["requests"] == 4 and state["known_usage"]["total_tokens"] == 40
    assert not state["waypoints"] and state["target_evidence"] is None


def test_abort_owns_its_decision_and_exhausted_budget_never_calls_the_api():
    state, _ = invoke([ASK, READ, ABORT])
    assert state["status"] == "abort" and state["error"] == ABORT[1]["reason"]
    state, captures = invoke([], max_requests=0)
    assert state["error"] == "MODEL_REQUEST_BUDGET" and not captures
    state, captures = invoke([], deadline=time.monotonic() - 1)
    assert state["error"] == "MODEL_TIMEOUT" and not captures
