"""Keep complete public dialogue across A/B without making live API requests."""
import json

import httpx
import pytest

from go2_demo.fresh_subgoal_planning import FreshSubgoalPlanner, SUBGOAL_PROMPT
from go2_demo.models import ModelError, ModelProtocolError
from go2_demo.types import ToolCall


def obs(stage, *, active=True):
    return {"local_odometry": {"translation_m": [0., 0.], "yaw_rad": 0.},
            "task_state": {"stage": stage, "active": {"id": stage + "-1", "status": "active",
                "criterion": "observe_target", "pending_route": None} if active else None},
            "memory_access": {"allowed": False, "intents": []},
            "private_evaluator": "DO_NOT_SEND_PRIVATE_STATE"}


def frame(index):
    return "data:image/png;base64,cHVibGlj" + str(index)


def tool(name="scan", args=None, index=0):
    return {"id": "call-" + str(index), "type": "function", "function": {
        "name": name, "arguments": json.dumps(args or {"yaw_rad": .2, "intention": "查看当前目标"})}}


def response(calls):
    return httpx.Response(200, json={"choices": [{"message": {"content": "继续检查当前阶段",
        "tool_calls": calls}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}})


@pytest.fixture
def client():
    created = []

    def make(handler, *, retain=True):
        model = FreshSubgoalPlanner("https://api.deepseek.com", "deepseek-flash", "",
                                   transport=httpx.MockTransport(handler), retain_context=retain)
        created.append(model)
        return model

    yield make
    for model in created:
        model.close()


def test_first_and_later_return_requests_keep_every_outbound_image_action_and_result(client):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        i = len(requests)
        if i == 9:
            return response([tool("set_subgoal", {"description": "返回 B", "criterion": "observe_target"}, i)])
        return response([tool(index=i)])

    model = client(handler)
    for i in range(8):
        call = model.choose(obs("A"), frame(i), None, [], stage="A", timeout_s=2)
        model.finish_turn(call, {"status": "handoff" if i == 7 else "ok",
                                "error_code": "A_stop_confirmed" if i == 7 else None,
                                "reason": "observed-response-" + str(i),
                                "private_evaluator": "DO_NOT_SEND_PRIVATE_RESULT"})
    model.context_handoff("A", "B")
    for i in range(2):
        call = model.choose(obs("B", active=bool(i)), frame(8 + i), {"waypoints": []}, [], stage="B", timeout_s=2)
        model.finish_turn(call, {"status": "ok", "memory_card": {"source": "public selected evidence"}})

    for packet in requests[8:]:
        text = json.dumps(packet, ensure_ascii=False)
        assert "当前是 B 阶段" in packet["messages"][0]["content"]
        assert "B 阶段已清空" not in packet["messages"][0]["content"]
        assert sum(m["role"] == "system" for m in packet["messages"]) == 1
        for i in range(8):
            assert frame(i) in text and "observed-response-" + str(i) in text
            assert '"id": "call-' + str(i + 1) + '"' in text
        assert "stage_transition" in text and "A_stop_confirmed" in text
        assert "DO_NOT_SEND_PRIVATE" not in text
    assert model.context_stats() == {"enabled": True, "retained_turns": 10, "retained_messages": 31,
                                    "retained_images": 10, "outbound_turns": 8, "outbound_images": 8,
                                    "stage_transitions": 1, "pending_turn": False}
    assert "public selected evidence" in json.dumps(requests[9])


def test_current_stage_is_explicit_and_old_tool_calls_are_not_reexecuted(client):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response([tool(index=len(requests))])

    model = client(handler)
    a = model.choose(obs("A"), frame(0), None, [], stage="A", timeout_s=2)
    assert "当前是 A 阶段" in requests[0]["messages"][0]["content"]
    model.finish_turn(a, {"status": "handoff", "error_code": "A_stop_confirmed"})
    model.context_handoff("A", "B")
    b = model.choose(obs("B"), frame(1), None, [], stage="B", timeout_s=2)
    assert b.call_id != a.call_id
    current = json.loads(requests[1]["messages"][-1]["content"][0]["text"])
    assert current["observation"]["task_state"]["stage"] == "B"
    model.finish_turn(b, {"status": "ok"})


def test_reset_mode_retains_original_stateless_payload(client):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response([tool(index=len(requests))])

    model = client(handler, retain=False)
    call = model.choose(obs("A"), frame(0), None, [], stage="A", timeout_s=2)
    model.finish_turn(call, {"status": "handoff", "error_code": "A_stop_confirmed"})
    model.context_handoff("A", "B")
    model.choose(obs("B"), frame(1), None, [], stage="B", timeout_s=2)
    assert len(requests[1]["messages"]) == 2
    assert requests[1]["messages"][0]["content"] == SUBGOAL_PROMPT
    assert frame(0) not in json.dumps(requests[1])
    assert model.context_stats()["retained_turns"] == 0


def test_schema_rejection_is_retained_with_error_result_and_no_dangling_call(client):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response([tool(args={"yaw_rad": 99., "intention": "错误幅度"}, index=1)])
        return response([tool(index=2)])

    model = client(handler)
    with pytest.raises(ModelProtocolError):
        model.choose(obs("A"), frame(0), None, [], stage="A", timeout_s=2)
    assert not model.context_stats()["pending_turn"]
    call = model.choose(obs("A"), frame(1), None, [], stage="A", timeout_s=2)
    messages = requests[1]["messages"]
    failed = next(m for m in messages if m.get("tool_call_id") == "call-1")
    assert json.loads(failed["content"])["error_code"] == "MODEL_PROTOCOL_ERROR"
    assert json.loads(failed["content"])["native_actions"] == 0
    model.finish_turn(call, {"status": "ok"})


@pytest.mark.parametrize("failure", ["http", "malformed", "multiple"])
def test_failed_requests_remain_recoverable_and_history_stays_well_formed(client, failure):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            if failure == "http":
                return httpx.Response(400)
            if failure == "malformed":
                return httpx.Response(200, text="invalid JSON")
            return response([tool(index=1), tool(index=2)])
        return response([tool(index=3)])

    model = client(handler)
    with pytest.raises(ModelError):
        model.choose(obs("A"), frame(0), None, [], stage="A", timeout_s=2)
    call = model.choose(obs("A"), frame(1), None, [], stage="A", timeout_s=2)
    messages = requests[1]["messages"]
    assert frame(0) in json.dumps(messages)
    pending = set()
    for message in messages:
        if message["role"] == "assistant":
            pending.update(call["id"] for call in message.get("tool_calls", []))
        elif message["role"] == "tool":
            assert message["tool_call_id"] in pending
            pending.remove(message["tool_call_id"])
    assert not pending
    model.finish_turn(call, {"status": "ok"})


def test_turn_pairing_and_handoff_are_checked_without_discarding_pending_data(client):
    model = client(lambda request: response([tool(index=1)]))
    call = model.choose(obs("A"), frame(0), None, [], stage="A", timeout_s=2)
    with pytest.raises(ModelProtocolError):
        model.choose(obs("A"), frame(1), None, [], stage="A", timeout_s=2)
    with pytest.raises(ModelProtocolError):
        model.context_handoff("A", "B")
    with pytest.raises(ModelProtocolError):
        model.finish_turn(ToolCall("another-call", call.name, call.arguments), {"status": "ok"})
    model.finish_turn(call, {"status": "handoff", "error_code": "A_stop_confirmed"})
    model.context_handoff("A", "B")
    with pytest.raises(ModelProtocolError):
        model.context_handoff("A", "B")
    assert model.context_stats()["retained_turns"] == 1
