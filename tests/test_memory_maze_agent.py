"""Model protocol checks with mocked HTTP only; no simulator or paid calls."""
import base64
from io import BytesIO
import json

import httpx
import numpy as np
from PIL import Image
import pytest

from go2_demo.memory_maze_agent import MazeClient, tool_schema
from go2_demo.models import ModelError, ModelProtocolError
from go2_demo.types import ToolCall


TOOLS = [tool_schema("act", "Perform one public action.", {"action": {"type": "integer"}})]


def reply(index=0, *, names=("act",)):
    return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
        "content": "Inspect the next view.",
        "tool_calls": [{"id": f"call-{index}-{j}", "type": "function", "function": {
            "name": name, "arguments": json.dumps({"action": 2})}} for j, name in enumerate(names)]}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}})


def client(handler, *, retain=True):
    return MazeClient("https://api.deepseek.com", "deepseek-flash", "",
                      system_prompt="Use public observations only.", retain_context=retain,
                      history_turns=0, transport=httpx.MockTransport(handler), trust_env=False)


def test_full_image_and_tool_history_is_retained_without_history_turn_limit():
    packets = []

    def handler(request):
        packets.append(json.loads(request.content))
        return reply(len(packets))

    model = client(handler)
    for index in range(9):
        image = np.full((3, 4, 3), index, dtype=np.uint8)
        public = {"step": index, "memory_state": {"public_note": "not a Go2 repair request"}}
        call = model.choose(public, image, TOOLS)
        assert model.last_decision.calls == [call]
        model.finish(call, {"status": "ok", "step": index})
        public["step"] = -1
        image[:] = 255
    messages = packets[-1]["messages"]
    assert len(messages) == 26  # system + eight complete triples + current user
    images = [part["image_url"]["url"] for msg in messages if isinstance(msg["content"], list)
              for part in msg["content"] if part["type"] == "image_url"]
    assert len(images) == 9
    for index, url in enumerate(images):
        decoded = np.asarray(Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))))
        assert (decoded == index).all()
    for message in messages:
        if message["role"] == "assistant":
            ids = {c["id"] for c in message["tool_calls"]}
            assert ids <= {m.get("tool_call_id") for m in messages if m["role"] == "tool"}
    assert model.context_stats() == {"turns": 9, "images": 9, "messages": 27}
    assert model.raw_usage == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    assert packets[0]["thinking"] == {"type": "disabled"}
    assert packets[0]["temperature"] == 0
    assert packets[0]["tool_choice"] == "required" and packets[0]["parallel_tool_calls"] is False


@pytest.mark.parametrize("failure", ["malformed", "multiple", "unknown", "transport"])
def test_failed_turn_preserves_observation_and_pairs_every_parsed_tool(failure):
    packets = []

    def handler(request):
        packets.append(json.loads(request.content))
        if len(packets) > 1:
            return reply(2)
        if failure == "transport":
            raise httpx.ConnectError("mock connection failed", request=request)
        if failure == "malformed":
            return httpx.Response(200, content=b"not valid JSON")
        return reply(1, names=("act", "act") if failure == "multiple" else ("missing",))

    model = client(handler)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ModelError):
        model.choose({"step": 1}, image, TOOLS)
    assert model.context_stats()["turns"] == 1
    assert model.context_stats()["images"] == 1
    if failure in {"multiple", "unknown"}:
        assert model.raw_usage["total_tokens"] == 15
    call = model.choose({"step": 2}, image, TOOLS)
    messages = packets[-1]["messages"]
    tools = [m for m in messages if m["role"] == "tool"]
    calls = [c for m in messages if m["role"] == "assistant" for c in m["tool_calls"]]
    assert len(tools) == len(calls)
    assert {m["tool_call_id"] for m in tools} == {c["id"] for c in calls}
    assert all(json.loads(m["content"])["executed"] is False for m in tools)
    assert json.loads(messages[1]["content"][0]["text"]) == {"step": 1}
    model.finish(call, {"status": "ok"})


def test_stateless_reader_sends_only_current_text_request():
    packets = []

    def handler(request):
        packets.append(json.loads(request.content))
        return reply(len(packets))

    model = client(handler, retain=False)
    call = model.choose({"question": "first"}, None, TOOLS)
    model.finish(call, {"answer": "first feedback"})
    model.choose({"question": "second"}, None, TOOLS)
    model.choose({"question": "third"}, None, TOOLS)
    assert all(len(p["messages"]) == 2 for p in packets)
    assert json.loads(packets[-1]["messages"][-1]["content"]) == {"question": "third"}
    assert model.context_stats() == {"turns": 0, "images": 0, "messages": 0}


def test_feedback_is_atomic_and_must_match_the_pending_call():
    model = client(lambda request: reply())
    call = model.choose({"step": 0}, None, TOOLS)
    with pytest.raises(ModelProtocolError, match="pending"):
        model.choose({"step": 1}, None, TOOLS)
    with pytest.raises(ModelProtocolError, match="match"):
        model.finish(ToolCall("different-id", call.name, call.arguments), {"status": "ok"})
    with pytest.raises(ModelProtocolError, match="finite"):
        model.finish(call, {"distance": float("nan")})
    assert model.context_stats() == {"turns": 0, "images": 0, "messages": 0}
    model.finish(call, {"status": "ok"})
    assert model.context_stats() == {"turns": 1, "images": 0, "messages": 3}


def test_tool_schema_keeps_optional_fields_and_is_independent_of_input():
    properties = {"action": {"type": "integer"}, "note": {"type": "string"}}
    all_required = tool_schema("act", "Act.", properties)
    selected_required = tool_schema("act", "Act.", properties, required=["action"])
    properties["action"]["type"] = "string"
    assert all_required["function"]["parameters"]["required"] == ["action", "note"]
    assert selected_required["function"]["parameters"]["required"] == ["action"]
    assert selected_required["function"]["parameters"]["properties"]["action"]["type"] == "integer"
