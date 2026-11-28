"""Offline stage handoffs through the real planner protocol and motion runner."""
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from go2_demo.fresh_episode_memory import FreshEpisodeMemory
from go2_demo.fresh_episode_run import StageFollower, play_stage
from go2_demo.fresh_subgoal_planning import FreshSubgoalPlanner
from test_fresh_episode_run import StageHarness


GOAL = {"description": "停靠当前阶段目标", "criterion": "dock_target"}
DOCK = {"waypoints": [{"x": 0., "y": 0.}], "after": "dock",
        "intention": "靠近当前阶段目标并停稳", "route_role": "primary"}
QUERY = {"question": "起点附近见到的 B 在哪里？", "intents": ["b_sighting"]}
CARD = {"evidence_ids": ["E001"], "waypoints": [], "answer": "B 曾在起点左前方可见"}


@pytest.fixture
def retained_run(monkeypatch):
    clients = []
    monkeypatch.setattr(StageFollower, "measure_docking_target", lambda self: (.96, 0., .96, 0.))
    monkeypatch.setattr("go2_demo.fresh_episode_run.TaskLabelRGBD.inspect",
                        lambda *a: ({"target_candidates": []}, {}))

    def create(choices, *, mode="empty", max_requests=4):
        choices = iter(choices)
        sent, events = [], []

        def handler(request):
            sent.append(json.loads(request.content))
            name, arguments = next(choices)
            call = {"id": f"action-{len(sent)}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)}}
            return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [call]},
                                                         "finish_reason": "tool_calls"}],
                                            "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                                                      "total_tokens": 13}})

        planner = FreshSubgoalPlanner("https://api.deepseek.com", "deepseek-flash", "",
                                     transport=httpx.MockTransport(handler), retain_context=True)
        clients.append(planner)
        # The memory's reset/freeze and gate state are real. Only sensor capture
        # and the reader's answer are substituted; no external service is used.
        memory = FreshEpisodeMemory(mode, "memory_ab_occluded_v1", reader=object() if mode == "query" else None)
        monkeypatch.setattr(memory, "observe", lambda *a: None)

        def query(*args, **kwargs):
            memory._card = deepcopy(CARD)
            return {"requests": 0, "unknown_usage_requests": 0, "known_usage": {},
                    "status": "ok", "error": None, "items": {"b_sighting": "available"},
                    "cache_hit": False, "card": memory.card}

        monkeypatch.setattr(memory, "handle", query)
        log = SimpleNamespace(write=lambda event, **fields: events.append((event, deepcopy(fields))),
                              save_frame=lambda frame, name: name)
        config = SimpleNamespace(max_requests=max_requests, max_consecutive_errors=2, max_native_actions=2,
                                 max_sim_seconds=470, max_wall_seconds=1117,
                                 navigation="subgoal", handoff_context="retain")
        return SimpleNamespace(planner=planner, memory=memory, harness=StageHarness(confirm_stop=True),
                               log=log, config=config, mode=mode, context={}, sent=sent, events=events)

    yield create
    for client in clients:
        client.close()


def play(run, stage):
    return play_stage(stage, run.mode, run.harness, run.log, run.memory, run.planner, run.config,
                      episode_context=run.context)


def payload(message):
    return json.loads(message["content"][0]["text"])


def test_return_request_retains_outbound_images_results_and_memory_answer(retained_run):
    run = retained_run([("memory_query", QUERY), ("set_subgoal", GOAL), ("follow_waypoints", DOCK),
                        ("abort", {"reason": "测试结束"})], mode="query")
    outbound = play(run, "A")
    assert outbound["termination"] == "handoff"
    assert outbound["task_state"]["active"]["status"] == "completed"
    run.memory.freeze(preserve_context=True)
    run.planner.context_handoff("A", "B")
    run.harness.frame.rgb.fill(128)
    returning = play(run, "B")
    assert returning["termination"] == "abort"

    messages = run.sent[-1]["messages"]
    images = [message for message in messages if isinstance(message.get("content"), list)]
    assert len(images) == 4
    assert images[0] == run.sent[0]["messages"][-1]
    assert images[0]["content"][1] != images[-1]["content"][1]
    assert [payload(message)["observation"]["task_state"]["stage"] for message in images] == ["A", "A", "A", "B"]
    results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
    assert results[0]["memory_card"] == CARD
    assert results[2]["status"] == "handoff"
    assert results[2]["error_code"] == "A_stop_confirmed"
    transition = [json.loads(message["content"]) for message in messages
                  if message["role"] == "user" and isinstance(message["content"], str)]
    assert transition[0]["event"] == "stage_transition"

    current = payload(messages[-1])
    assert current["memory_card"] == CARD
    assert current["observation"]["latest_memory_feedback"]["question"] == QUERY["question"]
    assert current["observation"]["latest_memory_feedback"]["status"] == "ok"
    state = current["observation"]["task_state"]
    assert state["previous_stage_state"] == outbound["task_state"]
    assert state["active"] is None and state["status"] == "needs_subgoal"
    assert current["recent_return_actions"] == []
    assert run.harness.dispatched == ["stop"]  # No retained A route executes at B.
    first_b = next(fields for event, fields in run.events if event == "planner_input" and fields["stage"] == "B")
    assert first_b["retained_context_turns"] == first_b["retained_context_images"] == 3
    assert returning["context_stats"]["pending_turn"] is False


def test_failed_decision_keeps_valid_dialogue_and_each_stage_has_fresh_budgets(retained_run):
    run = retained_run([("scan", {"yaw_rad": .3, "intention": "非法的无子目标动作"}),
                        ("set_subgoal", GOAL), ("follow_waypoints", DOCK),
                        ("set_subgoal", GOAL), ("follow_waypoints", DOCK)], max_requests=3)
    outbound = play(run, "A")
    assert outbound["termination"] == "handoff" and outbound["planner_requests"] == 3
    run.memory.freeze(preserve_context=True)
    run.planner.context_handoff("A", "B")
    returning = play(run, "B")
    assert returning["termination"] == "finish" and returning["planner_requests"] == 2
    assert outbound["native_attempts"] == {"stop": 1}
    assert returning["native_attempts"] == {"stop": 1, "finish": 1}
    assert run.harness.dispatched == ["stop", "stop", "finish"]
    assert run.harness.backend.sim_deadline_s == outbound["end_sim_s"] + run.config.max_sim_seconds
    first_b = payload(run.sent[3]["messages"][-1])
    assert first_b["observation"]["native_actions_remaining"] == 2
    assert first_b["memory_card"] is None
    assert first_b["observation"]["latest_memory_feedback"] is None
    assert first_b["observation"]["task_state"]["active"] is None

    messages = run.sent[-1]["messages"]
    calls = [call["id"] for message in messages if message["role"] == "assistant"
             for call in message["tool_calls"]]
    replies = [message for message in messages if message["role"] == "tool"]
    assert calls == [message["tool_call_id"] for message in replies]
    assert json.loads(replies[0]["content"])["error_code"] == "MODEL_PROTOCOL_ERROR"
    assert json.loads(replies[0]["content"])["native_actions"] == 0
    assert returning["context_stats"]["retained_turns"] == 5
    assert returning["context_stats"]["pending_turn"] is False
