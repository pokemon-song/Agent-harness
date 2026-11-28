"""Exercise the public planner protocol without making model API requests."""
from copy import deepcopy
import json

import httpx
import pytest

from go2_demo.fresh_subgoal_planning import FreshSubgoalPlanner, SUBGOAL_PROMPT
from go2_demo.fresh_subgoals import StageSubgoals
from go2_demo.models import ModelProtocolError


IMAGE = "data:image/png;base64,cHVibGlj"
GOAL = {"description": "重新看到当前目标", "criterion": "observe_target"}
ROUTE = {"waypoints": [{"x": 1., "y": .4}], "after": "inspect",
         "intention": "移动到柜体外侧观察", "route_role": "primary"}


def observation(stage="B", *, active=True, status="active", pending=False, memory=False):
    obs = {
        "local_odometry": {"translation_m": [2., -.4], "yaw_rad": .3},
        "memory_access": {"allowed": memory, "intents": ["route_reverse", "b_sighting"] if memory else []},
    }
    goals = StageSubgoals(stage)
    if active:
        goals.set_goal(GOAL, obs)
        if pending:
            goals.note_route(ROUTE)
        goals.active["status"] = "recovering" if status == "recovery_required" else status
    goals.recovery_required = status == "recovery_required"
    obs["task_state"] = goals.snapshot()
    return obs


@pytest.fixture
def response_planner():
    clients = []

    def create(name, args, *, extra_calls=None):
        sent = []

        def handler(request):
            sent.append(json.loads(request.content))
            calls = [{"id": "action", "type": "function", "function": {
                "name": name, "arguments": json.dumps(args)}}]
            calls.extend(extra_calls or [])
            return httpx.Response(200, json={"choices": [
                {"message": {"tool_calls": calls}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}})

        client = FreshSubgoalPlanner("https://api.deepseek.com", "deepseek-flash", "",
                                     transport=httpx.MockTransport(handler))
        clients.append(client)
        return client, sent

    yield create
    for client in clients:
        client.close()


def test_prompt_keeps_current_stage_state_and_physical_history_without_candidates(response_planner):
    planner, sent = response_planner("scan", {"yaw_rad": .3, "intention": "确认柜体边缘"})
    obs = observation()
    obs.update(exploration={"viewpoints": [{"waypoint": {"x": 99., "y": 99.}}], "private": "PROPOSAL"},
               secret_evaluator="PRIVATE", return_memory="OUTBOUND",
               depth_sectors={"source": "current_front_depth"},
               latest_memory_feedback={"status": "unavailable"})
    history = [{"tool": "scan", "stage": "A", "feedback": "OUTBOUND"}]
    for i in range(8):
        history.extend([
            {"tool": "continue_subgoal" if i % 2 else "follow_waypoints", "stage": "B",
             "feedback": f"physical-{i}", "private_evaluator": "PRIVATE"},
            {"tool": "memory_query", "stage": "B", "feedback": "QUERY"},
            {"tool": "set_subgoal", "stage": "B", "feedback": "GOAL_ONLY"},
        ])
    history.append({"tool": "previous_request", "stage": "B", "feedback": "correct the last call"})
    planner.choose(obs, IMAGE, None, history, stage="B", timeout_s=2)
    packet = sent[0]
    system, user = packet["messages"]
    payload = json.loads(user["content"][0]["text"])
    assert system["content"] == SUBGOAL_PROMPT
    assert payload["observation"]["task_state"] == obs["task_state"]
    assert payload["observation"]["depth_sectors"] == obs["depth_sectors"]
    assert [item["feedback"] for item in payload["recent_return_actions"]] == [f"physical-{i}" for i in range(2, 8)]
    assert payload["previous_error"] == "correct the last call"
    assert user["content"][1]["image_url"]["url"] == IMAGE
    serialized = json.dumps(packet)
    for forbidden in ("explore_viewpoint", "PROPOSAL", "PRIVATE", "OUTBOUND", "GOAL_ONLY", "QUERY"):
        assert forbidden not in serialized
    assert packet["tool_choice"] == "required" and packet["parallel_tool_calls"] is False


@pytest.mark.parametrize("stage", ["A", "B"])
@pytest.mark.parametrize("active,status", [(False, "active"), (True, "completed")])
def test_motion_requires_an_unfinished_subgoal(stage, active, status):
    obs = observation(stage, active=active, status=status, pending=True)
    before = deepcopy(obs)
    tools = FreshSubgoalPlanner.available_tools(stage, obs)
    assert {t["function"]["name"] for t in tools} == {"set_subgoal", "abort"}
    assert obs == before


def test_recovery_keeps_explicit_motion_but_disallows_blind_route_continuation():
    obs = observation(status="recovery_required", pending=True, memory=True)
    names = {t["function"]["name"] for t in FreshSubgoalPlanner.available_tools("B", obs)}
    assert names == {"set_subgoal", "abort", "scan", "follow_waypoints", "memory_query"}
    obs["task_state"]["active"]["status"] = "active"
    obs["task_state"]["recovery_required"] = False
    assert "continue_subgoal" in {t["function"]["name"] for t in FreshSubgoalPlanner.available_tools("B", obs)}


@pytest.mark.parametrize("name,args", [
    ("set_subgoal", {"description": "到达可见空地", "criterion": "reach_pose", "target_pose": {"x": 1., "y": .4, "yaw": .2}}),
    ("set_subgoal", GOAL),
    ("set_subgoal", {"description": "靠近当前目标并停稳", "criterion": "dock_target"}),
])
def test_initial_goal_is_model_selected_and_cannot_complete_by_declaration(response_planner, name, args):
    planner, sent = response_planner(name, args)
    call = planner.choose(observation(active=False), IMAGE, None, [], stage="B", timeout_s=2)
    assert call.name == name and call.arguments == args
    assert len(sent) == 1


def test_replacing_unfinished_goal_requires_reason_but_completed_goal_does_not(response_planner):
    planner, _ = response_planner("set_subgoal", GOAL)
    with pytest.raises(ModelProtocolError):
        planner.choose(observation(), IMAGE, None, [], stage="B", timeout_s=2)
    assert planner.choose(observation(status="completed"), IMAGE, None, [], stage="B", timeout_s=2).name == "set_subgoal"
    replacement, _ = response_planner("set_subgoal", {**GOAL, "reason": "先前位置已经不可通行，改为重新观察目标"})
    assert replacement.choose(observation(), IMAGE, None, [], stage="B", timeout_s=2).name == "set_subgoal"


@pytest.mark.parametrize("name,args", [
    ("follow_waypoints", ROUTE),
    ("follow_waypoints", {**ROUTE, "route_role": "recovery"}),
    ("scan", {"yaw_rad": -.3, "intention": "核对目标上次出现的朝向"}),
    ("continue_subgoal", {"intention": "沿已保存路线继续到下一个拐弯"}),
])
def test_physical_intent_and_route_role_survive_round_trip(response_planner, name, args):
    planner, _ = response_planner(name, args)
    call = planner.choose(observation(pending=True), IMAGE, None, [], stage="B", timeout_s=2)
    assert call.name == name and call.arguments == args


@pytest.mark.parametrize("name,args", [
    ("set_subgoal", {**GOAL, "completed": True}),
    ("set_subgoal", {**GOAL, "target_pose": {"x": 0., "y": 0.}}),
    ("set_subgoal", {"description": "到达空地", "criterion": "reach_pose"}),
    ("set_subgoal", {"description": "到达空地", "criterion": "reach_pose", "target_pose": {"x": True, "y": 0.}}),
    ("set_subgoal", {"description": " ", "criterion": "observe_target"}),
    ("set_subgoal", {"description": "目标完成", "criterion": "finish"}),
    ("scan", {"yaw_rad": .3}),
    ("scan", {"yaw_rad": .3, "intention": " "}),
    ("scan", {"yaw_rad": .3, "intention": "x" * 241}),
    ("scan", {"yaw_rad": .3, "intention": True}),
    ("follow_waypoints", {k: v for k, v in ROUTE.items() if k != "route_role"}),
    ("follow_waypoints", {**ROUTE, "route_role": "automatic"}),
    ("follow_waypoints", {**ROUTE, "after": "dock"}),
    ("continue_subgoal", {"intention": "继续", "completed": True}),
    ("explore_viewpoint", {"index": 1}),
    ("finish", {}),
])
def test_invalid_model_action_is_charged_without_execution(response_planner, name, args):
    planner, sent = response_planner(name, args)
    with pytest.raises(ModelProtocolError):
        planner.choose(observation(pending=True), IMAGE, None, [], stage="B", timeout_s=2)
    assert len(sent) == 1 and planner.raw_usage["total_tokens"] == 13


def test_docking_requires_dock_goal(response_planner):
    planner, _ = response_planner("follow_waypoints", {**ROUTE, "after": "dock"})
    obs = observation()
    obs["task_state"]["active"]["criterion"] = "dock_target"
    assert planner.choose(obs, IMAGE, None, [], stage="B", timeout_s=2).arguments["after"] == "dock"


def test_memory_access_and_query_scope_remain_enforced(response_planner):
    args = {"question": "提供返程路段和 B 的最后目击", "intents": ["route_reverse", "b_sighting"],
            "route_leg": 1, "sighting_window": "latest"}
    planner, _ = response_planner("memory_query", args)
    with pytest.raises(ModelProtocolError):
        planner.choose(observation(), IMAGE, None, [], stage="B", timeout_s=2)
    assert planner.choose(observation(memory=True), IMAGE, None, [], stage="B", timeout_s=2).arguments == args
    for invalid in ({**args, "intents": ["route_reverse", "route_reverse"]},
                    {**args, "intents": ["a_sighting"]}, {**args, "route_leg": True},
                    {**args, "sighting_window": "future"}):
        bad, _ = response_planner("memory_query", invalid)
        with pytest.raises(ModelProtocolError):
            bad.choose(observation(memory=True), IMAGE, None, [], stage="B", timeout_s=2)


def test_stage_mismatch_is_rejected_before_request(response_planner):
    planner, sent = response_planner("set_subgoal", GOAL)
    with pytest.raises(ValueError):
        planner.choose(observation("A"), IMAGE, None, [], stage="B", timeout_s=2)
    assert sent == []


def test_multiple_tools_do_not_bypass_one_decision_contract(response_planner):
    extra = [{"id": "second", "type": "function", "function": {"name": "scan", "arguments": json.dumps({
        "yaw_rad": .2, "intention": "再观察一次"})}}]
    planner, sent = response_planner("follow_waypoints", ROUTE, extra_calls=extra)
    with pytest.raises(ModelProtocolError):
        planner.choose(observation(), IMAGE, None, [], stage="B", timeout_s=2)
    assert len(sent) == 1
