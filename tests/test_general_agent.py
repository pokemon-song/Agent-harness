"""Offline end-to-end protocol tests, using real HTTP serialization with mock transport."""
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from go2_demo.general_agent import GeneralAgent, AgentLimits, SYSTEM_PROMPT, AGENT_UPDATES_PROMPT, READER_PROMPT, validate_task
from go2_demo.general_episode_run import load_config, MujocoTaskAdapter
from go2_demo.memory_maze_agent import MazeClient


TASK = {"id": "equipment_followup", "instruction": "Inspect the valve, then the drawer; the panel may be observed on the way.",
        "requirements": "Follow the public acceptance feedback.", "targets": [
            {"target_id": "valve_north", "description": "a circular valve"},
            {"target_id": "drawer_east", "description": "a cabinet drawer"},
            {"target_id": "panel_west", "description": "a rectangular panel"}]}


def response(name, args, index):
    return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})


def model(handler, reader=False):
    return MazeClient(base_url="https://offline.invalid/v1", api_key="offline-test", model="offline-test-model",
                      system_prompt=READER_PROMPT if reader else SYSTEM_PROMPT, retain_context=not reader,
                      transport=httpx.MockTransport(handler), trust_env=False)


class Executor:
    def __init__(self):
        self.harness = SimpleNamespace(frame=SimpleNamespace(rgb=np.zeros((5, 8, 3), np.uint8)))
        self.frame = 0
        self.xy = [0., 0.]
        self.grounds = []
        self.executed = []

    def observation(self):
        return {"image_id": f"frame-{self.frame}", "local_odometry": {"translation_m": self.xy.copy(), "yaw_rad": 0.},
                "grounded_targets": [{**g, "image_id": f"frame-{self.frame}"} for g in self.grounds],
                "robot_state": {"fallen": False, "speed_mps": 0., "yaw_rate_radps": 0.}}

    def ground_target(self, target_id, image_id, u, v):
        assert image_id == f"frame-{self.frame}"
        self.executed.append("ground_target")
        self.grounds = [{"target_id": target_id, "point_id": "point-1", "valid": True,
                        "identity_basis": "model_proposed", "geometry_only": True, "range_m": 1.}]
        return {"status": "geometry_grounded", "grounding": self.grounds[0]}

    def approach(self, target_id, desired_distance_m):
        self.executed.append("approach_target")
        self.frame += 1
        return {"status": "skill_completed", "skill": "approach_target", "target_id": target_id,
                "geometry_only": True, "reference": "tracked_surface_point"}

    def navigate(self, name, args, resume_endpoint=False):
        self.executed.append(name)
        self.frame += 1
        if name == "scan":
            return {"status": "ok", "completed_waypoints": 0}
        point = args["waypoints"][0]
        self.xy = [point["x"], point["y"]]
        return {"status": "ok" if len(args["waypoints"]) == 1 else "replan",
                "error_code": None if len(args["waypoints"]) == 1 else "execution_slice_complete",
                "completed_waypoints": 1}

    def finish(self, name, reason=None):
        self.executed.append(name)
        return {"status": name, "stopped": True}


def test_generic_three_targets_full_context_memory_and_partial_route():
    choices = iter([
        ("set_subgoal", {"description": "See valve", "criterion": "observe_target", "target_id": "valve_north"}),
        ("ground_target", {"target_id": "valve_north", "image_id": "frame-0", "u": 3., "v": 2.}),
        ("set_subgoal", {"description": "Approach its visible surface", "criterion": "approach_target",
                         "target_id": "valve_north", "desired_distance_m": 1.}),
        ("approach_target", {"target_id": "valve_north", "desired_distance_m": 1., "intention": "verify local geometry"}),
        ("set_subgoal", {"description": "Move to another view", "criterion": "reach_pose", "target_pose": {"x": 2., "y": 0.}}),
        ("follow_waypoints", {"waypoints": [{"x": 1., "y": 0.}, {"x": 2., "y": 0.}], "after": "inspect",
                              "route_role": "primary", "intention": "change position"}),
        ("memory_query", {"question": "What happened at the valve?", "target_ids": ["valve_north"]}),
        ("continue_subgoal", {"intention": "complete the pending route"}),
        ("finish", {"note": "end this protocol check"}),
    ])
    packets, reader_packets = [], []

    def plan(request):
        packets.append(json.loads(request.content))
        return response(*next(choices), len(packets))

    def read(request):
        packet = json.loads(request.content)
        reader_packets.append(packet)
        data = json.loads(packet["messages"][-1]["content"])
        assert data["events"]
        ids = [r["event_id"] for r in data["events"]]
        return response("answer_evidence", {"answer": "A model-proposed surface point was approached; task acceptance is still pending.",
                                             "event_ids": ids[:2]}, 1)

    executor = Executor()
    agent = GeneralAgent(TASK, model(plan), executor, reader=model(read, True),
        feedback_provider=lambda: {"milestones": [{"target_id": "valve_north", "passed": False, "target_xy": "SECRET_COORDINATES"}],
                                   "task_completed": False, "private_state": "SECRET_COORDINATES"})
    result = agent.run()
    assert result["termination"] == "finish"
    assert result["requests"] == {"planner": 9, "reader": 1}
    assert result["usage"]["total_tokens"] == 120
    assert result["unknown_usage_requests"] == 0
    assert "success" not in result  # only the independent environment may score a task
    assert executor.executed == ["ground_target", "approach_target", "follow_waypoints", "follow_waypoints", "finish"]
    assert len(reader_packets) == 1
    serialized = json.dumps(packets + reader_packets)
    assert "SECRET_COORDINATES" not in serialized
    assert "memory_ab_occluded_v1" not in serialized and "蓝球" not in serialized
    last_messages = packets[-1]["messages"]
    images = [part for m in last_messages if isinstance(m["content"], list) for part in m["content"] if part["type"] == "image_url"]
    assert len(images) == 9
    state_after_slice = json.loads(packets[6]["messages"][-1]["content"][0]["text"])["task_state"]
    assert state_after_slice["active"]["pending_route"]["completed_waypoints"] == 1
    state_after_read = json.loads(packets[7]["messages"][-1]["content"][0]["text"])["task_state"]
    assert state_after_slice["active"]["id"] == state_after_read["active"]["id"]
    assert result["task_state"]["active"]["status"] == "completed"


def test_shared_budget_counts_reader_and_never_permits_an_extra_call():
    packets = []
    def plan(request):
        p = json.loads(request.content)
        packets.append(p)
        return response("memory_query", {"question": "What was observed?"}, len(packets))
    def read(request):
        p = json.loads(request.content)
        payload = json.loads(p["messages"][-1]["content"])
        return response("answer_evidence", {"answer": "initial observation", "event_ids": [payload["events"][0]["event_id"]]}, 1)
    agent = GeneralAgent(TASK, model(plan), Executor(), reader=model(read, True), limits=AgentLimits(max_requests=2))
    result = agent.run()
    assert result["termination"] == "request_budget"
    assert result["requests"] == {"planner": 1, "reader": 1}


def test_unavailable_or_invalid_target_never_reaches_execution():
    packets = []
    def handler(request):
        packets.append(json.loads(request.content))
        return response("ground_target", {"target_id": "unknown", "image_id": "frame-0", "u": 2., "v": 2.}, len(packets))
    executor = Executor()
    agent = GeneralAgent(TASK, model(handler), executor, limits=AgentLimits(max_requests=3, max_consecutive_errors=2))
    assert agent.run()["termination"] == "model_error"
    assert executor.executed == []
    assert len(packets) == 2


def error_response(kind, index):
    if kind == "protocol":
        return response("scan", {"yaw_rad": .1, "intention": "no active subgoal"}, index)
    return response("ground_target", {"target_id": "unknown", "image_id": "frame-0", "u": 2., "v": 2.}, index)


@pytest.mark.parametrize("kind", ["protocol", "validation"])
def test_disabled_error_limit_can_recover_after_more_than_six_errors(kind):
    packets, events = [], []
    def handler(request):
        packets.append(json.loads(request.content))
        if len(packets) <= 7:
            return error_response(kind, len(packets))
        return response("finish", {"note": "recovered after errors"}, len(packets))
    executor, planner = Executor(), model(handler)
    agent = GeneralAgent(TASK, planner, executor,
                         limits=AgentLimits(max_requests=8, max_consecutive_errors=0),
                         log=lambda event, **fields: events.append({"event": event, **fields}))
    result = agent.run()
    assert result["termination"] == "finish"
    assert result["requests"] == {"planner": 8}
    assert executor.executed == ["finish"]
    assert planner._pending_user is None
    if kind == "protocol":
        errors = [e for e in events if e["event"] == "model_error"]
        assert len(errors) == 7 and errors[-1]["consecutive_errors"] == 7
        assert errors[-1]["proposed_tools"] == ["scan"]
        assert "not available" in errors[-1]["message"]


@pytest.mark.parametrize("kind", ["protocol", "validation"])
@pytest.mark.parametrize("error_limit,expected_count,termination", [
    (0, 3, "request_budget"), (2, 2, "model_error"),
])
def test_request_budget_and_optional_error_limit_both_count_failures(kind, error_limit, expected_count, termination):
    packets = []
    def handler(request):
        packets.append(json.loads(request.content))
        return error_response(kind, len(packets))
    executor, planner = Executor(), model(handler)
    result = GeneralAgent(TASK, planner, executor,
                          limits=AgentLimits(max_requests=3, max_consecutive_errors=error_limit)).run()
    assert result["termination"] == termination
    assert result["requests"] == {"planner": expected_count}
    assert len(packets) == expected_count and executor.executed == []
    assert planner._pending_user is None


@pytest.mark.parametrize("kind", ["protocol", "validation"])
def test_disabled_error_limit_still_obeys_wall_deadline(monkeypatch, kind):
    now, packets = [100.], []
    monkeypatch.setattr("go2_demo.general_agent.time.monotonic", lambda: now[0])
    def handler(request):
        packets.append(json.loads(request.content))
        now[0] = 102.
        return error_response(kind, len(packets))
    executor, planner = Executor(), model(handler)
    result = GeneralAgent(TASK, planner, executor,
                          limits=AgentLimits(max_consecutive_errors=0, max_wall_seconds=1.)).run()
    assert result["termination"] == "wall_budget"
    assert result["requests"] == {"planner": 1}
    assert executor.executed == [] and planner._pending_user is None


@pytest.mark.parametrize("value", [-1, True, 1.5, "0"])
def test_error_limit_requires_nonnegative_integer(value):
    with pytest.raises(ValueError):
        AgentLimits(max_consecutive_errors=value)


def test_plain_finish_does_not_create_subgoal_or_acceptance():
    agent = GeneralAgent(TASK, model(lambda r: response("finish", {"note": "premature"}, 1)), Executor())
    result = agent.run()
    assert result["termination"] == "finish"
    assert result["task_state"]["active"] is None
    assert "success" not in result


def test_time_budget_before_reader_closes_the_pending_tool_turn(monkeypatch):
    now = [100.]
    monkeypatch.setattr("go2_demo.general_agent.time.monotonic", lambda: now[0])
    def plan(request):
        now[0] = 102.
        return response("memory_query", {"question": "Read the observation"}, 1)
    def forbidden_reader(request):
        pytest.fail("Reader must not start after the wall deadline")
    planner = model(plan)
    agent = GeneralAgent(TASK, planner, Executor(), reader=model(forbidden_reader, True),
                         limits=AgentLimits(max_wall_seconds=1.))
    result = agent.run()
    assert result["termination"] == "wall_budget"
    assert result["requests"] == {"planner": 1}
    assert planner.context_stats()["turns"] == 1
    assert planner._pending_user is None


def test_memory_failure_after_physical_action_does_not_claim_it_was_unexecuted():
    planner = model(lambda r: response("ground_target", {
        "target_id": "valve_north", "image_id": "frame-0", "u": 2., "v": 2.}, 1))
    executor = Executor()
    agent = GeneralAgent(TASK, planner, executor, reader=model(lambda r: pytest.fail("No reader needed"), True))
    append = agent.memory.append
    def fail_action(kind, *args):
        if kind == "action":
            raise ValueError("offline capacity check")
        return append(kind, *args)
    agent.memory.append = fail_action
    result = agent.run()
    assert result["termination"] == "memory_error"
    assert executor.executed == ["ground_target"]
    feedback = json.loads(planner._dialogue[-1]["content"])
    assert feedback["executed"] is True
    assert feedback["status"] == "geometry_grounded"
    assert planner._pending_user is None


def test_environment_adapter_exposes_only_boolean_acceptance():
    adapter = object.__new__(MujocoTaskAdapter)
    adapter.config = {"task": TASK, "environment": {"checkpoint_target_ids": {"milestone_x": "drawer_east", "milestone_y": "valve_north"}}}
    adapter.evaluation = lambda: {"success": False, "target_xy": [666, 666], "checkpoints": {
        "milestone_x": {"passed": True, "distance": 777}, "milestone_y": {"passed": False, "hidden": 888}}}
    assert adapter.feedback() == {"milestones": [{"target_id": "drawer_east", "passed": True},
                                                 {"target_id": "valve_north", "passed": False}], "task_completed": False}


def test_configs_are_data_only_and_other_task_needs_no_core_changes():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    occluded = load_config(root / "configs/general_agent/occluded_return.json")
    single = load_config(root / "configs/general_agent/single_object.json")
    assert len(occluded["task"]["targets"]) == 2
    assert len(single["task"]["targets"]) == 1
    invalid = deepcopy(TASK)
    invalid["targets"][0]["true_position"] = [1, 2]
    with pytest.raises(ValueError):
        validate_task(invalid)


def test_agent_owns_route_progress_and_completion_through_real_tool_protocol():
    choices = iter([
        ("set_subgoal", {"description": "Inspect the far pose", "criterion": "reach_pose",
                         "target_pose": {"x": 2., "y": 0.}}),
        ("follow_waypoints", {"waypoints": [{"x": 1., "y": 0.}, {"x": 2., "y": 0.}],
                              "after": "inspect", "route_role": "primary", "intention": "move"}),
        ("update_subgoal", {"goal_id": "goal-1", "status": "active", "reason": "First waypoint measured reached",
                            "completed_waypoints": 1}),
        ("continue_subgoal", {"intention": "move to the second waypoint"}),
        ("update_subgoal", {"goal_id": "goal-1", "status": "completed", "reason": "Pose matches the selected goal",
                            "completed_waypoints": 2, "route_done": True}),
        ("finish", {"note": "local protocol check only"}),
    ])
    inputs = []
    def plan(request):
        packet = json.loads(request.content)
        inputs.append(json.loads(packet["messages"][-1]["content"][0]["text"]))
        return response(*next(choices), len(inputs))
    planner = model(plan)
    planner.system_prompt = AGENT_UPDATES_PROMPT
    executor = Executor()
    agent = GeneralAgent(TASK, planner, executor, subgoal_updates="agent",
                         feedback_provider=lambda: {"task_completed": False})
    result = agent.run()
    after_motion = inputs[2]["task_state"]["active"]
    assert after_motion["status"] == "active"
    assert after_motion["pending_route"]["completed_waypoints"] == 0
    assert len(after_motion["pending_route"]["waypoints"]) == 2
    after_update = inputs[3]["task_state"]["active"]
    assert after_update["pending_route"]["waypoints"] == [{"x": 2., "y": 0.}]
    assert inputs[4]["task_state"]["active"]["status"] == "active"  # even at the goal
    final = result["task_state"]["active"]
    assert final["status"] == "completed" and final["pending_route"] is None
    assert final["verification"]["source"] == "agent_judgment"
    assert final["verification"]["independent_task_success"] is None
    assert result["requests"] == {"planner": 6}
    assert executor.executed == ["follow_waypoints", "follow_waypoints", "finish"]
    assert inputs[-1]["observation"]["task_feedback"]["task_completed"] is False
    assert "success" not in result


def test_agent_decides_recovery_without_automatic_continue_gating():
    from go2_demo.general_agent import tools_for
    executor = Executor()
    agent = GeneralAgent(TASK, model(lambda request: pytest.fail("No API call")), executor,
                         subgoal_updates="agent")
    obs = agent.observation()
    agent.state.set_goal({"description": "Reach another pose", "criterion": "reach_pose",
                          "target_pose": {"x": 2., "y": 0.}}, obs)
    agent.state.note_route({"waypoints": [{"x": 2., "y": 0.}], "after": "inspect"})
    for _ in range(4):
        agent.state.record_action("scan", {"yaw_rad": .1}, obs, obs, {"status": "ok"})
    names = lambda: {t["function"]["name"] for t in tools_for(agent.state.snapshot(), agent.target_ids, False)}
    assert not agent.state.recovery_required and "continue_subgoal" in names()
    agent.state.update_goal({"goal_id": "goal-1", "status": "recovering",
                             "reason": "Repeated observations give no new information"}, obs)
    assert agent.state.recovery_required and "continue_subgoal" not in names()
    agent.state.update_goal({"goal_id": "goal-1", "status": "abandoned",
                             "reason": "Choose a different observation location"}, obs)
    assert not {"scan", "follow_waypoints", "continue_subgoal", "approach_target"} & names()
    assert "set_subgoal" in names()


def test_agent_update_schema_rejects_fake_prefix_types():
    from go2_demo.general_agent import tools_for, validate_schema
    state = {"update_policy": "agent", "active": {"id": "goal-1", "status": "active", "criterion": "reach_pose"}}
    schema = next(t["function"]["parameters"] for t in tools_for(state, (), False)
                  if t["function"]["name"] == "update_subgoal")
    base = {"goal_id": "goal-1", "status": "active", "reason": "Use measured route evidence"}
    for bad in (True, 1.5, "1", -1, 33):
        with pytest.raises(ValueError):
            validate_schema({**base, "completed_waypoints": bad}, schema)
    with pytest.raises(ValueError):
        validate_schema({**base, "route_done": "true"}, schema)
