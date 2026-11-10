"""Planning must read published memory and gate real Harness dispatch.

The backend and model here are explicit fixtures: there is no API or physics.
Assertions cover observable reads, rejected plans and physical-call counts.
"""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from go2_demo.harness import Harness
from go2_demo.md_memory import planner as planner_module
from go2_demo.md_memory.planner import MDPlanningAgent, PlanError, check_plan, execute_plan
from go2_demo.md_memory.store import MDMemoryStore
from go2_demo.models import ModelError
from go2_demo.robots.go2.profile import GO2_PROFILE
from go2_demo.types import Decision, SensorFrame, ToolCall


IMAGE = "data:image/png;base64,fixture-image-not-sent-to-provider"
PRIVATE_SENTINEL = "PRIVATE_REASONING_MUST_NOT_APPEAR_IN_LOGS"
TASK_PATH = "tasks/current-order.md"


class FixtureBackend:
    """Count execution calls while using the production profile and Harness."""

    def __init__(self, *, lateral_motion=True):
        self.profile = replace(
            GO2_PROFILE, lateral_motion=lateral_motion, point_tracking=False,
            height_pitch_control=False, posture=None,
            sensor=replace(GO2_PROFILE.sensor, rgb_width_px=8, rgb_height_px=6,
                           depth_enabled=False),
        )
        self.pose = np.zeros(3)
        self.calls = []
        self.captures = 0

    def capture(self):
        self.captures += 1
        return SensorFrame(
            frame_id=f"fixture-{self.captures}", sim_time_s=self.captures * .01,
            rgb=np.zeros((6, 8, 3), dtype=np.uint8), depth_m=None,
            intrinsics=None, camera_to_world=None, base_xy_yaw=self.pose.copy(),
            robot_state={"speed_mps": 0., "yaw_rate_radps": 0., "fallen": False},
        )

    def execute(self, call):
        self.profile.validate_action(call.name, call.arguments)
        self.calls.append(deepcopy(call))
        if call.name == "move_relative":
            self.pose[:2] += [call.arguments["forward_m"], call.arguments["lateral_m"]]
        elif call.name == "turn":
            self.pose[2] += call.arguments["yaw_rad"]
        return {"status": "succeeded", "measured": {"base_xy_yaw": self.pose.tolist()}}


class ScriptedPlannerModel:
    """Actions may inspect only the observation and returned tool history."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.requests = []
        self.system_prompt = ""
        self.close_count = 0

    def decide(self, observation, image_data_url, tools, history, *, timeout_s):
        self.requests.append({"observation": deepcopy(observation), "image": image_data_url,
                              "history": deepcopy(history), "timeout_s": timeout_s})
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        if callable(action):
            action = action(observation, history)
        if isinstance(action, Decision):
            return action
        name, arguments = action
        return Decision(
            calls=[ToolCall(f"fixture-call-{len(self.requests)}", name, deepcopy(arguments))],
            usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            model="scripted-planner-fixture", reasoning_content=PRIVATE_SENTINEL,
        )

    def close(self):
        self.close_count += 1


def publish(store, *, path=TASK_PATH, body="ACTIVE R1: move left 0.2 metres.",
            status="confirmed", critical=True):
    event = store.append_event("instruction", {"notice": body}, critical=critical)
    result = store.commit(
        store.snapshot()["revision"], event["seq"],
        [{"path": path, "body": body, "evidence_ids": [event["id"]], "status": status}],
        call_id=f"publish-{event['id']}",
    )
    return result, event


@pytest.fixture
def environment(tmp_path):
    store = MDMemoryStore(tmp_path / "memory", scope="world-fixture")
    publish(store)
    backend = FixtureBackend()
    harness = Harness(backend, "Use current published order memory.")
    return SimpleNamespace(store=store, backend=backend, harness=harness, directory=tmp_path)


def proposal(observation, document, *, action="move_relative", arguments=None):
    return {
        "based_on_revision": observation["memory_snapshot"]["revision"],
        "dependencies": [{"path": document["path"], "sha256": document["sha256"]}],
        "subgoal": "Follow the current published order.",
        "completion_condition": "Measured action completes and the robot is stopped.",
        "on_failure": "stop_and_replan", "action_name": action,
        "arguments": arguments if arguments is not None else {"forward_m": 0., "lateral_m": .2},
    }


def submit_from_read(*, action="move_relative", arguments=None):
    def response(observation, history):
        document = history[-1]["results"][0]["document"]
        return "submit_plan", proposal(observation, document, action=action, arguments=arguments)
    return response


def agent_for(env, actions, *, max_calls=None, log_name="planner.jsonl"):
    model = ScriptedPlannerModel(actions)
    agent = MDPlanningAgent(env.store, model, env.harness,
                            log_path=env.directory / log_name,
                            max_calls=max_calls or len(actions), request_timeout_s=5)
    return agent, model


def ready_plan(env, *, action="move_relative", arguments=None, log_name="planner.jsonl"):
    agent, model = agent_for(env, [("memory_read", {"path": TASK_PATH}),
                                  submit_from_read(action=action, arguments=arguments)],
                             log_name=log_name)
    plan = agent.plan("Follow the published order.", IMAGE, timeout_s=10)
    return plan, agent, model


def tool_results(env, filename="planner.jsonl"):
    return [result for row in (env.directory / filename).read_text().splitlines()
            for event in [json.loads(row)] if event["event"] == "tool_results"
            for result in event["results"]]


def test_known_document_hash_without_memory_read_is_not_permission_to_plan(environment):
    env = environment
    document = env.store.read(TASK_PATH)
    agent, _ = agent_for(env, [lambda observation, history:
                              ("submit_plan", proposal(observation, document))])
    with pytest.raises(PlanError, match="call budget"):
        agent.plan("Read memory first.", IMAGE)
    assert tool_results(env)[-1]["error_code"] == "UNREAD_MEMORY"
    assert env.backend.calls == []


def test_search_summary_does_not_replace_read_receipt(environment):
    env = environment

    def use_search_match(observation, history):
        return "submit_plan", proposal(observation, history[-1]["results"][0]["matches"][0])

    agent, _ = agent_for(env, [("memory_search", {"query": "ACTIVE"}), use_search_match])
    with pytest.raises(PlanError):
        agent.plan("Do not act on an unread search summary.", IMAGE)
    assert tool_results(env)[-1]["error_code"] == "UNREAD_MEMORY"
    assert env.backend.calls == []


def test_submit_is_read_only_and_execution_reuses_harness_deduplication(environment):
    env = environment
    before = env.store.snapshot()
    plan, agent, model = ready_plan(env)
    assert env.backend.calls == []
    assert env.store.snapshot() == before
    assert model.requests[1]["history"][0]["results"][0]["document"]["path"] == TASK_PATH

    first = execute_plan(plan, env.store, env.harness)
    second = execute_plan(plan, env.store, env.harness)
    assert first["status"] == second["status"] == "succeeded"
    assert second["deduplicated"] is True
    assert len(env.backend.calls) == 1
    assert env.backend.calls[0].call_id == plan.plan_id
    assert env.backend.pose.tolist() == [0., .2, 0.]
    agent.close()
    agent.close()
    assert model.close_count == 1


def test_unprocessed_critical_input_blocks_previously_accepted_plan(environment):
    env = environment
    plan, _, _ = ready_plan(env)
    env.store.append_event("user_revision", {"notice": "ACTIVE R2: reverse direction."}, critical=True)
    with pytest.raises(PlanError) as error:
        execute_plan(plan, env.store, env.harness)
    assert error.value.code == "MEMORY_PENDING"
    assert env.backend.calls == []


def test_critical_input_arriving_during_planning_blocks_submit(environment):
    env = environment

    def submit_after_new_input(observation, history):
        env.store.append_event("user_revision", {"notice": "Review the new instruction."}, critical=True)
        return submit_from_read()(observation, history)

    agent, _ = agent_for(env, [("memory_read", {"path": TASK_PATH}), submit_after_new_input])
    with pytest.raises(PlanError):
        agent.plan("Plan using memory.", IMAGE)
    assert tool_results(env)[-1]["error_code"] == "MEMORY_PENDING"
    assert env.backend.calls == []


def test_revised_dependency_rejects_old_plan_after_event_is_processed(environment):
    env = environment
    plan, _, _ = ready_plan(env)
    publish(env.store, body="ACTIVE R2: move right 0.2 metres.")
    assert env.store.snapshot()["pending_critical_events"] == []
    with pytest.raises(PlanError) as error:
        execute_plan(plan, env.store, env.harness)
    assert error.value.code == "MEMORY_CHANGED"
    assert env.backend.calls == []


def test_unrelated_published_document_does_not_invalidate_read_dependency(environment):
    env = environment
    plan, _, _ = ready_plan(env)
    old_hash = plan.dependencies[0]["sha256"]
    publish(env.store, path="knowledge/bench.md", body="A bench was observed in another area.")
    assert env.store.snapshot()["revision"] > plan.snapshot_revision
    assert env.store.read(TASK_PATH)["sha256"] == old_hash
    result = execute_plan(plan, env.store, env.harness)
    assert result["status"] == "succeeded"
    assert len(env.backend.calls) == 1


def test_noncritical_pending_input_does_not_block_an_unchanged_plan(environment):
    env = environment
    plan, _, _ = ready_plan(env)
    env.store.append_event("diagnostic", {"status": "log rotation completed"}, critical=False)
    assert execute_plan(plan, env.store, env.harness)["status"] == "succeeded"
    assert len(env.backend.calls) == 1


@pytest.mark.parametrize("status", ["unknown", "stale", "contradicted", "predicted"])
def test_uncertain_memory_cannot_authorize_translation(environment, status):
    env = environment
    publish(env.store, body="Instruction still requires confirmation.", status=status)
    agent, _ = agent_for(env, [("memory_read", {"path": TASK_PATH}), submit_from_read()])
    with pytest.raises(PlanError):
        agent.plan("Act only if evidence is usable.", IMAGE)
    assert tool_results(env)[-1]["error_code"] == "MEMORY_UNCERTAIN"
    assert env.backend.calls == []


def test_predicted_memory_allows_observation_without_translation(environment):
    env = environment
    publish(env.store, body="Target may remain behind the occluder.", status="predicted")
    plan, _, _ = ready_plan(env, action="observe", arguments={})
    captures = env.backend.captures
    assert execute_plan(plan, env.store, env.harness)["status"] == "succeeded"
    assert env.backend.captures == captures + 1
    assert env.backend.calls == []


@pytest.mark.parametrize("arguments", [
    {"forward_m": .31, "lateral_m": 0.},
    {"forward_m": .3, "lateral_m": .3},
    {"forward_m": 0., "lateral_m": True},
])
def test_demo_action_limits_reject_motion_before_backend_execution(environment, arguments):
    env = environment
    agent, _ = agent_for(env, [("memory_read", {"path": TASK_PATH}),
                              submit_from_read(arguments=arguments)])
    with pytest.raises(PlanError):
        agent.plan("Propose a short action.", IMAGE)
    assert tool_results(env)[-1]["error_code"] == "INVALID_ACTION"
    assert env.backend.calls == []


def test_same_world_memory_does_not_override_different_robot_capability(environment):
    env = environment
    backend = FixtureBackend(lateral_motion=False)
    env.harness = Harness(backend, "Use the same world memory on a robot without sideways motion.")
    agent, _ = agent_for(env, [("memory_read", {"path": TASK_PATH}), submit_from_read()])
    with pytest.raises(PlanError):
        agent.plan("Respect the current robot profile.", IMAGE)
    assert tool_results(env)[-1]["error_code"] == "INVALID_ACTION"
    assert backend.calls == []


def test_refresh_clears_receipts_from_previous_snapshot(environment):
    env = environment

    def refresh_after_publication(observation, history):
        publish(env.store, body="ACTIVE R2: move right 0.2 metres.")
        return "memory_refresh", {}

    def forge_current_hash(observation, history):
        return "submit_plan", proposal(observation, env.store.read(TASK_PATH))

    agent, _ = agent_for(env, [("memory_read", {"path": TASK_PATH}),
                              refresh_after_publication, forge_current_hash])
    with pytest.raises(PlanError):
        agent.plan("Refresh and actually read the new document.", IMAGE)
    assert tool_results(env)[-1]["error_code"] == "UNREAD_MEMORY"
    assert env.backend.calls == []


def test_expired_and_different_world_plans_never_reach_backend(environment):
    env = environment
    plan, _, _ = ready_plan(env)
    for invalid, code in [(replace(plan, expires_monotonic=0.), "PLAN_EXPIRED"),
                          (replace(plan, scope="another-world"), "SCOPE_CHANGED")]:
        with pytest.raises(PlanError) as error:
            execute_plan(invalid, env.store, env.harness)
        assert error.value.code == code
    assert env.backend.calls == []


@pytest.mark.parametrize("code", ["MODEL_TIMEOUT", "MODEL_PROTOCOL_ERROR"])
def test_model_failures_preserve_attempts_without_logging_private_content(environment, code):
    env = environment
    failure = ModelError("untrusted exception text " + PRIVATE_SENTINEL, code)
    agent, _ = agent_for(env, [("memory_read", {"path": TASK_PATH}), failure])
    with pytest.raises(ModelError) as error:
        agent.plan("Plan from published evidence.", IMAGE)
    assert error.value.code == code
    assert agent.stats["api_requests"] == 2
    assert agent.stats["completed_responses"] == 1
    assert agent.stats["usage"]["total_tokens"] == 14
    assert agent.stats["errors"][-1]["code"] == code
    content = (env.directory / "planner.jsonl").read_text()
    assert PRIVATE_SENTINEL not in content
    assert "reasoning_content" not in content
    events = [json.loads(row) for row in content.splitlines()]
    assert events[-1]["event"] == "model_error"
    assert events[-1]["code"] == code
    assert env.backend.calls == []


def test_response_after_planning_deadline_cannot_produce_plan(environment, monkeypatch):
    env = environment
    clock = [100.]
    monkeypatch.setattr(planner_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def late_submit(observation, history):
        clock[0] += 2.
        return submit_from_read()(observation, history)

    agent, _ = agent_for(env, [("memory_read", {"path": TASK_PATH}), late_submit])
    with pytest.raises(PlanError) as error:
        agent.plan("Discard late decisions.", IMAGE, timeout_s=1.)
    assert error.value.code == "PLANNER_DEADLINE"
    assert env.backend.calls == []
    assert not any("plan" in result for result in tool_results(env))


def test_planner_cannot_write_memory_or_directly_dispatch_motion(environment):
    env = environment
    before = env.store.snapshot()
    agent, _ = agent_for(env, [("memory_commit", {"body": "invented state"}),
                              ("move_relative", {"forward_m": .2, "lateral_m": 0.})])
    with pytest.raises(PlanError):
        agent.plan("Use only the planning tools.", IMAGE)
    assert [r["error_code"] for r in tool_results(env)] == ["UNKNOWN_TOOL", "UNKNOWN_TOOL"]
    assert env.store.snapshot() == before
    assert env.backend.calls == []
