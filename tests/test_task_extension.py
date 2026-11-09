"""Task tools share the main loop; their independent verdict never enters model context."""

from copy import deepcopy
import json

import numpy as np
import pytest

from go2_demo.runner import run_episode
from go2_demo.types import Decision, ToolCall
from test_memory_agent import make_agent, observation
from test_runner import EpisodeBackend, EpisodeModel


class ReceiptExtension:
    extension_id = "receipt-test-v1"

    def __init__(self):
        self.calls = []
        self.evaluations = 0
        self.receipt = None
        self.tools = [{"type": "function", "function": {
            "name": "submit_delivery", "description": "Submit the remembered delivery receipt.",
            "parameters": {"type": "object", "properties": {
                "order_id": {"type": "string"}, "destination": {"type": "string"},
                "receipt_code": {"type": "string"}},
                "required": ["order_id", "destination", "receipt_code"], "additionalProperties": False}}}]

    def execute(self, call):
        assert call.name == "submit_delivery"
        self.calls.append(call)
        self.receipt = deepcopy(call.arguments)
        # Receipt correctness is deliberately not reported by this public tool.
        return {"status": "succeeded", "submission_received": True}

    def evaluate(self):
        self.evaluations += 1
        return {"success": self.receipt == {"order_id": "order-7", "destination": "dock-b", "receipt_code": "code-19"},
                "private_note": "EVALUATOR_ONLY_TOKEN"}


def submit(call_id="submit", *, code="code-19"):
    return ToolCall(call_id, "submit_delivery", {
        "order_id": "order-7", "destination": "dock-b", "receipt_code": code})


def test_task_tool_is_idempotent_and_does_not_become_robot_memory_evidence(tmp_path):
    extension = ReceiptExtension()
    agent = make_agent(tmp_path, task_extension=extension)
    assert "submit_delivery" in {tool["function"]["name"] for tool in agent.tools}
    before = observation(agent)
    call = submit()
    first = agent.dispatch_many([call])
    assert first[0]["status"] == "succeeded"
    assert agent.record_public_turn(before, [call], first) is None
    assert observation(agent)["agent_context"]["public_evidence_refs"] == []
    assert agent.dispatch(call) == {**first[0], "deduplicated": True}
    assert len(extension.calls) == 1 and extension.evaluations == 0
    assert agent.dispatch(submit(code="changed"))["error_code"] == "CALL_ID_REUSED"
    conflict = agent.dispatch(ToolCall("submit", "memory_working_note", {"text": "changed domain"}))
    assert conflict["error_code"] == "CALL_ID_REUSED"
    agent.dispatch(ToolCall("finish", "finish", {}))
    assert agent.dispatch(submit("after-end"))["error_code"] == "EPISODE_ENDED"
    assert len(extension.calls) == 1 and extension.evaluations == 0


@pytest.mark.parametrize("collision", ["finish", "set_posture", "memory_search", "observe"])
def test_task_tools_cannot_shadow_robot_or_memory_names_even_when_capability_is_disabled(tmp_path, collision):
    extension = ReceiptExtension()
    extension.tools[0]["function"]["name"] = collision
    with pytest.raises(ValueError, match="collide"):
        make_agent(tmp_path, task_extension=extension)
    assert not extension.calls and extension.evaluations == 0


def test_task_memory_and_robot_tools_are_all_blocked_in_a_multi_tool_response(tmp_path):
    extension = ReceiptExtension()
    agent = make_agent(tmp_path, task_extension=extension)
    calls = [submit(), ToolCall("note", "memory_working_note", {"text": "plan"}),
             ToolCall("move", "move_relative", {"forward_m": .1, "lateral_m": 0.})]
    assert all(result["error_code"] == "ONE_TOOL_PER_TURN" for result in agent.dispatch_many(calls))
    assert not extension.calls and not agent.harness.backend.calls
    assert agent.store.info()["working_note"] == ""


class DeliveryModel(EpisodeModel):
    def __init__(self, extension, *, code="code-19"):
        super().__init__()
        self.turn = 0
        self.extension = extension
        self.code = code

    def decide(self, observation, image, tools, history, **kwargs):
        assert self.extension.evaluations == 0, "Task verdict was computed before final independent evaluation"
        assert "EVALUATOR_ONLY_TOKEN" not in json.dumps({"observation": observation, "history": history, "tools": tools})
        assert observation["agent_context"]["budget"]["remaining_calls"] == 5 - self.turn
        assert "submit_delivery" in {tool["function"]["name"] for tool in tools}
        call = (ToolCall("note", "memory_working_note", {"text": "remember the receipt"}) if self.turn == 0
                else ToolCall("move", "move_relative", {"forward_m": .1, "lateral_m": 0.}) if self.turn == 1
                else submit(code=self.code) if self.turn in (2, 3)
                else ToolCall("finish", "finish", {}))
        self.turn += 1
        return Decision([call], model=self.model, usage={"total_tokens": 10})


@pytest.mark.parametrize("code,success", [("code-19", True), ("wrong-receipt", False)])
def test_receipt_verdict_is_final_only_and_wrong_receipt_overrides_physical_success(tmp_path, code, success):
    extension = ReceiptExtension()
    backend = EpisodeBackend(score_success=True)
    model = DeliveryModel(extension, code=code)
    directory = tmp_path / "episode"
    result = run_episode(backend, model, mode="api", run_dir=directory, max_model_calls=5,
                         memory_mode="read_write", memory_dir=tmp_path / "memory", task_extension=extension)
    assert result["termination"] == "finish" and result["success"] is success
    assert result["evaluation"]["physical_success"] is True
    assert result["evaluation"]["task_extension"]["success"] is success
    assert result["evaluation"]["success"] is success
    assert result["model_calls"] == 5 and result["usage"]["total_tokens"] == 50
    assert result["memory_tool_calls"] == 1 and result["task_tool_calls"] == 2 and result["motion_calls"] == 1
    assert len(extension.calls) == 1 and extension.evaluations == 1
    assert model.closed and backend.closed
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["task_extension_id"] == extension.extension_id
    assert "EVALUATOR_ONLY_TOKEN" not in json.dumps(manifest)
    tools = json.loads((directory / "tools.json").read_text())
    assert next(tool for tool in tools if tool["function"]["name"] == "submit_delivery") == extension.tools[0]
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert "EVALUATOR_ONLY_TOKEN" not in json.dumps([e for e in events if e["event"] != "episode_end"])


def test_task_extension_requires_enabled_memory_before_any_model_request(tmp_path):
    extension = ReceiptExtension()
    with pytest.raises(ValueError, match="require memory_mode"):
        run_episode(EpisodeBackend(), EpisodeModel(after_decision=lambda: pytest.fail("Model called")),
                    mode="api", run_dir=tmp_path / "run", task_extension=extension)
    assert not extension.calls and extension.evaluations == 0
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("verdict", [RuntimeError("evaluation failed"), {"success": 1}, {"success": np.bool_(True)}, []])
def test_extension_evaluation_exception_or_non_native_success_is_a_failed_evaluation(tmp_path, verdict):
    extension = ReceiptExtension()
    def evaluate():
        extension.evaluations += 1
        if isinstance(verdict, BaseException):
            raise verdict
        return verdict
    extension.evaluate = evaluate
    result = run_episode(EpisodeBackend(score_success=True), EpisodeModel(), mode="api", run_dir=tmp_path / "run",
                         memory_mode="read_write", memory_dir=tmp_path / "memory", task_extension=extension)
    assert result["success"] is False and result["termination"] == "evaluation_error"
    assert result["evaluation"]["physical_success"] is True
    assert result["evaluation"]["task_extension"]["success"] is False
    assert result["model_calls"] == 1 and extension.evaluations == 1


def test_task_tool_consumes_the_last_model_call_without_implying_finish(tmp_path):
    extension = ReceiptExtension()
    class SubmitOnly(EpisodeModel):
        def decide(self, *args, **kwargs):
            return Decision([submit()], model=self.model)
    result = run_episode(EpisodeBackend(), SubmitOnly(), mode="api", run_dir=tmp_path / "run", max_model_calls=1,
                         memory_mode="read_write", memory_dir=tmp_path / "memory", task_extension=extension)
    assert result["termination"] == "budget_exhausted" and result["success"] is False
    assert result["model_calls"] == result["task_tool_calls"] == 1
    assert result["evaluation"]["physical_success"] and result["evaluation"]["task_extension"]["success"]


def test_no_extension_does_not_add_task_fields_to_the_original_evaluation(tmp_path):
    result = run_episode(EpisodeBackend(), EpisodeModel(), mode="api", run_dir=tmp_path / "run")
    assert result["success"] and "task_extension_id" not in result and "task_tool_calls" not in result
    assert "physical_success" not in result["evaluation"] and "task_extension" not in result["evaluation"]
