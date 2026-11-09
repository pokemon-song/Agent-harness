"""Memory decisions are explicit tool calls; these tests use no model or simulator."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from go2_demo.agent import MEMORY_TOOLS, MemoryAgent
from go2_demo.harness import Harness
from go2_demo.memory import MemoryScope, MemoryStore
from go2_demo.models import APIModel
from go2_demo.types import ToolCall
from test_robot_harness import MockBackend, mock_profile


def make_agent(tmp_path, *, mode="read_write", episode_id="test-episode", filename="memory.sqlite",
               task_extension=None):
    harness = Harness(MockBackend(mock_profile()), "Approach a visible object safely.")
    store = MemoryStore(tmp_path / filename, scope=MemoryScope.from_profile(harness.profile),
                        mode=mode, episode_id=episode_id)
    return MemoryAgent(harness, store, episode_id, task_extension=task_extension)


def observation(agent):
    return agent.observation(remaining_wall_s=300., remaining_calls=30)


def public_move(agent, call_id="move"):
    before = observation(agent)
    call = ToolCall(call_id, "move_relative", {"forward_m": .2, "lateral_m": 0.})
    results = agent.dispatch_many([call])
    reference = agent.record_public_turn(before, [call], results)
    assert isinstance(reference, str) and results[0]["status"] == "succeeded"
    return reference


def save(agent, reference, call_id="save"):
    return agent.dispatch(ToolCall(call_id, "memory_save", {
        "text": "One observed move requested 0.2 m. Check measured progress; this is not a general calibration.",
        "evidence_refs": [reference],
    }))


def test_constructor_and_observation_do_not_call_a_model_or_automatically_search(tmp_path, monkeypatch):
    monkeypatch.setattr(APIModel, "from_env", lambda **_: pytest.fail("Loaded a model or credentials"))
    monkeypatch.setattr(APIModel, "decide", lambda *_args, **_kwargs: pytest.fail("Hidden model call"))
    agent = make_agent(tmp_path)
    monkeypatch.setattr(agent.store, "search", lambda *_args, **_kwargs: pytest.fail("Automatic retrieval"))
    raw = agent.harness.observation()
    decorated = observation(agent)
    assert {k: v for k, v in decorated.items() if k != "agent_context"} == raw
    assert "agent_context" not in agent.harness.observation()
    assert decorated["agent_context"]["budget"] == {"remaining_wall_s": 300., "remaining_calls": 30}
    assert decorated["agent_context"]["public_evidence_refs"] == []
    assert decorated["agent_context"]["memory"]["record_count"] == 0
    assert MEMORY_TOOLS.issubset({t["function"]["name"] for t in agent.tools})
    assert "single main agent" in agent.system_prompt
    assert "no mandatory" in agent.system_prompt


def test_scope_and_episode_are_explicit_and_cannot_silently_switch_robots(tmp_path):
    agent = make_agent(tmp_path)
    with pytest.raises(TypeError, match="explicit Harness"):
        MemoryAgent(None, agent.store, "test-episode")
    with pytest.raises(ValueError, match="episode_id"):
        MemoryAgent(agent.harness, agent.store, "another-episode")
    other = Harness(MockBackend(replace(mock_profile(), controller_id="another-controller")), "task")
    with pytest.raises(ValueError, match="scope"):
        MemoryAgent(other, agent.store, "test-episode")


def test_empty_search_query_lists_bounded_candidates(tmp_path):
    agent = make_agent(tmp_path)
    saved = save(agent, public_move(agent))
    result = agent.dispatch(ToolCall("list-notes", "memory_search", {"query": "", "limit": 1}))
    assert result["status"] == "succeeded"
    assert len(result["matches"]) == 1 and result["matches"][0]["memory_id"] == saved["memory_id"]
    search = next(t["function"] for t in agent.tools if t["function"]["name"] == "memory_search")
    assert search["parameters"]["properties"]["query"]["minLength"] == 0
    assert "not semantic search" in search["description"]


def test_public_robot_evidence_and_explicit_save_are_separate_idempotent_decisions(tmp_path):
    agent = make_agent(tmp_path)
    reference = public_move(agent)
    assert agent.store.info()["record_count"] == 0
    assert observation(agent)["agent_context"]["public_evidence_refs"] == [reference]
    result = save(agent, reference)
    assert result["status"] == "succeeded"
    assert result["version"] == 1
    before = agent.snapshot()
    repeated = save(agent, reference)
    assert repeated == {**result, "deduplicated": True}
    assert agent.snapshot() == before
    assert agent.store.info()["record_count"] == 1
    assert len(agent.harness.backend.calls) == 1
    conflict = agent.dispatch(ToolCall("save", "move_relative", {"forward_m": .1, "lateral_m": 0.}))
    assert conflict["error_code"] == "CALL_ID_REUSED" and len(agent.harness.backend.calls) == 1


def test_working_notes_and_memory_answers_cannot_register_their_own_robot_evidence(tmp_path):
    agent = make_agent(tmp_path)
    call = ToolCall("note", "memory_working_note", {"text": "Unverified claim: the task is done."})
    before = observation(agent)
    result = agent.dispatch_many([call])
    assert result[0]["status"] == "succeeded"
    assert agent.record_public_turn(before, [call], result) is None
    assert observation(agent)["agent_context"]["public_evidence_refs"] == []
    result = save(agent, "fabricated-or-file-path-reference")
    assert result["status"] == "failed" and agent.store.info()["record_count"] == 0
    assert "Unverified claim" in observation(agent)["agent_context"]["memory"]["working_note"]


def test_revision_requires_the_expected_version_and_new_registered_public_evidence(tmp_path):
    agent = make_agent(tmp_path)
    first = save(agent, public_move(agent))
    reference = public_move(agent, call_id="move-again")
    arguments = {"memory_id": first["memory_id"], "expected_version": 1,
                 "text": "Two observed requests still require measured feedback; no universal movement correction is established.",
                 "reason": "Add a second observed move and retain the original limitation.",
                 "evidence_refs": [reference]}
    result = agent.dispatch(ToolCall("revise", "memory_revise", arguments))
    assert result["status"] == "succeeded" and result["version"] == 2
    snapshot = agent.snapshot()
    assert agent.dispatch(ToolCall("revise", "memory_revise", arguments)) == {**result, "deduplicated": True}
    assert agent.snapshot() == snapshot
    stale = agent.dispatch(ToolCall("stale-revision", "memory_revise", arguments))
    assert stale["error_code"] == "VERSION_CONFLICT"


def test_private_reasoning_evaluator_data_and_memory_context_are_not_saved_as_public_evidence(tmp_path):
    agent = make_agent(tmp_path)
    before = observation(agent)
    before["reasoning_content"] = "PRIVATE_COT_TOKEN"
    before["evaluation"] = {"success": True, "target_xy_m": [99., 99.]}
    before["robot_state"]["reasoning_content"] = "PRIVATE_COT_TOKEN"
    before["robot_state"]["evaluation"] = {"message": "PRIVATE_EVAL_TOKEN"}
    before["agent_context"]["memory"]["working_note"] = "MEMORY_SELF_CLAIM"
    call = ToolCall("move", "move_relative", {"forward_m": .2, "lateral_m": 0.})
    results = agent.dispatch_many([call])
    results[0]["evaluation"] = {"message": "PRIVATE_EVAL_TOKEN"}
    results[0]["measured"] = {
        "translation_start_frame_m": [.18, 0.], "reasoning_content": "PRIVATE_COT_TOKEN",
        "hidden": {"source": "privileged_simulator_evaluator", "value": "PRIVATE_EVAL_TOKEN"},
    }
    reference = agent.record_public_turn(before, [call], results)
    assert reference
    result = save(agent, reference)
    read = agent.dispatch(ToolCall("read", "memory_read", {"memory_id": result["memory_id"]}))
    encoded = json.dumps(read, allow_nan=False)
    assert "translation_start_frame_m" in encoded
    assert all(token not in encoded for token in ("PRIVATE_COT_TOKEN", "PRIVATE_EVAL_TOKEN", "MEMORY_SELF_CLAIM", "agent_context"))
    privileged = deepcopy(before)
    privileged["source"] = "privileged_simulator_evaluator"
    assert agent.record_public_turn(privileged, [call], results) is None
    results[0]["source"] = "privileged_simulator_evaluator"
    assert agent.record_public_turn(before, [call], results) is None


def test_readonly_search_read_and_ram_note_preserve_persistent_memory(tmp_path):
    writer = make_agent(tmp_path, episode_id="writer")
    result = save(writer, public_move(writer))
    writer.store.close()
    reader = make_agent(tmp_path, mode="read_only", episode_id="reader")
    baseline = reader.snapshot()
    search = reader.dispatch(ToolCall("search", "memory_search", {"query": "observed move", "limit": 2}))
    assert search["status"] == "succeeded"
    read = reader.dispatch(ToolCall("read", "memory_read", {"memory_id": result["memory_id"], "version": 1}))
    assert read["status"] == "succeeded"
    assert reader.dispatch(ToolCall("note", "memory_working_note", {"text": "Check current measurements first."}))["status"] == "succeeded"
    assert observation(reader)["agent_context"]["memory"]["working_note"] == "Check current measurements first."
    assert reader.snapshot() == baseline
    assert {t["function"]["name"] for t in reader.tools}.intersection(MEMORY_TOOLS) == {
        "memory_search", "memory_read", "memory_working_note"}
    assert save(reader, "not-registered", call_id="denied")["status"] == "failed"
    assert reader.snapshot() == baseline


@pytest.mark.parametrize("terminal,arguments", [("finish", {}), ("abort", {"reason": "Cannot safely continue."})])
def test_finish_or_abort_blocks_later_writes_without_an_automatic_reflection_call(tmp_path, terminal, arguments):
    agent = make_agent(tmp_path)
    reference = public_move(agent)
    call = ToolCall("terminal", terminal, arguments)
    terminal_result = agent.dispatch(call)
    before = agent.snapshot()
    calls = len(agent.harness.backend.calls)
    assert agent.dispatch(call) == {**terminal_result, "deduplicated": True}
    assert save(agent, reference)["error_code"] == "EPISODE_ENDED"
    assert agent.dispatch(ToolCall("late-note", "memory_working_note", {"text": "late"}))["error_code"] == "EPISODE_ENDED"
    assert agent.snapshot() == before and len(agent.harness.backend.calls) == calls


def test_one_tool_per_turn_applies_across_memory_and_robot_tools(tmp_path):
    agent = make_agent(tmp_path)
    calls = [ToolCall("note", "memory_working_note", {"text": "plan"}),
             ToolCall("move", "move_relative", {"forward_m": .2, "lateral_m": 0.})]
    results = agent.dispatch_many(calls)
    assert all(r["error_code"] == "ONE_TOOL_PER_TURN" for r in results)
    assert agent.harness.backend.calls == [] and agent.store.info()["working_note"] == ""
    assert agent.record_public_turn(observation(agent), calls, results) is None


@pytest.mark.parametrize("name,arguments", [
    ("memory_search", {"query": "move", "limit": True}),
    ("memory_search", {"query": "move", "limit": 11}),
    ("memory_read", {"memory_id": "id", "version": 0}),
    ("memory_save", {"text": "claim"}),
    ("memory_save", {"text": "claim", "evidence_refs": []}),
    ("memory_working_note", {"text": "plan", "unexpected": 1}),
])
def test_memory_schemas_reject_invalid_arguments_before_mutation(tmp_path, name, arguments):
    agent = make_agent(tmp_path)
    before = agent.snapshot()
    assert agent.dispatch(ToolCall("bad", name, arguments))["error_code"] == "INVALID_ARGUMENTS"
    assert agent.snapshot() == before and agent.harness.backend.calls == []
