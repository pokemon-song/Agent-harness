"""Retained handoff keeps evidence while separating current-stage execution."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from go2_demo.fresh_episode_memory import FreshEpisodeMemory
from go2_demo.fresh_subgoals import StageSubgoals
from test_fresh_episode_memory import SelectAll, harness, scoped
from test_fresh_subgoals import goal, obs


def recorded_memory():
    reader = SelectAll()
    memory = FreshEpisodeMemory("query", "memory_ab_occluded_v1", reader)
    candidate = {"task_label": "B", "description": "绿方块", "pixel_uv": [0, 0], "camera_xyz_m": [0., 0., 1.]}
    memory._notes.sensor = SimpleNamespace(inspect=lambda *args: ({"target_candidates": [candidate]}, {}))
    memory.observe(harness(0., "frame-00001"), 0)
    memory.observe(harness(1., "frame-00002"), 1)
    scoped(memory, "b_sighting", "route_outbound")
    return memory, reader


def test_retained_freeze_keeps_answer_cache_gate_and_frozen_outbound_documents():
    memory, reader = recorded_memory()
    # A repeated query closes the query gate until physical motion occurs.
    answer = scoped(memory, "b_sighting", "route_outbound")
    assert answer["cache_hit"] and not memory.query_access()["allowed"]
    before = deepcopy((memory.card, memory._cache, memory.query_access(), memory.documents()))
    memory.freeze(preserve_context=True)
    assert memory._frozen
    assert (memory.card, memory._cache, memory.query_access(), memory.documents()) == before
    memory.observe(harness(9., "frame-00003"), 2)
    assert memory.documents() == before[3]
    memory.after_physical_action()
    again = scoped(memory, "b_sighting", "route_outbound")
    assert again["cached_intents"] == ["b_sighting", "route_outbound"]
    assert again["requests"] == 0 and len(reader.inputs) == 1
    # A new reverse-route intent is still selected with return-stage semantics.
    memory.after_physical_action()
    reverse = scoped(memory, "route_reverse")
    assert reverse["requests"] == 1 and reader.inputs[-1]["phase"] == "return_to_B"
    assert [point["x"] for point in reverse["card"]["waypoints"]] == [1., 0.]


def test_default_freeze_still_clears_answer_cache_and_reopens_query_gate():
    memory, _ = recorded_memory()
    scoped(memory, "b_sighting", "route_outbound")
    docs = memory.documents()
    memory.freeze()
    assert memory.card is None and memory._cache == {}
    assert memory.query_access()["allowed"]
    assert memory.documents() == docs


def test_empty_memory_has_no_extra_evidence_when_context_is_retained():
    memory = FreshEpisodeMemory("empty", "memory_ab_occluded_v1")
    memory.freeze(preserve_context=True)
    assert memory.card is None and memory.documents() == {}
    assert memory.query_access()["reason"] == "no_external_memory"


def test_previous_goals_sightings_and_action_remain_visible_without_reexecuting_A():
    outbound = StageSubgoals("A")
    outbound.set_goal(goal("observe_target"), obs(target="A", frame="initial-A"))
    outbound.set_goal(goal(), obs(target="A", frame="approach-A"))
    route = {"waypoints": [{"x": 1., "y": 0.}], "after": "dock"}
    outbound.note_route(route)
    outbound.record_action("follow_waypoints", route, obs(target="A"), obs(1., target="A", frame="stop-A"),
                           {"status": "handoff", "error_code": "A_stop_confirmed", "completed_waypoints": 1},
                           intention="complete the approach to A")
    prior = outbound.snapshot()
    returning = StageSubgoals("B", prior_stage_state=prior)
    first = returning.observe(obs(1., target="A", frame="stop-A"))
    assert first["previous_stage_state"] == prior
    assert first["previous_stage_state"]["active"]["verification"]["source"] == "independent_A_handoff"
    assert first["previous_stage_state"]["last_target_sighting"]["image_id"] == "stop-A"
    assert first["previous_stage_state"]["last_action"]["intention"] == "complete the approach to A"
    assert first["active"] is None and returning.pending_route is None
    assert first["last_target_sighting"] is None and not first["current_target_visible"]
    assert first["status"] == "needs_subgoal" and first["recent_goals"] == []
    # The new B goal cannot be completed by the retained A visibility or success.
    returning.set_goal(goal("observe_target"), obs(target="A"))
    assert returning.active["target_label"] == "B" and returning.active["status"] == "active"
    returning.observe(obs(target="B", frame="now-B"))
    assert returning.active["status"] == "completed"
    assert returning.snapshot()["previous_stage_state"] == prior
    # Parent state and serialized snapshots do not alias retained execution data.
    prior["active"]["description"] = "mutated by caller"
    first["previous_stage_state"]["last_target_sighting"] = None
    assert returning.snapshot()["previous_stage_state"]["active"]["description"] != "mutated by caller"
    assert returning.snapshot()["previous_stage_state"]["last_target_sighting"] is not None


@pytest.mark.parametrize("stage,previous", [("A", {"stage": "A"}), ("B", {"stage": "B"}),
                                           ("B", {"stage": "A", "previous_stage_state": {"stage": "A"}})])
def test_previous_stage_context_cannot_form_recursive_or_cross_stage_execution(stage, previous):
    with pytest.raises(ValueError):
        StageSubgoals(stage, prior_stage_state=previous)
