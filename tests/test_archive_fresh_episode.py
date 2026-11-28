"""Audit the selected trial arms and fresh subgoal state without rendering."""
from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import archive_fresh_episode as archive


def initial(stage):
    return {"event": "planner_input", "stage": stage, "index": 1, "current_targets": [],
            "current_odometry": {"translation_m": [0., 0.], "yaw_rad": 0.}, "exploration": None,
            "task_state": {"stage": stage, "active": None, "recent_goals": [], "last_action": None,
                           "action_count": 0, "no_progress_actions": 0, "recovery_required": False,
                           "recovery_reason": None, "last_target_sighting": None}}


def test_subgoal_stage_audit_accepts_null_exploration_and_counts_goal_once():
    recorded = {"planner_requests": 1, "reader_requests": 0, "total_requests": 1, "native_attempts": {},
                "first_planner_target_candidate": None, "planner_tools": {}, "start_sim_s": 0., "end_sim_s": 1.,
                "path_m": 0., "boundary_refusals": 0, "depth_refusals": 0, "known_usage": {},
                "unknown_usage_requests": 0}
    goal = {"id": "A-1", "criterion": "reach_pose", "status": "completed", "verification": {"source": "public_odometry"}}
    events = [initial("A"), {"event": "subgoal_set", "stage": "A", "detail": {"task_state": {"active": goal}}},
              {"event": "subgoal_verification", "stage": "A", "task_state": {"active": goal}},
              {"event": "subgoal_verification", "stage": "A", "task_state": {"active": goal}}]
    result = archive.audit_stage(Path("trial"), "A", recorded, events)
    assert result["inputs_with_proposed_viewpoints"] == result["selected_viewpoints"] == 0
    assert result["subgoals"]["created"] == result["subgoals"]["completed"] == 1
    assert result["subgoals"]["verification_events"] == 2


@pytest.mark.parametrize("change", ["goal", "history", "cross_stage", "viewpoints", "sighting"])
def test_subgoal_handoff_rejects_retained_A_execution_state(change):
    events = [initial("A"), initial("B")]
    state = events[1]["task_state"]
    if change == "goal":
        state["active"] = {"id": "A-1"}
    elif change == "history":
        state["recent_goals"] = [{"id": "A-1"}]
    elif change == "cross_stage":
        state["stage"] = "A"
    elif change == "viewpoints":
        events[1]["exploration"] = {"viewpoints": [{"x": 1., "y": 0.}]}
    else:
        state["last_target_sighting"] = {"target_label": "A"}
    with pytest.raises(ValueError):
        archive.audit_subgoal_state(Path("trial"), events)


def test_subgoal_handoff_allows_a_current_B_sighting_without_outbound_history():
    events = [initial("A"), initial("B")]
    events[1]["current_targets"] = [{"task_label": "B"}]
    events[1]["task_state"]["last_target_sighting"] = {
        "target_label": "B", "observer_pose": {"x": 0., "y": 0., "yaw": 0.}}
    archive.audit_subgoal_state(Path("trial"), events)


def retained_events():
    a_first, a_last, b_first = initial("A"), initial("A"), initial("B")
    a_last["index"] = 2
    card = {"waypoints": [{"id": "E001", "x": .5, "y": -.2}]}
    feedback = {"question": "B was visible where?", "status": "available"}
    for row, count in ((a_first, 0), (a_last, 1), (b_first, 2)):
        row.update(retained_context_turns=count, retained_context_messages=count * 3, retained_context_images=count,
                   history_turns=count, memory_card=deepcopy(card) if count else None,
                   latest_memory_feedback=deepcopy(feedback) if count else None)
    final_a = {**deepcopy(a_last["task_state"]),
               "active": {"id": "A-1", "status": "completed", "criterion": "dock_target"},
               "action_count": 3}
    b_first["task_state"]["previous_stage_state"] = deepcopy(final_a)
    events = [a_first, {"event": "model_choice", "stage": "A", "tool": "memory_query"},
              a_last, {"event": "model_choice", "stage": "A", "tool": "follow_waypoints"}, b_first]
    manifest = {"handoff_context": "retain", "context_retention": "full_public_dialogue",
                "reset_model_history_at_A": False}
    return events, manifest, final_a


def test_retained_handoff_keeps_A_evidence_but_starts_a_new_B_target():
    events, manifest, final_a = retained_events()
    archive.audit_subgoal_state(Path("trial"), events, handoff_context="retain", outbound_state=final_a)
    archive.audit_handoff_context(Path("trial"), events, manifest)
    assert events[-1]["task_state"]["active"] is None
    assert events[-1]["task_state"]["previous_stage_state"]["active"]["id"] == "A-1"


@pytest.mark.parametrize("change", ["missing_A", "altered_A", "active_A", "wrong_stage"])
def test_retained_handoff_rejects_lost_or_reactivated_outbound_goal(change):
    events, _, final_a = retained_events()
    state = events[-1]["task_state"]
    if change == "missing_A":
        del state["previous_stage_state"]
    elif change == "altered_A":
        state["previous_stage_state"]["action_count"] = 99
    elif change == "active_A":
        state["active"] = deepcopy(final_a["active"])
    else:
        state["previous_stage_state"]["stage"] = "B"
    with pytest.raises(ValueError):
        archive.audit_subgoal_state(Path("trial"), events, handoff_context="retain", outbound_state=final_a)


@pytest.mark.parametrize("change", ["turns", "messages", "images", "decreasing", "card", "feedback", "manifest"])
def test_retained_handoff_rejects_context_or_memory_loss(change):
    events, manifest, _ = retained_events()
    first_b = events[-1]
    if change == "turns":
        first_b["retained_context_turns"] = 0
    elif change == "messages":
        first_b["retained_context_messages"] = 1
    elif change == "images":
        first_b["retained_context_images"] = 0
    elif change == "decreasing":
        events[2]["retained_context_messages"] = 20
    elif change == "card":
        first_b["memory_card"] = None
    elif change == "feedback":
        first_b["latest_memory_feedback"] = None
    else:
        manifest["reset_model_history_at_A"] = True
    with pytest.raises(ValueError):
        archive.audit_handoff_context(Path("trial"), events, manifest)


def test_retained_empty_arm_does_not_require_external_memory_card():
    events, manifest, _ = retained_events()
    for row in events:
        if row["event"] == "planner_input":
            row["memory_card"] = row["latest_memory_feedback"] = None
    archive.audit_handoff_context(Path("trial"), events, manifest)


def test_reset_handoff_still_rejects_retained_card():
    events, _, _ = retained_events()
    with pytest.raises(ValueError, match="nonempty history/card"):
        archive.audit_handoff_context(Path("trial"), events, {"handoff_context": "reset"})


def native(stage, request_id, **reply):
    return {"event": "native_action", "stage": stage,
            "detail": {"call_id": request_id, "reply": {"status": "succeeded", **reply}}}


def test_retained_episode_requires_unique_ids_across_A_and_B_even_when_reply_succeeded():
    # An identical repeated request can return a cached success without moving.
    events = [native("A", "route-control-000"), native("B", "route-control-000")]
    with pytest.raises(ValueError, match="duplicate native action call ID"):
        archive.audit_native_action_ids(Path("trial"), events, retain_context=True)
    # Historical reset runs deliberately clear that cache before the second ID.
    audit = archive.audit_native_action_ids(Path("trial"), events, retain_context=False)
    assert audit["scope"] == "stage" and audit["actions_checked"] == 2


def test_retained_episode_accepts_stage_namespaced_native_ids():
    events = [native("A", "route-A-control-000"), native("B", "route-B-control-000")]
    assert archive.audit_native_action_ids(Path("trial"), events, retain_context=True) == {
        "scope": "episode", "actions_checked": 2, "unique_call_ids": 2, "call_id_reuse_errors": 0}


@pytest.mark.parametrize("retain", [False, True])
def test_call_id_reused_failure_is_rejected_even_if_other_occurrence_is_missing(retain):
    events = [native("B", "route-control-000", status="failed", error_code="CALL_ID_REUSED")]
    with pytest.raises(ValueError, match="CALL_ID_REUSED"):
        archive.audit_native_action_ids(Path("trial"), events, retain_context=retain)


def test_native_ids_cannot_repeat_within_reset_stage():
    with pytest.raises(ValueError, match="duplicate native action"):
        archive.audit_native_action_ids(Path("trial"), [native("A", "same"), native("A", "same")])


def test_native_reply_must_correspond_to_current_request():
    with pytest.raises(ValueError, match="reply does not match"):
        archive.audit_native_action_ids(Path("trial"), [native("B", "new", call_id="old")])


def scored_episode(*, success=True, contacts=2):
    manifest = {"collision_policy": "record_and_penalize", "first_contact_terminates": False,
                "collision_penalty_points": 5, "points_per_checkpoint": 50,
                "completion_requirements": {"collision_free_required": False, "b_contact_ends_episode": False,
                                            "other_scene_contacts_end_episode": False, "collision_penalty_points": 5}}
    goals = 100 if success else 50
    metrics = {"success": success, "goal_points": goals, "collision_count": contacts,
               "collision_penalty": 5 * contacts, "score": max(0, goals - 5 * contacts),
               "independent_evaluation": {"collision": contacts > 0, "collision_count": contacts,
                                          "collision_events": [{"object": "cabinet"}] * contacts,
                                          "checkpoints": {"A": {"passed": True}, "B": {"passed": True}}}}
    return manifest, metrics


def test_collision_score_keeps_collided_completion_independent_of_reduced_score():
    manifest, metrics = scored_episode()
    assert archive.audit_collision_score(Path("trial"), manifest, metrics) == {
        "goal_points": 100, "collision_count": 2, "collision_penalty": 10, "score": 90}
    manifest, metrics = scored_episode(contacts=21)
    result = archive.audit_collision_score(Path("trial"), manifest, metrics)
    assert metrics["success"] and result["score"] == 0 and result["collision_penalty"] == 105


def test_geometric_B_checkpoint_without_episode_completion_gets_only_A_points():
    manifest, metrics = scored_episode(success=False)
    assert archive.audit_collision_score(Path("trial"), manifest, metrics)["goal_points"] == 50
    metrics.update(goal_points=100, score=90)
    with pytest.raises(ValueError, match="score arithmetic"):
        archive.audit_collision_score(Path("trial"), manifest, metrics)


@pytest.mark.parametrize("field", ["goal_points", "collision_count", "collision_penalty", "score"])
def test_collision_score_rejects_incorrect_contact_count_or_arithmetic(field):
    manifest, metrics = scored_episode()
    metrics[field] += 1
    with pytest.raises(ValueError, match="score arithmetic|collision count"):
        archive.audit_collision_score(Path("trial"), manifest, metrics)


@pytest.mark.parametrize("field", ["collision_free_required", "b_contact_ends_episode", "other_scene_contacts_end_episode"])
def test_penalty_policy_cannot_claim_contact_ends_or_disqualifies_episode(field):
    manifest, metrics = scored_episode()
    manifest["completion_requirements"][field] = True
    with pytest.raises(ValueError, match="inconsistent collision scoring policy"):
        archive.audit_collision_score(Path("trial"), manifest, metrics)


def test_collision_score_requires_new_policy_fields_but_accepts_legacy_metrics():
    manifest, metrics = scored_episode()
    del metrics["score"]
    with pytest.raises(ValueError, match="missing or invalid"):
        archive.audit_collision_score(Path("trial"), manifest, metrics)
    assert archive.audit_collision_score(Path("old-trial"), {}, {"success": False}) == {}


def test_selected_pair_packages_no_ledger_and_keeps_matching_source_versions(tmp_path, monkeypatch):
    raw, out = tmp_path / "runs", tmp_path / "results"
    raw.mkdir()
    called = []
    manifest = {"task": "memory_ab_occluded_v1", "seed": 0, "floor_size_m": [24., 18.], "model": "deepseek-flash",
                "api_limit_per_stage": 1024, "max_native_per_stage": 230, "max_sim_s_per_stage": 940,
                "max_wall_s_per_stage": 2234, "completion_requirements": {}, "navigation": "persistent_subgoals_v1",
                "fresh_driver_sha256": {"fresh_subgoals.py": "unchanged"}, "native_actions_per_slice": 6}

    def arm(source, destination, mode):
        called.append(mode)
        target = destination / mode
        target.mkdir(parents=True)
        archive.write(target / "manifest.json", manifest)
        archive.write(target / "audit.json", {"mode": mode})
        (target / "full.mp4").write_bytes(mode.encode())
        return {"episode_id": mode, "initial_image_sha256": "same-initial-frame",
                "A_confirmed_at_checkpoint": False, "B_entered": False,
                "B_passed_if_entered": None, "passed": False}

    monkeypatch.setattr(archive, "audit_arm", arm)
    monkeypatch.setattr("sys.argv", ["archive", str(raw), "--output", str(out), "--modes", "empty", "query"])
    archive.main()
    assert set(called) == {"empty", "query"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["modes"] == ["empty", "query"] and set(summary["arms"]) == {"empty", "query"}
    assert summary["settings"]["fresh_driver_sha256"] == manifest["fresh_driver_sha256"]
    with ZipFile(out / "paired-full-videos.zip") as videos:
        assert videos.namelist() == ["empty-full.mp4", "query-full.mp4"]
    with ZipFile(out / "episode-data.zip") as data:
        assert data.namelist() == ["summary.json", "empty/audit.json", "query/audit.json"]
    assert not (out / "three-full-videos.zip").exists()


@pytest.mark.parametrize("field", ["fresh_driver_sha256", "collision_policy", "collision_penalty_points",
                                  "points_per_checkpoint", "first_contact_terminates", "missing_collision_policy"])
def test_selected_pair_rejects_mismatched_navigation_versions_or_scoring_policies(tmp_path, monkeypatch, field):
    raw, out = tmp_path / "runs", tmp_path / "results"
    raw.mkdir()

    def arm(source, destination, mode):
        target = destination / mode
        target.mkdir(parents=True)
        manifest = {key: None for key in ("task", "seed", "floor_size_m", "model", "api_limit_per_stage",
                    "max_native_per_stage", "max_sim_s_per_stage", "max_wall_s_per_stage", "completion_requirements")}
        manifest["navigation"] = "persistent_subgoals_v1"
        if field == "missing_collision_policy":
            if mode == "query":
                manifest["collision_policy"] = "record_and_penalize"
        else:
            manifest[field] = {"fresh_subgoals.py": mode} if field == "fresh_driver_sha256" else mode
        archive.write(target / "manifest.json", manifest)
        return {"episode_id": mode, "initial_image_sha256": "same"}

    monkeypatch.setattr(archive, "audit_arm", arm)
    monkeypatch.setattr("sys.argv", ["archive", str(raw), "--output", str(out), "--modes", "empty", "query"])
    with pytest.raises(ValueError, match="settings differ"):
        archive.main()
